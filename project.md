# Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

or with conda
```bash
conda create -n charles-cubicasa python=3.11
conda activate charles-cubicasa
pip install -r requirements.txt
# pour mon système : version de torch compatible avec la version de CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```


# Creating the LMDB database
```bash
python create_lmdb.py --txt train.txt
python create_lmdb.py --txt val.txt
python create_lmdb.py --txt test.txt
```
For smaller subsets you can use the `sample_data.txt` file.

multiplewarnings while creating the LMDB database:
```bash
libpng warning: iCCP: profile 'ICC Profile': 'CMYK': invalid ICC profile color space
# and in validation set:
libpng warning: iCCP: profile 'ICC Profile': 'tech': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'rTRC': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'gTRC': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'bTRC': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'desc': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'wtpt': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'bkpt': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'rXYZ': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'gXYZ': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'bXYZ': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'dmnd': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'dmdd': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'vued': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'view': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'lumi': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'meas': ICC profile tag start not a multiple of 4
libpng warning: iCCP: profile 'ICC Profile': 'tech': ICC profile tag start not a multiple of 4
```
i fixed issue in data loader when creating the LMDB database by using uint16 instead of uint8 : wall_id increments for each Wall/Railing in the SVG; it reached 256, which doesn't fit in np.uint8 (0-255). Changing wall_ids to np.uint16.

database size: 26G

**Debug datasets** (separate LMDB at `data/cubicasa5k/debug/cubi_lmdb/`, run from the repo root):

```bash
python create_lmdb.py --txt debug/train.txt --data-path data/cubicasa5k/ --lmdb data/cubicasa5k/debug/cubi_lmdb/ --overwrite
python create_lmdb.py --txt debug/val.txt --data-path data/cubicasa5k/ --lmdb data/cubicasa5k/debug/cubi_lmdb/
python create_lmdb.py --txt debug/test.txt --data-path data/cubicasa5k/ --lmdb data/cubicasa5k/debug/cubi_lmdb/
```

**Simple debug training** on the debug dataset:

seems that we should use --image-size 256 for representative runs : this is the default.
```bash
python train_simple.py --debug --segmentation-map room --data-path data/cubicasa5k/debug/ --image-size 128 --n-epoch 3 --batch-size 3
```

generate images on tensorboard:
```bash
python train_simple.py   --segmentation-map room   --data-path data/cubicasa5k/debug/    --n-epoch 100   --batch-size 3   --debug --plot-samples
# launch tensorboard
tensorboard --logdir runs_cubi --port 6006
```

premier entraînement avec les valeurs par défaut sur 350 epochs
```bash
nohup python train_simple.py   --segmentation-map room  --n-epoch 350 --plot-samples &
```

weighted cross-entropy:
```bash
python train_simple.py   --segmentation-map room  --n-epoch 350 --plot-samples --criterion weighted-cross-entropy
```

Si on fait simplement la moyenne de l'inverse de la fréquence, on obtient des poids très déséquilibrés, surtout pour les icons.
poids:
```bash
# formula : 
weights = 1.0 / (freqs + 1e-6)
weights = weights / weights.mean() # keep weights mean equal to 1
# rooms
[0.0469, 0.3466, 0.2120, 0.3620, 0.2044, 0.2325, 0.7592, 0.5222, 5.3676, 1.3282, 2.4445, 0.1737]
# icons
[7.6781e-04, 4.8298e-02, 1.0481e-01, 3.6481e-02, 7.8813e-02, 3.3583e-01,  1.9796e-01, 1.1102e-01, 7.4211e-01, 1.2648e+00, 8.0791e+00]
```
Il vaut mieux calculer avec la racine carrée et mettre un "clamping" pour éviter les valeurs infinies. Mais cela donne des poids semblables à toutes les classes sauf la classe dominante. On perd de l'information.
```bash
# formula : 
weights = 1.0 / torch.sqrt(freqs + 1e-6)
weights = torch.clamp(weights, max=5.0)
weights = weights / weights.mean()
# rooms
[0.3873, 1.0526, 0.8231, 1.0757, 0.8083, 0.8621, 1.2492, 1.2492, 1.2492, 1.2492, 1.2492, 0.7451]
# icons
[0.2228, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777]
```
Il vaut mieux utiliser le "effective number of samples". E_n=(1-Beta)^N/(1-Beta) Beta=0.9999, N=nombre de classes, et prendre w_n = 1/E_n.
```bash
# formula : 
E_n = (1-Beta)^N/(1-Beta)
w_n = 1/E_n
# rooms
[0.3873, 1.0526, 0.8231, 1.0757, 0.8083, 0.8621, 1.2492, 1.2492, 1.2492, 1.2492, 1.2492, 0.7451]
# icons
[0.2228, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777, 1.0777]
```

tensorboard:
```bash
tensorboard --logdir runs_cubi --port 6006
```
logs:
```bash
tail -f runs_cubi/train.log
```

# Visualizing the LMDB database
```bash
python lmdb_viewer.py
```
Then open in the browser: http://localhost:8080/

# training
## single task training
We first try to train the model on a single task.
For this, we created a new `train_single_task.py` script based on the `train.py` script.
It takes a new argument `--task` to specify the task to train on. Tasks are:
- `wall` for the wall segmentation task
- `icon` for the icon segmentation task
- `junction_heatmap` for the junction heatmap that gathers all types of junctions.
- `opening_heatmap` for the opening heatmap that gathers all types of openings.
- `icon_heatmap` for the icon heatmap that gathers all types of icons.
As we are running only one task, we use a simple loss function (that don't need to be trained).
- `CrossEntropyLoss` for segmentation tasks (wall and icon)
- `MSELoss` for heatmap tasks (junction, icon and opening)
As optimizer, we keep the possibility to use the same optimizers as in the `train.py` script, and benchmark the results.
