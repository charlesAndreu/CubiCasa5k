# environnement virtuel python

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


# création de la base de données lmdb
```bash
python create_lmdb.py --txt train.txt
python create_lmdb.py --txt val.txt
python create_lmdb.py --txt test.txt
```
for more little datasets, you can use `sample_data.txt` file.git 

# visualisation de la base de données lmdb
```bash
python lmdb_viewer.py
```
puis dans le navigateur web: http://localhost:8080/

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
