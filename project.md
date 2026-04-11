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