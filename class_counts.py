import os
import pickle

import lmdb
import numpy as np
from tqdm import tqdm

# Must match train_simple.py / segmentation heads
N_ROOM_CLASSES = 12
N_ICON_CLASSES = 11


def _label_to_numpy_ch(label_2hw, ch: int) -> np.ndarray:
    return np.asarray(label_2hw[ch]).ravel()


def build_class_counts_dict(lmdb_path="data/cubicasa5k/cubi_lmdb"):
    """Sum of per-class pixel counts over the whole LMDB (per image: bincount, then add)."""
    lmdb_path = os.path.abspath(lmdb_path)
    sum_room = np.zeros(N_ROOM_CLASSES, dtype=np.float64)
    sum_icon = np.zeros(N_ICON_CLASSES, dtype=np.float64)
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=16,
    )
    try:
        n_entries = int(env.stat()["entries"])
        with env.begin(write=False) as txn:
            for _key, value in tqdm(
                txn.cursor(),
                total=n_entries,
                desc="Class counts",
                ncols=80,
            ):
                sample = pickle.loads(value)
                lab = sample["label"]
                r = np.clip(_label_to_numpy_ch(lab, 0), 0, N_ROOM_CLASSES - 1).astype(
                    np.int64
                )
                ic = np.clip(_label_to_numpy_ch(lab, 1), 0, N_ICON_CLASSES - 1).astype(
                    np.int64
                )
                rc = np.bincount(r, minlength=N_ROOM_CLASSES).astype(np.float64)
                sum_room += rc
                icc = np.bincount(ic, minlength=N_ICON_CLASSES).astype(np.float64)
                sum_icon += icc
    finally:
        env.close()
    return {"room": sum_room.tolist(), "icon": sum_icon.tolist()}
