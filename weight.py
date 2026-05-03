import json
import os
import pickle
import torch
import lmdb
import numpy as np
from tqdm import tqdm

# Must match train_simple.py / segmentation heads
N_ROOM_CLASSES = 12
N_ICON_CLASSES = 11

def build_class_counts_dict(lmdb_path="data/cubicasa5k/cubi_lmdb"):
    """Sum of per-class pixel counts over the whole LMDB (per image: bincount, then add)."""
    lmdb_path = os.path.abspath(lmdb_path)
    sum_room = np.zeros(N_ROOM_CLASSES, dtype=np.int64)
    sum_icon = np.zeros(N_ICON_CLASSES, dtype=np.int64)
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
                r = np.clip(np.asarray(lab[0]).ravel(), 0, N_ROOM_CLASSES - 1).astype(
                    np.int64
                )
                ic = np.clip(np.asarray(lab[1]).ravel(), 0, N_ICON_CLASSES - 1).astype(
                    np.int64
                )
                rc = np.bincount(r, minlength=N_ROOM_CLASSES)
                sum_room += rc
                icc = np.bincount(ic, minlength=N_ICON_CLASSES)
                sum_icon += icc
    finally:
        env.close()
    return {"room": sum_room.tolist(), "icon": sum_icon.tolist()}


class Weights:
    def __init__(self, counts):
        self.counts = counts

    def weights(self, method="effective_num"):
        if method == "effective_num":
            return self.effective_num()
        elif method == "inverse_sqrt_frequency":
            return self.inverse_sqrt_frequency()
        elif method == "inverse_double_sqrt_frequency":
            return self.inverse_double_sqrt_frequency()
        elif method == "inverse_frequency":
            return self.inverse_frequency()
        else:
            raise ValueError(f"Invalid method: {method}")

    def effective_num(self, beta=0.9999):
        effective_num = 1.0 - beta ** self.counts.float()
        weights = (1.0 - beta) / (effective_num + 1e-6)
        weights = weights / weights.mean()
        return weights

    def inverse_sqrt_frequency(self):
        weights = 1.0 / torch.sqrt(self.counts.float() + 1e-6)
        weights = weights / weights.mean()
        return weights

    def inverse_double_sqrt_frequency(self):
        weights = 1.0 / torch.pow(self.counts.float() + 1e-6, 0.25)
        weights = weights / weights.mean()
        return weights

    def inverse_frequency(self):
        weights = 1.0 / (self.counts.float() + 1e-6)
        weights = weights / weights.mean()
        return weights

if __name__ == "__main__":
    class_counts = build_class_counts_dict()
    with open("class_counts.json", "w") as f:
        json.dump(class_counts, f)
