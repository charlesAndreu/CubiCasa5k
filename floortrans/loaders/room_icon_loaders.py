import os
import pickle

import numpy as np
from numpy import genfromtxt
from torch.utils.data import Dataset
from torchvision.transforms import RandomChoice

from floortrans.loaders.augmentations import (
    ColorJitterTorch,
    Compose,
    RandomCropToSizeTorch,
    RandomRotations,
    ResizePaddedTorch,
)


def build_simple_train_augmentations(args) -> Compose:
    """Crop/resize + rotation + color jitter; no DictToTensor (no heatmap channels)."""
    sz = (args.image_size, args.image_size)
    if args.scale:
        return Compose(
            [
                RandomChoice(
                    [
                        RandomCropToSizeTorch(data_format="dict", size=sz),
                        ResizePaddedTorch((0, 0), data_format="dict", size=sz),
                    ]
                ),
                RandomRotations(format="cubi"),
                ColorJitterTorch(),
            ]
        )
    return Compose(
        [
            RandomCropToSizeTorch(data_format="dict", size=sz),
            RandomRotations(format="cubi"),
            ColorJitterTorch(),
        ]
    )


def build_simple_val_augmentations(args) -> Compose:
    """Deterministic resize/pad to ``image_size`` (no jitter, no heatmaps)."""
    sz = (args.image_size, args.image_size)
    return Compose([ResizePaddedTorch((0, 0), data_format="dict", size=sz)])


class _SimpleSegLMDBDataset(Dataset):
    """Read Cubi LMDB pickles; apply ``augmentations``; return image + single-channel label."""

    def __init__(
        self,
        data_path: str,
        txt_file: str,
        lmdb_env,
        augmentations,
        seg_channel: int,
    ):
        self.data_path = data_path.rstrip(os.sep) + os.sep
        self.folders = genfromtxt(self.data_path + txt_file, dtype="str")
        if self.folders.ndim == 0:
            self.folders = np.array([str(self.folders)])
        self.lmdb_env = lmdb_env
        self.augmentations = augmentations
        self.seg_channel = int(seg_channel)

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, index: int) -> dict:
        key = self.folders[index].encode("ascii")
        with self.lmdb_env.begin(write=False) as txn:
            blob = txn.get(key)
        sample = pickle.loads(blob)
        sample["heatmaps"] = {}
        if self.augmentations is not None:
            sample = self.augmentations(sample)
        image = sample["image"].float()
        image = 2 * (image / 255.0) - 1.0
        label = sample["label"][self.seg_channel : self.seg_channel + 1].long()
        return {
            "image": image,
            "label": label,
            "folder": self.folders[index],
        }


class RoomLoader(_SimpleSegLMDBDataset):
    """Room / wall raster (label channel 0) only; no heatmaps."""

    def __init__(self, data_path: str, txt_file: str, lmdb_env, augmentations):
        super().__init__(data_path, txt_file, lmdb_env, augmentations, seg_channel=0)


class IconLoader(_SimpleSegLMDBDataset):
    """Icon raster (label channel 1) only; no heatmaps."""

    def __init__(self, data_path: str, txt_file: str, lmdb_env, augmentations):
        super().__init__(data_path, txt_file, lmdb_env, augmentations, seg_channel=1)
