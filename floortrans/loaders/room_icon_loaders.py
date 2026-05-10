import os
import pickle

import numpy as np
from numpy import genfromtxt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomChoice

from floortrans.loaders.augmentations import (
    ColorJitterTorch,
    Compose,
    RandomCropToSizeTorch,
    RandomRotations,
    ResizePaddedTorch,
)

# Reduced ("mini") label spaces — keep here to avoid circular imports with ``dataloader``.
ROOM_MINI_MAPPING = {0: 0, 2: 1, 8: 1}  # bg -> 0; wall+railing -> 1
ROOM_MINI_DEFAULT_CLASS = 2  # rest -> 2
ICON_MINI_MAPPING = {0: 0, 1: 1, 2: 2}  # bg -> 0; window -> 1; door -> 2
ICON_MINI_DEFAULT_CLASS = 3  # rest -> 3


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
        mini: bool,
        mini_mapping: dict = None,
        mini_default_class: int = None,
    ):
        self.data_path = data_path.rstrip(os.sep) + os.sep
        self.folders = genfromtxt(self.data_path + txt_file, dtype="str")
        if self.folders.ndim == 0:
            self.folders = np.array([str(self.folders)])
        self.folders = np.array([str(f).strip() for f in self.folders], dtype=str)
        self.folders = self.folders[self.folders != ""]
        self.lmdb_env = lmdb_env
        self.augmentations = augmentations
        self.seg_channel = int(seg_channel)
        self.mini = mini
        self.mini_mapping = mini_mapping
        self.mini_default_class = mini_default_class

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, index: int) -> dict:
        key = self.folders[index].encode("ascii")
        with self.lmdb_env.begin(write=False) as txn:
            blob = txn.get(key)
        if blob is None:
            raise KeyError(
                f"LMDB key missing for '{self.folders[index]}' under {self.data_path}"
            )
        sample = pickle.loads(blob)
        sample["heatmaps"] = {}
        if self.augmentations is not None:
            sample = self.augmentations(sample)
        image = sample["image"].float()
        image = 2 * (image / 255.0) - 1.0
        label = sample["label"][self.seg_channel : self.seg_channel + 1].long()
        if self.mini:
            label = self.get_mini_label(label)
        return {
            "image": image,
            "label": label,
            "folder": self.folders[index],
        }

    def get_mini_label(self, label: torch.Tensor) -> torch.Tensor:
        """Map full-resolution class ids to mini ids; output shape (1, H, W) like non-mini."""
        plane = label[0]
        h, w = plane.shape
        new_label = torch.full(
            (h, w),
            self.mini_default_class,
            dtype=torch.long,
            device=plane.device,
        )
        for k, v in self.mini_mapping.items():
            new_label[plane == k] = v
        return new_label.unsqueeze(0)


class RoomLoader(_SimpleSegLMDBDataset):
    """Room / wall raster (label channel 0) only; no heatmaps."""

    def __init__(
        self, data_path: str, txt_file: str, lmdb_env, augmentations, mini=False
    ):
        if mini:
            mini_mapping = ROOM_MINI_MAPPING
            mini_default_class = ROOM_MINI_DEFAULT_CLASS
        else:
            mini_mapping = None
            mini_default_class = None
        super().__init__(
            data_path,
            txt_file,
            lmdb_env,
            augmentations,
            seg_channel=0,
            mini=mini,
            mini_mapping=mini_mapping,
            mini_default_class=mini_default_class,
        )


class IconLoader(_SimpleSegLMDBDataset):
    """Icon raster (label channel 1) only; no heatmaps."""

    def __init__(
        self, data_path: str, txt_file: str, lmdb_env, augmentations, mini=False
    ):
        if mini:
            mini_mapping = ICON_MINI_MAPPING
            mini_default_class = ICON_MINI_DEFAULT_CLASS
        else:
            mini_mapping = None
            mini_default_class = None
        super().__init__(
            data_path,
            txt_file,
            lmdb_env,
            augmentations,
            seg_channel=1,
            mini=mini,
            mini_mapping=mini_mapping,
            mini_default_class=mini_default_class,
        )
