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
    DictToTensor,
    RandomCropToSizeTorch,
    RandomRotations,
    ResizePaddedTorch,
)

# Wall-point channel layout (see plan_wall_training.md):
#   0-3: junction arity (1=dead-end, 2=corner/L, 3=T-junction, 4=X-junction)
#   4:   opening endpoint (door or window gap, type-agnostic for v1)
N_WALL_CHANNELS = 5

# House.get_number() (floortrans/loaders/house.py:1207-1208) encodes
# channel = (arity - 1) * 4 + orientation for the 13 junction channels (0-12),
# so arity = channel // 4 for channels 0-11, and channel 12 (X-junction) is arity 4.
ARITY_CHANNEL_BY_JUNCTION_CHANNEL = {ch: min(ch // 4, 3) for ch in range(13)}
# Opening corner channels (left/right/up/down), house.py:649-673, all merge into one.
OPENING_JUNCTION_CHANNELS = (13, 14, 15, 16)
OPENING_CHANNEL = 4


def remap_heatmap_dict_to_wall_channels(heatmaps: dict) -> dict:
    """21-channel junction-type point dict -> 5-channel (arity + opening) point dict."""
    wall_heatmaps = {ch: [] for ch in range(N_WALL_CHANNELS)}
    for channel, points in heatmaps.items():
        channel = int(channel)
        if channel in ARITY_CHANNEL_BY_JUNCTION_CHANNEL:
            wall_heatmaps[ARITY_CHANNEL_BY_JUNCTION_CHANNEL[channel]].extend(points)
        elif channel in OPENING_JUNCTION_CHANNELS:
            wall_heatmaps[OPENING_CHANNEL].extend(points)
        # channels 17-20 (icon corners): dropped, out of scope.
    return wall_heatmaps


def build_wall_train_augmentations(args) -> Compose:
    """Same structure as build_full_train_augmentations, remapped to 5 wall channels."""
    sz = (args.image_size, args.image_size)
    dict_to_tensor = DictToTensor(n_channels=N_WALL_CHANNELS, kernel_size=args.kernel_px)
    if args.scale:
        return Compose(
            [
                RandomChoice(
                    [
                        RandomCropToSizeTorch(data_format="dict", size=sz),
                        ResizePaddedTorch((0, 0), data_format="dict", size=sz),
                    ]
                ),
                RandomRotations(format="wall"),
                dict_to_tensor,
                ColorJitterTorch(),
            ]
        )
    return Compose(
        [
            RandomCropToSizeTorch(data_format="dict", size=sz),
            RandomRotations(format="wall"),
            dict_to_tensor,
            ColorJitterTorch(),
        ]
    )


def build_wall_val_augmentations(args) -> Compose:
    """Native LMDB resolution at val/test (DictToTensor only; no resize/pad)."""
    dict_to_tensor = DictToTensor(n_channels=N_WALL_CHANNELS, kernel_size=args.kernel_px)
    return Compose([dict_to_tensor])


class WallLoader(Dataset):
    """Wall-point-only loader: 5 gaussian heatmap channels (arity + opening), no room/icon."""

    def __init__(self, data_path: str, txt_file: str, lmdb_env, augmentations):
        self.data_path = data_path.rstrip(os.sep) + os.sep
        self.folders = genfromtxt(self.data_path + txt_file, dtype="str")
        if self.folders.ndim == 0:
            self.folders = np.array([str(self.folders)])
        self.folders = np.array([str(f).strip() for f in self.folders], dtype=str)
        self.folders = self.folders[self.folders != ""]
        self.lmdb_env = lmdb_env
        self.augmentations = augmentations

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
        sample.setdefault("scale", 1.0)
        # Remap 21-channel junction-type dict -> 5-channel arity/opening dict before any
        # geometric augmentation, so downstream code only ever sees the wall channel layout.
        sample["heatmaps"] = remap_heatmap_dict_to_wall_channels(sample["heatmaps"])

        if self.augmentations is not None:
            sample = self.augmentations(sample)

        image = sample["image"].float()
        image = 2 * (image / 255.0) - 1.0
        # DictToTensor concatenated the 5 wall heatmap channels on top of whatever
        # room/icon label tensor was carried through for spatial-shape reference
        # (unused here) — keep only the heatmap channels, same slicing idiom as
        # FullLoader's `label[:21]` (room_icon_loaders.py:263).
        label = sample["label"][:N_WALL_CHANNELS].float()

        return {
            "image": image,  # (3, H, W) in [-1, 1]
            "label": label,  # (5, H, W): [0:4]=arity heatmaps, [4]=opening-endpoint heatmap
            "folder": self.folders[index],
        }
