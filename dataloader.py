import os

import lmdb
from torch.utils.data import DataLoader

from floortrans.loaders.room_icon_loaders import (
    RoomLoader,
    IconLoader,
    build_simple_train_augmentations,
    build_simple_val_augmentations,
)


def build_cubi_casa5k_dataloaders(args, segmentation_map, device, logger):
    """Open LMDB, build train/val datasets and PyTorch ``DataLoader``s."""
    logger.info("Loading data...")
    root = args.data_path.rstrip(os.sep)
    lmdb_path = os.path.join(root, "cubi_lmdb")
    lmdb_env = lmdb.open(
        lmdb_path,
        readonly=True,
        max_readers=16,
        lock=False,
        readahead=True,
        meminit=False,
    )

    logger.info(
        "LMDB loader is %sLoader",
        "Room" if segmentation_map == "room" else "Icon",
    )
    train_aug = build_simple_train_augmentations(args)
    val_aug = build_simple_val_augmentations(args)
    LoaderCls = RoomLoader if segmentation_map == "room" else IconLoader
    train_set = LoaderCls(args.data_path, "train.txt", lmdb_env, train_aug)
    val_set = LoaderCls(args.data_path, "val.txt", lmdb_env, val_aug)

    if args.debug:
        num_workers = 0
        print("In debug mode.")
        logger.info("In debug mode.")
    else:
        num_workers = max(0, args.num_workers)

    logger.info(
        "DataLoader num_workers=%s prefetch_factor=%s",
        num_workers,
        max(2, int(args.prefetch_factor)) if num_workers > 0 else "n/a",
    )

    dl_common = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        dl_common["prefetch_factor"] = max(2, int(args.prefetch_factor))

    trainloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        **dl_common,
    )
    valloader = DataLoader(
        val_set,
        batch_size=1,
        **dl_common,
    )
    return trainloader, valloader
