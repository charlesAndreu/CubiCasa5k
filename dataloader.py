import os

import lmdb
from torch.utils.data import DataLoader

from floortrans.loaders.room_icon_loaders import (
    ICON_MINI_DEFAULT_CLASS,
    ICON_MINI_MAPPING,
    ROOM_MINI_DEFAULT_CLASS,
    ROOM_MINI_MAPPING,
    RoomLoader,
    IconLoader,
    build_simple_train_augmentations,
    build_simple_val_augmentations,
)


def n_segmentation_classes(segmentation_map):
    """Head output channels: full room/icon vs reduced mini label spaces."""
    if segmentation_map == "room-mini":
        return 3
    if segmentation_map == "icon-mini":
        return 4
    if segmentation_map.startswith("room"):
        return 12
    return 11


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
        "Room" if segmentation_map.startswith("room") else "Icon",
    )
    train_aug = build_simple_train_augmentations(args)
    val_aug = build_simple_val_augmentations(args)
    LoaderCls = RoomLoader if segmentation_map.startswith("room") else IconLoader
    # set label space (default or mini)
    mini = "mini" in segmentation_map
    logger.info(f"Using mini label space: {mini}")
    # build train and val datasets
    train_set = LoaderCls(args.data_path, "train.txt", lmdb_env, train_aug, mini=mini)
    val_set = LoaderCls(args.data_path, "val.txt", lmdb_env, val_aug, mini=mini)

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


def build_cubi_casa5k_eval_dataloaders(args, segmentation_map, device):
    """Open LMDB, build test dataset and PyTorch ``DataLoader``s."""
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

    print(
        f"LMDB eval loader is {'Room' if segmentation_map.startswith('room') else 'Icon'}Loader"
    )
    LoaderCls = RoomLoader if segmentation_map.startswith("room") else IconLoader
    test_set = LoaderCls(
        args.data_path,
        "test.txt",
        lmdb_env,
        None,
        mini="mini" in segmentation_map,
    )

    num_workers = max(0, args.num_workers)
    persistent_workers = num_workers > 0
    pin_memory = device.type == "cuda"

    return DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
