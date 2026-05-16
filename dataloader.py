import os

import lmdb
from torch.utils.data import DataLoader


from floortrans.loaders.room_icon_loaders import (
    RoomLoader,
    IconLoader,
    FullLoader,
    build_full_train_augmentations,
    build_full_val_augmentations,
    build_simple_train_augmentations,
    build_simple_val_augmentations,
)
from floortrans.loaders.augmentations import Compose, DictToTensor


def n_segmentation_classes(segmentation_map):
    """Head output channels: full room/icon vs reduced mini label spaces."""
    if segmentation_map == "room-mini":
        return 3
    if segmentation_map == "icon-mini":
        return 4
    if segmentation_map.startswith("room"):
        return 12
    return 11


def build_cubicasa5k_simple_dataloaders(args, segmentation_map, device, logger):
    """Open LMDB, build train/val datasets and PyTorch DataLoaders."""
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


def build_cubicasa5k_full_dataloaders(args, device, logger):
    """Open LMDB, build train/val datasets and PyTorch DataLoaders."""
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
    train_aug = build_full_train_augmentations(args)
    val_aug = build_full_val_augmentations(args)
    logger.info("Loading full data (heatmaps + room + icon)...")
    logger.info("Train at %sx%s; validation at native LMDB resolution", args.image_size, args.image_size)
    train_set = FullLoader(
        args.data_path, "train.txt", lmdb_env, augmentations=train_aug
    )
    val_set = FullLoader(args.data_path, "val.txt", lmdb_env, augmentations=val_aug)

    if args.debug:
        num_workers = 0
        print("In debug mode.")
        logger.info("In debug mode.")
    else:
        num_workers = max(0, args.num_workers)

    logger.info(
        "Full DataLoader num_workers=%s prefetch_factor=%s",
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


def build_cubicasa5k_full_eval_dataloaders(args, device):
    """Full test loader at native LMDB resolution (DictToTensor only)."""
    return build_cubicasa5k_full_eval_dataloaders_native_res(args, device)


def build_cubicasa5k_full_eval_dataloaders_native_res(args, device):
    """
    Test dataloader for full models at native (LMDB) image resolution.
    No resize/pad — only DictToTensor rasterises heatmap points. batch_size=1
    (variable image sizes mean any larger batch would require collation/padding).
    """
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
    eval_aug = build_full_val_augmentations(args)
    print("LMDB full eval loader (native res): FullLoader (heatmaps + room + icon)")
    test_set = FullLoader(args.data_path, "test.txt", lmdb_env, augmentations=eval_aug)

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


def build_cubicasa5k_simple_eval_dataloaders(args, segmentation_map, device):
    """Open LMDB, build test dataset and PyTorch DataLoaders."""
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
