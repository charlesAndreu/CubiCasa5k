import argparse
import copy
import logging
import os
import time

import lmdb
import torch
from torch.utils import data

from floortrans.loaders.room_icon_loaders import (
    IconLoader,
    RoomLoader,
    build_simple_train_augmentations,
)


def _build_train_loader(args, device: torch.device):
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
    train_aug = build_simple_train_augmentations(args)
    LoaderCls = RoomLoader if args.segmentation_map == "room" else IconLoader
    train_set = LoaderCls(args.data_path, "train.txt", lmdb_env, train_aug)

    if args.debug:
        num_workers = 0
    else:
        num_workers = max(0, args.num_workers)

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))

    return data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        **dl_kwargs,
    )


def run_benchmark(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = 0 if args.debug else max(0, args.num_workers)
    print("---")
    print(
        "config:",
        f"image_size={args.image_size}",
        f"batch_size={args.batch_size}",
        f"num_workers={nw}",
        f"scale={args.scale}",
        f"to_gpu={args.to_gpu}",
    )
    loader = _build_train_loader(args, device)

    # Warm-up: workers start, caches warm
    it = iter(loader)
    for _ in range(args.warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        _touch_batch(batch, args.to_gpu, device)

    if args.to_gpu and device.type == "cuda":
        torch.cuda.synchronize()

    it = iter(loader)
    n = 0
    total_samples = 0
    first_batch_shape = None
    t0 = time.perf_counter()
    for i in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        _touch_batch(batch, args.to_gpu, device)
        if first_batch_shape is None:
            first_batch_shape = tuple(batch["image"].shape)
        n += 1
        total_samples += batch["image"].shape[0]

    if args.to_gpu and device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    bps = n / elapsed
    sps = total_samples / elapsed
    print(f"device (for pin_memory): {device.type}")
    print(f"batch tensor shape (N,C,H,W): {first_batch_shape}")
    print(f"batches timed: {n}  warmup (not timed): {args.warmup_batches}")
    print(f"elapsed: {elapsed:.3f}s")
    print(f"batches/s: {bps:.2f}")
    print(f"samples/s: {sps:.2f}")
    print(f"ms/batch: {1000 * elapsed / n:.2f}")
    if args.to_gpu:
        print("(includes host->device copy + cuda synchronize per batch)")


def _touch_batch(batch, to_gpu: bool, device: torch.device) -> None:
    """Force tensors to be materialized / transferred so work is not optimized away."""
    img = batch["image"]
    lbl = batch["label"]
    if to_gpu and device.type == "cuda":
        img = img.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
    # Cheap reduction so the batch is fully read
    _ = float(img.sum()) + float(lbl.float().sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark train_simple-style DataLoader throughput."
    )
    parser.add_argument(
        "--segmentation-map",
        type=str,
        required=True,
        choices=["room", "icon"],
        help="Same as train_simple.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/cubicasa5k/",
        help="Directory containing cubi_lmdb/ and train.txt.",
    )
    parser.add_argument("--batch-size", type=int, default=26)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--scale",
        nargs="?",
        type=bool,
        default=False,
        const=True,
        help="Same RandomChoice augment as train_simple --scale.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="num_workers=0 (single process loading).",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=50,
        help="Number of batches to time (after warmup).",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Batches to run before starting the timer.",
    )
    parser.add_argument(
        "--to-gpu",
        action="store_true",
        help="Copy each batch to CUDA and synchronize (includes H2D in timing).",
    )
    parser.add_argument(
        "--sweep-image-sizes",
        type=int,
        nargs="+",
        metavar="N",
        default=None,
        help=(
            "Run one timed benchmark per size (e.g. 128 256 384). "
            "Same as repeating the command with different --image-size."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    sizes = args.sweep_image_sizes if args.sweep_image_sizes else [args.image_size]
    for i, sz in enumerate(sizes):
        run_args = copy.copy(args)
        run_args.image_size = sz
        if len(sizes) > 1:
            print(f"\n### sweep {i + 1}/{len(sizes)}: --image-size {sz}\n")
        run_benchmark(run_args)


if __name__ == "__main__":
    main()
