#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import lmdb
import matplotlib.pyplot as plt
import numpy as np


ROOM_COLORS = [
    "#DCDCDC",
    "#b3de69",
    "#000000",
    "#8dd3c7",
    "#fdb462",
    "#fccde5",
    "#80b1d3",
    "#808080",
    "#fb8072",
    "#696969",
    "#577a4d",
    "#ffffb3",
]

ICON_COLORS = [
    "#DCDCDC",
    "#8dd3c7",
    "#b15928",
    "#fdb462",
    "#ffff99",
    "#fccde5",
    "#80b1d3",
    "#808080",
    "#fb8072",
    "#696969",
    "#577a4d",
]

# Wall/junction channels in CubiCasa GT points dict
WALL_POINT_CHANNELS = list(range(0, 13))


def _hex_to_rgb01(hex_color: str):
    h = hex_color.lstrip("#")
    return [int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0]


def _indexed_to_rgb(index_map: np.ndarray, colors_hex):
    palette = np.array([_hex_to_rgb01(c) for c in colors_hex], dtype=np.float32)
    idx = np.clip(np.rint(index_map), 0, len(colors_hex) - 1).astype(np.int64)
    return palette[idx]


def _gaussian_kernel2d(size: int = 13, sigma: float = 0.25):
    if size <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    over_sigma = 1.0 / (sigma * size)
    mean = 0.5 * size + 0.5
    u = np.arange(1, size + 1, dtype=np.float32)
    du = (u - mean) * over_sigma
    du2, dv2 = np.broadcast_arrays(du[np.newaxis, :], du[:, np.newaxis])
    return np.exp(-0.5 * (du2 * du2 + dv2 * dv2)).astype(np.float32)


def _points_to_heatmap(points_dict, channels, height, width, kernel_size=13):
    out = np.zeros((height, width), dtype=np.float32)
    k = _gaussian_kernel2d(kernel_size)
    r = kernel_size // 2

    for ch in channels:
        for point in points_dict.get(ch, []):
            x = int(round(point[0]))
            y = int(round(point[1]))
            if x < 0 or y < 0 or x >= width or y >= height:
                continue

            y0 = max(0, y - r)
            y1 = min(height, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(width, x + r + 1)

            ky0 = y0 - (y - r)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (x - r)
            kx1 = kx0 + (x1 - x0)

            out[y0:y1, x0:x1] += k[ky0:ky1, kx0:kx1]
    return out


def _get_key(env, key_arg, index_arg):
    with env.begin(write=False) as txn:
        keys = [k.decode("utf-8") for k, _ in txn.cursor()]

    if not keys:
        raise RuntimeError("LMDB contains no keys.")

    if key_arg:
        if key_arg not in keys:
            raise KeyError(f"Key not found: {key_arg}")
        return key_arg

    idx = index_arg if index_arg is not None else 0
    if idx < 0 or idx >= len(keys):
        raise IndexError(f"Index {idx} out of range [0, {len(keys)-1}]")
    return keys[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Export one 2x2 GT panel from CubiCasa LMDB."
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        default="data/cubicasa5k/debug/cubi_lmdb",
        help="Path to LMDB folder (containing data.mdb).",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Exact LMDB key to export (overrides --index).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Key index to export when --key is not provided.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=13,
        help="Gaussian kernel size for wall-point heatmap smoothing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/lmdb_gt_panel.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    env = lmdb.open(
        args.lmdb_path,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=32,
    )

    key = _get_key(env, args.key, args.index)
    with env.begin(write=False) as txn:
        raw = txn.get(key.encode("utf-8"))
    if raw is None:
        raise RuntimeError(f"Missing key in LMDB: {key}")

    sample = pickle.loads(raw)
    image = np.asarray(sample["image"])  # CHW uint8
    label = np.asarray(sample["label"])  # (2, H, W) expected
    points = sample.get("heatmaps", {})

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image CHW with 3 channels, got {image.shape}")
    if label.ndim != 3 or label.shape[0] < 2:
        raise ValueError(f"Expected label with at least 2 channels, got {label.shape}")

    # Normalize points dict keys to int.
    points_int = {}
    if isinstance(points, dict):
        for k, v in points.items():
            try:
                points_int[int(k)] = v
            except Exception:
                continue

    image_rgb = np.moveaxis(image, 0, -1)
    room_rgb = _indexed_to_rgb(label[0], ROOM_COLORS)
    icon_rgb = _indexed_to_rgb(label[1], ICON_COLORS)
    h, w = label.shape[1], label.shape[2]
    wall_hm = _points_to_heatmap(
        points_int,
        WALL_POINT_CHANNELS,
        height=h,
        width=w,
        kernel_size=args.kernel_size,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax in axes.ravel():
        # Force identical geometry across all panels.
        ax.set_aspect("equal")

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(room_rgb)
    axes[0, 1].set_title("Room segmentation (GT)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(icon_rgb)
    axes[1, 0].set_title("Icon segmentation (GT)")
    axes[1, 0].axis("off")

    hm = axes[1, 1].imshow(wall_hm, cmap="hot")
    axes[1, 1].set_title("Wall points heatmap (GT, ch 0-12)")
    axes[1, 1].axis("off")
    # Avoid per-axis colorbar because it shrinks only this panel and breaks alignment.
    vmax = float(np.max(wall_hm)) if wall_hm.size else 0.0
    axes[1, 1].text(
        0.99,
        0.01,
        f"max={vmax:.3f}",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.4, edgecolor="none"),
    )

    fig.suptitle(f"LMDB sample: {key}", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
