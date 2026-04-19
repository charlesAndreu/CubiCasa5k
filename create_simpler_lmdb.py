import argparse
import os
import pickle
import shutil
from typing import Any

import lmdb
import torch.nn.functional as F
from tqdm import tqdm


def _slim_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed for simple room/icon training (no heatmaps)."""
    if "image" not in sample or "label" not in sample or "scale" not in sample:
        raise KeyError("sample must contain 'image', 'label', and 'scale'")
    label = sample["label"]
    if hasattr(label, "shape") and len(label.shape) >= 1 and label.shape[0] < 2:
        raise ValueError(
            f"expected label with at least 2 channels, got shape {tuple(label.shape)}"
        )

    out: dict[str, Any] = {
        "image": sample["image"],
        "label": label,
        "scale": sample["scale"],
    }
    if "folder" in sample:
        out["folder"] = sample["folder"]
    return out


def _resize_uniform_max_side(sample: dict[str, Any], max_side: int) -> dict[str, Any]:
    """
    Scale ``image`` and ``label`` so ``max(H, W) <= max_side`` (aspect preserved).
    Image: bilinear; label: nearest. Updates ``scale`` (pixels in original SVG per
    raster pixel — see ``FloorplanSVG``).
    """
    if max_side <= 0:
        return sample
    image = sample["image"]
    label = sample["label"]
    if not hasattr(image, "shape") or image.dim() != 3:
        raise ValueError(f"expected image (C,H,W) tensor, got {type(image)}")
    _, h, w = image.shape
    m = max(int(h), int(w))
    if m <= max_side:
        return sample

    s = max_side / float(m)
    new_h = max(1, int(round(h * s)))
    new_w = max(1, int(round(w * s)))

    img = image.float().unsqueeze(0)
    image_out = F.interpolate(
        img, size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    sample["image"] = image_out.to(dtype=image.dtype)

    lbl = label.float().unsqueeze(0)
    label_out = F.interpolate(lbl, size=(new_h, new_w), mode="nearest").squeeze(0)
    sample["label"] = label_out.round().to(dtype=label.dtype)

    sample["scale"] = float(sample["scale"]) * (w / float(new_w))
    return sample


def create_simpler_lmdb(
    source_lmdb: str,
    dest_dir: str,
    map_size: int = int(100e9),
    dry_run: bool = False,
    overwrite: bool = False,
    max_side: int = 768,
    dest_subdir: str = "simpler_lmdb",
) -> str:
    """
    Read all entries from ``source_lmdb`` and write slim copies under
    ``dest_dir`` / ``dest_subdir``.
    """
    source_lmdb = os.path.abspath(source_lmdb)
    dest_dir = os.path.abspath(dest_dir)
    dest_path = os.path.join(dest_dir, dest_subdir)

    if not os.path.isdir(source_lmdb):
        raise FileNotFoundError(f"source LMDB directory not found: {source_lmdb}")

    if dry_run:
        return dest_path

    if os.path.exists(dest_path):
        if not overwrite:
            raise FileExistsError(
                f"destination exists (use overwrite=True to replace): {dest_path}"
            )
        shutil.rmtree(dest_path)
    os.makedirs(dest_path, exist_ok=True)

    env_src = lmdb.open(
        source_lmdb, readonly=True, lock=False, readahead=False, max_readers=256
    )
    env_out = lmdb.open(dest_path, map_size=map_size)
    try:
        total = int(env_src.stat()["entries"])
        desc = "simpler_lmdb"
        if max_side > 0:
            desc += f"_max{max_side}"
        with env_src.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, blob in tqdm(cursor, total=total, desc=desc):
                if blob is None:
                    continue
                sample = pickle.loads(blob)
                slim = _slim_from_sample(sample)
                if max_side > 0:
                    slim = _resize_uniform_max_side(slim, max_side)
                payload = pickle.dumps(slim, protocol=pickle.HIGHEST_PROTOCOL)
                with env_out.begin(write=True) as txn_w:
                    txn_w.put(key, payload)
    finally:
        env_src.close()
        env_out.close()

    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create slimmer LMDB: drop heatmaps; optionally resize so max(H,W) <= max-side."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/cubicasa5k/cubi_lmdb",
        help="Path to existing Cubi LMDB directory (read-only).",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="data/cubicasa5k",
        help="Parent directory for the output LMDB folder.",
    )
    parser.add_argument(
        "--dest-subdir",
        type=str,
        default="simpler_lmdb",
        help="Output folder name under --dest-dir (e.g. simpler_lmdb_max768).",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=768,
        help=(
            "If > 0, scale each sample so max(height,width) is at most this (uniform). "
            "If 0, only strip heatmaps; keep native resolution."
        ),
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=int(100e9),
        help="map_size for new LMDB environment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print destination path; do not read/write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination LMDB directory if present.",
    )
    args = parser.parse_args()

    out = create_simpler_lmdb(
        args.source,
        args.dest_dir,
        map_size=args.map_size,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        max_side=args.max_side,
        dest_subdir=args.dest_subdir,
    )
    print("Created:" if not args.dry_run else "Would create:")
    print(" ", out)
    if not args.dry_run:
        print(
            "  train_simple loads <data-path>/cubi_lmdb/ — symlink or rename this output "
            "to cubi_lmdb, or use --dest-subdir cubi_lmdb under --dest-dir."
        )


if __name__ == "__main__":
    main()
