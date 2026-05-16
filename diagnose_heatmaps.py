"""
Quick diagnostic for full-model heatmap outputs.

Loads a trained checkpoint, runs forward on the first N test samples at native
resolution, and prints per-channel statistics for the 21 raw heatmap channels:

    min, max, baseline (median), 99.9th percentile (peaks), fraction > 0.4

This helps decide whether the model needs to be retrained with
`n_heatmap_channels=21` + sigmoid (and ideally `BCEWithLogitsLoss`).

For visual inspection, --save-png writes 21-panel grids per sample under
<run_dir>/heatmap_diagnostic/ (raw and sigmoid side by side).

Usage:
    python diagnose_heatmaps.py <run_dir> [--n-samples 3] [--save-png]
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_full import (
    FullSegEvaluator,
    N_HEATMAPS,
    load_eval_args,
)


def fmt(arr):
    return " ".join(f"{x:+.3f}" for x in arr)


def print_per_channel_stats(label, hm):
    """hm: (21, H, W) numpy."""
    mins = hm.min(axis=(1, 2))
    maxs = hm.max(axis=(1, 2))
    p999 = np.quantile(hm, 0.999, axis=(1, 2))
    p50 = np.median(hm, axis=(1, 2))
    frac_above = (hm > 0.4).mean(axis=(1, 2))

    print(f"\n{label}")
    print(f"  per-channel min:      [{fmt(mins)}]")
    print(f"  per-channel max:      [{fmt(maxs)}]")
    print(f"  per-channel p50:      [{fmt(p50)}]")
    print(f"  per-channel p99.9:    [{fmt(p999)}]")
    print(f"  per-channel frac>0.4: [{' '.join(f'{x:.4f}' for x in frac_above)}]")

    print(
        f"  ALL: min={hm.min():+.3f}  max={hm.max():+.3f}  "
        f"median={np.median(hm):+.3f}  p99.9={np.quantile(hm, 0.999):+.3f}  "
        f"frac>0.4={(hm > 0.4).mean():.4f}"
    )


def save_heatmap_grid(path, hm, title):
    """21 channels in a 3x7 grid, shared color scale."""
    vmin = float(hm.min())
    vmax = float(hm.max())
    fig, axes = plt.subplots(3, 7, figsize=(21, 9))
    for ch in range(N_HEATMAPS):
        ax = axes[ch // 7, ch % 7]
        im = ax.imshow(hm[ch], vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_title(f"ch {ch}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{title}  (vmin={vmin:.3f}, vmax={vmax:.3f})", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="training run dir with args.json + checkpoint")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="also save raw + sigmoid heatmap grids under <run_dir>/heatmap_diagnostic/",
    )
    args_cli = parser.parse_args()

    args = load_eval_args(args_cli.run_dir)
    evaluator = FullSegEvaluator(args)
    testloader = evaluator.dataloader_setup()
    model = evaluator.model_setup()
    model.eval()

    out_dir = os.path.join(args_cli.run_dir, "heatmap_diagnostic")
    if args_cli.save_png:
        os.makedirs(out_dir, exist_ok=True)

    device = evaluator.device
    all_raw = []
    all_sig = []

    with torch.no_grad():
        for idx, sample in enumerate(testloader):
            if idx >= args_cli.n_samples:
                break
            images = sample["image"].to(
                device, non_blocking=(device.type == "cuda")
            )
            outputs = model(images)  # (1, 28, h, w)
            hm_raw = outputs[0, :N_HEATMAPS].detach().cpu().float().numpy()
            hm_sig = (
                torch.sigmoid(outputs[0, :N_HEATMAPS]).detach().cpu().float().numpy()
            )
            all_raw.append(hm_raw)
            all_sig.append(hm_sig)

            print(
                f"\n========== Sample {idx}  shape={hm_raw.shape}  "
                f"folder={sample.get('folder', ['?'])[0]} =========="
            )
            print_per_channel_stats("RAW (model output, no activation)", hm_raw)
            print_per_channel_stats("SIGMOID(raw)", hm_sig)

            if args_cli.save_png:
                save_heatmap_grid(
                    os.path.join(out_dir, f"sample_{idx:03d}_raw.png"),
                    hm_raw,
                    f"Sample {idx} RAW",
                )
                save_heatmap_grid(
                    os.path.join(out_dir, f"sample_{idx:03d}_sigmoid.png"),
                    hm_sig,
                    f"Sample {idx} SIGMOID",
                )

    # aggregate verdict
    raw_concat = np.concatenate([h.reshape(N_HEATMAPS, -1) for h in all_raw], axis=1)
    sig_concat = np.concatenate([h.reshape(N_HEATMAPS, -1) for h in all_sig], axis=1)

    print("\n\n=================== AGGREGATE ===================")
    print_per_channel_stats("RAW   (all samples)", raw_concat.reshape(N_HEATMAPS, -1, 1))
    print_per_channel_stats("SIGMOID (all samples)", sig_concat.reshape(N_HEATMAPS, -1, 1))

    raw_peak = float(np.quantile(raw_concat, 0.999))
    raw_baseline = float(np.median(raw_concat))
    raw_min = float(raw_concat.min())
    raw_max = float(raw_concat.max())
    print("\n=================== VERDICT ===================")
    print(
        f"raw    overall min={raw_min:+.3f}  max={raw_max:+.3f}  "
        f"baseline(median)={raw_baseline:+.3f}  peak(p99.9)={raw_peak:+.3f}"
    )

    if 0.7 <= raw_peak and abs(raw_baseline) < 0.1 and raw_max < 1.5 and raw_min > -0.3:
        bucket = "CLEAN — model fits the [0,1] target well"
        advice = (
            "Keep training as-is. In eval, drop the sigmoid workaround in eval_full.py "
            "(pass raw heatmaps) and post-processing threshold 0.4 is fine."
        )
    elif raw_peak < 0.7 or abs(raw_baseline) > 0.1:
        bucket = "UNDERFIT / UNCALIBRATED"
        advice = (
            "Try: (a) keep current model, lower POSTPROC_THRESHOLD to ~0.5 of raw peak "
            "and cap postproc resolution to 256. (b) If still poor: retrain with "
            "n_heatmap_channels=21 + BCEWithLogitsLoss for heatmap channels."
        )
    else:
        bucket = "BLOWN OUT / SCATTERED"
        advice = (
            "Retrain. Switch to n_heatmap_channels=21 + BCEWithLogitsLoss. "
            "Current MSE-without-sigmoid is producing very out-of-range outputs."
        )
    print(f"\nbucket : {bucket}")
    print(f"advice : {advice}")


if __name__ == "__main__":
    main()
