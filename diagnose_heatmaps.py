"""
Heatmap threshold diagnostic for full-model eval / post-processing.

GT heatmaps are sparse Gaussians (a few junctions per channel). Post-processing
uses extract_local_max(..., threshold=0.4) — this script compares how many peaks
are found on GT vs prediction at several thresholds, using the same peak extractor.

Usage:
    python diagnose_heatmaps.py <run_dir> [--n-samples 5] [--save-png]

Outputs:
  - Per-sample tables: GT peak count vs pred peak count per channel (0–20)
  - Threshold sweep: total pred/GT peak ratio (active channels only)
  - Suggested POSTPROC_THRESHOLD for eval_full.py
"""

import argparse
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_full import FullSegEvaluator, N_HEATMAPS, load_eval_args
from floortrans.post_prosessing import extract_local_max

# Same grouping as post_prosessing (wall / opening / icon)
CHANNEL_GROUPS = {
    "wall (0–12)": (0, 13, True),
    "opening (13–16)": (13, 17, False),
    "icon (17–20)": (17, 21, True),
}

DEFAULT_THRESHOLDS = np.arange(0.15, 0.65, 0.05)
MAX_POINTS_PER_CHANNEL = 100


def count_peaks_channel(
    hm_hw: np.ndarray,
    threshold: float,
    close_point_suppression: bool,
) -> int:
    """Same peak picking as get_wall_lines / get_icon_polygon."""
    pts = extract_local_max(
        hm_hw,
        MAX_POINTS_PER_CHANNEL,
        [0, 0],
        threshold,
        close_point_suppression=close_point_suppression,
    )
    return len(pts)


def count_peaks_all_channels(hm_chw: np.ndarray, threshold: float) -> np.ndarray:
    """Per-channel peak counts (21,)."""
    counts = np.zeros(N_HEATMAPS, dtype=int)
    for ch in range(N_HEATMAPS):
        _, _, close_sup = _channel_close_sup(ch)
        counts[ch] = count_peaks_channel(
            hm_chw[ch], threshold, close_point_suppression=close_sup
        )
    return counts


def _channel_close_sup(ch: int) -> tuple:
    for _name, (lo, hi, close_sup) in CHANNEL_GROUPS.items():
        if lo <= ch < hi:
            return lo, hi, close_sup
    return 0, N_HEATMAPS, False


def upsample_pred_heatmaps(pred_chw: torch.Tensor, target_hw) -> np.ndarray:
    """Bilinear resize pred heatmaps to GT resolution (H, W)."""
    if pred_chw.shape[-2:] == target_hw:
        out = pred_chw
    else:
        out = F.interpolate(
            pred_chw.unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return out.detach().cpu().float().numpy()


def active_channels_mask(gt_counts: np.ndarray, min_gt_peaks: int = 1) -> np.ndarray:
    return gt_counts >= min_gt_peaks


def sweep_thresholds(
    pred_list,
    gt_list,
    thresholds,
) -> list[dict]:
    """For each threshold, aggregate pred vs GT peak counts over samples/channels."""
    rows = []
    for thr in thresholds:
        pred_total = 0
        gt_total = 0
        pred_active = 0
        gt_active = 0
        n_active_ch = 0

        for pred_hm, gt_hm in zip(pred_list, gt_list):
            pred_c = count_peaks_all_channels(pred_hm, thr)
            gt_c = count_peaks_all_channels(gt_hm, thr)
            active = active_channels_mask(gt_c, min_gt_peaks=1)

            pred_total += int(pred_c.sum())
            gt_total += int(gt_c.sum())
            pred_active += int(pred_c[active].sum())
            gt_active += int(gt_c[active].sum())
            n_active_ch += int(active.sum())

        ratio_all = pred_total / gt_total if gt_total > 0 else float("nan")
        ratio_active = (
            pred_active / gt_active if gt_active > 0 else float("nan")
        )

        rows.append(
            {
                "threshold": float(thr),
                "pred_peaks": pred_total,
                "gt_peaks": gt_total,
                "ratio_all": ratio_all,
                "pred_active": pred_active,
                "gt_active": gt_active,
                "ratio_active": ratio_active,
                "n_active_channel_instances": n_active_ch,
            }
        )
    return rows


def suggest_threshold(
    rows: list[dict],
    target_ratio: float = 1.05,
    prefer_over_detection: bool = True,
) -> float:
    """
  Pick threshold for post-processing. When prefer_over_detection is True (default),
  choose the highest threshold with pred/GT >= 1 on active channels — slight
  over-counting is preferred to missing junctions.
    """
    valid = [r for r in rows if r["gt_active"] > 0 and np.isfinite(r["ratio_active"])]
    if not valid:
        return 0.30

    if prefer_over_detection:
        at_least_one = [r for r in valid if r["ratio_active"] >= 1.0]
        if at_least_one:
            # Highest thr still meeting >= GT: fewer spurious peaks than very low thr
            return max(at_least_one, key=lambda r: r["threshold"])["threshold"]
        return min(valid, key=lambda r: -r["ratio_active"])["threshold"]

    return min(valid, key=lambda r: abs(r["ratio_active"] - target_ratio))["threshold"]


def print_channel_table(sample_idx, gt_c, pred_c, threshold):
    print(f"\n  Peaks at threshold={threshold:.2f}  (extract_local_max, same as post-proc)")
    print(f"  {'ch':>3}  {'group':<14}  {'GT':>5}  {'pred':>5}  {'pred/GT':>8}")
    for ch in range(N_HEATMAPS):
        gname = "?"
        for name, (lo, hi, _) in CHANNEL_GROUPS.items():
            if lo <= ch < hi:
                gname = name.split()[0]
                break
        gt_n = int(gt_c[ch])
        pr_n = int(pred_c[ch])
        ratio = f"{pr_n / gt_n:.2f}" if gt_n > 0 else ("—" if pr_n == 0 else "∞")
        print(f"  {ch:3d}  {gname:<14}  {gt_n:5d}  {pr_n:5d}  {ratio:>8}")


def save_overlay_peaks(path, gt_hm, pred_hm, ch, threshold):
    """GT vs pred for one channel with detected peak markers."""
    _, _, close_sup = _channel_close_sup(ch)
    gt_pts = extract_local_max(
        gt_hm[ch], MAX_POINTS_PER_CHANNEL, [0, 0], threshold, close_sup
    )
    pred_pts = extract_local_max(
        pred_hm[ch], MAX_POINTS_PER_CHANNEL, [0, 0], threshold, close_sup
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, hm, pts, title in (
        (axes[0], gt_hm[ch], gt_pts, "GT"),
        (axes[1], pred_hm[ch], pred_pts, "Pred"),
    ):
        ax.imshow(hm, cmap="inferno", vmin=0, vmax=max(1.0, hm.max()))
        for p in pts:
            ax.plot(p[0], p[1], "c+", markersize=12, markeredgewidth=2)
        ax.set_title(f"{title}  ch={ch}  n_peaks={len(pts)}")
        ax.axis("off")
    fig.suptitle(f"Peaks @ threshold={threshold:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="",
        help="comma-separated sweep values (default 0.15,0.20,...,0.60)",
    )
    parser.add_argument(
        "--ref-threshold",
        type=float,
        default=0.30,
        help="threshold for per-channel tables and overlays (matches eval_full default)",
    )
    args_cli = parser.parse_args()

    if args_cli.thresholds:
        thresholds = np.array([float(x) for x in args_cli.thresholds.split(",")])
    else:
        thresholds = DEFAULT_THRESHOLDS

    args = load_eval_args(args_cli.run_dir)
    evaluator = FullSegEvaluator(args)
    testloader = evaluator.dataloader_setup()
    model = evaluator.model_setup()
    model.eval()
    device = evaluator.device

    out_dir = os.path.join(args_cli.run_dir, "heatmap_diagnostic")
    if args_cli.save_png:
        os.makedirs(out_dir, exist_ok=True)

    pred_list = []
    gt_list = []

    with torch.no_grad():
        for idx, sample in enumerate(testloader):
            if idx >= args_cli.n_samples:
                break
            images = sample["image"].to(
                device, non_blocking=(device.type == "cuda")
            )
            labels = sample["label"]  # (1, 23, H, W)
            gt_hm = labels[0, :N_HEATMAPS].float().numpy()
            target_hw = gt_hm.shape[-2:]

            outputs = model(images)
            pred_hm = upsample_pred_heatmaps(outputs[0, :N_HEATMAPS], target_hw)

            pred_list.append(pred_hm)
            gt_list.append(gt_hm)

            folder = sample.get("folder", ["?"])
            folder = folder[0] if isinstance(folder, (list, tuple)) else folder
            print(
                f"\n{'=' * 72}\nSample {idx}  {target_hw[0]}x{target_hw[1]}  {folder}\n{'=' * 72}"
            )

            # Value stats (peaks are sparse — percentiles over all pixels mislead)
            print("\n  Value range (per channel max; GT should be ~1.0 at junctions):")
            print(f"    GT   max per ch: min={gt_hm.max(axis=(1,2)).min():.3f}  "
                  f"median={np.median(gt_hm.max(axis=(1,2))):.3f}  "
                  f"max={gt_hm.max(axis=(1,2)).max():.3f}")
            print(f"    Pred max per ch: min={pred_hm.max(axis=(1,2)).min():.3f}  "
                  f"median={np.median(pred_hm.max(axis=(1,2))):.3f}  "
                  f"max={pred_hm.max(axis=(1,2)).max():.3f}")

            gt_c = count_peaks_all_channels(gt_hm, args_cli.ref_threshold)
            pred_c = count_peaks_all_channels(pred_hm, args_cli.ref_threshold)
            print_channel_table(idx, gt_c, pred_c, args_cli.ref_threshold)

            if args_cli.save_png:
                # Save channels with most GT peaks for visual check
                top_ch = np.argsort(-gt_c)[:4]
                for ch in top_ch:
                    if gt_c[ch] == 0 and pred_c[ch] == 0:
                        continue
                    save_overlay_peaks(
                        os.path.join(
                            out_dir,
                            f"sample_{idx:03d}_ch{ch:02d}_thr{args_cli.ref_threshold:.2f}.png",
                        ),
                        gt_hm,
                        pred_hm,
                        ch,
                        args_cli.ref_threshold,
                    )

    print(f"\n\n{'=' * 72}\nTHRESHOLD SWEEP (aggregate over {len(pred_list)} samples)\n{'=' * 72}")
    print(
        "Only channels with GT>=1 peak at that threshold count toward 'active' columns.\n"
        "ratio_active ≈ 1.0 means pred finds as many peaks as GT on informative channels.\n"
    )
    print(
        f"{'thr':>6}  {'GT peaks':>9}  {'pred':>9}  {'ratio':>7}  "
        f"{'GT act':>8}  {'pred act':>9}  {'ratio act':>10}"
    )

    rows = sweep_thresholds(pred_list, gt_list, thresholds)
    for r in rows:
        print(
            f"{r['threshold']:6.2f}  {r['gt_peaks']:9d}  {r['pred_peaks']:9d}  "
            f"{r['ratio_all']:7.2f}  {r['gt_active']:8d}  {r['pred_active']:9d}  "
            f"{r['ratio_active']:10.2f}"
        )

    suggested = suggest_threshold(rows, target_ratio=1.0)
    ref_row = next((r for r in rows if abs(r["threshold"] - args_cli.ref_threshold) < 1e-6), None)
    if ref_row is None:
        ref_row = min(rows, key=lambda r: abs(r["threshold"] - args_cli.ref_threshold))

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")
    print(f"  Reference threshold {args_cli.ref_threshold:.2f}:  "
          f"ratio_active={ref_row['ratio_active']:.2f}  "
          f"(pred {ref_row['pred_active']} vs GT {ref_row['gt_active']} peaks on active channels)")

    if ref_row["ratio_active"] < 1.0:
        print("  → Under GT peak count at this threshold; lower thr if you want more points.")
    elif ref_row["ratio_active"] > 1.4:
        print("  → Heavy over-detection; raise threshold only if post-proc is too slow/noisy.")
    else:
        print("  → OK for a slight-over-detection policy (pred ≳ GT).")

    print(
        f"\n  Suggested POSTPROC_THRESHOLD ≈ {suggested:.2f}  "
        f"(prefer ≥1.0× GT peaks; eval_full.py uses policy from your sweep)"
    )
    r30 = next((r for r in rows if abs(r["threshold"] - 0.30) < 1e-6), None)
    r25 = next((r for r in rows if abs(r["threshold"] - 0.25) < 1e-6), None)
    if r30 and r25:
        print(
            f"  eval_full.py uses 0.30 (≈{r30['ratio_active']:.2f}× GT in this sweep). "
            f"Use 0.25 for more peaks (≈{r25['ratio_active']:.2f}× GT)."
        )

    print(
        "\n  Note: GT has only a few peaks per channel; global pixel stats (frac>0.4) "
        "are misleading. Trust peak counts and overlays in heatmap_diagnostic/."
    )


if __name__ == "__main__":
    main()
