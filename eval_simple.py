import json
import os
import sys
import logging
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import build_cubicasa5k_simple_eval_dataloaders, n_segmentation_classes
from entropy_heatmap import all_classes_entropy_heatmap
from floortrans.metrics import runningScore
from model import cubi_casa5k_simple_model
from train_simple import TRAIN_SIMPLE_CONFIG_DEFAULTS

matplotlib.use("Agg")

WALL_CLASSES = [2]
import matplotlib.pyplot as plt


def _tensor_to_rgb_numpy(image_chw):
    """Float tensor (3, H, W) → RGB array (H, W, 3) in [0, 1] for saving."""
    x = image_chw.detach().cpu().float().numpy().transpose(1, 2, 0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo > 1e-6:
        x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def _tensor_to_bgr_uint8(image_chw):
    """Float tensor (3, H, W) → BGR uint8 for OpenCV."""
    rgb01 = _tensor_to_rgb_numpy(image_chw)
    rgb_u8 = (rgb01 * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def _save_entropy_heatmap_png(path, entropy_hw, title):
    """Normalized entropy map in [0, 1] as a color heatmap PNG."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        entropy_hw,
        vmin=0.0,
        vmax=1.0,
        cmap="inferno",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized entropy")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_segmentation_map_png(path, seg_hw, n_classes):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        seg_hw,
        vmin=0,
        vmax=n_classes - 1,
        cmap=plt.cm.tab20,
        interpolation="nearest",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_eval_entropy_heatmap(out_dir, idx, logits_chw):
    """Save all-class entropy heatmap for a visualized sample."""
    stem = f"sample_{idx:05d}"
    hm_all = all_classes_entropy_heatmap(logits_chw).detach().cpu().numpy()
    _save_entropy_heatmap_png(
        os.path.join(out_dir, f"{stem}_entropy_all_classes.png"),
        hm_all,
        "Entropy (all classes)",
    )


def save_eval_sample_images(
    out_dir,
    idx,
    image_chw,
    seg_pred_hw,
    n_classes,
    include_wall=False,
):
    """
    PNG exports — *_input.png, *_segmentation.png;
    *_wall.png only when include_wall (room segmentation).
    """
    stem = f"sample_{idx:05d}"
    bgr = _tensor_to_bgr_uint8(image_chw)
    h, w = bgr.shape[:2]
    cv2.imwrite(os.path.join(out_dir, f"{stem}_input.png"), bgr)

    _save_segmentation_map_png(
        os.path.join(out_dir, f"{stem}_segmentation.png"),
        seg_pred_hw,
        n_classes,
    )

    if not include_wall:
        return

    model_wall = np.isin(seg_pred_hw, WALL_CLASSES).astype(np.uint8)
    if model_wall.shape[0] != h or model_wall.shape[1] != w:
        model_wall = cv2.resize(
            model_wall,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        model_wall = (model_wall > 0).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_wall.png"), model_wall * 255)


class SegmentationMapEvaluator:

    def __init__(self, args):
        self.segmentation_map = args.segmentation_map
        self.args = args
        self.n_output_channels = n_segmentation_classes(self.segmentation_map)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("eval")

    def prepare_segmentation_target(self, labels, output_hw):
        """
        Prepare the segmentation target for CrossEntropyLoss from RoomLoader/IconLoader
        labels (N, 1, H, W) or (N, H, W) (class indices).
        """
        t = labels.float()
        if t.dim() == 3:
            t = t.unsqueeze(1)
        if t.shape[2:] != output_hw:
            t = F.interpolate(t, size=output_hw, mode="nearest")
        return t.squeeze(1).long()

    def dataloader_setup(self):
        return build_cubicasa5k_simple_eval_dataloaders(
            self.args, self.segmentation_map, self.device
        )

    def load_state_dict_from_checkpoint(self):
        checkpoint = torch.load(self.args.weights, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def model_setup(self):
        return cubi_casa5k_simple_model(self.args, self.logger)

    def evaluate(self, results_dir):
        # ------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------

        testloader = self.dataloader_setup()
        self.model = self.model_setup()
        self.load_state_dict_from_checkpoint()
        running_metrics_raw = runningScore(self.n_output_channels)
        self.model.eval()

        save_wall = self.segmentation_map.startswith("room")
        n_samples = len(testloader.dataset)
        rng = np.random.default_rng(42)
        k_vis = min(3, n_samples)
        vis_indices = set(rng.choice(n_samples, size=k_vis, replace=False).tolist())
        vis_dir = os.path.join(results_dir, "eval_samples_seed42")
        os.makedirs(vis_dir, exist_ok=True)

        # ------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------

        entropy_all_sum = 0.0
        entropy_pixel_count = 0

        global_idx = 0
        with torch.no_grad():
            for _, samples in tqdm(
                enumerate(testloader),
                total=len(testloader),
                ncols=80,
                leave=False,
                desc="Eval",
            ):
                images = samples["image"].to(
                    self.device, non_blocking=(self.device.type == "cuda")
                )
                labels = samples["label"].to(
                    self.device, non_blocking=(self.device.type == "cuda")
                )
                outputs = self.model(images)
                target = self.prepare_segmentation_target(labels, outputs.shape[2:])

                pred = outputs.argmax(dim=1).long()
                map_pred = pred[0].detach().cpu().numpy()
                map_gt = target[0].detach().cpu().numpy()

                running_metrics_raw.update([map_gt], [map_pred])

                logits_chw = outputs[0]
                hm_all = all_classes_entropy_heatmap(logits_chw)
                n_pixels = hm_all.numel()
                entropy_all_sum += float(hm_all.sum().item())
                entropy_pixel_count += n_pixels

                if global_idx in vis_indices:
                    save_eval_sample_images(
                        vis_dir,
                        global_idx,
                        images[0],
                        map_pred,
                        self.n_output_channels,
                        include_wall=save_wall,
                    )
                    save_eval_entropy_heatmap(vis_dir, global_idx, logits_chw)

                global_idx += 1

        score, class_iou = running_metrics_raw.get_scores()
        confusion_matrix = running_metrics_raw.confusion_matrix.copy()
        mean_entropy_all_classes = (
            entropy_all_sum / entropy_pixel_count if entropy_pixel_count else 0.0
        )
        return score, class_iou, confusion_matrix, vis_dir, mean_entropy_all_classes


def load_eval_args(run_dir):
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path, "r") as f:
        run_args = json.load(f)
    merged = {**TRAIN_SIMPLE_CONFIG_DEFAULTS, **run_args}
    merged["weights"] = os.path.join(run_dir, "model_best_val_loss.pkl")
    merged["log_path"] = run_dir + "/eval.log"
    merged["num_workers"] = 4
    return SimpleNamespace(**merged)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_confusion_matrix_artifacts(run_dir, confusion_matrix, stem="confusion_matrix"):
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums != 0,
    )

    n_classes = confusion_matrix.shape[0]
    labels = [str(i) for i in range(n_classes)]

    fig_size = max(8, min(16, n_classes * 0.8))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(
        normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized score")

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion Matrix (row-normalized colors)",
    )

    for i in range(n_classes):
        for j in range(n_classes):
            count = int(confusion_matrix[i, j])
            pct = normalized[i, j] * 100.0
            text_color = "white" if normalized[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{count}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    fig.tight_layout()
    image_path = os.path.join(run_dir, f"{stem}.png")
    fig.savefig(image_path, dpi=200)
    plt.close(fig)

    raw_path = os.path.join(run_dir, f"{stem}_raw.json")
    with open(raw_path, "w") as f:
        json.dump({"confusion_matrix": confusion_matrix.tolist()}, f, indent=2)

    return image_path, raw_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python eval_simple.py <run_dir>")
    run_dir = sys.argv[1]
    args = load_eval_args(run_dir)

    evaluator = SegmentationMapEvaluator(args)
    score, class_iou, confusion_matrix, vis_dir, mean_entropy_all_classes = (
        evaluator.evaluate(results_dir=run_dir)
    )

    confusion_image_path, confusion_raw_path = save_confusion_matrix_artifacts(
        run_dir, confusion_matrix, stem="confusion_matrix"
    )

    results = {
        "score": _to_jsonable(score),
        "class_iou": _to_jsonable(class_iou),
        "mean_entropy_all_classes": mean_entropy_all_classes,
    }

    output_path = os.path.join(run_dir, "eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {output_path}")
    print(f"Saved confusion matrix image to {confusion_image_path}")
    print(f"Saved confusion matrix raw values to {confusion_raw_path}")
    print(f"Mean entropy (all classes): {mean_entropy_all_classes:.6f}")
    if vis_dir:
        extras = " / wall" if args.segmentation_map.startswith("room") else ""
        print(
            f"Saved eval samples (input / segmentation{extras} / entropy heatmap) "
            f"under {vis_dir}"
        )
