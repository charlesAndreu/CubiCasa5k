import json
import os
import sys
import logging
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml  # type: ignore[reportMissingModuleSource]
from tqdm import tqdm

from dataloader import build_cubi_casa5k_eval_dataloaders
from floortrans.metrics import runningScore
from model import cubi_casa5k_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SegmentationMapEvaluator:

    def __init__(self, args):
        self.segmentation_map = args.segmentation_map
        self.args = args
        self.n_output_channels = 12 if self.segmentation_map == "room" else 11
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("eval")

    def prepare_segmentation_target(self, labels, output_hw):
        """
        Prepare the segmentation target for CrossEntropyLoss from RoomLoader/IconLoader
        labels ``(N, 1, H, W)`` or ``(N, H, W)`` (class indices).
        """
        t = labels.float()
        if t.dim() == 3:
            t = t.unsqueeze(1)
        if t.shape[2:] != output_hw:
            t = F.interpolate(t, size=output_hw, mode="nearest")
        return t.squeeze(1).long()

    def dataloader_setup(self):
        return build_cubi_casa5k_eval_dataloaders(
            self.args, self.segmentation_map, self.device
        )

    def load_state_dict_from_checkpoint(self):
        checkpoint = torch.load(self.args.weights, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def model_setup(self):
        return cubi_casa5k_model(self.args, self.logger)

    def evaluate(self):
        # ------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------

        testloader = self.dataloader_setup()
        self.model = self.model_setup()
        self.load_state_dict_from_checkpoint()
        running_metrics_map_val = runningScore(self.n_output_channels)
        self.model.eval()

        # ------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------

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

                map_pred = outputs.argmax(dim=1)[0].detach().cpu().numpy()
                map_gt = target[0].detach().cpu().numpy()
                running_metrics_map_val.update([map_gt], [map_pred])

        score, class_iou = running_metrics_map_val.get_scores()
        confusion_matrix = running_metrics_map_val.confusion_matrix.copy()
        return score, class_iou, confusion_matrix


def load_eval_args(run_dir):
    defaults = {
        "segmentation_map": "room",
        "optimizer": "adam-patience-previous-best",
        "criterion": "cross-entropy",
        "weights_method": "inverse_sqrt_frequency",
        "focal_gamma": 2.0,
        "dice_weight": 1.0,
        "data_path": "data/cubicasa5k/",
        "n_epoch": 400,
        "batch_size": 26,
        "image_size": 256,
        "l_rate": 1e-3,
        "l_rate_drop": 200,
        "patience": 20,
        "furukawa_weights": None,
        "resume_from": None,
        "log_path": "runs_cubi/",
        "model": None,
        "debug": False,
        "num_workers": 16,
        "prefetch_factor": 4,
        "plot_samples": False,
        "scale": False,
    }

    args_path = os.path.join(run_dir, "args.json")
    with open(args_path, "r") as f:
        run_args = json.load(f)
    merged = {**defaults, **run_args}
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


def save_confusion_matrix_artifacts(run_dir, confusion_matrix):
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
    im = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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
    image_path = os.path.join(run_dir, "confusion_matrix.png")
    fig.savefig(image_path, dpi=200)
    plt.close(fig)

    raw_path = os.path.join(run_dir, "confusion_matrix_raw.json")
    with open(raw_path, "w") as f:
        json.dump({"confusion_matrix": confusion_matrix.tolist()}, f, indent=2)

    return image_path, raw_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python eval_simple.py <run_dir>")
    run_dir = sys.argv[1]
    args = load_eval_args(run_dir)

    evaluator = SegmentationMapEvaluator(args)
    score, class_iou, confusion_matrix = evaluator.evaluate()
    confusion_image_path, confusion_raw_path = save_confusion_matrix_artifacts(
        run_dir, confusion_matrix
    )
    results = {
        "score": _to_jsonable(score),
        "class_iou": _to_jsonable(class_iou),
        "confusion_matrix_image": os.path.basename(confusion_image_path),
        "confusion_matrix_raw": os.path.basename(confusion_raw_path),
    }

    output_path = os.path.join(run_dir, "eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {output_path}")
    print(f"Saved confusion matrix image to {confusion_image_path}")
    print(f"Saved confusion matrix raw values to {confusion_raw_path}")
