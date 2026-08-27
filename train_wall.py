import os
import logging
import json
import torch
import torch.nn.functional as F
from torch import amp
import numpy as np
import yaml  # type: ignore[reportMissingModuleSource]
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import cubi_casa5k_wall_model
from criterion import build_wall_criterion
from dataloader import build_cubicasa5k_wall_dataloaders
from optimizer import build_optimizer_and_scheduler
from training_tensorboard import WallTrainingTensorBoard
from floortrans.post_prosessing import extract_local_max

TRAIN_WALL_CONFIG_DEFAULTS = {
    "model": None,
    "segformer_model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "n_wall_channels": 5,
    "kernel_px": 7,
    "criterion": "mse",
    "point_confidence_threshold": 0.5,
    "point_match_tolerance_px": 8,
    "optimizer": "adam-patience-previous-best",
    "data_path": "data/cubicasa5k/",
    "n_epoch": 350,
    "batch_size": 26,
    "image_size": 256,
    "l_rate": 1e-3,
    "l_rate_drop": 200,
    "patience": 20,
    "furukawa_weights": None,
    "resume_from": None,
    "log_path": "runs_cubi_wall/",
    "plot_samples": True,
    "debug": False,
    "num_workers": 8,
    "prefetch_factor": 2,
    "scale": True,
}


def resize_heatmap_target(labels, output_hw):
    """Bilinear-resize a continuous gaussian-heatmap target to the model's output resolution
    (labels are float-valued in [0,1], not class indices — unlike train_simple's nearest-resize)."""
    if labels.shape[2:] == output_hw:
        return labels
    return F.interpolate(labels, size=output_hw, mode="bilinear", align_corners=False)


def _match_points(pred_points, gt_points, tol_px):
    """Greedy nearest-neighbor matching within tol_px; returns (tp, fp, fn)."""
    if not gt_points:
        return 0, len(pred_points), 0
    if not pred_points:
        return 0, 0, len(gt_points)

    pred_xy = np.array([[p[0], p[1]] for p in pred_points], dtype=np.float64)
    gt_xy = np.array([[p[0], p[1]] for p in gt_points], dtype=np.float64)
    dists = np.sqrt(((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=2))

    used_gt = set()
    tp = 0
    order = np.argsort(dists.min(axis=1))
    for pi in order:
        gi = int(np.argmin(dists[pi]))
        if dists[pi, gi] <= tol_px and gi not in used_gt:
            used_gt.add(gi)
            tp += 1
    fp = len(pred_points) - tp
    fn = len(gt_points) - tp
    return tp, fp, fn


def compute_point_metrics(pred_chw, target_chw, threshold, tol_px):
    """Sum TP/FP/FN across all heatmap channels for one sample, using the same
    extract_local_max peak-picker the rest of the repo uses for point extraction."""
    tp_total = fp_total = fn_total = 0
    n_channels = pred_chw.shape[0]
    for c in range(n_channels):
        pred_points = extract_local_max(pred_chw[c], num_points=100, info=[], heatmap_value_threshold=threshold)
        gt_points = extract_local_max(target_chw[c], num_points=100, info=[], heatmap_value_threshold=threshold)
        tp, fp, fn = _match_points(pred_points, gt_points, tol_px)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    return tp_total, fp_total, fn_total


class WallTrainer:

    def __init__(self, args, log_dir, writer, logger):
        self.args = args
        self.n_channels = args.n_wall_channels
        self.log_dir = log_dir
        self.tb = WallTrainingTensorBoard(writer)
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            self.grad_scaler = (
                None if self.amp_dtype == torch.bfloat16 else amp.GradScaler("cuda")
            )
        else:
            self.amp_dtype = None
            self.grad_scaler = None

    def dataloader_setup(self):
        return build_cubicasa5k_wall_dataloaders(self.args, self.device, self.logger)

    def model_setup(self):
        return cubi_casa5k_wall_model(self.args, self.logger)

    def criterion_setup(self):
        return build_wall_criterion(self.args, self.device, self.logger)

    def optimizer_setup(self):
        return build_optimizer_and_scheduler(self.args, self.model, self.criterion)

    def _amp_autocast(self):
        if self.use_amp:
            return amp.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    def save_checkpoint(self, filename, epoch, best_loss=None):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "criterion_state": self.criterion.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if best_loss is not None:
            state["best_loss"] = best_loss
        torch.save(state, os.path.join(self.log_dir, filename))

    def train(self):
        with open(self.log_dir + "/args.json", "w") as out:
            json.dump(vars(self.args), out, indent=4)
        self.logger.info("Using device: %s", self.device)

        trainloader, valloader = self.dataloader_setup()
        self.model = self.model_setup()
        self.criterion = self.criterion_setup()
        self.optimizer, self.scheduler = self.optimizer_setup()

        self.tb.log_args(self.args)
        if self.use_amp:
            self.logger.info("AMP enabled (dtype=%s, GradScaler=%s)", self.amp_dtype, self.grad_scaler is not None)
        else:
            self.logger.info("AMP disabled (CPU or CUDA unavailable; training in FP32)")

        first_best = True
        best_val_loss = np.inf
        no_improvement = 0

        for epoch in range(self.args.n_epoch):
            self.model.train()
            epoch_train_losses = []
            train_len = len(trainloader)
            for samples in tqdm(
                trainloader, total=train_len, ncols=80, leave=False,
                desc=f"Train ep {epoch + 1}/{self.args.n_epoch}",
            ):
                images = samples["image"].to(self.device, non_blocking=(self.device.type == "cuda"))
                labels = samples["label"].to(self.device, non_blocking=(self.device.type == "cuda"))
                self.optimizer.zero_grad(set_to_none=True)
                with self._amp_autocast():
                    outputs = self.model(images)
                    preds = torch.sigmoid(outputs)
                    target = resize_heatmap_target(labels, preds.shape[2:])
                    loss = self.criterion(preds, target)
                epoch_train_losses.append(loss.item())

                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            train_loss = float(np.mean(epoch_train_losses))
            self.logger.info("Epoch [%d/%d] Loss: %.6f", epoch + 1, self.args.n_epoch, train_loss)
            self.tb.log_training_scalars(epoch, train_loss, self.optimizer)

            # ------------------------------------------------------------
            # epoch validation
            # ------------------------------------------------------------
            self.model.eval()
            val_losses = []
            tp_total = fp_total = fn_total = 0
            val_len = len(valloader)
            for samples_val in tqdm(
                valloader, total=val_len, ncols=80, leave=False,
                desc=f"Val   ep {epoch + 1}/{self.args.n_epoch}",
            ):
                with torch.no_grad():
                    images_val = samples_val["image"].to(self.device, non_blocking=(self.device.type == "cuda"))
                    labels_val = samples_val["label"].to(self.device, non_blocking=(self.device.type == "cuda"))
                    with self._amp_autocast():
                        outputs = self.model(images_val)
                        preds = torch.sigmoid(outputs)
                        target = resize_heatmap_target(labels_val, preds.shape[2:])
                        loss = self.criterion(preds, target)
                    val_losses.append(loss.item())

                    pred_np = preds[0].detach().float().cpu().numpy()
                    target_np = target[0].detach().float().cpu().numpy()
                    tp, fp, fn = compute_point_metrics(
                        pred_np, target_np,
                        self.args.point_confidence_threshold,
                        self.args.point_match_tolerance_px,
                    )
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn

            val_loss_mean = float(np.mean(val_losses))
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            self.logger.info(
                "val_loss: %.6f  point precision=%.3f recall=%.3f f1=%.3f",
                val_loss_mean, precision, recall, f1,
            )
            self.tb.log_validation_loss(epoch, val_loss_mean)
            self.tb.log_point_metrics(epoch, precision, recall, f1)

            # ------------------------------------------------------------
            # Learning rate scheduler
            # ------------------------------------------------------------
            if self.args.optimizer == "adam-patience":
                self.scheduler.step(val_loss_mean)
            elif self.args.optimizer == "adam-patience-previous-best":
                if val_loss_mean < best_val_loss:
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement >= self.args.patience:
                    self.logger.info(
                        "No improvement for %d epochs; loading last best model and reducing learning rate.",
                        no_improvement,
                    )
                    checkpoint = torch.load(
                        os.path.join(self.log_dir, "model_best_val_loss.pkl"),
                        map_location=self.device, weights_only=False,
                    )
                    self.model.load_state_dict(checkpoint["model_state"])
                    for i, p in enumerate(self.optimizer.param_groups):
                        self.optimizer.param_groups[i]["lr"] = p["lr"] * 0.1
                    no_improvement = 0
            elif self.args.optimizer in ["sgd", "adam-scheduler"]:
                self.scheduler.step(epoch + 1)

            # ------------------------------------------------------------
            # Save best validation model
            # ------------------------------------------------------------
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                self.logger.info("New best val loss, saving model_best_val_loss.pkl...")
                self.save_checkpoint("model_best_val_loss.pkl", epoch + 1, best_loss=best_val_loss)
                self.tb.log_new_best_val_visualizations(
                    epoch, valloader, first_best, self.model, self.args, self.n_channels, self.device,
                )
                first_best = False

        self.logger.info("Last epoch done saving final model...")
        self.save_checkpoint("model_last_epoch.pkl", epoch + 1)
        self.tb.close()


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    config_path = "train_wall_config.yaml"
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}
    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a YAML mapping at top level.")
    defaults = TRAIN_WALL_CONFIG_DEFAULTS
    unknown_keys = sorted(set(config_data.keys()) - set(defaults.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown_keys}")

    args = SimpleNamespace(**{**defaults, **config_data})

    log_dir = args.log_path + "/" + time_stamp + "/"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger = logging.getLogger("train_wall")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir + "/train.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    trainer = WallTrainer(args, log_dir, writer, logger)
    trainer.train()
