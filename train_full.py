import sys
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
from floortrans.metrics import runningScore
from model import cubi_casa5k_full_model
from tensorboardX import SummaryWriter

from criterion import build_full_criterion
from dataloader import build_cubicasa5k_full_dataloaders
from optimizer import build_optimizer_and_scheduler
from training_tensorboard import FullTrainingTensorBoard

TRAIN_FULL_CONFIG_DEFAULTS = {
    "optimizer": "adam-patience-previous-best",
    "room_weights_method": "inverse_sqrt_frequency",
    "icon_weights_method": "inverse_sqrt_frequency",
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
    "debug": False,
    "num_workers": 8,
    "prefetch_factor": 2,
    "plot_samples": True,
    "scale": True,
}


def _seg_argmax_at_label_size(logits_chw, label_hw):
    """
    Argmax segmentation logits on the label grid (native resolution).
    The hourglass head can be 1–2 px smaller/larger than the input; upsample
    logits with bilinear (same idea as split_prediction), not the GT image.
    """
    if logits_chw.shape[-2:] != label_hw:
        logits_chw = F.interpolate(
            logits_chw.unsqueeze(0),
            size=label_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return logits_chw.argmax(dim=0).detach().cpu().numpy()


class Cubicasa5kFullTrainer:

    def __init__(self, args, log_dir, writer, logger):
        self.input_slice = [
            21,
            3,
            4,
        ]  # 21 heatmap channels, 3 room classes, 4 icon classes
        self.n_output_channels = sum(self.input_slice)
        self.args = args
        self.log_dir = log_dir
        self.tb = FullTrainingTensorBoard(writer, self.input_slice)
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
        return build_cubicasa5k_full_dataloaders(self.args, self.device, self.logger)

    def model_setup(self):
        return cubi_casa5k_full_model(self.args, self.logger)

    def criterion_setup(self):
        return build_full_criterion(
            self.args,
            self.input_slice,
            self.device,
            self.logger,
        )

    def optimizer_setup(self):
        return build_optimizer_and_scheduler(self.args, self.model, self.criterion)

    def _amp_autocast(self):
        """CUDA autocast when AMP is on; no-op context otherwise (safe on CPU)."""
        if self.use_amp:
            return amp.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    def save_checkpoint(self, filename, epoch, best_loss=None):
        """Save training checkpoint under log_dir. filename is a basename (e.g. 'model_last_epoch.pkl')."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "criterion_state": self.criterion.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if best_loss is not None:
            state["best_loss"] = best_loss
        path = os.path.join(self.log_dir, filename)
        torch.save(state, path)

    def train(self):
        # ------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------

        with open(self.log_dir + "/args.json", "w") as out:
            json.dump(vars(self.args), out, indent=4)
        self.logger.info("Using device: %s", self.device)

        trainloader, valloader = self.dataloader_setup()
        self.model = self.model_setup()
        self.criterion = self.criterion_setup()
        self.optimizer, self.scheduler = self.optimizer_setup()

        self.tb.log_args(self.args)
        if self.use_amp:
            self.logger.info(
                "AMP enabled (dtype=%s, GradScaler=%s)",
                self.amp_dtype,
                self.grad_scaler is not None,
            )
        else:
            self.logger.info(
                "AMP disabled (CPU or CUDA unavailable; training in FP32)"
            )

        # ------------------------------------------------------------
        # Training
        # ------------------------------------------------------------

        # set up variables for training
        first_best = True
        best_val_loss = np.inf  # best validation total_var (training-aligned objective)
        start_epoch = 0
        # runningScore tracks mIoU/pixel-acc for classification heads
        # heatmaps are regression (MSE) — already tracked via criterion loss
        running_metrics_room_val = runningScore(
            self.input_slice[1]
        )  # 3 room-mini classes
        running_metrics_icon_val = runningScore(
            self.input_slice[2]
        )  # 4 icon-mini classes
        no_improvement = 0

        # train for n_epochs (self.args.n_epoch)
        for epoch in range(start_epoch, self.args.n_epoch):
            self.model.train()
            epoch_train_scalars = []
            # ------------------------------------------------------------
            # epoch training
            # ------------------------------------------------------------
            train_len = len(trainloader)
            for i, samples in tqdm(
                enumerate(trainloader),
                total=train_len,
                ncols=80,
                leave=False,
                desc=f"Train ep {epoch + 1}/{self.args.n_epoch}",
            ):
                images = samples["image"].to(
                    self.device, non_blocking=(self.device.type == "cuda")
                )
                labels = samples["label"].to(
                    self.device, non_blocking=(self.device.type == "cuda")
                )
                self.optimizer.zero_grad(set_to_none=True)
                with self._amp_autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                epoch_train_scalars.append(self.criterion.get_loss_scalars())

                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            keys = epoch_train_scalars[0].keys()
            train_losses = {
                k: float(np.mean([d[k] for d in epoch_train_scalars])) for k in keys
            }

            self.logger.info(
                "Epoch [%d/%d] Loss (total_var): %.4f"
                % (epoch + 1, self.args.n_epoch, train_losses["total_var"])
            )

            self.tb.log_training_scalars(epoch, train_losses, self.optimizer)
            self.tb.log_uncertainty_vars(
                epoch, self.criterion.get_uncertainty_scalars()
            )

            # ------------------------------------------------------------
            # epoch validation
            # ------------------------------------------------------------
            self.model.eval()
            val_scalars = []
            val_len = len(valloader)
            for samples_val in tqdm(
                valloader,
                total=val_len,
                ncols=80,
                leave=False,
                desc=f"Val   ep {epoch + 1}/{self.args.n_epoch}",
            ):
                with torch.no_grad():
                    images_val = samples_val["image"].to(
                        self.device, non_blocking=(self.device.type == "cuda")
                    )
                    labels_val = samples_val["label"].to(
                        self.device, non_blocking=(self.device.type == "cuda")
                    )

                    with self._amp_autocast():
                        outputs = self.model(images_val)
                        loss = self.criterion(outputs, labels_val)
                    val_scalars.append(self.criterion.get_loss_scalars())

                    n_hm = self.input_slice[0]
                    room_end = n_hm + self.input_slice[1]
                    icon_end = room_end + self.input_slice[2]
                    label_hw = (labels_val.shape[2], labels_val.shape[3])

                    room_pred = _seg_argmax_at_label_size(
                        outputs[0, n_hm:room_end], label_hw
                    )
                    room_gt = labels_val[0, 21].long().detach().cpu().numpy()
                    running_metrics_room_val.update([room_gt], [room_pred])

                    icon_pred = _seg_argmax_at_label_size(
                        outputs[0, room_end:icon_end], label_hw
                    )
                    icon_gt = labels_val[0, 22].long().detach().cpu().numpy()
                    running_metrics_icon_val.update([icon_gt], [icon_pred])

            keys = val_scalars[0].keys()
            val_losses = {k: float(np.mean([d[k] for d in val_scalars])) for k in keys}
            val_loss_var = val_losses["total_var"]
            self.logger.info("val_loss (total_var): %.4f", val_loss_var)
            self.tb.log_validation_scalars(epoch, val_losses)

            val_improved = val_loss_var < best_val_loss
            if val_improved:
                best_val_loss = val_loss_var

            # ------------------------------------------------------------
            # Learning rate scheduler
            # ------------------------------------------------------------
            if self.args.optimizer == "adam-patience":
                self.scheduler.step(val_loss_var)
            elif self.args.optimizer == "adam-patience-previous-best":
                if val_improved:
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement >= self.args.patience:
                    self.logger.info(
                        "No no_improvement for "
                        + str(no_improvement)
                        + " loading last best model and reducing learning rate."
                    )
                    checkpoint = torch.load(
                        os.path.join(self.log_dir, "model_best_val_loss.pkl")
                    )
                    self.model.load_state_dict(checkpoint["model_state"])
                    for i, p in enumerate(self.optimizer.param_groups):
                        self.optimizer.param_groups[i]["lr"] = p["lr"] * 0.1
                    no_improvement = 0
            elif self.args.optimizer in ["sgd", "adam-scheduler"]:
                self.scheduler.step(epoch + 1)

            score, class_iou = running_metrics_room_val.get_scores()
            self.tb.log_validation_map_metrics(epoch, score, class_iou, head="room")
            running_metrics_room_val.reset()

            score, class_iou = running_metrics_icon_val.get_scores()
            self.tb.log_validation_map_metrics(epoch, score, class_iou, head="icon")
            running_metrics_icon_val.reset()

            # ------------------------------------------------------------
            # Save best validation checkpoint (total_var only)
            # ------------------------------------------------------------
            if val_improved:
                self.logger.info(
                    "New best val loss (total_var), saving model_best_val_loss.pkl..."
                )
                self.save_checkpoint(
                    "model_best_val_loss.pkl",
                    epoch + 1,
                    best_loss=best_val_loss,
                )
                self.tb.log_new_best_val_visualizations(
                    epoch,
                    valloader,
                    first_best,
                    self.model,
                    self.args,
                    "full",
                    self.n_output_channels,
                    self.device,
                )
                first_best = False

        # ------------------------------------------------------------
        # Save final model
        # ------------------------------------------------------------
        self.logger.info("Last epoch done saving final model...")
        self.save_checkpoint("model_last_epoch.pkl", epoch + 1)
        self.tb.close()


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    config_path = "train_full_config.yaml"
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}
    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a YAML mapping at top level.")
    defaults = TRAIN_FULL_CONFIG_DEFAULTS
    unknown_keys = sorted(set(config_data.keys()) - set(defaults.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown_keys}")

    args = SimpleNamespace(**{**defaults, **config_data})

    log_dir = args.log_path + "/" + time_stamp + "/"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir + "/train.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    trainer = Cubicasa5kFullTrainer(args, log_dir, writer, logger)
    trainer.train()
