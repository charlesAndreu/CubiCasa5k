import sys
import os
import logging
import json
import torch
import torch.nn.functional as F
import numpy as np
import yaml  # type: ignore[reportMissingModuleSource]
from datetime import datetime
from types import SimpleNamespace
from tqdm import tqdm
from floortrans.metrics import runningScore
from model import cubi_casa5k_model
from tensorboardX import SummaryWriter

from criterion import CrossEntropyLearnedWeightsLoss, build_criterion
from dataloader import build_cubi_casa5k_dataloaders
from optimizer import build_optimizer_and_scheduler
from tensorboard import TrainingTensorBoard


class SegmentationMapTrainer:

    def __init__(self, args, log_dir, writer, logger):
        self.segmentation_map = args.segmentation_map
        self.args = args
        self.log_dir = log_dir
        self.tb = TrainingTensorBoard(writer)
        self.logger = logger
        self.n_output_channels = 12 if self.segmentation_map == "room" else 11
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return build_cubi_casa5k_dataloaders(
            self.args, self.segmentation_map, self.device, self.logger
        )

    def model_setup(self):
        return cubi_casa5k_model(self.args, self.logger)

    def criterion_setup(self):
        return build_criterion(
            self.args,
            self.segmentation_map,
            self.n_output_channels,
            self.device,
            self.logger,
        )

    def optimizer_setup(self):
        return build_optimizer_and_scheduler(
            self.args, self.model, self.criterion
        )

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
        self.tb.add_graph(self.model, self.args.image_size, self.device)

        # ------------------------------------------------------------
        # Training
        # ------------------------------------------------------------

        # set up variables for training
        first_best = True
        best_val_loss = np.inf
        start_epoch = 0
        running_metrics_map_val = runningScore(self.n_output_channels)
        best_val_loss_variance = np.inf
        no_improvement = 0

        # train for n_epochs (self.args.n_epoch)
        for epoch in range(start_epoch, self.args.n_epoch):
            self.model.train()
            epoch_train_losses = []
            # ------------------------------------------------------------
            # epoch training
            # ------------------------------------------------------------
            for i, samples in tqdm(
                enumerate(trainloader),
                total=len(trainloader),
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
                # outputs are logits: (N, n_output_channels, H, W)
                outputs = self.model(images)
                # target is a long tensor (N, H, W) — one class index per pixel (channel 21 or 22)
                target = self.prepare_segmentation_target(labels, outputs.shape[2:])
                loss = self.criterion(outputs, target)
                epoch_train_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = float(np.mean(epoch_train_losses))

            self.logger.info(
                "Epoch [%d/%d] Loss: %.4f" % (epoch + 1, self.args.n_epoch, train_loss)
            )

            self.tb.log_training_scalars(epoch, train_loss, self.optimizer)

            # ------------------------------------------------------------
            # epoch validation
            # ------------------------------------------------------------
            self.model.eval()
            val_losses = []
            for i_val, samples_val in tqdm(
                enumerate(valloader),
                total=len(valloader),
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

                    outputs = self.model(images_val)
                    target = self.prepare_segmentation_target(
                        labels_val, outputs.shape[2:]
                    )
                    loss = self.criterion(outputs, target)
                    val_losses.append(loss.item())

                    # Per-pixel class predictions: (N, C, H, W) -> argmax over C
                    map_pred = outputs.argmax(dim=1)[0].detach().cpu().numpy()
                    map_gt = target[0].detach().cpu().numpy()
                    running_metrics_map_val.update([map_gt], [map_pred])

            val_loss_mean = float(np.mean(val_losses))
            self.logger.info("val_loss: %.4f" % val_loss_mean)
            self.tb.log_validation_loss(epoch, val_loss_mean)

            # ------------------------------------------------------------
            # Learning rate scheduler
            # ------------------------------------------------------------
            # adam-patience: reduce learning rate when validation loss plateaus
            if self.args.optimizer == "adam-patience":
                self.scheduler.step(val_loss_mean)
            # adam-patience-previous-best: reduce learning rate when validation loss plateaus and save the best model
            elif self.args.optimizer == "adam-patience-previous-best":
                if best_val_loss_variance > val_loss_mean:
                    best_val_loss_variance = val_loss_mean
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement >= self.args.patience:
                    self.logger.info(
                        "No no_improvement for "
                        + str(no_improvement)
                        + " loading last best model and reducing learning rate."
                    )
                    checkpoint = torch.load(self.log_dir + "/model_best_val_loss.pkl")
                    self.model.load_state_dict(checkpoint["model_state"])
                    for i, p in enumerate(self.optimizer.param_groups):
                        self.optimizer.param_groups[i]["lr"] = p["lr"] * 0.1
                    no_improvement = 0

            # sgd: reduce learning rate when validation loss plateaus
            # adam-scheduler: reduce learning rate when validation loss plateaus
            elif self.args.optimizer in ["sgd", "adam-scheduler"]:
                self.scheduler.step(epoch + 1)

            score, class_iou = running_metrics_map_val.get_scores()
            self.tb.log_validation_map_metrics(epoch, score, class_iou)
            running_metrics_map_val.reset()

            # ------------------------------------------------------------
            # Save best validation model
            # ------------------------------------------------------------
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                self.logger.info("New best val loss, saving model_best_val_loss.pkl...")
                if isinstance(self.criterion, CrossEntropyLearnedWeightsLoss):
                    w = np.exp(-self.criterion.s.detach().cpu().numpy()).reshape(-1)
                    self.logger.info(
                        "Learned per-class weights: [%s]",
                        ", ".join(f"{float(x):.4f}" for x in w),
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
                    self.segmentation_map,
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
    config_path = "train_simple_config.yaml"
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}
    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a YAML mapping at top level.")
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

    trainer = SegmentationMapTrainer(args, log_dir, writer, logger)
    trainer.train()
