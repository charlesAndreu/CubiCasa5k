import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pdf")
import sys
import os
import logging
import json
import argparse
import math
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from class_counts import Weights
from datetime import datetime
from torch.utils import data
from tqdm import tqdm

from floortrans.loaders.room_icon_loaders import (
    RoomLoader,
    IconLoader,
    build_simple_train_augmentations,
    build_simple_val_augmentations,
)
from floortrans.models import hg_furukawa_original
from floortrans.metrics import runningScore

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone().float())
        else:
            self.register_buffer("weight", None)
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SegmentationMapTrainer:

    def __init__(self, args, log_dir, writer, logger):
        self.segmentation_map = args.segmentation_map
        self.args = args
        self.log_dir = log_dir
        self.writer = writer  # None when TensorBoard is disabled
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

    def data_loader(self):
        self.logger.info("Loading data...")
        root = self.args.data_path.rstrip(os.sep)
        lmdb_path = os.path.join(root, "cubi_lmdb")
        lmdb_env = lmdb.open(
            lmdb_path,
            readonly=True,
            max_readers=16,
            lock=False,
            readahead=True,
            meminit=False,
        )

        self.logger.info(
            "LMDB loader is %sLoader",
            "Room" if self.segmentation_map == "room" else "Icon",
        )
        train_aug = build_simple_train_augmentations(self.args)
        val_aug = build_simple_val_augmentations(self.args)
        LoaderCls = RoomLoader if self.segmentation_map == "room" else IconLoader
        train_set = LoaderCls(self.args.data_path, "train.txt", lmdb_env, train_aug)
        val_set = LoaderCls(self.args.data_path, "val.txt", lmdb_env, val_aug)

        if self.args.debug:
            num_workers = 0
            print("In debug mode.")
            self.logger.info("In debug mode.")
        else:
            num_workers = max(0, self.args.num_workers)

        self.logger.info(
            "DataLoader num_workers=%s prefetch_factor=%s",
            num_workers,
            max(2, int(self.args.prefetch_factor)) if num_workers > 0 else "n/a",
        )

        _dl_common = dict(
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            persistent_workers=num_workers > 0,
        )
        if num_workers > 0:
            _dl_common["prefetch_factor"] = max(2, int(self.args.prefetch_factor))
        trainloader = data.DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            **_dl_common,
        )
        valloader = data.DataLoader(
            val_set,
            batch_size=1,
            **_dl_common,
        )
        return trainloader, valloader

    def model_setup(self):
        self.logger.info("Loading model...")
        self.logger.info(
            f"Using {self.n_output_channels} channels for {self.segmentation_map} segmentation map"
        )

        # load the original model with its 51 channels to be able to load the weights from the pre-trained model
        # however, we set n_heatmap_channels to 0 to avoid any sigmoid operation applied in conv4_ to these channels
        model = hg_furukawa_original(n_heatmap_channels=0, n_output_channels=51)
        resume = bool(self.args.resume_from)
        if not resume:
            model.init_weights()
            if self.args.furukawa_weights:
                self.logger.info(
                    "Loading furukawa model weights from checkpoint '{}'".format(
                        self.args.furukawa_weights
                    )
                )
                checkpoint = torch.load(self.args.furukawa_weights)
                model.load_state_dict(checkpoint["model_state"])
        else:
            self.logger.info(
                "Skipping init_weights / --furukawa-weights; will load full state from --resume-from"
            )
        # replace the last conv layer with a 1x1 conv layer to output the desired number of channels
        model.conv4_ = torch.nn.Conv2d(
            256, self.n_output_channels, bias=True, kernel_size=1
        )
        model.upsample = torch.nn.ConvTranspose2d(
            self.n_output_channels, self.n_output_channels, kernel_size=4, stride=4
        )
        if not resume:
            for m in [model.conv4_, model.upsample]:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        model.n_output_channels = self.n_output_channels

        self.model = model.to(self.device)

        if resume:
            self.logger.info(
                "Resuming model weights from checkpoint '%s'", self.args.resume_from
            )
            checkpoint = torch.load(self.args.resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])

    def draw_tensorboard_graph(self):
        if self.writer is None:
            return
        dummy = torch.zeros(
            (2, 3, self.args.image_size, self.args.image_size),
            device=self.device,
        )
        self.writer.add_graph(self.model, dummy)

    def tensorboard_log_args(self):
        if self.writer is None:
            return
        self.writer.add_text("parameters", str(vars(self.args)))

    @staticmethod
    def _tb_finite_float(x):
        """Plain float for TensorBoard; None if NaN/Inf (metrics often NaN per-class on small val sets)."""
        v = float(np.asarray(x).reshape(-1)[0])
        if not math.isfinite(v):
            return None
        return v

    def _log_tb_scalar(self, tag, value, step):
        if self.writer is None:
            return
        v = self._tb_finite_float(value)
        if v is not None:
            self.writer.add_scalar(tag, v, global_step=step)

    def tensorboard_log_training_scalars(self, epoch, train_loss):
        if self.writer is None:
            return
        step = 1 + epoch
        lr = self.optimizer.param_groups[0]["lr"]
        self._log_tb_scalar("training/loss", train_loss, step)
        self._log_tb_scalar("training/lr", lr, step)

    def tensorboard_log_validation_loss(self, epoch, val_loss_mean):
        self._log_tb_scalar("validation/loss", val_loss_mean, 1 + epoch)

    def tensorboard_log_validation_map_metrics(self, epoch, score, class_iou):
        if self.writer is None:
            return
        step = 1 + epoch
        for name, val in score.items():
            tag = "validation/map/general/" + name.replace(" ", "_")
            self._log_tb_scalar(tag, val, step)
        for cls, val in class_iou["Class IoU"].items():
            self._log_tb_scalar("validation/map/class_iou/" + str(cls), val, step)
        for cls, val in class_iou["Class Acc"].items():
            self._log_tb_scalar("validation/map/class_acc/" + str(cls), val, step)
        self.writer.flush()

    def tensorboard_log_new_best_val_visualizations(self, epoch, valloader, first_best):
        if self.writer is None or not self.args.plot_samples:
            return

        self.model.eval()
        for i, samples_val in enumerate(valloader):
            with torch.no_grad():
                if i == 4:
                    break
                images_val = samples_val["image"].to(
                    self.device, non_blocking=(self.device == "cuda")
                )
                labels_val = samples_val["label"].to(
                    self.device, non_blocking=(self.device == "cuda")
                )
                if first_best:
                    self.writer.add_image("Image " + str(i), images_val[0])
                    gt = labels_val[0, 0].detach().cpu().numpy()
                    fig = plt.figure(figsize=(10, 8))
                    plot = fig.add_subplot(111)
                    cax = plot.imshow(
                        gt,
                        vmin=0,
                        vmax=self.n_output_channels - 1,
                        cmap=plt.cm.tab20,
                    )
                    fig.colorbar(cax)
                    self.writer.add_figure(
                        "Image " + str(i) + " label/" + self.segmentation_map,
                        fig,
                    )
                outputs = self.model(images_val)
                pred_map = outputs[0].argmax(dim=0).detach().cpu().numpy()
                fig = plt.figure(figsize=(18, 12))
                plot = fig.add_subplot(111)
                cax = plot.imshow(
                    pred_map,
                    vmin=0,
                    vmax=self.n_output_channels - 1,
                    cmap=plt.cm.tab20,
                )
                fig.colorbar(cax)
                self.writer.add_figure(
                    "Image " + str(i) + " prediction/" + self.segmentation_map,
                    fig,
                    global_step=1 + epoch,
                )

    def setup_optimizer(self):
        if self.args.optimizer == "adam-patience":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.l_rate,
                eps=1e-8,
                betas=(0.9, 0.999),
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, "min", patience=self.args.patience, factor=0.5
            )
        elif self.args.optimizer == "adam-patience-previous-best":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.l_rate,
                eps=1e-8,
                betas=(0.9, 0.999),
            )
            self.scheduler = None
        elif self.args.optimizer == "sgd":

            def lr_drop(epoch):
                return (1 - epoch / self.args.n_epoch) ** 0.9

            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.l_rate,
                momentum=0.9,
                weight_decay=10**-4,
                nesterov=True,
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_drop
            )
        elif self.args.optimizer == "adam-scheduler":

            def lr_drop(epoch):
                return 0.5 ** np.floor(epoch / self.args.l_rate_drop)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.l_rate,
                eps=1e-8,
                betas=(0.9, 0.999),
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_drop
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optimizer}")

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

    def setup_loss_weights(self):
        if not self.args.weights_method:
            return None
        with open("class_counts.json", "r") as f:
            class_counts = json.load(f)
        counts = torch.tensor(
            class_counts[self.segmentation_map], dtype=torch.float32
        )
        weights = Weights(counts).weights(method=self.args.weights_method)
        logger.info(f"Setting up loss weights: {weights}")
        return weights

    def setup_criterion(self):
        criterion = self.args.criterion
        # cross-entropy loss
        if criterion == "cross-entropy":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        # weighted cross-entropy loss
        elif criterion == "weighted-cross-entropy":
            weights = self.setup_loss_weights()
            self.criterion = nn.CrossEntropyLoss(weight=weights).to(self.device)
        # focal loss
        elif criterion == "focal-loss":
            weights = self.setup_loss_weights()
            self.criterion = FocalLoss(
                gamma=self.args.focal_gamma, weight=weights, reduction="mean"
            ).to(self.device)
        # dice loss
        elif criterion == "dice-loss":
            pass
        else:
            raise ValueError(f"Invalid criterion: {criterion}")

    def train(self):

        with open(self.log_dir + "/args.json", "w") as out:
            json.dump(vars(self.args), out, indent=4)

        self.tensorboard_log_args()
        self.logger.info("Using device: %s", self.device)
        trainloader, valloader = self.data_loader()
        self.model_setup()
        self.draw_tensorboard_graph()
        self.setup_optimizer()
        self.setup_criterion()

        first_best = True
        best_val_loss = np.inf
        start_epoch = 0
        running_metrics_map_val = runningScore(self.n_output_channels)
        best_val_loss_variance = np.inf
        no_improvement = 0

        # train for n_epochs
        for epoch in range(start_epoch, self.args.n_epoch):
            self.model.train()
            epoch_train_losses = []
            # ------------------------------------------------------------
            # Training
            # ------------------------------------------------------------
            for i, samples in tqdm(
                enumerate(trainloader),
                total=len(trainloader),
                ncols=80,
                leave=False,
                desc=f"Train ep {epoch + 1}/{self.args.n_epoch}",
            ):
                images = samples["image"].to(
                    self.device, non_blocking=(self.device == "cuda")
                )
                labels = samples["label"].to(
                    self.device, non_blocking=(self.device == "cuda")
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

            self.tensorboard_log_training_scalars(epoch, train_loss)

            # ------------------------------------------------------------
            # Validation
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
                        self.device, non_blocking=(self.device == "cuda")
                    )
                    labels_val = samples_val["label"].to(
                        self.device, non_blocking=(self.device == "cuda")
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
            self.tensorboard_log_validation_loss(epoch, val_loss_mean)

            # Learning rate scheduler
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
            self.tensorboard_log_validation_map_metrics(epoch, score, class_iou)
            running_metrics_map_val.reset()

            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                self.logger.info("New best val loss, saving model_best_val_loss.pkl...")
                self.save_checkpoint(
                    "model_best_val_loss.pkl",
                    epoch + 1,
                    best_loss=best_val_loss,
                )
                self.tensorboard_log_new_best_val_visualizations(
                    epoch, valloader, first_best
                )

                first_best = False

        self.logger.info("Last epoch done saving final model...")
        self.save_checkpoint("model_last_epoch.pkl", epoch + 1)
        if self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument(
        "--segmentation-map",
        type=str,
        required=True,
        choices=["room", "icon"],
        help="The segmentation map to train on. Must be 'room' or 'icon'.",
    )
    parser.add_argument(
        "--optimizer",
        nargs="?",
        type=str,
        default="adam-patience-previous-best",
        help="Optimizer to use ['adam, sgd']",
    )
    parser.add_argument(
        "--criterion",
        nargs="?",
        type=str,
        default="cross-entropy",
        help="Criterion to use ['cross-entropy, weighted-cross-entropy, focal-loss, dice-loss']",
    )
    parser.add_argument(
        "--weights-method",
        nargs="?",
        type=str,
        default="inverse_sqrt_frequency",
        choices=["effective_num", "inverse_sqrt_frequency", "inverse_frequency"],
        help=(
            "Class weighting method for --criterion=weighted-cross-entropy "
            "(requires class_counts.json)."
        ),
    )
    parser.add_argument(
        "--focal-gamma",
        nargs="?",
        type=float,
        default=2.0,
        help="Gamma focusing parameter used when --criterion=focal-loss.",
    )
    parser.add_argument(
        "--data-path",
        nargs="?",
        type=str,
        default="data/cubicasa5k/",
        help="Path to data directory",
    )
    parser.add_argument(
        "--n-epoch", nargs="?", type=int, default=400, help="# of the epochs"
    )
    parser.add_argument(
        "--batch-size", nargs="?", type=int, default=26, help="Batch Size"
    )
    parser.add_argument(
        "--image-size", nargs="?", type=int, default=256, help="Image size in training"
    )
    parser.add_argument(
        "--l-rate", nargs="?", type=float, default=1e-3, help="Learning Rate"
    )
    parser.add_argument(
        "--l-rate-drop",
        nargs="?",
        type=float,
        default=200,
        help="Learning rate drop after how many epochs?",
    )
    parser.add_argument(
        "--patience",
        nargs="?",
        type=int,
        default=20,
        help="Learning rate drop patience",
    )
    parser.add_argument(
        "--furukawa-weights",
        nargs="?",
        type=str,
        default=None,
        help="Path to previously trained furukawa model weights file .pkl",
    )
    parser.add_argument(
        "--resume-from",
        nargs="?",
        type=str,
        default=None,
        help=(
            "Path to a train_simple checkpoint .pkl (e.g. model_best_val_loss.pkl) "
            "with the segmentation head; loaded after the model is built. "
            "Do not use for raw 51-ch Furukawa weights (use --furukawa-weights)."
        ),
    )
    parser.add_argument(
        "--log-path",
        nargs="?",
        type=str,
        default="runs_cubi/",
        help="Path to log directory",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        type=bool,
        default=False,
        const=True,
        help="Use DataLoader with num_workers=0 for easier debugging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help=(
            "DataLoader worker processes when not --debug. Each worker is limited to "
            "one compute thread to avoid pegging all CPU cores. Raise (e.g. 4–8) if the GPU starves."
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Per-worker batch prefetch when num_workers>0 (PyTorch requires >= 2).",
    )
    parser.add_argument(
        "--plot-samples",
        nargs="?",
        type=bool,
        default=False,
        const=True,
        help="Plot validation images and segmentation to TensorBoard.",
    )
    parser.add_argument(
        "--scale",
        nargs="?",
        type=bool,
        default=False,
        const=True,
        help="Rescale to 256x256 augmentation.",
    )
    args = parser.parse_args()

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
