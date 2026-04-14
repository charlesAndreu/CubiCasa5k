import matplotlib

matplotlib.use("pdf")
import sys
import os
import logging
import json
import argparse
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from floortrans.loaders.augmentations import (
    RandomCropToSizeTorch,
    ResizePaddedTorch,
    Compose,
    DictToTensor,
    ColorJitterTorch,
    RandomRotations,
)
from torchvision.transforms import RandomChoice
from torch.utils import data
from tqdm import tqdm

from floortrans.loaders import FloorplanSVG
from floortrans.models import hg_furukawa_original
from floortrans.metrics import runningScore

# from tensorboardX import SummaryWriter  # TensorBoard (disabled; study later)
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import matplotlib.pyplot as plt  # only used for TensorBoard plot_samples below


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
        Prepare the segmentation target for the CrossEntropyLoss.
        Room/icon map from full label stack -> (N, H, W) long for CrossEntropyLoss.

        labels: (N, 23, H, W). Channel 21 = room indices, 22 = icon indices.
        output_hw: (H, W) must match logits spatial size.
        """
        ch = 21 if self.segmentation_map == "room" else 22
        t = labels[:, ch : ch + 1, :, :].float()
        if t.shape[2:] != output_hw:
            t = F.interpolate(t, size=output_hw, mode="nearest")
        return t.squeeze(1).long()

    def data_loader(self):
        # Augmentation setup
        if self.args.scale:
            aug = Compose(
                [
                    RandomChoice(
                        [
                            RandomCropToSizeTorch(
                                data_format="dict",
                                size=(self.args.image_size, self.args.image_size),
                            ),
                            ResizePaddedTorch(
                                (0, 0),
                                data_format="dict",
                                size=(self.args.image_size, self.args.image_size),
                            ),
                        ]
                    ),
                    RandomRotations(format="cubi"),
                    DictToTensor(),
                    ColorJitterTorch(),
                ]
            )
        else:
            aug = Compose(
                [
                    RandomCropToSizeTorch(
                        data_format="dict",
                        size=(self.args.image_size, self.args.image_size),
                    ),
                    RandomRotations(format="cubi"),
                    DictToTensor(),
                    ColorJitterTorch(),
                ]
            )

        # Setup Dataloader
        # self.writer.add_text("parameters", str(vars(self.args)))  # TensorBoard
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
        train_set = FloorplanSVG(
            self.args.data_path,
            "train.txt",
            format="lmdb",
            augmentations=aug,
            lmdb_env=lmdb_env,
        )
        val_set = FloorplanSVG(
            self.args.data_path,
            "val.txt",
            format="lmdb",
            augmentations=DictToTensor(),
            lmdb_env=lmdb_env,
        )

        if self.args.debug:
            num_workers = 0
            print("In debug mode.")
            self.logger.info("In debug mode.")
        else:
            num_workers = 8

        trainloader = data.DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=(self.device == "cuda"),
        )
        valloader = data.DataLoader(
            val_set,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
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
        model.init_weights()
        if self.args.furukawa_weights:
            self.logger.info(
                "Loading furukawa model weights from checkpoint '{}'".format(
                    self.args.furukawa_weights
                )
            )
            checkpoint = torch.load(self.args.furukawa_weights)
            model.load_state_dict(checkpoint["model_state"])
        # replace the last conv layer with a 1x1 conv layer to output the desired number of channels
        model.conv4_ = torch.nn.Conv2d(
            256, self.n_output_channels, bias=True, kernel_size=1
        )
        model.upsample = torch.nn.ConvTranspose2d(
            self.n_output_channels, self.n_output_channels, kernel_size=4, stride=4
        )
        # initialize the weights of the last conv layer and the upsample layer that have been re-configured
        for m in [model.conv4_, model.upsample]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)

        model.n_output_channels = self.n_output_channels

        self.model = model.to(self.device)

    def draw_tensorboard_graph(self):
        # TensorBoard: computation graph (disabled; study later)
        # dummy = torch.zeros((2, 3, self.args.image_size, self.args.image_size)).cuda()
        # self.model(dummy)
        # self.writer.add_graph(self.model, dummy)
        pass

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

    def train(self):

        with open(self.log_dir + "/args.json", "w") as out:
            json.dump(vars(self.args), out, indent=4)

        self.logger.info("Using device: %s", self.device)
        trainloader, valloader = self.data_loader()
        self.model_setup()
        self.draw_tensorboard_graph()
        self.setup_optimizer()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        first_best = True
        best_val_loss = np.inf
        best_train_loss = np.inf
        best_acc = 0
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

            # TensorBoard: training scalars (disabled; study later)
            # current_lr = self.optimizer.param_groups[0]["lr"]
            # self.writer.add_scalars(
            #     "training",
            #     {"loss": float(train_loss), "lr": float(current_lr)},
            #     global_step=1 + epoch,
            # )

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
            # TensorBoard (disabled; study later)
            # self.writer.add_scalar(
            #     "validation/loss", val_loss_mean, global_step=1 + epoch
            # )

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
            # TensorBoard: validation metrics (disabled; study later)
            # self.writer.add_scalars(
            #     "validation/map/general", score, global_step=1 + epoch
            # )
            # self.writer.add_scalars(
            #     "validation/map/IoU",
            #     class_iou["Class IoU"],
            #     global_step=1 + epoch,
            # )
            # self.writer.add_scalars(
            #     "validation/map/Acc",
            #     class_iou["Class Acc"],
            #     global_step=1 + epoch,
            # )
            running_metrics_map_val.reset()

            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                self.logger.info("New best val loss, saving model_best_val_loss.pkl...")
                self.save_checkpoint(
                    "model_best_val_loss.pkl",
                    epoch + 1,
                    best_loss=best_val_loss,
                )
                # TensorBoard: plot_samples (matplotlib + add_image/add_figure) — disabled; study later
                # if self.args.plot_samples:
                #     import matplotlib.pyplot as plt
                #     for i, samples_val in enumerate(valloader):
                #         with torch.no_grad():
                #             if i == 4:
                #                 break
                #             images_val = samples_val["image"].cuda(non_blocking=True)
                #             labels_val = samples_val["label"].cuda(non_blocking=True)
                #             if first_best:
                #                 self.writer.add_image("Image " + str(i), images_val[0])
                #                 for j, l in enumerate(
                #                     labels_val.squeeze().detach().cpu().numpy()
                #                 ):
                #                     fig = plt.figure(figsize=(18, 12))
                #                     plot = fig.add_subplot(111)
                #                     if j < 21:
                #                         cax = plot.imshow(l, vmin=0, vmax=1)
                #                     else:
                #                         cax = plot.imshow(
                #                             l, vmin=0, vmax=19, cmap=plt.cm.tab20
                #                         )
                #                     fig.colorbar(cax)
                #                     self.writer.add_figure(
                #                         "Image " + str(i) + " label/Channel " + str(j),
                #                         fig,
                #                     )
                #             outputs = self.model(images_val)
                #             pred_map = outputs[0].argmax(dim=0).detach().cpu().numpy()
                #             fig = plt.figure(figsize=(18, 12))
                #             plot = fig.add_subplot(111)
                #             cax = plot.imshow(
                #                 pred_map,
                #                 vmin=0,
                #                 vmax=self.n_output_channels - 1,
                #                 cmap=plt.cm.tab20,
                #             )
                #             fig.colorbar(cax)
                #             self.writer.add_figure(
                #                 "Image " + str(i) + " prediction/" + self.segmentation_map,
                #                 fig,
                #                 global_step=1 + epoch,
                #             )

                first_best = False

            px_acc = score["Mean Acc"]
            if px_acc > best_acc:
                best_acc = px_acc
                self.logger.info("Best validation pixel accuracy found saving model...")
                self.save_checkpoint(
                    "model_best_val_acc.pkl",
                    epoch + 1,
                )

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                self.logger.info("Best training loss with variance...")
                self.save_checkpoint(
                    "model_best_train_loss_var.pkl",
                    epoch + 1,
                )

        self.logger.info("Last epoch done saving final model...")
        self.save_checkpoint("model_last_epoch.pkl", epoch + 1)


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
        "--data-path",
        nargs="?",
        type=str,
        default="data/cubicasa5k/",
        help="Path to data directory",
    )
    parser.add_argument(
        "--n-epoch", nargs="?", type=int, default=1000, help="# of the epochs"
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
        default=10,
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
        "--plot-samples",
        nargs="?",
        type=bool,
        default=False,
        const=True,
        help="Plot floorplan segmentations to Tensorboard.",
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
    # writer = SummaryWriter(log_dir)  # TensorBoard (disabled; study later)
    writer = None
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
