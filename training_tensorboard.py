"""TensorBoard logging for CubiCasa5k simple training (tensorboardX SummaryWriter)."""

import math

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _tb_finite_float(x):
    """Plain float for TensorBoard; None if NaN/Inf (metrics often NaN per-class on small val sets)."""
    v = float(np.asarray(x).reshape(-1)[0])
    if not math.isfinite(v):
        return None
    return v


class TrainingTensorBoard:
    """Wraps a ``SummaryWriter`` (or ``None`` when logging is disabled)."""

    def __init__(self, writer):
        self.writer = writer

    def log_args(self, args):
        if self.writer is None:
            return
        self.writer.add_text("parameters", str(vars(args)))

    def _log_scalar(self, tag, value, step):
        if self.writer is None:
            return
        v = _tb_finite_float(value)
        if v is not None:
            self.writer.add_scalar(tag, v, global_step=step)

    def log_training_scalars(self, epoch, train_loss, optimizer):
        if self.writer is None:
            return
        step = 1 + epoch
        lr = optimizer.param_groups[0]["lr"]
        self._log_scalar("training/loss", train_loss, step)
        self._log_scalar("training/lr", lr, step)

    def log_validation_loss(self, epoch, val_loss_mean):
        self._log_scalar("validation/loss", val_loss_mean, 1 + epoch)

    def log_validation_map_metrics(self, epoch, score, class_iou):
        if self.writer is None:
            return
        step = 1 + epoch
        for name, val in score.items():
            tag = "validation/map/general/" + name.replace(" ", "_")
            self._log_scalar(tag, val, step)
        for cls, val in class_iou["Class IoU"].items():
            self._log_scalar("validation/map/class_iou/" + str(cls), val, step)
        for cls, val in class_iou["Class Acc"].items():
            self._log_scalar("validation/map/class_acc/" + str(cls), val, step)
        self.writer.flush()

    def log_new_best_val_visualizations(
        self,
        epoch,
        valloader,
        first_best,
        model,
        args,
        segmentation_map,
        n_output_channels,
        device,
    ):
        if self.writer is None or not args.plot_samples:
            return

        model.eval()
        for i, samples_val in enumerate(valloader):
            with torch.no_grad():
                if i == 4:
                    break
                images_val = samples_val["image"].to(
                    device, non_blocking=(device.type == "cuda")
                )
                labels_val = samples_val["label"].to(
                    device, non_blocking=(device.type == "cuda")
                )
                if first_best:
                    self.writer.add_image("Image " + str(i), images_val[0])
                    gt = labels_val[0, 0].detach().cpu().numpy()
                    fig = plt.figure(figsize=(10, 8))
                    plot = fig.add_subplot(111)
                    cax = plot.imshow(
                        gt,
                        vmin=0,
                        vmax=n_output_channels - 1,
                        cmap=plt.cm.tab20,
                    )
                    fig.colorbar(cax)
                    self.writer.add_figure(
                        "Image " + str(i) + " label/" + segmentation_map,
                        fig,
                    )
                outputs = model(images_val)
                pred_map = outputs[0].argmax(dim=0).detach().cpu().numpy()
                fig = plt.figure(figsize=(18, 12))
                plot = fig.add_subplot(111)
                cax = plot.imshow(
                    pred_map,
                    vmin=0,
                    vmax=n_output_channels - 1,
                    cmap=plt.cm.tab20,
                )
                fig.colorbar(cax)
                self.writer.add_figure(
                    "Image " + str(i) + " prediction/" + segmentation_map,
                    fig,
                    global_step=1 + epoch,
                )

    def close(self):
        if self.writer is not None:
            self.writer.close()
