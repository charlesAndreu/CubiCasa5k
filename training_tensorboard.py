import math

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _tb_finite_float(x):
    v = float(np.asarray(x).reshape(-1)[0])
    if not math.isfinite(v):
        return None
    return v


def _pred_heatmap_sum_display(arr, noise_percentile=30.0):
    lo = float(np.percentile(arr, noise_percentile))
    return np.maximum(arr - lo, 0.0)


def _figure_heatmap_sum(map_hw):
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(float(map_hw.max()), 1e-6)
    im = ax.imshow(map_hw, vmin=0.0, vmax=vmax, cmap="magma", aspect="equal")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


class SimpleTrainingTensorBoard:

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

    def log_validation_map_metrics(self, epoch, score, class_iou, head=""):
        if self.writer is None:
            return
        step = 1 + epoch
        prefix = f"validation/map/{head}/" if head else "validation/map/"
        for name, val in score.items():
            self._log_scalar(prefix + "general/" + name.replace(" ", "_"), val, step)
        for cls, val in class_iou["Class IoU"].items():
            self._log_scalar(prefix + "class_iou/" + str(cls), val, step)
        for cls, val in class_iou["Class Acc"].items():
            self._log_scalar(prefix + "class_acc/" + str(cls), val, step)
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


class FullTrainingTensorBoard(SimpleTrainingTensorBoard):

    def __init__(self, writer, input_slice=(21, 4, 4)):
        super().__init__(writer)
        self.input_slice = tuple(int(x) for x in input_slice)

    def log_training_scalars(self, epoch, losses: dict, optimizer):
        if self.writer is None:
            return
        step = 1 + epoch
        self._log_scalar("training/loss/total", losses["total"], step)
        self._log_scalar("training/loss/rooms", losses["rooms"], step)
        self._log_scalar("training/loss/icons", losses["icons"], step)
        self._log_scalar("training/loss/heatmap", losses["heatmap"], step)
        self._log_scalar("training/loss/total_var", losses["total_var"], step)
        self._log_scalar("training/loss/rooms_var", losses["rooms_var"], step)
        self._log_scalar("training/loss/icons_var", losses["icons_var"], step)
        self._log_scalar("training/loss/heatmap_var", losses["heatmap_var"], step)
        self._log_scalar("training/lr", optimizer.param_groups[0]["lr"], step)

    def log_validation_scalars(self, epoch, losses: dict):
        if self.writer is None:
            return
        step = 1 + epoch
        self._log_scalar("validation/loss/total", losses["total"], step)
        self._log_scalar("validation/loss/rooms", losses["rooms"], step)
        self._log_scalar("validation/loss/icons", losses["icons"], step)
        self._log_scalar("validation/loss/heatmap", losses["heatmap"], step)
        self._log_scalar("validation/loss/total_var", losses["total_var"], step)

    def log_uncertainty_vars(self, epoch, uncertainty: dict):
        if self.writer is None:
            return
        step = 1 + epoch
        self._log_scalar("uncertainty/room_var", uncertainty["room_var"], step)
        self._log_scalar("uncertainty/icon_var", uncertainty["icon_var"], step)

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
        del n_output_channels  # kept from SimpleTrainingTensorBoard for API parity; no need here.

        if self.writer is None or not args.plot_samples:
            return

        n_hm, n_room, n_icon = self.input_slice
        room_logit_end = n_hm + n_room
        room_label_ch = n_hm
        icon_label_ch = n_hm + 1

        model.eval()
        step = 1 + epoch
        cmap = plt.cm.tab20

        for i, samples_val in enumerate(valloader):
            if i == 4:
                break
            with torch.no_grad():
                images_val = samples_val["image"].to(
                    device, non_blocking=(device.type == "cuda")
                )
                labels_val = samples_val["label"].to(
                    device, non_blocking=(device.type == "cuda")
                )
                if first_best:
                    self.writer.add_image("Image " + str(i), images_val[0])
                    for head, ch, vmax in (
                        ("room", room_label_ch, n_room - 1),
                        ("icon", icon_label_ch, n_icon - 1),
                    ):
                        gt = labels_val[0, ch].detach().cpu().numpy()
                        fig = plt.figure(figsize=(10, 8))
                        plot = fig.add_subplot(111)
                        cax = plot.imshow(gt, vmin=0, vmax=vmax, cmap=cmap)
                        fig.colorbar(cax)
                        self.writer.add_figure(
                            f"Image {i} label/{head}_{segmentation_map}",
                            fig,
                            global_step=step,
                        )
                        plt.close(fig)
                    hm_gt_sum = (
                        labels_val[0, :n_hm].sum(dim=0).detach().float().cpu().numpy()
                    )
                    fig_sum_gt = _figure_heatmap_sum(hm_gt_sum)
                    self.writer.add_figure(
                        f"Image {i} heatmaps_sum_gt_{segmentation_map}",
                        fig_sum_gt,
                        global_step=step,
                    )
                    plt.close(fig_sum_gt)

                # predicted heatmaps sum plot
                outputs = model(images_val)
                hm_pred_sum = (
                    torch.sigmoid(outputs[0, :n_hm])
                    .sum(dim=0)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                hm_pred_vis = _pred_heatmap_sum_display(hm_pred_sum)
                fig_sum_pred = _figure_heatmap_sum(hm_pred_vis)
                self.writer.add_figure(
                    f"Image {i} heatmaps_sum_pred_{segmentation_map}",
                    fig_sum_pred,
                    global_step=step,
                )
                plt.close(fig_sum_pred)

                # room and icon predictions plots
                for head, slc, vmax in (
                    ("room", slice(n_hm, room_logit_end), n_room - 1),
                    (
                        "icon",
                        slice(room_logit_end, room_logit_end + n_icon),
                        n_icon - 1,
                    ),
                ):
                    pred_map = outputs[0, slc].argmax(dim=0).detach().cpu().numpy()
                    fig = plt.figure(figsize=(18, 12))
                    plot = fig.add_subplot(111)
                    cax = plot.imshow(pred_map, vmin=0, vmax=vmax, cmap=cmap)
                    fig.colorbar(cax)
                    self.writer.add_figure(
                        f"Image {i} prediction/{head}_{segmentation_map}",
                        fig,
                        global_step=step,
                    )
                    plt.close(fig)
