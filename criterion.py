import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from floortrans.loaders.room_icon_loaders import (
    ICON_MINI_DEFAULT_CLASS,
    ICON_MINI_MAPPING,
    ROOM_MINI_DEFAULT_CLASS,
    ROOM_MINI_MAPPING,
)
from weight import (
    N_ICON_MINI_CLASSES,
    N_ROOM_MINI_CLASSES,
    Weights,
    aggregate_full_counts_to_mini,
)

from torch.nn.functional import mse_loss, cross_entropy, interpolate
import pandas as pd


def _class_weights_tensor(args, segmentation_map, logger, weights_method=None):
    method = weights_method or getattr(args, "weights_method", None)
    if not method:
        return None
    with open("class_counts.json", "r") as f:
        class_counts = json.load(f)
    counts_list = class_counts.get(segmentation_map)
    if counts_list is None and segmentation_map == "room-mini":
        counts_list = aggregate_full_counts_to_mini(
            class_counts["room"],
            ROOM_MINI_MAPPING,
            ROOM_MINI_DEFAULT_CLASS,
            N_ROOM_MINI_CLASSES,
        )
    elif counts_list is None and segmentation_map == "icon-mini":
        counts_list = aggregate_full_counts_to_mini(
            class_counts["icon"],
            ICON_MINI_MAPPING,
            ICON_MINI_DEFAULT_CLASS,
            N_ICON_MINI_CLASSES,
        )

    if counts_list is None:
        logger.warning(
            "No class_counts entry for '%s' (and no full-map fallback); frequency weights disabled.",
            segmentation_map,
        )
        return None

    counts = torch.tensor(counts_list, dtype=torch.float32)
    weights = Weights(counts).weights(method=method)
    logger.info("Setting up %s loss weights for %s: %s", method, segmentation_map, weights)
    return weights


class CrossEntropyAndTverskyLoss(nn.Module):
    def __init__(
        self,
        tversky_weight=0.5,
        alpha=0.6,
        beta=0.4,
        weight=None,
    ):
        super().__init__()
        self.tversky_weight = float(tversky_weight)
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone().float())
        else:
            self.register_buffer("weight", None)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)
        self.tversky = smp.losses.TverskyLoss(
            mode="multiclass",
            from_logits=True,
            alpha=alpha,
            beta=beta,
            smooth=1e-6,
        )

    def forward(self, logits, target):
        return self.cross_entropy(logits, target) + self.tversky_weight * self.tversky(
            logits, target
        )


class CrossEntropyAndDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0, weight=None):
        super().__init__()
        self.dice_weight = dice_weight
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone().float())
        else:
            self.register_buffer("weight", None)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)
        self.dice = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True, smooth=1e-6
        )

    def forward(self, logits, target):
        return (
            self.cross_entropy(logits, target)
            + self.dice(logits, target) * self.dice_weight
        )


class CrossEntropyLearnedWeightsLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.s = nn.Parameter(torch.ones(num_classes))

    # def forward(self, logits, target):
    #     ce = F.cross_entropy(logits, target, reduction="none")
    #     w = torch.exp(-self.s)
    #     wt = w[target]
    #     return (ce * wt).mean() + 0.5 * self.s.mean()

    def normalized_class_weights(self):
        w_raw = torch.exp(-self.s)
        return w_raw / (w_raw.mean() + 1e-8)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        w = self.normalized_class_weights()
        wt = w[target]
        reg = -0.5 * torch.log(w + 1e-8).mean()
        return (ce * wt).mean() + reg


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = float(gamma)
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone().float())
        else:
            self.register_buffer("weight", None)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()  # mean reduction


class UncertaintyCustomLoss(nn.Module):
    def __init__(
        self,
        input_slice=[21, 4, 4],
        target_slice=[21, 1, 1],
        sub=0,
        cuda=True,
        room_weight=None,
        icon_weight=None,
    ):
        super(UncertaintyCustomLoss, self).__init__()
        self.input_slice = input_slice
        self.target_slice = target_slice
        self.loss = None
        self.loss_rooms = None
        self.loss_icons = None
        self.loss_heatmap = None
        self.sub = sub
        self.cuda = cuda and torch.cuda.is_available()
        self.log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.log_vars_mse = nn.Parameter(
            torch.zeros(input_slice[0], dtype=torch.float32)
        )
        if room_weight is not None:
            self.register_buffer(
                "room_weight", room_weight.detach().clone().float()
            )
        else:
            self.register_buffer("room_weight", None)
        if icon_weight is not None:
            self.register_buffer(
                "icon_weight", icon_weight.detach().clone().float()
            )
        else:
            self.register_buffer("icon_weight", None)

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ct, ht, wt = target.size()
        if h != ht or w != wt:  # upsample labels
            target = target.unsqueeze(1)
            target = interpolate(target, size=(ct, h, w), mode="nearest")
            target = target.squeeze(1)

        pred_arr = torch.split(input, self.input_slice, 1)
        heatmap_pred, rooms_pred, icons_pred = pred_arr

        target_arr = torch.split(target, self.target_slice, 1)
        heatmap_target, rooms_target, icons_target = target_arr

        # removing empty dimension if batch size is 1
        rooms_target = torch.squeeze(rooms_target, 1)
        icons_target = torch.squeeze(icons_target, 1)

        # Segmentation labels to correct type
        if self.cuda:
            rooms_target = rooms_target.type(torch.cuda.LongTensor) - self.sub
            icons_target = icons_target.type(torch.cuda.LongTensor) - self.sub
        else:
            rooms_target = rooms_target.type(torch.LongTensor) - self.sub
            icons_target = icons_target.type(torch.LongTensor) - self.sub

        # as in original paper, variance is applied directly to the logits
        # in simple model, we use variance after calculating the loss
        self.loss_rooms_var = cross_entropy(
            input=rooms_pred * torch.exp(-self.log_vars[0]),
            target=rooms_target,
            weight=self.room_weight,
        )
        self.loss_icons_var = cross_entropy(
            input=icons_pred * torch.exp(-self.log_vars[1]),
            target=icons_target,
            weight=self.icon_weight,
        )

        # for logging purposes, we compute the loss without the variance
        self.loss_rooms = cross_entropy(
            input=rooms_pred, target=rooms_target, weight=self.room_weight
        )
        self.loss_icons = cross_entropy(
            input=icons_pred, target=icons_target, weight=self.icon_weight
        )

        self.loss_heatmap_var = self.homosced_heatmap_mse_loss(
            heatmap_pred, heatmap_target, self.log_vars_mse
        )
        self.loss_heatmap = mse_loss(input=heatmap_pred, target=heatmap_target)

        self.loss = self.loss_rooms + self.loss_icons + self.loss_heatmap
        # self.loss = self.loss_heatmap
        self.loss_var = (
            self.loss_rooms_var + self.loss_icons_var + self.loss_heatmap_var
        )
        # self.loss_var = self.loss_heatmap_var

        return self.loss_var

    def homosced_heatmap_mse_loss(self, input, target, logvars):
        # we have n heatmaps, i.e. n heatmap tasks
        n, ntasks, h, w = input.size()

        # make a 2d tensor from both input and target  so that we have n tasks cols
        preds = input.permute(0, 2, 3, 1).contiguous().view(-1, ntasks)
        targets = target.permute(0, 2, 3, 1).contiguous().view(-1, ntasks)

        # take elementwise subtraction and raise to the power of two
        diff = (preds - targets) ** 2

        # measure task dependent mse loss
        mse_loss_per_tasks = torch.sum(diff, 0) / (n * h * w)

        # apply uncertainty magic
        # w_mse_loss = torch.exp(-logvars) * mse_loss_per_tasks + logvars
        w_mse_loss = torch.exp(-logvars) * mse_loss_per_tasks + torch.log(
            1 + torch.exp(logvars)
        )

        # take sum and return it
        w_mse_loss_total = w_mse_loss.sum()

        return w_mse_loss_total

    def get_loss_scalars(self) -> dict:
        """Plain float scalars for each loss component (no pandas dependency)."""
        return {
            "total":       float(self.loss.item()),
            "rooms":       float(self.loss_rooms.item()),
            "icons":       float(self.loss_icons.item()),
            "heatmap":     float(self.loss_heatmap.item()),
            "total_var":   float(self.loss_var.item()),
            "rooms_var":   float(self.loss_rooms_var.item()),
            "icons_var":   float(self.loss_icons_var.item()),
            "heatmap_var": float(self.loss_heatmap_var.item()),
        }

    def get_uncertainty_scalars(self) -> dict:
        """Learned uncertainty variances as plain floats."""
        variance = torch.exp(self.log_vars.data)
        return {
            "room_var":  float(variance[0].item()),
            "icon_var":  float(variance[1].item()),
        }

    def get_loss(self):
        d = {
            "total loss": [self.loss.data],
            "room loss": [self.loss_rooms.data],
            "icon loss": [self.loss_icons.data],
            "heatmap loss": [self.loss_heatmap.data],
            "total loss with variance": [self.loss_var.data],
            "room loss with variance": [self.loss_rooms_var.data],
            "icon loss with variance": [self.loss_icons_var.data],
            "heatmap loss with variance": [self.loss_heatmap_var.data],
        }
        return pd.DataFrame(data=d)

    def get_var(self):
        variance = torch.exp(self.log_vars.data)
        mse_variance = torch.exp(self.log_vars_mse.data)
        d = {"room variance": [variance[0]], "icon variance": [variance[1]]}
        for i, m in enumerate(mse_variance):
            key = "heatmap " + str(i)
            d[key] = [m]

        return pd.DataFrame(data=d)

    def get_s(self):
        s = self.log_vars.data
        mse_s = self.log_vars_mse.data
        d = {"room s": [s[0]], "icon s": [s[1]]}
        for i, m in enumerate(mse_s):
            key = "heatmap s" + str(i)
            d[key] = [m]

        return pd.DataFrame(data=d)


class HeatmapMSELoss(nn.Module):
    """Plain per-pixel MSE between an already-sigmoided prediction and a gaussian
    heatmap target (both in [0,1])."""

    def forward(self, pred, target):
        return mse_loss(input=pred, target=target)


class HeatmapFocalLoss(nn.Module):
    """CornerNet/CenterNet-style penalty-reduced pixelwise focal loss for keypoint
    heatmaps. Plain MSE gives a weak, imbalanced gradient once the gaussian target is
    made thin/sharp (the positive region shrinks to a handful of pixels); this loss
    keeps full gradient at true peaks while down-weighting near-peak "soft negative"
    pixels via (1-target)**beta, instead of penalizing them as hard negatives.
    """

    def __init__(self, alpha=2.0, beta=4.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target):
        # Compute in float32 regardless of the caller's autocast dtype: under fp16/bf16
        # a sigmoid output can round to exactly 0.0/1.0, and the eps-clamp below can
        # itself round back to 0.0 at reduced precision -- log(0) = -inf, and that -inf
        # times a mask of 0 (a background pixel that should contribute nothing) is NaN,
        # not 0. float32 keeps the clamp bound meaningfully non-zero.
        pred = pred.float().clamp(self.eps, 1.0 - self.eps)
        target = target.float()
        pos_mask = target.eq(1.0).float()
        neg_mask = target.lt(1.0).float()
        neg_weights = torch.pow(1.0 - target, self.beta)

        pos_loss = torch.log(pred) * torch.pow(1.0 - pred, self.alpha) * pos_mask
        neg_loss = (
            torch.log(1.0 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_mask
        )

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


def build_wall_criterion(args, device, logger):
    """Heatmap-only criterion for train_wall.py (no room/icon heads)."""
    name = getattr(args, "criterion", "mse")
    if name == "mse":
        return HeatmapMSELoss().to(device)
    if name == "focal-heatmap":
        return HeatmapFocalLoss().to(device)
    raise ValueError(f"Invalid criterion: {name}")


def build_simple_criterion(args, segmentation_map, n_output_channels, device, logger):
    """Construct the training criterion from CLI-style args (criterion, weights_method, etc.)."""
    weight = _class_weights_tensor(args, segmentation_map, logger)
    name = args.criterion
    if name == "cross-entropy":
        return nn.CrossEntropyLoss(weight=weight).to(device)
    if name == "focal-loss":
        return FocalLoss(gamma=args.focal_gamma, weight=weight).to(device)
    if name == "cross-entropy-and-dice":
        dice_weight = getattr(args, "dice_weight", 1.0)
        return CrossEntropyAndDiceLoss(dice_weight=dice_weight, weight=weight).to(
            device
        )
    if name == "cross-entropy-and-tversky":
        tversky_weight = getattr(args, "tversky_weight", 0.5)
        tversky_alpha = getattr(args, "tversky_alpha", 0.6)
        tversky_beta = getattr(args, "tversky_beta", 0.4)
        return CrossEntropyAndTverskyLoss(
            tversky_weight=tversky_weight,
            alpha=tversky_alpha,
            beta=tversky_beta,
            weight=weight,
        ).to(device)
    if name == "cross-entropy-learned-weights":
        return CrossEntropyLearnedWeightsLoss(n_output_channels).to(device)
    raise ValueError(f"Invalid criterion: {name}")


def build_full_criterion(args, input_slice, device, logger):
    """Build UncertaintyCustomLoss with optional per-class weights for room and icon heads."""
    room_weight = _class_weights_tensor(
        args, "room-mini", logger, weights_method=getattr(args, "room_weights_method", None)
    )
    icon_weight = _class_weights_tensor(
        args, "icon-mini", logger, weights_method=getattr(args, "icon_weights_method", None)
    )
    return UncertaintyCustomLoss(
        input_slice=input_slice,
        room_weight=room_weight,
        icon_weight=icon_weight,
    ).to(device)
