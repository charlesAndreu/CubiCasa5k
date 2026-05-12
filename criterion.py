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


def _class_weights_tensor(args, segmentation_map, logger):
    if not args.weights_method:
        return None
    with open("class_counts.json", "r") as f:
        class_counts = json.load(f)
    counts_list = class_counts.get(segmentation_map)
    if counts_list is None and segmentation_map == "room-mini" :
        counts_list = aggregate_full_counts_to_mini(
            class_counts["room"],
            ROOM_MINI_MAPPING,
            ROOM_MINI_DEFAULT_CLASS,
            N_ROOM_MINI_CLASSES,
        )
    elif counts_list is None and segmentation_map == "icon-mini" :
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
    weights = Weights(counts).weights(method=args.weights_method)
    logger.info("Setting up loss weights: %s", weights)
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


def build_criterion(args, segmentation_map, n_output_channels, device, logger):
    """Construct the training criterion from CLI-style ``args`` (``criterion``, ``weights_method``, etc.)."""
    weight = _class_weights_tensor(args, segmentation_map, logger)
    name = args.criterion
    if name == "cross-entropy":
        return nn.CrossEntropyLoss(weight=weight).to(device)
    if name == "focal-loss":
        return FocalLoss(gamma=args.focal_gamma, weight=weight).to(device)
    if name == "cross-entropy-and-dice":
        dice_weight = getattr(args, "dice_weight", 1.0)
        return CrossEntropyAndDiceLoss(
            dice_weight=dice_weight, weight=weight
        ).to(device)
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
