import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from weight import Weights


def _class_weights_tensor(args, segmentation_map, logger):
    if not args.weights_method:
        return None
    with open("class_counts.json", "r") as f:
        class_counts = json.load(f)
    counts = torch.tensor(class_counts[segmentation_map], dtype=torch.float32)
    weights = Weights(counts).weights(method=args.weights_method)
    logger.info("Setting up loss weights: %s", weights)
    return weights

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

    def forward(self, logits, target):
        # Per-class multipliers must not be passed as `weight=` to F.cross_entropy when they
        # depend on parameters (PyTorch disallows grad through that argument); apply after CE.
        ce = F.cross_entropy(logits, target, reduction="none")
        w = torch.exp(-self.s)
        wt = w[target]
        return (ce * wt).mean() + 0.5 * self.s.mean()


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
    if name == "cross-entropy-learned-weights":
        return CrossEntropyLearnedWeightsLoss(n_output_channels).to(device)
    raise ValueError(f"Invalid criterion: {name}")
