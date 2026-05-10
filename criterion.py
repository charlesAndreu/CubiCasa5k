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
    if segmentation_map not in class_counts:
        logger.warning(
            "No class_counts entry for '%s'; frequency weights disabled.",
            segmentation_map,
        )
        return None
    counts = torch.tensor(class_counts[segmentation_map], dtype=torch.float32)
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
        """``w = exp(-s)`` with mean 1 over classes (used in ``forward``)."""
        w_raw = torch.exp(-self.s)
        return w_raw / (w_raw.mean() + 1e-8)

    def forward(self, logits, target):
        # Per-class multipliers must not be passed as `weight=` to F.cross_entropy when they
        # depend on parameters (PyTorch disallows grad through that argument); apply after CE.
        ce = F.cross_entropy(logits, target, reduction="none")
        w = self.normalized_class_weights()
        wt = w[target]
        # Mean-normalized w: regularize spread of s (not s.mean(); see legacy block above).
        # 0.2 changed from 0.5 to reduce the regularization strength in case of high class imbalance
        # TODO: return to 0.5 after further testing
        reg = -0.2 * torch.log(w + 1e-8).mean()
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
