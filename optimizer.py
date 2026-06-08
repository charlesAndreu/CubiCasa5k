import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _optimizer_parameters(model, criterion):
    """add any trainable parameters on the criterion (e.g. learned class weights)."""
    params = list(model.parameters())
    params.extend(criterion.parameters())
    return params


def build_optimizer_and_scheduler(args, model, criterion):
    params = _optimizer_parameters(model, criterion)
    name = args.optimizer
    # adam-patience: Adam + ReduceLROnPlateau on val loss
    if name == "adam-patience":
        optimizer = torch.optim.Adam(
            params,
            lr=args.l_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
        scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=args.patience, factor=0.5
        )
        return optimizer, scheduler

    # adam-patience-previous-best: Adam + manual patience logic
    # (reload best checkpoint + reduce LR by x0.1 after plateau)
    if name == "adam-patience-previous-best":
        optimizer = torch.optim.Adam(
            params,
            lr=args.l_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
        return optimizer, None

    # sgd: SGD with polynomial-style LambdaLR decay
    if name == "sgd":

        def lr_drop(epoch):
            return (1 - epoch / args.n_epoch) ** 0.9

        optimizer = torch.optim.SGD(
            params,
            lr=args.l_rate,
            momentum=0.9,
            weight_decay=10**-4,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_drop)
        return optimizer, scheduler

    # adam-scheduler: Adam with step-like LambdaLR decay
    if name == "adam-scheduler":

        def lr_drop_adam(epoch):
            return 0.5 ** np.floor(epoch / args.l_rate_drop)

        optimizer = torch.optim.Adam(
            params,
            lr=args.l_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_drop_adam)
        return optimizer, scheduler
    raise ValueError(f"Invalid optimizer: {name}")
