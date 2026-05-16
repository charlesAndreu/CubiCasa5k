import math

import torch
import torch.nn.functional as F


def all_classes_entropy_heatmap(logits: torch.Tensor):
    probs = F.softmax(logits, dim=0)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=0)
    n_classes = logits.shape[0]
    return entropy / math.log(n_classes)
