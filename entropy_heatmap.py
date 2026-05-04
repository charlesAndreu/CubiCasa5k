import math

import torch
import torch.nn.functional as F

LOG2 = math.log(2.0)


def top2_entropy_heatmap(logits: torch.Tensor):

    # Softmax over classes
    probs = F.softmax(logits, dim=0)  # [C, H, W]

    # Top-2 probabilities / classes
    top2_probs, top2_classes = torch.topk(probs, k=2, dim=0)

    p1 = top2_probs[0]
    p2 = top2_probs[1]

    c1 = top2_classes[0]
    c2 = top2_classes[1]

    # Renormalize locally over top-2
    s = p1 + p2

    p1_norm = p1 / (s + 1e-8)
    p2_norm = p2 / (s + 1e-8)

    # Binary entropy (bits), max log(2) when p1_norm = p2_norm = 0.5
    entropy = -(
        p1_norm * torch.log(p1_norm + 1e-8)
        + p2_norm * torch.log(p2_norm + 1e-8)
    )

    heatmap_norm = entropy / LOG2

    return heatmap_norm, c1, c2
