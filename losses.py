"""Objectives for HiPest-Net.

Stage I  : class-balanced focal loss on the composed 102-way log-probabilities
           plus a crop-level negative log-likelihood.
Stage II : plain negative log-likelihood (the long-tail correction comes from
           class-balanced sampling, as in decoupled training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import class_to_group


class ClassBalancedFocalLoss(nn.Module):
    """Focal loss weighted by the effective number of samples (Cui et al., 2019).

    Works on log-probabilities so it can be applied directly to the hierarchical
    decoding output. Weights are normalised to sum to the number of classes.
    """

    def __init__(self, class_counts, beta: float = 0.9999, gamma: float = 2.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        effective = 1.0 - torch.pow(beta, counts)
        w = (1.0 - beta) / effective.clamp_min(1e-12)
        w = w / w.sum() * len(counts)
        self.register_buffer("weight", w)
        self.gamma = gamma

    def forward(self, logp, target):
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -self.weight[target] * (1.0 - p_t).pow(self.gamma) * logp_t
        return loss.mean()


class HiPestLoss(nn.Module):
    def __init__(self, class_counts, kind: str = "cbfocal", crop_weight: float = 0.3,
                 beta: float = 0.9999, gamma: float = 2.0, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        assert kind in ("cbfocal", "ce")
        self.kind = kind
        self.crop_weight = crop_weight
        self.cb = ClassBalancedFocalLoss(class_counts, beta, gamma) if kind == "cbfocal" else None
        self.register_buffer("c2g", torch.as_tensor(class_to_group(), dtype=torch.long))

    def forward(self, logp, logp_crop, target):
        if self.kind == "cbfocal":
            loss = self.cb(logp, target)
        else:
            loss = F.nll_loss(logp, target)
            if self.smoothing > 0:
                loss = (1.0 - self.smoothing) * loss - self.smoothing * logp.mean()
        if logp_crop is not None and self.crop_weight > 0:
            loss = loss + self.crop_weight * F.nll_loss(logp_crop, self.c2g[target])
        return loss
