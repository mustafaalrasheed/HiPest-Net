"""Bidirectional feature pyramid network (Tan et al., EfficientDet, CVPR 2020).

Adapted to classification: every level is resampled with nearest-neighbour
up-sampling or adaptive max-pooling, so any input resolution is handled.
Level 0 is the highest-resolution map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BiFPNLayer(nn.Module):
    def __init__(self, ch: int, n_levels: int, eps: float = 1e-4):
        super().__init__()
        self.n = n_levels
        self.eps = eps
        self.w_td = nn.Parameter(torch.ones(n_levels - 1, 2))
        self.w_bu = nn.Parameter(torch.ones(max(n_levels - 2, 0), 3))
        self.w_bu_last = nn.Parameter(torch.ones(2))
        self.conv_td = nn.ModuleList([SepConv(ch) for _ in range(n_levels - 1)])
        self.conv_bu = nn.ModuleList([SepConv(ch) for _ in range(n_levels - 1)])

    def _fuse(self, w, xs):
        w = F.relu(w)
        w = w / (w.sum() + self.eps)
        out = w[0] * xs[0]
        for i in range(1, len(xs)):
            out = out + w[i] * xs[i]
        return out

    def forward(self, xs):
        n = self.n
        td = [None] * n
        td[-1] = xs[-1]
        for i in range(n - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=xs[i].shape[-2:], mode="nearest")
            td[i] = self.conv_td[i](self._fuse(self.w_td[i], [xs[i], up]))
        out = [None] * n
        out[0] = td[0]
        for i in range(1, n):
            down = F.adaptive_max_pool2d(out[i - 1], xs[i].shape[-2:])
            if i < n - 1:
                fused = self._fuse(self.w_bu[i - 1], [xs[i], td[i], down])
            else:
                fused = self._fuse(self.w_bu_last, [xs[i], down])
            out[i] = self.conv_bu[i - 1](fused)
        return out


class BiFPN(nn.Module):
    def __init__(self, in_channels, ch: int = 128, num_layers: int = 2):
        super().__init__()
        self.proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, ch, 1, bias=False), nn.BatchNorm2d(ch)) for c in in_channels]
        )
        self.layers = nn.ModuleList([BiFPNLayer(ch, len(in_channels)) for _ in range(num_layers)])

    def forward(self, feats):
        xs = [p(f) for p, f in zip(self.proj, feats)]
        for layer in self.layers:
            xs = layer(xs)
        return xs
