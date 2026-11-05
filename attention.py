"""Attention blocks used by HiPest-Net.

Stage 1: Coordinate Attention (Hou et al., CVPR 2021)
Stage 2: Channel Re-Attention + Spatial Multi-Region Attention (CRA-SMRA)
Stage 3: Large Kernel Attention (Guo et al., Computational Visual Media 2023)

Every block is wrapped in an identity-initialised gate (see IdentityGate) so the
pretrained ConvNeXt features are left untouched at the start of fine-tuning.

NOTE on CRA-SMRA: the block below is the formulation defined in the manuscript
(Section 3.2.2). It follows the design idea of re-attending to channels that a
first channel gate suppresses, followed by region-aware spatial gating. If you
want to use the exact formulation of a published CRA-SMRA module, replace the
CRASMRA class and cite that work accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityGate(nn.Module):
    """y = x + alpha * (module(x) - x), with a per-channel alpha initialised to 0.

    The wrapped attention block therefore starts as the identity mapping and
    its influence is learned, which keeps pretrained features intact.
    """

    def __init__(self, module: nn.Module, channels: int):
        super().__init__()
        self.module = module
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x + self.alpha * (self.module(x) - x)


class CoordAtt(nn.Module):
    """Coordinate attention: direction-aware pooling along H and W."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)  # N,C,H,1
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # N,C,W,1
        y = torch.cat([x_h, x_w], dim=2)  # N,C,H+W,1
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # N,mid,1,W
        a_h = torch.sigmoid(self.conv_h(y_h))  # N,C,H,1
        a_w = torch.sigmoid(self.conv_w(y_w))  # N,C,1,W
        return x * a_h * a_w


class ChannelReAttention(nn.Module):
    """Channel gate with a second pass over the channels the first gate suppressed.

    a = sigmoid(MLP1(avg) + MLP1(max))                 informative-channel gate
    r = sigmoid(MLP2(avg * (1 - a)))                   re-attention on the complement
    w = a + g * (1 - a) * r,   g = sigmoid(theta)      theta is a learned scalar
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp1 = nn.Sequential(nn.Linear(channels, mid), nn.GELU(), nn.Linear(mid, channels))
        self.mlp2 = nn.Sequential(nn.Linear(channels, mid), nn.GELU(), nn.Linear(mid, channels))
        self.theta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        a = torch.sigmoid(self.mlp1(avg) + self.mlp1(mx))
        r = torch.sigmoid(self.mlp2(avg * (1.0 - a)))
        g = torch.sigmoid(self.theta)
        w = a + g * (1.0 - a) * r
        return x * w[:, :, None, None]


class SpatialMultiRegionAttention(nn.Module):
    """Spatial gate informed by region descriptors at several grid resolutions."""

    def __init__(self, channels: int, regions=(1, 2, 4), kernel_size: int = 7):
        super().__init__()
        self.regions = tuple(regions)
        self.region_convs = nn.ModuleList([nn.Conv2d(channels, 1, 1) for _ in self.regions])
        self.local = nn.Conv2d(2 + len(self.regions), 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        size = x.shape[-2:]
        maps = [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)]
        for r, conv in zip(self.regions, self.region_convs):
            pooled = F.adaptive_avg_pool2d(x, r)
            maps.append(F.interpolate(conv(pooled), size=size, mode="bilinear", align_corners=False))
        return torch.sigmoid(self.local(torch.cat(maps, dim=1)))


class CRASMRA(nn.Module):
    """Channel Re-Attention followed by Spatial Multi-Region Attention."""

    def __init__(self, channels: int, reduction: int = 16, regions=(1, 2, 4)):
        super().__init__()
        self.cra = ChannelReAttention(channels, reduction)
        self.smra = SpatialMultiRegionAttention(channels, regions)

    def forward(self, x):
        x = self.cra(x)
        return x * self.smra(x)


class LKA(nn.Module):
    """Large kernel attention: 5x5 depth-wise + 7x7 dilated (d=3) depth-wise + 1x1.

    Together they approximate a dense 21x21 kernel (Guo et al.) at a fraction of
    its cost.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dw_d = nn.Conv2d(channels, channels, 7, padding=9, groups=channels, dilation=3)
        self.pw = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * self.pw(self.dw_d(self.dw(x)))
