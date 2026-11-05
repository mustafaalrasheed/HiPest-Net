"""HiPest-Net: hierarchical multi-scale attention network for pest recognition.

ConvNeXt-Tiny backbone with staged attention, a BiFPN neck, and a
taxonomy-aware (crop -> species) decoding head. Every component can be switched
off for the ablation study. Torchvision baselines with the same output
interface are provided for the comparison table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from attention import CRASMRA, LKA, CoordAtt, IdentityGate
from bifpn import BiFPN
from config import GROUP_SIZES, NUM_CLASSES

BACKBONE_NAMES = ("stem", "s1", "d2", "s2", "d3", "s3", "d4", "s4")


class HierarchicalHead(nn.Module):
    """Crop classifier plus one species classifier per crop.

    The species distribution of every crop is a softmax over that crop's classes
    only. The 102-way log-probability is the sum of the crop log-probability and
    the within-crop species log-probability, so training and inference use the
    same decoding rule.
    """

    def __init__(self, dim: int, group_sizes=GROUP_SIZES, drop: float = 0.2):
        super().__init__()
        self.crop = nn.Linear(dim, len(group_sizes))
        self.species = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(dim // 2, n)
                )
                for n in group_sizes
            ]
        )

    def forward(self, z):
        logp_crop = F.log_softmax(self.crop(z).float(), dim=1)  # B,G
        parts = []
        for g, head in enumerate(self.species):
            logp_sp = F.log_softmax(head(z).float(), dim=1)  # B,n_g
            parts.append(logp_sp + logp_crop[:, g : g + 1])
        return torch.cat(parts, dim=1), logp_crop  # B,C and B,G


class FlatHead(nn.Module):
    """Single C-way classifier (used when the hierarchical head is ablated)."""

    def __init__(self, dim: int, num_classes: int, drop: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, z):
        return F.log_softmax(self.fc(self.drop(z)).float(), dim=1), None


class HiPestNet(nn.Module):
    def __init__(
        self,
        group_sizes=GROUP_SIZES,
        pretrained: bool = True,
        use_coord: bool = True,
        use_cra: bool = True,
        use_lka: bool = True,
        use_bifpn: bool = True,
        hierarchical: bool = True,
        fusion: str = "skip",
        fpn_channels: int = 128,
        bifpn_layers: int = 2,
        drop: float = 0.2,
        num_classes=None,
        weights_tag: str = "in1k",
    ):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if (pretrained and weights_tag == "in1k") else None
        cn = convnext_tiny(weights=weights)
        if pretrained and weights_tag == "in22k":
            from weights22k import load_in22k_into
            load_in22k_into(cn)
        f = cn.features
        self.stem, self.s1, self.d2, self.s2 = f[0], f[1], f[2], f[3]
        self.d3, self.s3, self.d4, self.s4 = f[4], f[5], f[6], f[7]

        self.att1 = IdentityGate(CoordAtt(96), 96) if use_coord else nn.Identity()
        self.att2 = IdentityGate(CRASMRA(192), 192) if use_cra else nn.Identity()
        self.att3 = IdentityGate(LKA(384), 384) if use_lka else nn.Identity()

        assert fusion in ("skip", "neck")
        self.fusion = fusion if use_bifpn else "neck"
        if use_bifpn:
            self.neck = BiFPN((96, 192, 384, 768), fpn_channels, bifpn_layers)
            neck_dim = fpn_channels * 4
        else:
            self.neck = None
            neck_dim = 768
        if self.fusion == "skip":
            # pretrained stage-4 pathway kept intact; the multi-scale neck is added through a
            # zero-initialised gate, so the model starts as the plain backbone
            self.embed_skip = nn.LayerNorm(768)
            self.embed_neck = nn.LayerNorm(neck_dim)
            self.neck_gate = nn.Parameter(torch.zeros(neck_dim))
            emb_dim = 768 + neck_dim
        else:
            self.embed = nn.LayerNorm(neck_dim)
            emb_dim = neck_dim

        if hierarchical:
            self.head = HierarchicalHead(emb_dim, group_sizes, drop)
        else:
            self.head = FlatHead(emb_dim, num_classes or sum(group_sizes), drop)

    def features(self, x):
        x = self.stem(x)
        c1 = self.att1(self.s1(x))
        c2 = self.att2(self.s2(self.d2(c1)))
        c3 = self.att3(self.s3(self.d3(c2)))
        c4 = self.s4(self.d4(c3))
        return [c1, c2, c3, c4]

    def forward(self, x):
        feats = self.features(x)
        top = feats[-1].mean(dim=(2, 3))
        if self.fusion == "skip":
            outs = self.neck(feats)
            zn = torch.cat([o.mean(dim=(2, 3)) for o in outs], dim=1)
            z = torch.cat([self.embed_skip(top), self.neck_gate * self.embed_neck(zn)], dim=1)
        else:
            if self.neck is not None:
                outs = self.neck(feats)
                top = torch.cat([o.mean(dim=(2, 3)) for o in outs], dim=1)
            z = self.embed(top)
        return self.head(z)  # (log-probs [B,C], crop log-probs [B,G] or None)

    def reset_head(self):
        """Re-initialise the classification head (used before decoupled stage 2)."""
        for m in self.head.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


# ---------------------------------------------------------------- baselines
BASELINES = ("resnet50", "resnext50", "efficientnetv2_s", "swin_t", "convnext_t")


class TorchvisionBaseline(nn.Module):
    """Standard torchvision classifier returning (log-probabilities, None)."""

    def __init__(self, name: str, pretrained: bool = True, num_classes: int = NUM_CLASSES, weights_tag: str = "in1k"):
        super().__init__()
        if name == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            self.head_prefix = "net.fc"
        elif name == "resnext50":
            net = tvm.resnext50_32x4d(weights=tvm.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            self.head_prefix = "net.fc"
        elif name == "efficientnetv2_s":
            net = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
            self.head_prefix = "net.classifier.1"
        elif name == "swin_t":
            net = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
            net.head = nn.Linear(net.head.in_features, num_classes)
            self.head_prefix = "net.head"
        elif name == "convnext_t":
            net = tvm.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if (pretrained and weights_tag == "in1k") else None)
            if pretrained and weights_tag == "in22k":
                from weights22k import load_in22k_into
                load_in22k_into(net)
            net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
            self.head_prefix = "net.classifier.2"
        else:
            raise ValueError(f"unknown baseline {name}; choose from {BASELINES}")
        self.net = net

    def forward(self, x):
        return F.log_softmax(self.net(x).float(), dim=1), None


class TimmBaseline(nn.Module):
    """Any timm classifier, e.g. arch = "timm:vit_small_patch14_dinov2.lvd142m". Returns (log-probs, None)."""

    def __init__(self, spec: str, pretrained: bool = True, num_classes: int = NUM_CLASSES, img_size: int = 224,
                 drop_path: float = 0.0, grad_ckpt: bool = False):
        super().__init__()
        import timm

        name = spec.split(":", 1)[1]
        kw = dict(pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path)
        if any(t in name for t in ("vit", "deit", "eva", "swin", "beit")):
            kw["img_size"] = img_size
        self.net = timm.create_model(name, **kw)
        if grad_ckpt:
            self.net.set_grad_checkpointing(True)
        clf = self.net.get_classifier()
        self.head_prefix = next("net." + n for n, m in self.net.named_modules() if m is clf)
        self.depth = len(self.net.blocks) if hasattr(self.net, "blocks") else 0

    def layer_id(self, name: str) -> int:
        n = name[len("net."):]
        if n.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed")):
            return 0
        if n.startswith("blocks."):
            return int(n.split(".")[1]) + 1
        return self.depth + 1

    def forward(self, x):
        return F.log_softmax(self.net(x).float(), dim=1), None


def build_model(cfg: dict, pretrained: bool = True):
    """Build either HiPest-Net or a baseline from a plain config dict (CLI args)."""
    arch = cfg.get("arch", "hipest")
    if arch == "hipest":
        return HiPestNet(
            pretrained=pretrained,
            use_coord=not cfg.get("no_coord", False),
            use_cra=not cfg.get("no_cra", False),
            use_lka=not cfg.get("no_lka", False),
            use_bifpn=not cfg.get("no_bifpn", False),
            hierarchical=not cfg.get("flat_head", False),
            fusion=cfg.get("fusion", "skip"),
            weights_tag=cfg.get("weights", "in1k"),
        )
    if arch.startswith("timm:"):
        return TimmBaseline(arch, pretrained=pretrained, img_size=cfg.get("img_size", 224),
                            drop_path=cfg.get("drop_path", 0.0), grad_ckpt=cfg.get("grad_ckpt", False))
    return TorchvisionBaseline(arch, pretrained=pretrained, weights_tag=cfg.get("weights", "in1k"))


def build_param_groups(model: nn.Module, weight_decay: float, lr_backbone: float, lr_new: float, layer_decay: float = 1.0):
    """Parameter groups: {pretrained, new} x {decay, no decay}.

    For HiPest-Net the pretrained set is the ConvNeXt stages; for a baseline it is
    everything except the classification layer.
    """
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = 1.0
        if isinstance(model, (TorchvisionBaseline, TimmBaseline)):
            is_pretrained = not name.startswith(model.head_prefix)
            if is_pretrained and layer_decay < 1.0 and isinstance(model, TimmBaseline) and model.depth:
                scale = layer_decay ** (model.depth + 1 - model.layer_id(name))
        else:
            is_pretrained = name.split(".")[0] in BACKBONE_NAMES
        no_decay = (
            p.ndim <= 1
            or name.endswith("alpha")
            or "theta" in name
            or name.split(".")[-1].startswith("w_")
        )
        groups.setdefault((is_pretrained, no_decay, round(scale, 6)), []).append(p)
    out = []
    for (is_pretrained, no_decay, scale), params in groups.items():
        out.append(
            {
                "params": params,
                "lr": lr_backbone * scale if is_pretrained else lr_new,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return out
