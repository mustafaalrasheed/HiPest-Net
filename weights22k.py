"""ImageNet-22k pretrained ConvNeXt-Tiny weights for the torchvision model.

The checkpoint comes from timm (convnext_tiny.fb_in22k_ft_in1k). Its parameter names
differ from torchvision's, so the tensors are renamed here and loaded into the
torchvision network. The classifier layer is not transferred.
"""


def convert(timm_sd):
    out = {}
    for k, v in timm_sd.items():
        if k.startswith("head."):
            continue
        if k.startswith("stem.0."):
            nk = "features.0.0." + k[len("stem.0."):]
        elif k.startswith("stem.1."):
            nk = "features.0.1." + k[len("stem.1."):]
        elif k.startswith("stages."):
            parts = k.split(".")
            i = int(parts[1])
            if parts[2] == "downsample":
                nk = "features.%d.%s." % (2 * i, parts[3]) + ".".join(parts[4:])
            else:
                j, name = int(parts[3]), parts[4]
                base = "features.%d.%d." % (2 * i + 1, j)
                if name == "conv_dw":
                    nk = base + "block.0." + ".".join(parts[5:])
                elif name == "norm":
                    nk = base + "block.2." + ".".join(parts[5:])
                elif name == "mlp":
                    nk = base + ("block.3." if parts[5] == "fc1" else "block.5.") + ".".join(parts[6:])
                elif name == "gamma":
                    nk = base + "layer_scale"
                    v = v.reshape(-1, 1, 1)
                else:
                    raise KeyError(k)
        else:
            raise KeyError(k)
        out[nk] = v
    return out


def load_in22k_into(net):
    import timm

    m = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True)
    missing, unexpected = net.load_state_dict(convert(m.state_dict()), strict=False)
    bad = [k for k in missing if not k.startswith("classifier")]
    if bad or unexpected:
        raise RuntimeError("in22k weight conversion mismatch: missing %s unexpected %s" % (bad, unexpected))
    return net
