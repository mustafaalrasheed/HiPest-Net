"""Training loop helpers, metrics and efficiency measurements."""

import math
import time

import numpy as np
import torch

from config import GROUP_NAMES, NUM_CLASSES, class_to_group


def cosine_with_warmup(optimizer, warmup_steps, total_steps, floor=0.01):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, amp, clip=1.0, mode_fn=None, mix=None):
    if mode_fn is None:
        model.train()
    else:
        mode_fn(model)
    total_loss, seen, correct = 0.0, 0, 0
    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        lam, perm = 1.0, None
        if mix is not None and mix["prob"] > 0 and np.random.rand() < mix["prob"]:
            lam = float(np.random.beta(mix["alpha"], mix["alpha"]))
            perm = torch.randperm(imgs.size(0), device=imgs.device)
            h, w = imgs.shape[-2:]
            cut = math.sqrt(1.0 - lam)
            ch, cw = int(h * cut), int(w * cut)
            cy, cx = np.random.randint(h), np.random.randint(w)
            y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
            x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
            imgs[:, :, y1:y2, x1:x2] = imgs[perm][:, :, y1:y2, x1:x2]
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logp, logp_crop = model(imgs)
        loss = criterion(logp, logp_crop, y)
        if perm is not None:
            loss = lam * loss + (1.0 - lam) * criterion(logp, logp_crop, y[perm])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * y.numel()
        correct += (logp.argmax(1) == y).sum().item()
        seen += y.numel()
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    all_logp, all_y = [], []
    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logp, _ = model(imgs)
        all_logp.append(logp.float().cpu())
        all_y.append(y)
    return torch.cat(all_logp), torch.cat(all_y)


def shot_masks(counts, many_thr=None, few_thr=None):
    """Boolean class masks for the many / medium / few-shot groups.

    With explicit thresholds the usual rule is used (many: more than many_thr training
    images, few: fewer than few_thr). Without thresholds the classes are ranked by
    training-set size and cut into thirds, which always gives three non-empty groups;
    this is needed for IP102, where no class has fewer than 20 training images.
    """
    counts = np.asarray(counts)
    if many_thr is not None and few_thr is not None:
        return {
            "many": counts > many_thr,
            "medium": (counts <= many_thr) & (counts >= few_thr),
            "few": counts < few_thr,
        }
    order = np.argsort(-counts, kind="stable")
    k = len(counts) // 3
    many = np.zeros(len(counts), dtype=bool)
    few = np.zeros(len(counts), dtype=bool)
    many[order[:k]] = True
    few[order[len(counts) - k:]] = True
    return {"many": many, "medium": ~(many | few), "few": few}


def compute_metrics(logp, y, train_counts, many_thr=None, few_thr=None):
    """Top-1/top-5, macro-F1, many/medium/few-shot accuracy and per-crop accuracy."""
    c = NUM_CLASSES
    y_np = y.numpy()
    pred = logp.argmax(1).numpy()
    top5 = logp.topk(min(5, c), dim=1).indices.numpy()
    top1_acc = float((pred == y_np).mean())
    top5_acc = float((top5 == y_np[:, None]).any(axis=1).mean())

    cm = np.bincount(y_np * c + pred, minlength=c * c).reshape(c, c)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    predicted = cm.sum(0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros(c), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(c), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(c), where=denom > 0)
    present = support > 0
    macro_f1 = float(f1[present].mean())
    macro_recall = float(recall[present].mean())

    counts = np.asarray(train_counts)
    shot = shot_masks(counts, many_thr, few_thr)
    shot_acc = {}
    for name, cls_mask in shot.items():
        sample_mask = cls_mask[y_np]
        shot_acc[name] = float((pred[sample_mask] == y_np[sample_mask]).mean()) if sample_mask.any() else float("nan")

    c2g = np.asarray(class_to_group())
    g_true, g_pred = c2g[y_np], c2g[pred]
    crop_level_acc = float((g_true == g_pred).mean())
    per_crop = {}
    for g, name in enumerate(GROUP_NAMES):
        m = g_true == g
        per_crop[name] = float((pred[m] == y_np[m]).mean()) if m.any() else float("nan")

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "crop_level_acc": crop_level_acc,
        "many_shot": shot_acc["many"],
        "medium_shot": shot_acc["medium"],
        "few_shot": shot_acc["few"],
        "per_crop_acc": per_crop,
        "per_class_recall": recall.tolist(),
    }, cm


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


@torch.no_grad()
def count_gmacs(model, device, size=224):
    """GMACs for one image (torch FlopCounterMode reports FLOPs = 2 x MACs)."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2 / 1e9


@torch.no_grad()
def throughput(model, device, batch_size=64, size=224, iters=30, warmup=10, amp=True):
    """Images per second at the given batch size."""
    model.eval()
    x = torch.randn(batch_size, 3, size, size, device=device).contiguous(memory_format=torch.channels_last)
    use_amp = amp and device.type == "cuda"
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batch_size * iters / (time.time() - t0)
