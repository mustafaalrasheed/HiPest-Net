"""Two-stage training of HiPest-Net on IP102.

Stage I  : joint training of backbone, attention blocks, neck and heads with the
           class-balanced focal objective and instance-balanced sampling.
Stage II : decoupled classifier re-training. Backbone, attention blocks and neck
           are frozen; the heads are re-initialised and re-trained with
           class-balanced sampling and a plain likelihood objective.

Example
-------
python train.py --data-root /data/IP102 --out runs/full_s0 --seed 0

Ablations are obtained with the --no-* flags, --flat-head, --loss ce and
--skip-stage2 (see scripts/run_experiments.sh).
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.amp import GradScaler

from config import NUM_CLASSES
from data import build_transforms, class_balanced_sampler, make_dataset, make_loader
from engine import (compute_metrics, cosine_with_warmup, count_gmacs, count_params, predict,
                    throughput, train_one_epoch)
from losses import HiPestLoss
from model import BASELINES, build_model, build_param_groups


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", default="hipest",
                   help="hipest, a torchvision baseline (%s) or timm:<model name>" % ", ".join(BASELINES))
    p.add_argument("--layer-decay", type=float, default=1.0, help="layer-wise lr decay for timm ViTs")
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--init", default="", help="initialise all weights from a trained checkpoint (e.g. for a high-resolution fine-tune)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs1", type=int, default=100)
    p.add_argument("--epochs2", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-new", type=float, default=1e-3)
    p.add_argument("--lr-stage2", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--loss", choices=["cbfocal", "ce"], default="cbfocal")
    p.add_argument("--beta", type=float, default=0.9999)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--crop-weight", type=float, default=0.3)
    p.add_argument("--sampling-q-stage2", type=float, default=0.0, help="0 = class-balanced, 0.5 = sqrt")
    p.add_argument("--cutmix", type=float, default=0.0, help="probability of applying CutMix to a stage-I batch")
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--smoothing", type=float, default=0.0, help="label smoothing for the cross-entropy objective")
    p.add_argument("--fusion", choices=["skip", "neck"], default="skip",
                   help="skip: pretrained top feature + gated BiFPN features; neck: BiFPN features only")
    p.add_argument("--weights", choices=["in1k", "in22k"], default="in1k", help="pretraining of the ConvNeXt-Tiny backbone")
    p.add_argument("--no-coord", action="store_true")
    p.add_argument("--no-cra", action="store_true")
    p.add_argument("--no-lka", action="store_true")
    p.add_argument("--no-bifpn", action="store_true")
    p.add_argument("--flat-head", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore existing *_last.pt files and start from scratch (default: continue an interrupted run)")
    p.add_argument("--smoke", action="store_true", help="tiny run on a handful of batches")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def log(path, record):
    print(json.dumps(record))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def stage2_mode(model):
    """Frozen feature extractor in eval mode; only the head is trained."""
    model.eval()
    model.head.train()


def fit(tag, model, args, device, amp, train_loader, val_loader, criterion, groups, epochs,
        train_counts, out_dir, log_path, mode_fn=None, mix=None):
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, args.warmup_epochs * steps_per_epoch, epochs * steps_per_epoch)
    scaler = GradScaler("cuda", enabled=amp)
    best, start = -1.0, 0
    ckpt = os.path.join(out_dir, f"{tag}_best.pt")
    last = os.path.join(out_dir, f"{tag}_last.pt")

    # Continue an interrupted run from the last completed epoch.
    if not args.no_resume and os.path.isfile(last):
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best, start = state["best"], state["epoch"]
        print(f"resuming {tag} after epoch {start} (best val top-1 so far {best:.4f})", flush=True)

    for epoch in range(start, epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler,
                                          device, amp, mode_fn=mode_fn, mix=mix)
        logp, y = predict(model, val_loader, device, amp)
        m, _ = compute_metrics(logp, y, train_counts)
        log(log_path, {"stage": tag, "epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                       "val_top1": m["top1"], "val_macro_f1": m["macro_f1"], "val_few": m["few_shot"]})
        if m["top1"] > best:
            best = m["top1"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_top1": best,
                        "args": vars(args)}, ckpt)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best": best}, last + ".tmp")
        os.replace(last + ".tmp", last)  # atomic: an interrupted save never corrupts the checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return best


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    tf_train, tf_test = build_transforms(args.img_size)
    ds_train = make_dataset(args.data_root, "train", tf_train)
    ds_val = make_dataset(args.data_root, "val", tf_test)
    ds_test = make_dataset(args.data_root, "test", tf_test)
    train_counts = ds_train.class_counts()
    if args.smoke:
        for ds in (ds_train, ds_val, ds_test):
            ds.names, ds.labels = ds.names[:256], ds.labels[:256]
        args.epochs1, args.epochs2, args.warmup_epochs, args.workers = 1, 1, 0, 0
        train_counts = ds_train.class_counts()

    train_loader = make_loader(ds_train, args.batch_size, True, args.workers, seed=args.seed)
    val_loader = make_loader(ds_val, args.batch_size, False, args.workers)
    test_loader = make_loader(ds_test, args.batch_size, False, args.workers)

    if args.arch != "hipest":  # baselines: flat head, plain cross-entropy, single stage
        args.flat_head, args.loss, args.skip_stage2 = True, "ce", True
    model = build_model(vars(args), pretrained=not args.no_pretrained)
    if args.init:
        st = torch.load(args.init, map_location="cpu", weights_only=False)
        st = st.get("model", st) if isinstance(st, dict) else st
        missing, unexpected = model.load_state_dict(st, strict=False)
        print(f"initialised from {args.init}: missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
        assert len(missing) == 0, missing[:5]
    model = model.to(device).to(memory_format=torch.channels_last)
    total_m, _ = count_params(model)
    print(f"parameters: {total_m:.2f} M")

    # ---- Stage I ----
    counts_for_loss = np.maximum(train_counts, 1)
    criterion = HiPestLoss(counts_for_loss, args.loss, args.crop_weight, args.beta, args.gamma,
                          args.smoothing).to(device)
    groups = build_param_groups(model, args.weight_decay, args.lr_backbone, args.lr_new, args.layer_decay)
    best1 = fit("stage1", model, args, device, amp, train_loader, val_loader, criterion, groups,
                args.epochs1, train_counts, args.out, log_path,
                mix={"prob": args.cutmix, "alpha": args.cutmix_alpha})
    print(f"stage I best val top-1: {best1:.4f}")

    # ---- Stage II (decoupled classifier re-training) ----
    if not args.skip_stage2:
        for p in model.parameters():
            p.requires_grad = False
        model.reset_head()
        for p in model.head.parameters():
            p.requires_grad = True
        sampler = class_balanced_sampler(ds_train.labels, q=args.sampling_q_stage2, seed=args.seed)
        loader2 = make_loader(ds_train, args.batch_size, True, args.workers, sampler=sampler, seed=args.seed)
        crit2 = HiPestLoss(counts_for_loss, "ce", args.crop_weight).to(device)
        groups2 = [{"params": [p for p in model.head.parameters()], "lr": args.lr_stage2,
                    "weight_decay": args.weight_decay}]
        best2 = fit("stage2", model, args, device, amp, loader2, val_loader, crit2, groups2,
                    args.epochs2, train_counts, args.out, log_path, mode_fn=stage2_mode)
        print(f"stage II best val top-1: {best2:.4f}")

    # ---- Final test evaluation with the selected checkpoint ----
    logp, y = predict(model, test_loader, device, amp)
    metrics, cm = compute_metrics(logp, y, train_counts)
    total_m, trainable_m = count_params(model)
    metrics.update({
        "params_M": total_m,
        "gmacs_224": count_gmacs(model, device, args.img_size),
        "throughput_img_s": throughput(model, device, size=args.img_size, amp=amp),
        "seed": args.seed,
        "config": vars(args),
    })
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    np.save(os.path.join(args.out, "confusion_matrix.npy"), cm)
    np.save(os.path.join(args.out, "test_pred.npy"), logp.argmax(1).numpy())
    np.save(os.path.join(args.out, "test_labels.npy"), y.numpy())
    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "final.pt"))
    print(f"TEST top-1 {metrics['top1']*100:.2f} | top-5 {metrics['top5']*100:.2f} | "
          f"macro-F1 {metrics['macro_f1']*100:.2f} | few-shot {metrics['few_shot']*100:.2f}")


if __name__ == "__main__":
    main()
