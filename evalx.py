"""Post-hoc evaluation of trained checkpoints: test-time resolution, flip averaging, weight soups.

Everything that is chosen (resolution, flip, soup members) is chosen on the validation split.
The test split is then evaluated once per final configuration.  Class log-probabilities are
saved so that probability-averaged ensembles can be computed afterwards.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from data import IMAGENET_MEAN, IMAGENET_STD, make_dataset
from engine import compute_metrics
from model import build_model

GROUPS = {
    "base1k": ["v2_convnext_s0", "v2_convnext_s1", "v2_convnext_s2"],
    "hip1k": ["v2_hipest_s0", "v2_hipest_s1", "v2_hipest_s2"],
    "base22": ["in22k_convnext_s0", "in22k_convnext_s1"],
    "hip22": ["in22k_hipest_s0", "in22k_hipest_s1"],
}


def say(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def tf(size):
    return T.Compose([
        T.Resize(int(size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["args"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), ck["args"]


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def average(states):
    out = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            out[k] = sum(s[k].float() for s in states) / len(states)
        else:
            out[k] = states[0][k]
    return out


@torch.no_grad()
def infer(model, loader, device):
    """Returns (log-probs of the image, log-probs averaged with its horizontal flip, labels)."""
    plain, flipped, ys = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            a, _ = model(x)
            b, _ = model(torch.flip(x, dims=[3]))
        a, b = a.float().cpu(), b.float().cpu()
        plain.append(a)
        flipped.append((a + b) / 2)
        ys.append(y)
    return torch.cat(plain), torch.cat(flipped), torch.cat(ys)


def get_loader(args, split, size):
    ds = make_dataset(args.data_root, split, tf(size))
    return DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True), ds


def top1(lp, y):
    return float((lp.argmax(1) == y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs", required=True, help="directory that holds the run folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 256, 288, 320])
    ap.add_argument("--select-runs", nargs="+", default=["v2_convnext_s0", "v2_hipest_s0"])
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_path = os.path.join(a.out, "results.json")
    R = json.load(open(res_path)) if os.path.isfile(res_path) else {}
    save = lambda: json.dump(R, open(res_path, "w"), indent=1)
    train_counts = make_dataset(a.data_root, "train", None).class_counts()
    ckpt = lambda name: os.path.join(a.runs, name, "final.pt")

    # ---- 1. choose test-time resolution and flip on the validation split --------------------
    if "select" not in R:
        grid = {}
        for name in a.select_runs:
            model, _ = load_model(ckpt(name), device)
            for s in a.sizes:
                loader, _ = get_loader(a, "val", s)
                p, f, y = infer(model, loader, device)
                grid[f"{name}|{s}|0"], grid[f"{name}|{s}|1"] = top1(p, y), top1(f, y)
                say(f"select {name} size {s}: plain {grid[f'{name}|{s}|0']:.4f} flip {grid[f'{name}|{s}|1']:.4f}")
            del model
            torch.cuda.empty_cache()
        score = {(s, fl): float(np.mean([grid[f"{n}|{s}|{fl}"] for n in a.select_runs])) for s in a.sizes for fl in (0, 1)}
        best = max(score, key=lambda k: (round(score[k], 6), -k[0], -k[1]))
        R["select"] = {"grid": grid, "mean_val_top1": {f"{k[0]}|{k[1]}": v for k, v in score.items()},
                       "size": best[0], "flip": best[1]}
        save()
    size, flip = R["select"]["size"], R["select"]["flip"]
    say(f"chosen test-time config: size {size}, flip {flip}")

    # ---- 2. apply the chosen configuration to every model (validation and test once) ---------
    def evaluate(model, name, splits=("val", "test"), tag=None):
        tag = tag or name
        entry = {}
        for sp in splits:
            loader, _ = get_loader(a, sp, size)
            p, f, y = infer(model, loader, device)
            lp = f if flip else p
            m, _ = compute_metrics(lp, y, train_counts)
            entry[sp] = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            np.save(os.path.join(a.out, f"{tag}_{sp}.npy"), lp.half().numpy())
            np.save(os.path.join(a.out, f"labels_{sp}.npy"), y.numpy())
        R[tag] = entry
        save()
        say(f"{tag}: " + " | ".join(f"{sp} top-1 {entry[sp]['top1']*100:.2f}" for sp in entry))

    for g in a.groups:
        for name in GROUPS[g]:
            if name in R or not os.path.isfile(ckpt(name)):
                continue
            model, _ = load_model(ckpt(name), device)
            evaluate(model, name)
            del model
            torch.cuda.empty_cache()

    # ---- 3. weight soups per group: uniform and greedy (greedy chosen on validation) ---------
    yval = torch.from_numpy(np.load(os.path.join(a.out, "labels_val.npy")))
    for g in a.groups:
        names = [n for n in GROUPS[g] if n in R]
        if len(names) < 2 or f"soup_uniform_{g}" in R:
            continue
        states = {n: load_state(ckpt(n)) for n in names}
        model, _ = load_model(ckpt(names[0]), device)
        model.load_state_dict(average([states[n] for n in names]))
        evaluate(model, None, tag=f"soup_uniform_{g}")
        order = sorted(names, key=lambda n: -R[n]["val"]["top1"])
        chosen, best_val = [order[0]], R[order[0]]["val"]["top1"]
        for n in order[1:]:
            model.load_state_dict(average([states[c] for c in chosen + [n]]))
            loader, _ = get_loader(a, "val", size)
            p, f, _ = infer(model, loader, device)
            v = top1(f if flip else p, yval)
            say(f"greedy {g}: try {n}: val {v:.4f} (current {best_val:.4f})")
            if v >= best_val:
                chosen, best_val = chosen + [n], v
        model.load_state_dict(average([states[c] for c in chosen]))
        evaluate(model, None, tag=f"soup_greedy_{g}")
        R[f"soup_greedy_{g}"]["members"] = chosen
        save()
        del model, states
        torch.cuda.empty_cache()
    open(os.path.join(a.out, "DONE.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    say("done")


if __name__ == "__main__":
    main()
