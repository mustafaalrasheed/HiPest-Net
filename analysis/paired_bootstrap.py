"""Paired bootstrap for the difference in top-1 accuracy between two models on the same test set.

Usage:
    python analysis/paired_bootstrap.py runs/v2_hipest_s0 runs/v2_convnext_s0 [--n 10000] [--seed 0]

Each run folder must contain test_pred.npy and test_labels.npy (written by train.py).
Prints the difference A - B in percentage points with a 95% percentile interval and the number of
test images that only one of the two models classifies correctly. The interval reflects sampling of
the test images only, not variation between training seeds.
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ya = np.load(os.path.join(a.run_a, "test_labels.npy"))
    yb = np.load(os.path.join(a.run_b, "test_labels.npy"))
    assert (ya == yb).all(), "the two runs were evaluated on different test labels"
    ca = np.load(os.path.join(a.run_a, "test_pred.npy")) == ya
    cb = np.load(os.path.join(a.run_b, "test_pred.npy")) == ya
    rng = np.random.default_rng(a.seed)
    n = len(ya)
    d = np.empty(a.n)
    for i in range(a.n):
        idx = rng.integers(0, n, n)
        d[i] = (ca[idx].mean() - cb[idx].mean()) * 100
    print(f"A - B = {(ca.mean() - cb.mean()) * 100:.2f} points, 95% interval "
          f"[{np.percentile(d, 2.5):.2f}, {np.percentile(d, 97.5):.2f}]; "
          f"only A correct: {int((ca & ~cb).sum())}, only B correct: {int((cb & ~ca).sum())}, n = {n}")


if __name__ == "__main__":
    main()
