"""Lottery Ticket Hypothesis on sklearn digits (8x8) with a hand-rolled numpy MLP.

- 2-layer MLP (64 -> 128 -> 10), ReLU, softmax cross-entropy, Adam. All numpy.
- Iterative magnitude pruning (IMP), layerwise, biases never pruned.
- Conditions at each sparsity level:
    winning : IMP mask + rewind to original init
    reinit  : IMP mask + fresh random init
    randmask: random mask (same layerwise sparsity) + original init
- Multiple seeds. Results saved to results.json / masks_seed0.npz.
"""
import json
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HIDDEN = 128
EPOCHS = 60
BATCH = 64
LR = 1e-3
SEEDS = [0, 1, 2, 3, 4]
LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]  # overall sparsity targets

# ---------------- data ----------------
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
N, D = Xtr.shape
C = 10


def init_params(seed):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2.0 / D), size=(D, HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0 / HIDDEN), size=(HIDDEN, C))
    b2 = np.zeros(C)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(params, mask, Xb):
    W1 = params["W1"] * mask["W1"]
    W2 = params["W2"] * mask["W2"]
    z1 = Xb @ W1 + params["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ W2 + params["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)
    ez = np.exp(z2)
    p = ez / ez.sum(axis=1, keepdims=True)
    return z1, a1, p


def accuracy(params, mask, Xb, yb):
    _, _, p = forward(params, mask, Xb)
    return float((p.argmax(axis=1) == yb).mean())


def train(params0, mask, seed, record_curve=False):
    """Train a copy of params0 under mask. Returns (trained params, test acc, curve)."""
    params = {k: v.copy() for k, v in params0.items()}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    rng = np.random.default_rng(10_000 + seed)
    t = 0
    curve = []
    for epoch in range(EPOCHS):
        idx = rng.permutation(N)
        for s in range(0, N, BATCH):
            b = idx[s : s + BATCH]
            Xb, yb = Xtr[b], ytr[b]
            nb = len(b)
            z1, a1, p = forward(params, mask, Xb)
            # gradients (softmax CE)
            dz2 = p.copy()
            dz2[np.arange(nb), yb] -= 1.0
            dz2 /= nb
            gW2 = (a1.T @ dz2) * mask["W2"]
            gb2 = dz2.sum(axis=0)
            da1 = dz2 @ (params["W2"] * mask["W2"]).T
            dz1 = da1 * (z1 > 0)
            gW1 = (Xb.T @ dz1) * mask["W1"]
            gb1 = dz1.sum(axis=0)
            grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
            t += 1
            for k in params:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                mh = m[k] / (1 - 0.9**t)
                vh = v[k] / (1 - 0.999**t)
                params[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
        if record_curve:
            curve.append(accuracy(params, mask, Xte, yte))
    return params, accuracy(params, mask, Xte, yte), curve


def prune_layerwise(params, mask, target_sparsity):
    """New mask: within each weight matrix, keep the largest-|w| active weights
    so that each layer reaches target_sparsity."""
    new_mask = {k: v.copy() for k, v in mask.items()}
    for k in ("W1", "W2"):
        w = np.abs(params[k]) * mask[k]
        n = w.size
        n_keep = int(round(n * (1.0 - target_sparsity)))
        flat = w.flatten()
        # among active weights pick n_keep largest
        order = np.argsort(flat)[::-1]
        keep_idx = order[:n_keep]
        nm = np.zeros(n)
        nm[keep_idx] = 1.0
        new_mask[k] = nm.reshape(params[k].shape)
    return new_mask


def random_mask_like(mask_ref, seed):
    rng = np.random.default_rng(50_000 + seed)
    out = {}
    for k, v in mask_ref.items():
        if k in ("W1", "W2"):
            n_keep = int(v.sum())
            flat = np.zeros(v.size)
            idx = rng.choice(v.size, size=n_keep, replace=False)
            flat[idx] = 1.0
            out[k] = flat.reshape(v.shape)
        else:
            out[k] = v.copy()
    return out


def ones_mask():
    p = init_params(0)
    return {k: np.ones_like(v) for k, v in p.items()}


def main():
    results = {
        "levels": LEVELS,
        "seeds": SEEDS,
        "dense": {},
        "winning": {},
        "reinit": {},
        "randmask": {},
        "curves": {},  # curves at the curve_level for winning/reinit
    }
    curve_level = 0.95
    masks_seed0 = {}
    t0 = time.time()
    for seed in SEEDS:
        params0 = init_params(seed)
        mask = ones_mask()
        # dense training
        trained, acc_dense, _ = train(params0, mask, seed)
        results["dense"][str(seed)] = acc_dense
        print(f"seed {seed} dense acc {acc_dense:.4f}  ({time.time()-t0:.0f}s)")
        current_trained = trained
        for li, level in enumerate(LEVELS):
            mask = prune_layerwise(current_trained, mask, level)
            # winning ticket: rewind to init, train with mask
            rec = abs(level - curve_level) < 1e-9
            current_trained, acc_win, curve_w = train(
                params0, mask, seed, record_curve=rec
            )
            # reinit: same mask, fresh init
            reinit_params = init_params(seed + 1000 + li)
            _, acc_re, curve_r = train(reinit_params, mask, seed, record_curve=rec)
            # random mask + original init
            rmask = random_mask_like(mask, seed * 100 + li)
            _, acc_rm, _ = train(params0, rmask, seed)
            for name, acc in (
                ("winning", acc_win),
                ("reinit", acc_re),
                ("randmask", acc_rm),
            ):
                results[name].setdefault(str(level), {})[str(seed)] = acc
            if rec:
                results["curves"].setdefault("winning", {})[str(seed)] = curve_w
                results["curves"].setdefault("reinit", {})[str(seed)] = curve_r
            print(
                f"  seed {seed} sparsity {level:.2f}: win {acc_win:.4f} "
                f"reinit {acc_re:.4f} randmask {acc_rm:.4f} ({time.time()-t0:.0f}s)"
            )
            if seed == 0:
                masks_seed0[f"W1_{level}"] = mask["W1"]
                masks_seed0[f"W2_{level}"] = mask["W2"]
    with open(r"C:\tmp\lottery-ticket-numpy-experiment\results.json", "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(
        r"C:\tmp\lottery-ticket-numpy-experiment\masks_seed0.npz", **masks_seed0
    )
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
