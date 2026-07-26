"""本実験: lr=0.018(step0/0bで見つけた「不安定になるかならないか五分五分」のLR)で、
warmupなし・linear warmup・cosine warmupの3条件を、それぞれ10シードで訓練する。
モデルはsklearn digitsに対する64->128->128->128->128->10のReLU MLP(隠れ層4層)、
最適化はAdam(手書き)。warmup_frac=0.1(全1500ステップ中150ステップ)。
発散/崩壊の判定はtest_acc<0.5(chance levelは0.1なので、これでも十分「学習が壊れた」印)。
"""
import json

import numpy as np

from step0_lr_search import MLP, load_data, softmax_ce

TARGET_LR = 0.018
STEPS = 1500
WARMUP_FRAC = 0.1
BATCH_SIZE = 32
N_SEEDS = 20
FAIL_THRESH = 0.5  # test_acc未満なら「崩壊」


def train_run(lr, seed, warmup_kind, steps=STEPS, batch_size=BATCH_SIZE, warmup_frac=WARMUP_FRAC):
    Xtr, ytr, Xte, yte = load_data(0)
    r = np.random.default_rng(seed)
    m = MLP(seed)
    ps = m.params()
    mom = [np.zeros_like(p) for p in ps]
    vel = [np.zeros_like(p) for p in ps]
    B1, B2, EPS = 0.9, 0.999, 1e-8
    n = len(Xtr)
    n_layers = len(m.W)
    warmup_steps = max(int(steps * warmup_frac), 1)
    losses = np.empty(steps)
    for t in range(1, steps + 1):
        if warmup_kind == "none" or t > warmup_steps:
            cur_lr = lr
        elif warmup_kind == "linear":
            cur_lr = lr * t / warmup_steps
        elif warmup_kind == "cosine":
            frac = t / warmup_steps
            cur_lr = lr * 0.5 * (1 - np.cos(np.pi * frac))
        else:
            raise ValueError(warmup_kind)

        idx = r.integers(0, n, batch_size)
        Xb, yb = Xtr[idx], ytr[idx]
        logits, cache = m.forward(Xb)
        loss, dl, _ = softmax_ce(logits, yb)
        if not np.isfinite(loss):
            losses[t - 1 :] = np.nan
            break
        losses[t - 1] = loss

        grads = m.backward(cache, dl)
        gW, gb = grads[:n_layers], grads[n_layers:]
        for i, g in enumerate(gW):
            mom[i] = B1 * mom[i] + (1 - B1) * g
            vel[i] = B2 * vel[i] + (1 - B2) * g**2
            m.W[i] -= cur_lr * (mom[i] / (1 - B1**t)) / (np.sqrt(vel[i] / (1 - B2**t)) + EPS)
        for j, g in enumerate(gb):
            i = n_layers + j
            mom[i] = B1 * mom[i] + (1 - B1) * g
            vel[i] = B2 * vel[i] + (1 - B2) * g**2
            m.b[j] -= cur_lr * (mom[i] / (1 - B1**t)) / (np.sqrt(vel[i] / (1 - B2**t)) + EPS)

    logits, _ = m.forward(Xte)
    _, _, test_acc = softmax_ce(logits, yte)
    return losses, float(test_acc)


if __name__ == "__main__":
    conditions = ["none", "linear", "cosine"]
    all_losses = {}  # cond -> (N_SEEDS, STEPS)
    all_accs = {}  # cond -> list
    summary = {}

    for cond in conditions:
        loss_mat = np.empty((N_SEEDS, STEPS))
        accs = []
        for seed in range(N_SEEDS):
            losses, test_acc = train_run(TARGET_LR, seed, cond)
            loss_mat[seed] = losses
            accs.append(test_acc)
        all_losses[cond] = loss_mat
        all_accs[cond] = accs
        fails = sum(1 for a in accs if a < FAIL_THRESH)
        peaks = np.nanmax(loss_mat, axis=1)
        summary[cond] = {
            "fails": fails,
            "n": N_SEEDS,
            "accs": accs,
            "peaks": peaks.tolist(),
            "mean_acc": float(np.mean(accs)),
            "median_acc": float(np.median(accs)),
        }
        print(f"=== {cond} ===")
        print(f"  fails(test_acc<{FAIL_THRESH}): {fails}/{N_SEEDS}")
        print(f"  test_acc per seed: {[round(a, 3) for a in accs]}")
        print(f"  peak loss per seed: {[round(p, 2) for p in peaks]}")
        print(f"  mean_acc={np.mean(accs):.3f} median_acc={np.median(accs):.3f}")

    np.savez(
        "results.npz",
        **{f"loss_{c}": all_losses[c] for c in conditions},
    )
    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved results.npz, summary.json")
