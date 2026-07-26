"""
Catastrophic forgetting experiment.

Architecture matches transfer-learning-life-lesson.md: hand-rolled numpy MLP,
64-dim input (8x8 digits) -> 64 hidden units (ReLU) -> softmax output, Adam
optimizer implemented from scratch.

Difference from the transfer-learning post: here the output layer is a SINGLE
shared 10-way softmax head (digits 0-9), because catastrophic forgetting is
about a network that has to serve both tasks through the same output space,
not two independently-scored 5-way heads.

Task A = digits 0-4 (learned first, until good accuracy)
Task B = digits 5-9 (learned second, continuing to train the SAME network)

Two conditions during Task B training:
  - no_rehearsal: mini-batches drawn only from Task B train data
  - rehearsal: each mini-batch is 90% Task B samples + 10% Task A train samples

We track Task A test accuracy (and Task B test accuracy) throughout Task B
training for both conditions.
"""

import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RNG_SEED_MAIN = 0
HIDDEN = 64
N_CLASSES = 10
LR = 0.01
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8

PHASE1_EPOCHS = 400  # full-batch, matches transfer-learning post's source training
PHASE2_STEPS = 3000  # mini-batch steps of Task-B-only training
BATCH_SIZE = 32
REHEARSAL_FRAC = 0.1  # 10% of each Task-B batch is Task-A rehearsal data
RECORD_EVERY = 5  # record accuracy every N steps (for trajectory + gif)

TASK_A = [0, 1, 2, 3, 4]
TASK_B = [5, 6, 7, 8, 9]


# ---------------------------------------------------------------- data ----
def load_split(seed=0):
    digits = load_digits()
    X_all = digits.data / 16.0
    y_all = digits.target

    def split_for(classes):
        mask = np.isin(y_all, classes)
        X, y = X_all[mask], y_all[mask]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        return X_tr, y_tr, X_te, y_te

    A_Xtr, A_ytr, A_Xte, A_yte = split_for(TASK_A)
    B_Xtr, B_ytr, B_Xte, B_yte = split_for(TASK_B)
    return (A_Xtr, A_ytr, A_Xte, A_yte), (B_Xtr, B_ytr, B_Xte, B_yte)


# ---------------------------------------------------------------- model ---
def init_params(seed):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2.0 / 64), size=(64, HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0 / HIDDEN), size=(HIDDEN, N_CLASSES))
    b2 = np.zeros(N_CLASSES)
    return [W1, b1, W2, b2]


def forward(params, X):
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1
    h1 = np.maximum(0, z1)
    z2 = h1 @ W2 + b2
    z2 = z2 - z2.max(axis=1, keepdims=True)
    ez2 = np.exp(z2)
    probs = ez2 / ez2.sum(axis=1, keepdims=True)
    return probs, (X, z1, h1)


def backward(params, cache, y_onehot, probs):
    W1, b1, W2, b2 = params
    X, z1, h1 = cache
    n = X.shape[0]

    dz2 = (probs - y_onehot) / n
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)

    dh1 = dz2 @ W2.T
    dz1 = dh1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return [dW1, db1, dW2, db2]


class Adam:
    def __init__(self, params, lr=LR):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            self.m[i] = ADAM_B1 * self.m[i] + (1 - ADAM_B1) * grads[i]
            self.v[i] = ADAM_B2 * self.v[i] + (1 - ADAM_B2) * (grads[i] ** 2)
            mhat = self.m[i] / (1 - ADAM_B1 ** self.t)
            vhat = self.v[i] / (1 - ADAM_B2 ** self.t)
            params[i] -= self.lr * mhat / (np.sqrt(vhat) + ADAM_EPS)


def onehot(y, n_classes=N_CLASSES):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1
    return out


def accuracy(params, X, y):
    probs, _ = forward(params, X)
    pred = probs.argmax(axis=1)
    return (pred == y).mean()


def train_full_batch(params, opt, X, y, epochs):
    y1h = onehot(y)
    for _ in range(epochs):
        probs, cache = forward(params, X)
        grads = backward(params, cache, y1h, probs)
        opt.step(params, grads)
    return params


def clone_params(params):
    return [p.copy() for p in params]


def train_phase2(params, opt, B_Xtr, B_ytr, A_Xtr, A_ytr, A_Xte, A_yte, B_Xte, B_yte,
                  steps, batch_size, rehearsal_frac, rng):
    """Continue training on Task B. If rehearsal_frac > 0, mix in Task-A train
    samples into every batch. Track Task A / Task B test accuracy over steps."""
    n_b = len(B_Xtr)
    n_a = len(A_Xtr)
    n_rehearsal = int(round(batch_size * rehearsal_frac))
    n_fresh_b = batch_size - n_rehearsal

    trace_steps = []
    trace_a_acc = []
    trace_b_acc = []

    for step in range(1, steps + 1):
        b_idx = rng.integers(0, n_b, size=n_fresh_b)
        Xb, yb = B_Xtr[b_idx], B_ytr[b_idx]

        if n_rehearsal > 0:
            a_idx = rng.integers(0, n_a, size=n_rehearsal)
            Xa, ya = A_Xtr[a_idx], A_ytr[a_idx]
            Xbatch = np.concatenate([Xb, Xa], axis=0)
            ybatch = np.concatenate([yb, ya], axis=0)
        else:
            Xbatch, ybatch = Xb, yb

        y1h = onehot(ybatch)
        probs, cache = forward(params, Xbatch)
        grads = backward(params, cache, y1h, probs)
        opt.step(params, grads)

        if step % RECORD_EVERY == 0 or step == 1:
            trace_steps.append(step)
            trace_a_acc.append(accuracy(params, A_Xte, A_yte))
            trace_b_acc.append(accuracy(params, B_Xte, B_yte))

    return params, trace_steps, trace_a_acc, trace_b_acc


def first_step_below(trace_steps, trace_acc, threshold):
    for s, a in zip(trace_steps, trace_acc):
        if a < threshold:
            return s
    return None


def main():
    (A_Xtr, A_ytr, A_Xte, A_yte), (B_Xtr, B_ytr, B_Xte, B_yte) = load_split(seed=RNG_SEED_MAIN)
    print(f"task A: train n={len(A_Xtr)}, test n={len(A_Xte)}")
    print(f"task B: train n={len(B_Xtr)}, test n={len(B_Xte)}")

    # ---- Phase 1: train on Task A only ----
    params0 = init_params(seed=RNG_SEED_MAIN)
    opt0 = Adam(params0, lr=LR)
    params_after_a = train_full_batch(params0, opt0, A_Xtr, A_ytr, PHASE1_EPOCHS)

    a_train_acc = accuracy(params_after_a, A_Xtr, A_ytr)
    a_test_acc = accuracy(params_after_a, A_Xte, A_yte)
    b_test_acc_before = accuracy(params_after_a, B_Xte, B_yte)
    print(f"after phase1 (task A only, {PHASE1_EPOCHS} epochs): "
          f"A train acc={a_train_acc:.4f} A test acc={a_test_acc:.4f} "
          f"B test acc(untrained)={b_test_acc_before:.4f}")

    params_checkpoint = clone_params(params_after_a)

    # ---- Phase 2a: continue training on Task B only, NO rehearsal ----
    rng_no = np.random.default_rng(RNG_SEED_MAIN + 1)
    params_no_rehearsal = clone_params(params_checkpoint)
    opt_no = Adam(params_no_rehearsal, lr=LR)
    params_no_rehearsal, steps_no, a_acc_no, b_acc_no = train_phase2(
        params_no_rehearsal, opt_no, B_Xtr, B_ytr, A_Xtr, A_ytr, A_Xte, A_yte, B_Xte, B_yte,
        steps=PHASE2_STEPS, batch_size=BATCH_SIZE, rehearsal_frac=0.0, rng=rng_no,
    )

    # ---- Phase 2b: continue training on Task B, WITH rehearsal ----
    rng_re = np.random.default_rng(RNG_SEED_MAIN + 1)  # same seed for batch draws -> fair comparison
    params_rehearsal = clone_params(params_checkpoint)
    opt_re = Adam(params_rehearsal, lr=LR)
    params_rehearsal, steps_re, a_acc_re, b_acc_re = train_phase2(
        params_rehearsal, opt_re, B_Xtr, B_ytr, A_Xtr, A_ytr, A_Xte, A_yte, B_Xte, B_yte,
        steps=PHASE2_STEPS, batch_size=BATCH_SIZE, rehearsal_frac=REHEARSAL_FRAC, rng=rng_re,
    )

    print("\n--- trajectory samples (no rehearsal) ---")
    for s, a, b in list(zip(steps_no, a_acc_no, b_acc_no))[::20]:
        print(f"step={s:5d}  A_acc={a:.4f}  B_acc={b:.4f}")

    print("\n--- trajectory samples (rehearsal 10%) ---")
    for s, a, b in list(zip(steps_re, a_acc_re, b_acc_re))[::20]:
        print(f"step={s:5d}  A_acc={a:.4f}  B_acc={b:.4f}")

    print("\n--- summary (single seed run) ---")
    print(f"A acc at phase1 end (start of phase2):        {a_test_acc:.4f}")
    print(f"A acc final, NO rehearsal:                     {a_acc_no[-1]:.4f}")
    print(f"A acc final, WITH rehearsal:                   {a_acc_re[-1]:.4f}")
    print(f"B acc final, NO rehearsal:                     {b_acc_no[-1]:.4f}")
    print(f"B acc final, WITH rehearsal:                   {b_acc_re[-1]:.4f}")

    for thr in [0.9, 0.5, 0.3]:
        s_no = first_step_below(steps_no, a_acc_no, thr)
        s_re = first_step_below(steps_re, a_acc_re, thr)
        print(f"first step A_acc < {thr}: no_rehearsal={s_no}  rehearsal={s_re}")

    min_a_no = min(a_acc_no)
    min_a_re = min(a_acc_re)
    print(f"min A acc over phase2: no_rehearsal={min_a_no:.4f}  rehearsal={min_a_re:.4f}")

    # ---- dead ReLU analysis on Task A data, before vs after phase2 (both conditions) ----
    def dead_frac_on(params, X):
        _, (_, z1, h1) = forward(params, X)
        active = (h1 > 0)
        active_rate_per_unit = active.mean(axis=0)
        dead = (active_rate_per_unit == 0).mean()
        return dead, active_rate_per_unit.mean()

    dead_before, act_before = dead_frac_on(params_checkpoint, A_Xte)
    dead_no, act_no = dead_frac_on(params_no_rehearsal, A_Xte)
    dead_re, act_re = dead_frac_on(params_rehearsal, A_Xte)
    print(f"\ndead units on task A data: before phase2={dead_before:.4f} (active_rate={act_before:.4f}) "
          f"after no_rehearsal={dead_no:.4f} (active_rate={act_no:.4f}) "
          f"after rehearsal={dead_re:.4f} (active_rate={act_re:.4f})")

    # weight drift: L2 norm of change in W1 (hidden layer) from checkpoint
    w1_before = params_checkpoint[0]
    w1_no = params_no_rehearsal[0]
    w1_re = params_rehearsal[0]
    drift_no = np.linalg.norm(w1_no - w1_before)
    drift_re = np.linalg.norm(w1_re - w1_before)
    print(f"W1 drift (L2 norm): no_rehearsal={drift_no:.4f}  rehearsal={drift_re:.4f}")

    # ---- save trajectory data for plotting ----
    np.savez(
        "trajectory.npz",
        steps_no=np.array(steps_no), a_acc_no=np.array(a_acc_no), b_acc_no=np.array(b_acc_no),
        steps_re=np.array(steps_re), a_acc_re=np.array(a_acc_re), b_acc_re=np.array(b_acc_re),
        a_test_acc_start=a_test_acc, b_test_acc_before=b_test_acc_before,
    )

    summary = {
        "n_A_train": len(A_Xtr), "n_A_test": len(A_Xte),
        "n_B_train": len(B_Xtr), "n_B_test": len(B_Xte),
        "phase1_epochs": PHASE1_EPOCHS,
        "A_train_acc_after_phase1": a_train_acc,
        "A_test_acc_after_phase1": a_test_acc,
        "B_test_acc_before_phase2": b_test_acc_before,
        "A_test_acc_final_no_rehearsal": a_acc_no[-1],
        "A_test_acc_final_rehearsal": a_acc_re[-1],
        "B_test_acc_final_no_rehearsal": b_acc_no[-1],
        "B_test_acc_final_rehearsal": b_acc_re[-1],
        "A_test_acc_min_no_rehearsal": min_a_no,
        "A_test_acc_min_rehearsal": min_a_re,
        "dead_frac_before": dead_before,
        "dead_frac_no_rehearsal": dead_no,
        "dead_frac_rehearsal": dead_re,
        "w1_drift_no_rehearsal": drift_no,
        "w1_drift_rehearsal": drift_re,
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved trajectory.npz and summary.json")


if __name__ == "__main__":
    main()
