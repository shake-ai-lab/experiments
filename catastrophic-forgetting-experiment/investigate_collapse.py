"""
Follow-up investigation: why does task-A accuracy collapse to EXACTLY 0.0000
so fast (within ~15 steps) under no rehearsal? And what does the collapsed
model actually predict for task-A digits?

Reuses the same setup as experiment.py.
"""
import json

import numpy as np
from experiment import (
    load_split, init_params, forward, backward, Adam, onehot, accuracy,
    train_full_batch, clone_params, RNG_SEED_MAIN, LR, PHASE1_EPOCHS,
    BATCH_SIZE, TASK_A, TASK_B,
)

(A_Xtr, A_ytr, A_Xte, A_yte), (B_Xtr, B_ytr, B_Xte, B_yte) = load_split(seed=RNG_SEED_MAIN)

params0 = init_params(seed=RNG_SEED_MAIN)
opt0 = Adam(params0, lr=LR)
params_after_a = train_full_batch(params0, opt0, A_Xtr, A_ytr, PHASE1_EPOCHS)
checkpoint = clone_params(params_after_a)

# ---- fine-grained trace of first 60 steps, no rehearsal ----
rng = np.random.default_rng(RNG_SEED_MAIN + 1)
params = clone_params(checkpoint)
opt = Adam(params, lr=LR)
n_b = len(B_Xtr)

print("step | A_acc | mean logit(classes0-4) | mean logit(classes5-9) | b2[0:5] mean | b2[5:10] mean")
for step in range(1, 61):
    idx = rng.integers(0, n_b, size=BATCH_SIZE)
    Xb, yb = B_Xtr[idx], B_ytr[idx]
    y1h = onehot(yb)
    probs, cache = forward(params, Xb)
    grads = backward(params, cache, y1h, probs)
    opt.step(params, grads)

    if step in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60):
        a_acc = accuracy(params, A_Xte, A_yte)
        _, (_, z1, h1) = forward(params, A_Xte)
        W2, b2 = params[2], params[3]
        logits = h1 @ W2 + b2
        mean_logit_A = logits[:, TASK_A].mean()
        mean_logit_B = logits[:, TASK_B].mean()
        b2_A = params[3][TASK_A].mean()
        b2_B = params[3][TASK_B].mean()
        print(f"{step:4d} | {a_acc:.4f} | {mean_logit_A:8.3f} | {mean_logit_B:8.3f} | {b2_A:8.3f} | {b2_B:8.3f}")

params_no_rehearsal_60 = clone_params(params)

# continue to step 3000 total for confusion matrix at the end (reuse same rng stream)
for step in range(61, 3001):
    idx = rng.integers(0, n_b, size=BATCH_SIZE)
    Xb, yb = B_Xtr[idx], B_ytr[idx]
    y1h = onehot(yb)
    probs, cache = forward(params, Xb)
    grads = backward(params, cache, y1h, probs)
    opt.step(params, grads)

final_a_acc = accuracy(params, A_Xte, A_yte)
print(f"\nfinal A acc (step 3000): {final_a_acc:.4f}")

# what does the collapsed model predict for task-A digits now?
probs, _ = forward(params, A_Xte)
pred = probs.argmax(axis=1)
print("\nprediction distribution for task-A test images (true labels 0-4), after collapse:")
vals, counts = np.unique(pred, return_counts=True)
for v, c in zip(vals, counts):
    print(f"  predicted class {v}: {c} / {len(pred)} ({c/len(pred):.3f})")

# confusion: for each true task-A digit, what's the most common predicted class?
print("\nper true-digit breakdown:")
for true_cls in TASK_A:
    m = (A_yte == true_cls)
    sub_pred = pred[m]
    vals, counts = np.unique(sub_pred, return_counts=True)
    top = vals[np.argmax(counts)]
    top_frac = counts.max() / m.sum()
    print(f"  true={true_cls} (n={m.sum():3d}): most predicted as {top} ({top_frac:.3f}), full dist={dict(zip(vals.tolist(), counts.tolist()))}")

# output bias b2 final values, all 10 classes
print(f"\nfinal b2 (output bias), all 10 classes: {np.round(params[3], 3)}")
print(f"checkpoint b2 (after phase1, before phase2): {np.round(checkpoint[3], 3)}")
