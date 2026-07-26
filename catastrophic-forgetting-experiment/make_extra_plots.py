"""
Extra figures added after a critical re-read of the published post:

  0_digits.png        - what the 8x8 load_digits images actually look like
                        (Task A digits 0-4 / Task B digits 5-9), so a first-time
                        reader knows the data before the experiment starts.
  4_morph_examples.png - concrete before/after examples of the "morphing":
                        the SAME task-A test images, predicted correctly by the
                        phase-1 checkpoint, and predicted as a task-B digit by
                        the collapsed (no-rehearsal, step 3000) model.

Reruns phase 1 + phase 2 (no rehearsal) with the exact same seeds / rng streams
as experiment.py / investigate_collapse.py, so the collapsed model and its
prediction distribution are bit-for-bit identical to the published numbers
(asserted below against the hardcoded distribution in make_plots.py).
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import (
    load_split, init_params, forward, Adam, onehot, accuracy,
    train_full_batch, clone_params, backward,
    RNG_SEED_MAIN, LR, PHASE1_EPOCHS, BATCH_SIZE, TASK_A, TASK_B,
)

plt.rcParams["font.family"] = "Meiryo"

(A_Xtr, A_ytr, A_Xte, A_yte), (B_Xtr, B_ytr, B_Xte, B_yte) = load_split(seed=RNG_SEED_MAIN)

# ------------------------------------------------- figure 0: the raw data ----
fig, axes = plt.subplots(2, 5, figsize=(7.5, 3.9), dpi=110)
for row, (X, y, classes, label) in enumerate([
    (A_Xtr, A_ytr, TASK_A, "Task A"),
    (B_Xtr, B_ytr, TASK_B, "Task B"),
]):
    for col, cls in enumerate(classes):
        ax = axes[row, col]
        img = X[np.where(y == cls)[0][0]].reshape(8, 8)
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"「{cls}」", fontsize=14)
        if col == 0:
            ax.set_ylabel(label, fontsize=13)
fig.suptitle("load_digitsの中身: 8×8=64ピクセルの手書き数字", fontsize=14)
fig.tight_layout()
fig.savefig("0_digits.png")
plt.close(fig)
print("saved 0_digits.png")

# ------------------------------------- rebuild checkpoint + collapsed model --
params0 = init_params(seed=RNG_SEED_MAIN)
opt0 = Adam(params0, lr=LR)
params_after_a = train_full_batch(params0, opt0, A_Xtr, A_ytr, PHASE1_EPOCHS)
checkpoint = clone_params(params_after_a)
print(f"checkpoint A test acc = {accuracy(checkpoint, A_Xte, A_yte):.4f} (published: 0.9834)")

# phase 2, no rehearsal, 3000 steps — identical rng stream to experiment.py
rng = np.random.default_rng(RNG_SEED_MAIN + 1)
params = clone_params(checkpoint)
opt = Adam(params, lr=LR)
n_b = len(B_Xtr)
for step in range(1, 3001):
    idx = rng.integers(0, n_b, size=BATCH_SIZE)
    Xb, yb = B_Xtr[idx], B_ytr[idx]
    y1h = onehot(yb)
    probs, cache = forward(params, Xb)
    grads = backward(params, cache, y1h, probs)
    opt.step(params, grads)

final_a_acc = accuracy(params, A_Xte, A_yte)
print(f"collapsed A test acc = {final_a_acc:.4f} (published: 0.0000)")
assert final_a_acc == 0.0

probs_ckpt, _ = forward(checkpoint, A_Xte)
pred_ckpt = probs_ckpt.argmax(axis=1)
probs_coll, _ = forward(params, A_Xte)
pred_coll = probs_coll.argmax(axis=1)

# sanity check against the hardcoded distribution in make_plots.py
published_dist = {
    0: {5: 5, 6: 30, 7: 1},
    1: {6: 4, 8: 28, 9: 5},
    2: {5: 1, 6: 1, 7: 3, 8: 19, 9: 11},
    3: {5: 2, 7: 3, 8: 7, 9: 25},
    4: {5: 2, 6: 24, 7: 8, 8: 1, 9: 1},
}
for true_cls in TASK_A:
    m = A_yte == true_cls
    vals, counts = np.unique(pred_coll[m], return_counts=True)
    got = dict(zip(vals.tolist(), counts.tolist()))
    assert got == published_dist[true_cls], (true_cls, got)
print("prediction distribution matches the published numbers exactly")

# ------------------------------- figure 4: concrete morphing before/after ----
# for each true digit, pick a test image that the checkpoint classified
# correctly and the collapsed model sends to that digit's modal target class
modal_target = {0: 6, 1: 8, 2: 8, 3: 9, 4: 6}
fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.9), dpi=110)
for col, true_cls in enumerate(TASK_A):
    m = (A_yte == true_cls) & (pred_ckpt == true_cls) & (pred_coll == modal_target[true_cls])
    idx = np.where(m)[0][0]
    ax = axes[col]
    ax.imshow(A_Xte[idx].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"真の数字: {true_cls}\n崩壊前の予測: {true_cls}\n崩壊後の予測: {modal_target[true_cls]}",
                 fontsize=12)
fig.suptitle("同じ画像なのに、崩壊後は別の数字として読まれる", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("4_morph_examples.png", bbox_inches="tight")
plt.close(fig)
print("saved 4_morph_examples.png")
