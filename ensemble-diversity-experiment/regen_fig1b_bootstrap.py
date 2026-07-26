"""fig1b: ブートストラップ再抽出の before/after を1枚で見せる図。

左: trial 0 の訓練データ700点(クラス色つき) = two moons の「正体」も兼ねる
右: B条件のメンバー1人目が実際に引いたブートストラップ標本
    (resample_seed = trial*10000 + 200 + i = 200, step1_train.py と同じ式)
    点の大きさ=引かれた回数、灰色x=1回も引かれなかった点

キャッシュ済みの rep_trial.npz (trial 0 の X_tr/y_tr) を再利用するので、
学習をやり直す必要はなく、公開済みの数字は一切変わらない。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"

OUTDIR = "C:/tmp/ensemble-diversity-experiment"

rep = np.load(f"{OUTDIR}/rep_trial.npz")
X_tr, y_tr = rep["X_tr"], rep["y_tr"]
n = len(X_tr)
assert n == 700

# exactly the resample that member 1 of condition B saw in trial 0 (step1_train.py)
resample_seed = 0 * 10_000 + 200 + 0
r = np.random.default_rng(resample_seed)
idx = r.integers(0, n, size=n)
counts = np.bincount(idx, minlength=n)
n_unique = (counts > 0).sum()
print(f"resample_seed={resample_seed}: unique={n_unique}/{n} ({n_unique/n:.1%}), "
      f"max draws of one point={counts.max()}")

C0 = "#2e6ee4"  # class 0 (blue, matches condition-A blue family used in this post)
C1 = "#e4572e"  # class 1 (orange-red, matches condition-B family)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharex=True, sharey=True)

# ---- left: original training data ----
ax = axes[0]
for cls, col, name in [(0, C0, "class 0"), (1, C1, "class 1")]:
    m = y_tr == cls
    ax.scatter(X_tr[m, 0], X_tr[m, 1], s=12, color=col, alpha=0.75,
               edgecolor="none", label=name)
ax.set_title(f"元の訓練データ {n}点\n(条件Aの全メンバーはこれをそのまま使う)", fontsize=11)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

# ---- right: one member's bootstrap resample ----
ax = axes[1]
missed = counts == 0
# points never drawn: gray x
ax.scatter(X_tr[missed, 0], X_tr[missed, 1], s=14, color="#b0b0b0", marker="x",
           linewidth=0.9, alpha=0.8, label=f"1回も引かれず ({int(missed.sum())}点)")
for cls, col in [(0, C0), (1, C1)]:
    m = (y_tr == cls) & (~missed)
    ax.scatter(X_tr[m, 0], X_tr[m, 1], s=10 + 26 * (counts[m] - 1), color=col,
               alpha=0.75, edgecolor="white", linewidth=0.4)
ax.set_title(f"Bのメンバー1人目が引いたブートストラップ標本\n"
             f"(点の大きさ=引かれた回数、ユニーク{n_unique}点={n_unique/n:.0%})", fontsize=11)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("ブートストラップ再抽出のbefore/after: 同じ700点から、メンバーごとに少し違う「世界」が切り出される",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUTDIR}/fig1b_bootstrap.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig1b_bootstrap.png")
