import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

OUT = r"c:\Users\kouta\project\forsubwork\src\content\blog\images\lottery-ticket-numpy"

with open(r"C:\tmp\lottery-ticket-numpy-experiment\results.json") as f:
    r = json.load(f)

levels = r["levels"]
seeds = [str(s) for s in r["seeds"]]
COLORS = {"winning": "#1f77b4", "reinit": "#ff7f0e", "randmask": "#2ca02c"}
LABELS = {
    "winning": "当たりくじ(マスク+元の初期値)",
    "reinit": "同じマスク+ランダム再初期化",
    "randmask": "ランダムなマスク+元の初期値",
}


def stat(name, level):
    a = np.array([r[name][str(level)][s] for s in seeds])
    return a.mean(), a.std()


dense_mean = np.mean([r["dense"][s] for s in seeds])

# ---------------- fig 1: accuracy vs sparsity ----------------
fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(levels))
ax.axhline(dense_mean, color="#888888", ls="--", lw=1.5, zorder=1)
ax.annotate(
    f"枝刈りなし(密) {dense_mean*100:.1f}%",
    (0.1, dense_mean + 0.012),
    color="#666666",
    fontsize=10,
)
markers = {"winning": "o", "reinit": "s", "randmask": "^"}
for name in ("winning", "reinit", "randmask"):
    means = np.array([stat(name, lv)[0] for lv in levels])
    stds = np.array([stat(name, lv)[1] for lv in levels])
    ax.plot(
        x, means, marker=markers[name], color=COLORS[name], lw=2, ms=7,
        label=LABELS[name], zorder=3,
    )
    ax.fill_between(x, means - stds, means + stds, color=COLORS[name], alpha=0.15,
                    zorder=2, lw=0)
ax.set_xticks(x)
ax.set_xticklabels([f"{int(lv*100)}%" for lv in levels])
ax.set_xlabel("スパース率(枝刈りで消した重みの割合)")
ax.set_ylabel("テスト精度")
ax.set_ylim(0.05, 1.02)
ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
ax.legend(loc="lower left", fontsize=10, frameon=False)
ax.set_title("スパース率ごとのテスト精度(5シード平均±標準偏差)", fontsize=12)
ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
fig.tight_layout()
fig.savefig(f"{OUT}\\1.png", dpi=110)
plt.close(fig)

# ---------------- fig 2: heatmap ----------------
order = ["winning", "reinit", "randmask"]
M = np.array([[stat(name, lv)[0] for lv in levels] for name in order])
fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.imshow(M, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(
            j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=11,
            color="white" if v > 0.6 else "#1a355c",
        )
ax.set_xticks(np.arange(len(levels)))
ax.set_xticklabels([f"{int(lv*100)}%" for lv in levels])
ax.set_yticks(np.arange(3))
ax.set_yticklabels(["当たりくじ\n(マスク+元の初期値)", "同じマスク\n+ランダム再初期化",
                    "ランダムなマスク\n+元の初期値"], fontsize=10)
ax.set_xlabel("スパース率")
ax.set_title("テスト精度[%](5シード平均): スパース率 × 戦略", fontsize=12)
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.set_ylabel("精度", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}\\2.png", dpi=110)
plt.close(fig)

# ---------------- fig 3: learning curves at 95% ----------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ep = np.arange(1, 61)
for name in ("winning", "reinit"):
    c = np.array([r["curves"][name][s] for s in seeds])
    ax.plot(ep, c.mean(axis=0), color=COLORS[name], lw=2, label=LABELS[name])
    ax.fill_between(ep, c.mean(axis=0) - c.std(axis=0), c.mean(axis=0) + c.std(axis=0),
                    color=COLORS[name], alpha=0.15, lw=0)
ax.set_xlabel("エポック")
ax.set_ylabel("テスト精度")
ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
ax.set_title("スパース率95%での学習曲線(5シード平均±標準偏差)", fontsize=12)
ax.legend(loc="lower right", fontsize=10, frameon=False)
ax.grid(axis="y", color="#dddddd", lw=0.6)
fig.tight_layout()
fig.savefig(f"{OUT}\\3.png", dpi=110)
plt.close(fig)

# ---------------- GIF: mask evolution ----------------
masks = np.load(r"C:\tmp\lottery-ticket-numpy-experiment\masks_seed0.npz")
frames = [("0%(密)", np.ones((64, 128)))] + [
    (f"{int(lv*100)}%", masks[f"W1_{lv}"]) for lv in levels
]
frames += [frames[-1], frames[-1]]  # hold last frame longer
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=80,
                         gridspec_kw={"width_ratios": [2.2, 1]})


def draw(k):
    label, m = frames[k]
    for a in axes:
        a.clear()
    axes[0].imshow(m, cmap="gray_r", aspect="auto", interpolation="nearest",
                   vmin=0, vmax=1)
    axes[0].set_title(f"第1層の重み行列のマスク(黒=生存)  スパース率 {label}",
                      fontsize=9)
    axes[0].set_xlabel("隠れユニット(128)", fontsize=8)
    axes[0].set_ylabel("入力ピクセル(64)", fontsize=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    per_pixel = m.sum(axis=1).reshape(8, 8)
    axes[1].imshow(per_pixel, cmap="Blues", vmin=0, vmax=per_pixel.max() or 1)
    axes[1].set_title("入力ピクセルごとの\n生き残った結線の数", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.tight_layout()


anim = FuncAnimation(fig, draw, frames=len(frames))
anim.save(f"{OUT}\\mask.gif", writer=PillowWriter(fps=1))
plt.close(fig)

# ---------------- extra analysis for the article ----------------
m99 = masks["W1_0.99"]
per_pixel99 = m99.sum(axis=1).reshape(8, 8)
print("surviving connections per input pixel at 99% (8x8):")
print(per_pixel99.astype(int))
print("pixels with zero surviving connections:", int((per_pixel99 == 0).sum()))
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
var = X.var(axis=0).reshape(8, 8)
print("pixel variance (rounded):")
print(var.round(1))
alive = per_pixel99.flatten() > 0
print("mean variance of surviving pixels:", X.var(axis=0)[alive].mean().round(2))
print("mean variance of dead pixels:", X.var(axis=0)[~alive].mean().round(2))
corr = np.corrcoef(X.var(axis=0), m99.sum(axis=1))[0, 1]
print("corr(pixel variance, surviving connections at 99%):", round(corr, 3))
