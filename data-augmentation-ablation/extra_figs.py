import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
plt.rcParams["font.family"] = "Meiryo"

N_TRAIN = 15
NOISE = 0.25
SEED_ILLUSTRATE = 3

COLOR0 = "#4C72B0"
COLOR1 = "#C44E52"


def make_train(seed):
    return make_moons(n_samples=N_TRAIN, noise=NOISE, random_state=seed)


def augment(X, y, k, jitter_sigma, rot_max_deg, rng):
    Xs, ys = [X.copy()], [y.copy()]
    if k > 0:
        center = X.mean(axis=0)
        for _ in range(k):
            angles = rng.uniform(-rot_max_deg, rot_max_deg, size=len(X)) * np.pi / 180.0
            cos_a, sin_a = np.cos(angles), np.sin(angles)
            Xc = X - center
            Xrot = np.stack([
                Xc[:, 0] * cos_a - Xc[:, 1] * sin_a,
                Xc[:, 0] * sin_a + Xc[:, 1] * cos_a,
            ], axis=1) + center
            jitter = rng.normal(0, jitter_sigma, size=X.shape)
            Xs.append(Xrot + jitter)
            ys.append(y.copy())
    return np.vstack(Xs), np.concatenate(ys)


# =================================================================
# Fig 0: what does "two moons" even look like, and how sparse is
# our actual 15-point training sample relative to the true shape?
# =================================================================
X_dense, y_dense = make_moons(n_samples=1000, noise=0.05, random_state=0)
Xtr, ytr = make_train(SEED_ILLUSTRATE)

# shared view window for both panels, so aspect='equal' renders both
# plot boxes at the same size instead of auto-fitting each to its own
# (very different) data extent
all_x = np.concatenate([X_dense[:, 0], Xtr[:, 0]])
all_y = np.concatenate([X_dense[:, 1], Xtr[:, 1]])
pad_x = (all_x.max() - all_x.min()) * 0.08
pad_y = (all_y.max() - all_y.min()) * 0.08
XLIM = (all_x.min() - pad_x, all_x.max() + pad_x)
YLIM = (all_y.min() - pad_y, all_y.max() + pad_y)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
ax.scatter(X_dense[y_dense == 0, 0], X_dense[y_dense == 0, 1], c=COLOR0, s=10, alpha=0.6, label="class 0")
ax.scatter(X_dense[y_dense == 1, 0], X_dense[y_dense == 1, 1], c=COLOR1, s=10, alpha=0.6, label="class 1")
ax.set_title("two moonsの「正体」\n(密に1000点サンプリング, noise小)", fontsize=12.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
ax.set_aspect("equal")

ax = axes[1]
ax.scatter(X_dense[:, 0], X_dense[:, 1], c="#cccccc", s=8, alpha=0.4, zorder=1, label="真の形(参考・薄グレー)")
ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], c=COLOR0, edgecolors="k", s=90, zorder=3, label="訓練データ class 0")
ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], c=COLOR1, edgecolors="k", s=90, zorder=3, label="訓練データ class 1")
ax.set_title(f"今回の訓練データ\n(たった{N_TRAIN}点、noise={NOISE})", fontsize=12.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1, frameon=False)
ax.set_aspect("equal")

fig.suptitle("「三日月の形」自体はこう見える、その形をたった15点だけで学ばせている", fontsize=13, y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("0_intro_shape.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("saved 0_intro_shape.png")

# =================================================================
# Fig 0b: what augmentation literally does to the 15 points
# (before vs after, at the peak setting m=0.4, k=10)
# =================================================================
K_SHOW = 10
SIGMA_SHOW, ROT_SHOW = 0.12, 8.0  # m=0.4 setting (the peak found later in the post)
rng = np.random.default_rng(SEED_ILLUSTRATE + 999)
Xaug, yaug = augment(Xtr, ytr, K_SHOW, SIGMA_SHOW, ROT_SHOW, rng)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], c=COLOR0, edgecolors="k", s=90, zorder=3)
ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], c=COLOR1, edgecolors="k", s=90, zorder=3)
ax.set_title(f"拡張前: 元の訓練データ {N_TRAIN}点", fontsize=12.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-1.8, 2.8)
ax.set_ylim(-1.3, 1.8)
ax.set_aspect("equal")

ax = axes[1]
ax.scatter(Xaug[N_TRAIN:][yaug[N_TRAIN:] == 0, 0], Xaug[N_TRAIN:][yaug[N_TRAIN:] == 0, 1],
           c=COLOR0, s=16, alpha=0.45, zorder=2, label="合成コピー(ジッター+回転)")
ax.scatter(Xaug[N_TRAIN:][yaug[N_TRAIN:] == 1, 0], Xaug[N_TRAIN:][yaug[N_TRAIN:] == 1, 1],
           c=COLOR1, s=16, alpha=0.45, zorder=2)
ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], c=COLOR0, edgecolors="k", s=90, zorder=3, label="元の15点")
ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], c=COLOR1, edgecolors="k", s=90, zorder=3)
ax.set_title(f"拡張後: 元15点 + 合成{K_SHOW}倍 = {N_TRAIN * (K_SHOW + 1)}点\n(強さm=0.4: ジッター標準偏差{SIGMA_SHOW}, 回転±{ROT_SHOW:.0f}度)", fontsize=12.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-1.8, 2.8)
ax.set_ylim(-1.3, 1.8)
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
ax.set_aspect("equal")

fig.suptitle("データ拡張の中身: 元の点のまわりに「少しずらした偽の仲間」を増やす", fontsize=13, y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("0b_augment_example.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("saved 0b_augment_example.png")
