"""
Generate the 3 static figures + 1 GIF for the data-augmentation-ablation post.
Reuses results.json / final_confirm_results.json produced by experiment.py /
final_confirm.py, and re-fits a handful of models for the qualitative
decision-boundary figure and the GIF (fast: n<=15*41 points, lbfgs).
"""
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
plt.rcParams["font.family"] = "Meiryo"

N_TRAIN = 15
N_TEST = 2000
NOISE = 0.25
HIDDEN = (30, 30)
MAX_ITER = 1000

BEST_ALPHA_REG = 0.1873817422860383
BEST_K = 40
BEST_SIGMA = 0.12
BEST_ROT = 8.0
BEST_ALPHA_COMBO = 1.5199110829529332

with open("results.json") as f:
    R = json.load(f)
with open("final_confirm_results.json") as f:
    RC = json.load(f)


def make_train(seed):
    return make_moons(n_samples=N_TRAIN, noise=NOISE, random_state=seed)


def make_test(seed=999):
    return make_moons(n_samples=N_TEST, noise=NOISE, random_state=seed)


def fit(Xtr, ytr, alpha, seed):
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, alpha=alpha, max_iter=MAX_ITER,
                         solver="lbfgs", random_state=seed)
    clf.fit(Xtr, ytr)
    return clf


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


Xte, yte = make_test()

XX, YY = np.meshgrid(np.linspace(-1.8, 2.8, 220), np.linspace(-1.3, 1.8, 220))
GRID = np.c_[XX.ravel(), YY.ravel()]

COLOR_BASE = "#8a8a8a"
COLOR_REG = "#4C72B0"
COLOR_AUG = "#DD8452"
COLOR_COMBO = "#55A868"

# =================================================================
# Figure 1: qualitative decision boundaries, single representative
# training draw, 4 panels (baseline / reg-only / aug-only / combo)
# =================================================================
SEED_ILLUSTRATE = 3  # baseline test_acc was notably bad at this seed (see diag.py)
Xtr, ytr = make_train(SEED_ILLUSTRATE)

clf_base = fit(Xtr, ytr, alpha=1e-8, seed=SEED_ILLUSTRATE)
clf_reg = fit(Xtr, ytr, alpha=BEST_ALPHA_REG, seed=SEED_ILLUSTRATE)

rng_a = np.random.default_rng(SEED_ILLUSTRATE + 20000)
Xaug_a, yaug_a = augment(Xtr, ytr, BEST_K, BEST_SIGMA, BEST_ROT, rng_a)
clf_aug = fit(Xaug_a, yaug_a, alpha=1e-8, seed=SEED_ILLUSTRATE)

rng_c = np.random.default_rng(SEED_ILLUSTRATE + 30000)
Xaug_c, yaug_c = augment(Xtr, ytr, BEST_K, BEST_SIGMA, BEST_ROT, rng_c)
clf_combo = fit(Xaug_c, yaug_c, alpha=BEST_ALPHA_COMBO, seed=SEED_ILLUSTRATE)

panels = [
    ("(a) ベースライン\n(拡張なし・正則化なし)", clf_base, COLOR_BASE),
    ("(b) 正則化のみ", clf_reg, COLOR_REG),
    ("(c) データ拡張のみ", clf_aug, COLOR_AUG),
    ("(d) 両方", clf_combo, COLOR_COMBO),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
for ax, (title, clf, color) in zip(axes, panels):
    Z = clf.predict_proba(GRID)[:, 1].reshape(XX.shape)
    ax.contourf(XX, YY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.55)
    ax.contour(XX, YY, Z, levels=[0.5], colors=[color], linewidths=2.5)
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], c="#4C72B0", edgecolors="k", s=45, zorder=3)
    ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], c="#C44E52", edgecolors="k", s=45, zorder=3)
    te_acc = accuracy_score(yte, clf.predict(Xte))
    ax.set_title(f"{title}\ntest_acc={te_acc:.3f}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"15点だけの訓練データ(乱数seed={SEED_ILLUSTRATE})に対する決定境界の違い", fontsize=12)
fig.tight_layout()
fig.savefig("1.png", dpi=140)
plt.close(fig)
print("saved 1.png")

# =================================================================
# Figure 2: 2x2 grid — test acc & gap vs magnitude, and vs multiplier k
# =================================================================
mag = np.array(R["magnitude_sweep"])  # m, sigma, rot, train, test, gap
ksw = np.array(R["k_sweep"])          # k, train, test, gap

base_test = R["baseline"]["test_acc"]
base_gap = R["baseline"]["gap"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.plot(mag[:, 0], mag[:, 4], "o-", color=COLOR_AUG)
ax.axhline(base_test, color=COLOR_BASE, linestyle="--", label="ベースライン(拡張なし)")
ax.set_xlabel("拡張の強さ m (ジッター標準偏差=0.3m, 回転=20m度)")
ax.set_ylabel("テスト精度")
ax.set_title("拡張の強さ vs テスト精度 (倍率k=10固定)")
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.plot(mag[:, 0], mag[:, 5], "o-", color=COLOR_AUG)
ax.axhline(base_gap, color=COLOR_BASE, linestyle="--", label="ベースライン(拡張なし)")
ax.set_xlabel("拡張の強さ m")
ax.set_ylabel("train-testギャップ")
ax.set_title("拡張の強さ vs 過学習ギャップ (倍率k=10固定)")
ax.legend(fontsize=9)

ax = axes[1, 0]
ax.plot(ksw[:, 0], ksw[:, 2], "s-", color=COLOR_COMBO)
ax.axhline(base_test, color=COLOR_BASE, linestyle="--", label="ベースライン(拡張なし)")
ax.set_xscale("symlog")
ax.set_xlabel("1点あたりの合成コピー数 k (強さ m=0.4固定)")
ax.set_ylabel("テスト精度")
ax.set_title("拡張倍率 vs テスト精度")
ax.legend(fontsize=9)

ax = axes[1, 1]
ax.plot(ksw[:, 0], ksw[:, 3], "s-", color=COLOR_COMBO)
ax.axhline(base_gap, color=COLOR_BASE, linestyle="--", label="ベースライン(拡張なし)")
ax.set_xscale("symlog")
ax.set_xlabel("1点あたりの合成コピー数 k")
ax.set_ylabel("train-testギャップ")
ax.set_title("拡張倍率 vs 過学習ギャップ")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("2.png", dpi=140)
plt.close(fig)
print("saved 2.png")

# =================================================================
# Figure 3: final bar chart, 4 conditions, test acc & gap (60-trial mean+-std)
# =================================================================
names = ["baseline", "reg_only", "aug_only", "combo"]
labels = ["拡張なし\n正則化なし", "正則化のみ", "データ拡張\nのみ", "両方"]
colors = [COLOR_BASE, COLOR_REG, COLOR_AUG, COLOR_COMBO]

test_means = [RC[n]["test_mean"] for n in names]
test_stds = [RC[n]["test_std"] for n in names]
gap_means = [RC[n]["gap_mean"] for n in names]
gap_stds = [RC[n]["gap_std"] for n in names]
N = 60
test_sems = [s / np.sqrt(N) for s in test_stds]
gap_sems = [s / np.sqrt(N) for s in gap_stds]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

ax = axes[0]
bars = ax.bar(labels, test_means, yerr=test_sems, capsize=5, color=colors)
ax.set_ylabel("テスト精度")
ax.set_title("条件別テスト精度 (60試行平均±SEM)")
ax.set_ylim(0.75, 0.95)
# place value labels above the error bar cap so the bar doesn't strike through the digits
for b, v, s in zip(bars, test_means, test_sems):
    ax.text(b.get_x() + b.get_width() / 2, v + s + 0.004, f"{v:.3f}", ha="center", fontsize=10)

ax = axes[1]
bars = ax.bar(labels, gap_means, yerr=gap_sems, capsize=5, color=colors)
ax.set_ylabel("train-testギャップ")
ax.set_title("条件別過学習ギャップ (60試行平均±SEM)")
ax.set_ylim(0, 0.19)
for b, v, s in zip(bars, gap_means, gap_sems):
    ax.text(b.get_x() + b.get_width() / 2, v + s + 0.003, f"{v:.3f}", ha="center", fontsize=10)

fig.tight_layout()
fig.savefig("3.png", dpi=140)
plt.close(fig)
print("saved 3.png")

# =================================================================
# GIF: augmentation magnitude sweeping 0 -> 1, point cloud blurring +
# decision boundary changing (single representative training draw, k=10)
# =================================================================
K_GIF = 10
magnitudes_gif = np.concatenate([
    np.linspace(0.0, 0.4, 7),
    np.linspace(0.4, 1.0, 5),
])
Xtr_g, ytr_g = make_train(0)

XXg, YYg = np.meshgrid(np.linspace(-1.8, 2.8, 90), np.linspace(-1.3, 1.8, 90))
GRIDg = np.c_[XXg.ravel(), YYg.ravel()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.2), dpi=70)


def render(i):
    ax1.clear()
    ax2.clear()
    m = magnitudes_gif[i]
    jitter_sigma = m * 0.3
    rot_max_deg = m * 20.0
    rng = np.random.default_rng(12345 + i)
    Xaug, yaug = augment(Xtr_g, ytr_g, K_GIF, jitter_sigma, rot_max_deg, rng)
    clf = fit(Xaug, yaug, alpha=1e-8, seed=0)
    te_acc = accuracy_score(yte, clf.predict(Xte))
    gap = accuracy_score(ytr_g, clf.predict(Xtr_g)) - te_acc

    Z = clf.predict_proba(GRIDg)[:, 1].reshape(XXg.shape)
    ax1.contourf(XXg, YYg, Z, levels=np.linspace(0, 1, 9), cmap="RdBu_r", alpha=0.55)
    ax1.contour(XXg, YYg, Z, levels=[0.5], colors=["k"], linewidths=2)
    ax1.scatter(Xaug[yaug == 0, 0], Xaug[yaug == 0, 1], c="#4C72B0", s=10, alpha=0.5)
    ax1.scatter(Xaug[yaug == 1, 0], Xaug[yaug == 1, 1], c="#C44E52", s=10, alpha=0.5)
    ax1.scatter(Xtr_g[ytr_g == 0, 0], Xtr_g[ytr_g == 0, 1], c="#4C72B0", edgecolors="k", s=45, zorder=3)
    ax1.scatter(Xtr_g[ytr_g == 1, 0], Xtr_g[ytr_g == 1, 1], c="#C44E52", edgecolors="k", s=45, zorder=3)
    ax1.set_xlim(-1.8, 2.8)
    ax1.set_ylim(-1.3, 1.8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"拡張の強さ m={m:.2f} (元15点+合成{K_GIF}倍)", fontsize=10)

    idx = np.searchsorted(mag[:, 0], m)
    idx = min(idx, len(mag) - 1)
    # curves = the magnitude sweep (25-trial averages); title = the values of
    # THIS single displayed run, so label both to keep them from being confused
    ax2.plot(mag[:, 0], mag[:, 4], "o-", color=COLOR_AUG, label="test_acc (25回平均)")
    ax2.plot(mag[:, 0], mag[:, 5], "s-", color="#8172B2", label="gap (25回平均)")
    ax2.scatter([m], [te_acc], marker="D", s=45, facecolor="white",
                edgecolor=COLOR_AUG, linewidth=1.5, zorder=4, label="この1回分")
    ax2.axvline(m, color="k", linestyle=":")
    ax2.set_xlabel("拡張の強さ m")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(fontsize=7, loc="center right")
    ax2.set_title(f"この1回分: test_acc={te_acc:.3f}  gap={gap:.3f}", fontsize=9.5)

    fig.tight_layout()


anim = FuncAnimation(fig, render, frames=len(magnitudes_gif), interval=350)
anim.save("augmentation-sweep.gif", writer=PillowWriter(fps=3))
plt.close(fig)
print("saved augmentation-sweep.gif")
