import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "."

TRUE_FN = lambda x: np.sin(2 * np.pi * x)
NOISE_STD = 0.3


def make_data(n, rng):
    x = rng.uniform(0, 1, size=n)
    y = TRUE_FN(x) + rng.normal(0, NOISE_STD, size=n)
    return x, y


def design_matrix(x, degree):
    return np.vstack([x ** d for d in range(degree + 1)]).T


def fit_poly(x, y, degree, alpha=0.0):
    X = design_matrix(x, degree)
    # standardize columns (except bias) for numerical stability / fair ridge penalty
    mean = X[:, 1:].mean(axis=0)
    std = X[:, 1:].std(axis=0) + 1e-12
    Xs = X.copy()
    Xs[:, 1:] = (X[:, 1:] - mean) / std

    n_features = Xs.shape[1]
    reg = alpha * np.eye(n_features)
    reg[0, 0] = 0.0  # don't regularize the bias term
    w = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ y)

    def predict(xq):
        Xq = design_matrix(xq, degree)
        Xq[:, 1:] = (Xq[:, 1:] - mean) / std
        return Xq @ w

    return predict


def mse(predict, x, y):
    return float(np.mean((predict(x) - y) ** 2))


# ---------------------------------------------------------------------------
# Part 1: overfitting emerges as polynomial degree grows
# ---------------------------------------------------------------------------

N_TRAIN = 15
N_TEST = 300
DEGREES = list(range(1, 15))
N_SEEDS = 200

train_mse_all = np.zeros((N_SEEDS, len(DEGREES)))
test_mse_all = np.zeros((N_SEEDS, len(DEGREES)))

for s in range(N_SEEDS):
    rng = np.random.default_rng(s)
    x_tr, y_tr = make_data(N_TRAIN, rng)
    x_te, y_te = make_data(N_TEST, rng)
    for i, d in enumerate(DEGREES):
        predict = fit_poly(x_tr, y_tr, d, alpha=0.0)
        train_mse_all[s, i] = mse(predict, x_tr, y_tr)
        test_mse_all[s, i] = mse(predict, x_te, y_te)

train_mse_mean = train_mse_all.mean(axis=0)
test_mse_mean = test_mse_all.mean(axis=0)

print("=== degree sweep (averaged over %d seeds) ===" % N_SEEDS)
for d, tr, te in zip(DEGREES, train_mse_mean, test_mse_mean):
    print(f"degree={d:2d}  train_mse={tr:.4f}  test_mse={te:.4f}")

best_degree = DEGREES[int(np.argmin(test_mse_mean))]
print("best degree by mean test mse:", best_degree, "value:", test_mse_mean[np.argmin(test_mse_mean)])
print("degree=14 train_mse:", train_mse_mean[-1], "test_mse:", test_mse_mean[-1])

# figure: train/test mse vs degree (log scale)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(DEGREES, train_mse_mean, "o-", color="#4C72B0", label="訓練誤差 (train MSE)")
ax.plot(DEGREES, test_mse_mean, "o-", color="#DD8452", label="テスト誤差 (test MSE)")
ax.axvline(best_degree, color="gray", linestyle="--", linewidth=1, label=f"最良の次数 = {best_degree}")
ax.set_yscale("log")
ax.set_xlabel("多項式の次数 (degree)")
ax.set_ylabel("平均二乗誤差 (MSE, log scale)")
ax.set_title(f"次数を上げるほど訓練誤差は下がり続けるが、テスト誤差はU字を描く", fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_train_test_vs_degree.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part 2: illustrative single draw, 3-panel fit comparison
# ---------------------------------------------------------------------------

rng = np.random.default_rng(3)
x_tr, y_tr = make_data(N_TRAIN, rng)
x_grid = np.linspace(0, 1, 300)

panel_degrees = [1, 4, 14]
panel_labels = ["過小適合 (underfit)", "ちょうど良い (good fit)", "過学習 (overfit)"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, d, label in zip(axes, panel_degrees, panel_labels):
    predict = fit_poly(x_tr, y_tr, d, alpha=0.0)
    ax.plot(x_grid, TRUE_FN(x_grid), "--", color="gray", linewidth=1.5, label="真の関数")
    ax.plot(x_grid, predict(x_grid), color="#C44E52", linewidth=2, label=f"次数{d}のフィット")
    ax.scatter(x_tr, y_tr, color="#4C72B0", s=30, zorder=5, label="訓練データ")
    ax.set_title(f"{label}\n(degree={d})")
    ax.set_ylim(-2.2, 2.2)
    ax.legend(fontsize=8, loc="lower left")
fig.suptitle("同じ15点の訓練データに対する、次数の異なるフィット")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_three_panel_fits.png", dpi=150)
plt.close(fig)

single_train_mse = []
single_test_mse = []
x_te_illust, y_te_illust = make_data(N_TEST, np.random.default_rng(103))
for d in DEGREES:
    predict = fit_poly(x_tr, y_tr, d, alpha=0.0)
    single_train_mse.append(mse(predict, x_tr, y_tr))
    single_test_mse.append(mse(predict, x_te_illust, y_te_illust))

print("=== single illustrative draw (seed=3) ===")
for d, tr, te in zip(DEGREES, single_train_mse, single_test_mse):
    print(f"degree={d:2d}  train_mse={tr:.4f}  test_mse={te:.4f}")

# ---------------------------------------------------------------------------
# Part 3: GIF sweeping degree, fit curve + live mse readout
# ---------------------------------------------------------------------------

fig, (ax_fit, ax_mse) = plt.subplots(1, 2, figsize=(9.5, 4.2))
fig.subplots_adjust(wspace=0.3)

def animate(i):
    d = DEGREES[i]
    ax_fit.clear()
    ax_mse.clear()

    predict = fit_poly(x_tr, y_tr, d, alpha=0.0)
    ax_fit.plot(x_grid, TRUE_FN(x_grid), "--", color="gray", linewidth=1.2, label="真の関数")
    ax_fit.plot(x_grid, predict(x_grid), color="#C44E52", linewidth=2, label="フィット")
    ax_fit.scatter(x_tr, y_tr, color="#4C72B0", s=25, zorder=5, label="訓練データ")
    ax_fit.set_ylim(-3.5, 3.5)
    ax_fit.set_title(f"degree = {d}")
    ax_fit.legend(fontsize=8, loc="upper right")

    ax_mse.plot(DEGREES[: i + 1], single_train_mse[: i + 1], "o-", color="#4C72B0", label="train MSE")
    ax_mse.plot(DEGREES[: i + 1], single_test_mse[: i + 1], "o-", color="#DD8452", label="test MSE")
    ax_mse.set_xlim(0.5, max(DEGREES) + 0.5)
    ax_mse.set_ylim(0.001, max(single_test_mse) * 1.3)
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("degree")
    ax_mse.set_ylabel("MSE (log)")
    ax_mse.legend(fontsize=8, loc="upper left")
    ax_mse.set_title("誤差の推移")
    fig.tight_layout()
    return []

ani = animation.FuncAnimation(fig, animate, frames=len(DEGREES), interval=500)
ani.save(f"{OUT}/gif1_degree_sweep.gif", writer="pillow", dpi=80)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part 4: regularization (ridge) sweep at fixed high degree
# ---------------------------------------------------------------------------

FIXED_DEGREE = 14
ALPHAS = np.logspace(-6, 3, 40)

reg_train_mse_all = np.zeros((N_SEEDS, len(ALPHAS)))
reg_test_mse_all = np.zeros((N_SEEDS, len(ALPHAS)))

for s in range(N_SEEDS):
    rng = np.random.default_rng(1000 + s)
    x_tr_s, y_tr_s = make_data(N_TRAIN, rng)
    x_te_s, y_te_s = make_data(N_TEST, rng)
    for i, a in enumerate(ALPHAS):
        predict = fit_poly(x_tr_s, y_tr_s, FIXED_DEGREE, alpha=a)
        reg_train_mse_all[s, i] = mse(predict, x_tr_s, y_tr_s)
        reg_test_mse_all[s, i] = mse(predict, x_te_s, y_te_s)

reg_train_mean = reg_train_mse_all.mean(axis=0)
reg_test_mean = reg_test_mse_all.mean(axis=0)
best_alpha_idx = int(np.argmin(reg_test_mean))
best_alpha = ALPHAS[best_alpha_idx]

print("=== ridge alpha sweep at degree=%d (averaged over %d seeds) ===" % (FIXED_DEGREE, N_SEEDS))
for a, tr, te in zip(ALPHAS, reg_train_mean, reg_test_mean):
    print(f"alpha={a:.6f}  train_mse={tr:.4f}  test_mse={te:.4f}")
print("best alpha:", best_alpha, "test_mse:", reg_test_mean[best_alpha_idx])
print("no-regularization (alpha~1e-6) test_mse:", reg_test_mean[0])
print("heavy-regularization (alpha=1e3) test_mse:", reg_test_mean[-1])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ALPHAS, reg_train_mean, "o-", color="#4C72B0", markersize=3, label="訓練誤差")
ax.plot(ALPHAS, reg_test_mean, "o-", color="#DD8452", markersize=3, label="テスト誤差")
ax.axvline(best_alpha, color="gray", linestyle="--", linewidth=1, label=f"最良のalpha = {best_alpha:.3f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("正則化の強さ alpha (log scale)")
ax.set_ylabel("平均二乗誤差 (MSE, log scale)")
ax.set_title(f"degree={FIXED_DEGREE}固定でalphaを振ったときの誤差 ({N_SEEDS}回平均)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_ridge_alpha_sweep.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part 5: heatmap over (degree x alpha) grid
# ---------------------------------------------------------------------------

HEAT_DEGREES = list(range(1, 15))
HEAT_ALPHAS = np.logspace(-6, 3, 25)
HEAT_SEEDS = 60

heat_test = np.zeros((len(HEAT_DEGREES), len(HEAT_ALPHAS)))

for s in range(HEAT_SEEDS):
    rng = np.random.default_rng(5000 + s)
    x_tr_s, y_tr_s = make_data(N_TRAIN, rng)
    x_te_s, y_te_s = make_data(N_TEST, rng)
    for i, d in enumerate(HEAT_DEGREES):
        for j, a in enumerate(HEAT_ALPHAS):
            predict = fit_poly(x_tr_s, y_tr_s, d, alpha=a)
            heat_test[i, j] += mse(predict, x_te_s, y_te_s)

heat_test /= HEAT_SEEDS
heat_test_clipped = np.clip(heat_test, 0, 2.0)  # clip runaway values for readable color scale

best_flat_idx = np.unravel_index(np.argmin(heat_test), heat_test.shape)
print("=== degree x alpha heatmap ===")
print("best cell: degree=", HEAT_DEGREES[best_flat_idx[0]], "alpha=", HEAT_ALPHAS[best_flat_idx[1]], "test_mse=", heat_test[best_flat_idx])

fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(
    heat_test_clipped,
    aspect="auto",
    origin="lower",
    cmap="viridis_r",
    extent=[np.log10(HEAT_ALPHAS[0]), np.log10(HEAT_ALPHAS[-1]), HEAT_DEGREES[0], HEAT_DEGREES[-1]],
)
ax.scatter([np.log10(HEAT_ALPHAS[best_flat_idx[1]])], [HEAT_DEGREES[best_flat_idx[0]]], color="red", marker="*", s=200, label="最良点")
ax.set_xlabel("log10(alpha)")
ax.set_ylabel("次数 (degree)")
ax.set_title(f"テスト誤差の等高線 ({HEAT_SEEDS}回平均、2.0でクリップ)")
ax.legend()
fig.colorbar(im, ax=ax, label="test MSE")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_heatmap_degree_alpha.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part 6: bonus - does the ideal alpha shrink as training data grows?
# ---------------------------------------------------------------------------

N_TRAIN_LIST = [10, 15, 25, 50, 100, 300]
N_SEEDS_BONUS = 60

best_alphas_by_n = []
for n_train in N_TRAIN_LIST:
    test_mse_per_alpha = np.zeros(len(ALPHAS))
    for s in range(N_SEEDS_BONUS):
        rng = np.random.default_rng(20000 + s)
        x_tr_n, y_tr_n = make_data(n_train, rng)
        x_te_n, y_te_n = make_data(N_TEST, rng)
        for i, a in enumerate(ALPHAS):
            predict = fit_poly(x_tr_n, y_tr_n, FIXED_DEGREE, alpha=a)
            test_mse_per_alpha[i] += mse(predict, x_te_n, y_te_n)
    test_mse_per_alpha /= N_SEEDS_BONUS
    best_idx = int(np.argmin(test_mse_per_alpha))
    best_alphas_by_n.append(ALPHAS[best_idx])
    print(f"n_train={n_train:4d}  best_alpha={ALPHAS[best_idx]:.5f}  best_test_mse={test_mse_per_alpha[best_idx]:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(N_TRAIN_LIST, best_alphas_by_n, "o-", color="#55A868")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("訓練データ数 n (log scale)")
ax.set_ylabel("最良の正則化強さ alpha (log scale)")
ax.set_title(f"degree={FIXED_DEGREE}固定、データが増えるほど最適なalphaはどう変わるか")
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_best_alpha_vs_n.png", dpi=150)
plt.close(fig)

print("done")
