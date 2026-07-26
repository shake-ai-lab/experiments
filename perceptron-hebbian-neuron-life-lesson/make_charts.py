import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Meiryo"

# --- dataviz skill: palette & chrome tokens ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"    # categorical slot 1
GREEN = "#008300"   # categorical slot 2


def new_fig(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    return fig, ax


def style_title(ax, title):
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, fontweight="bold", pad=14, loc="left")


def style_labels(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10.5)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10.5)


def end_label(ax, x, y, text, color, dy=0, ha="left"):
    ax.annotate(
        text, xy=(x, y), xytext=(6, dy), textcoords="offset points",
        color=color, fontsize=10.5, fontweight="bold", va="center", ha=ha,
    )


# ============================================================
# 1) 重みノルムの推移 (dual-axisをやめて、伝えたい1つの事実に絞る)
# ============================================================
EPOCHS = 40
rng = np.random.default_rng(0)


def decide(w, b, X):
    return np.where(X @ w + b >= 0, 1.0, -1.0)


def make_data(n_pos, n_neg, seed):
    local_rng = np.random.default_rng(seed)
    pos = local_rng.normal(loc=[2, 2], scale=1.0, size=(n_pos, 2))
    neg = local_rng.normal(loc=[-2, -2], scale=1.0, size=(n_neg, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
    perm = local_rng.permutation(len(X))
    return X[perm], y[perm]


def perceptron_train(X, y, epochs, lr=0.1, shuffle_seed=123):
    w = np.zeros(X.shape[1]); b = 0.0
    norm_history = []
    shuffle_rng = np.random.default_rng(shuffle_seed)
    for epoch in range(epochs):
        order = shuffle_rng.permutation(len(X))
        for xi, yi in zip(X[order], y[order]):
            pred = 1 if (w @ xi + b) >= 0 else -1
            if pred != yi:
                w += lr * yi * xi
                b += lr * yi
        norm_history.append(np.linalg.norm(w))
    return w, b, norm_history


def hebbian_train(X, y, epochs, lr=0.1, shuffle_seed=123):
    w = np.zeros(X.shape[1]); b = 0.0
    norm_history = []
    shuffle_rng = np.random.default_rng(shuffle_seed)
    for epoch in range(epochs):
        order = shuffle_rng.permutation(len(X))
        for xi, yi in zip(X[order], y[order]):
            w += lr * yi * xi
            b += lr * yi
        norm_history.append(np.linalg.norm(w))
    return w, b, norm_history


X_bal, y_bal = make_data(500, 500, seed=1)
w_p, b_p, norm_p = perceptron_train(X_bal, y_bal, EPOCHS)
w_h, b_h, norm_h = hebbian_train(X_bal, y_bal, EPOCHS)

fig, ax = new_fig()
xs = np.arange(1, EPOCHS + 1)
ax.plot(xs, norm_p, color=BLUE, linewidth=2.2, solid_capstyle="round", label="perceptron学習則(誤差信号あり)")
ax.plot(xs, norm_h, color=GREEN, linewidth=2.2, solid_capstyle="round", label="ヘブ則(誤差信号なし)")
ax.scatter([EPOCHS], [norm_p[-1]], s=42, color=BLUE, zorder=5, edgecolor=SURFACE, linewidth=1.5)
ax.scatter([EPOCHS], [norm_h[-1]], s=42, color=GREEN, zorder=5, edgecolor=SURFACE, linewidth=1.5)
end_label(ax, EPOCHS, norm_p[-1], f"perceptron: {norm_p[-1]:.2f}", BLUE, dy=-4)
end_label(ax, EPOCHS, norm_h[-1], f"ヘブ則: {norm_h[-1]:,.0f}", GREEN, dy=6)
ax.set_xlim(0, EPOCHS + 9)
style_title(ax, "分類精度はほぼ同じ(99.5%)でも、重みの大きさは全く違う動きをする")
style_labels(ax, "エポック", "重みベクトルのノルム ||w||")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
plt.tight_layout()
plt.savefig("weight_norm_v2.png", dpi=150, facecolor=SURFACE)
plt.close()
print(f"perceptron final ||w||={norm_p[-1]:.2f}  hebbian final ||w||={norm_h[-1]:.2f}")

# ============================================================
# 2) 単層パーセプトロンはXORで頭打ち
# ============================================================
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([-1, 1, 1, -1], dtype=float)
X_xor_rep = np.tile(X_xor, (200, 1))
y_xor_rep = np.tile(y_xor, 200)


def perceptron_train_acc(X, y, epochs, lr=0.1, shuffle_seed=123):
    w = np.zeros(X.shape[1]); b = 0.0
    acc_history = []
    shuffle_rng = np.random.default_rng(shuffle_seed)
    for epoch in range(epochs):
        order = shuffle_rng.permutation(len(X))
        for xi, yi in zip(X[order], y[order]):
            pred = 1 if (w @ xi + b) >= 0 else -1
            if pred != yi:
                w += lr * yi * xi
                b += lr * yi
        acc_history.append((decide(w, b, X) == y).mean())
    return w, b, acc_history


w_x, b_x, acc_x = perceptron_train_acc(X_xor_rep, y_xor_rep, epochs=60, lr=0.1)

fig, ax = new_fig()
xs2 = np.arange(1, len(acc_x) + 1)
ax.axhline(1.0, color=INK_MUTED, linewidth=1, zorder=1)
ax.annotate("目標(完全分離)", xy=(60, 1.0), xytext=(6, 0), textcoords="offset points",
            color=INK_MUTED, fontsize=9.5, va="center")
ax.plot(xs2, acc_x, color=BLUE, linewidth=2.2, solid_capstyle="round")
ax.scatter([len(acc_x)], [acc_x[-1]], s=42, color=BLUE, zorder=5, edgecolor=SURFACE, linewidth=1.5)
end_label(ax, len(acc_x), acc_x[-1], f"{acc_x[-1]*100:.0f}%で振動", BLUE, dy=0)
ax.set_xlim(0, len(acc_x) + 14)
ax.set_ylim(0.15, 1.1)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
style_title(ax, "単層パーセプトロンはXORを解けず、精度が頭打ちになる")
style_labels(ax, "エポック", "XORデータでの精度")
plt.tight_layout()
plt.savefig("xor_perceptron_stuck_v2.png", dpi=150, facecolor=SURFACE)
plt.close()

# ============================================================
# 3) MLPの損失推移
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mlp_train(X, y01, hidden=4, epochs=20000, lr=1.0, seed=3):
    local_rng = np.random.default_rng(seed)
    W1 = local_rng.normal(0, 1, size=(X.shape[1], hidden)); b1 = np.zeros(hidden)
    W2 = local_rng.normal(0, 1, size=(hidden, 1)); b2 = np.zeros(1)
    loss_history = []
    for epoch in range(epochs):
        h = sigmoid(X @ W1 + b1)
        out = sigmoid(h @ W2 + b2).ravel()
        loss = np.mean((out - y01) ** 2)
        loss_history.append(loss)
        d_out = (out - y01) * out * (1 - out)
        d_W2 = h.T @ d_out.reshape(-1, 1) / len(X)
        d_b2 = d_out.mean()
        d_h = np.outer(d_out, W2.ravel()) * h * (1 - h)
        d_W1 = X.T @ d_h / len(X)
        d_b1 = d_h.mean(axis=0)
        W2 -= lr * d_W2; b2 -= lr * d_b2; W1 -= lr * d_W1; b1 -= lr * d_b1
    return (W1, b1, W2, b2), loss_history


y_xor01 = (y_xor + 1) / 2
(W1, b1, W2, b2), loss_hist = mlp_train(X_xor, y_xor01, hidden=4, epochs=20000, lr=1.0)

fig, ax = new_fig()
xs3 = np.arange(1, len(loss_hist) + 1)
ax.plot(xs3, loss_hist, color=BLUE, linewidth=2.2, solid_capstyle="round")
ax.scatter([len(loss_hist)], [loss_hist[-1]], s=42, color=BLUE, zorder=5, edgecolor=SURFACE, linewidth=1.5)
end_label(ax, len(loss_hist), loss_hist[-1], f"損失 {loss_hist[-1]:.4f}", BLUE, dy=8)
ax.set_xlim(0, len(loss_hist) * 1.18)
style_title(ax, "隠れ層を1つ足したMLPはXORを解いて損失がゼロに近づく")
style_labels(ax, "エポック", "損失(MSE)")
plt.tight_layout()
plt.savefig("xor_mlp_loss_v2.png", dpi=150, facecolor=SURFACE)
plt.close()

# ============================================================
# 4) 層を足す代わりに特徴量を1個足す
# ============================================================
X_xor_aug = np.column_stack([X_xor[:, 0], X_xor[:, 1], X_xor[:, 0] * X_xor[:, 1]])
X_xor_aug_rep = np.tile(X_xor_aug, (200, 1))
y_xor_aug_rep = np.tile(y_xor, 200)
w_aug, b_aug, acc_aug = perceptron_train_acc(X_xor_aug_rep, y_xor_aug_rep, epochs=60, lr=0.1)

fig, ax = new_fig()
xs4 = np.arange(1, len(acc_x) + 1)
ax.plot(xs4, acc_x, color=BLUE, linewidth=2.2, solid_capstyle="round", label="素のx1, x2だけ(層を足さない)")
ax.plot(xs4, acc_aug, color=GREEN, linewidth=2.2, solid_capstyle="round", label="x1*x2を足しただけ(層を足さない)")
ax.scatter([len(acc_x)], [acc_x[-1]], s=42, color=BLUE, zorder=5, edgecolor=SURFACE, linewidth=1.5)
ax.scatter([len(acc_aug)], [acc_aug[-1]], s=42, color=GREEN, zorder=5, edgecolor=SURFACE, linewidth=1.5)
ax.set_xlim(0, len(acc_x) + 14)
ax.set_ylim(0.15, 1.32)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
style_title(ax, "層を足さなくても、掛け算という特徴量を1つ足すだけでXORは解ける")
style_labels(ax, "エポック", "XORデータでの精度")
legend = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.99), fontsize=9.5, frameon=False, labelcolor=INK_SECONDARY)
plt.tight_layout()
plt.savefig("xor_feature_trick_v2.png", dpi=150, facecolor=SURFACE)
plt.close()

print("done")
