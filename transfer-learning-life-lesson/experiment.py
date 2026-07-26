import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import load_digits

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "."

digits = load_digits()
X_all = digits.data / 16.0
y_all = digits.target

SOURCE_CLASSES = [0, 1, 2, 3, 4]
TARGET_CLASSES = [5, 6, 7, 8, 9]
N_TEST_PER_CLASS = 100
HIDDEN = 64
N_CLASSES = 5


def subset(classes):
    mask = np.isin(y_all, classes)
    Xc, yc = X_all[mask], y_all[mask]
    remap = {c: i for i, c in enumerate(classes)}
    yc = np.array([remap[v] for v in yc])
    return Xc, yc


def split_train_pool_test(X, y, n_test_per_class, rng):
    test_idx, pool_idx = [], []
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        test_idx.extend(idx[:n_test_per_class])
        pool_idx.extend(idx[n_test_per_class:])
    return np.array(pool_idx), np.array(test_idx)


def draw_small_set(pool_idx, y, n_per_class, rng):
    chosen = []
    for c in range(N_CLASSES):
        cls_idx = pool_idx[y[pool_idx] == c]
        chosen.extend(rng.choice(cls_idx, size=n_per_class, replace=False))
    return np.array(chosen)


def one_hot(y, k):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


def init_params(rng, in_dim=64, hidden=HIDDEN, out_dim=N_CLASSES):
    return {
        "W1": rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, np.sqrt(2 / hidden), size=(hidden, out_dim)),
        "b2": np.zeros(out_dim),
    }


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(0, z1)
    z2 = a1 @ params["W2"] + params["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)
    probs = np.exp(z2) / np.exp(z2).sum(axis=1, keepdims=True)
    return z1, a1, probs


def accuracy(params, X, y):
    _, _, probs = forward(params, X)
    return float(np.mean(probs.argmax(axis=1) == y))


def cross_entropy(probs, y):
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def init_adam(params):
    return {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in params.items()}


def adam_step(params, grads, state, t, trainable, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    for k in trainable:
        state[k]["m"] = beta1 * state[k]["m"] + (1 - beta1) * grads[k]
        state[k]["v"] = beta2 * state[k]["v"] + (1 - beta2) * (grads[k] ** 2)
        m_hat = state[k]["m"] / (1 - beta1 ** t)
        v_hat = state[k]["v"] / (1 - beta2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train(params, X, y, trainable, epochs, lr=0.01, X_test=None, y_test=None, record_every=None):
    state = init_adam(params)
    history = []
    n = len(y)
    Y1h = one_hot(y, N_CLASSES)
    for t in range(1, epochs + 1):
        z1, a1, probs = forward(params, X)
        dz2 = (probs - Y1h) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ params["W2"].T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        adam_step(params, grads, state, t, trainable, lr=lr)

        if record_every and (t % record_every == 0 or t == 1):
            entry = {"epoch": t, "train_acc": accuracy(params, X, y)}
            if X_test is not None:
                entry["test_acc"] = accuracy(params, X_test, y_test)
            history.append(entry)
    return params, history


# ---------------------------------------------------------------------------
# Part 1: pretrain the source model on digits 0-4 (plenty of data)
# ---------------------------------------------------------------------------

rng0 = np.random.default_rng(0)
X_src, y_src = subset(SOURCE_CLASSES)
perm = rng0.permutation(len(y_src))
X_src, y_src = X_src[perm], y_src[perm]
n_src_test = 150
X_src_test, y_src_test = X_src[:n_src_test], y_src[:n_src_test]
X_src_train, y_src_train = X_src[n_src_test:], y_src[n_src_test:]

print(f"source train n={len(y_src_train)}, source test n={len(y_src_test)}")

source_params = init_params(rng0)
source_params, src_hist = train(
    source_params, X_src_train, y_src_train, trainable=["W1", "b1", "W2", "b2"],
    epochs=400, lr=0.02, X_test=X_src_test, y_test=y_src_test, record_every=20,
)
print("source final train/test acc:", src_hist[-1]["train_acc"], src_hist[-1]["test_acc"])

PRETRAINED_W1 = source_params["W1"].copy()
PRETRAINED_B1 = source_params["b1"].copy()

# ---------------------------------------------------------------------------
# Part 2: target task (digits 5-9), scratch vs frozen-transfer vs finetune-transfer
# ---------------------------------------------------------------------------

X_tgt, y_tgt = subset(TARGET_CLASSES)
rng1 = np.random.default_rng(1)
pool_idx, test_idx = split_train_pool_test(X_tgt, y_tgt, N_TEST_PER_CLASS, rng1)
X_tgt_test, y_tgt_test = X_tgt[test_idx], y_tgt[test_idx]
print(f"target pool n={len(pool_idx)}, target test n={len(test_idx)}")

N_PER_CLASS_LIST = [1, 2, 3, 5, 10, 20, 40]
N_SEEDS = 40
EPOCHS = 300

conditions = ["scratch", "frozen", "finetune"]
results = {c: {n: [] for n in N_PER_CLASS_LIST} for c in conditions}

for seed in range(N_SEEDS):
    rng = np.random.default_rng(1000 + seed)
    for n_per_class in N_PER_CLASS_LIST:
        train_idx = draw_small_set(pool_idx, y_tgt, n_per_class, rng)
        Xtr, ytr = X_tgt[train_idx], y_tgt[train_idx]

        # scratch: random init, train everything
        p_scratch = init_params(rng)
        p_scratch, _ = train(p_scratch, Xtr, ytr, trainable=["W1", "b1", "W2", "b2"], epochs=EPOCHS, lr=0.02)
        results["scratch"][n_per_class].append(accuracy(p_scratch, X_tgt_test, y_tgt_test))

        # frozen transfer: reuse pretrained W1/b1 untouched, train only W2/b2 (linear probe)
        p_frozen = init_params(rng)
        p_frozen["W1"], p_frozen["b1"] = PRETRAINED_W1.copy(), PRETRAINED_B1.copy()
        p_frozen, _ = train(p_frozen, Xtr, ytr, trainable=["W2", "b2"], epochs=EPOCHS, lr=0.02)
        results["frozen"][n_per_class].append(accuracy(p_frozen, X_tgt_test, y_tgt_test))

        # finetune transfer: start from pretrained W1/b1, train everything
        p_finetune = init_params(rng)
        p_finetune["W1"], p_finetune["b1"] = PRETRAINED_W1.copy(), PRETRAINED_B1.copy()
        p_finetune, _ = train(p_finetune, Xtr, ytr, trainable=["W1", "b1", "W2", "b2"], epochs=EPOCHS, lr=0.02)
        results["finetune"][n_per_class].append(accuracy(p_finetune, X_tgt_test, y_tgt_test))
    if (seed + 1) % 10 == 0:
        print(f"seed {seed + 1}/{N_SEEDS} done")

mean_acc = {c: [np.mean(results[c][n]) for n in N_PER_CLASS_LIST] for c in conditions}
std_acc = {c: [np.std(results[c][n]) for n in N_PER_CLASS_LIST] for c in conditions}

print("=== target accuracy by n_per_class (mean over %d seeds) ===" % N_SEEDS)
for i, n in enumerate(N_PER_CLASS_LIST):
    print(
        f"n_per_class={n:3d}  scratch={mean_acc['scratch'][i]:.4f}  "
        f"frozen={mean_acc['frozen'][i]:.4f}  finetune={mean_acc['finetune'][i]:.4f}"
    )

fig, ax = plt.subplots(figsize=(7.5, 5))
colors = {"scratch": "#C44E52", "frozen": "#55A868", "finetune": "#4C72B0"}
labels = {"scratch": "スクラッチ(ランダム初期化)", "frozen": "転移(特徴量は凍結、出力層のみ学習)", "finetune": "転移(全層をfine-tune)"}
for c in conditions:
    ax.errorbar(
        N_PER_CLASS_LIST, mean_acc[c], yerr=std_acc[c], marker="o", capsize=3,
        color=colors[c], label=labels[c],
    )
ax.set_xscale("log")
ax.set_xlabel("クラスあたりの訓練データ数 (log scale)")
ax.set_ylabel("テスト精度 (5クラス分類)")
ax.set_title(f"digits 5-9の分類精度: スクラッチ vs 転移学習 ({N_SEEDS}回平均)")
ax.legend()
ax.axhline(0.2, color="gray", linestyle=":", linewidth=1, label="偶然(20%)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_accuracy_vs_n.png", dpi=150)
plt.close(fig)

# gap between finetune-transfer and scratch, as n grows
gap = np.array(mean_acc["finetune"]) - np.array(mean_acc["scratch"])
print("=== finetune - scratch gap ===")
for n, g in zip(N_PER_CLASS_LIST, gap):
    print(f"n_per_class={n:3d}  gap={g:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(N_PER_CLASS_LIST, gap, "o-", color="#8172B2")
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xscale("log")
ax.set_xlabel("クラスあたりの訓練データ数 (log scale)")
ax.set_ylabel("精度差 (fine-tune転移 − スクラッチ)")
ax.set_title("スクラッチの優位性は、データが増えるほどほぼ消えていく")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_gap_vs_n.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part 3: visualize pretrained hidden-layer receptive fields
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(4, 8, figsize=(9, 5))
for i, ax in enumerate(axes.flat):
    w = PRETRAINED_W1[:, i].reshape(8, 8)
    ax.imshow(w, cmap="RdBu_r", vmin=-np.abs(w).max(), vmax=np.abs(w).max())
    ax.axis("off")
fig.suptitle("digits 0-4で事前学習した隠れ層32ユニット分の重み(受容野)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_receptive_fields.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part 4: GIF comparing training curves, scratch vs finetune, at small n
# ---------------------------------------------------------------------------

GIF_N_PER_CLASS = 1
rng_gif = np.random.default_rng(5008)
train_idx = draw_small_set(pool_idx, y_tgt, GIF_N_PER_CLASS, rng_gif)
Xtr, ytr = X_tgt[train_idx], y_tgt[train_idx]

p_scratch = init_params(rng_gif)
_, hist_scratch = train(
    p_scratch, Xtr, ytr, trainable=["W1", "b1", "W2", "b2"], epochs=EPOCHS, lr=0.02,
    X_test=X_tgt_test, y_test=y_tgt_test, record_every=5,
)

p_finetune = init_params(rng_gif)
p_finetune["W1"], p_finetune["b1"] = PRETRAINED_W1.copy(), PRETRAINED_B1.copy()
_, hist_finetune = train(
    p_finetune, Xtr, ytr, trainable=["W1", "b1", "W2", "b2"], epochs=EPOCHS, lr=0.02,
    X_test=X_tgt_test, y_test=y_tgt_test, record_every=5,
)

epochs_recorded = [h["epoch"] for h in hist_scratch]
scratch_test = [h["test_acc"] for h in hist_scratch]
finetune_test = [h["test_acc"] for h in hist_finetune]

print("=== gif run (n_per_class=3) final test acc ===")
print("scratch:", scratch_test[-1], "finetune:", finetune_test[-1])

fig, ax = plt.subplots(figsize=(7, 4.5))

def animate(i):
    ax.clear()
    ax.plot(epochs_recorded[: i + 1], scratch_test[: i + 1], color="#C44E52", label="スクラッチ")
    ax.plot(epochs_recorded[: i + 1], finetune_test[: i + 1], color="#4C72B0", label="転移(fine-tune)")
    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("テスト精度")
    ax.set_title(f"クラスあたり{GIF_N_PER_CLASS}件だけで学習したときの収束の違い")
    ax.legend(loc="lower right")
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=1)
    fig.tight_layout()
    return []

ani = animation.FuncAnimation(fig, animate, frames=len(epochs_recorded), interval=100)
ani.save(f"{OUT}/gif1_convergence.gif", writer="pillow", dpi=80)
plt.close(fig)

print("done")
