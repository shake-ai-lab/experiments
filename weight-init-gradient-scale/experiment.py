import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "Meiryo"
np.seterr(over="ignore", invalid="ignore")

WIDTH = 64
INPUT_DIM = 64
BATCH = 256
DEPTH = 50

# fixed input batch and fixed regression target, shared across all init schemes
data_rng = np.random.default_rng(0)
X_FIXED = data_rng.normal(0, 1, size=(BATCH, INPUT_DIM))
TARGET_FIXED = data_rng.normal(0, 1, size=(BATCH, WIDTH))

SCHEME_SEEDS = {"naive_small": 1, "naive_large": 2, "he": 3}
SCHEME_LABELS = {
    "naive_small": "naive-small (std=0.01固定)",
    "naive_large": "naive-large (std=1.5固定)",
    "he": "He初期化 (std=sqrt(2/fan_in))",
}


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def make_weights(scheme, depth=DEPTH, width=WIDTH, input_dim=INPUT_DIM, seed=0):
    rng = np.random.default_rng(seed)
    Ws = []
    for l in range(depth):
        fan_in = input_dim if l == 0 else width
        fan_out = width
        if scheme == "naive_small":
            std = 0.01
        elif scheme == "naive_large":
            std = 1.5
        elif scheme == "he":
            std = np.sqrt(2.0 / fan_in)
        else:
            raise ValueError(scheme)
        Ws.append(rng.normal(0, std, size=(fan_in, fan_out)))
    return Ws


def run(scheme, depth=DEPTH):
    Ws = make_weights(scheme, depth=depth, seed=SCHEME_SEEDS[scheme])
    a = X_FIXED
    acts = [a.copy()]
    zs = []
    act_std = [float(np.std(a))]
    for l, W in enumerate(Ws):
        z = a @ W
        a = relu(z)
        zs.append(z)
        acts.append(a.copy())
        s = np.std(a)
        act_std.append(float(s) if np.isfinite(s) else np.nan)

    # backward pass: MSE loss vs a fixed random target
    out = acts[-1]
    dA = (out - TARGET_FIXED) / out.shape[0]
    grad_std = [None] * (depth + 1)
    gs = float(np.std(dA))
    grad_std[depth] = gs if np.isfinite(gs) else np.nan
    dA_cur = dA
    for l in range(depth - 1, -1, -1):
        z = zs[l]
        dZ = dA_cur * relu_grad(z)
        dA_prev = dZ @ Ws[l].T
        s = np.std(dA_prev)
        grad_std[l] = float(s) if np.isfinite(s) else np.nan
        dA_cur = dA_prev

    return {"act_std": act_std, "grad_std": grad_std, "acts_full": acts}


def find_overflow_layer(scheme="naive_large", max_depth=400):
    Ws = make_weights(scheme, depth=max_depth, seed=SCHEME_SEEDS[scheme])
    a = X_FIXED
    for l, W in enumerate(Ws):
        z = a @ W
        a = relu(z)
        s = np.std(a)
        if not np.isfinite(s):
            return l + 1, None
    return None, float(np.std(a))


if __name__ == "__main__":
    results = {}
    for scheme in ["naive_small", "naive_large", "he"]:
        results[scheme] = run(scheme)

    print(f"activation=ReLU, width={WIDTH}, depth={DEPTH}, batch={BATCH}, input_dim={INPUT_DIM}")
    print()
    checkpoints = [0, 1, 5, 10, 20, 30, 50]
    for scheme in ["naive_small", "naive_large", "he"]:
        r = results[scheme]
        print(f"--- {scheme} ---")
        for c in checkpoints:
            a = r["act_std"][c]
            g = r["grad_std"][c]
            print(f"  layer {c:2d}: act_std={a:.6e}  grad_std={g:.6e}")
        print()

    layer, final_std = find_overflow_layer("naive_large", max_depth=400)
    print(f"naive_large pushed to depth=400: overflow(inf) first appears at layer {layer}" if layer
          else f"naive_large at depth=400 reached act_std={final_std:.3e} without overflow")
    print()

    # ---------- Chart 1: activation std vs layer depth (log-y), 3 lines ----------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    colors = {"naive_small": "#4C72B0", "naive_large": "#C44E52", "he": "#55A868"}
    layers_x = list(range(0, DEPTH + 1))
    for scheme in ["naive_small", "naive_large", "he"]:
        y = results[scheme]["act_std"]
        ax.plot(layers_x, y, label=SCHEME_LABELS[scheme], color=colors[scheme], linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("層の深さ(0=入力)")
    ax.set_ylabel("活性化の標準偏差(log scale)")
    ax.set_title("初期化方式ごとの活性化の標準偏差 vs 層の深さ(ReLU, 幅64, 50層)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("1_activation_std_vs_depth.png")
    plt.close(fig)

    # ---------- Chart 2: gradient std vs layer depth (log-y), 3 lines ----------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    for scheme in ["naive_small", "naive_large", "he"]:
        y = results[scheme]["grad_std"]
        ax.plot(layers_x, y, label=SCHEME_LABELS[scheme], color=colors[scheme], linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("層の深さ(0=入力側, 50=出力/損失側)")
    ax.set_ylabel("勾配(dL/da)の標準偏差(log scale)")
    ax.set_title("初期化方式ごとの勾配の標準偏差 vs 層の深さ(逆伝播)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("2_gradient_std_vs_depth.png")
    plt.close(fig)

    # ---------- GIF: histogram of activations at current max depth, sweeping depth ----------
    # Panels are stacked vertically (full-width) with large fonts so the x-axis
    # scale numbers -- the whole point of this animation -- stay readable in a
    # ~720px blog column. First/last frames are held so the start is visible.
    schemes_gif = ["naive_small", "naive_large", "he"]
    frame_depths = list(range(1, DEPTH + 1, 3)) + [DEPTH]
    frame_seq = [frame_depths[0]] * 4 + frame_depths + [frame_depths[-1]] * 6

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), dpi=100)

    def update(depth):
        for ax, scheme in zip(axes, schemes_gif):
            ax.clear()
            vals = results[scheme]["acts_full"][depth].ravel()
            ax.hist(vals, bins=40, color=colors[scheme], alpha=0.85)
            ax.set_title(f"{SCHEME_LABELS[scheme]}   std={np.std(vals):.2e}", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)
            ax.xaxis.get_offset_text().set_fontsize(12)
            ax.yaxis.get_offset_text().set_fontsize(12)
        fig.suptitle(f"深さ {depth} 層目の活性化ヒストグラム", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return []

    anim = FuncAnimation(fig, update, frames=frame_seq, interval=350)
    anim.save("depth_sweep_histogram.gif", writer=PillowWriter(fps=3))
    plt.close(fig)
    print("saved: 1_activation_std_vs_depth.png, 2_gradient_std_vs_depth.png, depth_sweep_histogram.gif")
