import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"

PE = np.load("C:/tmp/posenc-experiment/PE.npy")
max_pos, d_model = PE.shape

def fit_and_eval(k, n_train=60, n_test=30, seed=0):
    # positions available so that both pos and pos+k stay in range
    rng = np.random.default_rng(seed)
    valid_pos = np.arange(0, max_pos - k)
    rng.shuffle(valid_pos)
    train_pos = valid_pos[:n_train]
    test_pos = valid_pos[n_train:n_train + n_test]

    X_train = PE[train_pos]          # (n_train, d_model)  PE(pos)
    Y_train = PE[train_pos + k]      # (n_train, d_model)  PE(pos+k)

    # least squares: find M_k (d_model, d_model) minimizing || X_train @ M_k - Y_train ||
    M_k, *_ = np.linalg.lstsq(X_train, Y_train, rcond=None)

    X_test = PE[test_pos]
    Y_test = PE[test_pos + k]
    Y_pred = X_test @ M_k

    residual = np.abs(Y_pred - Y_test).max()
    return residual, M_k

ks = list(range(1, 51))
residuals = []
for k in ks:
    r, _ = fit_and_eval(k)
    residuals.append(r)

print("k, max abs residual on held-out positions:")
for k in [1, 2, 5, 10, 20, 30, 50]:
    print(f"  k={k:3d}  residual={residuals[k-1]:.3e}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ks, residuals, marker="o", markersize=3, color="seagreen")
ax.set_yscale("log")
ax.set_xlabel("相対距離 k")
ax.set_ylabel("held-out位置での最大誤差(対数軸)")
ax.set_title("PE(pos+k) を PE(pos) の線形変換 M_k で予測したときの誤差")
plt.savefig("C:/tmp/posenc-experiment/fig4_linear_residual.png", dpi=150, bbox_inches="tight")
print("saved fig4_linear_residual.png")

# GIF: for a fixed k, show predicted vs actual PE(pos+k) vector overlapping as we scan pos,
# and also show how well M_k generalizes to *unseen* base positions it wasn't fit on.
k_demo = 7
r, M_k = fit_and_eval(k_demo, n_train=60, n_test=30)
print(f"\ndemo k={k_demo}, held-out residual={r:.3e}")

scan_range = 120  # keep the gif small; enough to see the fit hold up
scan_positions = np.arange(0, scan_range)
preds = PE[scan_positions] @ M_k
actuals = PE[scan_positions + k_demo]

dims_to_show = [0, 1, 2, 3]  # first few dims, mix of sin/cos, high frequency -> visually informative

fig3, axes = plt.subplots(2, 2, figsize=(7, 4.2), sharex=True)
lines_actual = []
lines_pred = []
for ax, dim in zip(axes.flat, dims_to_show):
    la, = ax.plot([], [], color="steelblue", label="実際の PE(pos+k)")
    lp, = ax.plot([], [], color="crimson", linestyle="--", label="M_k · PE(pos) による予測")
    ax.set_xlim(0, scan_range)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"次元 {dim}", fontsize=9)
    lines_actual.append(la)
    lines_pred.append(lp)
axes[0, 0].legend(fontsize=7, loc="upper right")
fig3.suptitle(f"k={k_demo}: 学習済み線形変換 M_k は全ての pos で PE(pos+k) を再現できるか", fontsize=10)

def update(frame):
    upto = frame + 1
    for la, lp, dim in zip(lines_actual, lines_pred, dims_to_show):
        la.set_data(scan_positions[:upto], actuals[:upto, dim])
        lp.set_data(scan_positions[:upto], preds[:upto, dim])
    return lines_actual + lines_pred

n_frames = len(scan_positions)
frame_idxs = list(range(0, n_frames, 3)) + [n_frames - 1] * 8
ani = animation.FuncAnimation(fig3, update, frames=frame_idxs, interval=60, blit=False)
ani.save("C:/tmp/posenc-experiment/exp4_linear.gif", writer=animation.PillowWriter(fps=16), dpi=80)
print("saved exp4_linear.gif")
