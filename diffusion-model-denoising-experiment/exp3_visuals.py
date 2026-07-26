import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"

X0 = np.load("C:/tmp/diffusion-experiment/X0.npy")
alpha_bars = np.load("C:/tmp/diffusion-experiment/alpha_bars.npy")
T = len(alpha_bars)
rng = np.random.default_rng(2)

# ============ Figure: forward process (data -> noise) ============
show_ts = [0, 10, 25, 50, 75, 99]
fig, axes = plt.subplots(1, len(show_ts), figsize=(18, 3.2))
noise_fixed = rng.normal(0, 1, size=X0.shape)
for ax, t_idx in zip(axes, show_ts):
    ab = alpha_bars[t_idx]
    xt = np.sqrt(ab) * X0 + np.sqrt(1 - ab) * noise_fixed
    ax.scatter(xt[:, 0], xt[:, 1], s=3, alpha=0.5, color="steelblue")
    ax.set_title(f"t={t_idx}")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("forward process: 元データに少しずつノイズを加えていくと、完全なノイズに近づいていく")
plt.tight_layout()
plt.savefig("C:/tmp/diffusion-experiment/fig1_forward.png", dpi=150, bbox_inches="tight")
print("saved fig1_forward.png")

# ============ GIF: reverse process (noise -> data) ============
data = np.load("C:/tmp/diffusion-experiment/samples.npz")
ts_desc = list(range(T, -1, -1))  # T, T-1, ..., 0

fig2, ax2 = plt.subplots(figsize=(4.6, 4.4))
first = data[str(T)]
sc = ax2.scatter(first[:, 0], first[:, 1], s=5, alpha=0.55, color="crimson")
ax2.set_xlim(-4, 4); ax2.set_ylim(-4, 4)
ax2.set_xticks([]); ax2.set_yticks([])
title = ax2.set_title(f"reverse process: t={T}(完全なノイズ)")

frame_ts = ts_desc[::3] + [0] * 8
def update(t_idx):
    pts = data[str(t_idx)]
    sc.set_offsets(pts)
    title.set_text(f"reverse process: t={t_idx}" + ("(完全なノイズ)" if t_idx == T else ("(元データに近い)" if t_idx == 0 else "")))
    return sc,

ani = animation.FuncAnimation(fig2, update, frames=frame_ts, interval=70, blit=False)
ani.save("C:/tmp/diffusion-experiment/exp3_reverse.gif", writer=animation.PillowWriter(fps=13), dpi=68)
print("saved exp3_reverse.gif")
