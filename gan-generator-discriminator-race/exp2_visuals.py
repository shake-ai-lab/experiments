import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"

X_real = np.load("C:/tmp/gan-experiment/X_real.npy")
snaps = np.load("C:/tmp/gan-experiment/snapshots.npz")
heatmaps = np.load("C:/tmp/gan-experiment/d_heatmaps.npz")
grid_x = np.load("C:/tmp/gan-experiment/grid_x.npy")
grid_y = np.load("C:/tmp/gan-experiment/grid_y.npy")
d_losses = np.load("C:/tmp/gan-experiment/d_losses.npy")
g_losses = np.load("C:/tmp/gan-experiment/g_losses.npy")

keys = sorted(int(k) for k in snaps.files)

# ============ loss curves ============
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(d_losses, color="#2563eb", label="識別器(D)の損失", linewidth=1.2, alpha=0.85)
ax.plot(g_losses, color="#dc2626", label="生成器(G)の損失", linewidth=1.2, alpha=0.85)
ax.axhline(np.log(4), color="#2563eb", linestyle="--", linewidth=1, alpha=0.6, label="理論上の均衡値 ln4≈1.386(D)")
ax.axhline(np.log(2), color="#dc2626", linestyle="--", linewidth=1, alpha=0.6, label="理論上の均衡値 ln2≈0.693(G)")
ax.set_xlabel("イテレーション")
ax.set_ylabel("損失")
ax.set_title("識別器と生成器の損失の推移")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("C:/tmp/gan-experiment/fig1_losses.png", dpi=150, bbox_inches="tight")
print("saved fig1_losses.png")

# ============ combined GIF: D's confidence heatmap (background) + G's samples (foreground) ============
fig2, ax2 = plt.subplots(figsize=(4.6, 4.3))
first_hm = heatmaps[str(keys[0])]
im = ax2.imshow(first_hm, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                 origin="lower", cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75)
real_sc = ax2.scatter(X_real[:, 0], X_real[:, 1], s=2, color="black", alpha=0.15, label="本物データ")
fake_sc = ax2.scatter([], [], s=6, color="lime", edgecolor="black", linewidth=0.2, label="生成器の出力")
ax2.set_xlim(grid_x.min(), grid_x.max())
ax2.set_ylim(grid_y.min(), grid_y.max())
ax2.legend(loc="upper right", fontsize=8)
title = ax2.set_title("")
plt.colorbar(im, ax=ax2, fraction=0.046, label="識別器が「本物」と判定する確率")

def update(k):
    im.set_data(heatmaps[str(k)])
    pts = snaps[str(k)]
    fake_sc.set_offsets(pts[:600])
    title.set_text(f"iteration {k}")
    return im, fake_sc, title

frame_keys = keys[::2] + [keys[-1]] * 6
ani = animation.FuncAnimation(fig2, update, frames=frame_keys, interval=180, blit=False)
ani.save("C:/tmp/gan-experiment/exp2_catmouse.gif", writer=animation.PillowWriter(fps=5), dpi=65)
print("saved exp2_catmouse.gif")
