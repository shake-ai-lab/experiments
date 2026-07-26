"""(1) カオスゲームで点が増えて模様が浮かぶGIF
(2) 頂点へズームしても同じ模様が現れる自己相似ズームGIF"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
pts = np.load("sierpinski_pts.npy")  # 約50万点

# ---- GIF1: 点が増えていく ----
counts = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, len(pts)]
fig, ax = plt.subplots(figsize=(5.4, 5.2))


def update(fi):
    n = counts[fi]
    ax.clear()
    ax.scatter(pts[:n, 0], pts[:n, 1], s=0.3, color="#e4572e")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"カオスゲーム: {n:,}点", fontsize=12)
    return []


ani = animation.FuncAnimation(fig, update, frames=len(counts), interval=450)
ani.save("chaos_game.gif", writer="pillow", fps=2.2, dpi=80)
print("saved chaos_game.gif")
plt.close(fig)

# ---- GIF2: 自己相似ズーム (左下の頂点 (0.05,0.05) へ) ----
target = np.array([0.05, 0.05])
fig, ax = plt.subplots(figsize=(5.4, 5.2))
zooms = 2 ** np.linspace(0, 4, 33)  # 1x -> 16x


def update2(fi):
    z = zooms[fi]
    half = 0.5 / z
    cx, cy = target + half * 0.98  # 頂点が左下に来るように
    m = (
        (pts[:, 0] > cx - half) & (pts[:, 0] < cx + half)
        & (pts[:, 1] > cy - half) & (pts[:, 1] < cy + half)
    )
    sub = pts[m]
    ax.clear()
    ax.scatter(sub[:, 0], sub[:, 1], s=0.35, color="#e4572e")
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"ズーム {z:.1f}倍 (表示中の点: {m.sum():,})", fontsize=12)
    return []


ani2 = animation.FuncAnimation(fig, update2, frames=len(zooms), interval=150)
ani2.save("zoom.gif", writer="pillow", fps=7, dpi=80)
print("saved zoom.gif")
