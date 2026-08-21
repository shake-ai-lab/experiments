"""「いま出している手」と「ここまでの平均」を同じ三角形の上で同時に動かす。

この記事の主張はこの1枚に尽きる。軌道は外へ広がっていくのに、その平均は
ナッシュ均衡(★)へ吸い込まれていく。静止画だと2本の線が重なって読めないので、
時間を動かして見せる。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

plt.rcParams["font.family"] = "Meiryo"

DARK = "#3d4753"
BLUE = "#2f6f9e"
RED = "#ef476f"
LIGHT = "#c0c6cc"

xs = np.load("rps_x_eta0.05.npy")  # 10手ごとに間引いた軌道
avg = np.cumsum(xs, axis=0) / np.arange(1, len(xs) + 1)[:, None]
STRIDE = 200_000 // (len(xs) - 1)


def to_xy(p):
    r, pa, s = p[..., 0], p[..., 1], p[..., 2]
    return 0.5 * (2 * pa + s) / (r + pa + s), (np.sqrt(3) / 2) * s / (r + pa + s)


tx, ty = to_xy(xs)
ax_, ay_ = to_xy(avg)

# 前半に多くコマを割く(外へ広がるのが速いので、後半は変化が少ない)
N_FRAMES = 80
ends = np.unique(np.round(np.geomspace(30, len(xs) - 1, N_FRAMES)).astype(int))

tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
frames = []
for end in ends:
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=100)
    ax.plot(tri[:, 0], tri[:, 1], color=LIGHT, lw=1.6)
    nx, ny = to_xy(np.array([1 / 3, 1 / 3, 1 / 3]))
    ax.plot([nx], [ny], marker="*", ms=22, color=DARK, zorder=6)

    # 赤い軌跡は直近ぶんだけ残す。全部描くと後半で三角形が塗りつぶされ、
    # いま外へ広がっているという肝心の動きが見えなくなる。
    tail = max(0, end - 700)
    ax.plot(tx[tail:end], ty[tail:end], color=RED, lw=0.9, alpha=0.55, zorder=3)
    ax.plot(ax_[:end], ay_[:end], color=BLUE, lw=2.2, alpha=0.9, zorder=4)
    ax.plot([tx[end]], [ty[end]], marker="o", ms=11, color=RED, zorder=7)
    ax.plot([ax_[end]], [ay_[end]], marker="o", ms=11, color=BLUE, zorder=8)

    ax.text(-0.03, -0.07, "グー", ha="center", fontsize=14, color=DARK)
    ax.text(1.03, -0.07, "パー", ha="center", fontsize=14, color=DARK)
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "チョキ", ha="center", fontsize=14, color=DARK)
    ax.text(0.5, -0.16, f"{end * STRIDE:,} 手目", ha="center", fontsize=15, color=DARK)
    ax.text(0.02, 0.80, "● いま出している手", fontsize=14, color=RED)
    ax.text(0.02, 0.74, "● ここまでの平均", fontsize=14, color=BLUE)
    ax.text(0.02, 0.68, "★ ナッシュ均衡", fontsize=14, color=DARK)

    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-0.22, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

# 最後のコマは長めに止める
frames += [frames[-1]] * 8
imageio.mimsave("nash_spiral.gif", frames, duration=0.11, loop=0)
print("frames:", len(frames))
