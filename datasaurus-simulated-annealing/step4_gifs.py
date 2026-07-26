"""GIF生成: (1)4形状同時モーフィング2x2グリッド (2)統計量ライブ表示つきサーモン変形"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"

d = np.load("shape_results.npz")
dino = d["dino"]
SNAP = int(d["snap_every"])
names = [("circle", "円"), ("star", "星"), ("x_shape", "X"), ("salmon", "サーモン")]

# 全ターゲットのフレーム数を揃える (101フレーム、ストライド2 -> 51)
STRIDE = 2
frames = range(0, d["circle_snaps"].shape[0], STRIDE)
frame_list = list(frames)


def stats_of(pts):
    x, y = pts[:, 0], pts[:, 1]
    return (x.mean(), y.mean(), x.std(ddof=1), y.std(ddof=1), np.corrcoef(x, y)[0, 1])


# ---- GIF 1: 2x2同時モーフィング ----
fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.8))
scatters = []
for ax, (key, label) in zip(axes.flat, names):
    sc = ax.scatter([], [], s=5, color="#e4572e")
    ax.set_xlim(-5, 110)
    ax.set_ylim(-2, 108)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"dino → {label}", fontsize=10)
    scatters.append(sc)
title = fig.suptitle("", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])


def update(fi):
    f = frame_list[fi]
    for sc, (key, _) in zip(scatters, names):
        snaps = d[f"{key}_snaps"]
        sc.set_offsets(snaps[min(f, len(snaps) - 1)])
    title.set_text(
        f"平均・分散・相関は小数2桁で固定のまま   iter {min(f * SNAP, 1_500_000):,}"
    )
    return scatters


ani = animation.FuncAnimation(fig, update, frames=len(frame_list), interval=100)
ani.save("morph4.gif", writer="pillow", fps=10, dpi=80)
print("saved morph4.gif")
plt.close(fig)

# ---- GIF 2: サーモン変形 + 統計量ライブ表示 ----
fig = plt.figure(figsize=(7.2, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
ax = fig.add_subplot(gs[0])
ax_t = fig.add_subplot(gs[1])
ax_t.axis("off")
sc = ax.scatter([], [], s=6, color="#e4572e")
ax.set_xlim(-5, 110)
ax.set_ylim(-2, 108)
ax.set_xticks([])
ax.set_yticks([])
txt = ax_t.text(0.02, 0.5, "", fontsize=12, family="Meiryo", va="center", linespacing=2.0)
snaps = d["salmon_snaps"]


def update2(fi):
    f = min(frame_list[fi], len(snaps) - 1)
    pts = snaps[f]
    sc.set_offsets(pts)
    mx, my, sx, sy, r = stats_of(pts)
    txt.set_text(
        f"iter {min(f * SNAP, 1_500_000):>9,}\n\n"
        f"平均 x  = {mx:6.2f}\n平均 y  = {my:6.2f}\n"
        f"SD x    = {sx:6.2f}\nSD y    = {sy:6.2f}\n相関    = {r:6.2f}"
    )
    return sc, txt


ani2 = animation.FuncAnimation(fig, update2, frames=len(frame_list), interval=100)
ani2.save("salmon_stats.gif", writer="pillow", fps=10, dpi=80)
print("saved salmon_stats.gif")
