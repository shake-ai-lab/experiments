"""
New intro figure: what the experimental setup actually looks like in d=2.

Draws the exact same 2000 uniform points used in the article (seed=42),
marks the query point (the origin = a corner of the unit square), and
highlights the nearest and farthest points with their real distances.
Asserts that the min/max distances match the published table before saving.

Writes 0_setup.png directly to the blog images folder.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(
    HERE, "..", "..", "src", "content", "blog", "images",
    "curse-of-dimensionality-experiment", "0_setup.png"))

N, SEED, D = 2000, 42, 2
rng = np.random.default_rng(SEED)
pts = rng.uniform(0, 1, size=(N, D))
dist = np.linalg.norm(pts, axis=1)
i_min, i_max = int(np.argmin(dist)), int(np.argmax(dist))

# sanity: must reproduce the published numbers exactly
assert abs(dist[i_min] - 0.017057277820945443) < 1e-12, dist[i_min]
assert abs(dist[i_max] - 1.4048201968321268) < 1e-12, dist[i_max]
print(f"min={dist[i_min]:.4f}  max={dist[i_max]:.4f}  ratio={dist[i_max]/dist[i_min]:.2f}")

fig, ax = plt.subplots(figsize=(7.0, 6.4), dpi=110)
ax.scatter(pts[:, 0], pts[:, 1], s=6, color="#9db4c8", alpha=0.55, linewidths=0,
           label=f"一様乱数の{N}点")

# unit square outline
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#555555", linewidth=1.2)

# lines from origin to nearest / farthest
ax.plot([0, pts[i_min, 0]], [0, pts[i_min, 1]], color="#d1495b", linewidth=2.0, zorder=4)
ax.plot([0, pts[i_max, 0]], [0, pts[i_max, 1]], color="#1b6ca8", linewidth=2.0, zorder=4)
ax.scatter([pts[i_min, 0]], [pts[i_min, 1]], s=90, facecolor="#d1495b",
           edgecolor="black", zorder=5)
ax.scatter([pts[i_max, 0]], [pts[i_max, 1]], s=90, facecolor="#1b6ca8",
           edgecolor="black", zorder=5)

# query point: the origin (a corner of the square)
ax.scatter([0], [0], marker="*", s=380, facecolor="#f2b134",
           edgecolor="black", zorder=6)
ax.text(0.015, -0.055, "基準点 = 原点 (0,0)\n(正方形の角)",
        fontsize=12, ha="left", va="top")
ax.annotate(f"一番近い点\n距離 {dist[i_min]:.4f}", xy=pts[i_min], xytext=(0.16, 0.10),
            fontsize=12, color="#d1495b",
            arrowprops=dict(arrowstyle="->", color="#d1495b"))
ax.annotate(f"一番遠い点\n距離 {dist[i_max]:.4f}", xy=pts[i_max],
            xytext=(0.33, 0.84), fontsize=12, color="#1b6ca8",
            arrowprops=dict(arrowstyle="->", color="#1b6ca8"))

ax.text(0.5, -0.185,
        f"最大/最小の比 = {dist[i_max]:.4f} / {dist[i_min]:.4f} = {dist[i_max]/dist[i_min]:.2f}倍",
        fontsize=13, ha="center", va="top")

ax.set_title(f"実験の設定を d=2 で見る: 単位正方形 $[0,1]^2$ に{N}点、原点からの距離を測る",
             fontsize=13)
ax.set_xlim(-0.06, 1.06)
ax.set_ylim(-0.30, 1.06)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print("saved", OUT)
