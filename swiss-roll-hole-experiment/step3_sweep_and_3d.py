"""(1) 穴を0から成長させたときのIsomap/LLE埋め込みのスイープGIF (corr併記)
(2) intro用の静的3D+シート図 (fig1)
(3) インタラクティブplotly 3D (大きい穴)"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, LocallyLinearEmbedding

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

X_all, t_all = make_swiss_roll(n_samples=6000, noise=0.05, random_state=0)
y_all = X_all[:, 1]
order = np.argsort(t_all)
t_sorted = t_all[order]
s_sorted = cumulative_trapezoid(np.sqrt(1 + t_sorted**2), t_sorted, initial=0)
s_all = np.empty_like(s_sorted)
s_all[order] = s_sorted
s_c, y_c = s_all.max() * 0.55, (y_all.min() + y_all.max()) / 2


def cut(radius, seed=0):
    keep = (s_all - s_c) ** 2 + (y_all - y_c) ** 2 > radius**2
    idx = np.flatnonzero(keep)
    return np.random.default_rng(seed).choice(idx, size=1200, replace=False)


def best_corr(emb, t):
    return max(abs(np.corrcoef(emb[:, i], t)[0, 1]) for i in range(emb.shape[1]))


radii = np.arange(0, 9.5, 0.5)
frames = []
for R in radii:
    idx = cut(R)
    X, t, s, y = X_all[idx], t_all[idx], s_all[idx], y_all[idx]
    e_iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    e_lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=0).fit_transform(X)
    frames.append(
        dict(R=R, s=s, y=y, t=t, iso=e_iso, c_iso=best_corr(e_iso, t), lle=e_lle, c_lle=best_corr(e_lle, t))
    )
    print(f"R={R}: iso {frames[-1]['c_iso']:.3f}, lle {frames[-1]['c_lle']:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
titles = ["シート (真の座標)", "Isomap", "LLE"]
for ax, ti in zip(axes, titles):
    ax.set_xticks([])
    ax.set_yticks([])
scs = [axes[i].scatter([], [], s=3, c=[], cmap="turbo") for i in range(3)]


def update(fi):
    f = frames[fi]
    for i, (ax, data) in enumerate(
        zip(axes, [np.c_[f["s"], f["y"]], f["iso"], f["lle"]])
    ):
        ax.clear()
        ax.scatter(data[:, 0], data[:, 1], s=3, c=f["t"], cmap="turbo")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title(f"シート (穴の半径 {f['R']:.1f})", fontsize=10)
    axes[1].set_title(f"Isomap  corr={f['c_iso']:.3f}", fontsize=10)
    axes[2].set_title(f"LLE  corr={f['c_lle']:.3f}", fontsize=10)
    return []


ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=350)
ani.save("hole_sweep.gif", writer="pillow", fps=3, dpi=80)
print("saved hole_sweep.gif")
plt.close(fig)

# ---- fig1: 3D + シートの静的図 (大きい穴) ----
idx = cut(8.0)
X, t, s, y = X_all[idx], t_all[idx], s_all[idx], y_all[idx]
fig = plt.figure(figsize=(11, 4))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(X[:, 0], X[:, 2], X[:, 1], c=t, cmap="turbo", s=4)
ax1.set_title("3D空間での穴あきスイスロール", fontsize=11)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(s, y, c=t, cmap="turbo", s=4)
ax2.set_title("シートとして広げた真の姿 (弧長s × 幅y)", fontsize=11)
ax2.set_xlabel("弧長 s")
ax2.set_ylabel("幅 y")
fig.tight_layout()
fig.savefig("fig1_holed_roll.png", dpi=110, bbox_inches="tight")
print("saved fig1_holed_roll.png")

# ---- plotly interactive ----
figp = go.Figure()
figp.add_trace(
    go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode="markers",
        marker=dict(size=3.5, color=t, colorscale="Turbo", colorbar=dict(title="巻きの位置<br>(t)")),
        hoverinfo="skip",
    )
)
figp.update_layout(
    title="穴あきスイスロール。渦の中ほどに円形の穴が開いている(ドラッグで回転できます)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    margin=dict(l=0, r=0, t=60, b=0),
    height=650,
)
figp.write_html("swiss-roll-hole-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved swiss-roll-hole-3d.html")
