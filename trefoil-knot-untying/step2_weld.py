"""糸を太らせる(ノイズを増やす)と、交差部で偽のつながり(ショートカット)が生まれ、
Isomapの輪がどう壊れるかをスイープする。GIF + 定量曲線 + 3D出力。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph

plt.rcParams["font.family"] = "Meiryo"
N = 900


def trefoil(noise, seed=0):
    r = np.random.default_rng(seed)
    t = np.sort(r.uniform(0, 2 * np.pi, N))
    X = np.c_[
        np.sin(t) + 2 * np.sin(2 * t),
        np.cos(t) - 2 * np.cos(2 * t),
        -np.sin(3 * t),
    ]
    return X + r.normal(0, noise, X.shape), t


def circular_corr(emb, t):
    """埋め込みの偏角と結び目パラメータtの円環相関 (回転・鏡映を許す)"""
    c = emb - emb.mean(0)
    phi = np.arctan2(c[:, 1], c[:, 0])
    scores = []
    for sign in (1, -1):
        scores.append(np.abs(np.mean(np.exp(1j * (sign * phi - t)))))
    return max(scores)


def shortcut_count(X, t, k=10, thresh=0.5):
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity").tocoo()
    dt = np.abs(t[A.row] - t[A.col])
    dt = np.minimum(dt, 2 * np.pi - dt)
    return int((dt > thresh).sum()) // 2


noises = np.arange(0.02, 0.45, 0.03)
frames, sc_list, cc_list = [], [], []
for nz in noises:
    X, t = trefoil(nz)
    sc = shortcut_count(X, t)
    emb = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    cc = circular_corr(emb, t)
    frames.append((nz, X, t, emb, sc, cc))
    sc_list.append(sc)
    cc_list.append(cc)
    print(f"noise {nz:.2f}: shortcuts {sc:4d}, circ-corr {cc:.3f}")

# ---- 定量図 ----
fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
ax1.plot(noises, cc_list, color="#e4572e", marker="o", ms=4, label="輪の復元度 (円環相関)")
ax1.set_xlabel("糸の太さ (ノイズSD)")
ax1.set_ylabel("輪の復元度", color="#e4572e")
ax1.set_ylim(0, 1.05)
ax2 = ax1.twinx()
ax2.plot(noises, sc_list, color="#2e6ee4", marker="s", ms=4, label="ショートカットエッジ数")
ax2.set_ylabel("偽のつながり (ショートカットエッジ数)", color="#2e6ee4")
ax1.set_title("糸が太るとショートカットが生まれ、輪の復元が壊れる")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center left")
fig.tight_layout()
fig.savefig("fig3_weld_curve.png", dpi=110, bbox_inches="tight")
print("saved fig3_weld_curve.png")

# ---- GIF: 3D + Isomap ----
fig = plt.figure(figsize=(8.6, 4.2))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax2d = fig.add_subplot(1, 2, 2)


def update(fi):
    nz, X, t, emb, sc, cc = frames[fi]
    ax3d.clear()
    ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="hsv", s=3)
    ax3d.set_title(f"結び目 (ノイズ {nz:.2f})", fontsize=10)
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.view_init(elev=30, azim=-60)
    ax2d.clear()
    ax2d.scatter(emb[:, 0], emb[:, 1], c=t, cmap="hsv", s=3)
    ax2d.set_title(f"Isomap  ショートカット{sc}本 / 復元度{cc:.2f}", fontsize=10)
    ax2d.set_xticks([])
    ax2d.set_yticks([])
    return []


ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=400)
ani.save("weld.gif", writer="pillow", fps=2.5, dpi=80)
print("saved weld.gif")

# ---- plotly interactive (きれいな結び目) ----
X, t = trefoil(0.05)
figp = go.Figure(
    go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
        marker=dict(size=3.5, color=t, colorscale="HSV", colorbar=dict(title="位置 t")),
        hoverinfo="skip",
    )
)
figp.update_layout(
    title="三つ葉結び目 (ドラッグで回転できます)。位相的にはただの輪",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    margin=dict(l=0, r=0, t=60, b=0), height=650,
)
figp.write_html("trefoil-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved trefoil-3d.html")
