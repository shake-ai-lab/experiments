"""(1) スペクトラルクラスタリングの中身: グラフラプラシアンの固有ベクトル空間で
リングが2つの点塊に分離する様子を可視化
(2) ノイズを増やしていくと、どの手法がいつ壊れるか (ARI vs noise + GIF)
(3) インタラクティブ3D"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import eigh
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

plt.rcParams["font.family"] = "Meiryo"
R, N = 1.0, 400


def make_rings(noise=0.05, seed=0):
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi, N)
    ph = r.uniform(0, 2 * np.pi, N)
    ringA = np.c_[R * np.cos(th), R * np.sin(th), np.zeros(N)]
    ringB = np.c_[R + R * np.cos(ph), np.zeros(N), R * np.sin(ph)]
    X = np.vstack([ringA, ringB]) + r.normal(0, noise, (2 * N, 3))
    y = np.r_[np.zeros(N, int), np.ones(N, int)]
    return X, y


# ---- (1) 固有ベクトルの可視化 ----
X, y = make_rings(0.05)
A = kneighbors_graph(X, n_neighbors=10, mode="connectivity")
A = 0.5 * (A + A.T).toarray()
D = np.diag(A.sum(1))
L = D - A
w, v = eigh(L)
print("最小固有値5つ:", np.round(w[:5], 4))
fiedler = v[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
axes[0].plot(np.sort(fiedler), lw=1.5, color="#333")
axes[0].set_title("第2固有ベクトル(Fiedlerベクトル)の値を昇順に並べる", fontsize=10)
axes[0].set_xlabel("点の番号(ソート後)")
axes[0].axhline(0, color="#e4572e", ls="--", lw=1)
axes[1].scatter(v[:, 1], v[:, 2], c=y, cmap="coolwarm", s=6)
axes[1].set_title("固有ベクトル空間 (v2×v3) 色=真のリング", fontsize=10)
axes[1].set_xlabel("v2")
axes[1].set_ylabel("v3")
sc = axes[2].scatter(np.arange(len(fiedler)), fiedler, c=y, cmap="coolwarm", s=5)
axes[2].set_title("点ごとのFiedler値 (色=真のリング)", fontsize=10)
axes[2].set_xlabel("点の番号")
fig.suptitle("スペクトラルクラスタリングの中身: グラフの固有ベクトルがリングを分けている", fontsize=12)
fig.tight_layout()
fig.savefig("fig3_eigen.png", dpi=110, bbox_inches="tight")
print("saved fig3_eigen.png")

# ---- (2) ノイズスイープ ----
methods = {
    "k-means": lambda X: KMeans(2, n_init=10, random_state=0).fit_predict(X),
    "GMM": lambda X: GaussianMixture(2, covariance_type="full", random_state=0).fit_predict(X),
    "スペクトラル": lambda X: SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=10, random_state=0).fit_predict(X),
    "DBSCAN": lambda X: DBSCAN(eps=0.25, min_samples=5).fit_predict(X),
}
noises = np.arange(0.02, 0.31, 0.02)
SEEDS = [0, 1, 2]
curves = {m: [] for m in methods}
spectral_frames = []
for nz in noises:
    for m, fn in methods.items():
        aris = []
        for sd in SEEDS:
            Xs, ys = make_rings(nz, seed=sd)
            aris.append(adjusted_rand_score(ys, fn(Xs)))
        curves[m].append(np.mean(aris))
    Xs, ys = make_rings(nz, seed=0)
    pred = methods["スペクトラル"](Xs)
    spectral_frames.append((nz, Xs, pred, adjusted_rand_score(ys, pred)))
    print(f"noise {nz:.2f}: " + ", ".join(f"{m} {curves[m][-1]:.2f}" for m in methods))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
colors = {"k-means": "#999", "GMM": "#2e6ee4", "スペクトラル": "#e4572e", "DBSCAN": "#3aa655"}
for m, vals in curves.items():
    ax.plot(noises, vals, marker="o", ms=4, label=m, color=colors[m])
ax.set_xlabel("ノイズの標準偏差 (リング半径=1)")
ax.set_ylabel("ARI (3シード平均)")
ax.set_title("ノイズを増やすと、どの手法がいつ壊れるか")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig4_noise_sweep.png", dpi=110, bbox_inches="tight")
print("saved fig4_noise_sweep.png")

# スペクトラルの壊れる瞬間GIF
fig = plt.figure(figsize=(5.6, 5.2))
ax = fig.add_subplot(111, projection="3d")


def update(fi):
    nz, Xs, pred, ari = spectral_frames[fi]
    ax.clear()
    ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], c=pred, cmap="coolwarm", s=6)
    ax.set_title(f"スペクトラル / ノイズ {nz:.2f}  ARI={ari:.2f}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-60)
    return []


ani = animation.FuncAnimation(fig, update, frames=len(spectral_frames), interval=500)
ani.save("noise_break.gif", writer="pillow", fps=2, dpi=80)
print("saved noise_break.gif")

# ---- (3) plotly interactive ----
X, y = make_rings(0.05)
figp = go.Figure()
for lbl, name, color in [(0, "リングA", "#2e6ee4"), (1, "リングB", "#e4572e")]:
    m = y == lbl
    figp.add_trace(
        go.Scatter3d(
            x=X[m, 0], y=X[m, 1], z=X[m, 2], mode="markers",
            marker=dict(size=3.5, color=color), name=name, hoverinfo="skip",
        )
    )
figp.update_layout(
    title="鎖のように絡まった2つのリング (ドラッグで回転できます)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    margin=dict(l=0, r=0, t=60, b=0), height=650,
)
figp.write_html("linked-rings-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved linked-rings-3d.html")
