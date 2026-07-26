"""メビウスの輪(向き付け不可能な面)は平らに展開できるのか。
多様体学習シリーズ第4弾。ひねり0回(ただの帯=展開可能)との対照実験。
評価: (a) 帯に沿った位置uの復元度(円環相関) (b) 幅方向vの符号が一周で反転するか"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

N = 1600
R, WIDTH = 2.0, 0.9


def make_band(half_twists, seed=0, n=N, noise=0.01):
    """half_twists=0: ただの円筒帯, 1: メビウス, 2: 二回ひねり"""
    r = np.random.default_rng(seed)
    u = r.uniform(0, 2 * np.pi, n)  # 帯に沿った位置
    v = r.uniform(-WIDTH / 2, WIDTH / 2, n)  # 幅方向
    t = half_twists * u / 2
    x = (R + v * np.cos(t)) * np.cos(u)
    y = (R + v * np.cos(t)) * np.sin(u)
    z = v * np.sin(t)
    X = np.c_[x, y, z] + r.normal(0, noise, (n, 3))
    return X, u, v


def circ_corr(emb, u):
    c = emb - emb.mean(0)
    phi = np.arctan2(c[:, 1], c[:, 0])
    return max(abs(np.mean(np.exp(1j * (s * phi - u)))) for s in (1, -1))


CASES = [(0, "ひねり0回 (ただの帯)"), (1, "ひねり1回 (メビウスの輪)"), (2, "ひねり2回")]
methods = {
    "PCA": lambda X: PCA(2).fit_transform(X),
    "Isomap": lambda X: Isomap(n_neighbors=12, n_components=2).fit_transform(X),
    "LLE": lambda X: LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=0).fit_transform(X),
    "t-SNE": lambda X: TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(X),
}

data = {ht: make_band(ht) for ht, _ in CASES}
np.savez("bands.npz", **{f"{ht}_{k}": v for ht, (X, u, vv) in data.items()
                          for k, v in [("X", X), ("u", u), ("v", vv)]})

# ---- 図1: 3案件×4手法の埋め込み(色=幅方向v: 表裏の代理) ----
fig, axes = plt.subplots(3, 4, figsize=(14.5, 11))
scores = {}
for i, (ht, cname) in enumerate(CASES):
    X, u, v = data[ht]
    for j, (mname, fn) in enumerate(methods.items()):
        emb = fn(X)
        cc = circ_corr(emb, u)
        scores[(cname, mname)] = cc
        ax = axes[i, j]
        ax.scatter(emb[:, 0], emb[:, 1], c=v, cmap="coolwarm", s=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{mname} / {cname}\n輪の復元 {cc:.3f}", fontsize=11)
        print(f"{cname} {mname}: circ-corr {cc:.4f}")
fig.suptitle("色 = 帯の幅方向の位置(青が片側の縁、赤が反対の縁)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig2_embeddings.png", dpi=100, bbox_inches="tight")
print("saved fig2_embeddings.png")

# ---- 縁の連結性: 帯の「縁」は何本か ----
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

print("\n=== 縁(境界)の本数 ===")
edge_counts = {}
for ht, cname in CASES:
    X, u, v = data[ht]
    # 幅方向の端に近い点だけ取り出して連結成分を数える
    m = np.abs(np.abs(v) - WIDTH / 2) < 0.06
    Xe = X[m]
    A = kneighbors_graph(Xe, n_neighbors=6, mode="connectivity")
    ncomp = connected_components(A, directed=False)[0]
    edge_counts[cname] = ncomp
    print(f"  {cname}: 縁は {ncomp} 本 (点{m.sum()}個)")

# ---- 3D plotly (メビウス) ----
X, u, v = data[1]
fig3d = go.Figure(
    go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
        marker=dict(size=2.8, color=v, colorscale="RdBu",
                    colorbar=dict(title="幅方向の<br>位置")),
        hoverinfo="skip",
    )
)
fig3d.update_layout(
    title="メビウスの輪 (ドラッグで回転)。色は帯の幅方向の位置——一周すると青と赤が入れ替わる",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    margin=dict(l=0, r=0, t=60, b=0), height=650,
)
fig3d.write_html("mobius-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved mobius-3d.html")
np.savez("scores.npz", **{f"{a}|{b}": v for (a, b), v in scores.items()},
         edges=np.array([edge_counts[c] for _, c in CASES]))
