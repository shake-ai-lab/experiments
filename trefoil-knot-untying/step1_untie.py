"""三つ葉結び目(trefoil knot)は次元削減で「ほどける」のか。
結び目は位相的にはただの輪(1次元ループ)。内在的な距離だけを見る手法なら
円にほどけるはずだが、3D空間で交差する糸同士が近づく場所では、kNNグラフに
「ショートカット」が生まれて偽のつながりができる。それを実測する。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import kneighbors_graph

plt.rcParams["font.family"] = "Meiryo"
N = 900


def trefoil(noise=0.05, seed=0):
    r = np.random.default_rng(seed)
    t = np.sort(r.uniform(0, 2 * np.pi, N))
    X = np.c_[
        np.sin(t) + 2 * np.sin(2 * t),
        np.cos(t) - 2 * np.cos(2 * t),
        -np.sin(3 * t),
    ]
    return X + r.normal(0, noise, X.shape), t


X, t = trefoil()
np.savez("trefoil.npz", X=X, t=t)

# ショートカット検出: kNNグラフのエッジのうち、パラメータtの差が大きいもの
A = kneighbors_graph(X, n_neighbors=10, mode="connectivity").tocoo()
dt = np.abs(t[A.row] - t[A.col])
dt = np.minimum(dt, 2 * np.pi - dt)  # 円周上の距離
shortcuts = int((dt > 0.5).sum()) // 2
print(f"ショートカットエッジ (Δt>0.5): {shortcuts}本 / 全{A.nnz//2}本")

methods = {
    "PCA": lambda X: PCA(2).fit_transform(X),
    "Isomap": lambda X: Isomap(n_neighbors=10, n_components=2).fit_transform(X),
    "t-SNE": lambda X: TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(X),
    "UMAP": lambda X: umap.UMAP(n_neighbors=15, random_state=0).fit_transform(X),
}

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (name, fn) in zip(axes, methods.items()):
    emb = fn(X)
    ax.scatter(emb[:, 0], emb[:, 1], c=t, cmap="hsv", s=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=11)
    print(f"{name} done")
fig.suptitle("三つ葉結び目の2次元展開 (色相=結び目に沿った位置t。ほどければ虹色の輪になる)", fontsize=12)
fig.tight_layout()
fig.savefig("fig2_embeddings.png", dpi=110, bbox_inches="tight")
print("saved fig2_embeddings.png")
