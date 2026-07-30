"""テセラクトの16頂点を次元削減(PCA/MDS/Isomap/t-SNE)で2次元に落とし、
4次元の距離構造がどれだけ保たれるかを測る。
- 指標: Kruskalのstress-1(最適スケーリング後)、距離のPearson/Spearman相関
- 4D距離は 2√k (k=異なる座標の数, k=1..4) の4クラスしかない。
  埋め込み後にこの4クラスがどれだけ分離して残るかも見る
"""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})


def hypercube(d):
    V = np.array(list(itertools.product([-1, 1], repeat=d)), dtype=float)
    E = [(i, j) for i in range(len(V)) for j in range(i + 1, len(V))
         if np.sum(V[i] != V[j]) == 1]
    return V, E


def stress1(D_hi, D_lo):
    """Kruskal stress-1。埋め込みのスケール b は最適化してから測る"""
    b = np.dot(D_hi, D_lo) / np.dot(D_lo, D_lo)
    return np.sqrt(np.sum((D_hi - b * D_lo) ** 2) / np.sum(D_hi ** 2))


V4, E4 = hypercube(4)
D4 = pdist(V4)
kclass = np.round((D4 / 2) ** 2).astype(int)  # 異なる座標の数 k
print("=== 4Dでの距離構造 ===")
for k in range(1, 5):
    print(f"  k={k}: 距離 {2*np.sqrt(k):.4f} x {np.sum(kclass==k)}ペア")

methods = {
    "PCA": lambda X: PCA(2, random_state=0).fit_transform(X),
    "MDS": lambda X: MDS(2, random_state=0, normalized_stress=False,
                         n_init=8, max_iter=1000).fit_transform(X),
    "Isomap": lambda X: Isomap(n_neighbors=4, n_components=2).fit_transform(X),
    "t-SNE": lambda X: TSNE(2, perplexity=5, random_state=0, init="pca",
                            learning_rate="auto").fit_transform(X),
}

print("\n=== 16頂点を2次元へ ===")
print(f"{'手法':8s} {'stress-1':>9s} {'Pearson':>8s} {'Spearman':>9s}")
embs, stats = {}, {}
for name, fn in methods.items():
    emb = fn(V4)
    D2 = pdist(emb)
    s = stress1(D4, D2)
    pr = pearsonr(D4, D2)[0]
    sr = spearmanr(D4, D2)[0]
    embs[name], stats[name] = emb, (s, pr, sr)
    print(f"{name:8s} {s:9.4f} {pr:8.4f} {sr:9.4f}")

# PCAの固有値: 超立方体頂点は等方的 → 全固有値が等しいはず
ev = PCA(4).fit(V4).explained_variance_
print(f"\nPCAの固有値(4成分): {np.round(ev, 4)}  ← 全部同じ = 選ぶ軸に根拠がない")

# 対蹠点(k=4, 4D距離4.0)ペアが2Dでどれだけ近くに置かれたか
print("\n=== 距離クラス別の平均埋め込み距離(4D距離2.0のクラス平均=1に正規化) ===")
for name, emb in embs.items():
    D2 = pdist(emb)
    base = D2[kclass == 1].mean()
    rel = [D2[kclass == k].mean() / base for k in range(1, 5)]
    ideal = [np.sqrt(k) for k in range(1, 5)]
    print(f"  {name:8s} " + " ".join(f"k={k}:{r:.3f}(理想{i:.3f})"
                                     for k, r, i in zip(range(1, 5), rel, ideal)))
    # クラス間の逆転: 4D で遠いペアが 2D で近いペアより手前に来た割合
    inv = 0
    tot = 0
    for a in range(len(D4)):
        for b in range(a + 1, len(D4)):
            if kclass[a] != kclass[b]:
                tot += 1
                if (kclass[a] < kclass[b]) != (D2[a] < D2[b]):
                    inv += 1
    print(f"           順序の逆転: {inv}/{tot} ペア組 ({inv/tot:.1%})")

# ---- 図2: 4手法の埋め込み ----
fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for ax, (name, emb) in zip(axes.ravel(), embs.items()):
    s, pr, sr = stats[name]
    for i, j in E4:
        ax.plot(emb[[i, j], 0], emb[[i, j], 1], color="#bbb", lw=1.2, zorder=1)
    ax.scatter(emb[:, 0], emb[:, 1], c=V4[:, 3], cmap="RdBu", vmin=-1.45, vmax=1.45,
               s=90, zorder=2, edgecolors="#333", linewidths=0.6)
    ax.set_title(f"{name}\nstress-1 {s:.3f} / 距離相関(Spearman) {sr:.3f}", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
fig.suptitle("テセラクトの16頂点を2次元へ。灰色の線は4次元での辺(32本)\n"
             "点の色は第4座標 w (青=+1, 赤=-1)", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig2_embeddings.png", dpi=110, bbox_inches="tight")
print("\nsaved fig2_embeddings.png")

np.savez("embeddings.npz", **embs)
