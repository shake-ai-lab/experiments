"""次元を上げていったときの劣化を測る。
立方体(3D, 8頂点) → テセラクト(4D, 16頂点) → … → 9次元超立方体(512頂点) を
2次元に落とし、stress-1 と距離相関(Spearman)が次元とともにどう悪化するかを測る。
あわせて、頂点間距離の変動係数CV(次元の呪いの指標)も測り、
curse-of-dimensionality-experiment の結果と接続する。
"""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 15, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12})


def hypercube_vertices(d):
    return np.array(list(itertools.product([-1, 1], repeat=d)), dtype=float)


def stress1(D_hi, D_lo):
    b = np.dot(D_hi, D_lo) / np.dot(D_lo, D_lo)
    return np.sqrt(np.sum((D_hi - b * D_lo) ** 2) / np.sum(D_hi ** 2))


DIMS = list(range(3, 10))
rows = []
print(f"{'d':>2s} {'頂点':>4s} | {'MDS stress':>10s} {'MDS ρ':>7s} | "
      f"{'PCA stress':>10s} {'PCA ρ':>7s} | {'t-SNE ρ':>8s} | {'距離CV':>7s}")
for d in DIMS:
    V = hypercube_vertices(d)
    D = pdist(V)
    cv = D.std() / D.mean()
    e_mds = MDS(2, random_state=0, normalized_stress=False,
                n_init=4, max_iter=500).fit_transform(V)
    e_pca = PCA(2, random_state=0).fit_transform(V)
    e_tsne = TSNE(2, perplexity=min(30, len(V) - 2) if len(V) > 8 else 5,
                  random_state=0, init="pca", learning_rate="auto").fit_transform(V)
    s_mds, r_mds = stress1(D, pdist(e_mds)), spearmanr(D, pdist(e_mds))[0]
    s_pca, r_pca = stress1(D, pdist(e_pca)), spearmanr(D, pdist(e_pca))[0]
    r_tsne = spearmanr(D, pdist(e_tsne))[0]
    rows.append((d, len(V), s_mds, r_mds, s_pca, r_pca, r_tsne, cv))
    print(f"{d:2d} {len(V):4d} | {s_mds:10.4f} {r_mds:7.4f} | "
          f"{s_pca:10.4f} {r_pca:7.4f} | {r_tsne:8.4f} | {cv:7.4f}")

rows = np.array(rows)
np.savetxt("sweep_results.txt", rows,
           header="d n_vertices mds_stress mds_rho pca_stress pca_rho tsne_rho dist_cv",
           fmt="%.6f")

# ---- 図3: 縦スタック2段 ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 11))
d_ax = rows[:, 0]
ax1.plot(d_ax, rows[:, 2], "-o", ms=7, lw=2.2, color="#e4572e", label="MDS stress-1")
ax1.plot(d_ax, rows[:, 4], "-s", ms=7, lw=2.2, color="#2e6ee4", label="PCA stress-1")
ax1.set_ylabel("stress-1 (小さいほど距離構造を保持)")
ax1.set_title("超立方体を2次元に落としたときの歪みは、次元とともに単調に増える")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(bottom=0)
ax1.annotate(f"立方体(3D): {rows[0,2]:.3f}", xy=(3, rows[0, 2]),
             xytext=(3.2, rows[0, 2] - 0.06), fontsize=12,
             arrowprops=dict(arrowstyle="->", lw=1.4))
ax1.annotate(f"テセラクト(4D): {rows[1,2]:.3f}", xy=(4, rows[1, 2]),
             xytext=(4.3, rows[1, 2] - 0.05), fontsize=12,
             arrowprops=dict(arrowstyle="->", lw=1.4))

ax2.plot(d_ax, rows[:, 3], "-o", ms=7, lw=2.2, color="#e4572e", label="MDS")
ax2.plot(d_ax, rows[:, 5], "-s", ms=7, lw=2.2, color="#2e6ee4", label="PCA")
ax2.plot(d_ax, rows[:, 6], "-^", ms=7, lw=2.2, color="#3aa76d", label="t-SNE")
ax2b = ax2.twinx()
ax2b.plot(d_ax, rows[:, 7], ":d", ms=7, lw=2.2, color="#888", label="距離のCV (右軸)")
ax2b.set_ylabel("頂点間距離の変動係数CV", color="#666")
ax2b.set_ylim(bottom=0)
ax2.set_xlabel("超立方体の次元 d")
ax2.set_ylabel("距離のSpearman相関 ρ")
ax2.set_title("距離の順序も次第に壊れる。灰色の点線は「次元の呪い」の進行度")
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="center right")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig3_sweep.png", dpi=110, bbox_inches="tight")
print("\nsaved fig3_sweep.png")
