"""4次元のクラインの壺を次元削減で2次元に展開すると何が起きるか。
トーラス(向き付け可能な閉曲面)との対照で、PCA / Isomap / LLE / t-SNEを比較。
評価:
  (a) 輪に沿った位置uの復元度(円環相関)
  (b) 偽の近傍率: 埋め込みで近いのに表面上(測地距離)では遠い点の割合
おまけ: Isomapに3次元を許したら自己交差を再現するのか。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

G0 = 2.0
d = np.load("klein_data.npz")
geo = {"クラインの壺": np.load("geo_klein.npy"),
       "トーラス": np.load("geo_torus.npy")}
DATA = {"トーラス": (d["Xt"][:, :3], d["ut"], d["vt"]),      # wは恒等0なので3次元
        "クラインの壺": (d["Xk"], d["uk"], d["vk"])}          # 4次元のまま渡す

methods = {
    "PCA": lambda X: PCA(2).fit_transform(X),
    "Isomap": lambda X: Isomap(n_neighbors=10, n_components=2).fit_transform(X),
    "LLE": lambda X: LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                                            random_state=0).fit_transform(X),
    "t-SNE": lambda X: TSNE(2, perplexity=30, random_state=0,
                            init="pca").fit_transform(X),
}


def circ_corr(emb, u):
    c = emb - emb.mean(0)
    phi = np.arctan2(c[:, 1], c[:, 0])
    return max(abs(np.mean(np.exp(1j * (s * phi - u)))) for s in (1, -1))


def false_neighbor_rate(emb, G, g0=G0, k=8):
    """埋め込み上のk近傍のうち、表面上では g0 より遠い点の割合"""
    emb_s = (emb - emb.mean(0)) / emb.std(0)
    _, nn = cKDTree(emb_s).query(emb_s, k=k + 1)
    return float((G[np.arange(len(emb))[:, None], nn[:, 1:]] > g0).mean())


def tear_rate(emb, G, k=8, k_emb=24):
    """表面上のk近傍のうち、埋め込みのk_emb近傍に入っていない点の割合(引き裂き)"""
    n = len(emb)
    true_nn = np.argsort(G, axis=1)[:, 1:k + 1]
    emb_s = (emb - emb.mean(0)) / emb.std(0)
    _, enn = cKDTree(emb_s).query(emb_s, k=k_emb + 1)
    keep = np.zeros(n)
    for i in range(n):
        keep[i] = np.isin(true_nn[i], enn[i, 1:]).mean()
    return float(1 - keep.mean())


fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.6))
scores = {}
for i, (sname, (X, u, v)) in enumerate(DATA.items()):
    for j, (mname, fn) in enumerate(methods.items()):
        emb = fn(X)
        cc = circ_corr(emb, u)
        fr = false_neighbor_rate(emb, geo[sname])
        tr = tear_rate(emb, geo[sname])
        scores[(sname, mname)] = (cc, fr, tr)
        ax = axes[i, j]
        ax.scatter(emb[:, 0], emb[:, 1], c=np.cos(v), cmap="coolwarm", s=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{mname} / {sname}\n輪の復元 {cc:.3f} / 偽近傍 {fr*100:.1f}%",
                     fontsize=11)
        print(f"{sname} {mname}: circ-corr {cc:.4f}, 偽近傍率 {fr*100:.2f}%, "
              f"引き裂き率 {tr*100:.2f}%")
fig.suptitle("閉曲面を2次元に展開する (色 = チューブ周りの位置 cos v)", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig2_embed2d.png", dpi=100, bbox_inches="tight")
print("saved fig2_embed2d.png")

# ---- おまけ: Isomapに3次元を許すと、自己交差を再現するのか ----
print("\n=== Isomap 4次元→3次元: 出力の中で遠い点対はどこまで近づくか ===")
Xk, Gk = DATA["クラインの壺"][0], geo["クラインの壺"]
emb3 = Isomap(n_neighbors=10, n_components=3).fit_transform(Xk)
# 入力(4次元)とスケールを揃える: 測地距離の中央値比で正規化
D_in = np.sqrt(((Xk[:, None] - Xk[None]) ** 2).sum(-1))
D_out = np.sqrt(((emb3[:, None] - emb3[None]) ** 2).sum(-1))
iu = np.triu_indices(len(Xk), k=1)
scale = np.median(D_in[iu]) / np.median(D_out[iu])
D_out *= scale
far = Gk[iu] > G0
min_in, min_out = D_in[iu][far].min(), D_out[iu][far].min()
n_eps = int((D_out[iu][far] < 0.10).sum())
print(f"  入力(4次元): 遠い点対の最近接 {min_in:.3f}")
print(f"  Isomap 3次元出力(スケール補正後): 遠い点対の最近接 {min_out:.3f}, "
      f"接触(<0.10) {n_eps}件")
# Isomapが作った接触は、w落とし射影と同じ u=π 付近なのか
uk = d["uk"]
c_mask = (D_out[iu] < 0.10) & far
ci, cj = iu[0][c_mask], iu[1][c_mask]
uu = np.r_[uk[ci], uk[cj]]
print(f"  Isomapの接触点対のuの位置: 平均 {uu.mean():.3f} (π={np.pi:.3f}), "
      f"|u-π|の最大 {np.abs(uu - np.pi).max():.3f}")
np.savez("embed2d_scores.npz",
         **{f"{a}|{b}": np.array(val) for (a, b), val in scores.items()},
         isomap3d=np.array([min_in, min_out, n_eps]))
