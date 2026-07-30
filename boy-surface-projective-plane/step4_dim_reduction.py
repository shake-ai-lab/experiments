"""実射影平面を次元削減にかけると何が起きるか。

RP^2 は自己交差なしでは3次元にすら入らない曲面なので、
3次元点群(ボーイ曲面)から出発すると自己交差が kNN グラフを汚染してしまう。
そこでまず「自己交差のない本物の RP^2」をヴェロネーゼ埋め込みで6次元に作る:
    v(x,y,z) = (x², y², z², √2xy, √2yz, √2zx),  (x,y,z)は単位球面上
p と -p が同じ点に写るので、対蹠点の同一視がこの写像に組み込まれている。

実験:
  (A) RP^2 / 球面 / 円板 を PCA, Isomap, LLE, t-SNE で2次元化。
      偽近傍率(埋め込みで近いのに本来遠い)と迷子率(本来近いのに埋め込みで散る)を測る
  (B) メビウスの帯を「ひねって作る」(前回の記事の構成) vs「貼って作る」(対蹠点同一視)。
      同じ位相なのに、Isomap の嘘のつき方が変わるかを見る
  (C) ボーイ曲面の3D点群の kNN 汚染率 (自己交差が作る偽の近傍)
  (D) 向き付け可能性の数値判定: 局所接平面の向きを近傍グラフで伝播させ、
      「向きの二重被覆」の連結成分数を数える (向き付け可能なら2、不可能なら1)"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from boy_common import boy, param_dist, sample_disk
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})
rng = np.random.default_rng(0)
N = 1500
S2 = np.sqrt(2)


def qmap(P):
    """中心対称な点集合の対蹠点同一視 p ~ -p を実現する2次写像 (ヴェロネーゼ型)。"""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.c_[x**2, y**2, z**2, S2 * x * y, S2 * y * z, S2 * z * x]


def sphere_points(n, seed):
    r = np.random.default_rng(seed)
    P = r.normal(size=(n, 3))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


# ================= (A) RP^2 / 球面 / 円板 の埋め込みグリッド =================
P_rp2 = sphere_points(N, 1)
X_rp2 = qmap(P_rp2)                  # RP^2 (6次元、自己交差なし)
P_sph = sphere_points(N, 2)
X_sph = P_sph                        # 球面 (3次元)
r_d = np.sqrt(rng.uniform(0, 1, N))
a_d = rng.uniform(0, 2 * np.pi, N)
X_dsk = np.c_[r_d * np.cos(a_d), r_d * np.sin(a_d), np.zeros(N)]  # 円板


def intrinsic_dist(name, i, j):
    if name == "rp2":
        return np.arccos(np.clip(np.abs(np.sum(P_rp2[i] * P_rp2[j], axis=-1)), 0, 1))
    if name == "sphere":
        return np.arccos(np.clip(np.sum(P_sph[i] * P_sph[j], axis=-1), -1, 1))
    return np.linalg.norm(X_dsk[i] - X_dsk[j], axis=-1)


def true_knn(name, X, k=8):
    if name == "rp2":
        G = np.abs(P_rp2 @ P_rp2.T)
        np.fill_diagonal(G, -np.inf)
        return np.argsort(-G, axis=1)[:, :k]
    if name == "sphere":
        G = P_sph @ P_sph.T
        np.fill_diagonal(G, -np.inf)
        return np.argsort(-G, axis=1)[:, :k]
    return NearestNeighbors(n_neighbors=k + 1).fit(X_dsk).kneighbors(X_dsk)[1][:, 1:]


DIAM = {"rp2": np.pi / 2, "sphere": np.pi, "disk": 2.0}
datasets = [
    ("rp2", "実射影平面", X_rp2, np.abs(P_rp2[:, 2]), "viridis"),
    ("sphere", "球面", X_sph, P_sph[:, 2], "coolwarm"),
    ("disk", "円板", X_dsk, r_d, "viridis"),
]
methods = {
    "PCA": lambda X: PCA(2).fit_transform(X),
    "Isomap": lambda X: Isomap(n_neighbors=12, n_components=2).fit_transform(X),
    "LLE": lambda X: LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=0).fit_transform(X),
    "t-SNE": lambda X: TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(X),
}

print("=== (A) 2次元埋め込み: 偽近傍率 / 迷子率 ===")
print("偽近傍率 = 埋め込みの8近傍のうち、本来の距離が直径の30%より遠いもの")
print("迷子率   = 本来の8近傍のうち、埋め込みの24近傍にすら入らないもの (引き裂きの指標)")
fig, axes = plt.subplots(3, 4, figsize=(14.5, 11))
for i, (key, cname, X, col, cmap) in enumerate(datasets):
    tk = true_knn(key, X)
    for j, (mname, fn) in enumerate(methods.items()):
        emb = fn(X)
        idx = NearestNeighbors(n_neighbors=9).fit(emb).kneighbors(emb)[1]
        src = np.repeat(np.arange(N), 8)
        false_rate = (intrinsic_dist(key, src, idx[:, 1:].ravel()) > 0.3 * DIAM[key]).mean()
        idx24 = NearestNeighbors(n_neighbors=25).fit(emb).kneighbors(emb)[1][:, 1:]
        lost_rate = np.mean([len(set(tk[p]) - set(idx24[p])) / 8 for p in range(N)])
        ax = axes[i, j]
        ax.scatter(emb[:, 0], emb[:, 1], c=col, cmap=cmap, s=5)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{mname} / {cname}\n偽近傍 {false_rate*100:.1f}% / 迷子 {lost_rate*100:.1f}%", fontsize=11)
        print(f"  {cname} × {mname}: 偽近傍 {false_rate*100:.1f}%, 迷子 {lost_rate*100:.1f}%")
fig.suptitle("色: 実射影平面=|z| (明=キャップ)、球面=z、円板=中心からの距離", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig4_embeddings.png", dpi=100, bbox_inches="tight")
print("saved fig4_embeddings.png")

# ================= (B) ひねって作った帯 vs 貼って作った帯 =================
print("\n=== (B) 同じメビウスの帯を2通りの作り方で Isomap にかける ===")
NB2, R_B, H_B = 2000, 2.0, 0.45


def analyze_ring(emb, v, NBIN=40):
    c = emb - emb.mean(0)
    theta = np.arctan2(c[:, 1], c[:, 0])
    radius = np.hypot(c[:, 0], c[:, 1])
    bins = np.linspace(-np.pi, np.pi, NBIN + 1)
    agree, widths = np.full(NBIN, np.nan), np.full(NBIN, np.nan)
    for b in range(NBIN):
        m = (theta >= bins[b]) & (theta < bins[b + 1])
        if m.sum() < 8:
            continue
        side = np.sign(radius[m] - np.median(radius[m]))
        agree[b] = np.mean(side * np.sign(v[m]))
        widths[b] = np.percentile(radius[m], 95) - np.percentile(radius[m], 5)
    centers = (bins[:-1] + bins[1:]) / 2
    valid = ~np.isnan(agree)
    sgn = np.sign(agree[valid])
    flips = int((sgn[:-1] * sgn[1:] < 0).sum() + (sgn[-1] * sgn[0] < 0))
    return dict(centers=centers, agree=agree, widths=widths, flips=flips,
                mean_abs=float(np.nanmean(np.abs(agree))))


r1 = np.random.default_rng(11)
# (a) 貼らないただの帯 (対照): 中心対称にならないよう z 方向にずらして同じ2次写像を通す
th_a = r1.uniform(0, 2 * np.pi, NB2)
v_a = r1.uniform(-H_B, H_B, NB2)
X_a = qmap(np.c_[R_B * np.cos(th_a), R_B * np.sin(th_a), v_a + 3.0])
# (b) ひねって作ったメビウス (前回の記事と同じ構成、3次元)
u_b = r1.uniform(0, 2 * np.pi, NB2)
v_b = r1.uniform(-H_B, H_B, NB2)
X_b = np.c_[(R_B + v_b * np.cos(u_b / 2)) * np.cos(u_b),
            (R_B + v_b * np.cos(u_b / 2)) * np.sin(u_b),
            v_b * np.sin(u_b / 2)] + r1.normal(0, 0.01, (NB2, 3))
# (c) 貼って作ったメビウス: 円筒 (ひねりなし!) の対蹠点を2次写像で同一視
th_c = r1.uniform(0, np.pi, NB2)
v_c = r1.uniform(-H_B, H_B, NB2)
X_c = qmap(np.c_[R_B * np.cos(th_c), R_B * np.sin(th_c), v_c])

band_conditions = [
    ("貼らない帯 (アニュラス、対照)", X_a, v_a, "#2e6ee4"),
    ("ひねって作ったメビウス (前回の構成)", X_b, v_b, "#e4572e"),
    ("貼って作ったメビウス (対蹠点同一視)", X_c, v_c, "#7a28cb"),
]
fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.6))
for row, (bname, Xb, v, col) in enumerate(band_conditions):
    emb = Isomap(n_neighbors=12, n_components=2).fit_transform(Xb)
    res = analyze_ring(emb, v)
    wrel = res["widths"] / np.nanmedian(res["widths"])
    print(f"{bname}: 反転 {res['flips']}回, |一致度|平均 {res['mean_abs']:.2f}, "
          f"幅の絶対値 中央値 {np.nanmedian(res['widths']):.2f} 最小 {np.nanmin(res['widths']):.2f}")
    ax = axes[row, 0]
    ax.scatter(emb[:, 0], emb[:, 1], c=v, cmap="RdBu", s=5)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"Isomap: {bname}", fontsize=13.5)
    ax.set_aspect("equal")
    ax2 = axes[row, 1]
    valid = ~np.isnan(res["agree"])
    ax2.plot(res["centers"][valid], res["agree"][valid], "-o", ms=4, color=col, label="表裏の一致度")
    ax2.plot(res["centers"][valid], wrel[valid], "-s", ms=3, color="#888", label="帯の幅 (中央値=1)")
    ax2.axhline(0, color="#bbb", lw=1)
    ax2.set_ylim(-1.3, 1.9)
    ax2.set_xlabel("展開図上での角度")
    ax2.set_title(f"一致度の符号反転: {res['flips']}回, |一致度|平均 {res['mean_abs']:.2f}", fontsize=13)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(alpha=0.3)
fig.suptitle("同じメビウスの帯なのに、「ひねって作る」と「貼って作る」で Isomap の応答が変わる", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig5_band_two_ways.png", dpi=110, bbox_inches="tight")
print("saved fig5_band_two_ways.png")

# 貼って作った帯の幅がどれだけ潰れたかを、埋め込み前後の比で出す
emb_c = Isomap(n_neighbors=12, n_components=2).fit_transform(X_c)
emb_a = Isomap(n_neighbors=12, n_components=2).fit_transform(X_a)
res_c, res_a = analyze_ring(emb_c, v_c), analyze_ring(emb_a, v_a)
# 2次写像による幅方向のスケール(解析値): |dq(e_z)| = sqrt(2(|p|^2 + z^2))
w_int_c = 2 * H_B * 0.9 * np.sqrt(2 * R_B**2)          # 95-5percentile相当(0.9倍)
w_int_a = 2 * H_B * 0.9 * np.sqrt(2 * (R_B**2 + 9.0))
print(f"帯の幅の残存率(埋め込み幅/本来の幅): 貼らない帯 {np.nanmedian(res_a['widths'])/w_int_a:.2f}, "
      f"貼ったメビウス {np.nanmedian(res_c['widths'])/w_int_c:.3f}")

# t-SNE は貼って作った帯をどうするか
emb_t = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(X_c)
fig, ax = plt.subplots(figsize=(7.2, 7.2))
ax.scatter(emb_t[:, 0], emb_t[:, 1], c=v_c, cmap="RdBu", s=7)
ax.set_xticks([]), ax.set_yticks([])
ax.set_title("t-SNE は「貼って作ったメビウス」を破って開く\n(色: 帯の幅方向。白=中心線、濃い赤/青=縁)", fontsize=14)
fig.savefig("fig6_tsne_band.png", dpi=110, bbox_inches="tight")
print("saved fig6_tsne_band.png")

# ================= (C) ボーイ曲面3D点群の kNN 汚染 =================
print("\n=== (C) 3次元点群(ボーイ曲面)では kNN が自己交差で汚染される ===")
W3 = sample_disk(4000, seed=21)
X3 = boy(W3)
idx = NearestNeighbors(n_neighbors=9).fit(X3).kneighbors(X3)[1]
pdists = param_dist(W3[np.repeat(np.arange(len(X3)), 8)], W3[idx[:, 1:].ravel()])
contam = (pdists > 0.25).mean()
pts_hit = (param_dist(W3[idx[:, 1:]], W3[:, None]) > 0.25).any(axis=1).mean()
print(f"ボーイ曲面(3D immersion)の kNN: {contam*100:.1f}% が『別の枚』への短絡")
print(f"少なくとも1個の偽近傍を持つ点: {pts_hit*100:.1f}%")
Ps = sphere_points(4000, 22)
idxs = NearestNeighbors(n_neighbors=9).fit(Ps).kneighbors(Ps)[1]
gd = np.arccos(np.clip(np.sum(Ps[np.repeat(np.arange(4000), 8)] * Ps[idxs[:, 1:].ravel()], axis=1), -1, 1))
print(f"対照(球面、埋め込み): 測地的に遠い kNN {(gd > 0.5).mean()*100:.1f}%")

# ================= (D) 向き付け可能性の数値判定 =================
print("\n=== (D) 向きの二重被覆の連結成分数 (2=向き付け可能, 1=不可能) ===")


def orientation_cover(X, k=10):
    n = len(X)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[1]
    frames = np.empty((n, 2, X.shape[1]))
    for i in range(n):
        Q = X[idx[i]] - X[idx[i]].mean(0)
        frames[i] = np.linalg.svd(Q, full_matrices=False)[2][:2]
    edges = {(min(i, j), max(i, j)) for i in range(n) for j in idx[i, 1:]}
    rows, cols, br, bc = [], [], [], []
    for i, j in edges:
        s = np.linalg.det(frames[i] @ frames[j].T)
        if abs(s) < 1e-12:
            continue
        br.append(i), bc.append(j)
        rows += [2 * i, 2 * i + 1]
        cols += [2 * j, 2 * j + 1] if s > 0 else [2 * j + 1, 2 * j]
    n_base = connected_components(
        coo_matrix((np.ones(len(br)), (br, bc)), shape=(n, n)), directed=False)[0]
    n_cover = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n, 2 * n)), directed=False)[0]
    return n_base, n_cover


for name, X, expect in [
    ("円板", X_dsk, 2),
    ("球面", X_sph, 2),
    ("貼らない帯 (アニュラス)", X_a, 2),
    ("ひねって作ったメビウス", X_b, 1),
    ("貼って作ったメビウス", X_c, 1),
    ("実射影平面 (6次元)", X_rp2, 1),
]:
    nb_, nc_ = orientation_cover(X)
    verdict = "向き付け可能" if nc_ == 2 * nb_ else "向き付け不可能"
    print(f"  {name}: 台の成分 {nb_}, 二重被覆の成分 {nc_} → {verdict} (期待 {expect})")
