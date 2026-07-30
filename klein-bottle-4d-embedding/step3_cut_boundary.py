"""クラインの壺を「赤道」で切ると何が出てくるか。
チューブ周りの座標 v = ±π/2 の2本の円に沿って切ることを考える。
トーラスなら切り口は独立した2本の円。
クラインの壺では u が一周すると v の符号が反転して貼り合わさっているので、
「2本」の円が実はつながった1本の円になっているはず(メビウスの輪の縁と同じ構図)。
点群のkNNグラフの連結成分数と、成分ごとの周長で確かめる。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 15, "axes.labelsize": 13})

R, r = 2.0, 1.0
N_PER = 700  # 各切り口円あたりの点数
rng = np.random.default_rng(5)


def cut_curves(surface, noise=0.005):
    """v = +π/2 と v = -π/2 の円周上の点を生成"""
    pts, us, vs = [], [], []
    for v0 in (np.pi / 2, -np.pi / 2):
        # 円周に沿って等間隔(+微小ジッタ)。ランダムだと間隙でグラフが千切れる
        u = np.linspace(0, 2 * np.pi, N_PER, endpoint=False)
        u = (u + rng.normal(0, 0.002, N_PER)) % (2 * np.pi)
        v = np.full(N_PER, v0)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        if surface == "klein":
            z = r * np.sin(v) * np.cos(u / 2)
            w = r * np.sin(v) * np.sin(u / 2)
            X = np.c_[x, y, z, w]
        else:
            X = np.c_[x, y, r * np.sin(v)]
        pts.append(X)
        us.append(u)
        vs.append(v)
    X = np.vstack(pts) + rng.normal(0, noise, (2 * N_PER, len(pts[0][0])))
    return X, np.concatenate(us), np.concatenate(vs)


def analyze(X, k=4):
    A = kneighbors_graph(X, n_neighbors=k, mode="distance")
    A = A.maximum(A.T)
    ncomp, labels = connected_components(A, directed=False)
    lengths = []
    for c in range(ncomp):
        idx = np.flatnonzero(labels == c)
        G = shortest_path(A[idx][:, idx], method="D", directed=False)
        lengths.append(2 * G.max())  # 円周 ≒ 最遠グラフ距離の2倍
    return ncomp, labels, lengths


results = {}
for surface, label in [("torus", "トーラス"), ("klein", "クラインの壺")]:
    X, u, v = cut_curves(surface)
    ncomp, labels, lengths = analyze(X)
    results[surface] = (X, u, v, ncomp, labels, lengths)
    ls = ", ".join(f"{l:.2f}" for l in lengths)
    print(f"{label}: 切り口の本数 {ncomp}, 各周長 [{ls}]")
print(f"(参考) 理論値: トーラス 2πR = {2*np.pi*R:.2f} が2本, "
      f"クライン 4π√(R²+r²/4) = {4*np.pi*np.sqrt(R**2+r**2/4):.2f} が1本")

# 背景用の面の点
bg = {}
for surface in ("torus", "klein"):
    u = rng.uniform(0, 2 * np.pi, 1500)
    v = rng.uniform(0, 2 * np.pi, 1500)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = (r * np.sin(v) * np.cos(u / 2)) if surface == "klein" else r * np.sin(v)
    bg[surface] = np.c_[x, y, z]

# ---- 図: 縦スタック(トーラス上段 / クライン下段) ----
fig = plt.figure(figsize=(9.5, 11.5))
for i, (surface, label) in enumerate([("torus", "トーラス"), ("klein", "クラインの壺")]):
    X, u, v, ncomp, labels, lengths = results[surface]
    ax = fig.add_subplot(2, 1, i + 1, projection="3d")
    B = bg[surface]
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], color="#999", s=2, alpha=0.12)
    if surface == "torus":
        for c, col in zip(range(ncomp), ["#2e6ee4", "#e4572e"]):
            m = labels == c
            ax.scatter(X[m, 0], X[m, 1], X[m, 2], color=col, s=8,
                       label=f"切り口{c+1} (長さ {lengths[c]:.1f})")
        title = f"トーラス: 切り口は {ncomp} 本の円"
    else:
        # 1本の輪であることを、始点からのグラフ距離のグラデーションで見せる
        A = kneighbors_graph(X, n_neighbors=4, mode="distance")
        A = A.maximum(A.T)
        dist0 = shortest_path(A, method="D", directed=False, indices=0)
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=dist0, cmap="viridis", s=8)
        fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.0, label="始点からの道のり")
        title = (f"クラインの壺: 切り口は {ncomp} 本につながった円 "
                 f"(長さ {lengths[0]:.1f} ≒ 2倍)")
    ax.set_title(title, fontsize=15, pad=0)
    ax.view_init(elev=32, azim=-58)
    ax.set_box_aspect((np.ptp(B[:, 0]), np.ptp(B[:, 1]), np.ptp(B[:, 2]) * 1.6))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    if surface == "torus":
        ax.legend(fontsize=12, loc="upper right")
fig.suptitle("同じ場所(v=±π/2)で切っても、切り口の本数が違う", fontsize=16)
fig.subplots_adjust(left=0, right=0.95, top=0.93, bottom=0.0, hspace=0.06)
fig.savefig("fig3_cut_boundary.png", dpi=110, bbox_inches="tight")
print("saved fig3_cut_boundary.png")
