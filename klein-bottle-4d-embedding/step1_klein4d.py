"""クラインの壺は3次元では必ず自己交差するが、4次元なら交差しない——を実測する。
多様体学習シリーズ第5弾(メビウスの輪の続編)。

標準的な4次元埋め込み:
  x = (R + r cos v) cos u
  y = (R + r cos v) sin u
  z = r sin v cos(u/2)
  w = r sin v sin(u/2)
w軸を落として3次元に射影すると u=π の子午線で自己交差する。

定量化:
  4次元座標のkNNグラフで測地距離(表面に沿った距離)を計算し、
  「表面上では遠い(測地距離 > G0)のに空間内で接触する」点対を数える。
  3次元射影では最近接距離が0に落ち、4次元のままなら0にならないことを示す。
対照: トーラス(向き付け可能、w=0で済む=最初から3次元に収まる)。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 15, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})

N = 2400
R, r = 2.0, 1.0
G0 = 2.0    # 「表面上で遠い」とみなす測地距離の閾値
EPS = 0.10  # 「空間内で接触している」とみなす距離
rng = np.random.default_rng(0)


def make_klein(n=N, noise=0.01, seed=0):
    """クラインの壺の標準的な4次元埋め込み"""
    rg = np.random.default_rng(seed)
    u = rg.uniform(0, 2 * np.pi, n)
    v = rg.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v) * np.cos(u / 2)
    w = r * np.sin(v) * np.sin(u / 2)
    X4 = np.c_[x, y, z, w] + rg.normal(0, noise, (n, 4))
    return X4, u, v


def make_torus(n=N, noise=0.01, seed=1):
    """トーラス。同じ(R, r)で、wは恒等的に0(=3次元で足りる)"""
    rg = np.random.default_rng(seed)
    u = rg.uniform(0, 2 * np.pi, n)
    v = rg.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    w = np.zeros(n)
    X4 = np.c_[x, y, z, w] + rg.normal(0, noise, (n, 4))
    return X4, u, v


def geodesic_matrix(X, k=10):
    """kNNグラフ上の最短路で表面に沿った距離を近似"""
    A = kneighbors_graph(X, n_neighbors=k, mode="distance")
    A = A.maximum(A.T)
    ncomp = connected_components(A, directed=False)[0]
    assert ncomp == 1, f"グラフが非連結 (成分数 {ncomp})"
    return shortest_path(A, method="D", directed=False)


def far_pair_stats(P, G, g0=G0, eps=EPS):
    """測地距離 > g0 の点対について、空間内距離の統計を取る"""
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    far = G > g0
    iu = np.triu_indices_from(D, k=1)
    mask = far[iu]
    d_far = D[iu][mask]
    # 点ごとの「遠い相手との最近接距離」
    D_masked = np.where(far, D, np.inf)
    per_point_min = D_masked.min(1)
    n_contact = int((d_far < eps).sum())
    return dict(min=float(d_far.min()), n_contact=n_contact,
                per_point_min=per_point_min, n_far_pairs=int(mask.sum()))


print("=== 生成 ===")
Xk, uk, vk = make_klein()
Xt, ut, vt = make_torus()
print(f"クラインの壺: {N}点 (4次元) / トーラス: {N}点 (w=0)")
print(f"wの標準偏差: クライン {Xk[:, 3].std():.3f}, トーラス {Xt[:, 3].std():.3f}")

print("\n=== 測地距離 (4次元kNNグラフ, k=10) ===")
Gk = geodesic_matrix(Xk)
Gt = geodesic_matrix(Xt)
print(f"測地距離の最大値: クライン {Gk.max():.2f}, トーラス {Gt.max():.2f}")

print(f"\n=== 表面上で {G0} 以上離れた点対の、空間内での接近 (接触判定 < {EPS}) ===")
res = {}
res["torus3d"] = far_pair_stats(Xt[:, :3], Gt)
res["klein3d"] = far_pair_stats(Xk[:, :3], Gk)  # w軸を潰した3次元射影
res["klein4d"] = far_pair_stats(Xk, Gk)         # 4次元のまま
for name, label in [("torus3d", "トーラス (3次元)"),
                    ("klein3d", "クライン 3次元射影 (wを潰す)"),
                    ("klein4d", "クライン 4次元のまま")]:
    s = res[name]
    print(f"  {label}: 最近接距離 {s['min']:.4f}, 接触点対 {s['n_contact']}件"
          f" (遠い点対 {s['n_far_pairs']}組中)")

# 接触点対はどこにあるか
D3 = np.sqrt(((Xk[:, None, :3] - Xk[None, :, :3]) ** 2).sum(-1))
far = Gk > G0
iu = np.triu_indices(N, k=1)
contact = (D3[iu] < EPS) & far[iu]
ci, cj = iu[0][contact], iu[1][contact]
d4_contact = np.sqrt(((Xk[ci] - Xk[cj]) ** 2).sum(-1))
print(f"\n接触点対のuの位置: 平均 {np.mean((uk[ci] + uk[cj]) / 2):.3f} (π={np.pi:.3f}),"
      f" |u-π|の最大 {np.abs(np.r_[uk[ci], uk[cj]] - np.pi).max():.3f}")
print(f"接触点対の4次元距離: 最小 {d4_contact.min():.3f}, 平均 {d4_contact.mean():.3f}"
      f" (3次元では<{EPS}なのに)")
print(f"理論値 2r|sin v| との差: 最大 {np.abs(d4_contact - 2 * r * np.abs(np.sin(vk[ci]))).max():.3f}")

# 参考: サンプリング密度(最近傍距離の中央値)
from scipy.spatial import cKDTree

nn_k = cKDTree(Xk).query(Xk, k=2)[0][:, 1]
print(f"\n参考: 4次元クライン点群の最近傍距離の中央値 {np.median(nn_k):.3f}")

# ---- スケーリング: 点を増やすと3次元射影の最近接距離は0へ、4次元は下げ止まる ----
print("\n=== 点数を増やしたときの、遠い点対の最近接距離 (3シード平均) ===")
Ns = [600, 1200, 2400, 4800]
SEEDS = [10, 20, 30]
scale_res = {"3d": [], "4d": []}
for n in Ns:
    m3, m4 = [], []
    for sd in SEEDS:
        Xs, us, vs = make_klein(n=n, seed=sd + n)
        Gs = geodesic_matrix(Xs)
        m3.append(far_pair_stats(Xs[:, :3], Gs)["min"])
        m4.append(far_pair_stats(Xs, Gs)["min"])
    scale_res["3d"].append(m3)
    scale_res["4d"].append(m4)
    print(f"  N={n}: 3次元射影 {np.mean(m3):.4f} (個別 {['%.3f' % x for x in m3]}) "
          f"/ 4次元 {np.mean(m4):.4f}")
scale_res = {k: np.array(v) for k, v in scale_res.items()}

# ---- 図1: 遠い相手との最近接距離をuに沿って見る + まとめ + スケーリング(縦スタック) ----
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
ax = axes[0]
order_t, order_k = np.argsort(ut), np.argsort(uk)
ax.scatter(ut, res["torus3d"]["per_point_min"], s=6, color="#2e6ee4", alpha=0.5,
           label="トーラス (3次元)")
ax.scatter(uk, res["klein4d"]["per_point_min"], s=6, color="#2ea44f", alpha=0.5,
           label="クラインの壺 (4次元のまま)")
ax.scatter(uk, res["klein3d"]["per_point_min"], s=6, color="#e4572e", alpha=0.5,
           label="クラインの壺 (3次元射影)")
ax.axvline(np.pi, color="#888", lw=1, ls="--")
ax.annotate("u=πで0まで落ちる\n=別の場所同士が接触(自己交差)",
            xy=(np.pi, 0.05), xytext=(3.9, 0.62), fontsize=13,
            arrowprops=dict(arrowstyle="->", lw=1.8))
ax.set_xlabel("輪に沿った位置 u")
ax.set_ylabel("表面上で2.0以上離れた相手との最近接距離")
ax.set_title("各点から「表面上では遠い点」までの空間内距離", fontsize=15)
ax.legend(fontsize=12, loc="lower left", framealpha=0.95)
ax.grid(alpha=0.3)

ax = axes[1]
labels = ["トーラス\n(3次元)", "クラインの壺\n(3次元射影)", "クラインの壺\n(4次元のまま)"]
mins = [res["torus3d"]["min"], res["klein3d"]["min"], res["klein4d"]["min"]]
cols = ["#2e6ee4", "#e4572e", "#2ea44f"]
bars = ax.bar(labels, mins, color=cols, width=0.55)
for b, m, nc in zip(bars, mins, [res["torus3d"]["n_contact"],
                                 res["klein3d"]["n_contact"],
                                 res["klein4d"]["n_contact"]]):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.04,
            f"最近接 {m:.3f}\n接触 {nc}件", ha="center", fontsize=13)
ax.set_ylim(0, max(mins) * 1.35)
ax.set_ylabel("遠い点対の最近接距離")
ax.set_title(f"表面上で2.0以上離れた点対が、空間内でどこまで近づくか", fontsize=15)
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
for key, col, lab in [("4d", "#2ea44f", "4次元のまま"), ("3d", "#e4572e", "3次元射影")]:
    mean = scale_res[key].mean(1)
    ax.plot(Ns, mean, "-o", ms=7, color=col, label=lab)
    ax.fill_between(Ns, scale_res[key].min(1), scale_res[key].max(1),
                    color=col, alpha=0.18)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel("点の数 N")
ax.set_ylabel("遠い点対の最近接距離 (対数)")
ax.set_title("点を増やすほど3次元射影は0に向かい、4次元は下げ止まる", fontsize=15)
ax.annotate("真の交差がある\n→ 密に測るほど0へ", xy=(Ns[-1], scale_res["3d"].mean(1)[-1]),
            xytext=(1500, scale_res["3d"].mean(1)[-1] * 3.0), fontsize=13,
            arrowprops=dict(arrowstyle="->", lw=1.8))
ax.legend(fontsize=12)
ax.grid(alpha=0.3, which="both")
fig.suptitle("3次元に潰すと自己交差し、4次元のままなら交差しない", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig1_contact.png", dpi=110, bbox_inches="tight")
print("saved fig1_contact.png")

# ---- 図2: 3次元射影の見た目(色=w) 縦スタック ----
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(9.5, 11))
for i, (elev, azim, title) in enumerate([
        (28, -62, "クラインの壺の3次元射影 (色=捨てた第4軸w)"),
        (6, -92, "真横から見る: 内側の面(赤)と外側の面(青)が u=π の稜線で交差する")]):
    ax = fig.add_subplot(2, 1, i + 1, projection="3d")
    sc = ax.scatter(Xk[:, 0], Xk[:, 1], Xk[:, 2], c=Xk[:, 3], cmap="RdBu",
                    s=6, alpha=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=15, pad=0)
    ax.set_box_aspect((np.ptp(Xk[:, 0]), np.ptp(Xk[:, 1]), np.ptp(Xk[:, 2])))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.0, label="第4軸 w")
fig.subplots_adjust(left=0, right=0.95, top=0.97, bottom=0.0, hspace=0.02)
fig.savefig("fig0_projection.png", dpi=110, bbox_inches="tight")
print("saved fig0_projection.png")

# ---- インタラクティブ3D (plotly) ----
n3d = 4000
X3d, u3d, v3d = make_klein(n=n3d, noise=0.0, seed=7)
fig3d = go.Figure(
    go.Scatter3d(
        x=X3d[:, 0], y=X3d[:, 1], z=X3d[:, 2], mode="markers",
        marker=dict(size=2.8, color=X3d[:, 3], colorscale="RdBu",
                    colorbar=dict(title="第4軸 w")),
        hoverinfo="skip",
    )
)
fig3d.update_layout(
    title="クラインの壺の3次元射影 (ドラッグで回転)。色は捨てた第4軸w——"
          "交差している場所で青と赤が重なる=4次元では離れている",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    margin=dict(l=0, r=0, t=60, b=0), height=650,
)
fig3d.write_html("klein-bottle-4d-embedding.html", include_plotlyjs="cdn", full_html=True)
print("saved klein-bottle-4d-embedding.html")

np.savez_compressed("klein_data.npz", Xk=Xk, uk=uk, vk=vk, Xt=Xt, ut=ut, vt=vt)
np.save("geo_klein.npy", Gk.astype(np.float32))
np.save("geo_torus.npy", Gt.astype(np.float32))
print("saved klein_data.npz / geo_*.npy")
