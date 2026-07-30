"""ボーイ曲面を Bryant-Kusner 表示で生成し、
(a) 境界の対蹠点同一視 f(w)=f(-w) が成り立つこと
(b) 内部の点は同一視されないこと
(c) メッシュのオイラー標数 V-E+F が理論値と一致すること
を数値で確認する。多様体学習シリーズ第5弾(メビウスの輪の続き)。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from boy_common import POLE_RADIUS, boy, sample_disk

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

# ---- (a) 境界での対蹠点同一視 ----
th = np.linspace(0, np.pi, 4001)
wb = np.exp(1j * th)
gap_boundary = np.linalg.norm(boy(wb) - boy(-wb), axis=1)
print("=== 境界 |w|=1 での f(w) と f(-w) の距離 ===")
print(f"  最大 {gap_boundary.max():.2e}, 平均 {gap_boundary.mean():.2e}  (機械精度で一致 = 対蹠点が貼り合う)")

# ---- (b) 内部では同一視されない ----
for r in (0.3, 0.5, 0.8):
    wi = r * np.exp(1j * th)
    gap = np.linalg.norm(boy(wi) - boy(-wi), axis=1)
    print(f"  内部 |w|={r}: f(w) と f(-w) の距離 最小 {gap.min():.3f} (0にならない = 内部は貼り合わない)")

# ---- (c) オイラー標数 V-E+F ----
# パラメータ円板を極座標メッシュで三角形分割し、
# 「縁の対蹠点を同一視する/しない」「中心まで含める/アニュラスにする」を
# 切り替えて4種類の曲面を作る。


def polar_mesh_chi(m=24, n=48, inner=0, identify_outer=False):
    """極座標メッシュの V-E+F。inner>0 でアニュラス、identify_outer で縁の対蹠点同一視。"""
    assert n % 2 == 0

    def key(i, j):
        j = j % n
        if identify_outer and i == m:  # 縁: (m,j) と (m,j+n/2) を同じ頂点にする
            j = min(j, (j + n // 2) % n)
        return (i, j)

    tris = []
    i0 = inner
    if i0 == 0:  # 中心の1点から放射状に
        tris += [(("c",), key(1, j), key(1, j + 1)) for j in range(n)]
        i0 = 1
    for i in range(i0, m):
        for j in range(n):
            a, b = key(i, j), key(i, j + 1)
            c, d = key(i + 1, j), key(i + 1, j + 1)
            tris += [(a, b, c), (b, d, c)]
    V = len({v for t in tris for v in t})
    E = len({frozenset(p) for t in tris for p in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]})
    F = len(tris)
    return V - E + F


def sphere_mesh_chi(m=24, n=48):
    """対照: 球面(UVメッシュ、両極は1点)の V-E+F。"""
    def key(i, j):
        if i == 0:
            return ("N",)
        if i == m:
            return ("S",)
        return (i, j % n)

    tris = []
    for i in range(m):
        for j in range(n):
            a, b = key(i, j), key(i, j + 1)
            c, d = key(i + 1, j), key(i + 1, j + 1)
            for t in [(a, b, c), (b, d, c)]:
                if len(set(t)) == 3:
                    tris.append(t)
    V = len({v for t in tris for v in t})
    E = len({frozenset(p) for t in tris for p in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]})
    return V - E + len(tris)


print("\n=== オイラー標数 V-E+F (メッシュから直接数える) ===")
cases = [
    ("円板 (縁を貼らない)", dict(inner=0, identify_outer=False), 1),
    ("アニュラス (帯、縁を貼らない)", dict(inner=15, identify_outer=False), 0),
    ("メビウスの帯 (アニュラスの外縁の対蹠点を貼る)", dict(inner=15, identify_outer=True), 0),
    ("実射影平面 (円板の縁の対蹠点を貼る)", dict(inner=0, identify_outer=True), 1),
]
for name, kw, expect in cases:
    chi = polar_mesh_chi(**kw)
    print(f"  {name}: χ = {chi} (理論値 {expect})")
chi_s = sphere_mesh_chi()
print(f"  球面 (対照): χ = {chi_s} (理論値 2)")
print("  貼り合わせの検算: χ(メビウス) + χ(円板) - χ(円周) = 0 + 1 - 0 = 1 = χ(射影平面)")

# ---- 点群の生成と保存 ----
N = 40000
W = sample_disk(N, seed=0)
X = boy(W)
np.savez("boy_cloud.npz", W=W, X=X)
print(f"\n点群 {N} 点を生成 (3D範囲: x {X[:,0].min():.2f}..{X[:,0].max():.2f}, "
      f"y {X[:,1].min():.2f}..{X[:,1].max():.2f}, z {X[:,2].min():.2f}..{X[:,2].max():.2f})")
print(f"分母の零点(三重点の予言): 半径 {POLE_RADIUS:.6f}, 角度 0°, 120°, 240°")

# ---- 図1: 3方向からのビュー(色 = パラメータ円板の中心からの距離 |w|) ----
fig = plt.figure(figsize=(14.5, 5.4))
views = [(18, -60, "斜め上から"), (0, -90, "真横から"), (88, -90, "真上から")]
for k, (elev, azim, label) in enumerate(views):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.abs(W), cmap="viridis", s=1.2, alpha=0.6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(label, fontsize=14)
    ax.set_box_aspect((1, 1, 0.9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
cb = fig.colorbar(sc, ax=fig.axes, shrink=0.7, pad=0.02)
cb.set_label("パラメータ円板の中心からの距離 |w|", fontsize=12)
fig.suptitle("ボーイ曲面 (Bryant-Kusner 表示、40000点)。裏表のない閉曲面を3次元に押し込んだ姿", fontsize=15)
fig.savefig("fig1_surface.png", dpi=110, bbox_inches="tight")
print("saved fig1_surface.png")
