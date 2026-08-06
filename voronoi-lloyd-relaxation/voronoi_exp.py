# ボロノイ図: ランダムに撒いた点を「自分の縄張りの重心」へ動かし続けると、蜂の巣になる
#
# 実験1: Lloyd法(ボロノイ緩和)を回し、セル面積のばらつきと辺数の分布がどう動くか
# 実験2: ランダムな点のボロノイ図で、1セルあたりの平均辺数は本当に6になるか
# 実験3: 距離の測り方(ユークリッド/マンハッタン/チェビシェフ)を変えると縄張りはどう変わるか
#
# 実行: C:/bin/Python/python310-64/python.exe -u voronoi_exp.py

import json
from collections import Counter

import numpy as np
from scipy.spatial import Voronoi, cKDTree

OUT = {}
rng = np.random.default_rng(20260806)


def lloyd(points, steps, grid=600, record=()):
    """密な格子上で最近傍を割り当て、各セルの重心へ点を動かす(=一様分布上のk-means)。
    トーラス境界にはせず、単位正方形の中だけで扱う。"""
    g = (np.arange(grid) + 0.5) / grid
    gy, gx = np.meshgrid(g, g, indexing="ij")
    G = np.column_stack([gx.ravel(), gy.ravel()])
    hist = []
    snaps = {}
    p = points.copy()
    for t in range(steps + 1):
        idx = cKDTree(p).query(G)[1]
        counts = np.bincount(idx, minlength=len(p)).astype(float)
        area = counts / len(G)
        sx = np.bincount(idx, weights=G[:, 0], minlength=len(p))
        sy = np.bincount(idx, weights=G[:, 1], minlength=len(p))
        cv = float(area.std() / area.mean())
        hist.append(cv)
        if t in record:
            snaps[t] = p.copy()
        if t == steps:
            break
        nz = counts > 0
        p[nz] = np.column_stack([sx[nz] / counts[nz], sy[nz] / counts[nz]])
    return p, np.array(hist), snaps


def edge_counts(points):
    """各点が何本の辺を持つか(=ドロネー隣接数)。外周に触れるセルは除く。"""
    v = Voronoi(points)
    deg = Counter()
    for a, b in v.ridge_points:
        deg[a] += 1
        deg[b] += 1
    # 無限に伸びるセル(外周)を除く
    infinite = set()
    for i, reg in enumerate(v.point_region):
        r = v.regions[reg]
        if len(r) == 0 or -1 in r:
            infinite.add(i)
        else:
            pts = v.vertices[r]
            if pts.min() < -0.02 or pts.max() > 1.02:
                infinite.add(i)
    inner = [i for i in range(len(points)) if i not in infinite]
    return np.array([deg[i] for i in inner]), inner


# ================================================================ 実験1
N = 400
STEPS = 40
print(f"[実験1] {N}点をランダムに撒き、Lloyd法を{STEPS}回", flush=True)
pts0 = rng.random((N, 2))
rec = {0, 1, 2, 5, 10, 20, 40}
pf, cv_hist, snaps = lloyd(pts0, STEPS, grid=600, record=rec)
print("  セル面積の変動係数(標準偏差/平均)")
for t in sorted(rec):
    print(f"    {t:>3} 回: {cv_hist[t]:.4f}")
deg0, _ = edge_counts(pts0)
degf, _ = edge_counts(pf)
print(f"\n  辺の数: 初期 平均 {deg0.mean():.3f} / 6角形の割合 {(deg0==6).mean()*100:.1f}%"
      f"  ({len(deg0)}セル)")
print(f"          収束後 平均 {degf.mean():.3f} / 6角形の割合 {(degf==6).mean()*100:.1f}%"
      f"  ({len(degf)}セル)")
OUT["exp1"] = {"N": N, "steps": STEPS, "cv": cv_hist.tolist(),
               "deg0": np.bincount(deg0, minlength=13).tolist(),
               "degf": np.bincount(degf, minlength=13).tolist(),
               "deg0_mean": float(deg0.mean()), "degf_mean": float(degf.mean()),
               "hex0": float((deg0 == 6).mean()), "hexf": float((degf == 6).mean())}
np.save("pts0.npy", pts0)
np.save("ptsf.npy", pf)
np.save("snaps.npy", np.array([snaps[t] for t in sorted(rec)]))

# ================================================================ 実験2
print("\n[実験2] ランダム点のボロノイ図で、平均辺数は6になるか", flush=True)
rows = []
for n in [100, 300, 1000, 3000, 10000]:
    ms = []
    for s in range(5):
        p = np.random.default_rng(s * 97 + n).random((n, 2))
        d, inner = edge_counts(p)
        ms.append(d.mean())
    rows.append({"n": n, "mean": float(np.mean(ms)), "sd": float(np.std(ms))})
    print(f"  点の数 {n:>6}: 平均辺数 {np.mean(ms):.4f} (±{np.std(ms):.4f})")
print("  理論: 平面を分割した図形では、オイラーの多面体定理から平均辺数はちょうど6に近づく")
OUT["exp2"] = rows

# 面積の分布
p = np.random.default_rng(5).random((2000, 2))
g = (np.arange(900) + 0.5) / 900
gy, gx = np.meshgrid(g, g, indexing="ij")
G = np.column_stack([gx.ravel(), gy.ravel()])
idx = cKDTree(p).query(G)[1]
area = np.bincount(idx, minlength=len(p)) / len(G) * len(p)
print(f"\n  ランダム点のセル面積(平均1に規格化): 標準偏差 {area.std():.4f} / "
      f"最小 {area.min():.3f} / 最大 {area.max():.3f}")
OUT["area_random"] = area.tolist()

# ================================================================ 実験3
print("\n[実験3] 距離の測り方を変える", flush=True)
P3 = np.random.default_rng(11).random((60, 2))
gg = (np.arange(500) + 0.5) / 500
GY, GX = np.meshgrid(gg, gg, indexing="ij")
res3 = {}
for name, ordv in [("ユークリッド距離", 2), ("マンハッタン距離", 1), ("チェビシェフ距離", np.inf)]:
    dx = np.abs(GX[..., None] - P3[:, 0])
    dy = np.abs(GY[..., None] - P3[:, 1])
    if ordv == 2:
        d = dx * dx + dy * dy
    elif ordv == 1:
        d = dx + dy
    else:
        d = np.maximum(dx, dy)
    lab = np.argmin(d, axis=-1)
    # 境界の長さ(隣と違うラベルの画素数)で「縄張りの縁の複雑さ」を測る
    edge = ((lab[1:, :] != lab[:-1, :]).sum() + (lab[:, 1:] != lab[:, :-1]).sum())
    a = np.bincount(lab.ravel(), minlength=len(P3)) / lab.size * len(P3)
    res3[name] = {"edge_px": int(edge), "area_sd": float(a.std())}
    print(f"  {name:<14} 境界の総画素 {edge:>7,} / セル面積の標準偏差 {a.std():.4f}")
    np.save(f"lab_{ordv if ordv != np.inf else 'inf'}.npy", lab.astype(np.int16))
OUT["exp3"] = res3
np.save("pts3.npy", P3)

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json")
