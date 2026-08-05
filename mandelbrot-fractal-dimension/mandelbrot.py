# マンデルブロ集合: 面積は有限なのに、境界のフラクタル次元は2なのか
#
# 実験1: 面積をモンテカルロ/グリッドで推定し、最大反復回数を増やしたときの収束を見る
# 実験2: box-counting で境界のフラクタル次元を実測する
#         validator として「円」(境界次元=1.000) と「コッホ雪片」(境界次元=1.2619) を同じコードで測る
# 実験3: εを細かくしたときの局所傾きの推移(収束するか、上がり続けるか)
#
# 実行: C:/bin/Python/python310-64/python.exe mandelbrot.py

import json
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = {}


from common import (escape_time, circle_mask, koch_mask, boundary_box_count,
                    fit_dimension)  # noqa: E402

# ================================================================ 実験開始
EXTENT = 2.6
XMIN, XMAX = -2.0, 0.6
YMIN, YMAX = -1.3, 1.3
R = 4096
MAXITER = 500

print("=" * 70)
print(f"マンデルブロ集合を {R}x{R}、最大反復 {MAXITER} 回で計算する", flush=True)
t0 = time.time()
import os

if os.path.exists("escape_4096.npy"):
    esc = np.load("escape_4096.npy")
    print("  (キャッシュ escape_4096.npy を読み込んだ)")
else:
    esc = escape_time(XMIN, XMAX, YMIN, YMAX, R, R, MAXITER)
inside = esc == 0
print(f"  計算時間 {time.time() - t0:.1f} 秒 / 集合内とみなされた画素 {inside.sum():,}", flush=True)

# ---------------------------------------------------------------- 実験1: 面積
cell_area = (EXTENT / R) ** 2
area = inside.sum() * cell_area
print(f"\n[実験1] グリッド面積推定 (maxiter={MAXITER}): {area:.5f}")
print("  文献値(Monte-Carlo/ピクセル計数による推定): 約 1.5065")
RESULTS["area_grid"] = area

# 最大反復回数を増やすと面積推定はどう動くか(小さめの解像度で走査)
R2 = 1024
iters = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
areas = []
for m in iters:
    e = escape_time(XMIN, XMAX, YMIN, YMAX, R2, R2, m)
    a = (e == 0).sum() * (EXTENT / R2) ** 2
    areas.append(a)
    print(f"  maxiter={m:>5}: 面積推定 {a:.5f}")
RESULTS["area_vs_iter"] = {"iters": iters, "areas": areas}

# ---------------------------------------------------------------- 実験2: box counting
SIZES = [2**k for k in range(1, 9)]  # 2..256 画素
print(f"\n[実験2] box-counting (箱サイズ {SIZES} 画素)")

targets = {}
print("  検証1: 円 (境界の次元は厳密に 1)")
cm = circle_mask(R, EXTENT)
targets["円"] = cm
print("  検証2: コッホ雪片 (境界の次元は log4/log3 = 1.26186)")
km = koch_mask(R, order=6, extent=EXTENT)
targets["コッホ雪片"] = km
targets["マンデルブロ集合"] = inside

table = {}
for name, mask in targets.items():
    counts = boundary_box_count(mask, SIZES)
    D, eps, _ = fit_dimension(SIZES, counts, EXTENT, R)
    table[name] = {"sizes": SIZES, "counts": counts.tolist(), "eps": eps.tolist(), "D": float(D)}
    print(f"  {name:<16} 全体フィット D = {D:.4f}   箱数 = {counts.tolist()}")
RESULTS["boxcount"] = table

# ---------------------------------------------------------------- 実験3: 局所傾き
print("\n[実験3] 隣り合う2点だけで出した局所傾き (εを細かくすると次元推定はどう動くか)")
print(f"  {'箱サイズ(画素)':<18}", " ".join(f"{a}->{b}" for a, b in zip(SIZES[:-1], SIZES[1:])))
local = {}
for name, d in table.items():
    c = np.array(d["counts"], dtype=float)
    e = np.array(d["eps"])
    slopes = (np.log(c[:-1]) - np.log(c[1:])) / (np.log(1 / e[:-1]) - np.log(1 / e[1:]))
    local[name] = slopes.tolist()
    print(f"  {name:<16}", " ".join(f"{s:8.3f}" for s in slopes))
RESULTS["local_slopes"] = local

# ---------------------------------------------------------------- 周長
print("\n[実験4] 境界の『長さ』N(ε)*ε は収束するか")
peri = {}
for name, d in table.items():
    c = np.array(d["counts"], dtype=float)
    e = np.array(d["eps"])
    L = c * e
    peri[name] = L.tolist()
    print(f"  {name:<16}", " ".join(f"{v:9.2f}" for v in L))
RESULTS["perimeter"] = peri

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=1)

np.save("inside_4096.npy", inside)
np.save("escape_4096.npy", esc)
print("\n保存: results.json / inside_4096.npy / escape_4096.npy")
