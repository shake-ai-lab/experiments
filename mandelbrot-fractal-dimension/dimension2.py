# box-counting をやり直す。前回分かったこと:
#   箱サイズ1〜2画素では digitization の癖で傾きが必ず低く出る(円で0.70)。
#   箱サイズを4画素以上に絞ると、円は1.00、コッホ雪片は1.23(理論1.2619)に落ち着く。
#   → フィット範囲を 4〜64 画素に固定し、円とコッホで較正してからマンデルブロを測る。
#
# そして本題。解像度だけ上げてもマンデルブロの次元推定が1.0付近から動かなかった。
# 仮説: 有限の反復回数で「集合内」とした領域の輪郭は、本物の境界ではなく
#       等ポテンシャル線(なめらかな解析曲線)に近いので、次元1に見えている。
#       だとすれば、反復回数を上げるほど輪郭は本物の境界に寄り、次元推定が上がるはず。

import json
import time

import numpy as np

from common import escape_time, circle_mask, koch_mask

EXTENT = 2.6
XMIN, XMAX = -2.0, 0.6
YMIN, YMAX = -1.3, 1.3
FIT_SIZES = [4, 8, 16, 32, 64]


def boundary_pixels(inside):
    b = np.zeros_like(inside)
    b[1:, :] |= inside[1:, :] & ~inside[:-1, :]
    b[:-1, :] |= inside[:-1, :] & ~inside[1:, :]
    b[:, 1:] |= inside[:, 1:] & ~inside[:, :-1]
    b[:, :-1] |= inside[:, :-1] & ~inside[:, 1:]
    return b


def box_count(mask, sizes):
    R = mask.shape[0]
    return np.array(
        [int(mask[: (R // b) * b, : (R // b) * b]
             .reshape(R // b, b, R // b, b)
             .any(axis=(1, 3)).sum()) for b in sizes]
    )


def dimension(mask, extent, sizes=FIT_SIZES):
    R = mask.shape[0]
    counts = box_count(mask, sizes)
    eps = np.array(sizes, dtype=float) * extent / R
    ok = counts > 0
    D = np.polyfit(np.log(1 / eps[ok]), np.log(counts[ok]), 1)[0]
    return float(D), counts


out = {"fit_sizes_px": FIT_SIZES}

# ------------------------------------------------------------ 較正
print("[較正] 箱サイズ 4〜64 画素に絞ると、既知の図形は理論値に合うか", flush=True)
calib = {}
for R in [2048, 4096, 8192]:
    dc, _ = dimension(boundary_pixels(circle_mask(R, EXTENT)), EXTENT)
    dk, _ = dimension(boundary_pixels(koch_mask(R, order=7, extent=EXTENT)), EXTENT)
    calib[R] = {"円": dc, "コッホ雪片": dk}
    print(f"  {R:>5}x{R:<5} 円 D={dc:.4f} (理論1.0000) / コッホ雪片 D={dk:.4f} (理論1.2619)")
out["calibration"] = calib

# ------------------------------------------------------------ 反復回数スイープ
R = 4096
print(f"\n[実験A] 解像度を {R}x{R} に固定し、最大反復回数だけを上げる", flush=True)
iters = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
rows = []
for m in iters:
    t0 = time.time()
    esc = escape_time(XMIN, XMAX, YMIN, YMAX, R, R, m)
    inside = esc == 0
    bp = boundary_pixels(inside)
    D, counts = dimension(bp, EXTENT)
    area = inside.sum() * (EXTENT / R) ** 2
    rows.append({"maxiter": m, "D": D, "area": area,
                 "boundary_px": int(bp.sum()), "counts": counts.tolist()})
    print(f"  maxiter={m:>6}  D={D:.4f}  面積={area:.5f}  境界画素={bp.sum():>9,}  ({time.time()-t0:.0f}秒)",
          flush=True)
out["iter_sweep"] = {"resolution": R, "rows": rows}

# ------------------------------------------------------------ 解像度スイープ(反復回数は多め固定)
print("\n[実験B] 最大反復回数を 20000 に固定し、解像度だけを上げる", flush=True)
rows2 = []
for RR in [512, 1024, 2048, 4096, 8192]:
    t0 = time.time()
    esc = escape_time(XMIN, XMAX, YMIN, YMAX, RR, RR, 20000)
    bp = boundary_pixels(esc == 0)
    D, counts = dimension(bp, EXTENT)
    rows2.append({"resolution": RR, "D": D, "boundary_px": int(bp.sum()), "counts": counts.tolist()})
    print(f"  {RR:>5}x{RR:<5}  D={D:.4f}  境界画素={bp.sum():>9,}  ({time.time()-t0:.0f}秒)", flush=True)
out["res_sweep"] = {"maxiter": 20000, "rows": rows2}

with open("dimension2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n保存: dimension2.json")
