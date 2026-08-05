# box-counting でフラクタル次元を測り直す。
#
# 前回の失敗: 「内側の画素と外側の画素が両方入っている箱」を数える方式だと、
# 境界がちょうど箱の境目を通るとき(円の上下など)に箱を取りこぼし、
# 検証用の円で 1.000 のはずが 0.880 と低く出た。
# 今回は「境界画素」(内側で、かつ4近傍に外側がある画素)を先に作り、
# その画素集合を覆う箱を数える標準的なやり方に変える。
#
# さらに、解像度を 512 -> 8192 と上げながら同じ測り方を繰り返し、
# 推定値が収束するのか上がり続けるのかを見る。

import json
import time

import numpy as np

from common import escape_time, circle_mask, koch_mask

EXTENT = 2.6
XMIN, XMAX = -2.0, 0.6
YMIN, YMAX = -1.3, 1.3
MAXITER = 2000


def boundary_pixels(inside):
    """内側の画素のうち、上下左右のどれかが外側であるものを境界画素とする。"""
    b = np.zeros_like(inside)
    b[1:, :] |= inside[1:, :] & ~inside[:-1, :]
    b[:-1, :] |= inside[:-1, :] & ~inside[1:, :]
    b[:, 1:] |= inside[:, 1:] & ~inside[:, :-1]
    b[:, :-1] |= inside[:, :-1] & ~inside[:, 1:]
    # 画像の外周に接している内側の画素も境界扱いにはしない(枠で切れているだけなので)
    return b


def box_count(mask, sizes):
    R = mask.shape[0]
    out = []
    for b in sizes:
        k = R // b
        blk = mask[: k * b, : k * b].reshape(k, b, k, b)
        out.append(int(blk.any(axis=(1, 3)).sum()))
    return np.array(out)


def dimension(mask, sizes, extent, R):
    counts = box_count(mask, sizes)
    eps = np.array(sizes, dtype=float) * extent / R
    ok = counts > 0
    D = np.polyfit(np.log(1 / eps[ok]), np.log(counts[ok]), 1)[0]
    return float(D), counts, eps


SIZES = [1, 2, 4, 8, 16, 32]  # 画素単位。実座標での ε は解像度によって変わる
RESOLUTIONS = [512, 1024, 2048, 4096, 8192]

results = {"sizes_px": SIZES, "resolutions": RESOLUTIONS, "maxiter": MAXITER, "by_shape": {}}

for name in ["円", "コッホ雪片", "マンデルブロ集合"]:
    results["by_shape"][name] = {"D": [], "counts": [], "eps": []}

for R in RESOLUTIONS:
    print(f"\n=== 解像度 {R}x{R} ===", flush=True)
    t0 = time.time()
    esc = escape_time(XMIN, XMAX, YMIN, YMAX, R, R, MAXITER)
    masks = {
        "円": circle_mask(R, EXTENT),
        "コッホ雪片": koch_mask(R, order=7, extent=EXTENT),
        "マンデルブロ集合": esc == 0,
    }
    print(f"  マスク生成 {time.time() - t0:.1f} 秒", flush=True)
    for name, m in masks.items():
        bp = boundary_pixels(m)
        D, counts, eps = dimension(bp, SIZES, EXTENT, R)
        results["by_shape"][name]["D"].append(D)
        results["by_shape"][name]["counts"].append(counts.tolist())
        results["by_shape"][name]["eps"].append(eps.tolist())
        print(f"  {name:<16} D = {D:.4f}  境界画素 {bp.sum():>10,}  箱数 {counts.tolist()}")

print("\n" + "=" * 72)
print("解像度を上げたときの次元推定の推移")
print(f"{'図形':<16}" + "".join(f"{r:>10}" for r in RESOLUTIONS))
for name, d in results["by_shape"].items():
    print(f"{name:<16}" + "".join(f"{v:>10.4f}" for v in d["D"]))
print("\n理論値: 円 1.0000 / コッホ雪片 1.2619 (log4/log3) / マンデルブロ集合 2.0000 (Shishikura 1998)")

with open("dimension.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("\n保存: dimension.json")
