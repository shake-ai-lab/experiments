# ヒルベルト曲線: 1本の線で平面を埋めると、「近さ」はどこまで保たれるのか
#
# 実験1: 2次元で隣り合う画素が、1次元に並べ替えたあとどれだけ離れるか。
#         ヒルベルト順・Z階数(モートン順)・行優先を比べる
# 実験2: box-countingでヒルベルト曲線のフラクタル次元を測る(理論値2)
# 実験3: 実用の検証。画像をヒルベルト順に並べ替えてからgzipすると小さくなるか
#
# 実行: C:/bin/Python/python310-64/python.exe -u hilbert.py

import gzip
import json

import numpy as np

OUT = {}


from hilbert_core import hilbert_d2xy, hilbert_index, morton_index, row_index


ORDER = 8
N = 1 << ORDER
H = hilbert_index(ORDER)
M = morton_index(ORDER)
Rw = row_index(ORDER)
print(f"{N}x{N} の格子で比較する")

# 正しさの確認: ヒルベルト曲線は順番が1つ進むと必ず隣のマスに移る
d = np.arange(N * N)
hx, hy = hilbert_d2xy(ORDER, d)
step = np.abs(np.diff(hx)) + np.abs(np.diff(hy))
print(f"  ヒルベルト: 1歩あたりの移動距離 最小 {step.min()} / 最大 {step.max()} "
      f"(すべて1なら正しい)")
mx, my = np.nonzero(M == 0)
mo = np.argsort(M.ravel())
mxs, mys = np.unravel_index(mo, (N, N))
mstep = np.abs(np.diff(mxs)) + np.abs(np.diff(mys))
print(f"  Z階数    : 1歩あたりの移動距離 最小 {mstep.min()} / 最大 {mstep.max()}")

# ================================================================ 実験1
print("\n[実験1] 2次元で隣り合う画素は、1次元でどれだけ離れるか", flush=True)
rows = []
for name, idx in [("ヒルベルト順", H), ("Z階数(モートン順)", M), ("行優先", Rw)]:
    gap_h = np.abs(idx[:, 1:] - idx[:, :-1]).ravel()
    gap_v = np.abs(idx[1:, :] - idx[:-1, :]).ravel()
    gap = np.concatenate([gap_h, gap_v])
    rows.append({"name": name, "mean": float(gap.mean()), "median": float(np.median(gap)),
                 "p99": float(np.percentile(gap, 99)), "max": int(gap.max()),
                 "hist": np.histogram(np.log10(np.maximum(gap, 1)), bins=30,
                                      range=(0, 5))[0].tolist()})
    print(f"  {name:<18} 平均 {gap.mean():8.1f} / 中央値 {np.median(gap):6.0f} / "
          f"99パーセンタイル {np.percentile(gap,99):8.0f} / 最大 {gap.max():,}")
OUT["exp1"] = {"N": N, "rows": rows}

# 逆向き: 1次元で近いものは2次元でも近いか
print("\n  逆向き: 1次元で k 個離れた2点は、2次元でどれだけ離れているか", flush=True)
rows2 = []
for name, idx in [("ヒルベルト順", H), ("Z階数(モートン順)", M), ("行優先", Rw)]:
    order_map = np.argsort(idx.ravel())
    ys, xs = np.unravel_index(order_map, (N, N))
    line = []
    for k in [1, 4, 16, 64, 256]:
        dd = np.hypot(xs[k:] - xs[:-k], ys[k:] - ys[:-k])
        line.append(float(dd.mean()))
    rows2.append({"name": name, "dists": line})
    print(f"  {name:<18} " + " / ".join(f"k={k}: {v:6.2f}" for k, v in
                                        zip([1, 4, 16, 64, 256], line)))
OUT["exp1b"] = {"ks": [1, 4, 16, 64, 256], "rows": rows2}

# ================================================================ 実験2
print("\n[実験2] ヒルベルト曲線のフラクタル次元", flush=True)
res = []
for order in [3, 4, 5, 6, 7, 8, 9]:
    n = 1 << order
    dd = np.arange(n * n)
    x, y = hilbert_d2xy(order, dd)
    # 曲線を細かい格子(1024)の上に描いて、箱で数える
    G = 1024
    canvas = np.zeros((G, G), dtype=bool)
    px = np.round(x / (n - 1) * (G - 1)).astype(int)
    py = np.round(y / (n - 1) * (G - 1)).astype(int)
    # 線分を補間して塗る
    for i in range(len(px) - 1):
        canvas[py[i], px[i]] = True
    canvas[py[-1], px[-1]] = True
    sizes = [4, 8, 16, 32, 64]
    counts = np.array([canvas[: (G // b) * b, : (G // b) * b]
                       .reshape(G // b, b, G // b, b).any(axis=(1, 3)).sum() for b in sizes])
    eps = np.array(sizes, float) / G
    D = np.polyfit(np.log(1 / eps), np.log(counts), 1)[0]
    res.append({"order": order, "cells": int(n * n), "D": float(D)})
    print(f"  次数 {order} ({n}x{n}, {n*n:,}点): box-counting次元 {D:.4f}")
print("  理論値: 2.0000（曲線が平面を埋め尽くすので）")
OUT["exp2"] = res

# ================================================================ 実験3
print("\n[実験3] 画像をヒルベルト順に並べ替えてから圧縮する", flush=True)
rng = np.random.default_rng(7)
imgs = {}
# なめらかなグラデーション
yy, xx = np.mgrid[0:N, 0:N] / N
imgs["なめらかな勾配"] = ((np.sin(6 * xx) * np.cos(5 * yy) + 1) * 120).astype(np.uint8)
# 斑点(相関のある乱数)
from scipy import ndimage  # noqa: E402

imgs["ぼかした乱数"] = (ndimage.gaussian_filter(rng.random((N, N)), 6) * 900 % 256).astype(np.uint8)
# 完全な乱数
imgs["純粋な乱数"] = rng.integers(0, 256, (N, N), dtype=np.uint8)
# 同心円
imgs["同心円"] = ((np.hypot(xx - 0.5, yy - 0.5) * 900) % 256).astype(np.uint8)

rows3 = []
for name, img in imgs.items():
    sizes = {}
    for oname, idx in [("行優先", Rw), ("Z階数", M), ("ヒルベルト順", H)]:
        order_map = np.argsort(idx.ravel())
        seq = img.ravel()[order_map].tobytes()
        sizes[oname] = len(gzip.compress(seq, 9))
    base = sizes["行優先"]
    rows3.append({"image": name, **sizes,
                  "hilbert_gain": (base - sizes["ヒルベルト順"]) / base})
    print(f"  {name:<12} 行優先 {sizes['行優先']:>7,}B / Z階数 {sizes['Z階数']:>7,}B / "
          f"ヒルベルト {sizes['ヒルベルト順']:>7,}B "
          f"(行優先比 {(sizes['ヒルベルト順']-base)/base*100:+.1f}%)")
OUT["exp3"] = rows3

np.save("hilbert_idx.npy", H)
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json")
