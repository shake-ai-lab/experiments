# 実験Aの「落ち着くまでの世代数」が、ほとんど2000(=打ち切り)に張り付いてしまった。
# 振動子が常に人口を上下させるので「±3%以内に収まり続ける」という条件が厳しすぎた。
# 半減期のような、もっと素直な指標に変える。
# また実験Cで差の半径が150セル(=格子の半分)に達して飽和していたので、格子を広げて測り直す。

import json

import numpy as np

from core import step, run

OUT = {}

# ------------------------------------------------------------------ A': 減り方
N = 384
GENS = 10000
print(f"[実験A'] {N}x{N}、初期密度0.375で{GENS}世代まで回し、密度の減り方を見る", flush=True)
rng = np.random.default_rng(4242)
g = (rng.random((N, N)) < 0.375).astype(np.uint8)
_, pops = run(g, GENS)
dens = pops / (N * N)
for t in [0, 10, 50, 100, 500, 1000, 2000, 5000, 10000]:
    print(f"  {t:>6}世代: 密度 {dens[t]:.5f}")
OUT["long_run"] = {"N": N, "gens": GENS, "density_every_10": dens[::10].tolist()}

# 「落ち着くまで」を素直に定義し直す: 密度が最終値の2倍を下回る最初の世代
DENSITIES = [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.375, 0.45, 0.50,
             0.55, 0.62, 0.70, 0.78, 0.85]
print(f"\n[実験A''] 密度が「最終値の2倍」を下回るまでの世代数", flush=True)
rows = []
for p in DENSITIES:
    ts = []
    for s in range(3):
        rng = np.random.default_rng(hash((round(p * 1000), s, 7)) % (2**32))
        g = (rng.random((N, N)) < p).astype(np.uint8)
        _, pops = run(g, 2000)
        d = pops / (N * N)
        final = d[-200:].mean()
        idx = np.flatnonzero(d <= 2 * final)
        ts.append(int(idx[0]) if idx.size else 2000)
    rows.append({"p": p, "t_half_mean": float(np.mean(ts)), "t_half_sd": float(np.std(ts))})
    print(f"  初期密度 {p:.3f} -> {np.mean(ts):7.1f} 世代 (±{np.std(ts):.1f})", flush=True)
OUT["settle"] = rows

# ------------------------------------------------------------------ C': 光円錐
M = 801
CENTER = M // 2
GC = 350
print(f"\n[実験C'] {M}x{M}に広げて、1セルの違いが広がる速さを測り直す", flush=True)
rng = np.random.default_rng(777)
a = (rng.random((M, M)) < 0.375).astype(np.uint8)
b = a.copy()
b[CENTER, CENTER] ^= 1
yy, xx = np.mgrid[0:M, 0:M]
dist = np.maximum(np.abs(yy - CENTER), np.abs(xx - CENTER))
radii, counts = [], []
for t in range(GC):
    a, b = step(a), step(b)
    d = a != b
    counts.append(int(d.sum()))
    radii.append(int(dist[d].max()) if d.any() else 0)
radii = np.array(radii)
valid = radii < (CENTER - 20)
tt = np.arange(GC)[valid]
speed = np.polyfit(tt[tt >= 30], radii[valid][tt >= 30], 1)[0]
print(f"  {GC}世代後: 食い違うセル {counts[-1]:,} 個 / 差の半径 {radii[-1]} セル (飽和前の有効範囲 {valid.sum()} 世代)")
print(f"  半径の伸びる速さ: {speed:.4f} セル/世代  (理論上限 1.0 = ライフゲームの光速c)")
print(f"  差が最初に生まれてから10世代の半径: {radii[:10].tolist()}")
OUT["lightcone"] = {"M": M, "gens": GC, "radius": radii.tolist(), "diff_cells": counts,
                    "speed": float(speed)}

with open("results2.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results2.json")
