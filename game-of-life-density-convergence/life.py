# ライフゲーム: 初期密度を変えると、行き着く先も変わるのか
#
# 実験A: 初期密度を0.02〜0.95まで振り、十分時間が経ったあとの生存密度と、
#        落ち着くまでにかかった世代数を測る
# 実験B: 落ち着いたあとに残っているパターンの内訳を数える
# 実験C: 初期状態を1セルだけ変えた2つの宇宙を並走させ、差がどう広がるかを測る
#
# 実行: C:/bin/Python/python310-64/python.exe -u life.py

import json
import time
from collections import Counter

import numpy as np
from scipy import ndimage

rng_master = np.random.default_rng(20260805)


from core import step, run


RESULTS = {}

# ================================================================= 実験A
N = 384
GENS = 2000
DENSITIES = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.375, 0.45,
             0.50, 0.55, 0.62, 0.70, 0.78, 0.85, 0.90, 0.95]
SEEDS = 4
TAIL = 200  # 最後のこの世代数の平均を「落ち着いた密度」とみなす

print(f"[実験A] {N}x{N}のトーラス、{GENS}世代、初期密度{len(DENSITIES)}通り×{SEEDS}シード", flush=True)
rowsA = []
pop_curves = {}
for p in DENSITIES:
    finals, times = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(hash((round(p * 1000), s)) % (2**32))
        g0 = (rng.random((N, N)) < p).astype(np.uint8)
        t0 = time.time()
        g, pops = run(g0.copy(), GENS)
        final = pops[-TAIL:].mean() / (N * N)
        # 落ち着くまでの世代数: 以降ずっと最終平均の±3%以内に収まる最初の世代
        band = np.abs(pops / (N * N) - final) <= 0.03 * final
        settle = GENS
        for t in range(GENS - TAIL):
            if band[t:].all():
                settle = t
                break
        finals.append(final)
        times.append(settle)
        if s == 0:
            pop_curves[p] = (pops[::5] / (N * N)).tolist()
    rowsA.append({"p": p, "final_mean": float(np.mean(finals)), "final_sd": float(np.std(finals)),
                  "settle_mean": float(np.mean(times)), "settle_sd": float(np.std(times))})
    print(f"  初期密度 {p:.3f} -> 最終密度 {np.mean(finals):.5f} (±{np.std(finals):.5f})  "
          f"落ち着くまで {np.mean(times):.0f} 世代  ({time.time()-t0:.1f}秒/回)", flush=True)
RESULTS["density_sweep"] = {"N": N, "gens": GENS, "tail": TAIL, "seeds": SEEDS, "rows": rowsA}
RESULTS["pop_curves"] = {str(k): v for k, v in pop_curves.items()}

allf = [r["final_mean"] for r in rowsA]
print(f"\n  最終密度の範囲: {min(allf):.5f} 〜 {max(allf):.5f}  "
      f"(初期密度は {min(DENSITIES)} 〜 {max(DENSITIES)} と {max(DENSITIES)/min(DENSITIES):.0f}倍違う)")

# ================================================================= 実験B
print(f"\n[実験B] 落ち着いたあとに残っているパターンの内訳", flush=True)


def normalize(mask):
    """パターンを回転・反転・平行移動で正規化した署名にする"""
    best = None
    m = mask
    for _ in range(4):
        for cand in (m, m[:, ::-1]):
            b = np.argwhere(cand)
            c = cand[b[:, 0].min():b[:, 0].max() + 1, b[:, 1].min():b[:, 1].max() + 1]
            key = (c.shape, c.tobytes())
            if best is None or key < best:
                best = key
        m = np.rot90(m)
    return best


NAMES = {
    ((2, 2), 4): "ブロック", ((1, 3), 3): "ブリンカー", ((3, 1), 3): "ブリンカー",
    ((3, 4), 6): "ハチの巣", ((4, 4), 7): "パン", ((3, 3), 4): "舟/池の類",
    ((2, 3), 4): "たらい/船の類", ((3, 3), 5): "船/たらいの類", ((4, 4), 8): "池",
}
census = Counter()
period_count = Counter()
TRIALS = 6
for s in range(TRIALS):
    rng = np.random.default_rng(9000 + s)
    g = (rng.random((N, N)) < 0.375).astype(np.uint8)
    g, _ = run(g, 1500, record_pop=False)
    g1 = step(g)
    g2 = step(g1)
    lab, n = ndimage.label(g, structure=np.ones((3, 3)))
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        sub = (lab[sl] == i)
        cells = int(sub.sum())
        # 静止か振動かを判定(同じ切り出し窓で2世代先と比べる)
        stat = np.array_equal(g[sl] * sub, g2[sl] * sub) and np.array_equal(g[sl] * sub, g1[sl] * sub)
        osc = (not stat) and np.array_equal(g[sl] * sub, g2[sl] * sub)
        period_count["静止物" if stat else ("周期2の振動子" if osc else "その他")] += 1
        shape, _ = normalize(sub)
        census[NAMES.get((shape, cells), f"その他({shape[0]}x{shape[1]}, {cells}セル)")] += 1

total = sum(census.values())
print(f"  {TRIALS}回ぶんの灰から検出した物体 {total:,} 個")
top = census.most_common(8)
for name, c in top:
    print(f"    {name:<28} {c:>6,} 個 ({c/total*100:5.2f}%)")
print("  種類別:", dict(period_count))
RESULTS["census"] = {"total": total, "top": top, "kind": dict(period_count)}

# ================================================================= 実験C
print(f"\n[実験C] 初期状態を1セルだけ変えたら、差はどう広がるか", flush=True)
M = 301
CENTER = M // 2
rng = np.random.default_rng(777)
a0 = (rng.random((M, M)) < 0.375).astype(np.uint8)
b0 = a0.copy()
b0[CENTER, CENTER] ^= 1  # たった1セル反転させる
a, b = a0, b0
GC = 400
radii, counts = [], []
yy, xx = np.mgrid[0:M, 0:M]
dist = np.maximum(np.abs(yy - CENTER), np.abs(xx - CENTER))  # チェビシェフ距離
for t in range(GC):
    a, b = step(a), step(b)
    d = a != b
    counts.append(int(d.sum()))
    radii.append(int(dist[d].max()) if d.any() else 0)
print(f"  400世代後: 食い違っているセル {counts[-1]:,} 個 / 差の半径 {radii[-1]} セル")
fit = np.polyfit(np.arange(50, GC), radii[50:], 1)[0]
print(f"  半径の伸びる速さ(50世代目以降の回帰): {fit:.4f} セル/世代 (理論上限は 1.0 = ライフゲームの光速c)")
RESULTS["lightcone"] = {"M": M, "gens": GC, "radius": radii, "diff_cells": counts, "speed": float(fit)}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False)
print("\n保存: results.json")
