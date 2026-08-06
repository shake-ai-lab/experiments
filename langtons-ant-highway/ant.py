# ラングトンのアリ: 1万歩のカオスのあとに、なぜ突然「高速道路」ができるのか
#
# ルールは2行しかない。
#   白いマスに来たら: 右に90度回り、そのマスを黒くして、1歩進む
#   黒いマスに来たら: 左に90度回り、そのマスを白くして、1歩進む
#
# 実験1: 全部白の盤から始めて、ハイウェイに入るまでのステップ数を実測
# 実験2: 初期にランダムな黒マスを撒くと、ハイウェイ到達までの歩数はどう変わるか
# 実験3: ルールを「RL」以外の文字列に変えると何が起きるか(Turmite)
#
# 実行: C:/bin/Python/python310-64/python.exe -u ant.py

import json

import numpy as np

from core import run_ant, highway_start, DIRS

OUT = {}

# ================================================================ 実験1
STEPS = 20000
print(f"[実験1] 全部白の盤から {STEPS:,} ステップ", flush=True)
grid, traj, black, _ = run_ant(STEPS, N=560, record_every=True)
t0, disp = highway_start(traj)
print(f"  ハイウェイに入ったのは {t0:,} ステップ目 / 104ステップあたりの移動 {disp}")
print(f"  黒マスの数: 1,000歩で {black[999]:,} / 5,000歩で {black[4999]:,} / "
      f"{t0:,}歩で {black[t0]:,} / {STEPS:,}歩で {black[-1]:,}")
# 対称性がいつ壊れるか(盤の左右反転との一致率)
sym = []
g2, tr2, _, _ = run_ant(1200, N=200, record_every=False)
OUT["exp1"] = {"steps": STEPS, "highway_start": int(t0), "disp": list(disp),
               "black": black[::10].tolist()}

# ================================================================ 実験2
print("\n[実験2] 最初にランダムな黒マスを撒いてから走らせる", flush=True)
N2 = 560
rows = []
for p in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]:
    ts = []
    for s in range(6):
        rng = np.random.default_rng(1000 * s + int(p * 10000))
        init = (rng.random((N2, N2)) < p).astype(np.uint8) if p > 0 else None
        _, tr, _, _ = run_ant(60000, N=N2, init=init, record_every=False)
        t, dd = highway_start(tr)
        ts.append(t if t is not None else -1)
    ok = [t for t in ts if t > 0]
    rows.append({"p": p, "times": ts, "median": float(np.median(ok)) if ok else None,
                 "fail": int(len(ts) - len(ok))})
    med = f"{np.median(ok):,.0f}" if ok else "到達せず"
    print(f"  初期の黒マス率 {p*100:5.1f}%: ハイウェイ到達 中央値 {med} 歩 "
          f"(6回中 {len(ts)-len(ok)} 回は6万歩以内に到達せず)", flush=True)
OUT["exp2"] = {"rows": rows, "limit": 60000}

# ================================================================ 実験3
print("\n[実験3] ルール文字列を変える(Turmite)", flush=True)
RULES = ["RL", "RLR", "LLRR", "RRLL", "LRRRRRLLR", "LLRRRLRLRLLR", "RRLLLRLLLRRR"]
res3 = []
for rule in RULES:
    g, tr, _, _ = run_ant(60000, N=700, rule=rule, record_every=False)
    span_y = tr[:, 0].max() - tr[:, 0].min()
    span_x = tr[:, 1].max() - tr[:, 1].min()
    t, dd = highway_start(tr)
    filled = int((g > 0).sum())
    res3.append({"rule": rule, "span": [int(span_y), int(span_x)],
                 "highway": int(t) if t is not None else None,
                 "cells": filled})
    hw = f"{t:,}歩でハイウェイ" if t is not None else "ハイウェイ検出されず"
    print(f"  {rule:<14} 到達範囲 {span_y:>4}x{span_x:<4} / 塗られたマス {filled:>7,} / {hw}")
OUT["exp3"] = res3

np.save("grid_final.npy", grid)
np.save("traj.npy", traj)
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json / grid_final.npy / traj.npy")
