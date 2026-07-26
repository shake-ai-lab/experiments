# -*- coding: utf-8 -*-
"""(記憶枚数 P) x (手がかりのノイズ量) の想起成功率ヒートマップ用データ。

ランダムパターン + ヘブ則。成功判定は m >= 0.95。
"""
import json
import time
import numpy as np
from hopfield_common import (make_random_patterns, train, add_noise, recall,
                             overlap)

P_LIST = [10, 25, 50, 75, 100, 110, 120, 130, 140, 150, 160, 175, 200, 250, 300]
NOISE_LIST = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TRIALS = 12

rng = np.random.default_rng(7)
pats = make_random_patterns(300, seed=0)

grid = np.zeros((len(NOISE_LIST), len(P_LIST)))
t0 = time.time()
for j, P in enumerate(P_LIST):
    W = train(pats[:P])
    for i, nz in enumerate(NOISE_LIST):
        ok = 0
        for t in range(TRIALS):
            idx = int(rng.integers(P))
            cue = add_noise(pats[idx], nz, rng)
            s, sw, _ = recall(W, cue, rng)
            if overlap(s, pats[idx]) >= 0.95:
                ok += 1
        grid[i, j] = ok / TRIALS
    print(f"P={P} done: {np.round(grid[:, j], 2)} [{time.time()-t0:.0f}s]",
          flush=True)

np.savez("heatmap_results.npz", grid=grid, P_LIST=np.array(P_LIST),
         NOISE_LIST=np.array(NOISE_LIST))
with open("heatmap_results.json", "w", encoding="utf-8") as f:
    json.dump({"P": P_LIST, "noise": NOISE_LIST, "grid": grid.tolist()},
              f, indent=1)
print("done in", time.time() - t0, "s")
