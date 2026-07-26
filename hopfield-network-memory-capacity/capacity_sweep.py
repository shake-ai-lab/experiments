# -*- coding: utf-8 -*-
"""記憶枚数を1→300に増やしながら想起成功率を測る。

3条件:
  (a) ランダムパターン + ヘブ則
  (b) 文字パターン + ヘブ則
  (c) 文字パターン + 射影ルール
手がかり: 10%の画素を反転させたノイズ入りパターン。
成功判定: 収束状態と元パターンのオーバーラップ m >= 0.95。
失敗時は「別の記憶に収束」か「どの記憶とも違う偽アトラクタ」かも分類する。
"""
import json
import time
import numpy as np
from hopfield_common import (make_char_patterns, make_random_patterns, train,
                             train_projection, add_noise, recall,
                             classify_result, N)

NOISE = 0.10
TRIALS = 30
P_LIST = [1, 2, 3, 4, 5, 7, 10, 14, 20, 30, 45, 60, 80, 100, 110, 120, 130,
          140, 145, 150, 155, 160, 170, 180, 200, 230, 260, 300]

rng = np.random.default_rng(42)
char_pats, chars = make_char_patterns(300)
rand_pats = make_random_patterns(300, seed=0)

conditions = {
    "random_hebb": (rand_pats, train),
    "chars_hebb": (char_pats, train),
    "chars_projection": (char_pats, train_projection),
}

results = {}
t0 = time.time()
for name, (pats, train_fn) in conditions.items():
    rates, mean_ms, sp_counts, om_counts = [], [], [], []
    for P in P_LIST:
        W = train_fn(pats[:P])
        ok = sp = om = 0
        ms = []
        for t in range(TRIALS):
            idx = int(rng.integers(P)) if P > 1 else 0
            cue = add_noise(pats[idx], NOISE, rng)
            s, sw, _ = recall(W, cue, rng)
            label, m_cue, bi, ba = classify_result(s, pats[:P], idx)
            ms.append(m_cue)
            if label == "success":
                ok += 1
            elif label == "spurious":
                sp += 1
            else:
                om += 1
        rates.append(ok / TRIALS)
        mean_ms.append(float(np.mean(ms)))
        sp_counts.append(sp)
        om_counts.append(om)
        print(f"[{name}] P={P}: success={ok}/{TRIALS} "
              f"(spurious={sp}, other={om}, mean m={np.mean(ms):.3f}) "
              f"[{time.time()-t0:.0f}s]", flush=True)
    results[name] = {"P": P_LIST, "rate": rates, "mean_m": mean_ms,
                     "spurious": sp_counts, "other_memory": om_counts}

np.savez("capacity_results.npz", **{
    f"{name}_{k}": np.array(v) for name, r in results.items()
    for k, v in r.items()})
with open("capacity_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("done in", time.time() - t0, "s")
