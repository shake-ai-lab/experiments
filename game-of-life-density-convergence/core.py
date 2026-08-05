# ライフゲームの中核。1世代進める step と、まとめて回す run。

import numpy as np


def step(g):
    """トーラス境界のライフゲームを1世代進める。周囲8方向をrollで足すのが一番速い。"""
    n = (
        np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)
        + np.roll(np.roll(g, 1, 0), 1, 1) + np.roll(np.roll(g, 1, 0), -1, 1)
        + np.roll(np.roll(g, -1, 0), 1, 1) + np.roll(np.roll(g, -1, 0), -1, 1)
    )
    return ((n == 3) | (g.astype(bool) & (n == 2))).astype(np.uint8)


def run(g, gens, record_pop=True):
    pops = np.empty(gens + 1, dtype=np.int64)
    pops[0] = g.sum()
    for t in range(gens):
        g = step(g)
        pops[t + 1] = g.sum()
    return g, (pops if record_pop else None)


