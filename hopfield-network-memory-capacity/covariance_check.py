# -*- coding: utf-8 -*-
"""共分散則(バイアス補正ヘブ則)で文字パターンがどこまで覚えられるか。"""
import numpy as np
from hopfield_common import (make_char_patterns, train, add_noise, recall,
                             overlap, N)

pats, chars = make_char_patterns(300)
rng = np.random.default_rng(3)


def train_cov(patterns, a=None):
    X = patterns.astype(np.float32)
    if a is None:
        a = X.mean()
    C = X - a
    W = C.T @ C / N
    np.fill_diagonal(W, 0.0)
    return W


for P in [3, 5, 10, 20, 40, 60, 80, 100, 150, 200, 300]:
    W = train_cov(pats[:P])
    ok = 0
    ms = []
    for t in range(10):
        idx = int(rng.integers(P)) if P > 1 else 0
        cue = add_noise(pats[idx], 0.25, rng)
        s, sw, _ = recall(W, cue, rng)
        m = overlap(s, pats[idx])
        ms.append(m)
        if m >= 0.95:
            ok += 1
    print(f"cov chars P={P}: success {ok}/10, mean m={np.mean(ms):.3f}")
