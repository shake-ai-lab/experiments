# -*- coding: utf-8 -*-
"""
GIF用: k=10 に固定し、p を 100 -> 100,000 まで増やしながら、
特徴選択リークあり(A)と正しいCV(B)の精度分布(シード60個)がどう動くかを記録する。
"""
import json
import time

import numpy as np

from experiment import gen_data, corr_rank, fit_score, N_SPLITS
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

K = 10
N_SEEDS = 60
P_LIST = [int(round(v)) for v in np.geomspace(100, 100000, 13)]


def run_pair(X, y, seed):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))

    # A) 全データで選択
    idx = corr_rank(X, y)[:K]
    Xk = X[:, idx]
    acc_a = []
    for tr, te in folds:
        sc = StandardScaler().fit(Xk[tr])
        acc_a.append(fit_score(sc.transform(Xk[tr]), y[tr],
                               sc.transform(Xk[te]), y[te]))
    # B) fold内で選択
    acc_b = []
    for tr, te in folds:
        idx_tr = corr_rank(X[tr], y[tr])[:K]
        sc = StandardScaler().fit(X[tr][:, idx_tr])
        acc_b.append(fit_score(sc.transform(X[tr][:, idx_tr]), y[tr],
                               sc.transform(X[te][:, idx_tr]), y[te]))
    return float(np.mean(acc_a)), float(np.mean(acc_b))


def main():
    t0 = time.time()
    out = {}
    for p in P_LIST:
        leak, proper = [], []
        for seed in range(N_SEEDS):
            X, y = gen_data(p, seed=5_000_000 + 1000 * seed + p % 997)
            a, b = run_pair(X, y, seed)
            leak.append(a)
            proper.append(b)
        out[str(p)] = dict(leak=leak, proper=proper)
        print(f"p={p:>6d}  leak mean={np.mean(leak):.3f}  proper mean={np.mean(proper):.3f}"
              f"  ({time.time() - t0:.1f}s)", flush=True)

    with open("results_dist.json", "w") as f:
        json.dump(dict(k=K, n_seeds=N_SEEDS, p_list=P_LIST, data=out), f)


if __name__ == "__main__":
    main()
