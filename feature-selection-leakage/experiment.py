# -*- coding: utf-8 -*-
"""
特徴選択のデータリークを実測する実験。

n=100 サンプル x p 次元の純粋なガウスノイズと、独立なランダム2値ラベルを生成し、
以下の3条件で5-fold Stratified CV の精度を測る。

  A) leak_select : 全データでラベルとの相関上位 k 個を選んでから CV(特徴選択リーク)
  B) proper      : CV の各 fold の訓練側の中だけで選択(正しいやり方)
  C) leak_scale  : 標準化(StandardScaler)だけ全データで先に fit するリーク。
                   特徴選択は fold 内で正しく行う(前処理リークの害の大きさの対照)

分類器はロジスティック回帰。真の信号はゼロなので、正しい CV の期待精度は 0.5。
"""
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

N_SAMPLES = 100
N_SPLITS = 5
N_SEEDS = 20

P_LIST = [100, 316, 1000, 3162, 10000, 31623, 100000]
K_LIST = [10, 32, 100, 316, 1000]


def gen_data(p, seed):
    """純粋なノイズ X と、それとは完全に独立なラベル y。"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_SAMPLES, p))
    y = rng.integers(0, 2, N_SAMPLES)
    # StratifiedKFold が成立するよう、極端に偏ったラベルだけ引き直す
    while y.sum() < N_SPLITS or (N_SAMPLES - y.sum()) < N_SPLITS:
        y = rng.integers(0, 2, N_SAMPLES)
    return X, y


def corr_rank(X, y):
    """各特徴とラベルの相関の絶対値が大きい順の特徴インデックス。"""
    yc = y - y.mean()
    Xc = X - X.mean(axis=0)
    num = Xc.T @ yc
    denom = np.sqrt((Xc**2).sum(axis=0)) * np.sqrt((yc**2).sum())
    corr = np.abs(num / np.maximum(denom, 1e-12))
    return np.argsort(-corr)


def fit_score(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


def run_conditions(X, y, k_list, seed):
    """1つのデータセットに対して3条件 x 全kのCV精度を返す。"""
    p = X.shape[1]
    ks = [k for k in k_list if k <= p]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))

    # --- A) 特徴選択リーク: 全データで選んでから CV(スケーリングは fold 内で正しく) ---
    rank_full = corr_rank(X, y)
    acc_leak_sel = {}
    for k in ks:
        idx = rank_full[:k]
        Xk = X[:, idx]
        scores = []
        for tr, te in folds:
            sc = StandardScaler().fit(Xk[tr])
            scores.append(fit_score(sc.transform(Xk[tr]), y[tr],
                                    sc.transform(Xk[te]), y[te]))
        acc_leak_sel[k] = float(np.mean(scores))

    # --- B) 正しい CV: 選択もスケーリングも fold 内 ---
    # --- C) スケーリングリーク: scaler だけ全データで fit、選択は fold 内 ---
    X_scaled_full = StandardScaler().fit_transform(X)
    acc_proper = {k: [] for k in ks}
    acc_leak_scale = {k: [] for k in ks}
    for tr, te in folds:
        rank_tr = corr_rank(X[tr], y[tr])  # fold 内選択(B, C 共通)
        for k in ks:
            idx = rank_tr[:k]
            # B: fold 内スケーリング
            sc = StandardScaler().fit(X[tr][:, idx])
            acc_proper[k].append(fit_score(sc.transform(X[tr][:, idx]), y[tr],
                                           sc.transform(X[te][:, idx]), y[te]))
            # C: 全データでスケーリング済みの X を使う
            acc_leak_scale[k].append(fit_score(X_scaled_full[tr][:, idx], y[tr],
                                               X_scaled_full[te][:, idx], y[te]))
    return (acc_leak_sel,
            {k: float(np.mean(v)) for k, v in acc_proper.items()},
            {k: float(np.mean(v)) for k, v in acc_leak_scale.items()})


def main():
    t0 = time.time()
    results = []  # list of dicts: p, k, seed, cond, acc
    for p in P_LIST:
        for seed in range(N_SEEDS):
            X, y = gen_data(p, seed=1000 * seed + p % 997)
            a, b, c = run_conditions(X, y, K_LIST, seed)
            for k in a:
                results.append(dict(p=p, k=k, seed=seed,
                                    leak_select=a[k], proper=b[k],
                                    leak_scale=c[k]))
        print(f"p={p:>6d} done  ({time.time() - t0:.1f}s)", flush=True)

    with open("results_grid.json", "w") as f:
        json.dump(results, f)

    # サマリ表示
    print("\n=== mean over seeds (leak_select / proper / leak_scale) ===")
    for p in P_LIST:
        for k in K_LIST:
            rows = [r for r in results if r["p"] == p and r["k"] == k]
            if not rows:
                continue
            ls = np.mean([r["leak_select"] for r in rows])
            pr = np.mean([r["proper"] for r in rows])
            sc = np.mean([r["leak_scale"] for r in rows])
            print(f"p={p:>6d} k={k:>4d}  leak_sel={ls:.3f}  proper={pr:.3f}  leak_scale={sc:.3f}")


if __name__ == "__main__":
    main()
