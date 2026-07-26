"""Condorcet's Jury Theorem: N人の独立な投票者(それぞれ確率pで正しい)が
多数決を取ったとき、集団としての正解率はどうなるか。

理論: p>0.5ならN->infで集団正解率->1、p<0.5ならN->infで集団正解率->0、
p=0.5ならずっと0.5。これを実際にシミュレーションして確かめる。

このスクリプトは2つのことをやる:
  1. (N, p) の格子全体で、N人分のBernoulli(p)投票を実際にサンプリングして
     多数決の正解率を実測する(3D曲面・2D折れ線の両方の元データになる)
  2. いくつかの代表点で「投票者ごとの明示的な配列」を作って多数決を取り、
     二項分布の解析解(scipy.stats.binom)とどれだけ一致するかを検証する
"""
import numpy as np
from scipy.stats import binom

rng = np.random.default_rng(0)

N_VALUES = np.arange(1, 102, 2)          # 1, 3, 5, ..., 101 (奇数だけ = 同数票なし)
P_VALUES = np.round(np.arange(0.30, 0.901, 0.01), 4)  # 0.30, 0.31, ..., 0.90
TRIALS = 8000

print(f"grid: N={len(N_VALUES)}点 (1〜101), p={len(P_VALUES)}点 (0.30〜0.90), trials={TRIALS}")

# --- 1. 格子全体のシミュレーション -----------------------------------------
# N人の投票者のうち「正しい」人数は Binomial(N, p) に従う(iid Bernoulli(p)のN個の和)。
# 投票者1人1人の配列を作らずとも、rng.binomial(N, p, size=trials) で
# 「trials回、N人中何人が正しかったか」を直接サンプリングできる。
# これは "N個の独立なBernoulli(p)を引いて足す" のと数学的に同一の乱数生成であり、
# 格子全体(51 x 61 = 3111通り)を現実的な時間で回すために使う。
acc_grid = np.zeros((len(N_VALUES), len(P_VALUES)))
for i, N in enumerate(N_VALUES):
    for j, p in enumerate(P_VALUES):
        correct_votes = rng.binomial(N, p, size=TRIALS)
        majority_correct = correct_votes > N // 2  # Nは奇数なので同数票は起きない
        acc_grid[i, j] = majority_correct.mean()

np.savez("grid_results.npz", N_VALUES=N_VALUES, P_VALUES=P_VALUES, acc_grid=acc_grid)
print("saved grid_results.npz")

# --- 2. 明示的な投票者配列での検証 -----------------------------------------
# ここでは手を抜かず、本当に (trials, N) 形状の「投票者ごとの正誤」配列を作り、
# 行ごとに多数決を取る。二項分布の厳密解 P(X > N//2) と比較する。
def explicit_vote_simulation(N, p, trials, seed):
    r = np.random.default_rng(seed)
    votes = r.random((trials, N)) < p           # 各投票者が正しいか(独立Bernoulli(p))
    n_correct = votes.sum(axis=1)                 # 正しい投票者の人数
    majority_correct = n_correct > N / 2
    return majority_correct.mean()

print("\n=== 明示的シミュレーション vs 解析解(二項分布) ===")
spot_checks = [(1, 0.3), (1, 0.7), (11, 0.3), (11, 0.7), (51, 0.3), (51, 0.7),
               (101, 0.3), (101, 0.55), (101, 0.7), (101, 0.5), (1, 0.5), (101, 0.45)]
for N, p in spot_checks:
    emp = explicit_vote_simulation(N, p, trials=20000, seed=hash((N, p)) % (2**32))
    analytic = binom.sf(N // 2, N, p)  # P(X > N//2) = 多数決が正しい確率
    print(f"N={N:4d}  p={p:.2f}   simulated={emp:.4f}   analytic={analytic:.4f}   diff={abs(emp-analytic):.4f}")
