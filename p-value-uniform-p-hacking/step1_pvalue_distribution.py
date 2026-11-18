"""手順1: 自作 t 検定の検証、帰無・対立のときの p値の分布、検出力。"""
import numpy as np
from scipy import stats
from core import simulate_p, ttest_pooled

SIMS = 100_000
N = 20

# --- 自作の t 検定が scipy と一致するか
rng = np.random.default_rng(0)
x = rng.standard_normal((SIMS, N)) + 0.3
y = rng.standard_normal((SIMS, N))
t_mine, p_mine = ttest_pooled(x, y)
res = stats.ttest_ind(x, y, axis=1, equal_var=True)
print(f"自作 vs scipy.ttest_ind: {SIMS} 件で p の最大絶対差 = {np.abs(p_mine - res.pvalue).max():.2e}, "
      f"t の最大絶対差 = {np.abs(t_mine - res.statistic).max():.2e}")

# --- 帰無（差なし）の p値の分布: 種を5つ変えて測る
print("\n[差なし d=0, 各群 n=20, 10万回]")
fr = []
for seed in range(5):
    p = simulate_p(np.random.default_rng(100 + seed), SIMS, N, 0.0)
    fr.append((p < 0.05).mean())
    if seed == 0:
        counts, _ = np.histogram(p, bins=20, range=(0, 1))
        print("  20区間の度数(seed100):", counts.tolist())
        print("  p<0.05: %.4f  p<0.01: %.4f  p>0.95: %.4f  p<0.5: %.4f" % (
            (p < .05).mean(), (p < .01).mean(), (p > .95).mean(), (p < .5).mean()))
        ks = stats.kstest(p, "uniform")
        print(f"  一様分布との KS 統計量 = {ks.statistic:.4f} (p={ks.pvalue:.3f})")
print("  p<0.05 の割合(5つの種):", " ".join(f"{v*100:.2f}%" for v in fr),
      f"  範囲 {min(fr)*100:.2f}-{max(fr)*100:.2f}%")

# --- 本当に差があるとき
print("\n[本当に差があるとき] 検出力 = p<0.05 の割合 (10万回) と、非心 t 分布の理論値")
for d, n in [(0.2, 20), (0.5, 20), (0.8, 20), (0.5, 64), (0.2, 394)]:
    p = simulate_p(np.random.default_rng(7), SIMS, n, d)
    df = 2 * n - 2
    nc = d * np.sqrt(n / 2)
    tc = stats.t.ppf(0.975, df)
    theory = stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
    print(f"  d={d} n={n:>3}: 実測 {(p < .05).mean()*100:5.1f}%  理論 {theory*100:5.1f}%  "
          f"(p<0.01: {(p < .01).mean()*100:.1f}%, p>0.5: {(p > .5).mean()*100:.1f}%)")

# 検出力80%に必要な人数（理論）
for d in (0.2, 0.5, 0.8):
    for n in range(2, 2000):
        df = 2 * n - 2
        tc = stats.t.ppf(0.975, df)
        nc = d * np.sqrt(n / 2)
        if stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc) >= 0.8:
            print(f"  検出力80%に必要な人数(各群) d={d}: n={n}")
            break
