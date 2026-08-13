"""「標本サイズが30以上なら正規分布とみなしてよい」を実測で検証する。

母集団の分布を7種類用意し、標本サイズ n を変えながら標本平均を2万回ずつ作る。
その分布が標準正規分布からどれだけ離れているかを、
  - コルモゴロフ・スミルノフ統計量(累積分布の最大のズレ)
  - 歪度(左右の非対称さ)
の2つで測る。理論上は「標本平均の歪度 = 母集団の歪度 / sqrt(n)」になるはずなので、
それも一緒に確かめる。

結果は results.json に落とす。図は figures.py。
"""

import json

import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260813)
TRIALS = 20_000  # 各 (分布, n) で作る標本平均の個数
NS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]

# 母集団。歪度の小さい順に並べてある。
# sampler は (行数, 列数) を受けてその形の乱数を返す。
DISTS = [
    ("一様分布 U(0,1)", lambda r, c: RNG.uniform(0.0, 1.0, (r, c)), 0.0),
    ("コイン投げ p=0.5", lambda r, c: RNG.binomial(1, 0.5, (r, c)).astype(float), 0.0),
    ("指数分布 Exp(1)", lambda r, c: RNG.exponential(1.0, (r, c)), 2.0),
    ("カイ二乗分布 df=1", lambda r, c: RNG.chisquare(1, (r, c)), np.sqrt(8.0)),
    ("対数正規 σ=1", lambda r, c: RNG.lognormal(0.0, 1.0, (r, c)), 6.1848),
    ("まれな事象 p=0.01", lambda r, c: RNG.binomial(1, 0.01, (r, c)).astype(float), 9.8489),
    # コーシー分布には平均も分散も存在しない。中心極限定理の前提を満たさない。
    ("コーシー分布", lambda r, c: RNG.standard_cauchy((r, c)), None),
]

# 2万件のKS検定の5%棄却点。これを下回れば「正規分布と区別がつかない」と言える。
KS_CRITICAL = 1.36 / np.sqrt(TRIALS)
# 実用上の目安として、累積分布のズレが1%未満なら形はほぼ正規とみなす。
KS_PRACTICAL = 0.01

CHUNK = 2_000_000  # 一度に作る乱数の上限(メモリ対策)


def sample_means(sampler, n, trials):
    """標本サイズ n の標本平均を trials 個作る"""
    per_chunk = max(1, CHUNK // n)
    out = np.empty(trials)
    done = 0
    while done < trials:
        rows = min(per_chunk, trials - done)
        out[done : done + rows] = sampler(rows, n).mean(axis=1)
        done += rows
    return out


results = {}
for name, sampler, pop_skew in DISTS:
    rows = []
    for n in NS:
        m = sample_means(sampler, n, TRIALS)
        # 標準化は実測の平均と標準偏差で行う。位置と広がりを揃えたうえで
        # 「形」だけが正規分布とどれだけ違うかを見たいため。
        # コーシー分布は分散が存在せず標準偏差が標本ごとに暴れるので、
        # 中央値と四分位範囲で揃える(正規分布なら IQR = 1.349σ)。
        if pop_skew is None:
            centre = np.median(m)
            q1, q3 = np.percentile(m, [25, 75])
            scale = (q3 - q1) / 1.3489795
        else:
            centre, scale = m.mean(), m.std(ddof=1)
        z = (m - centre) / scale
        ks = stats.kstest(z, "norm").statistic
        rows.append(
            {
                "n": n,
                "ks": float(ks),
                "skew": float(stats.skew(m)),
                "kurtosis": float(stats.kurtosis(m)),  # 正規分布なら0(過剰尖度)
                # 裾の重さの実感用: 標準化した標本平均が3を超えた割合。
                # 正規分布なら0.27%になるはず。
                "beyond_3sigma": float(np.mean(np.abs(z) > 3)),
            }
        )
        print(f"{name} n={n:5d} KS={ks:.4f} skew={rows[-1]['skew']:+.3f}", flush=True)

    results[name] = {
        "pop_skew": pop_skew,
        "rows": rows,
        # 各基準を最初に満たした n。満たさなければ None。
        "n_for_critical": next((r["n"] for r in rows if r["ks"] < KS_CRITICAL), None),
        "n_for_practical": next((r["n"] for r in rows if r["ks"] < KS_PRACTICAL), None),
    }

# 標本平均のヒストグラム(図用)。代表的な分布と n だけ保存する。
hist = {}
for name, sampler, pop_skew in DISTS:
    hist[name] = {}
    for n in [1, 5, 30, 100, 1000]:
        m = sample_means(sampler, n, TRIALS)
        if pop_skew is None:
            centre = np.median(m)
            q1, q3 = np.percentile(m, [25, 75])
            scale = (q3 - q1) / 1.3489795
        else:
            centre, scale = m.mean(), m.std(ddof=1)
        z = (m - centre) / scale
        counts, edges = np.histogram(np.clip(z, -5, 5), bins=80, range=(-5, 5), density=True)
        hist[name][str(n)] = {"counts": counts.tolist(), "edges": edges.tolist()}

json.dump(
    {
        "trials": TRIALS,
        "ns": NS,
        "ks_critical": float(KS_CRITICAL),
        "ks_practical": KS_PRACTICAL,
        "results": results,
        "hist": hist,
    },
    open("results.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
print("\nwrote results.json")

for name, r in results.items():
    print(f"{name:22s} 歪度{str(r['pop_skew']):>7s}  実用基準を満たすn: {r['n_for_practical']}")
