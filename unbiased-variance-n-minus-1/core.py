"""n で割る分散と n-1 で割る分散を、母分散が既知の母集団から何十万回も標本を取って比べる。

出力: results.json（記事の数字はすべてここから）
"""
import json
import math

import numpy as np
from scipy.special import gammaln

SIGMA = 10.0          # 母標準偏差（母分散 100）
MU = 50.0
NS = [2, 3, 5, 10, 30]
REPS = 400_000        # 1 つの種あたりの標本の数
SEEDS = [0, 1, 2, 3, 4]
CHUNK = 100_000


def my_var(x, ddof):
    """自作: 標本平均からの偏差平方和を (n - ddof) で割る。x は (reps, n)。"""
    n = x.shape[1]
    m = x.mean(axis=1, keepdims=True)
    return ((x - m) ** 2).sum(axis=1) / (n - ddof)


def c4(n):
    return math.sqrt(2.0 / (n - 1)) * math.exp(gammaln(n / 2) - gammaln((n - 1) / 2))


def run_seed(seed, n):
    """1 つの種・1 つの n で、REPS 個の標本から各推定量の平均などを返す。"""
    rng = np.random.default_rng([seed, n])
    acc = {k: 0.0 for k in ["v0", "v1", "s", "ss_true", "ss_mean", "mse0", "mse1", "mse2", "mean_s2n1_sq"]}
    maxdiff = 0.0
    done = 0
    while done < REPS:
        r = min(CHUNK, REPS - done)
        x = rng.normal(MU, SIGMA, size=(r, n))
        v0 = my_var(x, 0)
        v1 = my_var(x, 1)
        # numpy と突き合わせる
        maxdiff = max(maxdiff, np.abs(v0 - x.var(axis=1, ddof=0)).max(), np.abs(v1 - x.var(axis=1, ddof=1)).max())
        ss_mean = v0 * n
        ss_true = ((x - MU) ** 2).sum(axis=1)
        acc["v0"] += v0.sum()
        acc["v1"] += v1.sum()
        acc["s"] += np.sqrt(v1).sum()
        acc["ss_true"] += ss_true.sum()
        acc["ss_mean"] += ss_mean.sum()
        s2 = SIGMA ** 2
        acc["mse0"] += ((ss_mean / n - s2) ** 2).sum()
        acc["mse1"] += ((ss_mean / (n - 1) - s2) ** 2).sum()
        acc["mse2"] += ((ss_mean / (n + 1) - s2) ** 2).sum()
        done += r
    out = {k: v / REPS for k, v in acc.items()}
    out["maxdiff_numpy"] = float(maxdiff)
    return out


def main():
    res = {"sigma": SIGMA, "reps": REPS, "seeds": SEEDS, "n": {}}
    s2 = SIGMA ** 2
    for n in NS:
        per = [run_seed(sd, n) for sd in SEEDS]
        def avg(k): return float(np.mean([p[k] for p in per]))
        def rng_(k): return [float(min(p[k] for p in per)), float(max(p[k] for p in per))]
        row = {
            "ratio_n": avg("v0") / s2, "ratio_n_range": [a / s2 for a in rng_("v0")],
            "theory_ratio_n": (n - 1) / n,
            "var_n1": avg("v1"), "var_n1_range": rng_("v1"),
            "var_n": avg("v0"),
            "s_mean": avg("s"), "s_mean_range": rng_("s"),
            "s_ratio": avg("s") / SIGMA, "c4": c4(n),
            "ss_true_per_n": avg("ss_true") / n, "ss_mean_per_n": avg("ss_mean") / n,
            "mse_n": avg("mse0"), "mse_n1": avg("mse1"), "mse_n2": avg("mse2"),
            "theory_mse_n1": 2 * s2 ** 2 / (n - 1),
            "theory_mse_n": s2 ** 2 * (2 * n - 1) / n ** 2,
            "theory_mse_n2": 2 * s2 ** 2 / (n + 1),
            "maxdiff_numpy": max(p["maxdiff_numpy"] for p in per),
        }
        res["n"][n] = row
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for n, r in res["n"].items():
        print(f"n={n:2d} ratio_n={r['ratio_n']:.4f} (theory {r['theory_ratio_n']:.4f}) range[{r['ratio_n_range'][0]:.4f},{r['ratio_n_range'][1]:.4f}] "
              f"var_n1={r['var_n1']:.2f} [{r['var_n1_range'][0]:.2f},{r['var_n1_range'][1]:.2f}] "
              f"s={r['s_mean']:.3f} ({r['s_ratio']:.4f} vs c4 {r['c4']:.4f}) "
              f"ss_true/n={r['ss_true_per_n']:.2f} ss_mean/n={r['ss_mean_per_n']:.2f} "
              f"mse n/n-1/n+1={r['mse_n']:.0f}/{r['mse_n1']:.0f}/{r['mse_n2']:.0f} "
              f"(theory {r['theory_mse_n']:.0f}/{r['theory_mse_n1']:.0f}/{r['theory_mse_n2']:.0f}) maxdiff={r['maxdiff_numpy']:.1e}")


if __name__ == "__main__":
    main()
