"""95%信頼区間の「含む割合」を実測する。
1) 100本の区間 GIF  2) z と t の被覆率  3) 歪んだ母集団での被覆率  4) 95%と99%の幅
"""
import sys
import numpy as np
from scipy import stats

REPS = 100_000
CHUNK = 5_000


def intervals(x, level, use_t=True):
    n = x.shape[1]
    m = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    q = stats.t.ppf(0.5 + level / 2, n - 1) if use_t else stats.norm.ppf(0.5 + level / 2)
    h = q * s / np.sqrt(n)
    return m - h, m + h


def coverage(sampler, n, level, use_t, true_mu, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    hit = 0
    width = 0.0
    done = 0
    while done < reps:
        k = min(CHUNK, reps - done)
        lo, hi = intervals(sampler(rng, (k, n)), level, use_t)
        hit += int(((lo <= true_mu) & (true_mu <= hi)).sum())
        width += float((hi - lo).sum())
        done += k
    p = hit / reps
    se = np.sqrt(p * (1 - p) / reps)
    return p, se, width / reps


MU, SD = 50.0, 10.0


def normal(rng, shape):
    return rng.normal(MU, SD, shape)


# 強く歪んだ分布: 対数正規 (sigma=1.5)。真の平均 exp(s^2/2)
LS = 1.5
LOGN_MU = float(np.exp(LS**2 / 2))


def lognorm(rng, shape):
    return rng.lognormal(0.0, LS, shape)


def main():
    print("== 1) 正規母集団(平均50, SD10) t区間 95% n=10, 100本 (seed=7)")
    rng = np.random.default_rng(7)
    lo, hi = intervals(rng.normal(MU, SD, (100, 10)), 0.95)
    miss = int(((MU < lo) | (MU > hi)).sum())
    print(f"外れた本数 {miss}/100")

    print("\n== 2) 正規母集団: n別 z(1.96) vs t の被覆率 (95%), 10万回")
    print("n, z, t, 理論z(自由度n-1のt分布で1.96を使った場合)")
    for n in [3, 5, 10, 20, 30, 100]:
        pz, sz, _ = coverage(normal, n, 0.95, False, MU, seed=1)
        pt, st, _ = coverage(normal, n, 0.95, True, MU, seed=2)
        th = 1 - 2 * stats.t.sf(stats.norm.ppf(0.975) * np.sqrt(n) / np.sqrt(n), n - 1)
        # 理論: 区間 mean±1.96 s/sqrt(n) が mu を含む <=> |T| <= 1.96, T~t(n-1)
        print(f"n={n:4d}  z={pz:.4f}±{1.96*sz:.4f}  t={pt:.4f}±{1.96*st:.4f}  理論z={th:.4f}")

    print("\n== 3) 対数正規(sigma=1.5)母集団: t区間 95% の被覆率")
    sk = stats.skew(np.random.default_rng(0).lognormal(0, LS, 2_000_000))
    print(f"真の平均={LOGN_MU:.4f}")
    for n in [5, 10, 30, 100, 300, 1000, 3000]:
        reps = REPS if n <= 1000 else 20_000
        ps = []
        for seed in (11, 12, 13):
            p, se, _ = coverage(lognorm, n, 0.95, True, LOGN_MU, reps=reps, seed=seed)
            ps.append(p)
        print(f"n={n:5d} reps={reps} 3種の値 {[round(v,4) for v in ps]} 平均={np.mean(ps):.4f}")

    print("\n== 4) 幅: 95% vs 99% (正規, SD10)")
    for n in [5, 10, 30, 100]:
        _, _, w95 = coverage(normal, n, 0.95, True, MU, seed=3)
        p99, _, w99 = coverage(normal, n, 0.99, True, MU, seed=4)
        print(f"n={n:4d} 幅95={w95:.3f} 幅99={w99:.3f} 比={w99/w95:.4f} 99%被覆={p99:.4f}"
              f" 理論比={stats.t.ppf(0.995,n-1)/stats.t.ppf(0.975,n-1):.4f}")
    # 同じ幅を縮めるのに必要なnの比 (z近似)
    print("99%で95%と同じ幅にするnの倍率(z近似)", (stats.norm.ppf(0.995) / stats.norm.ppf(0.975)) ** 2)

    print("\n== 5) scipy.stats.t.interval との一致")
    x = np.random.default_rng(5).normal(MU, SD, (1000, 10))
    lo, hi = intervals(x, 0.95)
    mx = 0.0
    for i in range(1000):
        a, b = stats.t.interval(0.95, 9, loc=x[i].mean(), scale=stats.sem(x[i]))
        mx = max(mx, abs(a - lo[i]), abs(b - hi[i]))
    print("最大差", mx)


if __name__ == "__main__":
    main()
