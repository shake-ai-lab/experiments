"""p値の実験の共通部品: 自作の t 検定（等分散を仮定する Student 版）と乱数。"""
import numpy as np
from scipy import stats


def ttest_pooled(x, y):
    """x, y: 形が (..., n) の配列。最後の軸が標本。戻り値は (t, p)。"""
    nx, ny = x.shape[-1], y.shape[-1]
    mx, my = x.mean(-1), y.mean(-1)
    vx, vy = x.var(-1, ddof=1), y.var(-1, ddof=1)
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    t = (mx - my) / np.sqrt(sp2 * (1 / nx + 1 / ny))
    p = 2 * stats.t.sf(np.abs(t), nx + ny - 2)
    return t, p


def simulate_p(rng, sims, n, d):
    """2群（各 n 人、標準偏差1）。群1の平均は d、群2は 0。d=0 なら本当は差が無い。"""
    x = rng.standard_normal((sims, n)) + d
    y = rng.standard_normal((sims, n))
    return ttest_pooled(x, y)[1]
