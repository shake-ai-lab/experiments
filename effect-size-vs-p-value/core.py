"""効果量 d と p 値の自作実装（2群・等分散を仮定する Student の t 検定）。"""
import numpy as np
from scipy import stats


def cohen_d(a, b):
    """Cohen の d = (平均差) / プールした標準偏差。a, b は (反復, n) の配列でも 1 次元でもよい。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = a.shape[-1], b.shape[-1]
    v1 = a.var(axis=-1, ddof=1)
    v2 = b.var(axis=-1, ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (a.mean(axis=-1) - b.mean(axis=-1)) / sp


def t_and_p(a, b):
    """d から t を作り、t 分布の上側確率から両側 p を出す。"""
    n1, n2 = np.asarray(a).shape[-1], np.asarray(b).shape[-1]
    d = cohen_d(a, b)
    t = d / np.sqrt(1 / n1 + 1 / n2)
    p = 2 * stats.t.sf(np.abs(t), n1 + n2 - 2)
    return t, p, d


def simulate(true_d, n, reps, rng):
    """処置群 N(true_d,1)、対照群 N(0,1)、各群 n 人を reps 回。d の推定値と p を返す。"""
    treat = rng.normal(true_d, 1.0, size=(reps, n))
    ctrl = rng.normal(0.0, 1.0, size=(reps, n))
    t, p, d = t_and_p(treat, ctrl)
    return d, p
