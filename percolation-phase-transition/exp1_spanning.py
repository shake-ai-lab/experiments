# -*- coding: utf-8 -*-
"""
2次元サイトパーコレーションの相転移を実測する。

- L x L 格子の各サイトを確率 p で占有
- scipy.ndimage.label でクラスタ検出
- 上下を貫通するクラスタの出現確率と最大クラスタサイズを p ごとに測定
- 各試行で「初めて貫通する p (p_span)」を二分探索で求め、
  転移点と転移幅の格子サイズ依存性(有限サイズ効果)を定量化する
"""
import time
import numpy as np
from scipy import ndimage

rng = np.random.default_rng(20260808)

P_GRID = np.round(np.arange(0.50, 0.70001, 0.005), 4)  # 41点
L_LIST = [64, 128, 256, 512, 1024]
TRIALS = {64: 1000, 128: 600, 256: 300, 512: 150, 1024: 80}

STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 4近傍


def spans_vertically(labels):
    """上端と下端の行に共通の(0でない)ラベルがあれば貫通。"""
    top = np.unique(labels[0])
    bot = np.unique(labels[-1])
    common = np.intersect1d(top, bot)
    return np.any(common > 0)


def measure_trial(r, p_grid):
    """1つの乱数場 r に対し、各 p での貫通有無と最大クラスタ比率を返す。"""
    span = np.zeros(len(p_grid), dtype=bool)
    maxfrac = np.zeros(len(p_grid))
    n_sites = r.size
    for i, p in enumerate(p_grid):
        occ = r < p
        labels, n = ndimage.label(occ, structure=STRUCT)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        maxfrac[i] = sizes.max() / n_sites
        span[i] = spans_vertically(labels)
    return span, maxfrac


def p_span_bisect(r, lo=0.40, hi=0.80, tol=1e-4):
    """この乱数場で初めて上下貫通する占有率を二分探索。"""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        occ = r < mid
        labels, _ = ndimage.label(occ, structure=STRUCT)
        if spans_vertically(labels):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


results = {}
for L in L_LIST:
    t0 = time.time()
    n_tr = TRIALS[L]
    span_count = np.zeros(len(P_GRID))
    maxfrac_sum = np.zeros(len(P_GRID))
    p_spans = np.zeros(n_tr)
    for t in range(n_tr):
        r = rng.random((L, L))
        span, maxfrac = measure_trial(r, P_GRID)
        span_count += span
        maxfrac_sum += maxfrac
        p_spans[t] = p_span_bisect(r)
    results[L] = dict(
        span_prob=span_count / n_tr,
        maxfrac=maxfrac_sum / n_tr,
        p_spans=p_spans,
    )
    print(f"L={L:5d}  trials={n_tr}  "
          f"p_span mean={p_spans.mean():.4f}  std={p_spans.std():.4f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

np.savez(
    "C:/tmp/percolation-phase-transition-experiment/exp1_results.npz",
    p_grid=P_GRID,
    L_list=np.array(L_LIST),
    **{f"span_{L}": results[L]["span_prob"] for L in L_LIST},
    **{f"maxfrac_{L}": results[L]["maxfrac"] for L in L_LIST},
    **{f"pspans_{L}": results[L]["p_spans"] for L in L_LIST},
)

# ---- 解析: p_half(貫通確率50%の点), 転移幅, p_c外挿 ----
print("\n=== analysis ===")
p_half = {}
width = {}
for L in L_LIST:
    ps = np.sort(results[L]["p_spans"])
    p_half[L] = np.median(ps)
    q25, q75 = np.percentile(ps, [25, 75])
    width[L] = q75 - q25
    # 固定 p での貫通確率の例
    sp = results[L]["span_prob"]
    print(f"L={L:5d}  p_half(median p_span)={p_half[L]:.4f}  "
          f"IQR width={width[L]:.4f}  std={ps.std():.4f}")

# 有限サイズスケーリング: p_half(L) = p_c + a * L^(-1/nu), nu=4/3
x = np.array([L ** (-3.0 / 4.0) for L in L_LIST])
y = np.array([p_half[L] for L in L_LIST])
A = np.vstack([x, np.ones_like(x)]).T
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
pc_extrap = coef[1]
print(f"\np_c extrapolated (nu=4/3): {pc_extrap:.5f}  (theory 0.59275)")
print(f"largest-L median p_span:   {p_half[L_LIST[-1]]:.5f}")

# 転移幅のスケーリング指数: width ~ L^(-1/nu) → log-log 傾き
lw = np.log(np.array([width[L] for L in L_LIST]))
lL = np.log(np.array(L_LIST, dtype=float))
slope = np.polyfit(lL, lw, 1)[0]
print(f"transition width scaling: width ~ L^({slope:.3f})  (theory -0.75)")
