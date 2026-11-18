"""手順3: 他の p値ハッキング。
(a) 結果の指標が k 個あり、一番 p が小さいものだけ報告する（指標同士は独立と仮定）。理論値 1-0.95^k と突き合わせる。
(b) 外れ値を除くかどうかを見てから選ぶ（全データ と 各群で平均から2SD超を除いたもの の、小さい方の p を採用）。
すべて本当は差が無い(d=0)、各群 n=20。"""
import numpy as np
from scipy import stats
from core import ttest_pooled

N = 20
SIMS = 100_000
SEEDS = [21, 22, 23, 24, 25]

# ---- (a) 指標の数 k
KS = [1, 2, 3, 5, 10, 20]
print("(a) 指標 k 個から最小の p を報告。偽陽性率(%) 平均 [種ごとの最小-最大] / 理論値 1-0.95^k")
rows = {k: [] for k in KS}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((SIMS, max(KS), N))
    y = rng.standard_normal((SIMS, max(KS), N))
    p = ttest_pooled(x, y)[1]  # (SIMS, 20)
    for k in KS:
        rows[k].append((p[:, :k].min(1) < 0.05).mean())
    if seed == SEEDS[0]:
        # 自作が scipy と一致
        r = stats.ttest_ind(x[:2000, 0], y[:2000, 0], axis=1)
        print(f"  自作 vs scipy: 2000件で p の最大絶対差 = {np.abs(p[:2000, 0] - r.pvalue).max():.2e}")
import json
json.dump({str(k): float(np.mean(rows[k])) for k in KS}, open("multiple_outcomes.json", "w"))
for k in KS:
    v = np.array(rows[k]) * 100
    print(f"  k={k:>2}: {v.mean():5.1f} [{v.min():.1f}-{v.max():.1f}]  理論 {100*(1-0.95**k):5.1f}")


# ---- (b) 外れ値の除き方を見てから選ぶ
def masked_p(x, y, mx_, my_):
    nx, ny = mx_.sum(1), my_.sum(1)
    ax = (x * mx_).sum(1) / nx
    ay = (y * my_).sum(1) / ny
    vx = (((x - ax[:, None]) ** 2) * mx_).sum(1) / (nx - 1)
    vy = (((y - ay[:, None]) ** 2) * my_).sum(1) / (ny - 1)
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    t = (ax - ay) / np.sqrt(sp2 * (1 / nx + 1 / ny))
    return 2 * stats.t.sf(np.abs(t), nx + ny - 2)


print("\n(b) 全データの p と 2SD超の外れ値を除いた p、小さい方を採用 (各群n=20)")
out = []
all_alone, removed_alone, rm_frac = [], [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed + 100)
    x = rng.standard_normal((SIMS, N))
    y = rng.standard_normal((SIMS, N))
    full = np.ones_like(x, dtype=bool)
    pa = masked_p(x, y, full, full)
    kx = np.abs(x - x.mean(1, keepdims=True)) <= 2 * x.std(1, ddof=1, keepdims=True)
    ky = np.abs(y - y.mean(1, keepdims=True)) <= 2 * y.std(1, ddof=1, keepdims=True)
    pb = masked_p(x, y, kx, ky)
    if seed == SEEDS[0]:
        i = 5
        ref = stats.ttest_ind(x[i][kx[i]], y[i][ky[i]]).pvalue
        print(f"  除外後の自作 p と scipy の差(1例) = {abs(ref - pb[i]):.2e}")
    all_alone.append((pa < 0.05).mean())
    removed_alone.append((pb < 0.05).mean())
    out.append((np.minimum(pa, pb) < 0.05).mean())
    rm_frac.append(1 - (kx.sum() + ky.sum()) / (2 * x.size))
for name, v in [("全データだけ", all_alone), ("除外後だけ", removed_alone), ("小さい方を採用", out)]:
    v = np.array(v) * 100
    print(f"  {name}: {v.mean():.2f}% [{v.min():.2f}-{v.max():.2f}]")
print(f"  除外された点の割合: {np.mean(rm_frac)*100:.2f}%")
