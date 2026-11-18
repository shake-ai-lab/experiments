"""実験本体。図を ../../forsubwork/src/content/blog/images/effect-size-vs-p-value/ に出す。"""
import os
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import cohen_d, t_and_p, simulate

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
OUT = r"C:\Users\kouta\project\forsubwork\src\content\blog\images\effect-size-vs-p-value"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]

# ---- 0. scipy との突き合わせ ----
rng = np.random.default_rng(12345)
maxdt = maxdp = 0.0
for _ in range(1000):
    n1, n2 = rng.integers(5, 200, size=2)
    a = rng.normal(rng.normal(), rng.uniform(0.5, 3), n1)
    b = rng.normal(rng.normal(), rng.uniform(0.5, 3), n2)
    t, p, d = t_and_p(a, b)
    r = stats.ttest_ind(a, b, equal_var=True)
    maxdt = max(maxdt, abs(float(t) - r.statistic))
    maxdp = max(maxdp, abs(float(p) - r.pvalue))
    # d は定義どおりの素朴な計算とも突き合わせる
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    assert abs(float(d) - (a.mean() - b.mean()) / sp) < 1e-12
print(f"[0] scipy.stats.ttest_ind との最大差: t {maxdt:.2e}, p {maxdp:.2e} (1000 組)")

# ---- 1. 真の d=0.1 で人数を増やす ----
TRUE_D = 0.1
NS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
REPS = 2000
res1 = {}  # seed -> {n: (d array, p array)}
for s in SEEDS:
    rng = np.random.default_rng(s)
    res1[s] = {}
    for n in NS:
        res1[s][n] = simulate(TRUE_D, n, REPS, rng)
print(f"\n[1] 真の d={TRUE_D}、各群 n 人、{REPS} 回 × 種{SEEDS}")
print("n, 有意割合(種ごと), 中央値p, d推定の平均, d推定の2.5-97.5%点")
sig_by_n = {}
for n in NS:
    sigs = [np.mean(res1[s][n][1] < 0.05) for s in SEEDS]
    sig_by_n[n] = sigs
    dall = np.concatenate([res1[s][n][0] for s in SEEDS])
    pall = np.concatenate([res1[s][n][1] for s in SEEDS])
    lo, hi = np.percentile(dall, [2.5, 97.5])
    print(f"{n:6d}  sig={['%.3f' % x for x in sigs]}  medp={np.median(pall):.3g}  "
          f"dmean={dall.mean():.4f}  d95=[{lo:.3f},{hi:.3f}]")
# 理論の検出力（非心 t）
print("理論の検出力（非心t）:")
for n in NS:
    df = 2 * n - 2
    nc = TRUE_D * np.sqrt(n / 2)
    tc = stats.t.ppf(0.975, df)
    pw = stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
    print(f"  n={n:6d} power={pw:.3f}")

# 図1: 上段 p値（対数）、下段 d 推定
fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
ax = axes[0]
s0 = 0
for i, n in enumerate(NS):
    p = res1[s0][n][1]
    x = np.full(300, i) + np.random.default_rng(i).uniform(-0.25, 0.25, 300)
    ax.scatter(x, p[:300], s=6, alpha=0.35, color="#4c72b0")
    ax.plot([i - 0.35, i + 0.35], [np.median(p)] * 2, color="#c44e52", lw=2)
ax.axhline(0.05, color="k", ls="--", lw=1)
ax.text(len(NS) - 0.5, 0.05 / 6, "p = 0.05", ha="right", fontsize=11)
ax.set_yscale("log")
ax.set_ylim(1e-25, 2)
ax.set_ylabel("p 値(対数軸)", fontsize=13)
ax.set_title(f"真の差は d={TRUE_D} で固定。人数だけを増やした(各 {300} 回を表示、赤線は中央値)", fontsize=13)
ax.tick_params(labelsize=11)
ax = axes[1]
for i, n in enumerate(NS):
    d = res1[s0][n][0]
    x = np.full(300, i) + np.random.default_rng(i).uniform(-0.25, 0.25, 300)
    ax.scatter(x, d[:300], s=6, alpha=0.35, color="#55a868")
    ax.plot([i - 0.35, i + 0.35], [np.mean(d)] * 2, color="#c44e52", lw=2)
ax.axhline(TRUE_D, color="k", ls="--", lw=1)
ax.text(len(NS) - 0.5, TRUE_D + 0.12, f"真の d = {TRUE_D}(破線)", ha="right", fontsize=11)
ax.set_ylabel("d の推定値", fontsize=13)
ax.set_xlabel("各群の人数 n", fontsize=13)
ax.set_xticks(range(len(NS)))
ax.set_xticklabels([f"{n:,}" for n in NS], fontsize=11)
ax.tick_params(labelsize=11)
ax.set_title("同じ実験の d の推定値は、人数が増えるほど真の値のまわりに絞られる", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "1.png"), dpi=130)
plt.close(fig)

# ---- 2. 目安 0.2/0.5/0.8 の実感 ----
print("\n[2] d と分布の重なり（理論 vs 乱数 200 万人ずつ）")
rng = np.random.default_rng(7)
N = 2_000_000
rows = []
for d in (0.2, 0.5, 0.8):
    a = rng.normal(d, 1, N)
    b = rng.normal(0, 1, N)
    ps_th = stats.norm.cdf(d / np.sqrt(2))
    ovl_th = 2 * stats.norm.cdf(-d / 2)
    u3_th = stats.norm.cdf(d)
    ps_sim = np.mean(a > b)
    # 重なり: 2 つのヒストグラム密度の min の面積
    bins = np.linspace(-6, 7, 1301)
    ha, _ = np.histogram(a, bins, density=True)
    hb, _ = np.histogram(b, bins, density=True)
    ovl_sim = np.sum(np.minimum(ha, hb)) * (bins[1] - bins[0])
    u3_sim = np.mean(a > np.median(b))
    print(f"d={d}: 重なり 理論{ovl_th:.4f} 実測{ovl_sim:.4f} | 優越確率 理論{ps_th:.4f} 実測{ps_sim:.4f} | "
          f"U3(処置群の平均が対照群の何%点か) 理論{u3_th:.4f} 実測{u3_sim:.4f}")
    rows.append(d)

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
xs = np.linspace(-4, 5, 900)
for ax, d in zip(axes, (0.2, 0.5, 0.8)):
    ya = stats.norm.pdf(xs, d, 1)
    yb = stats.norm.pdf(xs, 0, 1)
    ax.fill_between(xs, np.minimum(ya, yb), color="#9e9e9e", alpha=0.55, label="重なり")
    ax.plot(xs, yb, color="#4c72b0", lw=2.2, label="対照群")
    ax.plot(xs, ya, color="#c44e52", lw=2.2, label="処置群")
    ps = stats.norm.cdf(d / np.sqrt(2))
    ovl = 2 * stats.norm.cdf(-d / 2)
    ax.set_title(f"d = {d}: 分布の重なり {ovl*100:.0f}%、ランダムに1人ずつ選ぶと処置群が上回る確率 {ps*100:.0f}%",
                 fontsize=12.5)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=11)
    ax.tick_params(labelsize=11)
axes[-1].set_xlabel("測定値(標準偏差を 1 とした単位)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "2.png"), dpi=130)
plt.close(fig)

# ---- 3. 勝者の呪い ----
print("\n[3] 有意になった研究だけの d（真の d=0.2、各群 n 人、各 400,000 回 × 種{0,1,2}）")
REPS3 = 400_000
res3 = {}
for n in (20, 50, 100, 200, 500):
    ds, ps = [], []
    per_seed = []
    for s in SEEDS:
        rng = np.random.default_rng(100 + s)
        d, p = simulate(0.2, n, REPS3 // 4 if n >= 200 else REPS3, rng)
        ds.append(d); ps.append(p)
        sig = p < 0.05
        per_seed.append((sig.mean(), d[sig].mean(), d[sig & (d > 0)].mean()))
    d = np.concatenate(ds); p = np.concatenate(ps)
    sig = p < 0.05
    res3[n] = (d, p)
    print(f"n={n:4d}: 有意割合 {['%.4f' % x[0] for x in per_seed]} 全体 {sig.mean():.4f} | "
          f"d平均(全回) {d.mean():.4f} | 有意回だけの d平均 {d[sig].mean():.4f} "
          f"(種別 {['%.4f' % x[1] for x in per_seed]}) | 有意かつ d>0: {d[sig & (d>0)].mean():.4f} | "
          f"有意回で d<0 の割合 {np.mean(d[sig] < 0):.5f}")

d, p = res3[20]
sig = p < 0.05
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
bins = np.linspace(-1.2, 1.6, 113)
axes[0].hist(d, bins, color="#4c72b0", alpha=0.85)
axes[0].axvline(0.2, color="k", ls="--", lw=1.5)
axes[0].axvline(d.mean(), color="#c44e52", lw=2)
axes[0].set_title(f"全ての研究(n=20, 真の d=0.2): d の平均 = {d.mean():.3f}(黒破線が真の値)", fontsize=12.5)
axes[1].hist(d[~sig], bins, color="#bdbdbd", alpha=0.9, label=f"有意でなかった {np.mean(~sig)*100:.0f}%")
axes[1].hist(d[sig], bins, color="#c44e52", alpha=0.9, label=f"p<0.05 になった {np.mean(sig)*100:.0f}%")
axes[1].axvline(0.2, color="k", ls="--", lw=1.5)
axes[1].axvline(d[sig].mean(), color="#8b0000", lw=2)
axes[1].set_title(f"有意になった研究だけを集めると: d の平均 = {d[sig].mean():.3f}", fontsize=12.5)
axes[1].legend(fontsize=11, loc="upper left")
axes[1].set_xlabel("d の推定値", fontsize=13)
for ax in axes:
    ax.set_ylabel("研究の数", fontsize=12)
    ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "3.png"), dpi=130)
plt.close(fig)

# 図4: 人数ごとの「有意になった研究の d 平均」
fig, ax = plt.subplots(figsize=(8, 5))
ns3 = sorted(res3)
means = [res3[n][0][res3[n][1] < 0.05].mean() for n in ns3]
shares = [np.mean(res3[n][1] < 0.05) for n in ns3]
ax.plot(range(len(ns3)), means, "o-", color="#c44e52", lw=2, ms=8)
for i, (m, sh) in enumerate(zip(means, shares)):
    ax.annotate(f"{m:.2f}\n(有意 {sh*100:.0f}%)", (i, m), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=11)
ax.axhline(0.2, color="k", ls="--", lw=1)
ax.text(0, 0.19, "真の d = 0.2(破線)", ha="left", va="top", fontsize=11)
ax.set_xticks(range(len(ns3)))
ax.set_xticklabels([str(n) for n in ns3], fontsize=11)
ax.set_xlabel("各群の人数 n", fontsize=13)
ax.set_ylabel("p<0.05 になった研究の d 平均", fontsize=12)
ax.set_ylim(0.1, 0.95)
ax.set_title("人数が少ないほど、有意になった研究の効果量は過大になる", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "4.png"), dpi=130)
plt.close(fig)
print("\n図を出力:", OUT)
