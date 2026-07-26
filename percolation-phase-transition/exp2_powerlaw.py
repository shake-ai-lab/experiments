# -*- coding: utf-8 -*-
"""
臨界点近傍でのクラスタサイズ分布がべき則 n(s) ~ s^(-tau) に乗るかを確認する。
理論値: tau = 187/91 ≈ 2.055 (2次元パーコレーションの普遍指数)

p = 0.55(臨界未満) / 0.5927(臨界) / 0.65(臨界超過) の3点で、
L=2048 の格子でクラスタサイズ分布を測る。貫通(最大)クラスタは分布から除外。
"""
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

rng = np.random.default_rng(42)
STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

L = 2048
TRIALS = 20
P_LIST = [0.55, 0.5927, 0.65]

dist = {}
for p in P_LIST:
    all_sizes = []
    for t in range(TRIALS):
        occ = rng.random((L, L)) < p
        labels, n = ndimage.label(occ, structure=STRUCT)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        # 最大クラスタ(臨界以上では貫通クラスタ)は除外するのが慣例
        sizes[np.argmax(sizes)] = 0
        all_sizes.append(sizes[sizes > 0])
    all_sizes = np.concatenate(all_sizes)
    dist[p] = all_sizes
    print(f"p={p}: clusters={len(all_sizes)}  max size={all_sizes.max()}", flush=True)

# 対数ビンで n(s) (サイズ s のクラスタ数 / 総サイト数) を計算
def log_binned(sizes, n_sites, nbins=40):
    smax = sizes.max()
    bins = np.unique(np.round(np.logspace(0, np.log10(smax), nbins)).astype(int))
    hist, edges = np.histogram(sizes, bins=np.append(bins, bins[-1] + 1))
    widths = np.diff(np.append(bins, bins[-1] + 1)).astype(float)
    centers = np.sqrt(bins * np.append(bins[1:], bins[-1] + 1))
    ns = hist / widths / n_sites
    mask = hist > 0
    return centers[mask], ns[mask]

n_sites = TRIALS * L * L
curves = {p: log_binned(dist[p], n_sites) for p in P_LIST}

# 臨界点での傾き(tau)を s=10..3000 の範囲でフィット
s, ns = curves[0.5927]
mask = (s >= 10) & (s <= 3000)
tau_fit = -np.polyfit(np.log(s[mask]), np.log(ns[mask]), 1)[0]
print(f"\ntau measured at p=0.5927: {tau_fit:.3f}  (theory 2.055)")

# 参考: 臨界未満での同範囲フィットの残差(直線に乗らないことの定量化)
def fit_resid(p, lo=10, hi=3000):
    s, ns = curves[p]
    m = (s >= lo) & (s <= hi)
    if m.sum() < 3:
        return np.nan, np.nan
    c = np.polyfit(np.log(s[m]), np.log(ns[m]), 1)
    resid = np.log(ns[m]) - np.polyval(c, np.log(s[m]))
    return -c[0], np.sqrt(np.mean(resid ** 2))

for p in P_LIST:
    slope, rms = fit_resid(p)
    print(f"p={p}: fitted slope={slope:.3f}, RMS residual(log)={rms:.3f}")

np.savez("C:/tmp/percolation-phase-transition-experiment/exp2_results.npz",
         **{f"s_{p}": curves[p][0] for p in P_LIST},
         **{f"ns_{p}": curves[p][1] for p in P_LIST},
         tau_fit=tau_fit)

# ---- 図: クラスタサイズ分布 ----
fig, ax = plt.subplots(figsize=(7, 5.2), dpi=110)
colors = {0.55: "#4a90d9", 0.5927: "#e8590c", 0.65: "#37b24d"}
labels_ = {0.55: "p = 0.55(臨界未満)", 0.5927: "p = 0.5927(臨界点)",
           0.65: "p = 0.65(臨界超過)"}
for p in P_LIST:
    s, ns = curves[p]
    ax.loglog(s, ns, "o-", ms=4, lw=1.3, color=colors[p], label=labels_[p],
              alpha=0.9)
# 理論傾きのガイド線
sg = np.array([10, 3000])
ax.loglog(sg, 2.2e-2 * sg ** (-2.055), "--", color="#868e96", lw=1.5,
          label=r"傾き $-2.055$(理論値 $\tau$)")
ax.set_xlabel("クラスタサイズ s(サイト数)", fontsize=12)
ax.set_ylabel("クラスタ数密度 n(s)", fontsize=12)
ax.set_title(f"クラスタサイズ分布(L=2048)— 臨界点だけが直線に乗る", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("C:/tmp/percolation-phase-transition-experiment/fig_powerlaw.png")
print("saved fig_powerlaw.png")
