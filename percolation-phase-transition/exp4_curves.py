# -*- coding: utf-8 -*-
"""
exp1 の結果から図を作る。
- fig_spanning.png: 貫通確率 vs p の曲線群(縦2段: 貫通確率 / 最大クラスタ比率)
- fig_finite_size.png: 転移点の外挿と転移幅のスケーリング(縦2段)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

d = np.load("C:/tmp/percolation-phase-transition-experiment/exp1_results.npz")
p = d["p_grid"]
L_list = list(d["L_list"])
PC = 0.59275

# L(桁が変わる系列)は青系の明→暗で塗る
cmap = plt.get_cmap("viridis")
colors = {L: cmap(0.88 - 0.82 * i / (len(L_list) - 1))
          for i, L in enumerate(L_list)}

# ---------- 図: 貫通確率と最大クラスタ ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9.6), dpi=110, sharex=True)
for L in L_list:
    ax1.plot(p, d[f"span_{L}"], "-", lw=2, color=colors[L], label=f"L = {L}")
    ax2.plot(p, d[f"maxfrac_{L}"], "-", lw=2, color=colors[L], label=f"L = {L}")
for ax in (ax1, ax2):
    ax.axvline(PC, color="#c92a2a", ls="--", lw=1.3, alpha=0.8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, title="格子サイズ", title_fontsize=11)
ax1.text(PC + 0.002, 0.06, "理論臨界点\np = 0.5927", color="#c92a2a", fontsize=11)
ax1.set_ylabel("上下を貫通するクラスタが出る確率", fontsize=12)
ax1.set_title("格子が大きいほど、転移は「崖」になる", fontsize=14)
ax2.set_ylabel("最大クラスタのサイズ / 全サイト数", fontsize=12)
ax2.set_xlabel("占有率 p", fontsize=12)
fig.tight_layout()
fig.savefig("C:/tmp/percolation-phase-transition-experiment/fig_spanning.png")

# ---------- 図: 有限サイズスケーリング ----------
p_half = np.array([np.median(d[f"pspans_{L}"]) for L in L_list])
width = np.array([np.percentile(d[f"pspans_{L}"], 75)
                  - np.percentile(d[f"pspans_{L}"], 25) for L in L_list])
Ls = np.array(L_list, dtype=float)

x = Ls ** (-3.0 / 4.0)
A = np.vstack([x, np.ones_like(x)]).T
coef, *_ = np.linalg.lstsq(A, p_half, rcond=None)
pc_extrap = coef[1]

slope, intercept = np.polyfit(np.log(Ls), np.log(width), 1)

# ブートストラップで中央値と外挿値の誤差を見積もる
rng = np.random.default_rng(0)
B = 3000
boot_pc = np.zeros(B)
boot_med = np.zeros((B, len(L_list)))
for b in range(B):
    meds = []
    for j, L in enumerate(L_list):
        ps = d[f"pspans_{L}"]
        meds.append(np.median(rng.choice(ps, size=len(ps), replace=True)))
    meds = np.array(meds)
    boot_med[b] = meds
    c, *_ = np.linalg.lstsq(A, meds, rcond=None)
    boot_pc[b] = c[1]
med_err = boot_med.std(axis=0)
pc_err = boot_pc.std()
print(f"pc_extrap={pc_extrap:.5f} +- {pc_err:.5f}")
print("median errors:", np.round(med_err, 5))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 9.2), dpi=110)
xs = np.linspace(0, x.max() * 1.1, 100)
ax1.plot(xs, coef[0] * xs + coef[1], "-", color="#adb5bd", lw=1.5)
ax1.errorbar(x, p_half, yerr=med_err, fmt="o", ms=8, color="#1c7ed6",
             ecolor="#74a9d8", elinewidth=1.5, capsize=4)
for xi, yi, L in zip(x, p_half, L_list):
    ax1.annotate(f"L={L}", (xi, yi), textcoords="offset points",
                 xytext=(8, -14), fontsize=10)
ax1.axhline(PC, color="#c92a2a", ls="--", lw=1.3)
ax1.text(0.028, PC - 0.00015, "理論値 0.59275", color="#c92a2a", fontsize=11,
         va="top")
ax1.errorbar([0], [pc_extrap], yerr=[pc_err], fmt="s", ms=9, color="#e8590c",
             ecolor="#e8590c", elinewidth=1.5, capsize=4, zorder=5)
ax1.annotate(f"外挿値 {pc_extrap:.5f}", (0, pc_extrap),
             textcoords="offset points", xytext=(10, 8),
             fontsize=11, color="#e8590c")
ax1.set_xlabel(r"$L^{-3/4}$(左端が無限大の格子)", fontsize=12)
ax1.set_ylabel("貫通確率が50%になる占有率", fontsize=12)
ax1.set_title("転移点を無限サイズへ外挿する(誤差棒はブートストラップ)",
              fontsize=13)
ax1.grid(alpha=0.25)

ax2.loglog(Ls, width, "o", ms=9, color="#1c7ed6")
lfit = np.exp(intercept) * Ls ** slope
ax2.loglog(Ls, lfit, "-", color="#adb5bd", lw=1.5,
           label=f"実測の傾き {slope:.2f}(理論 −0.75)")
ax2.set_xlabel("格子サイズ L", fontsize=12)
ax2.set_ylabel("転移幅(p_spanの四分位範囲)", fontsize=12)
ax2.set_title("転移の「鈍さ」はべき則で消えていく", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("C:/tmp/percolation-phase-transition-experiment/fig_finite_size.png")

print(f"pc_extrap={pc_extrap:.5f}")
print(f"width slope={slope:.3f}")
for L in L_list:
    sp = d[f"span_{L}"]
    i58 = np.argmin(np.abs(p - 0.58)); i61 = np.argmin(np.abs(p - 0.61))
    print(f"L={L}: span(0.58)={sp[i58]:.3f} span(0.61)={sp[i61]:.3f} "
          f"p_half={np.median(d[f'pspans_{L}']):.4f} width={np.percentile(d[f'pspans_{L}'],75)-np.percentile(d[f'pspans_{L}'],25):.4f}")
print("saved figures")
