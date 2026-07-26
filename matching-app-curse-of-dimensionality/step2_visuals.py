"""可視化一式:
(1) 雑多な質問(ノイズ次元)を増やすほど、本当の最良マッチの順位がどう埋もれていくか(平均順位+ランダム基準線)
(2) hit@1(ぴったり1位を当てられる確率)とhit@10(トップ10に入る確率)の低下
(3) d_noise=0,10,50,200での順位分布ヒストグラム小組(小さいマルチプル)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11, "lines.linewidth": 2.0})

OUTDIR = "C:/tmp/matching-app-curse-of-dimensionality"
d = np.load(f"{OUTDIR}/results.npz")

d_noise = d["d_noise"]
mean_ranks = d["mean_ranks"]
std_ranks = d["std_ranks"]
hit1_rates = d["hit1_rates"]
hit10_rates = d["hit10_rates"]
N = int(d["N"])
K_SIGNAL = int(d["K_SIGNAL"])
R = int(d["R"])
random_baseline_rank = float(d["random_baseline_rank"])
random_baseline_hit1 = float(d["random_baseline_hit1"])
random_baseline_hit10 = float(d["random_baseline_hit10"])

COL = "#e4572e"
COL2 = "#2e6ee4"

xpos = np.arange(len(d_noise))
xlabels = [str(v) for v in d_noise]

# ---------------- (1) mean rank vs noise dims ----------------
fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.axhline(random_baseline_rank, color="#999", lw=1.3, ls="--", label=f"完全ランダム基準 (期待順位={random_baseline_rank:.1f}位)")
ax.axhline(1, color="#2fa96b", lw=1.3, ls=":", label="理想(常に1位を当てる)")
ax.plot(xpos, mean_ranks, "o-", color=COL, lw=2.2, ms=7, zorder=5, label="本当の最良マッチの平均順位")
ax.fill_between(xpos, mean_ranks - std_ranks, mean_ranks + std_ranks, color=COL, alpha=0.15)
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels, fontsize=9.5)
ax.set_xlabel(f"「雑多な質問」の数(ノイズ次元数)  ※本当に相性を決める価値観は{K_SIGNAL}次元固定")
ax.set_ylabel(f"本当の最良マッチの順位(候補者N={N}人中、{R}試行平均)")
ax.set_title("たった1問の無関係な質問を足しただけで、本当の最良マッチは平均31位まで埋もれた", fontsize=12)
ax.legend(fontsize=9.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig1_rank_degradation.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig1_rank_degradation.png")

# ---------------- (2) hit@1 / hit@10 rates ----------------
fig, ax = plt.subplots(figsize=(9.5, 5.0))
ax.plot(xpos, hit1_rates * 100, "o-", color=COL, lw=2.2, ms=7, label="hit@1: ちょうど1位に推薦できた確率")
ax.plot(xpos, hit10_rates * 100, "s-", color=COL2, lw=2.2, ms=7, label="hit@10: トップ10に入れられた確率")
ax.axhline(random_baseline_hit1 * 100, color=COL, lw=1, ls=":", alpha=0.7)
ax.axhline(random_baseline_hit10 * 100, color=COL2, lw=1, ls=":", alpha=0.7)
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels, fontsize=9.5)
ax.set_xlabel(f"「雑多な質問」の数(ノイズ次元数)  ※本当に相性を決める価値観は{K_SIGNAL}次元固定")
ax.set_ylabel("確率 (%)")
ax.set_title("hit@1は100%→1%、hit@10は100%→5%まで急落(点線はランダム推薦の基準線)", fontsize=12)
ax.legend(fontsize=9.5)
ax.set_ylim(-3, 103)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig2_hit_rates.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig2_hit_rates.png")

# ---------------- (3) rank distribution small multiples ----------------
# 2x2グリッド。d=0のパネルだけ「1位に300試行全部」というスケール(0-300)で、
# 残り3パネルは共通の小さいスケールにして分布の形が読めるようにする。
# 一様分布の期待値(300試行/30ビン=10試行/ビン)の線を引いて「くじ引きにどこまで近いか」を見せる。
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
axes = axes.ravel()
settings = [("0", "雑多な質問 0問 (価値観3次元のみ)"), ("12", "雑多な質問 12問"),
            ("60", "雑多な質問 60問"), ("200", "雑多な質問 200問")]
bins = np.linspace(1, N, 31)
n_bins = len(bins) - 1
uniform_per_bin = R / n_bins  # 完全一様(くじ引き)なら1ビンあたりこの試行数
small_ymax = 0
for key, _ in settings[1:]:
    counts, _edges = np.histogram(d[f"ranks_d{key}"], bins=bins)
    small_ymax = max(small_ymax, counts.max())
small_ymax = small_ymax * 1.25
for i, (ax, (key, title)) in enumerate(zip(axes, settings)):
    arr = d[f"ranks_d{key}"]
    ax.hist(arr, bins=bins, color=COL, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(arr.mean(), color="black", lw=1.4, ls="--")
    ax.set_title(f"{title}  平均{arr.mean():.1f}位", fontsize=12.5)
    ax.tick_params(labelsize=11)
    if i == 0:
        ax.set_ylim(0, R * 1.05)
        ax.annotate("300試行すべて1位", xy=(6, 297), xytext=(60, 265),
                    fontsize=11.5, arrowprops=dict(arrowstyle="->", color="#555"))
    else:
        ax.set_ylim(0, small_ymax)
        ax.axhline(uniform_per_bin, color="#888", lw=1.2, ls=":")
        if i == 1:
            ax.annotate(f"完全一様(くじ引き)なら\n1本あたり{uniform_per_bin:.0f}試行", xy=(160, uniform_per_bin),
                        xytext=(150, uniform_per_bin + 9), fontsize=10.5, color="#555",
                        arrowprops=dict(arrowstyle="->", color="#888"))
        ax.annotate("※縦軸は左上と異なる", xy=(0.98, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9.5, color="#888")
for i, ax in enumerate(axes):
    if i >= 2:
        ax.set_xlabel("本当の最良マッチの順位", fontsize=12)
    if i % 2 == 0:
        ax.set_ylabel(f"試行数(全{R}試行中)", fontsize=12)
fig.suptitle("雑多な質問を増やすほど、本当の最良マッチの順位が\nN=300人の中に散らばっていく(黒破線=平均順位)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUTDIR}/fig3_rank_histograms.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig3_rank_histograms.png")

print("ALL VISUALS DONE")
