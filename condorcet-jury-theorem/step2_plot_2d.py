"""2D折れ線: 固定pごとにNを増やしたときの集団正解率の推移。
p>0.5とp<0.5で運命が真逆に分かれる様子を1枚で見せる。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"

data = np.load("grid_results.npz")
N_VALUES, P_VALUES, acc_grid = data["N_VALUES"], data["P_VALUES"], data["acc_grid"]

target_ps = [0.70, 0.55, 0.50, 0.45, 0.30]
colors = {0.70: "#1f77b4", 0.55: "#2ca02c", 0.50: "#7f7f7f", 0.45: "#ff7f0e", 0.30: "#d62728"}

fig, ax = plt.subplots(figsize=(9, 5.6))
for p in target_ps:
    j = np.argmin(np.abs(P_VALUES - p))
    curve = acc_grid[:, j]
    ax.plot(N_VALUES, curve, label=f"p = {p:.2f}", color=colors[p], linewidth=2.6)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2)
ax.set_xlabel("投票者の人数 N", fontsize=13)
ax.set_ylabel("多数決の正解率(実測)", fontsize=13)
ax.set_title("p>0.5は1へ、p<0.5は0へ——多数決は個々の実力差を増幅する", fontsize=13.5)
ax.tick_params(labelsize=11)
ax.set_ylim(-0.03, 1.03)

# 収束先を直接注釈する。
# 注釈テキストは必ず「曲線と曲線の間の空き地」に置く(曲線がテキストを貫通しないように):
#   - 青(p=0.70)の3行 → 青曲線(≈1.0)と緑曲線(≦0.84)の間の帯
#   - 緑(p=0.55)の2行 → 緑曲線と0.5の点線の間の帯
#   - 灰・橙 → 0.5の点線と橙曲線の間の帯
#   - 赤の3行 → 橙曲線と赤曲線(≈0)の間の帯
ax.annotate("p=0.70: N=101で正解率1.0000\n(個人の実力70%から出発して\nほぼ完全な集団に)",
            xy=(101, acc_grid[-1, np.argmin(np.abs(P_VALUES - 0.70))]), xytext=(42, 0.86),
            fontsize=10.5, color=colors[0.70],
            arrowprops=dict(arrowstyle="->", color=colors[0.70]))
ax.annotate("p=0.55: じわじわ1に向かう\n(N=101で0.84)",
            xy=(101, acc_grid[-1, np.argmin(np.abs(P_VALUES - 0.55))]), xytext=(45, 0.56),
            fontsize=10.5, color=colors[0.55],
            arrowprops=dict(arrowstyle="->", color=colors[0.55]))
ax.annotate("p=0.50: 何人集めても0.5のまま", xy=(101, 0.50), xytext=(58, 0.40),
            fontsize=10.5, color=colors[0.50],
            arrowprops=dict(arrowstyle="->", color=colors[0.50]))
ax.annotate("p=0.45: じわじわ0に向かう\n(N=101で0.16)",
            xy=(101, acc_grid[-1, np.argmin(np.abs(P_VALUES - 0.45))]), xytext=(63, 0.26),
            fontsize=10.5, color=colors[0.45],
            arrowprops=dict(arrowstyle="->", color=colors[0.45]))
ax.annotate("p=0.30: N=101で正解率0.0000\n(個人はまだ30%当てるのに、\n集団はほぼ絶対に間違う)",
            xy=(101, 0.005), xytext=(33, 0.035),
            fontsize=10.5, color=colors[0.30],
            arrowprops=dict(arrowstyle="->", color=colors[0.30]))

ax.legend(fontsize=11, loc="center left", bbox_to_anchor=(0.02, 0.62))
fig.tight_layout()
fig.savefig("fig_2d_lines.png", dpi=150, bbox_inches="tight")
print("saved fig_2d_lines.png")

# 数値を書き出す(記事本文用)
print("\n--- N=101での正解率一覧 ---")
for p in target_ps:
    j = np.argmin(np.abs(P_VALUES - p))
    print(f"p={p:.2f}: {acc_grid[-1, j]:.4f}")
