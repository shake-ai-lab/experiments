import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"

OUTDIR = "C:/tmp/ensemble-diversity-experiment"

agg = np.load(f"{OUTDIR}/results_aggregate.npz")
R, M, H_SMALL = int(agg["R"]), int(agg["M"]), int(agg["H_SMALL"])
H_LIST_B = agg["H_LIST_B"]

COL_A = "#2e6ee4"
COL_B = "#e4572e"

fig, ax = plt.subplots(figsize=(8, 9))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.5, 0.985, "総パラメータ数はどちらも330 (=4H+1の合計)\n「同じ予算」でのA/B対決",
        ha="center", va="top", fontsize=13)

n_a = M  # one box per member: A and B both actually have M=10 members
box_w, gap = 0.06, 0.075
start_x = 0.5 - (n_a - 1) * gap / 2 - box_w / 2
ax.text(0.5, 0.90, "A: shallow diversity (seedのみ)", ha="center", fontsize=15, fontweight="bold", color=COL_A)
for i in range(n_a):
    cx = start_x + i * gap
    ax.add_patch(plt.Rectangle((cx, 0.76), box_w, 0.10, fc="#dfe8f7", ec=COL_A, lw=1.3))
ax.text(0.5, 0.685, f"全メンバー 隠れ{H_SMALL}ユニット固定\n同じ訓練データ700点をそのまま使用\n乱数初期化seedだけを変える (M={M})",
        ha="center", va="top", fontsize=12.5, linespacing=1.6)
ax.annotate("", xy=(0.5, 0.545), xytext=(0.5, 0.595), arrowprops=dict(arrowstyle="->", lw=1.8, color=COL_A))
ax.text(0.5, 0.515, "予測を平均", ha="center", fontsize=12.5, color=COL_A)

ax.plot([0.03, 0.97], [0.485, 0.485], color="#999", lw=1, ls="--")

widths = H_LIST_B
n_b = len(widths)
gap_b = 0.075
start_x_b = 0.5 - (n_b - 1) * gap_b / 2 - box_w / 2
base_y = 0.276  # bar baseline
h_min, h_max = float(min(widths)), float(max(widths))
ax.text(0.5, 0.406, "B: real diversity (bagging+幅)", ha="center", fontsize=15, fontweight="bold", color=COL_B)
for i, h in enumerate(widths):
    cx = start_x_b + i * gap_b
    hgt = 0.02 + (h - h_min) / (h_max - h_min + 1e-9) * (0.093 - 0.02)
    ax.add_patch(plt.Rectangle((cx, base_y), box_w, hgt, fc="#fbe3dc", ec=COL_B, lw=1.3))
ax.text(0.5, 0.206, f"メンバーごとに隠れユニット数を変える ({int(h_min)}〜{int(h_max)}, 合計80=Aと同じ)\n"
                    "各メンバーは訓練データのブートストラップ再抽出(700点、重複あり)で学習",
        ha="center", va="top", fontsize=12.5, linespacing=1.6)
ax.annotate("", xy=(0.5, 0.076), xytext=(0.5, 0.122), arrowprops=dict(arrowstyle="->", lw=1.8, color=COL_B))
ax.text(0.5, 0.048, "予測を平均", ha="center", fontsize=12.5, color=COL_B)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig0_setup.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("saved", f"{OUTDIR}/fig0_setup.png")
