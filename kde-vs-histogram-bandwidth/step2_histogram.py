"""同じサンプルに対して、ヒストグラムのビン幅だけを変える。
粗すぎ(幅2.4)/適切(幅0.8)/細かすぎ(幅0.15)の3パターン。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 11.5, "lines.linewidth": 2})

MU1, SD1, MU2, SD2, W = -2.0, 1.0, 2.2, 1.0, 0.5
def true_pdf(g):
    return W * stats.norm.pdf(g, MU1, SD1) + (1 - W) * stats.norm.pdf(g, MU2, SD2)

x = np.load("x_sample.npy")
N = len(x)
lo, hi = x.min() - 0.3, x.max() + 0.3
g = np.linspace(lo, hi, 800)
p_true = true_pdf(g)

configs = [
    ("粗すぎ: 幅2.4 → 双峰が消えて単峰に見える", 2.4, "#4c72b0"),
    ("適切: 幅0.8 → 双峰がはっきり見える", 0.8, "#3aa655"),
    ("細かすぎ: 幅0.15 → ノイズで山だらけ", 0.15, "#c44e52"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax, (title, bw, color) in zip(axes, configs):
    bins = np.arange(lo, hi + bw, bw)
    counts, edges = np.histogram(x, bins=bins)
    ax.hist(x, bins=bins, density=True, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.plot(g, p_true, color="#333", lw=1.6, ls="--", alpha=0.8, label="真の密度")
    ax.set_title(f"{title}\n(bins={len(bins)-1}本)")
    ax.set_yticks([])
    ax.set_xlabel("x")
    print(f"{title}: bins={len(bins)-1}, counts={counts.tolist()}")

axes[0].annotate("2つの山が\n1つに埋没", xy=(-0.5, 0.09), xytext=(-4.3, 0.16),
                  fontsize=10.5, color="#1a3a6b",
                  arrowprops=dict(arrowstyle="->", color="#1a3a6b", lw=1.6))
axes[1].annotate("谷がくっきり\n(2つの山)", xy=(0.4, 0.05), xytext=(-4.0, 0.19),
                  fontsize=10.5, color="#1e5c30",
                  arrowprops=dict(arrowstyle="->", color="#1e5c30", lw=1.6))
axes[2].annotate("サンプリング\nノイズの偽の山", xy=(-2.135, 0.362), xytext=(0.7, 0.30),
                  fontsize=10.5, color="#8b2f35", ha="left",
                  arrowprops=dict(arrowstyle="->", color="#8b2f35", lw=1.6))
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle(f"同じ{N}点のサンプルが、ビン幅を変えるだけで「山1つ」にも「山2つ」にも「山だらけ」にも見える", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("fig1_hist_compare.png", dpi=110, bbox_inches="tight")
print("saved fig1_hist_compare.png")
