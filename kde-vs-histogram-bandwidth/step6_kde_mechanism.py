"""KDEの仕組みを図解する: 各データ点の上にガウス核を置き、全部足すと曲線になる。
220点全部だと核が潰れて見えないので、同じサンプルの最初の12点だけを取り出し、
細い核(帯域幅0.36)と太い核(帯域幅1.93)で「核→合計」の関係を上下2段で見せる。
ブログの720px幅カラムでも読めるよう、横並びではなく縦積み+大きめフォント。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.family"] = "Meiryo"

MU1, SD1, MU2, SD2, W = -2.0, 1.0, 2.2, 1.0, 0.5
N, SEED = 220, 42
rng = np.random.default_rng(SEED)
is_comp1 = rng.random(N) < W
x_full = np.where(is_comp1, rng.normal(MU1, SD1, N), rng.normal(MU2, SD2, N))

M = 12
x = x_full[:M]  # 実サンプルの最初の12点(両方の山にまたがっている)
g = np.linspace(-6.5, 6.5, 1000)

configs = [
    (0.36, "#3aa655", "#1e5c30",
     "細めの核 (帯域幅=0.36): 1点ごとの山が残りつつ、合計には谷ができる"),
    (1.93, "#4c72b0", "#1a3a6b",
     "太い核 (帯域幅=1.93): 山どうしがべったり重なり、合計は1つの山にならされる"),
]

fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.6))
for ax, (bw, color, dark, title) in zip(axes, configs):
    total = np.zeros_like(g)
    for xi in x:
        k = stats.norm.pdf(g, xi, bw) / M
        total += k
        ax.plot(g, k, color=color, lw=1.3, alpha=0.55)
    ax.plot(g, total, color=dark, lw=3, label=f"{M}個の核の合計 (=KDEの曲線)")
    ax.plot(x, np.full_like(x, -0.008 * total.max() / 0.28), "|",
            color="#333", markersize=12, mew=1.8)
    ax.set_title(title, fontsize=14)
    ax.set_yticks([])
    ax.set_xlim(-6.5, 6.5)
    ax.legend(fontsize=12, loc="upper right", framealpha=1.0)

axes[0].annotate("核1個 = データ1点の上に\n置いた小さなガウスの山",
                 xy=(x[0], stats.norm.pdf(0, 0, 0.36) / M * 0.55),
                 xytext=(-6.2, 0.185), fontsize=12, color="#555",
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
axes[1].set_xlabel("x", fontsize=12)
fig.suptitle(f"KDEの正体: 各データ点に小さなガウスの山(核)を置いて、全部足すだけ\n"
             f"(仕組みが見えるよう、{N}点のうち最初の{M}点だけで図解)", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("fig4_kde_mechanism.png", dpi=110, bbox_inches="tight")
print("saved fig4_kde_mechanism.png")
print(f"12点: {np.round(np.sort(x), 2).tolist()}")
