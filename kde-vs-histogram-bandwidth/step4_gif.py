"""ヒストグラムのビン幅とKDEの帯域幅を、粗い(左)→適切→細かい(右)へ
同時にスイープするGIF。同じサンプルの「見え方」が滑らかに変わっていく様子を見せる。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.family"] = "Meiryo"

MU1, SD1, MU2, SD2, W = -2.0, 1.0, 2.2, 1.0, 0.5
def true_pdf(g):
    return W * stats.norm.pdf(g, MU1, SD1) + (1 - W) * stats.norm.pdf(g, MU2, SD2)

x = np.load("x_sample.npy")
N = len(x)
lo, hi = x.min() - 0.3, x.max() + 0.3
g = np.linspace(lo, hi, 500)
p_true = true_pdf(g)

# 「粗い」→「細かい」に対応するビン幅・KDE factorのペアを対数間隔で用意
STEPS = 11
bws = np.geomspace(2.8, 0.14, STEPS)
factors = np.geomspace(0.85, 0.07, STEPS)
GOOD_STEP = 5  # このあたりが「適切」ゾーン
print("bin widths:", np.round(bws, 2))
print("kde factors:", np.round(factors, 2))

# 最初と「適切」ゾーンで少し長く止める(開始状態が見えるように)
order = [0, 0] + list(range(STEPS)) + [GOOD_STEP] * 2 + list(range(GOOD_STEP, STEPS)) + [STEPS - 1] * 2

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=80)

def draw(i):
    step = order[i]
    bw, factor = bws[step], factors[step]
    for ax in axes:
        ax.clear()
    bins = np.arange(lo, hi + bw, bw)
    axes[0].hist(x, bins=bins, density=True, color="#4c72b0", edgecolor="white", linewidth=0.4, alpha=0.85)
    axes[0].plot(g, p_true, color="#e4572e", lw=1.8, ls="--")
    axes[0].set_title(f"ヒストグラム: ビン幅={bw:.2f}", fontsize=11)

    y = stats.gaussian_kde(x, bw_method=factor)(g)
    axes[1].plot(g, y, color="#3aa655", lw=2.2)
    axes[1].fill_between(g, y, color="#3aa655", alpha=0.25)
    axes[1].plot(g, p_true, color="#e4572e", lw=1.8, ls="--")
    axes[1].set_title(f"KDE: 帯域幅={factor*x.std():.2f}", fontsize=11)

    for ax in axes:
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 0.42)
        ax.set_xlabel("x")
    tag = "適切ゾーン" if step == GOOD_STEP else ("粗い(見えない)" if step < GOOD_STEP - 2 else ("細かい(ノイズ)" if step > GOOD_STEP + 2 else ""))
    fig.suptitle(f"同じ{N}点のサンプル。粒度を変えるだけで物語が変わる  [{tag}]", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return []

ani = animation.FuncAnimation(fig, draw, frames=len(order), interval=220)
ani.save("sweep.gif", writer="pillow", fps=4, dpi=72)
print("saved sweep.gif")
