"""KDEのbw_method(factor)を変えて、細すぎ/適切/太すぎの3パターンを比較。
さらにscipyのデフォルト(Scott則)・Silverman則が、この標本でどこに落ちるかを検証する。"""
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
g = np.linspace(x.min() - 2.5, x.max() + 2.5, 2000)
p_true = true_pdf(g)

def valley_peak_ratio(y):
    idx = [i for i in range(1, len(y) - 1) if y[i] > y[i-1] and y[i] >= y[i+1]]
    vals = sorted(((g[i], y[i]) for i in idx), key=lambda t: -t[1])[:2]
    if len(vals) < 2:
        return None, len(idx)
    (x1, y1), (x2, y2) = sorted(vals, key=lambda t: t[0])
    i1, i2 = np.searchsorted(g, x1), np.searchsorted(g, x2)
    valley = y[i1:i2].min()
    return valley / min(y1, y2), len(idx)

# ---- fig2: 3パターン比較 ----
configs = [
    ("細すぎ: factor=0.08 → 各点がそのままノイズの山に", 0.08, "#c44e52"),
    ("適切(手動): factor=0.15 → 双峰をきれいに再現", 0.15, "#3aa655"),
    ("太すぎ: factor=0.8 → 双峰が完全に埋もれる", 0.8, "#4c72b0"),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax, (title, factor, color) in zip(axes, configs):
    kde = stats.gaussian_kde(x, bw_method=factor)
    y = kde(g)
    ratio, npeaks = valley_peak_ratio(y)
    ax.plot(g, y, color=color, lw=2.2)
    ax.fill_between(g, y, color=color, alpha=0.25)
    ax.plot(g, p_true, color="#333", lw=1.6, ls="--", alpha=0.8, label="真の密度")
    ax.plot(x, np.full_like(x, -0.005), "|", color="#333", markersize=6, alpha=0.4)
    bw = factor * x.std()
    ax.set_title(f"{title}\n(帯域幅={bw:.2f}, 山の数={npeaks})")
    ax.set_yticks([])
    ax.set_xlabel("x")
    r = f"{ratio:.2f}" if ratio is not None else "N/A(単峰)"
    print(f"{title}: factor={factor}, bw={bw:.3f}, npeaks={npeaks}, valley/peak比={r}")

axes[0].annotate("個々の点がそのまま\nノイズの山に", xy=(-3.65, 0.077), xytext=(-6.3, 0.19),
                  fontsize=10, color="#8b2f35",
                  arrowprops=dict(arrowstyle="->", color="#8b2f35", lw=1.6))
axes[1].annotate("谷が明確\n(真の密度に近い形)", xy=(0.4, 0.05), xytext=(-4.3, 0.19),
                  fontsize=10, color="#1e5c30",
                  arrowprops=dict(arrowstyle="->", color="#1e5c30", lw=1.6))
axes[2].annotate("←双峰の情報が\n完全に消える", xy=(0.5, 0.115), xytext=(2.0, 0.16),
                  fontsize=10, color="#1a3a6b",
                  arrowprops=dict(arrowstyle="->", color="#1a3a6b", lw=1.6))
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle(f"同じ{N}点のサンプルが、KDEの帯域幅を変えるだけで「ノイズの山だらけ」にも「双峰」にも「単峰」にも見える", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("fig2_kde_compare.png", dpi=110, bbox_inches="tight")
print("saved fig2_kde_compare.png")

# ---- fig3: 自動選択則(Scott/Silverman) vs 手動 ----
scott_factor = stats.gaussian_kde(x, bw_method="scott").factor
silverman_factor = stats.gaussian_kde(x, bw_method="silverman").factor
manual_factor = 0.15

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(g, p_true, color="#333", lw=2, ls="--", label="真の密度")
for factor, label, color in [
    (manual_factor, f"手動で選んだ帯域幅 (factor={manual_factor})", "#3aa655"),
    (scott_factor, f"scipyデフォルト = Scott則 (factor={scott_factor:.3f})", "#e4572e"),
    (silverman_factor, f"Silverman則 (factor={silverman_factor:.3f})", "#7b52ab"),
]:
    y = stats.gaussian_kde(x, bw_method=factor)(g)
    ratio, npeaks = valley_peak_ratio(y)
    ax.plot(g, y, color=color, lw=2.2, label=f"{label} — 谷/山比={ratio:.2f}" if ratio else label)

ax.set_yticks([])
ax.set_xlabel("x")
ax.set_ylim(-0.05, None)
ax.legend(fontsize=9.5, loc="upper right", framealpha=1.0)  # 半透明だと背後の曲線が凡例の文字に透けるため不透明に
ax.set_title("自動選択則(Scott/Silverman)は双峰を消しはしないが、谷をかなり浅くしてしまう", fontsize=12.5)
ax.annotate("手動選択(緑)は谷が深い", xy=(0.13, 0.054), xytext=(-1.9, -0.04),
            fontsize=10, color="#1e5c30", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1e5c30", lw=1.6))
fig.tight_layout()
fig.savefig("fig3_selector_compare.png", dpi=110, bbox_inches="tight")
print("saved fig3_selector_compare.png")
print(f"scott_factor={scott_factor:.4f}, silverman_factor={silverman_factor:.4f}")
