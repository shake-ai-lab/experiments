"""English-labelled versions of every figure, the sweep GIF and the slider.
Same sample (x_sample.npy from step1_data.py, seed 42), same computations as
step1-step6; only the labels differ. Outputs go to ./en/.
Run step1_data.py first if x_sample.npy is missing."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams.update({"axes.titlesize": 11, "lines.linewidth": 2})
os.makedirs("en", exist_ok=True)

MU1, SD1, MU2, SD2, W = -2.0, 1.0, 2.2, 1.0, 0.5
N, SEED = 220, 42


def true_pdf(g):
    return W * stats.norm.pdf(g, MU1, SD1) + (1 - W) * stats.norm.pdf(g, MU2, SD2)


x = np.load("x_sample.npy")
# check the saved sample is the seed-42 draw
rng = np.random.default_rng(SEED)
is1 = rng.random(N) < W
x_chk = np.where(is1, rng.normal(MU1, SD1, N), rng.normal(MU2, SD2, N))
assert np.allclose(x, x_chk)

# ---------------- fig0: setup ----------------
g = np.linspace(-6.5, 6.5, 2000)
p = true_pdf(g)
valley = p[(g > MU1) & (g < MU2)].min()
print(f"true valley/peak = {valley / min(p[g < 0].max(), p[g > 0].max()):.3f}")
print(f"sample N={N} mean={x.mean():.2f} std={x.std():.3f} (ddof=0) / {x.std(ddof=1):.3f} (ddof=1)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(g, p, color="#e4572e", lw=2.2, label="True density (mixture of two normals)")
ax.plot(x, np.full_like(x, -0.01), "|", color="#333", markersize=10, alpha=0.6,
        label=f"The {N} points actually drawn")
ax.set_yticks([])
ax.set_xlabel("x")
ax.set_ylim(-0.03, 0.26)
ax.legend(fontsize=10, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0))
ax.set_title(f"Two clear peaks in truth. But given only {N} points,\ncould you say for sure it is bimodal?", fontsize=12.5)
fig.tight_layout()
fig.savefig("en/1.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---------------- fig1: histogram, three bin widths ----------------
lo, hi = x.min() - 0.3, x.max() + 0.3
gg = np.linspace(lo, hi, 800)
p_true = true_pdf(gg)
configs = [
    ("Too coarse: width 2.4\nthe two peaks merge into one", 2.4, "#4c72b0"),
    ("Just right: width 0.8\ntwo peaks clearly visible", 0.8, "#3aa655"),
    ("Too fine: width 0.15\nnoise makes a forest of peaks", 0.15, "#c44e52"),
]
fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
for ax, (title, bw, color) in zip(axes, configs):
    bins = np.arange(lo, hi + bw, bw)
    counts, _ = np.histogram(x, bins=bins)
    ax.hist(x, bins=bins, density=True, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.plot(gg, p_true, color="#333", lw=1.6, ls="--", alpha=0.8, label="True density")
    ax.set_title(f"{title}  ({len(bins) - 1} bins)", fontsize=12.5)
    ax.set_yticks([])
    ax.set_xlabel("x")
    print(f"hist width {bw}: bins={len(bins) - 1}, mean count/bin={N / (len(bins) - 1):.1f}, max={counts.max()}, counts={counts.tolist()}")
axes[0].legend(fontsize=10, loc="upper right")
fig.suptitle(f"The same {N} points look like one peak, two peaks, or a forest of peaks,\ndepending only on the bin width",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("en/2.png", dpi=100, bbox_inches="tight")
plt.close(fig)


def count_peaks(y):
    return sum(1 for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1])


gk = np.linspace(x.min() - 2.5, x.max() + 2.5, 2000)
pk = true_pdf(gk)


def valley_peak_ratio(y, grid):
    idx = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1]]
    vals = sorted(((grid[i], y[i]) for i in idx), key=lambda t: -t[1])[:2]
    if len(vals) < 2:
        return None, len(idx)
    (x1, y1), (x2, y2) = sorted(vals, key=lambda t: t[0])
    i1, i2 = np.searchsorted(grid, x1), np.searchsorted(grid, x2)
    return y[i1:i2].min() / min(y1, y2), len(idx)


# ---------------- fig2: KDE, three factors ----------------
kconf = [
    ("Too narrow: factor 0.08\nevery point becomes its own bump", 0.08, "#c44e52"),
    ("Just right: factor 0.15\ntwo peaks and a clear valley", 0.15, "#3aa655"),
    ("Too wide: factor 0.8\nthe two peaks are completely gone", 0.8, "#4c72b0"),
]
fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
for ax, (title, factor, color) in zip(axes, kconf):
    y = stats.gaussian_kde(x, bw_method=factor)(gk)
    ratio, npk = valley_peak_ratio(y, gk)
    ax.plot(gk, y, color=color, lw=2.2)
    ax.fill_between(gk, y, color=color, alpha=0.25)
    ax.plot(gk, pk, color="#333", lw=1.6, ls="--", alpha=0.8, label="True density")
    ax.plot(x, np.full_like(x, -0.005), "|", color="#333", markersize=6, alpha=0.4)
    bw = factor * x.std(ddof=1)
    ax.set_title(f"{title}\n(bandwidth {bw:.2f}, {npk} peak{'s' if npk != 1 else ''})", fontsize=12.5)
    ax.set_yticks([])
    ax.set_xlabel("x")
    print(f"kde factor {factor}: bw={bw:.3f} (ddof=1 std) / {factor * x.std():.3f} (ddof=0), peaks={npk}, valley/peak={ratio}")
axes[0].legend(fontsize=10, loc="upper right")
fig.suptitle(f"The same {N} points look noisy, bimodal, or unimodal,\ndepending only on the KDE bandwidth", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("en/3.png", dpi=100, bbox_inches="tight")
plt.close(fig)

# ---------------- fig3: automatic selectors ----------------
scott = stats.gaussian_kde(x, bw_method="scott").factor
silv = stats.gaussian_kde(x, bw_method="silverman").factor
fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.plot(gk, pk, color="#333", lw=2, ls="--", label="True density")
for factor, label, color in [
    (0.15, "Hand-picked (factor=0.15)", "#3aa655"),
    (scott, f"scipy default = Scott (factor={scott:.3f})", "#e4572e"),
    (silv, f"Silverman (factor={silv:.3f})", "#7b52ab"),
]:
    y = stats.gaussian_kde(x, bw_method=factor)(gk)
    ratio, _ = valley_peak_ratio(y, gk)
    ax.plot(gk, y, color=color, lw=2.2, label=f"{label}: valley/peak = {ratio:.2f}")
    print(f"selector factor={factor:.4f} valley/peak={ratio:.3f}")
ax.set_yticks([])
ax.set_xlabel("x")
ax.set_ylim(-0.05, 0.36)
ax.legend(fontsize=9.5, loc="upper right", framealpha=1.0)
ax.set_title("Scott and Silverman keep both peaks but make the valley much shallower", fontsize=12)
ax.annotate("hand-picked (green)\nhas a deep valley", xy=(0.13, 0.054), xytext=(-1.9, -0.04),
            fontsize=10, color="#1e5c30", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1e5c30", lw=1.6))
fig.tight_layout()
fig.savefig("en/4.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"scott={scott:.4f} silverman={silv:.4f}")

# Freedman-Diaconis / Scott bin widths and width-1.5 counts
iqr = np.subtract(*np.percentile(x, [75, 25]))
fd = 2 * iqr / N ** (1 / 3)
sc = 3.49 * x.std(ddof=1) / N ** (1 / 3)
print(f"FD width={fd:.2f}, Scott width={sc:.2f}")
b15 = np.arange(lo, hi + 1.5, 1.5)
print("width 1.5 counts:", np.histogram(x, bins=b15)[0].tolist())

# ---------------- fig4: KDE mechanism, first 12 points ----------------
M = 12
xm = x[:M]
g2 = np.linspace(-6.5, 6.5, 1000)
mconf = [
    (0.36, "#3aa655", "#1e5c30", "Narrow kernels (bandwidth 0.36): bumps stay distinct, a valley survives"),
    (1.93, "#4c72b0", "#1a3a6b", "Wide kernels (bandwidth 1.93): bumps smear into one hill"),
]
fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.6))
for ax, (bw, color, dark, title) in zip(axes, mconf):
    total = np.zeros_like(g2)
    for xi in xm:
        k = stats.norm.pdf(g2, xi, bw) / M
        total += k
        ax.plot(g2, k, color=color, lw=1.3, alpha=0.55)
    ax.plot(g2, total, color=dark, lw=3, label=f"Sum of {M} kernels (= the KDE curve)")
    ax.plot(xm, np.full_like(xm, -0.008 * total.max() / 0.28), "|", color="#333", markersize=12, mew=1.8)
    ax.set_title(title, fontsize=12.5)
    ax.set_yticks([])
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-0.01, total.max() * 1.45)
    ax.legend(fontsize=11, loc="upper left", framealpha=1.0)
axes[0].annotate("one kernel = a small Gaussian\nplaced on one data point",
                 xy=(xm[0], stats.norm.pdf(0, 0, 0.36) / M * 0.55),
                 xytext=(3.7, 0.10), fontsize=11, color="#555",
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
axes[1].set_xlabel("x", fontsize=12)
fig.suptitle(f"What a KDE is: put a small Gaussian on every data point, then add them up\n"
             f"(drawn with only the first {M} of the {N} points so the kernels stay visible)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("en/5.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"12 points: {np.round(np.sort(xm), 2).tolist()}")

# ---------------- sweep GIF + slider ----------------
STEPS = 11
bws = np.geomspace(2.8, 0.14, STEPS)
factors = np.geomspace(0.85, 0.07, STEPS)
GOOD = 5
gs = np.linspace(lo, hi, 500)
ps = true_pdf(gs)
order = [0, 0] + list(range(STEPS)) + [GOOD] * 2 + list(range(GOOD, STEPS)) + [STEPS - 1] * 2
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=80)


def draw(i):
    step = order[i]
    bw, factor = bws[step], factors[step]
    for ax in axes:
        ax.clear()
    bins = np.arange(lo, hi + bw, bw)
    axes[0].hist(x, bins=bins, density=True, color="#4c72b0", edgecolor="white", linewidth=0.4, alpha=0.85)
    axes[0].plot(gs, ps, color="#e4572e", lw=1.8, ls="--")
    axes[0].set_title(f"Histogram: bin width = {bw:.2f}", fontsize=11)
    y = stats.gaussian_kde(x, bw_method=factor)(gs)
    axes[1].plot(gs, y, color="#3aa655", lw=2.2)
    axes[1].fill_between(gs, y, color="#3aa655", alpha=0.25)
    axes[1].plot(gs, ps, color="#e4572e", lw=1.8, ls="--")
    axes[1].set_title(f"KDE: bandwidth = {factor * x.std(ddof=1):.2f}", fontsize=11)
    for ax in axes:
        ax.set_yticks([])
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 0.42)
        ax.set_xlabel("x")
    tag = "just right" if step == GOOD else ("too coarse" if step < GOOD - 2 else ("too fine (noise)" if step > GOOD + 2 else ""))
    fig.suptitle(f"The same {N} points. Only the granularity changes  [{tag}]", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return []


ani = animation.FuncAnimation(fig, draw, frames=len(order), interval=220)
ani.save("en/sweep.gif", writer="pillow", fps=4, dpi=72)
plt.close(fig)
print("bin widths:", np.round(bws, 2).tolist())
print("kde bandwidths:", np.round(factors * x.std(ddof=1), 2).tolist())

sf = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "KDE"))
gp = np.linspace(lo, hi, 400)
pp = true_pdf(gp)
for i, (bw, factor) in enumerate(zip(bws, factors)):
    bins = np.arange(lo, hi + bw, bw)
    cnt, edges = np.histogram(x, bins=bins, density=True)
    cen = (edges[:-1] + edges[1:]) / 2
    vis = i == 0
    sf.add_trace(go.Bar(x=cen, y=cnt, width=bw * 0.95, marker_color="#4c72b0", name="Histogram",
                        visible=vis, showlegend=False), row=1, col=1)
    sf.add_trace(go.Scatter(x=gp, y=pp, mode="lines", line=dict(color="#e4572e", dash="dash", width=1.6),
                            name="True density", visible=vis, showlegend=(i == 0)), row=1, col=1)
    y = stats.gaussian_kde(x, bw_method=factor)(gp)
    sf.add_trace(go.Scatter(x=gp, y=y, mode="lines", line=dict(color="#3aa655", width=2.2), fill="tozeroy",
                            fillcolor="rgba(58,166,85,0.25)", name="KDE", visible=vis, showlegend=False), row=1, col=2)
    sf.add_trace(go.Scatter(x=gp, y=pp, mode="lines", line=dict(color="#e4572e", dash="dash", width=1.6),
                            name="True density", visible=vis, showlegend=False), row=1, col=2)
per = 4
total = STEPS * per
steps = []
for i, (bw, factor) in enumerate(zip(bws, factors)):
    steps.append(dict(method="update", label=f"{i}",
                      args=[{"visible": [j // per == i for j in range(total)]}]))
sf.update_layout(
    sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="Granularity step: "), pad=dict(t=40))],
    title=dict(text="Drag the slider: coarse, just right, fine<br>The same 220 points look this different", font=dict(size=13)),
    height=560, margin=dict(l=30, r=30, t=90, b=60), legend=dict(orientation="h", x=1, xanchor="right", y=1, yanchor="top", bgcolor="rgba(255,255,255,0.8)"),
)
for c in [1, 2]:
    sf.update_yaxes(showticklabels=False, range=[0, 0.42], row=1, col=c)
    sf.update_xaxes(range=[lo, hi], row=1, col=c)
sf.write_html("en/kde-hist-slider-en.html", include_plotlyjs="cdn", full_html=True)
print("done")
