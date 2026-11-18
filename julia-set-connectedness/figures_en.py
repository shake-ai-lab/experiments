# English-label versions of figures 1, 2, 4 (dimension) and the interactive slider.
# Computation is identical to figures.py. Needs results.json from julia.py.
# The GIF and hero image contain no text, so figures.py's versions are reused as is.

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

from core import julia_escape

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"
CMAP = LinearSegmentedColormap.from_list(
    "shake", ["#050a12", "#0d2340", "#1b4a78", "#2f81b5", "#7cc0e0", "#e3f0f8", "#ffffff"])


def escape_counts(c, n, maxiter, rad=1.6):
    x = np.linspace(-rad, rad, n)
    Z = (x[None, :] + 1j * x[:, None]).ravel()
    idx = np.arange(Z.size)
    Zc = Z.copy()
    out = np.zeros(Z.size, dtype=np.int32)
    for i in range(maxiter):
        Zc = Zc * Zc + c
        esc = Zc.real ** 2 + Zc.imag ** 2 > 4.0
        if esc.any():
            out[idx[esc]] = i + 1
            k = ~esc
            idx, Zc = idx[k], Zc[k]
            if idx.size == 0:
                break
    return out.reshape(n, n)


def colorize(e, maxiter):
    v = e.astype(float)
    v[e == 0] = np.nan
    v = np.log1p(v) / np.log1p(maxiter)
    img = CMAP(np.clip(v * 1.8, 0, 1))
    img[np.isnan(v)] = (0.02, 0.03, 0.05, 1.0)
    return img


G = R["gridA"]
nc = np.array(G["ncomp"])
md = np.array(G["mand"])
ext = [G["cx"][0], G["cx"][1], G["cy"][0], G["cy"][1]]

# ---------------------------------------------------------------- fig1 (stacked vertically)
print("fig1", flush=True)
pred = nc == 1
agree = (pred == md).mean() * 100
miss = int((~pred & md).sum())
fp = int((pred & ~md).sum())
print(f"agreement {agree:.2f}%  missed {miss}  false positives {fp}")
fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.2))
axes[0].imshow(pred, cmap="binary", origin="lower", extent=ext, interpolation="nearest")
axes[0].set_title("c whose Julia set has\nexactly 1 component", fontsize=12.5)
axes[1].imshow(md, cmap="binary", origin="lower", extent=ext, interpolation="nearest")
axes[1].set_title("Mandelbrot set,\ndrawn by its definition", fontsize=12.5)
diff = np.zeros(pred.shape + (3,))
diff[...] = 1.0
diff[pred & md] = (0.15, 0.18, 0.22)
diff[~pred & md] = (0.937, 0.278, 0.435)
diff[pred & ~md] = (0.117, 0.654, 0.490)
axes[2].imshow(diff, origin="lower", extent=ext, interpolation="nearest")
axes[2].set_title(f"Mismatch: red = {miss} missed,\ngreen = {fp} false positives", fontsize=12.5)
for a in axes:
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle(f"c on a {nc.shape[0]}x{nc.shape[0]} grid; only the number of Julia-set components "
             f"was counted (agreement {agree:.2f}%)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig1_reconstruct_en.png", dpi=125)
plt.close(fig)

# ---------------------------------------------------------------- fig2 gallery
print("fig2", flush=True)
samples = [
    (complex(-0.123, 0.745), "Douady rabbit\nc = -0.123+0.745i", "inside"),
    (complex(-1.0, 0.0), "Basilica\nc = -1", "inside"),
    (complex(0.0, 1.0), "Dendrite\nc = i", "on the boundary of"),
    (complex(-0.75, 0.11), "Tailed shape\nc = -0.75+0.11i", "inside"),
    (complex(0.3, 0.5), "Scattered dust\nc = 0.3+0.5i", "outside"),
    (complex(-0.8, 0.4), "Scattered dust\nc = -0.8+0.4i", "outside"),
]
fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.9))
for ax, (c, name, where) in zip(axes.ravel(), samples):
    e = escape_counts(c, 460, 400, rad=1.55)
    ax.imshow(colorize(e, 400), origin="lower", interpolation="nearest")
    K = julia_escape(c, -2, 2, -2, 2, 260, 300)
    _, n = ndimage.label(K, structure=np.ones((3, 3)))
    frac = K.mean() * 100
    col = ACC if where == "outside" else ("#e8a33d" if where.startswith("on") else "#7cc0e0")
    ax.set_title(f"{name}\n{where} M / {n:,} components", fontsize=11.5, color=col)
    print(f"  {name.splitlines()[1]}: {where} M, "
          f"components {n}, filled {frac:.2f}%")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("The same z -> z^2 + c, with only c changed", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=3.0)
fig.savefig("fig2_gallery_en.png", dpi=118)
plt.close(fig)

# ---------------------------------------------------------------- fig4 dimension
print("fig4", flush=True)
rows = R["dim"]["rows"]
cv = np.array([r["c"] for r in rows])
dc = np.array([r["D_corrected"] for r in rows])
ru = np.array([r["ruelle"] for r in rows])
o = np.argsort(cv)
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(cv[o], ru[o], "--", color=SUB, lw=2.0, label="Ruelle approximation  1 + |c|^2/(4 ln 2)")
ax.plot(cv[o], dc[o], "o-", color=ACC, lw=2.4, ms=8,
        label="Box-counting measurement (circle bias corrected)")
ax.axhline(1.0, ls=":", color="#c0c6cc")
ax.set_ylim(0.91, 1.22)
ax.annotate("c = 0 is the unit circle (dimension 1)", (0.0, 1.0), xytext=(-0.3, 0.925),
            fontsize=10, color="#5c6672", arrowprops=dict(arrowstyle="-", color="#9aa3ad"))
ax.annotate("c = -0.75: cusp\nof the cardioid\n(on the boundary of M)", (-0.75, dc[o][0]),
            xytext=(-0.745, 1.012), fontsize=10, color=ACC, va="bottom",
            arrowprops=dict(arrowstyle="-", color=ACC, alpha=0.5))
ax.set_xlabel("c (on the real axis)")
ax.set_ylabel("Boundary dimension of the Julia set")
ax.set_title("The farther c is from 0, the more frayed the Julia set", fontsize=13)
ax.legend(fontsize=9.5, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig4_dimension_en.png", dpi=130)
plt.close(fig)
for r in sorted(rows, key=lambda r: r["c"]):
    print(f"  c={r['c']:+.2f}: measured {r['D_corrected']:.4f}  Ruelle {r['ruelle']:.4f}")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import os  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

CS = [complex(-0.123, 0.745), complex(-0.4, 0.6), complex(-0.7269, 0.1889),
      complex(-1.0, 0.0), complex(0.285, 0.01), complex(0.0, 0.8),
      complex(-0.8, 0.156), complex(0.35, 0.35)]
NS = 150
frames = []
for i, c in enumerate(CS):
    e = escape_counts(c, NS, 200, rad=1.55)
    z = np.log1p(e.astype(float))
    z[e == 0] = 0.0
    frames.append(go.Frame(data=[go.Heatmap(z=z, showscale=False,
                                            colorscale=[[0, "#050a12"], [0.3, "#1b4a78"],
                                                        [0.65, "#2f81b5"], [1, "#e3f0f8"]])],
                           name=f"{c.real:+.3f}{c.imag:+.3f}i"))
fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(
    title=dict(text="How the Julia set changes with c<br>(move the slider)", font=dict(size=14)),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    height=580, margin=dict(l=10, r=10, t=50, b=70), paper_bgcolor="rgba(0,0,0,0)",
    sliders=[dict(active=0, currentvalue=dict(prefix="c = "),
                  steps=[dict(method="animate", label=f.name,
                              args=[[f.name], dict(mode="immediate",
                                                   frame=dict(duration=0, redraw=True))])
                         for f in frames])])
fig.write_html("julia-slider-en.html", include_plotlyjs="cdn")
print(f"  julia-slider-en.html {os.path.getsize('julia-slider-en.html')/1024:.0f}KB")
print("done")
