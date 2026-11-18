# English-label version of figures.py. Same computation and seeds; only the text differs.
# Output goes to ./en/ (run turing.py first to create results.json).
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import ndimage

from core import simulate  # noqa: F401

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("en", exist_ok=True)
R = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB, GOOD = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae", "#06a77d"
CM = "turbo"
PAT, UNI, DEAD = "pattern", "uniform (all high)", "died out"


def classify(d):
    """Three classes by area fraction. 100% is a uniformly filled state, not a pattern."""
    if d["frac"] > 0.90:
        return UNI
    if d["frac"] < 0.02:
        return DEAD
    return PAT


def wavelength(v):
    """Peak of the radially averaged spectrum (parabolic interpolation), in grid cells."""
    N = v.shape[0]
    f = np.abs(np.fft.fftshift(np.fft.fft2(v - v.mean())))
    c = N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(yy - c, xx - c).astype(int)
    prof = np.bincount(rr.ravel(), f.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    prof[:3] = 0
    i = int(np.argmax(prof[: N // 2]))
    if 1 <= i < N // 2 - 1:
        y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
        denom = (y0 - 2 * y1 + y2)
        d = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        d = 0.0
    return N / (i + d) if (i + d) > 0 else np.nan


FS, KS = R["phase"]["F"], R["phase"]["k"]
counts = {PAT: 0, UNI: 0, DEAD: 0}
table = {}
for F in FS:
    for k in KS:
        d = R["phase"]["grid"][f"{F:.3f}_{k:.3f}"]
        c = classify(d)
        table[(F, k)] = c
        counts[c] += 1
print("classification:", counts, flush=True)

# ---------------------------------------------------------------- fig1
print("fig1", flush=True)
fig, axes = plt.subplots(len(FS), len(KS), figsize=(11.2, 12.6))
for i, F in enumerate(FS):
    for j, k in enumerate(KS):
        _, v, _ = simulate(F, k, 9000, N=110, seed=7)
        ax = axes[i][j]
        ax.imshow(v, cmap=CM, vmin=0, vmax=0.5, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        c = table[(F, k)]
        col = {PAT: ACC, UNI: SUB, DEAD: "#c0c6cc"}[c]
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(2.4)
        if i == 0:
            ax.set_title(f"k={k:.3f}", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"F={F:.3f}", fontsize=11)
fig.suptitle("Only F (feed) and k (kill) change.\n"
             f"Red frame = pattern ({counts[PAT]}), gray = uniform ({counts[UNI]}), "
             f"pale = died out ({counts[DEAD]})", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("en/fig1_phase.png", dpi=100)
plt.close(fig)

# ---------------------------------------------------------------- fig2
print("fig2", flush=True)
cases = [(0.030, 0.055), (0.022, 0.051), (0.038, 0.061), (0.046, 0.062), (0.026, 0.058),
         (0.034, 0.065)]
fig, axes = plt.subplots(2, 3, figsize=(12.6, 9.6))
for ax, (F, k) in zip(axes.ravel(), cases):
    _, v, _ = simulate(F, k, 14000, N=256, seed=3)
    wl = wavelength(v)
    m = v > 0.25
    _, n = ndimage.label(m)
    ax.imshow(v, cmap=CM, vmin=0, vmax=0.5, interpolation="bilinear")
    ax.set_title(f"F={F:.3f}  k={k:.3f}\nwavelength {wl:.1f} cells / {n:,} blobs / area {m.mean()*100:.1f}%",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"  F={F:.3f} k={k:.3f}: wavelength {wl:.2f} / blobs {n} / area {m.mean()*100:.1f}%", flush=True)
fig.suptitle("Same equations, two numbers changed (256x256 grid, 14,000 steps)", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.955]); fig.subplots_adjust(hspace=0.22)
fig.savefig("en/fig2_cases.png", dpi=112)
plt.close(fig)

# ---------------------------------------------------------------- fig3
print("fig3", flush=True)
rows3 = []
for r in [1.0, 1.1, 1.2, 1.35, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]:
    _, v, _ = simulate(0.030, 0.055, 9000, N=160, seed=11, du=0.16, dv=0.16 / r)
    m = v > 0.25
    rows3.append({"ratio": r, "frac": float(m.mean()), "wl": float(wavelength(v))})
    print(f"  Du/Dv={r:.2f}: area {m.mean()*100:5.2f}% / wavelength {wavelength(v):.1f}", flush=True)
rr = np.array([x["ratio"] for x in rows3])
ff = np.array([x["frac"] for x in rows3]) * 100
fig = plt.figure(figsize=(8.6, 9.6))
a1 = fig.add_axes([0.10, 0.50, 0.86, 0.43])
a1.plot(rr, ff, "o-", color=ACC, lw=2.4, ms=8)
a1.axvline(1.0, ls=":", color=SUB)
a1.annotate("Equal diffusion speeds:\nno pattern at all (0.00%)", (1.0, 1),
            textcoords="offset points", xytext=(20, 40), fontsize=12, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK))
a1.set_xlabel("Diffusion ratio Du/Dv (how much faster the inhibitor spreads)", fontsize=12)
a1.set_ylabel("Area covered by pattern (%)", fontsize=12)
a1.set_title("Without a speed difference, no pattern forms", fontsize=14)
a1.grid(alpha=0.3)
for i, r in enumerate([1.0, 1.5, 3.0]):
    _, v, _ = simulate(0.030, 0.055, 9000, N=160, seed=11, du=0.16, dv=0.16 / r)
    sub = fig.add_axes([0.06 + i * 0.32, 0.07, 0.28, 0.28])
    sub.imshow(v, cmap=CM, vmin=0, vmax=0.5, interpolation="bilinear")
    sub.set_xticks([]); sub.set_yticks([])
    sub.set_title(f"Du/Dv={r:g}", fontsize=12.5)
fig.text(0.5, 0.01, "Same F, k and initial state; only the diffusion ratio changes",
         ha="center", fontsize=12, color="#3d4753")
fig.savefig("en/fig3_ratio.png", dpi=118)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
FR = 44
marks = np.unique(np.round(np.geomspace(40, 16000, FR)).astype(int))
_, _, snaps = simulate(0.030, 0.055, int(marks[-1]), N=200, seed=5, snap_at=set(marks.tolist()))
keys = sorted(snaps)
fig, ax = plt.subplots(figsize=(4.8, 4.8))
ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(snaps[keys[0]], cmap=CM, vmin=0, vmax=0.5, interpolation="bilinear")
txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=12,
              fontweight="bold", color="white")


def fr(i):
    im.set_data(snaps[keys[i]])
    txt.set_text(f"step {keys[i]:,}")
    return im, txt


FuncAnimation(fig, fr, frames=len(keys), blit=False).save(
    "en/turing.gif", writer=PillowWriter(fps=6), dpi=78)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402

im2 = Image.open("en/turing.gif")
fl = [f.copy().convert("RGB").quantize(colors=64) for f in ImageSequence.Iterator(im2)]
fl[0].save("en/turing.gif", save_all=True, append_images=fl[1:], loop=0, duration=180, optimize=True)
print(f"  turing.gif {os.path.getsize('en/turing.gif')/1024:.0f}KB / {len(fl)} frames")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

sel = [(0.022, 0.051), (0.026, 0.055), (0.030, 0.055), (0.034, 0.060),
       (0.038, 0.061), (0.046, 0.062)]
frames = []
for F, k in sel:
    _, v, _ = simulate(F, k, 12000, N=128, seed=3)
    z = np.rint(np.clip(v, 0, 0.5) * 400).astype(int)
    frames.append(go.Frame(data=[go.Heatmap(z=z.tolist(), zmin=0, zmax=200,
                                            colorscale="Turbo", showscale=False)],
                           name=f"F={F:.3f}, k={k:.3f}"))
fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(
    title=dict(text="Pattern for each feed F and kill k<br>(use the slider below)",
               font=dict(size=14)),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    height=560, margin=dict(l=10, r=10, t=48, b=70), paper_bgcolor="rgba(0,0,0,0)",
    sliders=[dict(active=0, currentvalue=dict(prefix=""),
                  steps=[dict(method="animate", label=f.name,
                              args=[[f.name], dict(mode="immediate",
                                                   frame=dict(duration=0, redraw=True))])
                         for f in frames])])
fig.write_html("en/turing-slider-en.html", include_plotlyjs="cdn")
print("done")
