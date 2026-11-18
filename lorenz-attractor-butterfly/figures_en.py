"""English-labelled figures for the Lorenz article.

Same computation and seeds as figures.py; only the labels differ.
Needs results.json (lorenz.py) and lyap.json (lyap.py).
Writes en_1.png, en_2.png, en_3.png, en_lorenz.gif and en_lorenz-3d.html.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
L = json.load(open("lyap.json", encoding="utf-8"))
INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def deriv(s):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z], axis=-1)


def integrate(s0, dt, n):
    s = np.array(s0, float)
    out = np.empty((n + 1,) + s.shape)
    out[0] = s
    for i in range(n):
        k1 = deriv(s)
        k2 = deriv(s + dt / 2 * k1)
        k3 = deriv(s + dt / 2 * k2)
        k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = s
    return out


DT = 0.002
base = integrate([1.0, 1.0, 20.0], DT, 20000)[-1]

# ---------------------------------------------------------------- fig1 (stacked)
print("fig1", flush=True)
t = np.array(R["exp1"]["t"])
d = np.array(R["exp1"]["d"])
pair = integrate(np.stack([base, base + np.array([1e-9, 0, 0])]), DT, 14000)
fig = plt.figure(figsize=(8.6, 10.6))
a1 = fig.add_subplot(2, 1, 1)
a1.semilogy(t, d, color=ACC, lw=2.4)
a1.axhline(1.0, ls="--", color=SUB, lw=1.5)
a1.text(0.8, 1.8, "distance 1.0 (a visible difference)", fontsize=12, color="#5c6672")
a1.axvline(R["exp1"]["t_visible"], ls=":", color=INK, lw=1.6)
a1.annotate(f"t = {R['exp1']['t_visible']:.1f}", (R["exp1"]["t_visible"], 1e-7),
            textcoords="offset points", xytext=(8, 0), color=INK, fontsize=13,
            fontweight="bold")
a1.set_xlabel("model time", fontsize=12)
a1.set_ylabel("distance between the two orbits", fontsize=12)
a1.set_title("The initial difference was 0.000000001", fontsize=14)
a1.grid(alpha=0.3, which="both")

a2 = fig.add_subplot(2, 1, 2, projection="3d")
a2.plot(pair[:, 0, 0], pair[:, 0, 1], pair[:, 0, 2], lw=0.6, color=INK, alpha=0.9, label="orbit 1")
a2.plot(pair[:, 1, 0], pair[:, 1, 1], pair[:, 1, 2], lw=0.6, color=ACC, alpha=0.75, label="orbit 2")
a2.set_title("Two orbits running separately on the same shape", fontsize=14)
a2.set_xticklabels([]); a2.set_yticklabels([]); a2.set_zticklabels([])
from matplotlib.lines import Line2D  # noqa: E402
a2.legend(handles=[Line2D([0], [0], color=INK, lw=2.5, label="orbit 1"),
                   Line2D([0], [0], color=ACC, lw=2.5, label="orbit 2 (offset by 1e-9)")],
          loc="upper left", fontsize=11, bbox_to_anchor=(0.0, 0.95))
fig.tight_layout()
fig.savefig("en_1.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig2
print("fig2", flush=True)
rows = R["exp2"]["rows"]
ee = np.array([r["eps"] for r in rows])
hz = np.array([r["horizon"] for r in rows])
fig, ax = plt.subplots(figsize=(8.4, 5.4))
ax.plot(ee, hz, "o-", color=ACC, lw=2.4, ms=9)
ax.set_xscale("log")
ax.invert_xaxis()
for e, h in zip(ee, hz):
    ax.annotate(f"{h:.1f}", (e, h), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=11, color="#3d4753")
ax.annotate(f"Each 10x gain in precision buys\nonly {R['exp2']['slope']:.1f} more model time",
            (1e-9, 12), fontsize=12.5, color=INK, fontweight="bold", ha="center")
ax.set_xlabel("initial offset (more precise to the right)", fontsize=12)
ax.set_ylabel("time until the distance reaches 1.0", fontsize=12)
ax.set_title("Even from the double-precision limit (1e-15),\nthe orbits part by t = 34", fontsize=13.5)
ax.set_ylim(0, 41)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("en_2.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig3 (stacked)
print("fig3", flush=True)
fig, (b1, b2) = plt.subplots(2, 1, figsize=(8.4, 10.4))
r = np.array(R["exp3"]["r"])
C = np.array(R["exp3"]["C"])
ok = (C > 1e-4) & (C < 0.1)
b1.loglog(r, C, "o-", color=INK, lw=2.0, ms=6)
b1.loglog(r[ok], C[ok], "o", color=ACC, ms=9, label=f"fitted range (slope {R['exp3']['dim']:.3f})")
b1.set_xlabel("radius r", fontsize=12)
b1.set_ylabel("fraction of point pairs closer than r, C(r)", fontsize=12)
b1.set_title(f"Measured correlation dimension {R['exp3']['dim']:.3f} (literature: about 2.05)", fontsize=13)
b1.legend(fontsize=11, loc="lower right")
b1.grid(alpha=0.3, which="both")

run = np.array(L["running"])
x = (np.arange(len(run)) + 1) * 20 * 0.5
b2.plot(x, run, color=WARN, lw=2.2)
b2.axhline(0.906, ls="--", color=SUB, lw=1.6)
b2.text(x[-1], 0.918, "literature value 0.906", ha="right", fontsize=11.5, color="#5c6672")
b2.axhline(0.7604, ls=":", color=ACC, lw=1.8)
b2.text(x[len(x) // 4], 0.745, "0.760 from a single run", fontsize=11.5, color=ACC)
b2.set_ylim(0.7, 1.05)
b2.set_xlabel("time used for the average", fontsize=12)
b2.set_ylabel("estimate of the largest Lyapunov exponent", fontsize=12)
b2.set_title(f"Re-measuring and averaging settles at {L['lambda_benettin']:.4f}", fontsize=13)
b2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("en_3.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
FR = 60
seg = pair[::28]          # whole run, t = 0 .. 28 (the orbits separate near t = 24)
fig = plt.figure(figsize=(4.9, 4.9))
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()


def frame(k):
    ax.clear()
    ax.set_axis_off()
    n = int(len(seg) * (k + 1) / FR)
    ax.plot(seg[:n, 0, 0], seg[:n, 0, 1], seg[:n, 0, 2], lw=1.0, color=INK)
    ax.plot(seg[:n, 1, 0], seg[:n, 1, 1], seg[:n, 1, 2], lw=1.0, color=ACC)
    ax.scatter(*seg[n - 1, 0], s=26, color=INK)
    ax.scatter(*seg[n - 1, 1], s=26, color=ACC)
    dd = np.linalg.norm(seg[n - 1, 0] - seg[n - 1, 1])
    ax.set_xlim(-22, 22); ax.set_ylim(-28, 28); ax.set_zlim(2, 50)
    ax.set_title(f"t = {n * 28 * DT:5.1f}   distance {dd:.2e}", fontsize=12)
    ax.view_init(elev=20, azim=-60 + k * 1.2)


FuncAnimation(fig, frame, frames=FR, blit=False).save(
    "en_lorenz.gif", writer=PillowWriter(fps=6), dpi=76)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402

im = Image.open("en_lorenz.gif")
fr = [f.copy().convert("RGB").quantize(colors=48) for f in ImageSequence.Iterator(im)]
fr[0].save("en_lorenz.gif", save_all=True, append_images=fr[1:], loop=0, duration=170, optimize=True)
print(f"  en_lorenz.gif {os.path.getsize('en_lorenz.gif')/1024:.0f}KB / {len(fr)} frames")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

tr = integrate(base, DT, 120000)[::12]
fig = go.Figure(go.Scatter3d(
    x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode="lines",
    line=dict(width=2, color=np.arange(len(tr)),
              colorscale=[[0, "#ef476f"], [0.4, "#e8a33d"], [0.75, "#2f6f9e"], [1, "#0b1622"]]),
    hoverinfo="skip"))
fig.update_layout(
    title=dict(text="Lorenz attractor<br>(color shows elapsed time; drag to rotate)", font=dict(size=14)),
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
               camera=dict(eye=dict(x=1.4, y=-1.5, z=0.6))),
    height=600, margin=dict(l=0, r=0, t=45, b=0), paper_bgcolor="rgba(0,0,0,0)")
fig.write_html("en_lorenz-3d.html", include_plotlyjs="cdn")
print("done")
