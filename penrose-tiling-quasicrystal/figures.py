import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection

from penrose_core import PHI, subdivide, initial_wheel

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
R2 = json.load(open("results2.json", encoding="utf-8"))
INK, ACC, WARN, SUB, GOOD = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae", "#06a77d"
THICK, THIN = "#f2c14e", "#2f6f9e"


def polys(tris):
    a, b = [], []
    for k, A, B, C in tris:
        p = [(A.real, A.imag), (B.real, B.imag), (C.real, C.imag)]
        (a if k == 0 else b).append(p)
    return a, b


def draw(ax, tris, lw=0.35):
    a, b = polys(tris)
    ax.add_collection(PolyCollection(a, facecolors=THICK, edgecolors="#26303c", lw=lw))
    ax.add_collection(PolyCollection(b, facecolors=THIN, edgecolors="#26303c", lw=lw))
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- fig1
print("fig1", flush=True)
tris = initial_wheel()
gens = {}
for g in range(1, 9):
    tris = subdivide(tris)
    if g in (1, 3, 5, 8):
        gens[g] = list(tris)
fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.6))
for ax, g in zip(axes.ravel(), [1, 3, 5, 8]):
    draw(ax, gens[g], lw=0.5 if g < 5 else 0.15)
    n0 = sum(1 for t in gens[g] if t[0] == 0)
    n1 = len(gens[g]) - n0
    ax.set_title(f"{g}回細分  （36°の三角 {n0:,} / 108°の三角 {n1:,}）", fontsize=12.5)
fig.suptitle("同じ2種類の三角形を、同じ規則で割り続けるだけ", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig1_tiling.png", dpi=112)
plt.close(fig)

# ---------------------------------------------------------------- fig2
print("fig2", flush=True)
rows = R["exp1"]
g = [r["gen"] for r in rows]
ratio = [1 / r["ratio"] for r in rows]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
a1.plot(g, ratio, "o-", color=ACC, lw=2.4, ms=9)
a1.axhline(PHI, ls="--", color="#3d4753", lw=1.8)
a1.text(len(g), PHI + 0.03, f"黄金比 φ = {PHI:.8f}", ha="right", fontsize=11.5,
        color="#3d4753")
a1.set_xlabel("細分の回数")
a1.set_ylabel("108°の三角形 ÷ 36°の三角形")
a1.set_title("個数の比は黄金比へ", fontsize=13)
a1.grid(alpha=0.3)
err = [abs(1 / r["ratio"] - PHI) for r in rows]
a2.semilogy(g, err, "o-", color=INK, lw=2.4, ms=9)
a2.set_xlabel("細分の回数")
a2.set_ylabel("黄金比との差（対数）")
a2.set_title(f"10回細分で差は {err[-1]:.1e}", fontsize=13)
a2.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig2_golden.png", dpi=126)
plt.close(fig)

# ---------------------------------------------------------------- fig3 回折
print("fig3", flush=True)
I = np.load("diffraction.npy")
spec = np.array(R2["spec"])
fig, (b1, b2) = plt.subplots(1, 2, figsize=(12.6, 5.6))
b1.imshow(np.log1p(I / I.max() * 400), cmap="inferno", origin="lower")
b1.set_xticks([]); b1.set_yticks([])
b1.set_title("頂点の配置から計算した回折パターン", fontsize=13)
b2.bar(np.arange(1, len(spec)), spec[1:] / spec[1:].max(),
       color=[ACC if i == 10 else INK for i in range(1, len(spec))])
b2.set_xlabel("角度方向の対称性の次数")
b2.set_ylabel("強さ（最大を1に規格化）")
b2.set_title("10次が最も強い＝10回対称", fontsize=13)
b2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig3_diffraction.png", dpi=122)
plt.close(fig)

# ---------------------------------------------------------------- fig4 周期性
print("fig4", flush=True)
fig, ax = plt.subplots(figsize=(7.6, 4.6))
vals = [R2["penrose_match"] * 100, R2["lattice_match"] * 100]
bars = ax.barh(["ペンローズ・タイル", "周期的な三角格子"], vals, color=[ACC, SUB], height=0.5)
for b, v in zip(bars, vals):
    ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center",
            fontsize=14, fontweight="bold", color="#3d4753")
ax.set_xlim(0, 118)
ax.axvline(100, ls=":", color="#3d4753")
ax.set_xlabel("平行移動でぴったり重なった頂点の割合（最も良い移動量で）")
ax.set_title("1,500通りの平行移動を試して、いちばん重なった結果", fontsize=13.5)
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig("fig4_periodicity.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
tris = initial_wheel()
seq = [list(tris)]
for g in range(8):
    tris = subdivide(tris)
    seq.append(list(tris))
fig, ax = plt.subplots(figsize=(4.9, 4.9))


def fr(i):
    ax.clear()
    draw(ax, seq[i], lw=max(0.12, 1.2 - i * 0.15))
    ax.set_title(f"{i} 回細分（{len(seq[i]):,} 個）", fontsize=13)
    return []


FuncAnimation(fig, fr, frames=len(seq), blit=False).save(
    "penrose.gif", writer=PillowWriter(fps=1.4), dpi=80)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im = Image.open("penrose.gif")
fl = [f.copy().convert("RGB").quantize(colors=16) for f in ImageSequence.Iterator(im)]
fl[0].save("penrose.gif", save_all=True, append_images=fl[1:], loop=0, duration=900,
           optimize=True)
print(f"  penrose.gif {os.path.getsize('penrose.gif')/1024:.0f}KB / {len(fl)}フレーム")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

t7 = initial_wheel()
for _ in range(6):
    t7 = subdivide(t7)
xs, ys, col = [], [], []
for k, A, B, C in t7:
    xs += [A.real, B.real, C.real, None]
    ys += [A.imag, B.imag, C.imag, None]
fig = go.Figure()
for kind, colr, nm in [(0, THICK, "36°の三角形"), (1, THIN, "108°の三角形")]:
    X, Y = [], []
    for k, A, B, C in t7:
        if k != kind:
            continue
        X += [A.real, B.real, C.real, A.real, None]
        Y += [A.imag, B.imag, C.imag, A.imag, None]
    fig.add_trace(go.Scattergl(x=X, y=Y, mode="lines", fill="toself",
                               line=dict(color="#26303c", width=0.6),
                               fillcolor=colr, name=nm, hoverinfo="skip"))
fig.update_layout(
    title=dict(text="ペンローズ・タイル（6回細分。拡大しても同じ模様は繰り返しません）",
               font=dict(size=14)),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    height=640, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)")
fig.write_html("penrose-plot.html", include_plotlyjs="cdn")
print(f"  penrose-plot.html {os.path.getsize('penrose-plot.html')/1024:.0f}KB")

# ---------------------------------------------------------------- hero
print("hero", flush=True)
th = initial_wheel()
for _ in range(7):
    th = subdivide(th)
fig = plt.figure(figsize=(15.36, 10.24), facecolor="#101820")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor("#101820")
a, b = polys(th)
ax.add_collection(PolyCollection(a, facecolors="#f2c14e", edgecolors="#101820", lw=0.25))
ax.add_collection(PolyCollection(b, facecolors="#e0573f", edgecolors="#101820", lw=0.25))
ax.set_xlim(-0.76, 0.76); ax.set_ylim(-0.5, 0.5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.savefig("hero.png", dpi=100, facecolor="#101820")
plt.close(fig)
print(f"  hero.png {os.path.getsize('hero.png')/1024:.0f}KB")
print("完了")
