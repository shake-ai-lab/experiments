import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
BL = json.load(open("bylength.json", encoding="utf-8"))
G = np.load("grid.npy")
P = np.load("prime.npy")
N = P.shape[0]
base = P.mean()
INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"

# ---------------------------------------------------------------- fig1 実物と対照
print("fig1", flush=True)
rng = np.random.default_rng(3)
Rm = rng.random((N, N)) < base
fig, axes = plt.subplots(2, 1, figsize=(7.4, 15.0))
for ax, img, ttl in [(axes[0], P, f"素数を渦巻きに並べた（{N}×{N}、{int(P.sum()):,}個）"),
                     (axes[1], Rm, f"同じ密度でランダムに撒いた（対照群、{int(Rm.sum()):,}個）")]:
    ax.imshow(~img, cmap="gray", interpolation="nearest")
    ax.set_title(ttl, fontsize=15, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("点の数はほぼ同じ。斜めの筋があるのはどちらか", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig("fig1_spiral.png", dpi=95)
plt.close(fig)

# ---------------------------------------------------------------- fig2 直線密度
print("fig2", flush=True)
s = R["exp1"]["summary"]
names = list(s.keys())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.2))
x = np.arange(len(names))
a1.bar(x - 0.2, [s[n]["max"] * 100 for n in names], width=0.38, color=ACC, label="実測の最大")
a1.bar(x + 0.2, [s[n]["rand_max"] * 100 for n in names], width=0.38, color=SUB,
       label="ランダム対照200回の最大")
a1.axhline(base * 100, ls="--", color="#3d4753", lw=1.6)
a1.text(len(names) - 0.5, base * 100 + 0.6, f"全体の素数密度 {base*100:.2f}%", ha="right",
        fontsize=10.5, color="#3d4753")
a1.set_xticks(x)
a1.set_xticklabels(names, fontsize=11)
a1.set_ylabel("その向きで最も濃い直線の素数率（%）")
a1.set_title("斜めの線だけが飛び抜けている", fontsize=13)
a1.legend(fontsize=10)
a1.grid(alpha=0.3, axis="y")

labels = [r["label"] for r in BL["by_length"]]
tops = [r["top"][0][0] * 100 for r in BL["by_length"]]
a2.bar(labels, tops, color=INK, width=0.55)
for i, (lab, r) in enumerate(zip(labels, BL["by_length"])):
    a2.text(i, r["top"][0][0] * 100 + 0.6, f"{r['top'][0][0]*100:.1f}%\n(長さ{r['top'][0][1]})",
            ha="center", fontsize=10.5, color="#3d4753")
a2.axhline(BL["long_rand_max"] * 100, ls="--", color=ACC, lw=1.8)
a2.text(2.45, BL["long_rand_max"] * 100 + 0.6, f"長い線でのランダム上限 {BL['long_rand_max']*100:.1f}%",
        ha="right", fontsize=10.5, color=ACC)
a2.set_ylim(0, 48)
a2.set_ylabel("最も濃い右下がり対角の素数率（%）")
a2.set_title("短い線が有利なわけではなかった", fontsize=13)
a2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig2_density.png", dpi=125)
plt.close(fig)

# ---------------------------------------------------------------- fig3 オイラー多項式
print("fig3", flush=True)
rows = R["exp3"]["rows"]
up = [r["upper"] for r in rows]
fr = [r["prime_frac"] * 100 for r in rows]
topc = R["exp3"]["top_c"]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
b1.plot(up, fr, "o-", color=ACC, lw=2.4, ms=9)
for u, f in zip(up, fr):
    b1.annotate(f"{f:.1f}%", (u, f), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10.5, color="#3d4753")
b1.plot(up, [100 / np.log(u ** 2 / 2) for u in up], "--", color=SUB, lw=2.0,
        label="同じ大きさの数が素数である確率")
b1.set_xscale("log")
b1.set_ylim(0, 108)
b1.set_xlabel("n の範囲（0〜この数まで）")
b1.set_ylabel("n²+n+41 が素数になる割合（%）")
b1.set_title("オイラーの多項式は n=0〜39 で全部素数", fontsize=13)
b1.legend(fontsize=10)
b1.grid(alpha=0.3, which="both")

cs = [c for c, f in topc[:8]]
fs = [f * 100 for c, f in topc[:8]]
b2.bar([str(c) for c in cs], fs, color=[ACC if c == 41 else INK for c in cs])
for i, f in enumerate(fs):
    b2.text(i, f + 0.6, f"{f:.1f}", ha="center", fontsize=10, color="#3d4753")
b2.set_xlabel("n²+n+c の c")
b2.set_ylabel("n=0〜999 で素数になる割合（%）")
b2.set_title("c を1から999まで全部試した上位8個", fontsize=13)
b2.set_ylim(0, 66)
b2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig3_euler.png", dpi=125)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
FR = 40
sizes = np.unique(np.round(np.geomspace(11, N, FR)).astype(int))
sizes = sizes[sizes % 2 == 1]
c = N // 2
fig, ax = plt.subplots(figsize=(4.8, 4.8))
ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(~P[c - 5:c + 6, c - 5:c + 6], cmap="gray", interpolation="nearest")
txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=12,
              fontweight="bold", color=ACC)


def fr_fn(i):
    h = sizes[i] // 2
    sub = P[c - h:c + h + 1, c - h:c + h + 1]
    im.set_data(~sub)
    im.set_extent([0, sub.shape[1], sub.shape[0], 0])
    ax.set_xlim(0, sub.shape[1]); ax.set_ylim(sub.shape[0], 0)
    txt.set_text(f"{sizes[i]}×{sizes[i]}   1〜{sizes[i]**2:,}")
    return im, txt


FuncAnimation(fig, fr_fn, frames=len(sizes), blit=False).save(
    "ulam.gif", writer=PillowWriter(fps=5), dpi=80)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im2 = Image.open("ulam.gif")
fl = [f.copy().convert("RGB").quantize(colors=8) for f in ImageSequence.Iterator(im2)]
fl[0].save("ulam.gif", save_all=True, append_images=fl[1:], loop=0, duration=230, optimize=True)
print(f"  ulam.gif {os.path.getsize('ulam.gif')/1024:.0f}KB / {len(fl)}フレーム")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

M = 161
h = M // 2
sub = P[c - h:c + h + 1, c - h:c + h + 1]
subg = G[c - h:c + h + 1, c - h:c + h + 1]
ys, xs = np.nonzero(sub)
fig = go.Figure(go.Scattergl(
    x=xs, y=-ys, mode="markers",
    marker=dict(size=5, color="#2f6f9e"),
    text=[f"{int(subg[y, x]):,}" for y, x in zip(ys, xs)],
    hovertemplate="この位置の数: %{text}（素数）<extra></extra>"))
fig.update_layout(
    title=dict(text=f"ウラムの螺旋（中心から{M}×{M}、点は素数。拡大・ホバーできます）",
               font=dict(size=14)),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    height=620, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)")
fig.write_html("ulam-plot.html", include_plotlyjs="cdn")
print(f"  ulam-plot.html {os.path.getsize('ulam-plot.html')/1024:.0f}KB")

# ---------------------------------------------------------------- hero
print("hero", flush=True)
img = np.zeros((N, N, 3), dtype=np.uint8)
img[...] = (12, 18, 28)
img[P] = (246, 205, 96)
pil = Image.fromarray(img).resize((1024, 1024), Image.NEAREST)
canvas = Image.new("RGB", (1536, 1024), (12, 18, 28))
canvas.paste(pil, ((1536 - 1024) // 2, 0))
canvas.save("hero.png")
print(f"  hero.png {os.path.getsize('hero.png')/1024:.0f}KB")
print("完了")
