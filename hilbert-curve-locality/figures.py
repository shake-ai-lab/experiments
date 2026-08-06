import gzip
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from hilbert_core import hilbert_d2xy, hilbert_index, morton_index, row_index

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
D2 = json.load(open("dim2.json", encoding="utf-8"))
INK, ACC, WARN, SUB, GOOD = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae", "#06a77d"

# ---------------------------------------------------------------- fig1 3つの並べ方
print("fig1", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
for ax, (name, order) in zip(axes, [("ヒルベルト順", 5), ("Z階数（モートン順）", 5), ("行優先", 5)]):
    n = 1 << order
    if name.startswith("ヒ"):
        d = np.arange(n * n)
        x, y = hilbert_d2xy(order, d)
    elif name.startswith("Z"):
        M = morton_index(order)
        o = np.argsort(M.ravel())
        y, x = np.unravel_index(o, (n, n))
    else:
        Rw = row_index(order)
        o = np.argsort(Rw.ravel())
        y, x = np.unravel_index(o, (n, n))
    ax.plot(x, y, lw=1.1, color=INK)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}（{n}×{n}）", fontsize=13)
fig.suptitle("平面のマスに順番をつける3つのやり方", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig1_orders.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- fig2 局所性
print("fig2", flush=True)
rows = R["exp1"]["rows"]
names = [r["name"] for r in rows]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))
x = np.arange(len(names))
for i, key, lab, col in [(0, "median", "中央値", GOOD), (1, "mean", "平均", INK),
                         (2, "max", "最大", ACC)]:
    a1.bar(x + (i - 1) * 0.27, [max(r[key], 0.6) for r in rows], width=0.25, color=col,
           label=lab)
a1.set_yscale("log")
a1.set_xticks(x)
a1.set_xticklabels(names, fontsize=10.5)
a1.set_ylabel("2次元で隣り合う画素の、1次元での距離")
a1.set_title("中央値ではヒルベルトが圧勝、最大では逆転される", fontsize=13)
a1.legend(fontsize=10)
a1.grid(alpha=0.3, axis="y", which="both")
for i, r in enumerate(rows):
    a1.text(i - 0.27, max(r["median"], 0.6) * 1.25, f"{r['median']:.0f}", ha="center",
            fontsize=10, color="#3d4753")
    a1.text(i + 0.27, r["max"] * 1.25, f"{r['max']:,}", ha="center", fontsize=10,
            color="#3d4753")

ks = R["exp1b"]["ks"]
for r, col in zip(R["exp1b"]["rows"], [GOOD, INK, SUB]):
    a2.plot(ks, r["dists"], "o-", lw=2.4, ms=8, color=col, label=r["name"])
a2.set_xscale("log")
a2.set_yscale("log")
a2.set_xlabel("1次元で離れている距離 k")
a2.set_ylabel("そのときの2次元での平均距離")
a2.set_title("逆向きに見るとヒルベルトが明確に勝つ", fontsize=13)
a2.annotate("行優先のk=256は\n「ちょうど1行下」に当たる偶然",
            (256, 1.0), textcoords="offset points", xytext=(-10, 26), ha="right",
            fontsize=10, color=SUB, arrowprops=dict(arrowstyle="->", color=SUB))
a2.legend(fontsize=10)
a2.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig2_locality.png", dpi=122)
plt.close(fig)

# ---------------------------------------------------------------- fig3 次元
print("fig3", flush=True)
G = 2048
sizes = np.array([2, 4, 8, 16, 32, 64, 128], float)
fig, ax = plt.subplots(figsize=(8.6, 5.4))
cols = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(D2)))
for rec, c in zip(D2, cols):
    counts = np.array(rec["counts"], float)
    ax.plot(G / sizes, counts, "o-", lw=2.0, ms=6, color=c,
            label=f"次数 {rec['order']}（{1<<rec['order']}×{1<<rec['order']}）")
xx = np.array([16, 1024])
ax.plot(xx, xx * 250, "--", color="#3d4753", lw=1.6)
ax.text(300, 300 * 250 * 1.3, "傾き1（ただの線）", fontsize=11, color="#3d4753")
ax.plot(xx, xx ** 2 * 0.25, ":", color=ACC, lw=2.0)
ax.text(40, 40 ** 2 * 0.25 * 0.35, "傾き2（面を埋めている）", fontsize=11, color=ACC)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("1/ε（箱を細かくするほど右）")
ax.set_ylabel("曲線を覆う箱の数")
ax.set_title("同じ曲線が、粗く見れば面・細かく見れば線", fontsize=13.5)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig3_dimension.png", dpi=128)
plt.close(fig)

# ---------------------------------------------------------------- fig4 圧縮
print("fig4", flush=True)
rows3 = R["exp3"]
labels = [r["image"] for r in rows3]
fig, ax = plt.subplots(figsize=(9.2, 5.0))
x = np.arange(len(labels))
for i, key, col in [(0, "行優先", INK), (1, "Z階数", SUB), (2, "ヒルベルト順", ACC)]:
    ax.bar(x + (i - 1) * 0.27, [r[key] / 1024 for r in rows3], width=0.25, color=col,
           label=key)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("gzip後のサイズ（KB）")
ax.set_title("局所性を保つ並べ方は、圧縮では負けた", fontsize=13.5)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")
for i, r in enumerate(rows3):
    ax.text(i + 0.27, r["ヒルベルト順"] / 1024 + 0.8,
            f"{(r['ヒルベルト順']-r['行優先'])/r['行優先']*100:+.0f}%",
            ha="center", fontsize=10.5, color=ACC, fontweight="bold")
fig.tight_layout()
fig.savefig("fig4_compress.png", dpi=128)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
fig, ax = plt.subplots(figsize=(4.9, 4.9))
FR = 36
order = 6
n = 1 << order
d = np.arange(n * n)
hx, hy = hilbert_d2xy(order, d)
marks = np.unique(np.round(np.geomspace(4, n * n, FR)).astype(int))


def fr(i):
    ax.clear()
    m = marks[i]
    ax.plot(hx[:m], hy[:m], lw=1.2, color=INK)
    ax.plot(hx[m - 1], hy[m - 1], "o", ms=6, color=ACC)
    ax.set_xlim(-1, n); ax.set_ylim(-1, n)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{m:,} / {n*n:,} マス目", fontsize=13)
    return []


FuncAnimation(fig, fr, frames=len(marks), blit=False).save(
    "hilbert.gif", writer=PillowWriter(fps=6), dpi=78)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im = Image.open("hilbert.gif")
fl = [f.copy().convert("RGB").quantize(colors=16) for f in ImageSequence.Iterator(im)]
fl[0].save("hilbert.gif", save_all=True, append_images=fl[1:], loop=0, duration=180,
           optimize=True)
print(f"  hilbert.gif {os.path.getsize('hilbert.gif')/1024:.0f}KB / {len(fl)}フレーム")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

o = 6
n = 1 << o
d = np.arange(n * n)
hx, hy = hilbert_d2xy(o, d)
fig = go.Figure(go.Scattergl(
    x=hx, y=hy, mode="lines",
    line=dict(width=2, color="#2f6f9e"),
    hoverinfo="skip"))
fig.add_trace(go.Scattergl(x=hx[::64], y=hy[::64], mode="markers",
                           marker=dict(size=5, color="#ef476f"),
                           text=[f"{v:,}番目" for v in d[::64]],
                           hovertemplate="%{text}<extra></extra>", showlegend=False))
fig.update_layout(
    title=dict(text=f"{n}×{n} のヒルベルト曲線（拡大して、折り返しの入れ子構造を見てください）",
               font=dict(size=14)),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    height=620, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
fig.write_html("hilbert-plot.html", include_plotlyjs="cdn")
print(f"  hilbert-plot.html {os.path.getsize('hilbert-plot.html')/1024:.0f}KB")

# ---------------------------------------------------------------- hero
print("hero", flush=True)
o = 7
n = 1 << o
d = np.arange(n * n)
hx, hy = hilbert_d2xy(o, d)
fig = plt.figure(figsize=(15.36, 10.24), facecolor="#0b1220")
ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
ax.set_facecolor("#0b1220")
seg = 256
cm = plt.get_cmap("turbo")
for i in range(0, len(hx) - 1, seg):
    ax.plot(hx[i:i + seg + 1], hy[i:i + seg + 1], lw=2.0,
            color=cm(0.08 + 0.85 * i / len(hx)))
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig("hero.png", dpi=100, facecolor="#0b1220")
plt.close(fig)
print(f"  hero.png {os.path.getsize('hero.png')/1024:.0f}KB")
print("完了")
