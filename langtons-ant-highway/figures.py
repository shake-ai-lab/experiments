import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from core import run_ant

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB, GOOD = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae", "#06a77d"
HW = R["exp1"]["highway_start"]

# ---------------------------------------------------------------- fig1
print("fig1", flush=True)
black = np.array(R["exp1"]["black"])
step = np.arange(len(black)) * 10
fig, ax = plt.subplots(figsize=(8.8, 5.2))
ax.axvspan(0, 500, color=GOOD, alpha=0.10)
ax.axvspan(500, HW, color=WARN, alpha=0.10)
ax.axvspan(HW, step[-1], color=ACC, alpha=0.10)
ax.plot(step, black, color=INK, lw=2.2)
ax.axvline(HW, ls="--", color=ACC, lw=1.8)
ax.annotate(f"{HW:,}歩でハイウェイが始まる", (HW, 400), textcoords="offset points",
            xytext=(10, 0), color=ACC, fontsize=11.5, fontweight="bold")
for x, t, col in [(250, "対称な模様", GOOD), (5000, "カオス", WARN), (15000, "ハイウェイ", ACC)]:
    ax.text(x, 1780, t, ha="center", fontsize=12, color=col, fontweight="bold")
ax.set_xlabel("ステップ")
ax.set_ylabel("黒いマスの数")
ax.set_title("同じルールのまま、3つの局面を通る", fontsize=13.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig1_phases.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig2 スナップショット
print("fig2", flush=True)
snaps = {}
N = 560
grid = np.zeros((N, N), dtype=np.uint8)
DIRS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
pos = np.array([N // 2, N // 2])
d = 0
want = [400, 2000, 6000, 11000, 20000]
for t in range(1, max(want) + 1):
    c = grid[pos[0], pos[1]]
    d = (d + (1 if c == 0 else -1)) % 4
    grid[pos[0], pos[1]] = 1 - c
    pos = (pos + DIRS[d]) % N
    if t in want:
        snaps[t] = grid.copy()

fig, axes = plt.subplots(len(want), 1, figsize=(6.2, 5.0 * len(want)))
for ax, t in zip(axes, want):
    g = snaps[t]
    ys, xs = np.nonzero(g)
    pad = 12
    y0, y1 = ys.min() - pad, ys.max() + pad
    x0, x1 = xs.min() - pad, xs.max() + pad
    ax.imshow(g[y0:y1, x0:x1], cmap="binary", interpolation="nearest")
    ax.set_title(f"{t:,} ステップ  （黒 {int(g.sum()):,} マス）", fontsize=15, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("同じルールを回し続けただけ", fontsize=17)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig("fig2_snapshots.png", dpi=98)
plt.close(fig)

# ---------------------------------------------------------------- fig3
print("fig3", flush=True)
rows = R["exp2"]["rows"]
p = np.array([r["p"] for r in rows]) * 100
med = np.array([r["median"] if r["median"] else np.nan for r in rows])
fail = np.array([r["fail"] for r in rows])
fig, ax = plt.subplots(figsize=(8.6, 5.2))
okm = ~np.isnan(med)
ax.plot(p[okm], med[okm], "o-", color=INK, lw=2.4, ms=9)
for xx, yy, ff in zip(p, med, fail):
    if not np.isnan(yy):
        ax.annotate(f"{yy:,.0f}", (xx, yy), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=10, color="#3d4753")
bad = np.isnan(med)
ax.scatter(p[bad], np.full(bad.sum(), 3e5), marker="x", s=110, color=ACC, lw=2.6)
ax.axhline(3e5, ls=":", color=ACC, lw=1.6)
ax.text(50, 3.4e5, "×＝30万歩まで回してもハイウェイが現れなかった", ha="right",
        fontsize=10.5, color=ACC)
ax.set_yscale("log")
ax.set_xscale("symlog", linthresh=0.1)
ax.set_ylim(2e3, 6e5)
ax.set_xlabel("最初に撒いた黒マスの割合（%）")
ax.set_ylabel("ハイウェイが始まるまでの歩数（中央値）")
ax.set_title("盤が2%散らかっているだけで、ハイウェイは30万歩でも現れなくなる", fontsize=13)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig3_noise.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig4 ルール別
print("fig4", flush=True)
rules = [r["rule"] for r in R["exp3"]]
fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4))
for ax, rec in zip(axes.ravel(), R["exp3"]):
    g, tr, _, _ = run_ant(60000, N=1400, rule=rec["rule"], record_every=False)
    ys, xs = np.nonzero(g)
    pad = 8
    ax.imshow(g[ys.min() - pad:ys.max() + pad, xs.min() - pad:xs.max() + pad],
              cmap="cubehelix_r", interpolation="nearest")
    hw = f"{rec['highway']:,}歩でハイウェイ" if rec["highway"] else "ハイウェイなし"
    ax.set_title(f"ルール {rec['rule']}\n{hw} ／ 塗られたマス {rec['cells']:,}",
                 fontsize=11.5, color=ACC if rec["highway"] else "#3d4753")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("曲がり方の並びを変えただけで、出てくる模様はまるで違う（各6万歩）", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig4_rules.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
N = 300
grid = np.zeros((N, N), dtype=np.uint8)
pos = np.array([N // 2 + 40, N // 2 + 40])
d = 0
FR = 44
marks = np.unique(np.round(np.geomspace(60, 13000, FR)).astype(int))
frames = []
t = 0
for target in marks:
    while t < target:
        c = grid[pos[0], pos[1]]
        d = (d + (1 if c == 0 else -1)) % 4
        grid[pos[0], pos[1]] = 1 - c
        pos = (pos + DIRS[d]) % N
        t += 1
    frames.append((t, grid.copy(), pos.copy()))

fig, ax = plt.subplots(figsize=(4.8, 4.8))
ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(frames[0][1][60:260, 60:260], cmap="binary", interpolation="nearest",
               vmin=0, vmax=1)
txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=12,
              fontweight="bold", color=ACC)


def fr_fn(k):
    t, g, _ = frames[k]
    im.set_data(g[60:260, 60:260])
    txt.set_text(f"{t:,} ステップ" + ("   ← ハイウェイ" if t >= HW else ""))
    return im, txt


FuncAnimation(fig, fr_fn, frames=len(frames), blit=False).save(
    "ant.gif", writer=PillowWriter(fps=5), dpi=80)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im2 = Image.open("ant.gif")
fl = [f.copy().convert("RGB").quantize(colors=8) for f in ImageSequence.Iterator(im2)]
fl[0].save("ant.gif", save_all=True, append_images=fl[1:], loop=0, duration=230, optimize=True)
print(f"  ant.gif {os.path.getsize('ant.gif')/1024:.0f}KB / {len(fl)}フレーム")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

traj = np.load("traj.npy")[:20000]
sub = traj[::6]
fig = go.Figure(go.Scatter3d(
    x=sub[:, 1], y=sub[:, 0], z=np.arange(len(sub)) * 6, mode="lines",
    line=dict(width=3, color=np.arange(len(sub)),
              colorscale=[[0, "#06a77d"], [0.45, "#e8a33d"], [0.55, "#ef476f"], [1, "#ef476f"]]),
    hoverinfo="skip"))
fig.update_layout(
    title=dict(text="アリの足取りを時空図にする（高さがステップ数。最後に一直線に伸びるのがハイウェイ）",
               font=dict(size=13.5)),
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="ステップ",
               camera=dict(eye=dict(x=1.5, y=-1.4, z=0.7)),
               aspectratio=dict(x=1, y=1, z=1.3)),
    height=600, margin=dict(l=0, r=0, t=48, b=0), paper_bgcolor="rgba(0,0,0,0)")
fig.write_html("ant-3d.html", include_plotlyjs="cdn")
print(f"  ant-3d.html {os.path.getsize('ant-3d.html')/1024:.0f}KB")

# ---------------------------------------------------------------- hero
print("hero", flush=True)
g, tr, _, _ = run_ant(45000, N=1200, record_every=False)
ys, xs = np.nonzero(g)
pad = 14
sub = g[ys.min() - pad:ys.max() + pad, xs.min() - pad:xs.max() + pad]
h, w = sub.shape
fig = plt.figure(figsize=(15.36, 10.24), facecolor="#0e1620")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor("#0e1620")
ax.imshow(sub, cmap=matplotlib.colors.ListedColormap(["#0e1620", "#f2c14e"]),
          interpolation="nearest", aspect="auto")
ax.set_xticks([]); ax.set_yticks([])
fig.savefig("hero.png", dpi=100, facecolor="#0e1620")
plt.close(fig)
print(f"  hero.png {os.path.getsize('hero.png')/1024:.0f}KB")
print("完了")
