import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from newton_core import newton_basins, boundary_mask

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB, GOOD = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae", "#06a77d"
PAL3 = ListedColormap(["#1b1f26", "#2f6f9e", "#ef476f", "#e8a33d"])
roots3 = [np.exp(2j * np.pi * k / 3) for k in range(3)]
f3 = lambda z: z ** 3 - 1          # noqa: E731
d3 = lambda z: 3 * z ** 2          # noqa: E731


def shade(w, it, cmap=PAL3):
    """根ごとの色に、収束の速さで明るさをつける"""
    base = cmap((w.astype(int) + 1) / 3.0)[:, :, :3]
    v = 1.0 - np.clip(it / (it.max() * 0.55), 0, 0.75)
    return base * v[:, :, None]


# ---------------------------------------------------------------- fig1
print("fig1", flush=True)
w = np.load("w3.npy")
it = np.load("it3.npy")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.3))
axes[0].imshow(shade(w, it), origin="lower", extent=[-2, 2, -2, 2])
axes[0].add_patch(plt.Rectangle((-0.35, 0.35), 0.35, 0.35, fill=False, ec="white", lw=1.8))
axes[0].set_title("z³−1=0 をニュートン法で解いたときの行き先", fontsize=13)
wz, itz = newton_basins(f3, d3, roots3, 900, extent=0.175)
# 中心をずらして拡大(枠の中身)
x = np.linspace(-0.35, 0.0, 900)
y = np.linspace(0.35, 0.70, 900)
Z = (x[None, :] + 1j * y[:, None]).ravel()
which = np.full(Z.size, -1, dtype=np.int8)
iters = np.full(Z.size, 60, dtype=np.int16)
idx = np.arange(Z.size)
for i in range(60):
    Z = Z - f3(Z) / d3(Z)
    done = np.zeros(Z.size, dtype=bool)
    for r, root in enumerate(roots3):
        hit = (np.abs(Z - root) < 1e-6) & ~done
        which[idx[hit]] = r
        iters[idx[hit]] = i + 1
        done |= hit
    if done.any():
        keep = ~done
        idx, Z = idx[keep], Z[keep]
        if idx.size == 0:
            break
wz = which.reshape(900, 900)
itz = iters.reshape(900, 900)
axes[1].imshow(shade(wz, itz), origin="lower")
axes[1].set_title("白枠の中を拡大（同じ構造が入れ子になっている）", fontsize=13)
for a in axes:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("fig1_basins.png", dpi=118)
plt.close(fig)

# ---------------------------------------------------------------- fig2 Wada
print("fig2", flush=True)
rows = R["exp2"]
fig, ax = plt.subplots(figsize=(9.0, 5.2))
for RR, col in zip([512, 1024, 2048], [SUB, INK, ACC]):
    sub = [r for r in rows if r["R"] == RR]
    ax.plot([r["rad"] for r in sub], [r["frac3"] * 100 for r in sub], "o-",
            lw=2.4, ms=9, color=col, label=f"解像度 {RR}×{RR}")
ax.axhline(R["exp2_control"] * 100, ls="--", color="#3d4753", lw=1.8)
ax.text(4, R["exp2_control"] * 100 + 3, f"対照：ふつうに3等分した境界 {R['exp2_control']*100:.2f}%",
        ha="right", fontsize=11, color="#3d4753")
ax.set_ylim(-4, 108)
ax.set_xticks([1, 2, 4])
ax.set_xlabel("境界画素のまわりを見る半径（画素）")
ax.set_ylabel("3つの領域すべてが見えた境界画素の割合（%）")
ax.set_title("どの国境も、3か国すべてに接している", fontsize=13.5)
ax.legend(fontsize=10.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig2_wada.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig3
print("fig3", flush=True)
w5 = np.load("w5.npy")
w2 = np.load("w2.npy")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.3))
P5 = ListedColormap(["#1b1f26", "#2f6f9e", "#ef476f", "#e8a33d", "#06a77d", "#7b5ea7"])
axes[0].imshow(P5((w5.astype(int) + 1) / 5.0), origin="lower")
axes[0].set_title(f"z⁵−1=0（根が5つ）／境界次元 {R['exp3']['D5']:.3f}", fontsize=13)
img = np.zeros(w2.shape + (3,))
P4 = ListedColormap(["#ef476f", "#2f6f9e", "#e8a33d", "#06a77d"])
img = P4((w2.astype(int) + 1) / 4.0)[:, :, :3]
img[w2 == -1] = (0.95, 0.25, 0.35)
axes[1].imshow(img, origin="lower")
axes[1].set_title(f"z³−2z+2（赤＝200回まわしても収束しない {R['exp3']['none_trap']*100:.2f}%）",
                  fontsize=13)
for a in axes:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("fig3_others.png", dpi=118)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
CX, CY = -0.2, 0.52
FR = 34
half = np.geomspace(1.6, 3e-4, FR)
fig, ax = plt.subplots(figsize=(4.8, 4.8))
ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(np.zeros((340, 340, 3)), origin="lower")
txt = ax.text(0.03, 0.965, "", transform=ax.transAxes, va="top", color="white",
              fontsize=11.5, fontweight="bold")


def fr(k):
    h = half[k]
    n = 340
    x = np.linspace(CX - h, CX + h, n)
    y = np.linspace(CY - h, CY + h, n)
    Z = (x[None, :] + 1j * y[:, None]).ravel()
    wh = np.full(Z.size, -1, dtype=np.int8)
    itt = np.full(Z.size, 80, dtype=np.int16)
    idx = np.arange(Z.size)
    for i in range(80):
        Z = Z - f3(Z) / d3(Z)
        done = np.zeros(Z.size, dtype=bool)
        for r, root in enumerate(roots3):
            hit = (np.abs(Z - root) < 1e-6) & ~done
            wh[idx[hit]] = r
            itt[idx[hit]] = i + 1
            done |= hit
        if done.any():
            keep = ~done
            idx, Z = idx[keep], Z[keep]
            if idx.size == 0:
                break
    im.set_data(shade(wh.reshape(n, n), itt.reshape(n, n)))
    txt.set_text(f"×{1.6/h:,.0f}")
    return im, txt


FuncAnimation(fig, fr, frames=FR, blit=False).save(
    "newton.gif", writer=PillowWriter(fps=5), dpi=78)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im2 = Image.open("newton.gif")
fl = [f.copy().convert("RGB").quantize(colors=48) for f in ImageSequence.Iterator(im2)]
fl[0].save("newton.gif", save_all=True, append_images=fl[1:], loop=0, duration=200,
           optimize=True)
print(f"  newton.gif {os.path.getsize('newton.gif')/1024:.0f}KB / {len(fl)}フレーム")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

n = 150
wq, iq = newton_basins(f3, d3, roots3, n, extent=1.4)
z = np.log1p(iq.astype(float))
fig = go.Figure(go.Surface(
    z=z, surfacecolor=wq.astype(float),
    colorscale=[[0, "#2f6f9e"], [0.5, "#ef476f"], [1, "#e8a33d"]],
    showscale=False,
    lighting=dict(ambient=0.6, diffuse=0.8), hoverinfo="skip"))
fig.update_layout(
    title=dict(text="高さ＝収束までの反復回数、色＝行き着いた根（回せます）", font=dict(size=14)),
    scene=dict(xaxis_title="Re(z₀)", yaxis_title="Im(z₀)", zaxis_title="log(反復回数)",
               camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
               aspectratio=dict(x=1, y=1, z=0.5)),
    height=580, margin=dict(l=0, r=0, t=45, b=0), paper_bgcolor="rgba(0,0,0,0)")
fig.write_html("newton-3d.html", include_plotlyjs="cdn")
print(f"  newton-3d.html {os.path.getsize('newton-3d.html')/1024:.0f}KB")

# ---------------------------------------------------------------- hero
print("hero", flush=True)
W, H = 1536, 1024
x = np.linspace(-1.6 * W / H, 1.6 * W / H, W)
y = np.linspace(-1.6, 1.6, H)
Z = (x[None, :] + 1j * y[:, None]).ravel()
wh = np.full(Z.size, -1, dtype=np.int8)
itt = np.full(Z.size, 80, dtype=np.int16)
idx = np.arange(Z.size)
for i in range(80):
    Z = Z - f3(Z) / d3(Z)
    done = np.zeros(Z.size, dtype=bool)
    for r, root in enumerate(roots3):
        hit = (np.abs(Z - root) < 1e-8) & ~done
        wh[idx[hit]] = r
        itt[idx[hit]] = i + 1
        done |= hit
    if done.any():
        keep = ~done
        idx, Z = idx[keep], Z[keep]
        if idx.size == 0:
            break
wh = wh.reshape(H, W)
itt = itt.reshape(H, W)
cols = np.array([[0.18, 0.44, 0.62], [0.94, 0.28, 0.44], [0.91, 0.64, 0.24]])
img = cols[np.clip(wh, 0, 2)]
v = 1.0 - np.clip(itt / 26.0, 0, 0.82)
img = np.clip(img * v[:, :, None] * 1.25, 0, 1)
Image.fromarray((img * 255).astype(np.uint8)).save("hero.png")
print(f"  hero.png {os.path.getsize('hero.png')/1024:.0f}KB")
print("完了")
