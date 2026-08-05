# 記事用の図とGIFを作る。
#   fig1: マンデルブロ集合の全体像 + あとで次元を測る3地点
#   fig2: box-counting の log-log プロット(円・コッホ雪片・マンデルブロ)
#   fig3: 最大反復回数を上げたときの 面積 と 次元推定 の動き
#   zoom.gif: Misiurewicz点へのズーム

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from common import escape_time

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

EXTENT = 2.6
XMIN, XMAX = -2.0, 0.6
YMIN, YMAX = -1.3, 1.3

# ブログのモノトーン寄りの配色に合わせつつ、集合内は黒、外は反復回数で青〜白
CMAP = LinearSegmentedColormap.from_list(
    "shake", ["#0b1622", "#1b3a5c", "#2f6f9e", "#7fb2d4", "#dceaf4", "#ffffff"]
)


def colorize(esc, maxiter):
    v = esc.astype(float)
    v[esc == 0] = np.nan  # 集合内
    v = np.log1p(v) / np.log1p(maxiter)
    img = CMAP(v)
    img[np.isnan(v)] = (0.02, 0.03, 0.05, 1.0)  # 集合内は黒
    return img


# ---------------------------------------------------------------- fig1
print("fig1: 全体像", flush=True)
R = 2000
esc = escape_time(XMIN, XMAX, YMIN, YMAX, R, R, 800)
fig, ax = plt.subplots(figsize=(7.2, 7.2))
ax.imshow(colorize(esc, 800), extent=[XMIN, XMAX, YMIN, YMAX], origin="lower")
theta = 2.0
smooth = np.exp(1j * theta) / 2 - np.exp(2j * theta) / 4
marks = [
    (smooth.real, smooth.imag, "なめらかな縁", "#ffd166"),
    (-0.10109636384562, 0.95628651080914, "Misiurewicz点", "#ef476f"),
    (-0.75, 0.01, "シーホースバレー", "#06d6a0"),
]
for x, y, label, col in marks:
    ax.plot(x, y, "o", ms=9, mfc="none", mec=col, mew=2.2)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(12, 10),
                color=col, fontsize=11, fontweight="bold")
ax.set_xlabel("Re(c)")
ax.set_ylabel("Im(c)")
ax.set_title("マンデルブロ集合と、あとで次元を測る3地点", fontsize=13)
fig.tight_layout()
fig.savefig("fig1_overview.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig2
print("fig2: box-counting log-log", flush=True)
d2 = json.load(open("dimension2.json", encoding="utf-8"))
calib = d2["calibration"]["4096"]
row = [r for r in d2["iter_sweep"]["rows"] if r["maxiter"] == 20000][0]
sizes = np.array(d2["fit_sizes_px"], dtype=float)
eps = sizes * EXTENT / 4096

series = [
    ("円 (理論 1.0000)", None, "#8d99ae"),
    ("コッホ雪片 (理論 1.2619)", None, "#ffd166"),
    ("マンデルブロ集合 (理論 2.0000)", row["counts"], "#ef476f"),
]
# 円とコッホの箱数は dimension2.json に入っていないので測り直す
from common import circle_mask, koch_mask  # noqa: E402


def boundary_pixels(inside):
    b = np.zeros_like(inside)
    b[1:, :] |= inside[1:, :] & ~inside[:-1, :]
    b[:-1, :] |= inside[:-1, :] & ~inside[1:, :]
    b[:, 1:] |= inside[:, 1:] & ~inside[:, :-1]
    b[:, :-1] |= inside[:, :-1] & ~inside[:, 1:]
    return b


def box_count(mask, sz):
    RR = mask.shape[0]
    return np.array([int(mask[: (RR // b) * b, : (RR // b) * b]
                         .reshape(RR // b, b, RR // b, b).any(axis=(1, 3)).sum()) for b in sz])


series[0] = (series[0][0], box_count(boundary_pixels(circle_mask(4096, EXTENT)),
                                     [int(s) for s in sizes]), series[0][2])
series[1] = (series[1][0], box_count(boundary_pixels(koch_mask(4096, 7, EXTENT)),
                                     [int(s) for s in sizes]), series[1][2])

fig, ax = plt.subplots(figsize=(7.6, 5.6))
for label, counts, col in series:
    c = np.array(counts, dtype=float)
    D = np.polyfit(np.log(1 / eps), np.log(c), 1)[0]
    ax.plot(1 / eps, c, "o-", color=col, lw=2, ms=7, label=f"{label}  実測 D={D:.3f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("1/ε  (箱を細かくするほど右)")
ax.set_ylabel("境界を覆うのに必要な箱の数 N(ε)")
ax.set_title("box-counting: 傾きがそのままフラクタル次元になる", fontsize=13)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("fig2_boxcount.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig3
print("fig3: 反復回数スイープ", flush=True)
rows = d2["iter_sweep"]["rows"]
it = [r["maxiter"] for r in rows]
Ds = [r["D"] for r in rows]
areas = [r["area"] for r in rows]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 7.4), sharex=True)
a1.plot(it, areas, "o-", color="#2f6f9e", lw=2, ms=6)
a1.axhline(1.5065, ls="--", color="#8d99ae", lw=1.5)
a1.annotate("文献値 ≈ 1.5065", (it[2], 1.5065), textcoords="offset points",
            xytext=(0, -18), color="#5c6672", fontsize=10)
a1.set_ylabel("面積の推定値")
a1.set_title("反復回数を増やすと、面積は収束する", fontsize=13)
a1.grid(alpha=0.3)

a2.plot(it, Ds, "o-", color="#ef476f", lw=2, ms=6)
a2.axhline(2.0, ls="--", color="#8d99ae", lw=1.5)
a2.annotate("理論値 = 2 (Shishikura 1998)", (it[0], 2.0), textcoords="offset points",
            xytext=(6, -18), color="#5c6672", fontsize=10)
a2.set_xscale("log")
a2.set_ylim(0.9, 2.15)
a2.set_xlabel("最大反復回数")
a2.set_ylabel("box-counting で測った次元")
a2.set_title("しかし次元は、途中で上がってから逆に下がって止まる", fontsize=13)
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig3_iter_sweep.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- zoom.gif
print("zoom.gif: Misiurewicz点へのズーム", flush=True)
CX, CY = -0.10109636384562, 0.95628651080914
FRAMES = 42
RG = 420
half = np.geomspace(1.4, 2e-6, FRAMES)

fig, ax = plt.subplots(figsize=(5.0, 5.0))
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(np.zeros((RG, RG, 4)), origin="lower")
txt = ax.text(0.03, 0.955, "", transform=ax.transAxes, color="white",
              fontsize=12, fontweight="bold", va="top")


def frame(k):
    w = half[k]
    mi = int(300 + 900 * np.log10(1.4 / w))
    e = escape_time(CX - w, CX + w, CY - w, CY + w, RG, RG, mi)
    im.set_data(colorize(e, mi))
    txt.set_text(f"×{1.4 / w:,.0f}   最大反復 {mi}")
    return im, txt


ani = FuncAnimation(fig, frame, frames=FRAMES, blit=False)
ani.save("zoom.gif", writer=PillowWriter(fps=6), dpi=84)
plt.close(fig)
print("完了: fig1_overview.png / fig2_boxcount.png / fig3_iter_sweep.png / zoom.gif")
