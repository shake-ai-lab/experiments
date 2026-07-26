# -*- coding: utf-8 -*-
"""
見せ場のGIF: 同じ乱数場で p を少しずつ上げ、クラスタが合体して
ある瞬間に上下を貫通する様子をアニメーションにする。

クラスタの色は「クラスタ内で最小の乱数値」から決める。
合体すると小さい方の乱数値(=先にできた芯)の色を引き継ぐので、
フレーム間で色が安定し、合体の様子が追える。
貫通クラスタは黄色でハイライトする。
"""
import io
import colorsys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "Meiryo"

STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
OUT = "C:/tmp/percolation-phase-transition-experiment/"

BG = np.array([0.075, 0.09, 0.11])       # 空きサイト: 暗い背景
SPAN_COLOR = np.array([1.0, 0.83, 0.23])  # 貫通クラスタ: 黄色


def spanning_label(labels):
    common = np.intersect1d(np.unique(labels[0]), np.unique(labels[-1]))
    common = common[common > 0]
    return common[0] if len(common) else 0


def render_rgb(r, p, sat=0.55, val=0.78):
    """占有率 p での格子をRGB画像(クラスタ色分け)にする。"""
    occ = r < p
    labels, n = ndimage.label(occ, structure=STRUCT)
    img = np.tile(BG, (*r.shape, 1))
    if n == 0:
        return img, 0, labels
    idx = np.arange(1, n + 1)
    minr = ndimage.minimum(r, labels, index=idx)          # クラスタごとの最小乱数値
    hues = (minr * 9973.0) % 1.0                          # 安定な色ハッシュ
    lut = np.zeros((n + 1, 3))
    for i, h in enumerate(hues):
        lut[i + 1] = colorsys.hsv_to_rgb(h, sat, val)
    sp = spanning_label(labels)
    if sp:
        lut[sp] = SPAN_COLOR
    img = lut[labels]
    img[~occ] = BG
    return img, sp, labels


# ================= GIF =================
L = 160
rng = np.random.default_rng(7)
r = rng.random((L, L))

p_frames = np.round(np.arange(0.50, 0.6601, 0.005), 4)

frames = []
span_seen = False
for p in p_frames:
    img, sp, _ = render_rgb(r, p)
    fig, ax = plt.subplots(figsize=(4.2, 4.55), dpi=80)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    if sp:
        ax.set_title(f"p = {p:.3f} — 貫通!", fontsize=13, color="#c92a2a",
                     fontweight="bold")
    else:
        ax.set_title(f"p = {p:.3f}", fontsize=13)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    frame = Image.open(buf).convert("RGB")
    frames.append((frame, bool(sp)))
    if sp and not span_seen:
        span_seen = True
        print(f"GIF: first spanning at p={p}", flush=True)

# ホールド(最初・貫通の瞬間・最後を長めに見せる)
seq = []
durations = []
first_span_idx = next(i for i, (_, s) in enumerate(frames) if s)
for i, (f, s) in enumerate(frames):
    seq.append(f)
    if i == 0 or i == len(frames) - 1:
        durations.append(1200)
    elif i == first_span_idx:
        durations.append(1500)
    else:
        durations.append(160)

pal_frames = [f.quantize(colors=128, method=Image.MAXCOVERAGE, dither=0)
              for f in seq]
pal_frames[0].save(OUT + "percolation.gif", save_all=True,
                   append_images=pal_frames[1:], duration=durations, loop=0,
                   optimize=True)
import os
print(f"GIF size: {os.path.getsize(OUT + 'percolation.gif')/1e6:.2f} MB")

# ================= スナップショット図(縦スタック3枚) =================
L2 = 512
rng2 = np.random.default_rng(123)
r2 = rng2.random((L2, L2))
ps = [0.55, 0.5927, 0.65]
titles = ["p = 0.55 — 小さな島が無数に浮かぶ",
          "p = 0.5927(臨界点)— あらゆるスケールのクラスタが共存",
          "p = 0.65 — ほぼ全体がひとつのクラスタ(黄色)に"]
fig, axes = plt.subplots(3, 1, figsize=(6.4, 20.4), dpi=100)
for ax, p, t in zip(axes, ps, titles):
    img, sp, _ = render_rgb(r2, p)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(t, fontsize=15)
fig.tight_layout(pad=1.2)
fig.savefig(OUT + "fig_snapshots.png")
print("saved fig_snapshots.png")
