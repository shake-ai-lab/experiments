"""
Regenerate sweep.gif with a y-limit that fits the tallest histogram bar.

The original make_plots.py hard-coded ax.set_ylim(0, 500), but from d~50
onward the central bin holds far more than 500 of the 2000 points, so the
bar was clipped flat against the top of the axes. This script reuses the
cached distances.npz (same data, same seed) and only fixes the y-limit,
so every reported number stays identical.

Run from this directory. Reads cached data from C:/tmp/curse-of-dimensionality-experiment/
and writes the GIF directly to the blog images folder.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "C:/tmp/curse-of-dimensionality-experiment"
OUT = os.path.normpath(os.path.join(
    HERE, "..", "..", "src", "content", "blog", "images",
    "curse-of-dimensionality-experiment", "sweep.gif"))

with open(os.path.join(HERE, "results.json")) as f:
    results = json.load(f)
by_d = {r["d"]: r for r in results}

npz = np.load(os.path.join(CACHE, "distances.npz"))
DIMS_GIF = sorted(int(k) for k in npz.files)

# same frame selection as make_plots.py
gif_dims = DIMS_GIF[::2]
if gif_dims[-1] != DIMS_GIF[-1]:
    gif_dims.append(DIMS_GIF[-1])

BINS = 35
RANGE = (0.0, 2.2)

# global max bin count across all frames -> ylim that never clips
max_count = 0
for d in gif_dims:
    dist = npz[str(d)]
    normed = dist / dist.mean()
    counts, _ = np.histogram(normed, bins=BINS, range=RANGE)
    max_count = max(max_count, counts.max())
YMAX = int(np.ceil(max_count * 1.06 / 100.0) * 100)
print("max bin count:", max_count, "-> ylim:", YMAX)

fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=72)

def update(i):
    ax.clear()
    d = gif_dims[i]
    dist = npz[str(d)]
    normed = dist / dist.mean()
    ax.hist(normed, bins=BINS, range=RANGE, color="#1b6ca8", alpha=0.75)
    ax.set_xlim(*RANGE)
    ax.set_ylim(0, YMAX)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1)
    r = by_d.get(d, {}).get("ratio_max_min")
    cv = by_d.get(d, {}).get("cv")
    if r is None:
        rr = dist.max() / dist.min()
        rcv = dist.std() / dist.mean()
    else:
        rr, rcv = r, cv
    ax.set_title(f"次元数 d = {d:>4}   最大/最小比 = {rr:6.2f}   CV = {rcv:.3f}")
    ax.set_xlabel("原点からの距離 ÷ 平均距離")
    ax.set_ylabel("点の個数")
    return []

anim = animation.FuncAnimation(fig, update, frames=len(gif_dims), interval=350)
anim.save(OUT, writer=animation.PillowWriter(fps=3))
plt.close(fig)
print("saved", OUT, "frames:", len(gif_dims))

# shrink: reduce palette size via PIL (same as make_plots.py)
from PIL import Image
im = Image.open(OUT)
frames = []
try:
    while True:
        frames.append(im.copy().convert("P", palette=Image.ADAPTIVE, colors=64))
        im.seek(im.tell() + 1)
except EOFError:
    pass
frames[0].save(OUT, save_all=True, append_images=frames[1:],
               duration=350, loop=0, optimize=True)
print("recompressed", OUT)
