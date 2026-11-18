# Resolution test: does drawing a Julia set at higher resolution fix the pixel-connectedness count?
# For each c, count 8-connected components of the filled Julia set at several raster sizes.
# English labels (DejaVu Sans). Run: python -u resolution.py

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from core import julia_escape

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

CASES = [
    ("Basilica  c = -1", complex(-1.0, 0.0), "#ef476f"),
    ("Douady rabbit  c = -0.123+0.745i", complex(-0.123, 0.745), "#e8a33d"),
    ("c = -0.5+0.5i (main cardioid)", complex(-0.5, 0.5), "#2f6f9e"),
    ("c = 0 (unit circle)", complex(0.0, 0.0), "#8d99ae"),
]
SIZES = [120, 240, 480, 960, 1920]
MAXITER = 300

res = {}
print("components (8-connected) by resolution; window [-2,2]^2, maxiter", MAXITER)
for name, c, _ in CASES:
    row = []
    for n in SIZES:
        K = julia_escape(c, -2, 2, -2, 2, n, MAXITER)
        _, k = ndimage.label(K, structure=np.ones((3, 3)))
        row.append(int(k))
    res[name] = row
    print(f"  {name:36s}", row, flush=True)

fig, ax = plt.subplots(figsize=(8.4, 5.2))
for name, c, col in CASES:
    ax.plot(SIZES, res[name], "o-", color=col, lw=2.4, ms=7, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(SIZES)
ax.set_xticklabels([str(s) for s in SIZES])
ax.set_xlabel("Raster resolution (pixels per side, log scale)")
ax.set_ylabel("Connected components counted (log scale)")
ax.set_title("Finer pixels make connected Julia sets look more broken", fontsize=13)
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig3_resolution.png", dpi=130)
plt.close(fig)
json.dump(res, open("resolution.json", "w"))
