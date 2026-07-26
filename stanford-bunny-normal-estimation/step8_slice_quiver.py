"""Step 8: draw the normals as ACTUAL ARROWS on a thin cross-section slice,
before (naive PCA) vs after (MST orientation). The rest of the post visualizes
normals only as RGB colors, which is abstract; this figure makes the sign-flip
mechanism concrete: arrows pointing randomly in/out vs consistently outward.

Uses only cached arrays (bunny_points.npy, normals_naive.npy, normals_oriented.npy),
so no experiment is re-run and no published number changes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Meiryo'
from PIL import Image

def shrink_png(path, colors=256):
    im = Image.open(path).convert('RGB')
    im = im.quantize(colors=colors, method=Image.MEDIANCUT)
    im.save(path, optimize=True)

pts = np.load('bunny_points.npy')
normals_naive = np.load('normals_naive.npy')
normals_oriented = np.load('normals_oriented.npy')

# --- slice: 4mm-thick band around the median z (cuts the side-profile outline) ---
z0 = np.median(pts[:, 2])
band = 0.002
mask = np.abs(pts[:, 2] - z0) < band
P = pts[mask]
print(f'slice: |z - {z0:.4f}| < {band}, {mask.sum()} points')

# thin the arrows greedily so they stay legible (min 4mm apart in the x-y plane)
order = np.argsort(P[:, 1])  # deterministic order
keep = []
min_d = 0.004
for i in order:
    p = P[i, :2]
    if all(np.linalg.norm(p - P[j, :2]) >= min_d for j in keep):
        keep.append(i)
keep = np.array(keep)
print(f'arrows drawn: {len(keep)}')

ALEN = 0.011  # arrow length in metres

def draw_panel(ax, normals, title, color):
    # context: full point cloud silhouette, very faint
    ax.scatter(pts[:, 0], pts[:, 1], s=2, color='0.85', linewidths=0)
    # the slice itself
    ax.scatter(P[:, 0], P[:, 1], s=10, color='0.25', linewidths=0, zorder=3)
    # arrows: in-plane component of the normal, normalized to fixed length
    v = normals[mask][keep][:, :2]
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * ALEN
    ax.quiver(P[keep, 0], P[keep, 1], v[:, 0], v[:, 1],
              angles='xy', scale_units='xy', scale=1,
              width=0.0045, color=color, zorder=4)
    ax.set_title(title, fontsize=15)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(2, 1, figsize=(7.2, 12.6))
draw_panel(axes[0], normals_naive,
           '素朴なPCA法線: 隣同士でも矢印が内外バラバラ', 'crimson')
draw_panel(axes[1], normals_oriented,
           'MST伝播後: ほぼすべての矢印が外向きに揃う', 'darkgreen')

# shared limits + equal aspect so both panels render at identical size
pad = ALEN + 0.008
x0, x1 = pts[:, 0].min() - pad, pts[:, 0].max() + pad
y0, y1 = pts[:, 1].min() - pad, pts[:, 1].max() + pad
for ax in axes:
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('4.png', dpi=100)
plt.close()
shrink_png('4.png')
print('saved 4.png')
