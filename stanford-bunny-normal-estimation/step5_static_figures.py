"""Step 5: static annotated PNG figures for the blog post."""
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
idx = np.load('knn_idx_k15.npy')

def to_rgb(normals):
    return np.clip((normals + 1) / 2, 0, 1)

# ---------- Figure 1: naive (sign-ambiguous) vs oriented, X-Y projection ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))

axes[0].scatter(pts[:, 0], pts[:, 1], c=to_rgb(normals_naive), s=4)
axes[0].set_title('PCA法線をそのまま使うと符号がバラバラ\n(隣接点なのに色が反転)', fontsize=14)
axes[0].set_aspect('equal')
axes[0].set_xticks([]); axes[0].set_yticks([])
# annotate a couple of visibly speckled regions
for (x, y, dx, dy, tx, ty) in [(-0.02, 0.10, 0.025, 0.02, 0.01, 0.135),
                                (0.01, 0.06, 0.02, 0.015, 0.045, 0.085)]:
    axes[0].annotate('色が入り乱れる', xy=(x, y), xytext=(tx, ty),
                      fontsize=11, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

axes[1].scatter(pts[:, 0], pts[:, 1], c=to_rgb(normals_oriented), s=4)
axes[1].set_title('MST伝播で向きを揃えると滑らかに\n(37.1%→0.6%の反転率)', fontsize=14)
axes[1].set_aspect('equal')
axes[1].set_xticks([]); axes[1].set_yticks([])
axes[1].annotate('同じ面は同じ色に', xy=(-0.02, 0.12), xytext=(0.015, 0.155),
                  fontsize=11, color='darkgreen',
                  arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

plt.tight_layout()
plt.savefig('1.png', dpi=110)
plt.close()
shrink_png('1.png')
print('saved 1.png')

# ---------- Figure 2: k-sweep small multiples (k=5, 20, 130) ----------
data = np.load('ksweep_results.npz')
show_ks = [5, 20, 130]
labels = ['k=5: 小さすぎる\n(ノイズが目立つ)', 'k=20: ちょうどよい\n(耳の輪郭が残る)', 'k=130: 大きすぎる\n(耳が塗りつぶされる)']

fig, axes = plt.subplots(1, 3, figsize=(13.5, 6))
for ax, k, label in zip(axes, show_ks, labels):
    normals_k = data[f'normals_k{k}']
    ax.scatter(pts[:, 0], pts[:, 1], c=to_rgb(normals_k), s=4)
    ax.set_title(label, fontsize=14)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

# annotate the ear region on all three panels
ear_xy = (-0.045, 0.16)
label_bbox = dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.85)
axes[0].annotate('耳がざらつく', xy=ear_xy, xytext=(-0.11, 0.075), fontsize=11, color='red',
                  bbox=label_bbox, arrowprops=dict(arrowstyle='->', color='red', lw=1.5,
                  shrinkA=8, shrinkB=2))
axes[1].annotate('耳の稜線がくっきり', xy=ear_xy, xytext=(-0.11, 0.075), fontsize=11, color='darkgreen',
                  bbox=label_bbox, arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5,
                  shrinkA=8, shrinkB=2))
axes[2].annotate('耳が本体に溶ける', xy=ear_xy, xytext=(-0.11, 0.075), fontsize=11, color='red',
                  bbox=label_bbox, arrowprops=dict(arrowstyle='->', color='red', lw=1.5,
                  shrinkA=8, shrinkB=2))

plt.tight_layout()
plt.savefig('2.png', dpi=110)
plt.close()
shrink_png('2.png')
print('saved 2.png')

# ---------- Figure 3: quantitative sweet-spot plot ----------
ks = data['ks']
roughness = data['roughness']
ear_curv = data['ear_curv']

fig, ax1 = plt.subplots(figsize=(9, 5.5))
color1 = 'tab:blue'
ax1.set_xlabel('k (近傍点数)', fontsize=12)
ax1.set_ylabel('法線のガタつき (roughness, 小さいほど滑らか)', color=color1, fontsize=12)
ax1.plot(ks, roughness, 'o-', color=color1, lw=2.5, markersize=7)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('耳の稜線の曲率推定値 (大きいほど輪郭が残る)', color=color2, fontsize=12)
ax2.plot(ks, ear_curv, 's-', color=color2, lw=2.5, markersize=7)
ax2.tick_params(axis='y', labelcolor=color2)

peak_k = ks[np.argmax(ear_curv)]
peak_v = ear_curv.max()
ax2.annotate(f'k={peak_k}あたりが山\n(これ以上大きくすると耳が消える)',
             xy=(peak_k, peak_v), xytext=(peak_k * 0.35, peak_v * 1.28),
             fontsize=11, color=color2, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9),
             arrowprops=dict(arrowstyle='->', color=color2, lw=1.5, shrinkB=8))
ax2.set_ylim(top=peak_v * 1.55)

plt.title('小さいkはガタつき、大きすぎるkは耳の輪郭を消す', fontsize=14)
fig.tight_layout()
plt.savefig('3.png', dpi=110)
plt.close()
shrink_png('3.png')
print('saved 3.png')
