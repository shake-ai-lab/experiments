import numpy as np
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

X_full = np.load("C:/tmp/swiss-roll-experiment/X.npy")
t_full = np.load("C:/tmp/swiss-roll-experiment/t.npy")
idx = rng.choice(len(X_full), size=800, replace=False)
X = X_full[idx]
t = t_full[idx]

iso = Isomap(n_neighbors=12, n_components=2)
emb = iso.fit_transform(X)  # (800, 2), the "correct" unrolled coordinates

# center/scale both so the morph looks natural (comparable spatial scale)
X_c = X - X.mean(axis=0)
X_c = X_c / np.abs(X_c).max()
emb_c = emb - emb.mean(axis=0)
emb_c = emb_c / np.abs(emb_c).max()

start = X_c  # original 3D swiss-roll coordinates
end = np.column_stack([emb_c[:, 0], emb_c[:, 1], np.zeros(len(emb_c))])  # isomap 2D result, laid flat at z=0

n_frames = 60

def ease(a):
    return a * a * (3 - 2 * a)  # smoothstep easing

fig = plt.figure(figsize=(5.5, 5.0))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(start[:, 0], start[:, 1], start[:, 2], c=t, cmap="turbo", s=10)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
title = ax.set_title("3次元のスイスロールを、Isomapが復元した2次元配置へと変形する")

def update(frame):
    a = ease(min(frame / (n_frames - 1), 1.0))
    pts = (1 - a) * start + a * end
    sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
    ax.view_init(elev=20 + 10 * a, azim=-60 + 50 * a)
    title.set_text(f"3次元のスイスロールを、Isomapが復元した2次元配置へと変形する ({int(a*100)}%)")
    return sc,

frame_idxs = list(range(0, n_frames, 2)) + [n_frames - 1] * 10
ani = animation.FuncAnimation(fig, update, frames=frame_idxs, interval=60, blit=False)
ani.save("C:/tmp/swiss-roll-experiment/exp3_unroll.gif", writer=animation.PillowWriter(fps=14), dpi=70)
print("saved exp3_unroll.gif")
