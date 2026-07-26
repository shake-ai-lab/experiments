import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

# subsample for a tractable pairwise-distance animation
X_full = np.load("C:/tmp/swiss-roll-experiment/X.npy")
t_full = np.load("C:/tmp/swiss-roll-experiment/t.npy")
idx = rng.choice(len(X_full), size=400, replace=False)
X = X_full[idx]
t = t_full[idx]

# --- build the k-NN graph and geodesic (shortest-path) distances, exactly what Isomap does internally ---
knn_graph = kneighbors_graph(X, n_neighbors=10, mode="distance")
D_geo = shortest_path(knn_graph, method="D", directed=False)
finite_max = D_geo[np.isfinite(D_geo)].max()
D_geo[~np.isfinite(D_geo)] = finite_max * 1.5  # patch any disconnected pairs
print("geodesic distance matrix ready, shape:", D_geo.shape, "max:", D_geo.max())

n = len(X)
pairs_i, pairs_j = np.triu_indices(n, k=1)
d_geo_pairs = D_geo[pairs_i, pairs_j]

# init embedding from PCA (a reasonable, non-cheating starting point) then optimize with Adam
pca = PCA(n_components=2, random_state=0)
Y = pca.fit_transform(X) * 0.1  # start small/compressed so the "unrolling" is visually dramatic
Y += rng.normal(0, 0.5, size=Y.shape)

def stress_and_grad(Y):
    diff = Y[pairs_i] - Y[pairs_j]
    d2d = np.linalg.norm(diff, axis=1) + 1e-9
    err = d2d - d_geo_pairs
    stress = np.sum(err ** 2)
    coef = (2 * err / d2d)[:, None] * diff
    grad = np.zeros_like(Y)
    np.add.at(grad, pairs_i, coef)
    np.add.at(grad, pairs_j, -coef)
    return stress, grad

# Adam optimizer (same implementation family as the optimizer-race post)
m = np.zeros_like(Y)
v = np.zeros_like(Y)
b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.08

n_iters = 300
snapshots = []
stresses = []
for it in range(1, n_iters + 1):
    stress, grad = stress_and_grad(Y)
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    m_hat = m / (1 - b1 ** it)
    v_hat = v / (1 - b2 ** it)
    Y = Y - lr * m_hat / (np.sqrt(v_hat) + eps)
    stresses.append(stress)
    if it % 2 == 0 or it == 1:
        snapshots.append(Y.copy())

print(f"stress: start={stresses[0]:.1f}  end={stresses[-1]:.1f}")
np.save("C:/tmp/swiss-roll-experiment/mds_final_Y.npy", Y)
np.save("C:/tmp/swiss-roll-experiment/mds_t_sub.npy", t)

# ---- GIF: watch the tangled 2D points untangle into the flattened rectangle ----
fig, ax = plt.subplots(figsize=(6, 5.4))
all_xy = np.concatenate(snapshots, axis=0)
xlim = (all_xy[:, 0].min() - 1, all_xy[:, 0].max() + 1)
ylim = (all_xy[:, 1].min() - 1, all_xy[:, 1].max() + 1)
sc = ax.scatter(snapshots[0][:, 0], snapshots[0][:, 1], c=t, cmap="turbo", s=18)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title("勾配降下法(Adam)で、測地距離を再現する2D配置を探す過程")
txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=10)

def update(frame):
    sc.set_offsets(snapshots[frame])
    txt.set_text(f"iteration {frame*2}\nstress={stresses[frame*2-1] if frame>0 else stresses[0]:.0f}")
    return sc, txt

frame_idxs = list(range(len(snapshots))) + [len(snapshots) - 1] * 8
ani = animation.FuncAnimation(fig, update, frames=frame_idxs, interval=60, blit=False)
ani.save("C:/tmp/swiss-roll-experiment/exp3_unroll.gif", writer=animation.PillowWriter(fps=16), dpi=75)
print("saved exp3_unroll.gif")
