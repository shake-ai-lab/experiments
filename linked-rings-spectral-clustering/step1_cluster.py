"""鎖のように絡まった2つのリングを、k-means / GMM / スペクトラル / DBSCAN で
2クラスタに分けさせる。評価は真のリング所属とのAdjusted Rand Index (ARI)。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

R = 1.0
N = 400  # 1リングあたり


def make_rings(noise=0.05, seed=0):
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi, N)
    ph = r.uniform(0, 2 * np.pi, N)
    ringA = np.c_[R * np.cos(th), R * np.sin(th), np.zeros(N)]
    ringB = np.c_[R + R * np.cos(ph), np.zeros(N), R * np.sin(ph)]
    X = np.vstack([ringA, ringB]) + r.normal(0, noise, (2 * N, 3))
    labels = np.r_[np.zeros(N, int), np.ones(N, int)]
    return X, labels


X, y = make_rings(noise=0.05)
np.savez("rings.npz", X=X, y=y)

methods = {
    "k-means": lambda X: KMeans(2, n_init=10, random_state=0).fit_predict(X),
    "GMM": lambda X: GaussianMixture(2, covariance_type="full", random_state=0).fit_predict(X),
    "スペクトラル": lambda X: SpectralClustering(
        2, affinity="nearest_neighbors", n_neighbors=10, random_state=0
    ).fit_predict(X),
    "DBSCAN": lambda X: DBSCAN(eps=0.25, min_samples=5).fit_predict(X),
}

fig = plt.figure(figsize=(13, 3.6))
results = {}
for i, (name, fn) in enumerate(methods.items()):
    pred = fn(X)
    ari = adjusted_rand_score(y, pred)
    results[name] = ari
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=pred, cmap="coolwarm", s=6)
    ax.set_title(f"{name}  ARI={ari:.2f}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-60)
    print(f"{name}: ARI {ari:.3f}, clusters {len(set(pred))}")
fig.suptitle("絡まった2つのリングを2クラスタに分けられるか (色=予測クラスタ)", fontsize=12)
fig.tight_layout()
fig.savefig("fig2_methods.png", dpi=110, bbox_inches="tight")
print("saved fig2_methods.png")
