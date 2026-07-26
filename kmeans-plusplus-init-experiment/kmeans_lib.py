"""From-scratch k-means (Lloyd's algorithm) + two initialization strategies."""
import numpy as np


def make_dataset(rng):
    """5 true clusters, well separated but with very unequal population:
    one big diffuse blob (200 pts) surrounded by four small tight blobs
    (12 pts each) at compass points, all far from each other relative to
    their spread.

    This is a genuine trap for Forgy-style random init: ~80% of the data
    points live in the big blob, so k=5 random initial centroids (sampled
    uniformly from the data) very often place 2+ centroids inside the big
    blob while missing one of the small blobs entirely. Because every
    blob is far from every other blob, a missed small blob gets dragged
    into whichever centroid ends up nearest -- a stable, clearly bad
    local optimum with much higher inertia than the true partition.
    """
    specs = [
        dict(center=(0.0, 0.0), n=200, std=1.2),    # big, diffuse
        dict(center=(15.0, 0.0), n=12, std=0.4),    # small, tight, east
        dict(center=(-15.0, 0.0), n=12, std=0.4),   # small, tight, west
        dict(center=(0.0, 15.0), n=12, std=0.4),    # small, tight, north
        dict(center=(0.0, -15.0), n=12, std=0.4),   # small, tight, south
    ]
    X_parts, y_parts = [], []
    for i, s in enumerate(specs):
        pts = rng.normal(loc=s["center"], scale=s["std"], size=(s["n"], 2))
        X_parts.append(pts)
        y_parts.append(np.full(s["n"], i))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def init_random(X, k, rng):
    """Forgy method: k distinct random data points as initial centroids."""
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()


def init_kmeanspp(X, k, rng):
    """k-means++ init: first centroid uniform random, each subsequent
    centroid sampled with probability proportional to squared distance
    to the nearest already-chosen centroid."""
    n = len(X)
    centroids = np.empty((k, X.shape[1]))
    first = rng.integers(n)
    centroids[0] = X[first]
    closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest_sq / closest_sq.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
        new_sq = np.sum((X - centroids[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centroids


def kmeans_lloyd(X, init_centroids, max_iter=300, tol=1e-8, record_history=False):
    """Standard Lloyd's algorithm k-means, run to convergence."""
    centroids = init_centroids.copy()
    history = [centroids.copy()] if record_history else None
    labels = None
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = centroids.copy()
        for j in range(len(centroids)):
            mask = labels == j
            if mask.any():
                new_centroids[j] = X[mask].mean(axis=0)
            # if a cluster loses all points, keep its centroid where it is
        shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
        centroids = new_centroids
        if record_history:
            history.append(centroids.copy())
        if shift < tol:
            break
    dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)
    inertia = ((X - centroids[labels]) ** 2).sum()
    if record_history:
        return centroids, labels, inertia, history
    return centroids, labels, inertia
