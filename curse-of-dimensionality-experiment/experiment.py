"""
curse-of-dimensionality-experiment

Generate N points uniformly in [0,1]^d for a sweep of dimensions d.
For a fixed query point (origin), compute distances from the query to all N points.
Report max/min ratio and coefficient of variation (std/mean) of those distances,
as d grows. This is the classic "distance concentration" fact behind the curse
of dimensionality: in high-D, nearest and farthest neighbor distances become
nearly indistinguishable.
"""
import numpy as np
import json

N = 2000
SEED = 42
DIMS_REPORT = [2, 3, 5, 10, 20, 50, 100, 300]
# denser sweep for the GIF (log-spaced, includes the report dims)
DIMS_GIF = sorted(set(
    list(np.unique(np.round(np.logspace(np.log10(2), np.log10(300), 22)).astype(int)))
    + DIMS_REPORT
))

def distances_from_origin(d, N, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, size=(N, d))
    query = np.zeros(d)  # origin
    dist = np.linalg.norm(pts - query, axis=1)
    return dist

def stats_for(d):
    dist = distances_from_origin(d, N, SEED)
    dmax, dmin, dmean, dstd = dist.max(), dist.min(), dist.mean(), dist.std()
    ratio = dmax / dmin
    cv = dstd / dmean
    return {
        "d": int(d),
        "N": int(N),
        "max": float(dmax),
        "min": float(dmin),
        "mean": float(dmean),
        "std": float(dstd),
        "ratio_max_min": float(ratio),
        "cv": float(cv),
    }, dist

if __name__ == "__main__":
    results = []
    all_dists = {}
    for d in DIMS_GIF:
        s, dist = stats_for(d)
        results.append(s)
        all_dists[d] = dist

    # print report table
    print(f"{'d':>5} {'max':>10} {'min':>10} {'mean':>10} {'std':>10} {'ratio':>10} {'cv':>10}")
    for s in results:
        if s["d"] in DIMS_REPORT:
            print(f"{s['d']:>5} {s['max']:>10.4f} {s['min']:>10.4f} {s['mean']:>10.4f} "
                  f"{s['std']:>10.4f} {s['ratio_max_min']:>10.5f} {s['cv']:>10.5f}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez("distances.npz", **{str(d): all_dists[d] for d in DIMS_GIF})
    print("\nSaved results.json and distances.npz")
    print("DIMS_GIF:", DIMS_GIF)
