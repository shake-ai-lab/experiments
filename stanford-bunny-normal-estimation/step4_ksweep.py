"""Step 4: sweep the neighborhood size k and look at how it affects normal quality:
too small -> noisy (scan noise dominates a tiny neighborhood), too large -> over-smoothed
(loses real fine detail like the ear rim). Orients each k's normals with the same MST
propagation so the RGB coloring is comparable across panels.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order

def estimate_normals_pca(pts, k, nn_idx_cache={}):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pts)
    _, idx = nn.kneighbors(pts)
    normals = np.zeros_like(pts)
    curvature = np.zeros(len(pts))
    for i in range(len(pts)):
        nbrs = pts[idx[i]]
        mean = nbrs.mean(axis=0)
        cov = (nbrs - mean).T @ (nbrs - mean) / len(nbrs)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]
        curvature[i] = eigvals[0] / eigvals.sum()
    return normals, curvature, idx

def orient_normals_mst(pts, normals, idx):
    n = len(pts)
    rows, cols, weights = [], [], []
    for i in range(n):
        for j in idx[i][1:]:
            w = 1.0 - abs(np.dot(normals[i], normals[j]))
            rows.append(i); cols.append(j); weights.append(w)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    graph = graph.minimum(graph.T)
    mst = minimum_spanning_tree(graph)
    mst = mst + mst.T
    root = int(np.argmax(pts[:, 1]))
    order, predecessors = breadth_first_order(mst, i_start=root, directed=False)
    oriented = normals.copy()
    if oriented[root, 1] < 0:
        oriented[root] *= -1
    for node in order:
        if node == root:
            continue
        parent = predecessors[node]
        if parent == -9999:
            continue
        if np.dot(oriented[parent], oriented[node]) < 0:
            oriented[node] *= -1
    return oriented

if __name__ == '__main__':
    pts = np.load('bunny_points.npy')
    n = len(pts)

    # fixed small evaluation graph (independent of estimation k) to measure noise
    nn_fixed = NearestNeighbors(n_neighbors=7).fit(pts)
    _, eval_idx = nn_fixed.kneighbors(pts)

    ear_mask = pts[:, 1] > np.percentile(pts[:, 1], 95)  # ear-tip region (see inspect_views.py)

    ks = [3, 5, 8, 12, 15, 20, 30, 45, 60, 90, 130]
    results = {}
    roughness_list = []
    ear_curv_list = []
    for k in ks:
        normals, curvature, idx = estimate_normals_pca(pts, k)
        oriented = orient_normals_mst(pts, normals, idx)

        # roughness: mean angular disagreement among a FIXED small neighbor set
        # (measures how noisy/jittery the normal field is, independent of k)
        dots = np.abs(np.sum(oriented[eval_idx[:, 0]][:, None, :] * oriented[eval_idx[:, 1:]], axis=2))
        roughness = 1.0 - dots.mean()

        ear_curv = curvature[ear_mask].mean()

        results[k] = {'normals': oriented, 'curvature': curvature}
        roughness_list.append(roughness)
        ear_curv_list.append(ear_curv)
        print(f'k={k:4d}  roughness={roughness:.4f}  ear-region curvature={ear_curv:.4f}')

    np.savez('ksweep_results.npz',
             ks=np.array(ks),
             roughness=np.array(roughness_list),
             ear_curv=np.array(ear_curv_list),
             **{f'normals_k{k}': results[k]['normals'] for k in ks},
             **{f'curv_k{k}': results[k]['curvature'] for k in ks})
    print('saved ksweep_results.npz')
