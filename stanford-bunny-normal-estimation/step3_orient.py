"""Step 3: fix the sign ambiguity via MST-based orientation propagation
(the classic Hoppe et al. 1992 approach): build a k-NN graph weighted by
1 - |dot(n_i, n_j)| (cheap edges = normals that already roughly agree up to sign),
take its minimum spanning tree, then BFS from a root and flip each child's
normal sign to agree with its parent.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order

def orient_normals_mst(pts, normals, idx):
    n = len(pts)
    rows, cols, weights = [], [], []
    for i in range(n):
        for j in idx[i][1:]:
            w = 1.0 - abs(np.dot(normals[i], normals[j]))
            rows.append(i)
            cols.append(j)
            weights.append(w)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    # symmetrize by taking the elementwise min so the graph is undirected
    graph = graph.minimum(graph.T)

    mst = minimum_spanning_tree(graph)
    mst = mst + mst.T  # make traversable both directions

    # root: highest point (top of an ear) tends to be an unambiguous, flat-ish region
    root = int(np.argmax(pts[:, 1]))
    order, predecessors = breadth_first_order(mst, i_start=root, directed=False)

    oriented = normals.copy()
    # force the root to point "up" (+y) as an arbitrary but sensible global convention
    if oriented[root, 1] < 0:
        oriented[root] *= -1

    visited = {root}
    for node in order:
        if node == root:
            continue
        parent = predecessors[node]
        if parent == -9999:
            continue  # disconnected component's own root, leave as-is for now
        if np.dot(oriented[parent], oriented[node]) < 0:
            oriented[node] *= -1
        visited.add(node)

    n_components_touched = len(order)
    return oriented, n_components_touched, root

if __name__ == '__main__':
    pts = np.load('bunny_points.npy')
    normals = np.load('normals_naive.npy')
    idx = np.load('knn_idx_k15.npy')

    oriented, n_visited, root = orient_normals_mst(pts, normals, idx)
    print(f'root point index: {root}, position: {pts[root]}')
    print(f'points reached by BFS from root: {n_visited}/{len(pts)}')

    # Re-measure the "flip rate" among immediate k-NN pairs, same metric as step2
    flips = 0
    total_pairs = 0
    for i in range(len(pts)):
        for j in idx[i][1:]:
            total_pairs += 1
            if np.dot(oriented[i], oriented[j]) < 0:
                flips += 1
    print(f'oriented normals: {flips}/{total_pairs} neighbor pairs '
          f'({100*flips/total_pairs:.1f}%) still point into opposite hemispheres')

    np.save('normals_oriented.npy', oriented)

    # Sanity check on a known-orientation subset: for points near the very top of
    # the back (a locally near-flat, near-horizontal patch), the y-component of a
    # correctly-outward normal should be predominantly positive.
    top_mask = pts[:, 1] > np.percentile(pts[:, 1], 97)
    print(f'top-of-back patch ({top_mask.sum()} pts): mean oriented normal y = '
          f'{oriented[top_mask, 1].mean():.3f} (naive: {normals[top_mask, 1].mean():.3f})')
