"""Step 2: estimate normals via local PCA (k-NN neighborhoods), naive/unoriented version.
Shows the classic sign-ambiguity problem: each normal's sign is only known up to +/-,
so neighboring normals can point in opposite directions.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_normals_pca(pts, k=15):
    """For each point, take k nearest neighbors, PCA the local neighborhood,
    and take the eigenvector of the smallest eigenvalue as the normal direction.
    Sign is arbitrary (whichever eigh() happens to return)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pts)
    _, idx = nn.kneighbors(pts)  # idx[:,0] is the point itself
    normals = np.zeros_like(pts)
    curvature = np.zeros(len(pts))
    for i in range(len(pts)):
        nbrs = pts[idx[i]]
        mean = nbrs.mean(axis=0)
        cov = (nbrs - mean).T @ (nbrs - mean) / len(nbrs)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue -> normal direction
        curvature[i] = eigvals[0] / eigvals.sum()  # crude curvature/flatness measure
    return normals, curvature, idx

if __name__ == '__main__':
    pts = np.load('bunny_points.npy')
    print('points:', len(pts))

    normals, curvature, idx = estimate_normals_pca(pts, k=15)
    print('normals shape:', normals.shape)
    print('mean curvature measure:', curvature.mean())

    # Quantify the sign-ambiguity problem: for each point, check how many of its
    # k nearest neighbors have a normal pointing in the OPPOSITE hemisphere
    # (negative dot product). A perfectly oriented surface would have ~0 flips
    # among immediate neighbors (normals vary smoothly).
    flips = 0
    total_pairs = 0
    for i in range(len(pts)):
        for j in idx[i][1:]:
            total_pairs += 1
            if np.dot(normals[i], normals[j]) < 0:
                flips += 1
    print(f'naive (unoriented) normals: {flips}/{total_pairs} neighbor pairs '
          f'({100*flips/total_pairs:.1f}%) point into opposite hemispheres')

    np.save('normals_naive.npy', normals)
    np.save('curvature.npy', curvature)
    np.save('knn_idx_k15.npy', idx)
