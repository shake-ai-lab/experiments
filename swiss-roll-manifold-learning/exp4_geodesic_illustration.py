import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import plotly.graph_objects as go

rng = np.random.default_rng(0)
X = np.load("C:/tmp/swiss-roll-experiment/X.npy")
t = np.load("C:/tmp/swiss-roll-experiment/t.npy")

knn_graph = kneighbors_graph(X, n_neighbors=8, mode="distance")
D_geo, predecessors = shortest_path(knn_graph, method="D", directed=False, return_predecessors=True)

# find a pair that is close in straight-line 3D distance but far apart along the surface
# (adjacent windings of the coil, NOT just close because of the independent "height" axis)
diff = X[:, None, :] - X[None, :, :]
euclid_mat = np.linalg.norm(diff, axis=2)
height_diff = np.abs(X[:, 1:2] - X[:, 1:2].T)  # X[:,1] is the roll's independent height axis
t_diff = np.abs(t[:, None] - t[None, :])       # how far apart along the winding (arc length-ish)

ratio_mat = D_geo / (euclid_mat + 1e-9)
np.fill_diagonal(ratio_mat, 0)
ratio_mat[euclid_mat < 1.0] = 0
ratio_mat[euclid_mat > 3.0] = 0
ratio_mat[height_diff > 1.0] = 0   # keep height comparable, isolate the "different winding" effect
ratio_mat[t_diff < 5.0] = 0        # require them to actually be on different windings
i, j = np.unravel_index(np.argmax(ratio_mat), ratio_mat.shape)
de, dg, ratio = euclid_mat[i, j], D_geo[i, j], ratio_mat[i, j]
print(f"picked pair i={i} j={j}  euclid_dist={de:.3f}  geodesic_dist={dg:.3f}  ratio={ratio:.2f}  dt={abs(t[i]-t[j]):.2f}")

# reconstruct the shortest path chain of points from i to j
path_idx = [j]
while path_idx[-1] != i:
    path_idx.append(predecessors[i, path_idx[-1]])
path_idx = path_idx[::-1]
path_pts = X[path_idx]
print("path length (# hops):", len(path_idx) - 1)

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=X[:, 0], y=X[:, 1], z=X[:, 2],
    mode="markers", marker=dict(size=2.5, color=t, colorscale="Turbo", opacity=0.5),
    hoverinfo="skip", showlegend=False,
))
# straight-line (wrong) distance
fig.add_trace(go.Scatter3d(
    x=[X[i, 0], X[j, 0]], y=[X[i, 1], X[j, 1]], z=[X[i, 2], X[j, 2]],
    mode="lines", line=dict(color="red", width=8, dash="dash"),
    name=f"直線距離 = {de:.2f}(誤り)",
))
# geodesic (correct) path along the surface
fig.add_trace(go.Scatter3d(
    x=path_pts[:, 0], y=path_pts[:, 1], z=path_pts[:, 2],
    mode="lines+markers", line=dict(color="limegreen", width=8),
    marker=dict(size=3, color="limegreen"),
    name=f"面に沿った測地距離 = {dg:.2f}(正しい)",
))
fig.add_trace(go.Scatter3d(
    x=[X[i, 0], X[j, 0]], y=[X[i, 1], X[j, 1]], z=[X[i, 2], X[j, 2]],
    mode="markers+text", marker=dict(size=6, color="black"),
    text=["A", "B"], textposition="top center", showlegend=False,
))

fig.update_layout(
    title=f"直線距離では近い(={de:.2f})のに、面に沿うと実際はずっと遠い(={dg:.2f})2点の例",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    legend=dict(x=0.02, y=0.98),
    margin=dict(l=0, r=0, t=60, b=0),
    height=650,
)
fig.write_html("C:/tmp/swiss-roll-experiment/geodesic-illustration-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved geodesic-illustration-3d.html")
