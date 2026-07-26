import numpy as np
from sklearn.datasets import make_swiss_roll
import plotly.graph_objects as go

X, t = make_swiss_roll(n_samples=1500, noise=0.05, random_state=0)
np.save("C:/tmp/swiss-roll-experiment/X.npy", X)
np.save("C:/tmp/swiss-roll-experiment/t.npy", t)
print("X shape:", X.shape, "t range:", t.min(), t.max())

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=X[:, 0], y=X[:, 1], z=X[:, 2],
    mode="markers",
    marker=dict(size=3.5, color=t, colorscale="Turbo", colorbar=dict(title="巻きの位置<br>(t)")),
    hoverinfo="skip",
))
fig.update_layout(
    title="スイスロール(3次元データ)。色は「本当は1次元に近い、渦の中での位置」を表す(ドラッグで回転できます)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    margin=dict(l=0, r=0, t=60, b=0),
    height=650,
)
fig.write_html("C:/tmp/swiss-roll-experiment/swiss-roll-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved swiss-roll-3d.html")
