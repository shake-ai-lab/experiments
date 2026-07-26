import numpy as np
import plotly.graph_objects as go

def f(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

names = ["SGD", "Momentum", "RMSprop", "Adam"]
colors = {"SGD": "#6b7280", "Momentum": "#2563eb", "RMSprop": "#16a34a", "Adam": "#dc2626"}
paths = {n: np.load(f"C:/tmp/optimizer-race-experiment/path_{n.lower()}.npy") for n in names}

xs = np.linspace(-2.0, 2.0, 55)
ys = np.linspace(-1.0, 3.0, 55)
X, Y = np.meshgrid(xs, ys)
Z = np.log10(f(X, Y) + 1)

fig = go.Figure()

fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale="Greys",
    opacity=0.55,
    showscale=False,
    hoverinfo="skip",
))

for n in names:
    p = paths[n]
    # subsample for a lighter file
    p_sub = p[::4]
    z = np.log10(f(p_sub[:, 0], p_sub[:, 1]) + 1)
    fig.add_trace(go.Scatter3d(
        x=p_sub[:, 0], y=p_sub[:, 1], z=z,
        mode="lines",
        line=dict(color=colors[n], width=6),
        name=n,
    ))
    fig.add_trace(go.Scatter3d(
        x=[p[-1, 0]], y=[p[-1, 1]], z=[np.log10(f(p[-1, 0], p[-1, 1]) + 1)],
        mode="markers",
        marker=dict(color=colors[n], size=6),
        showlegend=False,
        hovertext=f"{n} 最終地点",
        hoverinfo="text",
    ))

fig.add_trace(go.Scatter3d(
    x=[1], y=[1], z=[np.log10(f(1, 1) + 1)],
    mode="markers",
    marker=dict(color="black", size=7, symbol="diamond"),
    name="最小値 (1,1)",
))

fig.update_layout(
    title="Rosenbrock関数の谷を4つの最適化アルゴリズムが降りていく様子(ドラッグで回転できます)",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="log10(loss+1)",
        camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    height=650,
)

fig.write_html("C:/tmp/optimizer-race-experiment/optimizer-race-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved optimizer-race-3d.html")
