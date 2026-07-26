import numpy as np
import plotly.graph_objects as go

X_real = np.load("C:/tmp/gan-experiment/X_real.npy")
heatmaps = np.load("C:/tmp/gan-experiment/d_heatmaps.npz")
grid_x = np.load("C:/tmp/gan-experiment/grid_x.npy")
grid_y = np.load("C:/tmp/gan-experiment/grid_y.npy")

snap_early, snap_late = "1", "8000"
GX, GY = np.meshgrid(grid_x, grid_y)

fig = go.Figure()

fig.add_trace(go.Surface(
    x=GX, y=GY, z=heatmaps[snap_early],
    colorscale="RdBu_r", cmin=0, cmax=1, opacity=0.9,
    name=f"iteration {snap_early}",
    showscale=True, colorbar=dict(title='D(x)<br>「本物」<br>と判定<br>する確率', x=1.0),
))

fig.add_trace(go.Scatter3d(
    x=X_real[::4, 0], y=X_real[::4, 1], z=np.zeros(len(X_real[::4])),
    mode="markers", marker=dict(size=2, color="black"),
    name="本物データ(参考、z=0に配置)",
))

fig.update_layout(
    title="学習序盤(iteration=1)の識別器が学んだ「本物らしさ」の地形。two moonsの三日月の稜線がそのまま山になっている(ドラッグで回転できます)",
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="D(x,y)(本物である確率)",
        camera=dict(eye=dict(x=1.5, y=-1.6, z=1.1)),
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    height=650,
)
fig.write_html("C:/tmp/gan-experiment/gan-discriminator-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved gan-discriminator-3d.html")
