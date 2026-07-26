"""(N, p) -> 多数決の正解率 を回転できる3D曲面として表示。
p=0.5の平面との交線が「賢くなる側」と「悪くなる側」の境界になる。"""
import numpy as np
import plotly.graph_objects as go

data = np.load("grid_results.npz")
N_VALUES, P_VALUES, acc_grid = data["N_VALUES"], data["P_VALUES"], data["acc_grid"]

# acc_grid shape: (len(N_VALUES), len(P_VALUES)) -> plotlyのSurfaceは z[y_idx, x_idx]
gx, gy = np.meshgrid(N_VALUES, P_VALUES, indexing="ij")  # gx=N, gy=p

fig = go.Figure(go.Surface(
    x=gx, y=gy, z=acc_grid,
    colorscale="RdBu", cmin=0, cmax=1,
    colorbar=dict(title="正解率"),
))

# p=0.5 の等高線(境界)を強調する黒い線を重ねる
j50 = np.argmin(np.abs(P_VALUES - 0.5))
fig.add_trace(go.Scatter3d(
    x=N_VALUES, y=np.full_like(N_VALUES, P_VALUES[j50], dtype=float), z=acc_grid[:, j50],
    mode="lines", line=dict(color="black", width=6),
    name="p=0.50(境界: ここだけ人数を増やしても変わらない)",
))

fig.update_layout(
    title="集団の正解率曲面。p=0.5を境に、賢くなる側と悪くなる側に分かれる(ドラッグで回転)",
    scene=dict(
        xaxis_title="投票者の人数 N",
        yaxis_title="1人あたりの正答確率 p",
        zaxis_title="多数決の正解率",
        aspectmode="cube",
    ),
    legend=dict(y=0.02, x=0.02),
    margin=dict(l=0, r=0, t=60, b=0),
    height=650,
)
fig.write_html("condorcet-surface-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved condorcet-surface-3d.html")

import os
print("size(KB):", round(os.path.getsize("condorcet-surface-3d.html") / 1024, 1))
