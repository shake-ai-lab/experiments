import numpy as np
import plotly.graph_objects as go

rng = np.random.default_rng(3)
data = np.load("C:/tmp/diffusion-experiment/samples.npz")
T = 100

n_show = 180
idx = rng.choice(1500, size=n_show, replace=False)

ts = list(range(T, -1, -1))
traj = np.stack([data[str(t)][idx] for t in ts], axis=1)  # (n_show, len(ts), 2)

final_x = traj[:, -1, 0]
colors = np.where(final_x < 0, "#2563eb", "#dc2626")  # color by which moon they end up on

fig = go.Figure()
for k in range(n_show):
    fig.add_trace(go.Scatter3d(
        x=traj[k, :, 0], y=traj[k, :, 1], z=ts,
        mode="lines",
        line=dict(color=colors[k], width=2),
        opacity=0.35,
        showlegend=False,
        hoverinfo="skip",
    ))

fig.add_trace(go.Scatter3d(
    x=traj[:, 0, 0], y=traj[:, 0, 1], z=[T] * n_show,
    mode="markers", marker=dict(size=3, color="gray"),
    name=f"t={T}(開始・完全なノイズ)",
))
fig.add_trace(go.Scatter3d(
    x=traj[:, -1, 0], y=traj[:, -1, 1], z=[0] * n_show,
    mode="markers", marker=dict(size=4, color=colors),
    name="t=0(生成結果)",
))

fig.update_layout(
    title="拡散モデルの逆過程: ノイズ(上)から2つの三日月形(下)へ、各点がどう動いたか(ドラッグで回転できます)",
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="t(拡散ステップ、下ほどノイズが少ない)",
        zaxis=dict(autorange="reversed"),
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    height=680,
)
fig.write_html("C:/tmp/diffusion-experiment/diffusion-trajectories-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved diffusion-trajectories-3d.html")
