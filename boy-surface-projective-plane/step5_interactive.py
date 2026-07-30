"""ブログ埋め込み用のインタラクティブ3D (plotly)。
円板パート/メビウス帯パートの色分け + 継ぎ目 + 自己交差線 + 三重点。
凡例クリックでパーツごとに表示を切り替えられる。"""
import numpy as np
import plotly.graph_objects as go
from boy_common import boy, sample_disk

RHO = 0.65
W = sample_disk(30000, seed=3)
X = boy(W)
band = np.abs(W) >= RHO

d = np.load("self_intersection.npz")
mid, center = d["mid"], d["center"]
# 自己交差線は間引いて軽くする
sel = np.random.default_rng(0).choice(len(mid), size=min(4000, len(mid)), replace=False)
th_c = np.linspace(0, 2 * np.pi, 400)
seam = boy(RHO * np.exp(1j * th_c))

fig = go.Figure([
    go.Scatter3d(
        x=X[~band, 0], y=X[~band, 1], z=X[~band, 2], mode="markers",
        marker=dict(size=2.2, color="#e4572e", opacity=0.55),
        name="円板パート", hoverinfo="skip"),
    go.Scatter3d(
        x=X[band, 0], y=X[band, 1], z=X[band, 2], mode="markers",
        marker=dict(size=2.2, color="#2e6ee4", opacity=0.3),
        name="メビウスの帯パート", hoverinfo="skip"),
    go.Scatter3d(
        x=seam[:, 0], y=seam[:, 1], z=seam[:, 2], mode="lines",
        line=dict(color="#111111", width=7),
        name="貼り合わせの継ぎ目", hoverinfo="skip"),
    go.Scatter3d(
        x=mid[sel, 0], y=mid[sel, 1], z=mid[sel, 2], mode="markers",
        marker=dict(size=2.6, color="#0e8a5f", opacity=0.9),
        name="自己交差線 (三葉のプロペラ)", hoverinfo="skip"),
    go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]], mode="markers",
        marker=dict(size=9, color="#111111", symbol="diamond"),
        name="三重点 (3枚が重なる点)", hoverinfo="skip"),
])
fig.update_layout(
    title=dict(
        text="ボーイ曲面 = 実射影平面を3次元に押し込んだ姿 (ドラッグで回転、凡例クリックで表示切替)",
        font=dict(size=14)),
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    legend=dict(itemsizing="constant", font=dict(size=13), y=0.9),
    margin=dict(l=0, r=0, t=60, b=0), height=650,
)
fig.write_html("boy-surface-projective-plane.html", include_plotlyjs="cdn", full_html=True)
print("saved boy-surface-projective-plane.html")
