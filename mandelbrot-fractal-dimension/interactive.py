# 記事に埋め込む回転できる3D。
# 脱出時間を高さにすると、マンデルブロ集合の境界が「絶壁」として見える。
# 集合の内側は最大値で平らな台地、外側は谷。境界はその間の崖。

import numpy as np
import plotly.graph_objects as go

from common import escape_time

# シーホースバレー(海馬の谷)。境界のひだが最も分かりやすい場所のひとつ
CX, CY, W = -0.745, 0.113, 0.022
N = 170
MAXITER = 1500

esc = escape_time(CX - W, CX + W, CY - W, CY + W, N, N, MAXITER)
z = esc.astype(float)
z[esc == 0] = MAXITER  # 集合内は最大値の台地にする
z = np.log10(z + 1)

x = np.linspace(CX - W, CX + W, N)
y = np.linspace(CY - W, CY + W, N)

fig = go.Figure(
    go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#0b1622"], [0.35, "#1b3a5c"], [0.6, "#2f6f9e"],
                    [0.82, "#7fb2d4"], [1, "#ffffff"]],
        showscale=False,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.9),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=1)),
    )
)
fig.update_layout(
    title=dict(text="脱出までの反復回数を高さにしたマンデルブロ集合の境界（シーホースバレー）",
               font=dict(size=14)),
    scene=dict(
        xaxis_title="Re(c)", yaxis_title="Im(c)", zaxis_title="log10(脱出までの反復回数)",
        camera=dict(eye=dict(x=1.45, y=-1.45, z=0.95)),
        aspectratio=dict(x=1, y=1, z=0.55),
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=560,
    paper_bgcolor="rgba(0,0,0,0)",
)
fig.write_html("mandelbrot-3d.html", include_plotlyjs="cdn")
print("保存: mandelbrot-3d.html")

import os

print(f"ファイルサイズ: {os.path.getsize('mandelbrot-3d.html') / 1024:.0f} KB")
