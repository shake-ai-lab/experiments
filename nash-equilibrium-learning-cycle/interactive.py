"""刻み幅を切り替えて軌道を見比べられる三角図を作る。

静止画は14周ぶんで切っているので「その先どうなるか」が見えない。ここでは
全200,000手を入れて、拡大・ホバーで自分で確かめられるようにする。
"""

import base64
import hashlib
import os
import numpy as np
import plotly.graph_objects as go

# ブログ側に置いてある plotly の実体。integrity 属性を計算するのに中身が要る。
# 環境に合わせて PLOTLY_VENDOR で差し替えられるようにしておく。
VENDOR = os.environ.get("PLOTLY_VENDOR", "public/vendor/plotly-3.7.0.min.js")
ETAS = ["0.002", "0.01", "0.05", "0.2"]
LABELS = {
    "0.002": "刻み幅 0.002（ほぼ閉じた輪）",
    "0.01": "刻み幅 0.01",
    "0.05": "刻み幅 0.05",
    "0.2": "刻み幅 0.2（角に張りつく）",
}
COLORS = {"0.002": "#06a77d", "0.01": "#2f6f9e", "0.05": "#e8a33d", "0.2": "#ef476f"}

LAPS = 20


def cut_at_laps(xs, laps=LAPS):
    """最初の何周かで切る。200,000手を全部描くと線が重なって三角形が塗りつぶされ、
    拡大もホバーも意味をなさなくなる。"""
    lead = xs.argmax(axis=1)
    switch = np.flatnonzero(np.diff(lead) != 0)
    if len(switch) < laps * 3 + 1:
        return len(xs)
    return int(switch[laps * 3]) + 1


fig = go.Figure()
for i, eta in enumerate(ETAS):
    xs = np.load(f"rps_x_eta{eta}.npy")
    step = np.arange(len(xs)) * (200_000 // (len(xs) - 1))
    end = cut_at_laps(xs)
    xs, step = xs[:end], step[:end]
    keep = slice(None, None, 2)
    fig.add_trace(
        go.Scatterternary(
            a=np.round(xs[keep, 2], 4),
            b=np.round(xs[keep, 0], 4),
            c=np.round(xs[keep, 1], 4),
            mode="lines",
            line=dict(color=COLORS[eta], width=1),
            name=LABELS[eta],
            visible=(i == 2),
            hovertemplate="%{text}手目<br>グー %{b:.3f} / パー %{c:.3f} / チョキ %{a:.3f}<extra></extra>",
            text=[f"{s:,}" for s in step[keep]],
        )
    )

fig.add_trace(
    go.Scatterternary(
        a=[1 / 3], b=[1 / 3], c=[1 / 3],
        mode="markers+text",
        marker=dict(color="#3d4753", size=16, symbol="star"),
        text=["ナッシュ均衡"],
        textposition="bottom center",
        textfont=dict(size=13, color="#3d4753"),
        name="ナッシュ均衡 (1/3, 1/3, 1/3)",
        hovertemplate="ナッシュ均衡 1/3ずつ<extra></extra>",
        showlegend=False,
    )
)

buttons = []
for i, eta in enumerate(ETAS):
    vis = [j == i for j in range(len(ETAS))] + [True]
    buttons.append(dict(label=f"η = {eta}", method="update", args=[{"visible": vis}]))

fig.update_layout(
    ternary=dict(
        sum=1,
        aaxis=dict(title="チョキ", min=0, ticks="outside", tickformat=".1f"),
        baxis=dict(title="グー", min=0, ticks="outside", tickformat=".1f"),
        caxis=dict(title="パー", min=0, ticks="outside", tickformat=".1f"),
        bgcolor="white",
    ),
    updatemenus=[
        dict(
            type="buttons", direction="right", buttons=buttons,
            x=0.5, xanchor="center", y=1.14, yanchor="top", showactive=True,
            bgcolor="#f3f5f7", bordercolor="#c0c6cc",
        )
    ],
    annotations=[
        dict(
            text="学習の刻み幅を切り替えて、最初の20周ぶんの軌道を見比べる",
            x=0.5, y=1.21, xref="paper", yref="paper",
            showarrow=False, font=dict(size=15, color="#3d4753"),
        )
    ],
    margin=dict(l=40, r=40, t=110, b=40),
    paper_bgcolor="white",
    showlegend=False,
    height=620,
)

html = fig.to_html(include_plotlyjs=False, full_html=False, config={"displayModeBar": False})

src = open(VENDOR, "rb").read()
integrity = "sha256-" + base64.b64encode(hashlib.sha256(src).digest()).decode()
page = (
    '<html>\n<head><meta charset="utf-8" />'
    '<meta name="robots" content="noindex" /></head>\n<body style="margin:0">\n'
    '<script>window.PlotlyConfig = {MathJaxConfig: "local"};</script>\n'
    f'<script charset="utf-8" src="/vendor/plotly-3.7.0.min.js" integrity="{integrity}" '
    'crossorigin="anonymous"></script>\n'
    f"{html}\n</body>\n</html>\n"
)
open("nash-simplex.html", "w", encoding="utf-8").write(page)
print("wrote nash-simplex.html", len(page), "bytes; integrity", integrity)
