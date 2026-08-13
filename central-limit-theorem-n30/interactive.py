"""記事に埋め込む対話図。標本サイズ n をスライダーで動かして、
標本平均の分布が正規分布に寄っていく(あるいは寄らない)のを見るためのもの。

出力: clt-slider.html (ブログの public/interactive/ に置く)
"""

import numpy as np
import plotly.graph_objects as go

RNG = np.random.default_rng(20260813)
TRIALS = 20_000
NS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
BINS, LIM = 70, 4.5

DISTS = [
    ("一様分布", lambda r, c: RNG.uniform(0, 1, (r, c)), False),
    ("指数分布", lambda r, c: RNG.exponential(1.0, (r, c)), False),
    ("対数正規", lambda r, c: RNG.lognormal(0.0, 1.0, (r, c)), False),
    ("まれな事象 p=0.01", lambda r, c: RNG.binomial(1, 0.01, (r, c)).astype(float), False),
    ("コーシー分布", lambda r, c: RNG.standard_cauchy((r, c)), True),
]

edges = np.linspace(-LIM, LIM, BINS + 1)
centres = (edges[:-1] + edges[1:]) / 2
xs = np.linspace(-LIM, LIM, 300)
normal = np.exp(-(xs**2) / 2) / np.sqrt(2 * np.pi)


def standardised_means(sampler, n, heavy_tail):
    per_chunk = max(1, 2_000_000 // n)
    out = np.empty(TRIALS)
    done = 0
    while done < TRIALS:
        rows = min(per_chunk, TRIALS - done)
        out[done : done + rows] = sampler(rows, n).mean(axis=1)
        done += rows
    if heavy_tail:  # 分散が存在しないので四分位範囲で揃える
        q1, q3 = np.percentile(out, [25, 75])
        return (out - np.median(out)) / ((q3 - q1) / 1.3489795)
    return (out - out.mean()) / out.std(ddof=1)


# 分布 × n ぶんのヒストグラムを先に全部作る
data = {}
for name, sampler, heavy in DISTS:
    for n in NS:
        z = standardised_means(sampler, n, heavy)
        counts, _ = np.histogram(np.clip(z, -LIM, LIM), bins=edges, density=True)
        data[(name, n)] = counts
    print(name, "done", flush=True)

fig = go.Figure()
# 各分布ぶんの棒グラフを重ねて置き、表示/非表示で切り替える。
# 初期表示は最初の分布の n=1。
for di, (name, _, _) in enumerate(DISTS):
    fig.add_trace(
        go.Bar(
            x=centres,
            y=data[(name, NS[0])],
            marker_color="#ef476f",
            marker_line_width=0,
            opacity=0.8,
            name=name,
            visible=(di == 0),
            hovertemplate="z=%{x:.2f}<br>密度 %{y:.3f}<extra></extra>",
        )
    )
fig.add_trace(
    go.Scatter(
        x=xs, y=normal, mode="lines", line=dict(color="#2f6f9e", width=3), name="正規分布",
        hoverinfo="skip",
    )
)

# スライダー: n を選ぶと、表示中の分布の棒の高さを差し替える
steps = []
for n in NS:
    steps.append(
        dict(
            method="restyle",
            label=str(n),
            args=[{"y": [data[(name, n)] for name, _, _ in DISTS] + [normal]}],
        )
    )

buttons = []
for di, (name, _, _) in enumerate(DISTS):
    vis = [i == di for i in range(len(DISTS))] + [True]
    buttons.append(dict(method="restyle", label=name, args=[{"visible": vis}]))

fig.update_layout(
    sliders=[
        dict(
            active=0,
            currentvalue={"prefix": "標本サイズ n = ", "font": {"size": 17}},
            pad={"t": 44},
            steps=steps,
        )
    ],
    updatemenus=[
        dict(
            buttons=buttons,
            direction="right",
            x=0,
            xanchor="left",
            y=1.16,
            yanchor="top",
            showactive=True,
            font={"size": 13},
        )
    ],
    bargap=0.02,
    showlegend=False,
    margin=dict(l=45, r=20, t=64, b=10),
    xaxis=dict(title="標準化した標本平均", range=[-LIM, LIM]),
    yaxis=dict(title="密度", range=[0, 0.62]),
    height=540,
    plot_bgcolor="white",
)
fig.update_xaxes(gridcolor="#eee", zeroline=False)
fig.update_yaxes(gridcolor="#eee", zeroline=False)

fig.write_html("clt-slider.html", include_plotlyjs="cdn", full_html=True)
print("wrote clt-slider.html")
