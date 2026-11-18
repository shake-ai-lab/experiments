"""ヒストグラムのビン幅とKDEの帯域幅を、1つのスライダーで同時に動かせるplotly版。
GIFでは一瞬で通り過ぎる「適切」ゾーンを、自分の手で止めて確認できるようにする。"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

MU1, SD1, MU2, SD2, W = -2.0, 1.0, 2.2, 1.0, 0.5
def true_pdf(g):
    return W * stats.norm.pdf(g, MU1, SD1) + (1 - W) * stats.norm.pdf(g, MU2, SD2)

x = np.load("x_sample.npy")
N = len(x)
lo, hi = x.min() - 0.3, x.max() + 0.3
g = np.linspace(lo, hi, 400)
p_true = true_pdf(g)

STEPS = 11
bws = np.geomspace(2.8, 0.14, STEPS)
factors = np.geomspace(0.85, 0.07, STEPS)

fig = make_subplots(rows=1, cols=2, subplot_titles=("ヒストグラム", "KDE"))

n_traces_per_step = 3  # hist bars, kde line, true pdf line (drawn twice, once per subplot, but count total)
for i, (bw, factor) in enumerate(zip(bws, factors)):
    bins = np.arange(lo, hi + bw, bw)
    counts, edges = np.histogram(x, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    vis = i == 0
    fig.add_trace(go.Bar(x=centers, y=counts, width=bw * 0.95, marker_color="#4c72b0",
                          name="ヒストグラム", visible=vis, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=g, y=p_true, mode="lines", line=dict(color="#e4572e", dash="dash", width=1.6),
                              name="真の密度", visible=vis, showlegend=(i == 0)), row=1, col=1)

    y = stats.gaussian_kde(x, bw_method=factor)(g)
    fig.add_trace(go.Scatter(x=g, y=y, mode="lines", line=dict(color="#3aa655", width=2.2), fill="tozeroy",
                              fillcolor="rgba(58,166,85,0.25)", name="KDE", visible=vis, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=g, y=p_true, mode="lines", line=dict(color="#e4572e", dash="dash", width=1.6),
                              name="真の密度", visible=vis, showlegend=False), row=1, col=2)

n_traces_per_step = 4
total = STEPS * n_traces_per_step

steps = []
for i, (bw, factor) in enumerate(zip(bws, factors)):
    label = "粗い" if i <= 1 else ("細かい" if i >= STEPS - 2 else ("適切" if abs(i - 5) <= 1 else ""))
    steps.append(dict(
        method="update",
        label=f"{i}",
        args=[{"visible": [j // n_traces_per_step == i for j in range(total)]}],
    ))

fig.update_layout(
    sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="粒度ステップ: "), pad=dict(t=40))],
    title=dict(text="スライダーで粗い→適切→細かいを行き来する<br>同じ220点の「見え方」がここまで変わる", font=dict(size=13)),
    height=560,
    margin=dict(l=30, r=30, t=90, b=60),
    legend=dict(orientation="h", x=1, xanchor="right", y=1, yanchor="top", bgcolor="rgba(255,255,255,0.8)"),
)
for c in [1, 2]:
    fig.update_yaxes(showticklabels=False, range=[0, 0.42], row=1, col=c)
    fig.update_xaxes(range=[lo, hi], row=1, col=c)

fig.write_html("kde-hist-slider.html", include_plotlyjs="cdn", full_html=True)
print("saved kde-hist-slider.html")
