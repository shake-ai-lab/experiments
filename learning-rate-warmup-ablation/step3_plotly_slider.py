"""LRスイープのインタラクティブ版。スライダーでLRを動かすと、代表シード(seed=0)の
損失曲線がどう変わるかを見られる。lr=0.012あたりまでは滑らかに下がるが、
0.015を過ぎたあたりから序盤にスパイクが混じり始め、0.03以上では暴れっぱなしになる様子を
自分の手で動かして確認できるようにした。
"""
import numpy as np
import plotly.graph_objects as go

sweep = np.load("C:/tmp/learning-rate-warmup-ablation-experiment/sweep_results.npz")
lrs = sweep["lrs"]

fig = go.Figure()
for i, lr in enumerate(lrs):
    curve = sweep[f"repcurve_lr{lr}"]
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(curve)),
            y=curve,
            mode="lines",
            visible=(i == 0),
            line=dict(color="#dc2626" if lr >= 0.018 else "#2563eb", width=1.6),
            hovertemplate="step %{x}<br>loss %{y:.3f}<extra></extra>",
        )
    )

steps = []
for i, lr in enumerate(lrs):
    tag = "(本実験で採用)" if abs(lr - 0.018) < 1e-9 else ""
    steps.append(
        dict(
            method="update",
            label=f"{lr:g}",
            args=[
                {"visible": [j == i for j in range(len(lrs))]},
                {"title": f"lr = {lr:g} {tag} での損失推移(1シード、warmupなし)"},
            ],
        )
    )

fig.update_layout(
    sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="学習率 = "))],
    title="lr = 0.001 での損失推移(1シード、warmupなし)",
    xaxis=dict(title="ステップ"),
    yaxis=dict(title="損失(対数軸)", type="log", range=[-2.3, 1.7]),
    margin=dict(l=55, r=20, t=70, b=40),
    height=560,
)
fig.write_html(
    "C:/tmp/learning-rate-warmup-ablation-experiment/lr-warmup-sweep.html",
    include_plotlyjs="cdn",
    full_html=True,
)
print("saved lr-warmup-sweep.html")
