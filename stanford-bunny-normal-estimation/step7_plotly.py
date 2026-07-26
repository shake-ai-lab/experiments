"""Step 7: interactive plotly visualizations.
1. bunny-normals-naive.html   - unoriented (sign-ambiguous) normals, RGB-colored, rotatable 3D
2. bunny-normals-oriented.html - MST-propagated oriented normals, RGB-colored, rotatable 3D
3. bunny-ksweep-slider.html   - slider over k, scrub/pause through the k-sweep
"""
import numpy as np
import plotly.graph_objects as go

pts_full = np.load('bunny_points.npy')
normals_naive_full = np.load('normals_naive.npy')
normals_oriented_full = np.load('normals_oriented.npy')

# subsample for interactive HTML file size (8171 points x float precision adds up
# fast once plotly serializes them inline; ~4000 keeps the bunny shape recognizable)
rng = np.random.default_rng(1)
sub_idx = rng.choice(len(pts_full), size=4000, replace=False)
pts = np.round(pts_full[sub_idx], 4)
normals_naive = normals_naive_full[sub_idx]
normals_oriented = normals_oriented_full[sub_idx]

def to_rgb_strings(normals):
    rgb = np.clip((normals + 1) / 2, 0, 1)
    rgb255 = (rgb * 255).astype(int)
    return [f'rgb({r},{g},{b})' for r, g, b in rgb255]

def make_scatter_html(normals, title, out_path):
    colors = to_rgb_strings(normals)
    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 2], z=pts[:, 1],
        mode='markers',
        marker=dict(size=2.2, color=colors),
        hoverinfo='skip',
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white',
    )
    fig.write_html(out_path, include_plotlyjs='cdn')
    print('saved', out_path)

make_scatter_html(normals_naive, '法線を色にすると符号がバラバラ(未整合)',
                   'bunny-normals-naive.html')
make_scatter_html(normals_oriented, '向きを揃えた法線(MST伝播で整合)',
                   'bunny-normals-oriented.html')

# ---------- k-sweep slider ----------
data = np.load('ksweep_results.npz')
ks_all = data['ks'].tolist()
ks = [k for k in ks_all if k in (3, 8, 20, 60, 130)]  # subset of the sweep, for file size

# smaller subsample just for the multi-frame slider (5 frames x N points adds up fast)
sub_idx_small = rng.choice(len(pts_full), size=2000, replace=False)
pts_small = np.round(pts_full[sub_idx_small], 4)

frames = []
init_colors = to_rgb_strings(data[f'normals_k{ks[0]}'][sub_idx_small])
fig = go.Figure(
    data=[go.Scatter3d(
        x=pts_small[:, 0], y=pts_small[:, 2], z=pts_small[:, 1],
        mode='markers',
        marker=dict(size=2.5, color=init_colors),
        hoverinfo='skip',
    )],
)

steps = []
for i, k in enumerate(ks):
    colors = to_rgb_strings(data[f'normals_k{k}'][sub_idx_small])
    frames.append(go.Frame(
        data=[go.Scatter3d(x=pts_small[:, 0], y=pts_small[:, 2], z=pts_small[:, 1],
                            mode='markers', marker=dict(size=2.5, color=colors), hoverinfo='skip')],
        name=str(k),
    ))
    steps.append(dict(
        method='animate',
        args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=f'k={k}',
    ))

fig.frames = frames
fig.update_layout(
    title='kを変えると法線がどう変わるか(スライダーで動かせます)',
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode='data',
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor='white',
    sliders=[dict(
        active=0,
        steps=steps,
        currentvalue=dict(prefix='k = '),
    )],
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        buttons=[dict(label='再生', method='animate',
                       args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]),
                 dict(label='停止', method='animate',
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])],
    )],
)
fig.write_html('bunny-ksweep-slider.html', include_plotlyjs='cdn')
print('saved bunny-ksweep-slider.html')
