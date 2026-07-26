"""学習済み32ユニットMLPの確率曲面をplotlyの3Dで出力"""
import numpy as np
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier

rng = np.random.default_rng(0)
R, EYE_R = 1.0, 0.14


def yinyang_label(X):
    x, y = X[:, 0], X[:, 1]
    up = np.hypot(x, y - R / 2)
    dn = np.hypot(x, y + R / 2)
    label = np.where(up < R / 2, 1, np.where(dn < R / 2, 0, (x > 0).astype(int)))
    eye_up = np.hypot(x, y - R / 2) < EYE_R
    eye_dn = np.hypot(x, y + R / 2) < EYE_R
    return np.where(eye_up, 0, np.where(eye_dn, 1, label))


def sample(n, seed):
    r = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        cand = r.uniform(-R, R, (n * 2, 2))
        cand = cand[np.hypot(cand[:, 0], cand[:, 1]) < R]
        pts.extend(cand.tolist())
    X = np.array(pts[:n])
    return X, yinyang_label(X)


X_tr, y_tr = sample(2000, 0)
net = MLPClassifier((32,), activation="tanh", max_iter=8000, random_state=0, learning_rate_init=0.01)
net.fit(X_tr, y_tr)

g = np.linspace(-1.05, 1.05, 55)
gx, gy = np.meshgrid(g, g)
grid = np.c_[gx.ravel(), gy.ravel()]
p = net.predict_proba(grid)[:, 1].reshape(gx.shape)
mask = np.hypot(gx, gy) < R
z = np.where(mask, p, np.nan)

fig = go.Figure(go.Surface(x=gx, y=gy, z=z, colorscale="RdBu_r", cmin=0, cmax=1,
                           colorbar=dict(title="P(赤クラス)")))
fig.update_layout(
    title="学習済みニューラルネットの確率曲面。陰陽の渦が地形になり、目玉が窪み/出っ張りとして現れる (ドラッグで回転)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="P", aspectmode="cube"),
    margin=dict(l=0, r=0, t=60, b=0), height=650,
)
fig.write_html("yinyang-surface-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved yinyang-surface-3d.html")
