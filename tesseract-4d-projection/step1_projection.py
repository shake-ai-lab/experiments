"""テセラクト(4次元超立方体)を4D座標で生成し、4D→3D射影で何が失われるかを測る。
- 頂点16 (±1,±1,±1,±1)、辺32、面24、胞8 を組み合わせ論で検証
- 平行投影(wを落とす)と透視投影(w方向の視点から)の両方で
  辺の長さの歪み・頂点まわりの角度の歪みを数値化する
- ついでにインタラクティブ3D(plotly, wx平面回転のアニメーション)も出力
"""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 15, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})


def hypercube(d):
    """d次元超立方体: 頂点(±1)^d と、1座標だけ異なる頂点対=辺"""
    V = np.array(list(itertools.product([-1, 1], repeat=d)), dtype=float)
    E = [(i, j) for i in range(len(V)) for j in range(i + 1, len(V))
         if np.sum(V[i] != V[j]) == 1]
    return V, E


def k_face_count(d, k):
    """d次元超立方体のk次元面の個数 = C(d,k) * 2^(d-k)"""
    from math import comb
    return comb(d, k) * 2 ** (d - k)


V4, E4 = hypercube(4)
print("=== 構成要素の検証 ===")
print(f"頂点: {len(V4)} (理論値 {k_face_count(4,0)})")
print(f"辺:   {len(E4)} (理論値 {k_face_count(4,1)})")
print(f"面:   理論値 {k_face_count(4,2)}")
print(f"胞:   理論値 {k_face_count(4,3)}")
L4 = np.array([np.linalg.norm(V4[i] - V4[j]) for i, j in E4])
print(f"4Dでの辺の長さ: すべて {L4.min():.4f} 〜 {L4.max():.4f} (一定)")


def proj_parallel(V):
    """平行投影: w座標を捨てる"""
    return V[:, :3]


def proj_perspective(V, c=3.0):
    """透視投影: w=cに置いた視点から w=0 の3D空間へ。倍率 c/(c-w)"""
    s = c / (c - V[:, 3])
    return V[:, :3] * s[:, None]


def edge_lengths(P, E):
    return np.array([np.linalg.norm(P[i] - P[j]) for i, j in E])


def vertex_angles(P, V, E):
    """各頂点に集まる辺ペアのなす角(4Dではすべて90°)。長さ0の辺は除外"""
    angles = []
    for v in range(len(V)):
        dirs = []
        for i, j in E:
            if i == v or j == v:
                vec = P[j] - P[i] if i == v else P[i] - P[j]
                n = np.linalg.norm(vec)
                if n > 1e-9:
                    dirs.append(vec / n)
        for a, b in itertools.combinations(dirs, 2):
            cosv = np.clip(np.dot(a, b), -1, 1)
            angles.append(np.degrees(np.arccos(cosv)))
    return np.array(angles)


print("\n=== 4D→3D 射影による辺の歪み ===")
results = {}
for name, P in [("平行投影", proj_parallel(V4)), ("透視投影(c=3)", proj_perspective(V4))]:
    L = edge_lengths(P, E4)
    ratio = L / 2.0  # 4Dでは全辺が長さ2
    uniq, cnt = np.unique(np.round(L, 4), return_counts=True)
    print(f"\n[{name}]")
    for u, c in zip(uniq, cnt):
        print(f"  投影後の長さ {u:.4f} (元の{u/2:.1%}) x {c}本")
    ang = vertex_angles(P, V4, E4)
    # 4Dで隣接する辺は全て直交(90°)。射影後のずれを見る
    print(f"  辺長の伸縮率: 最小 {ratio.min():.4f} 最大 {ratio.max():.4f} "
          f"(最大/最小 = {'inf' if ratio.min() < 1e-9 else f'{ratio.max()/ratio.min():.2f}'})")
    print(f"  頂点まわりの角度: 最小 {ang.min():.2f}° 最大 {ang.max():.2f}° "
          f"(4Dでは全ペア90°) / 90°からの平均ずれ {np.abs(ang-90).mean():.2f}°")
    results[name] = (P, L, ang)

# ---- 図1: 平行投影と透視投影(縦スタック) ----
fig = plt.figure(figsize=(11, 13))
w = V4[:, 3]
for row, (name, key, note) in enumerate([
    ("平行投影 (wを捨てる)", "平行投影", "w=+1とw=-1の立方体が完全に重なる"),
    ("透視投影 (w=3の視点から)", "透視投影(c=3)", "w=+1が外側の大きい立方体、w=-1が内側の小さい立方体"),
]):
    P, L, ang = results[key]
    ax = fig.add_subplot(2, 2, 2 * row + 1, projection="3d")
    for (i, j), l in zip(E4, L):
        col = "#e4572e" if V4[i, 3] != V4[j, 3] else ("#2e6ee4" if V4[i, 3] > 0 else "#7ba7f0")
        ax.plot(*zip(P[i], P[j]), color=col, lw=1.8)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.where(w > 0, "#2e6ee4", "#9bb9f2"), s=30)
    ax.set_title(f"{name}\n{note}", fontsize=13)
    ax.set_box_aspect((1, 1, 1))
    ax2 = fig.add_subplot(2, 2, 2 * row + 2)
    ax2.hist(L, bins=np.linspace(-0.05, 2.15, 45), color="#2e6ee4", edgecolor="white")
    ax2.axvline(2.0, color="#e4572e", lw=2, ls="--", label="4Dでの長さ (全辺2.0)")
    ax2.set_xlabel("投影後の辺の長さ")
    ax2.set_ylabel("辺の本数")
    ax2.set_title(f"辺長の分布 ({name})", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
fig.suptitle("同じテセラクトでも、射影の方式で「嘘のつき方」が違う", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig1_projections.png", dpi=110, bbox_inches="tight")
print("\nsaved fig1_projections.png")

# ---- インタラクティブ3D: wx平面回転 + 透視投影のアニメーション ----
STEPS = 72
thetas = np.linspace(0, 2 * np.pi, STEPS, endpoint=False)


def rot4(plane, th):
    R = np.eye(4)
    a, b = plane
    R[a, a] = R[b, b] = np.cos(th)
    R[a, b] = -np.sin(th)
    R[b, a] = np.sin(th)
    return R


AX = {"x": 0, "y": 1, "z": 2, "w": 3}


def frame_data(th):
    Vr = V4 @ rot4((AX["x"], AX["w"]), th).T
    P = proj_perspective(Vr)
    ex, ey, ez = [], [], []
    for i, j in E4:
        ex += [P[i, 0], P[j, 0], None]
        ey += [P[i, 1], P[j, 1], None]
        ez += [P[i, 2], P[j, 2], None]
    return P, Vr[:, 3], ex, ey, ez


P0, w0, ex0, ey0, ez0 = frame_data(0.0)
edge_trace = go.Scatter3d(x=ex0, y=ey0, z=ez0, mode="lines",
                          line=dict(color="#888", width=4), hoverinfo="skip")
vert_trace = go.Scatter3d(
    x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], mode="markers",
    marker=dict(size=6, color=w0, colorscale="RdBu", cmin=-1.45, cmax=1.45,
                colorbar=dict(title="第4の座標 w")),
    hoverinfo="skip")

frames = []
for k, th in enumerate(thetas):
    P, ww, ex, ey, ez = frame_data(th)
    frames.append(go.Frame(
        name=f"{k}",
        data=[go.Scatter3d(x=ex, y=ey, z=ez),
              go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2],
                           marker=dict(size=6, color=ww, colorscale="RdBu",
                                       cmin=-1.45, cmax=1.45))]))

R = 1.6
fig3d = go.Figure(data=[edge_trace, vert_trace], frames=frames)
fig3d.update_layout(
    title="テセラクト(4次元超立方体)の透視投影。再生で xw 平面の4次元回転——3次元の回転では絶対に起きない「内側の立方体が外側と入れ替わる」動きが見える。ドラッグで3D視点も回せる",
    scene=dict(
        xaxis=dict(range=[-R, R], title="x"),
        yaxis=dict(range=[-R, R], title="y"),
        zaxis=dict(range=[-R, R], title="z"),
        aspectmode="cube"),
    margin=dict(l=0, r=0, t=80, b=0), height=620,
    updatemenus=[dict(
        type="buttons", showactive=False, y=0, x=0, xanchor="left", yanchor="top",
        buttons=[
            dict(label="▶ 4次元回転を再生", method="animate",
                 args=[None, dict(frame=dict(duration=60, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label="■ 停止", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")]),
        ])],
    sliders=[dict(
        active=0, y=0, x=0.28, len=0.7,
        currentvalue=dict(prefix="回転角: ", suffix="°"),
        steps=[dict(method="animate", label=f"{int(np.degrees(th))}",
                    args=[[f"{k}"], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))])
               for k, th in enumerate(thetas)])],
)
fig3d.write_html("tesseract-4d-projection.html", include_plotlyjs="cdn", full_html=True)
print("saved tesseract-4d-projection.html")
