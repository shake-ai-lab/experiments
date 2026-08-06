import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB, GOOD = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae", "#06a77d"
snaps = np.load("snaps.npy")
REC = [0, 1, 2, 5, 10, 20, 40]


def draw(ax, pts, title, color=INK):
    v = Voronoi(pts)
    voronoi_plot_2d(v, ax=ax, show_points=False, show_vertices=False,
                    line_colors="#8d99ae", line_width=0.9, point_size=0)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.2, color=color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12.5)


# ---------------------------------------------------------------- fig1
print("fig1", flush=True)
show = [0, 2, 10, 40]
fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.4))
for ax, t in zip(axes.ravel(), show):
    i = REC.index(t)
    draw(ax, snaps[i], f"{t} 回目  （面積のばらつき {R['exp1']['cv'][t]:.3f}）",
         ACC if t == 40 else INK)
fig.suptitle("「自分の縄張りの重心へ動く」を繰り返すだけ", fontsize=14.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig1_lloyd.png", dpi=118)
plt.close(fig)

# ---------------------------------------------------------------- fig2
print("fig2", flush=True)
cv = np.array(R["exp1"]["cv"])
d0 = np.array(R["exp1"]["deg0"], dtype=float)
df = np.array(R["exp1"]["degf"], dtype=float)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
a1.plot(cv, color=ACC, lw=2.6)
a1.set_xlabel("繰り返し回数")
a1.set_ylabel("セル面積の変動係数（標準偏差÷平均）")
a1.set_title(f"ばらつきは {cv[0]:.3f} から {cv[-1]:.3f} へ", fontsize=13)
a1.grid(alpha=0.3)
x = np.arange(len(d0))
m = (x >= 3) & (x <= 11)
a2.bar(x[m] - 0.2, d0[m] / d0.sum() * 100, width=0.38, color=SUB, label="ランダムに撒いた直後")
a2.bar(x[m] + 0.2, df[m] / df.sum() * 100, width=0.38, color=ACC, label="40回動かしたあと")
a2.axvline(6, ls=":", color="#3d4753")
a2.set_xlabel("そのセルが持つ辺の数")
a2.set_ylabel("割合（%）")
a2.set_title(f"六角形の割合 {R['exp1']['hex0']*100:.1f}% → {R['exp1']['hexf']*100:.1f}%",
             fontsize=13)
a2.legend(fontsize=10)
a2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig2_stats.png", dpi=125)
plt.close(fig)

# ---------------------------------------------------------------- fig3
print("fig3", flush=True)
rows = R["exp2"]
n = [r["n"] for r in rows]
mm = [r["mean"] for r in rows]
ss = [r["sd"] for r in rows]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.errorbar(n, mm, yerr=ss, fmt="o-", color=INK, lw=2.4, ms=9, capsize=4)
ax.axhline(6.0, ls="--", color=ACC, lw=1.8)
ax.text(n[-1], 6.02, "理論値 6", ha="right", color=ACC, fontsize=11.5, fontweight="bold")
for xx, yy in zip(n, mm):
    ax.annotate(f"{yy:.3f}", (xx, yy), textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=10, color="#3d4753")
ax.set_xscale("log")
ax.set_ylim(5.7, 6.06)
ax.set_xlabel("撒いた点の数")
ax.set_ylabel("1セルあたりの平均辺数")
ax.set_title("点をいくら増やしても、平均辺数は6を超えない", fontsize=13.5)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig3_six.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig4 距離
print("fig4", flush=True)
labs = [("ユークリッド距離", "lab_2.npy"), ("マンハッタン距離", "lab_1.npy"),
        ("チェビシェフ距離", "lab_inf.npy")]
P3 = np.load("pts3.npy")
fig, axes = plt.subplots(3, 1, figsize=(6.4, 18.6))
for ax, (name, f) in zip(axes, labs):
    lab = np.load(f)
    ax.imshow(lab, cmap="tab20c", interpolation="nearest", origin="lower",
              extent=[0, 1, 0, 1])
    ax.plot(P3[:, 0], P3[:, 1], "o", ms=4.5, color="#101820")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}（境界の総画素 {R['exp3'][name]['edge_px']:,}）", fontsize=15, pad=8)
fig.suptitle("同じ60点。距離の測り方だけを変えた", fontsize=16.5)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig("fig4_metrics.png", dpi=92)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
pts = np.load("pts0.npy").copy()
g = (np.arange(400) + 0.5) / 400
gy, gx = np.meshgrid(g, g, indexing="ij")
G = np.column_stack([gx.ravel(), gy.ravel()])
frames = [pts.copy()]
for t in range(34):
    idx = cKDTree(pts).query(G)[1]
    cnt = np.bincount(idx, minlength=len(pts)).astype(float)
    sx = np.bincount(idx, weights=G[:, 0], minlength=len(pts))
    sy = np.bincount(idx, weights=G[:, 1], minlength=len(pts))
    nz = cnt > 0
    pts[nz] = np.column_stack([sx[nz] / cnt[nz], sy[nz] / cnt[nz]])
    frames.append(pts.copy())

fig, ax = plt.subplots(figsize=(4.9, 4.9))


def fr(k):
    ax.clear()
    draw(ax, frames[k], f"{k} 回目")
    return []


FuncAnimation(fig, fr, frames=len(frames), blit=False).save(
    "voronoi.gif", writer=PillowWriter(fps=5), dpi=78)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im = Image.open("voronoi.gif")
fl = [f.copy().convert("RGB").quantize(colors=24) for f in ImageSequence.Iterator(im)]
fl[0].save("voronoi.gif", save_all=True, append_images=fl[1:], loop=0, duration=200,
           optimize=True)
print(f"  voronoi.gif {os.path.getsize('voronoi.gif')/1024:.0f}KB / {len(fl)}フレーム")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

P = np.load("ptsf.npy")
v = Voronoi(P)
xs, ys = [], []
for a, b in v.ridge_vertices:
    if a >= 0 and b >= 0:
        p1, p2 = v.vertices[a], v.vertices[b]
        if max(abs(p1).max(), abs(p2).max()) < 1.4:
            xs += [p1[0], p2[0], None]
            ys += [p1[1], p2[1], None]
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#8d99ae", width=1),
                         hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers",
                         marker=dict(size=5, color="#ef476f"),
                         hovertemplate="種点 (%{x:.3f}, %{y:.3f})<extra></extra>",
                         showlegend=False))
fig.update_layout(
    title=dict(text="40回動かしたあとのボロノイ図（拡大できます）", font=dict(size=14)),
    xaxis=dict(range=[0, 1], visible=False), yaxis=dict(range=[0, 1], visible=False,
                                                        scaleanchor="x"),
    height=620, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)")
fig.write_html("voronoi-plot.html", include_plotlyjs="cdn")
print(f"  voronoi-plot.html {os.path.getsize('voronoi-plot.html')/1024:.0f}KB")

# ---------------------------------------------------------------- hero
print("hero", flush=True)
rng = np.random.default_rng(4)
H = rng.random((260, 2)) * np.array([1.5, 1.0])
gh = (np.arange(1024) + 0.5) / 1024
GY, GX = np.meshgrid(gh, gh * 1.5, indexing="ij")
lab = cKDTree(H).query(np.column_stack([GX.ravel(), GY.ravel()]))[1].reshape(1024, 1024)
lab = np.repeat(lab, 1, axis=0)
big = np.zeros((1024, 1536), dtype=np.int32)
gh2 = (np.arange(1536) + 0.5) / 1536 * 1.5
GY2, GX2 = np.meshgrid(gh, gh2, indexing="ij")
big = cKDTree(H).query(np.column_stack([GX2.ravel(), GY2.ravel()]))[1].reshape(1024, 1536)
cm = plt.get_cmap("magma")
col = (cm((big % 37) / 37.0)[:, :, :3] * 255).astype(np.uint8)
edge = np.zeros_like(big, dtype=bool)
edge[1:, :] |= big[1:, :] != big[:-1, :]
edge[:, 1:] |= big[:, 1:] != big[:, :-1]
col[edge] = (18, 22, 30)
Image.fromarray(col).save("hero.png")
print(f"  hero.png {os.path.getsize('hero.png')/1024:.0f}KB")
print("完了")
