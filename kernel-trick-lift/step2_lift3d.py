"""主役ビジュアル: (1) ドーナツが3Dに持ち上がり平面で切られる回せるplotly
(2) 持ち上げアニメーション mp4+GIF"""
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

d = np.load("data.npz")
Xc, yc = d["Xc"], d["yc"]
res = np.load("res.npz")
PLANE_Z = float(res["plane_z"])
z_lift = (Xc**2).sum(1)

# ---- plotly: 持ち上げ + 分離平面 ----
fig = go.Figure()
for lbl, name, col in [(0, "外側の輪", "#2e6ee4"), (1, "内側の輪", "#e4572e")]:
    m = yc == lbl
    fig.add_trace(
        go.Scatter3d(
            x=Xc[m, 0], y=Xc[m, 1], z=z_lift[m], mode="markers",
            marker=dict(size=3.5, color=col), name=name, hoverinfo="skip",
        )
    )
g = np.linspace(-1.4, 1.4, 12)
gx, gy = np.meshgrid(g, g)
fig.add_trace(
    go.Surface(
        x=gx, y=gy, z=np.full_like(gx, PLANE_Z), opacity=0.35,
        colorscale=[[0, "#888"], [1, "#888"]], showscale=False, name="分離平面",
    )
)
fig.update_layout(
    title=f"z = x² + y² で持ち上げると、平面1枚 (z={PLANE_Z:.2f}) でスパッと切れる (ドラッグで回転)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z = x²+y²", aspectmode="cube"),
    margin=dict(l=0, r=0, t=60, b=0), height=650,
    legend=dict(x=0.02, y=0.98),
)
fig.write_html("kernel-lift-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved kernel-lift-3d.html")

# ---- 持ち上げアニメ (mp4 + GIF): z を 0 -> x^2+y^2 に補間、回転視点 ----
fig = plt.figure(figsize=(6, 5.6))
ax = fig.add_subplot(111, projection="3d")
FR = 50
colors = np.where(yc == 0, "#2e6ee4", "#e4572e")


def update(fi):
    ax.clear()
    a = min(1.0, fi / (FR * 0.6))  # 前半60%で持ち上げ完了
    z = a * z_lift
    ax.scatter(Xc[:, 0], Xc[:, 1], z, c=colors, s=7)
    if a >= 1.0:
        ax.plot_surface(gx, gy, np.full_like(gx, PLANE_Z), alpha=0.3, color="#888")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(0, 1.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=12 + 8 * np.sin(fi / FR * np.pi), azim=-60 + fi * 1.6)
    ax.set_title(
        "2次元では絡まった輪が…" if a < 0.35 else
        ("3次元へ持ち上がっていく" if a < 1.0 else "平面1枚で一刀両断できる"),
        fontsize=14,
    )
    return []


ani = animation.FuncAnimation(fig, update, frames=FR + 25, interval=100)
ani.save("lift.mp4", writer=animation.FFMpegWriter(fps=12, bitrate=1800), dpi=100)
ani.save("lift.gif", writer="pillow", fps=9, dpi=72)
print("saved lift.mp4 / lift.gif")
