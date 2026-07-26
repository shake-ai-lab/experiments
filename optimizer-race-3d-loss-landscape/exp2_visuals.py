import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.family"] = "Meiryo"

def f(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

names = ["SGD", "Momentum", "RMSprop", "Adam"]
colors = {"SGD": "#6b7280", "Momentum": "#2563eb", "RMSprop": "#16a34a", "Adam": "#dc2626"}
paths = {n: np.load(f"C:/tmp/optimizer-race-experiment/path_{n.lower()}.npy") for n in names}

# --- domain / surface (log(f+1) to tame the huge dynamic range) ---
xs = np.linspace(-2.0, 2.0, 220)
ys = np.linspace(-1.0, 3.0, 220)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)
logZ = np.log10(Z + 1)

# ============ Figure 1: static 3D surface with all 4 paths ============
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, logZ, cmap="Greys", alpha=0.55, linewidth=0, antialiased=True, rstride=3, cstride=3)
for n in names:
    p = paths[n]
    z = np.log10(f(p[:, 0], p[:, 1]) + 1)
    ax.plot(p[:, 0], p[:, 1], z, color=colors[n], linewidth=2.2, label=n)
    ax.scatter([p[-1, 0]], [p[-1, 1]], [z[-1]], color=colors[n], s=40)
ax.scatter([1], [1], [np.log10(f(1, 1) + 1)], color="black", marker="*", s=180, label="最小値 (1,1)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("log10(loss+1)")
ax.set_title("Rosenbrock関数の谷を、4つの最適化アルゴリズムがどう降りていくか")
ax.legend(loc="upper left", fontsize=9)
ax.view_init(elev=35, azim=-60)
plt.tight_layout()
plt.savefig("C:/tmp/optimizer-race-experiment/fig1_surface3d.png", dpi=150, bbox_inches="tight")
print("saved fig1_surface3d.png")

# ============ Figure 2: loss vs iteration (log scale) ============
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
for n in names:
    p = paths[n]
    loss = f(p[:, 0], p[:, 1])
    ax2.plot(loss, color=colors[n], label=n, linewidth=1.8)
ax2.set_yscale("log")
ax2.set_xlabel("イテレーション")
ax2.set_ylabel("損失(対数軸)")
ax2.set_title("損失の減り方の比較")
ax2.legend()
plt.tight_layout()
plt.savefig("C:/tmp/optimizer-race-experiment/fig2_loss_curve.png", dpi=150, bbox_inches="tight")
print("saved fig2_loss_curve.png")

# ============ GIF: race on a 2D contour plot ============
fig3, ax3 = plt.subplots(figsize=(5.5, 4.8))
levels = np.logspace(0, np.log10(logZ.max() * 10 + 1), 25)
cs = ax3.contourf(X, Y, Z, levels=np.percentile(Z, np.linspace(0, 100, 40)), cmap="Greys", alpha=0.75)
ax3.plot(1, 1, marker="*", color="black", markersize=16, zorder=5)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("最適化アルゴリズムのレース(等高線上)")

lines = {}
dots = {}
for n in names:
    (ln,) = ax3.plot([], [], color=colors[n], linewidth=1.8, label=n)
    (dot,) = ax3.plot([], [], marker="o", color=colors[n], markersize=8)
    lines[n] = ln
    dots[n] = dot
ax3.legend(loc="upper right", fontsize=9)
step_text = ax3.text(0.02, 0.02, "", transform=ax3.transAxes, fontsize=10)

stride = 8
max_len = max(len(p) for p in paths.values())
frame_idxs = list(range(0, max_len, stride)) + [max_len - 1] * 6

def update(frame):
    upto = frame + 1
    for n in names:
        p = paths[n]
        seg = p[:upto]
        lines[n].set_data(seg[:, 0], seg[:, 1])
        dots[n].set_data([seg[-1, 0]], [seg[-1, 1]])
    step_text.set_text(f"iteration {frame}")
    return list(lines.values()) + list(dots.values()) + [step_text]

ani = animation.FuncAnimation(fig3, update, frames=frame_idxs, interval=50, blit=False)
ani.save("C:/tmp/optimizer-race-experiment/exp2_race.gif", writer=animation.PillowWriter(fps=16), dpi=70)
print("saved exp2_race.gif")
