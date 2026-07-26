import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "."


def ackley(pos):
    x, y = pos[..., 0], pos[..., 1]
    r = np.sqrt(0.5 * (x ** 2 + y ** 2))
    term1 = -20 * np.exp(-0.2 * r)
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + np.e + 20


def grad_ackley(pos, h=1e-5):
    gx = (ackley(pos + np.array([h, 0])) - ackley(pos - np.array([h, 0]))) / (2 * h)
    gy = (ackley(pos + np.array([0, h])) - ackley(pos - np.array([0, h]))) / (2 * h)
    return np.array([gx, gy])


BOUND = 5.0


def run_gd(start, iters, lr):
    pos = start.copy()
    traj = [pos.copy()]
    for _ in range(iters):
        g = grad_ackley(pos)
        pos = pos - lr * g
        pos = np.clip(pos, -BOUND, BOUND)
        traj.append(pos.copy())
    return np.array(traj)


def run_momentum(start, iters, lr, beta):
    pos = start.copy()
    v = np.zeros(2)
    traj = [pos.copy()]
    for _ in range(iters):
        g = grad_ackley(pos)
        v = beta * v - lr * g
        pos = pos + v
        pos = np.clip(pos, -BOUND, BOUND)
        traj.append(pos.copy())
    return np.array(traj)


def run_sa(start, iters, step_size, T0, cooling, rng):
    pos = start.copy()
    f_pos = ackley(pos)
    traj = [pos.copy()]
    T = T0
    for _ in range(iters):
        candidate = pos + rng.normal(0, step_size, size=2)
        candidate = np.clip(candidate, -BOUND, BOUND)
        f_cand = ackley(candidate)
        delta = f_cand - f_pos
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-8)):
            pos, f_pos = candidate, f_cand
        T *= cooling
        traj.append(pos.copy())
    return np.array(traj)


GLOBAL_MIN = np.array([0.0, 0.0])
SUCCESS_TOL = 0.5

ITERS = 500
LR = 0.05
BETA = 0.9
SA_STEP = 0.3
SA_T0 = 5.0
SA_COOLING = 0.985

# ---------------------------------------------------------------------------
# Part 1: success rate across many random starting points
# ---------------------------------------------------------------------------

N_TRIALS = 200
rng_starts = np.random.default_rng(0)
starts = rng_starts.uniform(-BOUND, BOUND, size=(N_TRIALS, 2))

results = {"gd": [], "momentum": [], "sa": []}
finals = {"gd": [], "momentum": [], "sa": []}

for i, s in enumerate(starts):
    traj_gd = run_gd(s, ITERS, LR)
    traj_mom = run_momentum(s, ITERS, LR, BETA)
    rng_sa = np.random.default_rng(10000 + i)
    traj_sa = run_sa(s, ITERS, SA_STEP, SA_T0, SA_COOLING, rng_sa)

    finals["gd"].append(traj_gd[-1])
    finals["momentum"].append(traj_mom[-1])
    finals["sa"].append(traj_sa[-1])

    results["gd"].append(np.linalg.norm(traj_gd[-1] - GLOBAL_MIN) < SUCCESS_TOL)
    results["momentum"].append(np.linalg.norm(traj_mom[-1] - GLOBAL_MIN) < SUCCESS_TOL)
    results["sa"].append(np.linalg.norm(traj_sa[-1] - GLOBAL_MIN) < SUCCESS_TOL)

success_rate = {k: np.mean(v) for k, v in results.items()}
print("=== success rate (reach within", SUCCESS_TOL, "of global min), n_trials=", N_TRIALS, "===")
for k, v in success_rate.items():
    print(f"{k}: {v:.3f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
labels = {"gd": "通常のGD", "momentum": "Momentum", "sa": "焼きなまし法(SA)"}
colors = {"gd": "#C44E52", "momentum": "#4C72B0", "sa": "#55A868"}
xs = list(labels.keys())
ax.bar([labels[k] for k in xs], [success_rate[k] for k in xs], color=[colors[k] for k in xs])
ax.set_ylabel("大域最適解に到達できた割合")
ax.set_title(f"200箇所のランダムな開始点から、大域最適解にたどり着けた割合", fontsize=11)
ax.set_ylim(0, 1.12)
for i, k in enumerate(xs):
    ax.text(i, success_rate[k] + 0.03, f"{success_rate[k]*100:.1f}%", ha="center")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_success_rate.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part 2: momentum coefficient sweep
# ---------------------------------------------------------------------------

BETAS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
mom_success = []
for beta in BETAS:
    succ = []
    for s in starts:
        traj = run_momentum(s, ITERS, LR, beta)
        succ.append(np.linalg.norm(traj[-1] - GLOBAL_MIN) < SUCCESS_TOL)
    mom_success.append(np.mean(succ))
    print(f"beta={beta:.2f}  success_rate={mom_success[-1]:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(BETAS, mom_success, "o-", color="#4C72B0")
ax.set_xlabel("momentum係数 beta")
ax.set_ylabel("大域最適解に到達できた割合")
ax.set_title("momentumを強めるほど脱出しやすくなるとは限らない")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_momentum_sweep.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part 3: representative trajectory comparison on a contour plot
# ---------------------------------------------------------------------------

# find a start point where gd fails but sa succeeds, for a clean illustrative example
chosen_idx = None
for i, s in enumerate(starts):
    if not results["gd"][i] and results["sa"][i] and not results["momentum"][i]:
        chosen_idx = i
        break
if chosen_idx is None:
    for i, s in enumerate(starts):
        if not results["gd"][i] and results["sa"][i]:
            chosen_idx = i
            break

print("chosen illustrative start index:", chosen_idx, "start:", starts[chosen_idx])
s = starts[chosen_idx]
traj_gd = run_gd(s, ITERS, LR)
traj_mom = run_momentum(s, ITERS, LR, BETA)
rng_sa_plot = np.random.default_rng(10000 + chosen_idx)
traj_sa = run_sa(s, ITERS, SA_STEP, SA_T0, SA_COOLING, rng_sa_plot)

grid = np.linspace(-BOUND, BOUND, 300)
GX, GY = np.meshgrid(grid, grid)
GPOS = np.stack([GX, GY], axis=-1)
Z = ackley(GPOS)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
cf = ax.contourf(GX, GY, Z, levels=40, cmap="viridis")
ax.plot(traj_gd[:, 0], traj_gd[:, 1], color="#C44E52", linewidth=2, label="通常のGD")
ax.plot(traj_mom[:, 0], traj_mom[:, 1], color="#4C72B0", linewidth=2, label="Momentum")
ax.plot(traj_sa[:, 0], traj_sa[:, 1], color="#55A868", linewidth=1, alpha=0.8, label="焼きなまし法(SA)")
ax.scatter([s[0]], [s[1]], color="white", edgecolor="black", s=80, zorder=5, label="開始点")
ax.scatter([0], [0], marker="*", color="gold", edgecolor="black", s=250, zorder=5, label="大域最適解")
ax.set_title("同じ開始点からの3手法の軌跡")
ax.legend(loc="upper right", fontsize=8)
fig.colorbar(cf, ax=ax, label="Ackley関数の値")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_trajectories.png", dpi=150)
plt.close(fig)

print("gd final:", traj_gd[-1], "f=", ackley(traj_gd[-1]))
print("momentum final:", traj_mom[-1], "f=", ackley(traj_mom[-1]))
print("sa final:", traj_sa[-1], "f=", ackley(traj_sa[-1]))

# ---------------------------------------------------------------------------
# Part 4: interactive 3D plotly surface
# ---------------------------------------------------------------------------

grid3d = np.linspace(-BOUND, BOUND, 60)
GX3, GY3 = np.meshgrid(grid3d, grid3d)
GPOS3 = np.stack([GX3, GY3], axis=-1)
Z3 = ackley(GPOS3)

fig3d = go.Figure(data=[go.Surface(x=GX3, y=GY3, z=Z3, colorscale="Viridis", showscale=False)])
fig3d.add_trace(go.Scatter3d(
    x=traj_sa[:, 0], y=traj_sa[:, 1], z=ackley(traj_sa) + 0.3,
    mode="lines", line=dict(color="lime", width=4), name="SA軌跡",
))
fig3d.update_layout(
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="loss"),
    margin=dict(l=0, r=0, t=30, b=0),
    title="Ackley関数の損失地形(エッグクレート状に無数の局所最適解が並ぶ)",
)
fig3d.write_html(f"{OUT}/ackley-3d.html", include_plotlyjs="cdn")
print("saved ackley-3d.html")

# ---------------------------------------------------------------------------
# Part 5: GIF, many particles (GD vs SA) converging or getting stuck
# ---------------------------------------------------------------------------

N_PARTICLES = 40
rng_particles = np.random.default_rng(7)
particle_starts = rng_particles.uniform(-BOUND, BOUND, size=(N_PARTICLES, 2))

RECORD_EVERY = 5
gd_trajs = np.array([run_gd(s, ITERS, LR) for s in particle_starts])
sa_trajs = []
for i, s in enumerate(particle_starts):
    rng_sa_i = np.random.default_rng(20000 + i)
    sa_trajs.append(run_sa(s, ITERS, SA_STEP, SA_T0, SA_COOLING, rng_sa_i))
sa_trajs = np.array(sa_trajs)

frame_idxs = list(range(0, ITERS + 1, 10))

grid_small = np.linspace(-BOUND, BOUND, 120)
GXs, GYs = np.meshgrid(grid_small, grid_small)
GPOSs = np.stack([GXs, GYs], axis=-1)
Zs = ackley(GPOSs)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))

def animate(fi):
    idx = frame_idxs[fi]
    for ax, trajs, title in [(axes[0], gd_trajs, "通常のGD"), (axes[1], sa_trajs, "焼きなまし法(SA)")]:
        ax.clear()
        ax.contourf(GXs, GYs, Zs, levels=20, cmap="viridis")
        ax.scatter(trajs[:, idx, 0], trajs[:, idx, 1], color="white", edgecolor="black", s=18, zorder=5)
        ax.scatter([0], [0], marker="*", color="gold", edgecolor="black", s=150, zorder=6)
        ax.set_title(f"{title} (iter={idx})", fontsize=10)
        ax.set_xlim(-BOUND, BOUND)
        ax.set_ylim(-BOUND, BOUND)
    fig.tight_layout()
    return []

ani = animation.FuncAnimation(fig, animate, frames=len(frame_idxs), interval=150)
ani.save(f"{OUT}/gif1_particles.gif", writer="pillow", dpi=70)
plt.close(fig)
print("saved gif1_particles.gif")

print("done")
