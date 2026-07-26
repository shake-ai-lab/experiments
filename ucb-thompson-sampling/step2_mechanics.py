"""(1) Thompsonの事後分布が試行とともに尖っていくGIF
(2) UCBの不確実性ボーナスが縮んでいく様子のスナップショット図"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams["font.family"] = "Meiryo"
K = 10
STEPS = 2000
rng = np.random.default_rng(7)
q_true = rng.normal(0, 1, K)
opt_arm = int(np.argmax(q_true))
print("q_true:", np.round(q_true, 2), "best:", opt_arm)

# Thompsonを1回走らせ、途中の事後分布を記録
q_est = np.zeros(K)
n = np.zeros(K)
snap_at = sorted(
    set(list(range(1, 51, 2)) + list(range(50, 300, 10)) + list(range(300, STEPS + 1, 50)))
)
snaps = []
for t in range(1, STEPS + 1):
    post_var = 1.0 / (1.0 + n)
    post_mean = (n * q_est) / (1.0 + n)
    a = int(np.argmax(rng.normal(post_mean, np.sqrt(post_var))))
    r = rng.normal(q_true[a], 1)
    n[a] += 1
    q_est[a] += (r - q_est[a]) / n[a]
    if t in snap_at:
        pv = 1.0 / (1.0 + n)
        pm = (n * q_est) / (1.0 + n)
        snaps.append((t, pm.copy(), np.sqrt(pv), n.copy()))

xs = np.linspace(-2.5, 2.5, 300)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
cmap = plt.get_cmap("tab10")


def update(fi):
    t, pm, ps, cnt = snaps[fi]
    ax.clear()
    for k in range(K):
        col = "#e4572e" if k == opt_arm else "#bbb"
        lw = 2.2 if k == opt_arm else 1.0
        ax.plot(xs, norm.pdf(xs, pm[k], ps[k]), color=col, lw=lw)
        ax.axvline(q_true[k], color=col, ls=":", lw=0.7, alpha=0.5)
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("腕の期待報酬の信念")
    ax.set_ylabel("確率密度")
    ax.set_title(
        f"Thompson samplingの信念 (step {t})  最適腕(赤)を引いた回数 {int(cnt[opt_arm])}/{t}",
        fontsize=11,
    )
    return []


ani = animation.FuncAnimation(fig, update, frames=len(snaps), interval=120)
ani.save("thompson_beliefs.gif", writer="pillow", fps=8, dpi=80)
print("saved thompson_beliefs.gif")
plt.close(fig)

# ---- UCBのスナップショット ----
q_est = np.zeros(K)
n = np.zeros(K)
c = 2.0
shots = {}
for t in range(STEPS):
    if t < K:
        a = t
    else:
        a = int(np.argmax(q_est + c * np.sqrt(np.log(t + 1) / n)))
    r = rng.normal(q_true[a], 1)
    n[a] += 1
    q_est[a] += (r - q_est[a]) / n[a]
    if t + 1 in (20, 200, 2000):
        shots[t + 1] = (q_est.copy(), c * np.sqrt(np.log(t + 2) / n), n.copy())

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (t, (qe, bonus, cnt)) in zip(axes, shots.items()):
    colors = ["#e4572e" if k == opt_arm else "#9ab" for k in range(K)]
    ax.bar(range(K), qe, color=colors, alpha=0.85)
    ax.errorbar(range(K), qe, yerr=[np.zeros(K), bonus], fmt="none", ecolor="#333", capsize=3, lw=1.2)
    ax.scatter(range(K), q_true, color="k", marker="_", s=180, label="真の期待報酬")
    ax.set_title(f"step {t}", fontsize=11)
    ax.set_xlabel("腕")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"{int(x)}" for x in cnt], fontsize=8)
    ax.set_ylim(-2.2, 3.2)
    if t == 20:
        ax.set_ylabel("推定値 + 不確実性ボーナス")
        ax.legend(fontsize=9)
fig.suptitle("UCB: 縦棒=推定値、ヒゲ=不確実性ボーナス (x軸ラベル=その腕を引いた回数)。引かない腕のヒゲは伸び、引いた腕は縮む", fontsize=11)
fig.tight_layout()
fig.savefig("fig2_ucb_bounds.png", dpi=110, bbox_inches="tight")
print("saved fig2_ucb_bounds.png")
