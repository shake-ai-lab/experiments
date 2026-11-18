import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from experiment import intervals, coverage, normal, lognorm, MU, LOGN_MU

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
OUT = r"C:\Users\kouta\project\forsubwork\src\content\blog\images\confidence-interval-95-meaning"
os.makedirs(OUT, exist_ok=True)
BLUE, RED, GRAY = "#2b6cb0", "#d9381e", "#888888"

# ---- 1) GIF: 100本の区間を1本ずつ
rng = np.random.default_rng(7)
lo, hi = intervals(rng.normal(MU, 10.0, (100, 10)), 0.95)
miss = (MU < lo) | (MU > hi)
fig, ax = plt.subplots(figsize=(7, 8.5), dpi=70)


def draw(k):
    ax.clear()
    ax.axvline(MU, color="black", lw=1.5)
    for i in range(k):
        c = RED if miss[i] else BLUE
        ax.plot([lo[i], hi[i]], [i, i], color=c, lw=2)
    ax.set_ylim(100, -2)
    ax.set_xlim(20, 80)
    ax.set_yticks([])
    ax.set_xlabel("値（黒線が母集団の真の平均 50）")
    ax.set_title(f"95%信頼区間（n=10）を{k}本作った：真の値を外したのは {int(miss[:k].sum())} 本（赤）", fontsize=12)


frames = list(range(0, 101)) + [100] * 25
ani = FuncAnimation(fig, lambda f: draw(f), frames=frames, interval=80)
ani.save(os.path.join(OUT, "intervals.gif"), writer=PillowWriter(fps=12))
plt.close(fig)
print("miss", int(miss.sum()))

# ---- 2) z vs t (正規)
ns = [3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
pz = [coverage(normal, n, 0.95, False, MU, seed=1)[0] for n in ns]
pt = [coverage(normal, n, 0.95, True, MU, seed=2)[0] for n in ns]
fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
ax.axhline(0.95, color=GRAY, ls="--")
ax.plot(ns, pz, "o-", color=RED, label="z（1.96）を使った区間")
ax.plot(ns, pt, "s-", color=BLUE, label="t 分布を使った区間")
for n, v in zip(ns, pz):
    if n in (3, 5, 10, 30):
        ax.annotate(f"{v:.3f}", (n, v), textcoords="offset points", xytext=(0, -16), ha="center", color=RED)
ax.set_xscale("log")
ax.set_xticks([3, 5, 10, 20, 50, 100])
ax.set_xticklabels([3, 5, 10, 20, 50, 100])
ax.set_ylim(0.78, 0.98)
ax.set_xlabel("標本サイズ n（正規分布の母集団）")
ax.set_ylabel("真の平均を含んだ割合（10万回）")
ax.set_title("少人数で 1.96 を使うと「95%」は名ばかりになる")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "z_vs_t.png"))
plt.close(fig)

# ---- 3) 歪んだ母集団
ns2 = [5, 10, 30, 100, 300, 1000, 3000]
res = []
for n in ns2:
    reps = 100_000 if n <= 1000 else 20_000
    res.append(np.mean([coverage(lognorm, n, 0.95, True, LOGN_MU, reps=reps, seed=s)[0] for s in (11, 12, 13)]))
fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
ax.axhline(0.95, color=GRAY, ls="--")
ax.plot(ns2, res, "o-", color=RED, label="t 区間（強く歪んだ母集団）")
for n, v in zip(ns2, res):
    ax.annotate(f"{v:.3f}", (n, v), textcoords="offset points", xytext=(0, -16), ha="center", color=RED)
ax.set_xscale("log")
ax.set_xticks(ns2)
ax.set_xticklabels(ns2)
ax.set_ylim(0.6, 1.0)
ax.set_xlabel("標本サイズ n（対数正規分布、σ=1.5）")
ax.set_ylabel("真の平均を含んだ割合")
ax.set_title("n=30 でも 79%。95% に届くには n が数千要る")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "skewed.png"))
plt.close(fig)
print("done")
