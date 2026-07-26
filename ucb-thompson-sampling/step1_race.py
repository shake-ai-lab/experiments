"""ε-greedy vs UCB vs Thompson sampling の3つ巴。
前回のε-greedy記事と同じ10本腕ガウスバンディット (q*~N(0,1), 報酬~N(q*,1))。
後悔(regret)の伸び方と最適腕選択率を比較する。"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs(".", exist_ok=True)
plt.rcParams["font.family"] = "Meiryo"

K = 10
STEPS = 2000
RUNS = 1000


def run_one(algo, seed, eps=0.1, c=2.0):
    rng = np.random.default_rng(seed)
    q_true = rng.normal(0, 1, K)
    best = q_true.max()
    q_est = np.zeros(K)
    n = np.zeros(K)
    # Thompson用: 事前 N(0,1)、既知分散1 -> 事後 N(mu, 1/(1+n))
    regret = np.zeros(STEPS)
    optimal = np.zeros(STEPS)
    opt_arm = np.argmax(q_true)
    for t in range(STEPS):
        if algo == "eps":
            if rng.random() < eps:
                a = rng.integers(K)
            else:
                a = np.argmax(q_est)
        elif algo == "ucb":
            if t < K:
                a = t  # 最初に全腕を1回ずつ
            else:
                a = np.argmax(q_est + c * np.sqrt(np.log(t + 1) / n))
        else:  # thompson
            post_var = 1.0 / (1.0 + n)
            post_mean = (n * q_est) / (1.0 + n)
            a = np.argmax(rng.normal(post_mean, np.sqrt(post_var)))
        r = rng.normal(q_true[a], 1)
        n[a] += 1
        q_est[a] += (r - q_est[a]) / n[a]
        regret[t] = best - q_true[a]
        optimal[t] = a == opt_arm
    return regret, optimal


algos = {"ε-greedy (ε=0.1)": ("eps", "#999"), "UCB (c=2)": ("ucb", "#2e6ee4"), "Thompson": ("thompson", "#e4572e")}
results = {}
for name, (algo, color) in algos.items():
    reg = np.zeros(STEPS)
    opt = np.zeros(STEPS)
    for s in range(RUNS):
        r, o = run_one(algo, seed=s)
        reg += r
        opt += o
    results[name] = dict(cum_regret=np.cumsum(reg / RUNS), opt=opt / RUNS, color=color)
    print(f"{name}: 累積後悔@{STEPS} {results[name]['cum_regret'][-1]:.1f}, 最適腕率(終盤) {results[name]['opt'][-100:].mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
for name, res in results.items():
    axes[0].plot(res["cum_regret"], label=name, color=res["color"])
    axes[1].plot(np.convolve(res["opt"], np.ones(20) / 20, mode="valid"), label=name, color=res["color"])
axes[0].set_xlabel("ステップ")
axes[0].set_ylabel("累積後悔 (1000試行平均)")
axes[0].set_title("後悔の伸び方: 線形に伸び続けるか、頭打ちになるか")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].set_xlabel("ステップ")
axes[1].set_ylabel("最適腕の選択率")
axes[1].set_title("最適腕をどれだけ選べているか")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig1_race.png", dpi=110, bbox_inches="tight")
np.savez("race.npz", **{n: r["cum_regret"] for n, r in results.items()})
print("saved fig1_race.png")
