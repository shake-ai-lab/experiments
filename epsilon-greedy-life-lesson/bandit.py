import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

rng = np.random.default_rng(42)

K = 10          # 腕(選択肢)の数
STEPS = 1000    # 1回の試行あたりのステップ数
RUNS = 2000     # 独立試行の回数(平均を取ってならす)


def run_bandit(epsilon, steps=STEPS, seed=None, decay=False):
    """1回分のε-greedyバンディットを実行し、報酬と最適行動選択率の履歴を返す"""
    local_rng = np.random.default_rng(seed)
    q_true = local_rng.normal(0, 1, size=K)       # 各腕の本当の期待報酬(未知という設定)
    optimal_action = np.argmax(q_true)

    q_est = np.zeros(K)     # 推定価値
    n_pulls = np.zeros(K)   # 各腕を引いた回数

    rewards = np.zeros(steps)
    is_optimal = np.zeros(steps)

    for t in range(steps):
        eps = epsilon / (1 + t / 200) if decay else epsilon

        if local_rng.random() < eps:
            action = local_rng.integers(K)          # 探索:ランダムに選ぶ
        else:
            action = np.argmax(q_est)                # 活用:今のベストを選ぶ

        reward = local_rng.normal(q_true[action], 1)  # 実際に引いて得られる報酬

        n_pulls[action] += 1
        q_est[action] += (reward - q_est[action]) / n_pulls[action]  # 平均を更新

        rewards[t] = reward
        is_optimal[t] = (action == optimal_action)

    return rewards, is_optimal


def average_over_runs(epsilon, runs=RUNS, decay=False):
    all_rewards = np.zeros((runs, STEPS))
    all_optimal = np.zeros((runs, STEPS))
    for i in range(runs):
        r, o = run_bandit(epsilon, seed=1000 + i, decay=decay)
        all_rewards[i] = r
        all_optimal[i] = o
    return all_rewards.mean(axis=0), all_optimal.mean(axis=0) * 100


settings = [
    ("epsilon = 0 (完全greedy)", 0.0, False),
    ("epsilon = 0.01", 0.01, False),
    ("epsilon = 0.1", 0.1, False),
    ("epsilon = 0.3", 0.3, False),
    ("epsilon = 0.3 -> 減衰", 0.3, True),
]

results = {}
for label, eps, decay in settings:
    avg_r, avg_opt = average_over_runs(eps, decay=decay)
    results[label] = (avg_r, avg_opt)
    print(f"{label:24s} final_avg_reward={avg_r[-100:].mean():.4f}  final_optimal_rate={avg_opt[-100:].mean():.2f}%")

# --- 図1: 平均報酬の推移 ---
plt.figure(figsize=(9, 5))
for label, (avg_r, _) in results.items():
    plt.plot(avg_r, label=label)
plt.xlabel("ステップ")
plt.ylabel("平均報酬")
plt.title(f"epsilonごとの平均報酬の推移({RUNS}試行平均)")
plt.legend()
plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150)
plt.close()

# --- 図2: 最適行動を選んだ割合 ---
plt.figure(figsize=(9, 5))
for label, (_, avg_opt) in results.items():
    plt.plot(avg_opt, label=label)
plt.xlabel("ステップ")
plt.ylabel("最適な腕を選んだ割合(%)")
plt.title(f"epsilonごとの最適行動選択率({RUNS}試行平均)")
plt.legend()
plt.tight_layout()
plt.savefig("optimal_action_curve.png", dpi=150)
plt.close()

print("\n--- summary table ---")
for label, (avg_r, avg_opt) in results.items():
    print(f"{label:24s} cumulative_reward={avg_r.sum():.1f}")
