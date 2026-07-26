import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

K = 10
STEPS = 500


def run_bandit(q_true, epsilon, seed):
    local_rng = np.random.default_rng(seed)
    optimal_action = np.argmax(q_true)

    q_est = np.zeros(K)
    n_pulls = np.zeros(K)
    chosen = np.zeros(STEPS, dtype=int)
    running_avg = np.zeros(STEPS)
    total = 0.0

    for t in range(STEPS):
        if local_rng.random() < epsilon:
            action = local_rng.integers(K)
        else:
            action = np.argmax(q_est)

        reward = local_rng.normal(q_true[action], 1)
        n_pulls[action] += 1
        q_est[action] += (reward - q_est[action]) / n_pulls[action]

        total += reward
        chosen[t] = action
        running_avg[t] = total / (t + 1)

    return chosen, running_avg, optimal_action, q_est


# 「最初の一回でハズレを引いて、そのまま固まってしまう」ケースを探す
found = None
for seed in range(0, 2000):
    trial_rng = np.random.default_rng(seed)
    q_true = trial_rng.normal(0, 1, size=K)
    optimal = np.argmax(q_true)

    chosen, avg, _, q_est = run_bandit(q_true, epsilon=0.0, seed=seed + 5000)
    late_choice = chosen[100:]
    stuck_action = np.bincount(late_choice, minlength=K).argmax()
    stuck_rate = np.mean(late_choice == stuck_action)

    if stuck_action != optimal and stuck_rate > 0.99 and (q_true[optimal] - q_true[stuck_action]) > 0.8:
        found = seed
        break

print("seed used:", found)

trial_rng = np.random.default_rng(found)
q_true = trial_rng.normal(0, 1, size=K)
optimal = np.argmax(q_true)
print("q_true:", np.round(q_true, 2))
print("optimal action:", optimal, "true value:", round(q_true[optimal], 2))

chosen_g, avg_g, _, q_est_g = run_bandit(q_true, epsilon=0.0, seed=found + 5000)
chosen_e, avg_e, _, q_est_e = run_bandit(q_true, epsilon=0.1, seed=found + 5000)

stuck_action = np.bincount(chosen_g[100:], minlength=K).argmax()
print("greedy locked onto action:", stuck_action, "true value:", round(q_true[stuck_action], 2))
print("greedy final running avg reward:", round(avg_g[-1], 3))
print("epsilon-greedy final running avg reward:", round(avg_e[-1], 3))
print("greedy picked optimal action in last 300 steps:", np.mean(chosen_g[-300:] == optimal) * 100, "%")
print("eps-greedy picked optimal action in last 300 steps:", np.mean(chosen_e[-300:] == optimal) * 100, "%")

plt.figure(figsize=(9, 5))
plt.plot(avg_g, label="epsilon = 0 (完全greedy) : 最初のハズレに固執")
plt.plot(avg_e, label="epsilon = 0.1 : たまに他の腕も試す")
plt.axhline(q_true[optimal], color="gray", linestyle="--", linewidth=1, label="真の最適な腕の期待報酬")
plt.xlabel("ステップ")
plt.ylabel("その時点までの平均報酬")
plt.title("たった1回、序盤の運が悪かっただけで固まってしまう例(単一試行)")
plt.legend()
plt.tight_layout()
plt.savefig("trap_run.png", dpi=150)
plt.close()
