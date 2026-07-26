import numpy as np
import matplotlib.pyplot as plt
from step1_simulate import simulate, report

plt.rcParams["font.family"] = "Meiryo"

hours_a, score_a, hours_b, score_b = simulate()
res = report(hours_a, score_a, hours_b, score_b)

hours_all = np.concatenate([hours_a, hours_b])
score_all = np.concatenate([score_a, score_b])

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)

ax.scatter(hours_a, score_a, s=22, alpha=0.65, color="#2563eb", label=f"上級者グループ (n={len(hours_a)})")
ax.scatter(hours_b, score_b, s=22, alpha=0.65, color="#f97316", label=f"初級者グループ (n={len(hours_b)})")

xs_a = np.linspace(hours_a.min(), hours_a.max(), 50)
xs_b = np.linspace(hours_b.min(), hours_b.max(), 50)
xs_all = np.linspace(hours_all.min(), hours_all.max(), 50)

ax.plot(xs_a, res["intercept_a"] + res["slope_a"] * xs_a, color="#1e3a8a", lw=2.5,
        label=f"上級者の回帰直線 (r={res['r_a']:.2f})")
ax.plot(xs_b, res["intercept_b"] + res["slope_b"] * xs_b, color="#c2410c", lw=2.5,
        label=f"初級者の回帰直線 (r={res['r_b']:.2f})")
ax.plot(xs_all, res["intercept_pool"] + res["slope_pool"] * xs_all, color="#111827", lw=3,
        linestyle="--", label=f"全体をプールした回帰直線 (r={res['r_pool']:.2f})")

ax.set_xlabel("週あたりの学習アプリ利用時間 (時間)")
ax.set_ylabel("到達テストのスコア")
ax.set_title("グループ内では正、全体では負——Simpsonのパラドックス")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("1.png", dpi=140)
print("saved 1.png")
