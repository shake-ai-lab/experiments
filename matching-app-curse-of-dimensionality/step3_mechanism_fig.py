"""追加図: 「雑多な質問1問がランキングをどう壊すか」を1試行分だけ具体的に見せる散布図。

step1_simulate.py の d_noise=1 の実験(seed 1000+1)とまったく同じ乱数列を再生し、
300試行の中から「本当の最良マッチの順位」が全体平均(30.8位)に一番近い試行を選んで、
その試行の全候補者300人について
  x軸: 価値観3次元だけの距離(本当の相性)
  y軸: 価値観3次元+雑多な質問1問の距離(アプリが見ている距離)
を散布図にする。full_dist^2 = true_dist^2 + (雑多な質問の差)^2 なので点は必ず対角線の上側に乗り、
「上に持ち上げられた量」がそのまま雑多な質問1問のたまたまの不一致、という構図になる。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"

OUTDIR = "C:/tmp/matching-app-curse-of-dimensionality"
N = 300
K_SIGNAL = 3
D_NOISE = 1
R = 300
SEED = 1000 + D_NOISE  # step1と同一

COL = "#e4572e"
GREEN = "#2fa96b"


def gen_trial(rng):
    query_signal = rng.uniform(0, 1, size=K_SIGNAL)
    cand_signal = rng.uniform(0, 1, size=(N, K_SIGNAL))
    query_noise = rng.uniform(0, 1, size=D_NOISE)
    cand_noise = rng.uniform(0, 1, size=(N, D_NOISE))
    true_dist = np.linalg.norm(cand_signal - query_signal, axis=1)
    full_query = np.concatenate([query_signal, query_noise])
    full_cand = np.concatenate([cand_signal, cand_noise], axis=1)
    full_dist = np.linalg.norm(full_cand - full_query, axis=1)
    return true_dist, full_dist


# --- pass 1: step1と同じ順で300試行流して、平均順位に一番近い試行を選ぶ ---
rng = np.random.default_rng(SEED)
ranks = []
for _ in range(R):
    true_dist, full_dist = gen_trial(rng)
    best_true = int(np.argmin(true_dist))
    order = np.argsort(full_dist)
    ranks.append(int(np.where(order == best_true)[0][0]) + 1)
ranks = np.array(ranks)
mean_rank = ranks.mean()
pick = int(np.argmin(np.abs(ranks - mean_rank)))
print(f"mean rank over {R} trials = {mean_rank:.1f}; picked trial #{pick} with rank {ranks[pick]}")

# --- pass 2: 選んだ試行のデータを再生 ---
rng = np.random.default_rng(SEED)
for t in range(pick + 1):
    true_dist, full_dist = gen_trial(rng)
best_true = int(np.argmin(true_dist))
rank_best = int(np.where(np.argsort(full_dist) == best_true)[0][0]) + 1
overtakers = full_dist < full_dist[best_true]  # アプリが本当の最良マッチより上位に置いた候補
n_over = int(overtakers.sum())
assert rank_best == n_over + 1 == ranks[pick]

fig, ax = plt.subplots(figsize=(9.0, 6.4))
lim = max(true_dist.max(), full_dist.max()) * 1.05
ax.plot([0, lim], [0, lim], ls="--", color="#aaa", lw=1.2, zorder=1)
others = ~overtakers
others[best_true] = False
ax.scatter(true_dist[others], full_dist[others], s=26, color="#c9c9c9",
           edgecolor="white", linewidth=0.4, zorder=2, label="その他の候補者")
ax.scatter(true_dist[overtakers], full_dist[overtakers], s=42, color=COL,
           edgecolor="white", linewidth=0.6, zorder=3,
           label=f"アプリが本当の最良マッチより上位に置いた候補({n_over}人)")
ax.scatter([true_dist[best_true]], [full_dist[best_true]], s=260, marker="*",
           color=GREEN, edgecolor="black", linewidth=0.8, zorder=4,
           label="本当の最良マッチ(価値観距離が最小)")
ax.axhline(full_dist[best_true], color=GREEN, lw=1.2, ls=":", zorder=1)
ax.annotate(
    f"この線より下の{n_over}人が全員先に推薦される\n→ 本当の最良マッチはアプリ順位{rank_best}位",
    xy=(true_dist[best_true], full_dist[best_true]),
    xytext=(0.04, lim * 0.72),
    fontsize=11.5, arrowprops=dict(arrowstyle="->", color="#333"))
ax.annotate("対角線 = 雑多な質問の答えが\nたまたま完全一致だった場合",
            xy=(lim * 0.72, lim * 0.72), xytext=(lim * 0.80, lim * 0.50),
            fontsize=10.5, color="#666", arrowprops=dict(arrowstyle="->", color="#aaa"))
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("価値観3次元だけの距離(本当の相性。左ほど相性が良い)", fontsize=12)
ax.set_ylabel("価値観3次元+雑多な質問1問の距離(アプリが見ている距離)", fontsize=12)
ax.set_title("雑多な質問1問の「たまたまの不一致」が、各候補者を対角線から上に押し上げて\n順位を入れ替える(300試行のうち平均に近い1試行の全候補者300人)", fontsize=12.5)
ax.legend(fontsize=10.5, loc="lower right")
ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig4_mechanism_scatter.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"saved fig4_mechanism_scatter.png (trial rank={rank_best}, overtakers={n_over})")
