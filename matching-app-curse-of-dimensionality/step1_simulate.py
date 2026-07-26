"""マッチングアプリの「質問を増やすほど良いマッチングができる」は本当か、を
次元の呪い(高次元では距離が意味を失う)の観点からシミュレーションする。

設定:
  - 本当に相性を決めるのは k_signal=3個の「価値観」次元(コア・コンパティビリティ)だけ、とする
  - マッチングアプリはそれとは別に d_noise個の「雑多な質問」(朝型/夜型、犬派/猫派、等)
    を追加で聞いてくるが、これらは相性そのものには寄与しないノイズ次元とする
  - 候補者プールN=300人、それぞれ signal座標(k_signal次元)とnoise座標(d_noise次元)を
    独立に[0,1]一様分布から持つ
  - 「本当の最良マッチ」= signal座標だけでのユークリッド距離が一番近い候補者
  - 「アプリが実際に薦める最良マッチ」= signal+noiseを全部含めたフル次元での
    ユークリッド距離が一番近い候補者(=素朴な全次元マッチングアルゴリズム)
  - d_noiseを0から200まで振り、本当の最良マッチがフル次元ランキングで何位に
    埋もれてしまうか(rank)、ちょうど1位を当てられるか(hit@1)、
    トップ10には入るか(hit@10)を、各設定R=300試行の平均で測る
"""
import numpy as np

OUTDIR = "C:/tmp/matching-app-curse-of-dimensionality"
N = 300  # candidate pool size
K_SIGNAL = 3  # true compatibility dimensions
NOISE_SWEEP = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200]
R = 300  # trials per noise-dimension setting


def run_trial(rng, d_noise):
    query_signal = rng.uniform(0, 1, size=K_SIGNAL)
    cand_signal = rng.uniform(0, 1, size=(N, K_SIGNAL))
    true_dist = np.linalg.norm(cand_signal - query_signal, axis=1)
    best_true_idx = int(np.argmin(true_dist))

    if d_noise > 0:
        query_noise = rng.uniform(0, 1, size=d_noise)
        cand_noise = rng.uniform(0, 1, size=(N, d_noise))
        full_query = np.concatenate([query_signal, query_noise])
        full_cand = np.concatenate([cand_signal, cand_noise], axis=1)
    else:
        full_query, full_cand = query_signal, cand_signal

    full_dist = np.linalg.norm(full_cand - full_query, axis=1)
    order = np.argsort(full_dist)  # order[0] = app's #1 recommendation
    rank_of_best_true = int(np.where(order == best_true_idx)[0][0]) + 1  # 1-indexed
    hit1 = (order[0] == best_true_idx)
    hit10 = best_true_idx in order[:10]
    return rank_of_best_true, hit1, hit10


results = []
for d_noise in NOISE_SWEEP:
    rng = np.random.default_rng(1000 + d_noise)
    ranks, hit1s, hit10s = [], [], []
    for trial in range(R):
        rank, hit1, hit10 = run_trial(rng, d_noise)
        ranks.append(rank)
        hit1s.append(hit1)
        hit10s.append(hit10)
    ranks = np.array(ranks)
    hit1_rate = np.mean(hit1s)
    hit10_rate = np.mean(hit10s)
    print(f"d_noise={d_noise:4d}  total_dims={K_SIGNAL + d_noise:4d}  "
          f"mean_rank={ranks.mean():6.1f}±{ranks.std():5.1f}  "
          f"hit@1={hit1_rate:.4f}  hit@10={hit10_rate:.4f}")
    results.append(dict(d_noise=d_noise, ranks=ranks, hit1_rate=hit1_rate, hit10_rate=hit10_rate))

mean_ranks = np.array([r["ranks"].mean() for r in results])
std_ranks = np.array([r["ranks"].std() for r in results])
hit1_rates = np.array([r["hit1_rate"] for r in results])
hit10_rates = np.array([r["hit10_rate"] for r in results])
d_noise_arr = np.array(NOISE_SWEEP)

random_baseline_rank = (N + 1) / 2
random_baseline_hit1 = 1.0 / N
random_baseline_hit10 = 10.0 / N
print(f"\nrandom baseline (pure noise ranking): mean_rank={random_baseline_rank:.1f}  "
      f"hit@1={random_baseline_hit1:.4f}  hit@10={random_baseline_hit10:.4f}")

# find approx d_noise where mean rank first crosses 10 (true best match typically drops out of top-10)
cross10 = None
for i in range(len(d_noise_arr) - 1):
    if mean_ranks[i] < 10 <= mean_ranks[i + 1]:
        # linear interpolation between the two points
        x0, x1 = d_noise_arr[i], d_noise_arr[i + 1]
        y0, y1 = mean_ranks[i], mean_ranks[i + 1]
        cross10 = x0 + (10 - y0) * (x1 - x0) / (y1 - y0)
        break
print(f"approx d_noise where mean rank crosses 10: {cross10}")

# save per-trial rank arrays for a couple of representative noise settings (for histogram figure)
rank_dist_by_dnoise = {str(r["d_noise"]): r["ranks"] for r in results if r["d_noise"] in (0, 12, 60, 200)}

np.savez(
    f"{OUTDIR}/results.npz",
    d_noise=d_noise_arr, mean_ranks=mean_ranks, std_ranks=std_ranks,
    hit1_rates=hit1_rates, hit10_rates=hit10_rates,
    N=N, K_SIGNAL=K_SIGNAL, R=R,
    random_baseline_rank=random_baseline_rank,
    random_baseline_hit1=random_baseline_hit1,
    random_baseline_hit10=random_baseline_hit10,
    cross10=cross10 if cross10 is not None else -1.0,
    ranks_d0=rank_dist_by_dnoise.get("0"),
    ranks_d12=rank_dist_by_dnoise.get("12"),
    ranks_d60=rank_dist_by_dnoise.get("60"),
    ranks_d200=rank_dist_by_dnoise.get("200"),
)
print("saved results.npz")
