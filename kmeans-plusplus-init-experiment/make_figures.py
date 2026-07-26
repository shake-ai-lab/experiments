import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import adjusted_rand_score
from kmeans_lib import make_dataset, init_random, init_kmeanspp, kmeans_lloyd

plt.rcParams["font.family"] = "Meiryo"

rng_data = np.random.default_rng(42)
X, y_true = make_dataset(rng_data)
k = 5
n_runs = 200
BAD_THRESH = 1000.0

# ---------- Figure 0: dataset overview (what the data actually looks like) ----------
colors = plt.cm.tab10(np.linspace(0, 1, k))
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=130)
for j in range(k):
    m = y_true == j
    ax.scatter(X[m, 0], X[m, 1], s=16, color=colors[j], alpha=0.75)
ann_kw = dict(ha="center", fontsize=12.5, color="#333333")
ax.text(0, 4.9, "中心: 200点 (σ=1.2)", **ann_kw)
ax.text(15, 2.1, "東: 12点 (σ=0.4)", **ann_kw)
ax.text(-15, 2.1, "西: 12点 (σ=0.4)", **ann_kw)
ax.text(0, 17.0, "北: 12点 (σ=0.4)", **ann_kw)
ax.text(0, -18.3, "南: 12点 (σ=0.4)", **ann_kw)
ax.annotate("", xy=(13.5, 0), xytext=(4.3, 0),
            arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.2))
ax.text(8.9, 0.9, "中心間の距離15", ha="center", fontsize=11.5, color="#666666")
ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
ax.set_aspect("equal")
ax.tick_params(labelsize=11)
ax.set_title("実験データ: 大きく薄い塊1つ + 遠く離れた小さく濃い塊4つ(全248点)", fontsize=13)
fig.tight_layout()
fig.savefig("0_dataset.png")
plt.close(fig)

# ---------- Stats: where do the 200 random inits actually land? ----------
outer_hits = []
for i in range(n_runs):
    rng = np.random.default_rng(1000 + i)
    idx = rng.choice(len(X), size=k, replace=False)
    outer_hits.append(int((y_true[idx] != 0).sum()))
outer_hits = np.array(outer_hits)
print("random init: how many of the 5 initial centroids hit an outer blob")
for v in range(6):
    print(f"  {v} outer hits: {(outer_hits == v).sum():3d} runs ({(outer_hits == v).mean()*100:.1f}%)")

def run_many(init_fn, n_runs, seed0):
    inertias, aris, all_labels, all_centroids = [], [], [], []
    for i in range(n_runs):
        rng = np.random.default_rng(seed0 + i)
        init_c = init_fn(X, k, rng)
        centroids, labels, inertia = kmeans_lloyd(X, init_c)
        aris.append(adjusted_rand_score(y_true, labels))
        inertias.append(inertia)
        all_labels.append(labels)
        all_centroids.append(centroids)
    return np.array(inertias), np.array(aris), all_labels, all_centroids

inertia_rand, ari_rand, labels_rand, cent_rand = run_many(init_random, n_runs, seed0=1000)
inertia_pp, ari_pp, labels_pp, cent_pp = run_many(init_kmeanspp, n_runs, seed0=5000)

best_inertia = min(inertia_rand.min(), inertia_pp.min())
bad_rand = inertia_rand > BAD_THRESH
bad_pp = inertia_pp > BAD_THRESH

print("=== SUMMARY ===")
print(f"global best inertia: {best_inertia:.2f}")
print(f"random: bad frac = {bad_rand.mean()*100:.1f}%  mean inertia={inertia_rand.mean():.1f}  std={inertia_rand.std():.1f}")
print(f"pp:     bad frac = {bad_pp.mean()*100:.1f}%  mean inertia={inertia_pp.mean():.1f}  std={inertia_pp.std():.1f}")
print(f"random ari mean={ari_rand.mean():.3f}  perfect(ari=1.0) frac={ (ari_rand==1.0).mean()*100:.1f}%")
print(f"pp     ari mean={ari_pp.mean():.3f}  perfect(ari=1.0) frac={ (ari_pp==1.0).mean()*100:.1f}%")

# ---------- Figure 1: inertia distribution histogram ----------
fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
bins = np.linspace(400, 6000, 60)
ax.hist(inertia_rand, bins=bins, alpha=0.6, label=f"ランダム初期化 (n=200, 不良{bad_rand.mean()*100:.0f}%)", color="#d95f5f")
ax.hist(inertia_pp, bins=bins, alpha=0.6, label=f"k-means++ (n=200, 不良{bad_pp.mean()*100:.0f}%)", color="#4f7fbf")
ax.axvline(best_inertia, color="black", linestyle="--", linewidth=1, label=f"最良inertia={best_inertia:.0f}")
ax.set_xlabel("最終inertia(クラスタ内二乗距離の総和)")
ax.set_ylabel("試行回数")
ax.set_title("初期化方法ごとの最終inertia分布(200試行)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("1_inertia_hist.png")
plt.close(fig)

# ---------- Figure 2: side-by-side bad-random vs good-pp clustering ----------
# pick worst random run (max inertia) as the "genuinely bad" example
bad_idx = int(np.argmax(inertia_rand))
# pick a representative (median-ish, perfect) pp run
good_candidates = [i for i in range(n_runs) if ari_pp[i] == 1.0]
good_idx = good_candidates[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=130)
colors = plt.cm.tab10(np.linspace(0, 1, k))

ax = axes[0]
lab = labels_rand[bad_idx]
for j in range(k):
    m = lab == j
    ax.scatter(X[m, 0], X[m, 1], s=14, color=colors[j], alpha=0.8)
cen = cent_rand[bad_idx]
ax.scatter(cen[:, 0], cen[:, 1], marker="x", s=140, color="black", linewidths=2.5)
ax.set_title(f"ランダム初期化の一例(不良)\ninertia={inertia_rand[bad_idx]:.0f} (最良比 {inertia_rand[bad_idx]/best_inertia:.1f}倍) / ARI={ari_rand[bad_idx]:.2f}")
ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
ax.set_aspect("equal")

ax = axes[1]
lab = labels_pp[good_idx]
for j in range(k):
    m = lab == j
    ax.scatter(X[m, 0], X[m, 1], s=14, color=colors[j], alpha=0.8)
cen = cent_pp[good_idx]
ax.scatter(cen[:, 0], cen[:, 1], marker="x", s=140, color="black", linewidths=2.5)
ax.set_title(f"k-means++初期化の一例(良好)\ninertia={inertia_pp[good_idx]:.0f} (最良比 {inertia_pp[good_idx]/best_inertia:.2f}倍) / ARI={ari_pp[good_idx]:.2f}")
ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
ax.set_aspect("equal")

fig.suptitle("同じデータに対する、悪い局所最適解と良い局所最適解の実例(×は最終セントロイド)")
fig.tight_layout()
fig.savefig("2_side_by_side.png")
plt.close(fig)

print(f"\nbad example: idx={bad_idx} inertia={inertia_rand[bad_idx]:.1f} ari={ari_rand[bad_idx]:.3f}")
print(f"good example: idx={good_idx} inertia={inertia_pp[good_idx]:.1f} ari={ari_pp[good_idx]:.3f}")

# ---------- GIF: animate the bad run's centroids converging ----------
rng = np.random.default_rng(1000 + bad_idx)
init_c = init_random(X, k, rng)
centroids_final, labels_final, inertia_final, history = kmeans_lloyd(X, init_c, record_history=True)
print(f"\nGIF source run: {len(history)} Lloyd steps, final inertia={inertia_final:.1f}")

# recompute labels at each step of history for coloring
def labels_at(centroids):
    dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return dists.argmin(axis=1)

# drop trailing duplicate history entries (the last Lloyd pass moves nothing --
# it only confirms convergence), so the GIF frame count matches visible updates
while len(history) >= 2 and np.allclose(history[-1], history[-2]):
    history.pop()
n_updates = len(history) - 1  # centroid updates actually shown (frame 0 = init)
print(f"GIF frames: {len(history)} (init + {n_updates} centroid updates)")

fig, ax = plt.subplots(figsize=(7, 4.6), dpi=100)

def draw(frame_idx):
    ax.clear()
    cen = history[frame_idx]
    lab = labels_at(cen)
    for j in range(k):
        m = lab == j
        ax.scatter(X[m, 0], X[m, 1], s=12, color=colors[j], alpha=0.8)
    ax.scatter(cen[:, 0], cen[:, 1], marker="x", s=130, color="black", linewidths=2.5)
    ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=10)
    if frame_idx == 0:
        tag = "初期配置"
    elif frame_idx == n_updates:
        tag = "収束"
    else:
        tag = f"重心更新 {frame_idx}回目"
    ax.set_title(f"ランダム初期化からLloyd法で悪い局所最適解へ | {tag} ({frame_idx}/{n_updates})",
                 fontsize=12.5)
    fig.tight_layout()

frames = []
for i in range(len(history)):
    draw(i)
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
plt.close(fig)

durations = [1600] + [900] * (len(frames) - 2) + [2600]
frames[0].save("bad_convergence.gif", save_all=True, append_images=frames[1:],
               duration=durations, loop=0)

import os
print("\nGIF size KB:", os.path.getsize("bad_convergence.gif") / 1024)
