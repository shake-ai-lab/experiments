import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

PE = np.load("C:/tmp/posenc-experiment/PE.npy")
max_pos, d_model = PE.shape

def cos_sim(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

# fix a base position, look at similarity to all others
base = 50
sims = np.array([cos_sim(PE[base], PE[p]) for p in range(max_pos)])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.arange(max_pos), sims, color="steelblue")
ax.axvline(base, color="crimson", linestyle="--", linewidth=1, label=f"基準位置 pos={base}")
ax.set_xlabel("位置 pos")
ax.set_ylabel(f"PE({base}) との コサイン類似度")
ax.set_title("基準位置から離れるほど類似度がどう変化するか")
ax.legend()
plt.savefig("C:/tmp/posenc-experiment/fig3_similarity.png", dpi=150, bbox_inches="tight")
print("saved fig3_similarity.png")

# average over many base positions, as function of |delta|, to get a cleaner trend
deltas = np.arange(0, 80)
avg_sims = []
for d in deltas:
    vals = []
    for base in range(0, max_pos - d):
        vals.append(cos_sim(PE[base], PE[base + d]))
    avg_sims.append(np.mean(vals))
avg_sims = np.array(avg_sims)

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(deltas, avg_sims, color="darkorange")
ax2.set_xlabel("相対距離 |Δpos|")
ax2.set_ylabel("平均コサイン類似度")
ax2.set_title("相対距離ごとの平均類似度(全pos区間で平均)")
plt.savefig("C:/tmp/posenc-experiment/fig3b_avg_similarity.png", dpi=150, bbox_inches="tight")
print("saved fig3b_avg_similarity.png")

print("similarity at delta=1,5,10,20,40,79:")
for d in [1, 5, 10, 20, 40, 79]:
    print(f"  delta={d:3d}  avg_sim={avg_sims[d]:.4f}")
