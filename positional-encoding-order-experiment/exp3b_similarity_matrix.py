import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

PE = np.load("C:/tmp/posenc-experiment/PE.npy")

n = 150
P = PE[:n]
norms = np.linalg.norm(P, axis=1, keepdims=True)
Pn = P / norms
sim_matrix = Pn @ Pn.T  # (n, n) cosine similarity between every pair of positions

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("位置 pos_j")
ax.set_ylabel("位置 pos_i")
ax.set_title(f"位置エンコーディング同士のコサイン類似度行列 (pos=0〜{n-1})")
plt.colorbar(im, ax=ax, label="コサイン類似度")
plt.savefig("C:/tmp/posenc-experiment/fig3_similarity_matrix.png", dpi=150, bbox_inches="tight")
print("saved fig3_similarity_matrix.png")
