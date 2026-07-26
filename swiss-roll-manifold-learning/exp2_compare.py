import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE

plt.rcParams["font.family"] = "Meiryo"

X = np.load("C:/tmp/swiss-roll-experiment/X.npy")
t = np.load("C:/tmp/swiss-roll-experiment/t.npy")

methods = {
    "PCA(線形・比較用)": PCA(n_components=2, random_state=0),
    "Isomap": Isomap(n_neighbors=10, n_components=2),
    "LLE": LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=0),
    "t-SNE": TSNE(n_components=2, perplexity=30, random_state=0, init="pca"),
}

fig, axes = plt.subplots(1, 4, figsize=(19, 5))
for ax, (name, model) in zip(axes, methods.items()):
    emb = model.fit_transform(X)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=t, cmap="turbo", s=6)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
    print(f"{name}: embedding range x=({emb[:,0].min():.1f},{emb[:,0].max():.1f}) y=({emb[:,1].min():.1f},{emb[:,1].max():.1f})")
plt.colorbar(sc, ax=axes, fraction=0.02, label="巻きの位置 t")
plt.savefig("C:/tmp/swiss-roll-experiment/fig2_compare.png", dpi=150, bbox_inches="tight")
print("saved fig2_compare.png")
