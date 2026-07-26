import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

kv = KeyedVectors.load("C:/tmp/posenc-experiment/glove50.kv")
sentence = ["the", "animal", "did", "not", "cross", "the", "street", "because", "it", "was", "too", "tired"]
X = np.stack([kv[w] for w in sentence])

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def self_attention(X):
    d_k = X.shape[1]
    return softmax(X @ X.T / np.sqrt(d_k))

attn_orig = self_attention(X)

rng = np.random.default_rng(1)
perm = rng.permutation(len(sentence))
attn_shuf = self_attention(X[perm])

# "un-shuffle" back to original word order so we can compare directly against attn_orig
inv_perm = np.argsort(perm)
attn_shuf_restored = attn_shuf[np.ix_(inv_perm, inv_perm)]

diff = np.abs(attn_orig - attn_shuf_restored).max()
print("max abs diff after un-shuffling back:", diff)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, mat, title in zip(
    axes,
    [attn_orig, attn_shuf_restored, np.abs(attn_orig - attn_shuf_restored)],
    ["元の語順でattention計算", "シャッフルして計算→並び戻したもの", "差分(絶対値)"],
):
    im = ax.imshow(mat, cmap="viridis" if title != "差分(絶対値)" else "Reds")
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))
    ax.set_xticklabels(sentence, rotation=90)
    ax.set_yticklabels(sentence)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("C:/tmp/posenc-experiment/fig1_permutation_heatmaps.png", dpi=150, bbox_inches="tight")
print("saved fig1_permutation_heatmaps.png")
