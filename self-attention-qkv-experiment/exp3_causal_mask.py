import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(1)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

seq_len = 10
d_k = 32
Q = rng.normal(0, 1, size=(seq_len, d_k))
K = rng.normal(0, 1, size=(seq_len, d_k))
scores = Q @ K.T / np.sqrt(d_k)

attn_bidir = softmax(scores)

mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
scores_masked = scores.copy()
scores_masked[mask] = -np.inf
attn_causal = softmax(scores_masked)

print("bidirectional attention, row sums:", attn_bidir.sum(axis=1))
print("causal attention, row sums:", attn_causal.sum(axis=1))
print("\ncausal: first token (index 0) attention row:", attn_causal[0])
print("causal: last token (index 9) attention row:", attn_causal[9])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, attn, title in zip(axes, [attn_bidir, attn_causal], ["双方向(BERT型): 全位置を参照可能", "causal mask(GPT型): 自分より前だけ参照可能"]):
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=max(attn_bidir.max(), attn_causal.max()))
    ax.set_xlabel("Key位置(参照先)")
    ax.set_ylabel("Query位置(参照元)")
    ax.set_title(title)
plt.colorbar(im, ax=axes, fraction=0.025)
plt.savefig("C:/tmp/attention-experiment/fig3_causal_mask.png", dpi=150, bbox_inches="tight")
print("\nsaved fig3_causal_mask.png")
