import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

kv = KeyedVectors.load("C:/tmp/attention-experiment/glove50.kv")

sentence = ["the", "animal", "did", "not", "cross", "the", "street", "because", "it", "was", "too", "tired"]
X = np.stack([kv[w] for w in sentence])  # (seq_len, 50) real GloVe embeddings
print("embedding matrix shape:", X.shape)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# --- raw attention: no Q/K projection at all, just embeddings themselves as Q=K=V ---
d_k = X.shape[1]
scores_raw = X @ X.T / np.sqrt(d_k)
attn_raw = softmax(scores_raw)

it_idx = sentence.index("it")
print("\n[raw embeddings, no projection] attention from 'it' to each word:")
for w, p in sorted(zip(sentence, attn_raw[it_idx]), key=lambda t: -t[1]):
    print(f"  {w:10s} {p:.4f}")

# --- random (untrained) Q/K/V projection, as in a freshly-initialized transformer layer ---
rng = np.random.default_rng(0)
d_model = X.shape[1]
d_proj = 16
Wq = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_proj))
Wk = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_proj))
Wv = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_proj))

Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores_proj = Q @ K.T / np.sqrt(d_proj)
attn_proj = softmax(scores_proj)

print("\n[random untrained Q/K projection] attention from 'it' to each word:")
for w, p in sorted(zip(sentence, attn_proj[it_idx]), key=lambda t: -t[1]):
    print(f"  {w:10s} {p:.4f}")

output = attn_proj @ V
print("\noutput vector shape (context-mixed representation of each token):", output.shape)

# figure: two heatmaps side by side (raw vs random-projected), full attention matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, attn, title in zip(axes, [attn_raw, attn_proj], ["Q=K=生のGloVe埋め込み(射影なし)", "ランダム初期化のQ/K射影後"]):
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=attn.max())
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))
    ax.set_xticklabels(sentence, rotation=90)
    ax.set_yticklabels(sentence)
    ax.set_title(title)
plt.colorbar(im, ax=axes, fraction=0.025)
plt.savefig("C:/tmp/attention-experiment/fig1_qkv_basic.png", dpi=150, bbox_inches="tight")
print("\nsaved fig1_qkv_basic.png")
