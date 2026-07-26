import numpy as np
from gensim.models import KeyedVectors

kv = KeyedVectors.load("C:/tmp/posenc-experiment/glove50.kv")
PE = np.load("C:/tmp/posenc-experiment/PE.npy")

sentence = ["the", "animal", "did", "not", "cross", "the", "street", "because", "it", "was", "too", "tired"]
X = np.stack([kv[w] for w in sentence])  # (12, 50)
d_model = X.shape[1]
X_pos = X + PE[:len(sentence), :d_model]

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def self_attention(X):
    d_k = X.shape[1]
    scores = X @ X.T / np.sqrt(d_k)
    return softmax(scores)

attn_no_pe = self_attention(X)
attn_with_pe = self_attention(X_pos)

it_idx = sentence.index("it")
print("'it' attention WITHOUT positional encoding:")
for w, p in sorted(zip(sentence, attn_no_pe[it_idx]), key=lambda t: -t[1]):
    print(f"  {w:10s} {p:.4f}")

print("\n'it' attention WITH positional encoding added (same word vectors, position now baked in):")
for w, p in sorted(zip(sentence, attn_with_pe[it_idx]), key=lambda t: -t[1]):
    print(f"  {w:10s} {p:.4f}")

print("\nchange per word (with - without):")
for w, a, b in zip(sentence, attn_no_pe[it_idx], attn_with_pe[it_idx]):
    print(f"  {w:10s} {b - a:+.4f}")
