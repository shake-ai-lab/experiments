import numpy as np
from gensim.models import KeyedVectors

kv = KeyedVectors.load("C:/tmp/posenc-experiment/glove50.kv")

sentence = ["the", "animal", "did", "not", "cross", "the", "street", "because", "it", "was", "too", "tired"]
X = np.stack([kv[w] for w in sentence])  # (12, 50)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def self_attention(X):
    d_k = X.shape[1]
    scores = X @ X.T / np.sqrt(d_k)
    attn = softmax(scores)
    return attn, attn @ X

attn_orig, out_orig = self_attention(X)

rng = np.random.default_rng(1)
perm = rng.permutation(len(sentence))
X_shuffled = X[perm]
attn_shuf, out_shuf = self_attention(X_shuffled)

# attn_shuf[i,j] should equal attn_orig[perm[i], perm[j]]
attn_orig_permuted = attn_orig[np.ix_(perm, perm)]
diff_attn = np.abs(attn_shuf - attn_orig_permuted).max()

# output rows should just be permuted, not changed in value
out_orig_permuted = out_orig[perm]
diff_out = np.abs(out_shuf - out_orig_permuted).max()

print("shuffled order:", [sentence[i] for i in perm])
print("max |attn_shuffled - attn_original[perm,perm]| =", diff_attn)
print("max |output_shuffled - output_original[perm]|   =", diff_out)

print("\n'it' attention distribution, original order:")
it_idx = sentence.index("it")
for w, p in sorted(zip(sentence, attn_orig[it_idx]), key=lambda t: -t[1]):
    print(f"  {w:10s} {p:.4f}")

it_idx_shuf = list(perm).index(it_idx)
print("\n'it' attention distribution, shuffled order (same word, same weights, different position):")
for w, p in sorted(zip([sentence[i] for i in perm], attn_shuf[it_idx_shuf]), key=lambda t: -t[1]):
    print(f"  {w:10s} {p:.4f}")
