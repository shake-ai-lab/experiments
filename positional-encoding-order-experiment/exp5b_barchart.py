import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

kv = KeyedVectors.load("C:/tmp/posenc-experiment/glove50.kv")
PE = np.load("C:/tmp/posenc-experiment/PE.npy")

sentence = ["the", "animal", "did", "not", "cross", "the", "street", "because", "it", "was", "too", "tired"]
X = np.stack([kv[w] for w in sentence])
d_model = X.shape[1]
X_pos = X + PE[:len(sentence), :d_model]

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def self_attention(X):
    d_k = X.shape[1]
    return softmax(X @ X.T / np.sqrt(d_k))

attn_no_pe = self_attention(X)
attn_with_pe = self_attention(X_pos)

it_idx = sentence.index("it")
w_no = attn_no_pe[it_idx]
w_with = attn_with_pe[it_idx]

# order words by their distance from "it" in the sentence, to make the "locality bias" visible
order = np.argsort(np.abs(np.arange(len(sentence)) - it_idx))
labels = [sentence[i] for i in order]

x = np.arange(len(sentence))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width / 2, w_no[order], width, label="位置エンコーディングなし", color="steelblue")
ax.bar(x + width / 2, w_with[order], width, label="位置エンコーディングあり", color="crimson")
ax.set_xticks(x)
ax.set_xticklabels([f"{lbl}\n({'it自身' if i==it_idx else ('+' if i>it_idx else '') + str(i-it_idx)})" for i, lbl in zip(order, labels)], fontsize=9)
ax.set_ylabel("'it' からのattention重み")
ax.set_title("位置エンコーディングを足すと、'it' の注目はどう変わるか(itからの相対位置順に並べ替え)")
ax.legend()
# highlight animal specifically
animal_x = list(order).index(sentence.index("animal"))
ax.annotate("animal(変化なし)", xy=(animal_x, max(w_no[sentence.index("animal")], w_with[sentence.index("animal")]) + 0.01),
            ha="center", fontsize=9, color="black")
plt.tight_layout()
plt.savefig("C:/tmp/posenc-experiment/fig5_it_barchart.png", dpi=150, bbox_inches="tight")
print("saved fig5_it_barchart.png")
