import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import plotly.graph_objects as go

kv = KeyedVectors.load("C:/tmp/attention-experiment/glove50.kv")

sentence = ["the", "animal", "did", "not", "cross", "the", "street", "because", "it", "was", "too", "tired"]
X = np.stack([kv[w] for w in sentence])

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# same random Q/K projection as exp1 (untrained transformer layer)
rng = np.random.default_rng(0)
d_model = X.shape[1]
d_proj = 16
Wq = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_proj))
Wk = rng.normal(0, 1 / np.sqrt(d_model), size=(d_model, d_proj))
Q = X @ Wq
K = X @ Wk
scores = Q @ K.T / np.sqrt(d_proj)
attn = softmax(scores)

it_idx = sentence.index("it")
weights = attn[it_idx]  # attention from "it" to every word (including itself)

# PCA the raw 50d embeddings down to 3D just for a spatial, intuitive picture
pca = PCA(n_components=3, random_state=0)
coords = pca.fit_transform(X)
print("explained variance ratio:", pca.explained_variance_ratio_, "sum:", pca.explained_variance_ratio_.sum())

it_pos = coords[it_idx]

fig = go.Figure()

# lines from "it" to every other word, opacity/width driven by attention weight
for i, w in enumerate(sentence):
    if i == it_idx:
        continue
    fig.add_trace(go.Scatter3d(
        x=[it_pos[0], coords[i, 0]], y=[it_pos[1], coords[i, 1]], z=[it_pos[2], coords[i, 2]],
        mode="lines",
        line=dict(color=f"rgba(220,60,40,{0.15 + 0.85 * weights[i] / weights.max():.3f})",
                  width=2 + 18 * weights[i] / weights.max()),
        hoverinfo="skip",
        showlegend=False,
    ))

fig.add_trace(go.Scatter3d(
    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
    mode="markers+text",
    text=sentence,
    textposition="top center",
    marker=dict(
        size=8 + 40 * weights / weights.max(),
        color=weights,
        colorscale="YlOrRd",
        colorbar=dict(title='"it"からの<br>attention重み'),
        line=dict(color="black", width=1),
    ),
    hovertext=[f"{w}: attn={p:.4f}" for w, p in zip(sentence, weights)],
    hoverinfo="text",
    showlegend=False,
))

fig.update_layout(
    title='"it" (中心) から各単語への注目度。点の大きさ・色・線の太さがattention重み(ランダム初期化Q/K射影後)',
    scene=dict(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)",
        zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.0f}%)",
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    height=650,
)

fig.write_html("C:/tmp/attention-experiment/self-attention-3d.html", include_plotlyjs="cdn", full_html=True)
print("saved self-attention-3d.html")
