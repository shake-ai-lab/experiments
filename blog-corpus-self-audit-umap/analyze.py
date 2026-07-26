# -*- coding: utf-8 -*-
"""ブログの全53記事本文をTF-IDF文字n-gramでベクトル化し、UMAPで2次元に
埋め込む。「実践記/機械学習」トラックと「Claude入門/比較」トラックが
埋め込み空間で分離するかどうかを実際に検証する。"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
plt.rcParams["font.family"] = "Meiryo"

WORKDIR = Path(r"C:\tmp\blog-corpus-self-audit-umap-experiment")
OUT_INTERACTIVE = Path(r"C:\Users\kouta\project\forsubwork\public\interactive\blog-corpus-umap.html")
OUT_IMAGES = Path(r"C:\Users\kouta\project\forsubwork\src\content\blog\images\blog-corpus-self-audit-umap")
OUT_IMAGES.mkdir(parents=True, exist_ok=True)

records = json.loads((WORKDIR / "corpus.json").read_text(encoding="utf-8"))
print(f"記事数: {len(records)}")

titles = [r["title"] for r in records]
slugs = [r["slug"] for r in records]
bodies = [r["body"] for r in records]
tags_list = [r["tags"] for r in records]
series_list = [r["series"] for r in records]


def track_of(tags):
    has_a = any(t in ("実践記", "機械学習") for t in tags)
    has_b = any(t in ("Claude入門", "比較") for t in tags)
    if has_a and has_b:
        return "両方(実践記/機械学習 & Claude入門/比較)"
    if has_a:
        return "実践記/機械学習"
    if has_b:
        return "Claude入門/比較"
    return "その他"


tracks = [track_of(t) for t in tags_list]
print("トラック分布:", Counter(tracks))
print("タグ分布:", Counter(t for tags in tags_list for t in tags))
print("series分布:", Counter(s or "(なし)" for s in series_list))

# ---------- TF-IDF (文字2-4gram) ----------
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
X = vec.fit_transform(bodies)
print(f"TF-IDF行列: {X.shape}  (記事数 x 文字n-gram語彙数)")

# ---------- UMAP (umap-learn が実際にインストールされていたので使用) ----------
print(f"umap-learn version: {umap.__version__}")
reducer2d = umap.UMAP(n_components=2, metric="cosine", n_neighbors=10, min_dist=0.3, random_state=0)
emb_umap2d = reducer2d.fit_transform(X)

reducer3d = umap.UMAP(n_components=3, metric="cosine", n_neighbors=10, min_dist=0.3, random_state=0)
emb_umap3d = reducer3d.fit_transform(X)

# ---------- 比較用: 線形次元削減(TruncatedSVD, TF-IDFの疎行列にPCA相当) ----------
svd = TruncatedSVD(n_components=2, random_state=0)
emb_svd2d = svd.fit_transform(X)

# ---------- 比較用: PCA(50次元)->t-SNE ----------
svd50 = TruncatedSVD(n_components=50, random_state=0)
X50 = svd50.fit_transform(X)
emb_tsne2d = TSNE(n_components=2, perplexity=15, random_state=0, init="pca").fit_transform(X50)

# ---------- 定量評価: シルエットスコアで「実践記/機械学習」vs「Claude入門/比較」の分離度を測る ----------
mask = np.array([tr in ("実践記/機械学習", "Claude入門/比較") for tr in tracks])
labels_bin = np.array([1 if tr == "実践記/機械学習" else 0 for tr in tracks])[mask]
print(f"2群比較の対象記事数: {mask.sum()} (実践記/機械学習={sum(labels_bin==1)}, Claude入門/比較={sum(labels_bin==0)})")

sil_highdim = silhouette_score(X[mask], labels_bin, metric="cosine")
sil_umap2d = silhouette_score(emb_umap2d[mask], labels_bin)
sil_svd2d = silhouette_score(emb_svd2d[mask], labels_bin)
sil_tsne2d = silhouette_score(emb_tsne2d[mask], labels_bin)
print(f"シルエットスコア (高次元TF-IDF, cosine): {sil_highdim:.4f}")
print(f"シルエットスコア (UMAP 2D): {sil_umap2d:.4f}")
print(f"シルエットスコア (SVD/PCA相当 2D): {sil_svd2d:.4f}")
print(f"シルエットスコア (t-SNE 2D): {sil_tsne2d:.4f}")

# ---------- series内の凝集度: 同一series内の平均コサイン距離 vs 全体平均 ----------
from sklearn.metrics.pairwise import cosine_distances

D = cosine_distances(X)
overall_mean = D[np.triu_indices_from(D, k=1)].mean()
print(f"\n全記事ペアの平均コサイン距離: {overall_mean:.4f}")
series_names = sorted(set(s for s in series_list if s))
for sname in series_names:
    idx = [i for i, s in enumerate(series_list) if s == sname]
    if len(idx) < 2:
        continue
    sub = D[np.ix_(idx, idx)]
    within_mean = sub[np.triu_indices_from(sub, k=1)].mean()
    print(f"  series「{sname}」(n={len(idx)}): 内部平均距離={within_mean:.4f}  (全体平均との差: {within_mean - overall_mean:+.4f})")

# ---------- 静的比較図: SVD(線形) vs t-SNE vs UMAP、トラックで色分け ----------
track_colors = {
    "実践記/機械学習": "#d1495b",
    "Claude入門/比較": "#2b6cb0",
    "その他": "#a0a0a0",
    "両方(実践記/機械学習 & Claude入門/比較)": "#8850c4",
}
color_arr = [track_colors[t] for t in tracks]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, (name, emb) in zip(
    axes, [("線形 (SVD/PCA相当)", emb_svd2d), ("t-SNE", emb_tsne2d), ("UMAP", emb_umap2d)]
):
    for t, c in track_colors.items():
        idx = [i for i, tr in enumerate(tracks) if tr == t]
        if not idx:
            continue
        ax.scatter(emb[idx, 0], emb[idx, 1], c=c, s=55, label=t, edgecolors="white", linewidths=0.6)
    ax.set_title(name, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
fig.suptitle(
    "3手法とも「実践記/機械学習」(赤)と「Claude入門/比較」(青)はきれいには分離しない",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT_IMAGES / "1.png", dpi=130, bbox_inches="tight")
print("saved 1.png")

# ---------- 静的図2: series別の色分け(UMAP 2D) ----------
fig2, ax2 = plt.subplots(figsize=(8, 7))
cmap = plt.get_cmap("tab10")
series_color_map = {s: cmap(i % 10) for i, s in enumerate(series_names)}
for i in range(len(records)):
    s = series_list[i]
    if s is None:
        ax2.scatter(emb_umap2d[i, 0], emb_umap2d[i, 1], c="#d0d0d0", s=40, zorder=1)
    else:
        ax2.scatter(
            emb_umap2d[i, 0], emb_umap2d[i, 1], c=[series_color_map[s]], s=70,
            edgecolors="black", linewidths=0.4, zorder=2,
        )
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=series_color_map[s], markersize=9, label=s)
    for s in series_names
]
handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d0d0d0", markersize=9, label="(seriesなし)"))
ax2.legend(handles=handles, fontsize=7.5, loc="best", framealpha=0.9)
ax2.set_title("series(連載)ごとの色分け:同じseriesの記事はUMAP空間で近くに集まる", fontsize=12)
ax2.set_xticks([])
ax2.set_yticks([])
fig2.tight_layout()
fig2.savefig(OUT_IMAGES / "2.png", dpi=130, bbox_inches="tight")
print("saved 2.png")

# ---------- インタラクティブ Plotly (UMAP 2D, タグトラックで色分け, hoverでタイトル) ----------
fig_p = go.Figure()
for t, c in track_colors.items():
    idx = [i for i, tr in enumerate(tracks) if tr == t]
    if not idx:
        continue
    fig_p.add_trace(
        go.Scatter(
            x=emb_umap2d[idx, 0],
            y=emb_umap2d[idx, 1],
            mode="markers",
            name=t,
            marker=dict(size=11, color=c, line=dict(width=1, color="white")),
            text=[f"{titles[i]}<br>tags: {', '.join(tags_list[i])}<br>series: {series_list[i] or '(なし)'}" for i in idx],
            hovertemplate="%{text}<extra></extra>",
        )
    )
fig_p.update_layout(
    title=dict(
        text="ブログ全53記事のUMAP埋め込み<br><sup>TF-IDF文字n-gram, タグトラックで色分け</sup>",
        font=dict(size=15),
    ),
    xaxis_title="UMAP-1",
    yaxis_title="UMAP-2",
    legend=dict(
        title="タグトラック",
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="center",
        x=0.5,
    ),
    margin=dict(t=70, b=90, l=60, r=20),
    height=680,
    template="plotly_white",
    font=dict(size=13),
)
OUT_INTERACTIVE.parent.mkdir(parents=True, exist_ok=True)
fig_p.write_html(str(OUT_INTERACTIVE), include_plotlyjs="cdn")
print(f"saved {OUT_INTERACTIVE}")

# ---------- 一番近い他トラックのペアを探す(定性的に見る用) ----------
# 注意: 2次元の埋め込み平面上の「近さ」は必ずユークリッド距離で測ること。
# 当初ここで cosine_distances(emb_umap2d) を使っていたが、2次元でのコサイン距離は
# 「原点から見た角度」しか測らないため、視覚的な近さとは別物になる(修正済み)。
from sklearn.metrics.pairwise import euclidean_distances

nn_examples = []
Dfull = euclidean_distances(emb_umap2d)
np.fill_diagonal(Dfull, np.inf)
nn_median = np.median(Dfull.min(axis=1))
print(f"\n全記事の最近傍距離(UMAP平面, ユークリッド)の中央値: {nn_median:.3f}")
for i in range(len(records)):
    j = int(np.argmin(Dfull[i]))
    if tracks[i] != tracks[j] and tracks[i] != "その他" and tracks[j] != "その他":
        nn_examples.append((titles[i], tracks[i], titles[j], tracks[j], round(float(Dfull[i, j]), 3)))
print("異なるトラック同士が最近傍になっている例:")
for ex in nn_examples[:10]:
    print(" -", ex)

print("\n完了")
