# -*- coding: utf-8 -*-
"""追補図(3.png): 線形次元削減とUMAPの2次元配置を「本文文字数」で塗り、
「タグの分離に見えるものは記事の長さの軸と強く相関している」ことを可視化する。

corpus.json(C:\\tmp\\blog-corpus-self-audit-umap-experiment)を再利用し、
analyze.py と同じパラメータ・同じ乱数シードで埋め込みを再計算する
(シルエットスコアが本文掲載値と一致することを確認済み=同一の埋め込み)。
"""
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
plt.rcParams["font.family"] = "Meiryo"

WORKDIR = Path(r"C:\tmp\blog-corpus-self-audit-umap-experiment")
OUT_IMAGES = Path(r"C:\Users\kouta\project\forsubwork\src\content\blog\images\blog-corpus-self-audit-umap")

records = json.loads((WORKDIR / "corpus.json").read_text(encoding="utf-8"))
titles = [r["title"] for r in records]
bodies = [r["body"] for r in records]
lens = np.array([len(b) for b in bodies])

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
X = vec.fit_transform(bodies)

emb_umap2d = umap.UMAP(
    n_components=2, metric="cosine", n_neighbors=10, min_dist=0.3, random_state=0
).fit_transform(X)
emb_svd2d = TruncatedSVD(n_components=2, random_state=0).fit_transform(X)

r_svd0 = np.corrcoef(emb_svd2d[:, 0], lens)[0, 1]
r_umap0 = np.corrcoef(emb_umap2d[:, 0], lens)[0, 1]
r_umap1 = np.corrcoef(emb_umap2d[:, 1], lens)[0, 1]
print(f"線形 第1軸 vs 文字数: r={r_svd0:+.3f}")
print(f"UMAP 第1軸 vs 文字数: r={r_umap0:+.3f} / 第2軸: r={r_umap1:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
norm = plt.Normalize(vmin=lens.min(), vmax=lens.max())
for ax, (name, emb, note) in zip(
    axes,
    [
        ("線形 (SVD/PCA相当)", emb_svd2d, f"第1軸(横)と文字数の相関 r={r_svd0:.2f}"),
        ("UMAP", emb_umap2d, f"各軸と文字数の相関 r={r_umap0:+.2f} / {r_umap1:+.2f}"),
    ],
):
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=lens, cmap="Purples", norm=norm,
        s=60, edgecolors="#555555", linewidths=0.6,
    )
    ax.set_title(name, fontsize=13)
    ax.text(
        0.5, -0.06, note, transform=ax.transAxes,
        ha="center", va="top", fontsize=11, color="#444444",
    )
    ax.set_xticks([])
    ax.set_yticks([])
cbar = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02)
cbar.set_label("本文の文字数", fontsize=12)
fig.suptitle(
    "同じ2次元配置を「本文の文字数」で塗ると、タグの分かれ目と同じ向きに濃淡の勾配が現れる",
    fontsize=13,
)
fig.savefig(OUT_IMAGES / "3.png", dpi=130, bbox_inches="tight")
print("saved 3.png")

# --- 記事中の関連する数値の検証出力 ---
Dhigh = cosine_distances(X)
De = euclidean_distances(emb_umap2d)
np.fill_diagonal(De, np.inf)
hi = next(i for i, t in enumerate(titles) if "ハルシネーション" in t)
order = np.argsort(De[hi])[:3]
print("\nハルシネーション記事のUMAP平面上の最近傍3記事(ユークリッド距離):")
for k in order:
    print(f"  d={De[hi, k]:.3f}  {titles[k][:40]}")
nnd = De.min(axis=1)
print(f"全記事の最近傍距離の中央値: {np.median(nnd):.3f}")
ag = next(i for i, t in enumerate(titles) if "エージェント" in t and "Skill" in t)
print(f"ハルシネーション×エージェント/Skill: 2D距離={De[hi, ag]:.3f}, 高次元コサイン類似度={1 - Dhigh[hi, ag]:.4f}")

tre = next(i for i, t in enumerate(titles) if "結び目" in t)
row = np.array(X[tre].todense()).ravel()
feats = np.array(vec.get_feature_names_out())
top = np.argsort(row)[::-1][:8]
print(f"\n「三つ葉結び目」記事のTF-IDF上位n-gram: {[feats[k] for k in top]}")
dh = Dhigh[tre].copy()
dh[tre] = np.inf
near = np.argsort(dh)[:2]
print("「三つ葉結び目」に高次元で一番近い2記事:", [titles[k][:30] for k in near])
