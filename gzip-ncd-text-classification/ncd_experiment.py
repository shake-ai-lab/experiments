"""gzip(zlib)による正規化圧縮距離(NCD) + kNN でブログ記事を分類する追試。

- NCD(x,y) = (C(xy) - min(C(x),C(y))) / max(C(x),C(y))
- タスクA: 「実践記/機械学習」vs「Claude入門/比較」の2トラック分類(leave-one-out)
- タスクB: series(連載)6クラス分類(leave-one-out)
- 対照群: TF-IDF文字n-gram + ロジスティック回帰 / TF-IDFコサイン距離 + 1NN
- 「k=2のタイブレークが実質top-2精度」問題(Ken Schutteの指摘)の再現
- 日本語UTF-8の圧縮特性の観察
結果はすべて results.json に保存する。
"""
import json
import zlib
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

EXP_DIR = Path(r"C:\tmp\gzip-ncd-text-classification-experiment")
LEVEL = 9  # zlib最高圧縮

records = json.loads((EXP_DIR / "corpus.json").read_text(encoding="utf-8"))
n = len(records)
print(f"コーパス: {n}記事")

texts = [r["body"] for r in records]
raw_bytes = [t.encode("utf-8") for t in texts]


def C(b: bytes) -> int:
    return len(zlib.compress(b, LEVEL))


# ---- 圧縮サイズと NCD 行列 ----
c_single = np.array([C(b) for b in raw_bytes])
ncd = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        cxy = C(raw_bytes[i] + raw_bytes[j])
        ncd[i, j] = (cxy - min(c_single[i], c_single[j])) / max(c_single[i], c_single[j])
print("NCD行列 計算完了")
np.save(EXP_DIR / "ncd_matrix.npy", ncd)

# 対称化版(可視化・シルエット用)
ncd_sym = (ncd + ncd.T) / 2
np.fill_diagonal(ncd_sym, 0.0)

# ---- UTF-8 / 圧縮特性の観察 ----
char_lens = np.array([len(t) for t in texts])
byte_lens = np.array([len(b) for b in raw_bytes])
utf8_stats = {
    "mean_bytes_per_char": float(np.mean(byte_lens / char_lens)),
    "mean_char_len": float(char_lens.mean()),
    "mean_byte_len": float(byte_lens.mean()),
    "max_byte_len": int(byte_lens.max()),
    "mean_compressed_bytes_per_char": float(np.mean(c_single / char_lens)),
    "mean_compression_ratio": float(np.mean(c_single / byte_lens)),
    # gzip/zlibの参照ウィンドウは32KB。連結x+yがそれを超えるペアの割合
    "pairs_over_32kb": float(
        np.mean((byte_lens[:, None] + byte_lens[None, :]) > 32768)
    ),
    "docs_over_16kb": int(np.sum(byte_lens > 16384)),
    "self_ncd_mean": float(np.mean(np.diag(ncd))),  # NCD(x,x)は0にならない
    "self_ncd_max": float(np.max(np.diag(ncd))),
}
print("UTF-8/圧縮特性:", json.dumps(utf8_stats, indent=1))

# ---- ラベル定義(前回のUMAP記事と同じ規則) ----
def track(r):
    tags = set(r["tags"])
    if tags & {"実践記", "機械学習"}:
        return "実験系"
    if tags & {"Claude入門", "比較"}:
        return "入門系"
    return "その他"

tracks = [track(r) for r in records]
print("トラック分布:", Counter(tracks))


# ---- kNN 評価関数(leave-one-out) ----
def loo_knn(dist, labels, idx, k, mode, n_trials=1000, seed=0):
    """dist: 全体距離行列, idx: 使用する記事のインデックス, mode:
    'proper'  = 多数決、同数タイは最近傍のラベル
    'top2bug' = k=2専用: 上位2件のどちらかが正解ラベルなら正解扱い(top-2精度)
    'random'  = 多数決、同数タイはランダム(n_trials回平均)
    """
    idx = np.asarray(idx)
    sub = dist[np.ix_(idx, idx)]
    labs = [labels[i] for i in idx]
    m = len(idx)
    if mode == "random":
        rng = random.Random(seed)
        total = 0.0
        for _ in range(n_trials):
            correct = 0
            for i in range(m):
                order = sorted((j for j in range(m) if j != i), key=lambda j: sub[i, j])
                top = [labs[j] for j in order[:k]]
                cnt = Counter(top)
                best = max(cnt.values())
                cands = [l for l, c in cnt.items() if c == best]
                pred = rng.choice(cands)
                correct += pred == labs[i]
            total += correct / m
        return total / n_trials, None
    preds = []
    for i in range(m):
        order = sorted((j for j in range(m) if j != i), key=lambda j: sub[i, j])
        top = [labs[j] for j in order[:k]]
        if mode == "top2bug":
            # Schutteが指摘したnpc_gzipのバグ相当: 上位k件に正解が含まれれば正解
            preds.append(labs[i] if labs[i] in top else top[0])
        else:  # proper
            cnt = Counter(top)
            best = max(cnt.values())
            cands = [l for l, c in cnt.items() if c == best]
            if len(cands) == 1:
                preds.append(cands[0])
            else:  # タイは最近傍
                preds.append(next(labs[j] for j in order if labs[j] in cands))
    acc = float(np.mean([p == t for p, t in zip(preds, labs)]))
    return acc, preds


def per_class_recall(preds, labs):
    out = {}
    for c in sorted(set(labs)):
        ii = [i for i, l in enumerate(labs) if l == c]
        out[c] = float(np.mean([preds[i] == labs[i] for i in ii]))
    return out


# ---- TF-IDF 対照群 ----
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
X_all = vec.fit_transform(texts)
print(f"TF-IDF行列: {X_all.shape}")
tfidf_dist = cosine_distances(X_all)


def loo_logreg(X, labels, idx, class_weight=None, C_reg=1.0):
    idx = np.asarray(idx)
    labs = [labels[i] for i in idx]
    preds = []
    for i in range(len(idx)):
        train = [j for j in range(len(idx)) if j != i]
        clf = LogisticRegression(max_iter=2000, C=C_reg, class_weight=class_weight)
        clf.fit(X[idx[train]], [labs[j] for j in train])
        preds.append(clf.predict(X[idx[[i]]])[0])
    acc = float(np.mean([p == t for p, t in zip(preds, labs)]))
    return acc, preds


results = {"utf8_stats": utf8_stats, "n_docs": n}

# ================= タスクA: 2トラック分類 =================
idx_a = [i for i, t in enumerate(tracks) if t != "その他"]
labs_a = [tracks[i] for i in idx_a]
print(f"\nタスクA: {len(idx_a)}記事 ({Counter(labs_a)})")
majority = max(Counter(labs_a).values()) / len(labs_a)

task_a = {"n": len(idx_a), "counts": dict(Counter(labs_a)), "majority_baseline": majority}
for k in (1, 3, 5):
    acc, preds = loo_knn(ncd, tracks, idx_a, k, "proper")
    task_a[f"ncd_knn_k{k}"] = acc
    if k == 1:
        task_a["ncd_knn_k1_recall"] = per_class_recall(preds, labs_a)
acc2b, preds2b = loo_knn(ncd, tracks, idx_a, 2, "top2bug")
task_a["ncd_knn_k2_top2bug"] = acc2b
task_a["ncd_knn_k2_top2bug_recall"] = per_class_recall(preds2b, labs_a)
acc2r, _ = loo_knn(ncd, tracks, idx_a, 2, "random")
task_a["ncd_knn_k2_randomtie"] = acc2r

acc, preds = loo_knn(tfidf_dist, tracks, idx_a, 1, "proper")
task_a["tfidf_knn_k1"] = acc
task_a["tfidf_knn_k1_recall"] = per_class_recall(preds, labs_a)
acc2b_t, _ = loo_knn(tfidf_dist, tracks, idx_a, 2, "top2bug")
task_a["tfidf_knn_k2_top2bug"] = acc2b_t
acc, preds = loo_logreg(X_all, tracks, idx_a)
task_a["tfidf_logreg"] = acc
task_a["tfidf_logreg_recall"] = per_class_recall(preds, labs_a)
acc, preds = loo_logreg(X_all, tracks, idx_a, class_weight="balanced")
task_a["tfidf_logreg_balanced"] = acc
task_a["tfidf_logreg_balanced_recall"] = per_class_recall(preds, labs_a)
results["task_a"] = task_a
print(json.dumps(task_a, ensure_ascii=False, indent=1))

# ================= タスクB: series 6クラス分類 =================
idx_b = [i for i, r in enumerate(records) if r["series"]]
labs_b = [records[i]["series"] for i in idx_b]
series_labels = [r["series"] or "" for r in records]
print(f"\nタスクB: {len(idx_b)}記事 ({Counter(labs_b)})")
majority_b = max(Counter(labs_b).values()) / len(labs_b)

task_b = {"n": len(idx_b), "counts": dict(Counter(labs_b)), "majority_baseline": majority_b}
acc, preds_b_ncd = loo_knn(ncd, series_labels, idx_b, 1, "proper")
task_b["ncd_knn_k1"] = acc
acc2b, _ = loo_knn(ncd, series_labels, idx_b, 2, "top2bug")
task_b["ncd_knn_k2_top2bug"] = acc2b
acc, preds_b_tfidf = loo_knn(tfidf_dist, series_labels, idx_b, 1, "proper")
task_b["tfidf_knn_k1"] = acc
acc, _ = loo_logreg(X_all, series_labels, idx_b)
task_b["tfidf_logreg"] = acc
acc, _ = loo_logreg(X_all, series_labels, idx_b, class_weight="balanced")
task_b["tfidf_logreg_balanced"] = acc
acc, _ = loo_logreg(X_all, series_labels, idx_b, class_weight="balanced", C_reg=10.0)
task_b["tfidf_logreg_balanced_C10"] = acc
task_b["ncd_errors"] = [
    {"slug": records[idx_b[i]]["slug"], "true": labs_b[i], "pred": preds_b_ncd[i]}
    for i in range(len(idx_b)) if preds_b_ncd[i] != labs_b[i]
]
results["task_b"] = task_b
print(json.dumps(task_b, ensure_ascii=False, indent=1))

# ================= UMAP記事との照合 =================
# 2トラックのシルエットスコア(NCD対称化距離、precomputed)
sil_idx = np.array(idx_a)
sil = silhouette_score(
    ncd_sym[np.ix_(sil_idx, sil_idx)],
    [tracks[i] for i in sil_idx],
    metric="precomputed",
)
# series内平均NCD vs 全体平均NCD
iu = np.triu_indices(n, k=1)
overall_mean_ncd = float(ncd_sym[iu].mean())
series_ncd = {}
for s in sorted(set(labs_b)):
    ii = np.array([i for i in idx_b if records[i]["series"] == s])
    sub = ncd_sym[np.ix_(ii, ii)]
    m = len(ii)
    series_ncd[s] = {
        "n": m,
        "within_mean_ncd": float(sub[np.triu_indices(m, k=1)].mean()),
    }
umap_cross = {
    "silhouette_track_ncd": float(sil),
    "overall_mean_ncd": overall_mean_ncd,
    "series_within_ncd": series_ncd,
}
results["umap_cross_check"] = umap_cross
print(json.dumps(umap_cross, ensure_ascii=False, indent=1))

# ---- 記事の長さとトラックの関係(交絡の確認) ----
len_by_track = {
    t: float(np.mean([char_lens[i] for i in range(n) if tracks[i] == t]))
    for t in ("実験系", "入門系")
}
results["char_len_by_track"] = len_by_track
print("トラック別平均文字数:", len_by_track)

(EXP_DIR / "results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("\n保存完了 -> results.json")
