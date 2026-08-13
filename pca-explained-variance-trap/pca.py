"""「累積寄与率80%まで取れば十分」を検証する。

主成分分析は分散の大きい方向から順に残す。だが分散の大きさと「知りたいことが
入っているか」は別の話で、この2つがずれると寄与率を基準にした打ち切りが壊れる。

3つ調べる。
  A: 分離に効く方向の分散をわざと小さくした合成データで、寄与率がどう外すか
  B: 実データ(手書き数字・乳がん)で、累積寄与率と分類精度がどこでずれるか
  C: 標準化しないと何が起きるか(単位の大きい列に主成分が乗っ取られる)

結果は results.json に落とす。図は figures.py。
"""

import json

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260814)
CV = 5
out = {}


def accuracy(X, y):
    """分類精度を交差検証で測る。

    主成分に落としたあとで再び標準化してはいけない。標準化は各成分の分散を1に
    揃えてしまう(=白色化)ので、分散の小さい成分が増幅され、「分散が小さい方向を
    捨てるとどうなるか」という問い自体が消える。実際それをやると全成分を使った
    場合の精度が元データと一致せず(0.9093 対 0.9204)、比較にならなかった。
    標準化は主成分に落とす前に1回だけ。L2つきロジスティック回帰は回転で不変なので、
    全成分を残した場合は元データと同じ精度になる。
    """
    clf = LogisticRegression(max_iter=5000)
    return float(cross_val_score(clf, X, y, cv=CV).mean())


# ---------------------------------------------------------------- A 合成データ
# 20次元。全列に共通で乗る強い因子が1つあり、これが分散のほとんどを占める。
# クラスを分けているのはその因子と直交する「前半10列と後半10列の差」で、
# こちらの分散は小さい。
#
# 最初は列ごとに分散を変えて仕込んだが、それは失敗だった。標準化が列の分散を
# 1に揃えてしまうので、仕込んだ大小がそのまま消える。分散の偏りは列の単位では
# なく**列どうしの相関**から作らないと、標準化のあとまで残らない。
N, DIM = 1200, 20
DELTA = 0.35  # クラス差の大きさ。大きくすると信号の分散も増えて罠が緩む
y = RNG.integers(0, 2, N)
factor = RNG.normal(0, 1, (N, 1))
X = 3.0 * factor + RNG.normal(0, 1, (N, DIM))
contrast = np.r_[np.ones(DIM // 2), -np.ones(DIM // 2)]  # 共通因子と直交する方向
X += (y[:, None] - 0.5) * 2 * DELTA * contrast

Xsyn = StandardScaler().fit_transform(X)
pca_syn = PCA().fit(Xsyn)
Z = pca_syn.transform(Xsyn)
syn_rows = []
for i in range(DIM):
    syn_rows.append(
        {
            "component": i + 1,
            "ratio": float(pca_syn.explained_variance_ratio_[i]),
            # その主成分 1本だけで分類したときの精度
            "alone": accuracy(Z[:, [i]], y),
        }
    )
cum = np.cumsum(pca_syn.explained_variance_ratio_)
k80 = int(np.searchsorted(cum, 0.80) + 1)
out["synthetic"] = {
    "dim": DIM,
    "rows": syn_rows,
    "cum80_k": k80,
    "acc_at_80": accuracy(Z[:, :k80], y),
    "acc_all": accuracy(Z, y),
    "acc_raw": accuracy(Xsyn, y),
    # 信号を入れた方向に対応する主成分は、寄与率で何番目か
    "best_alone_component": int(max(syn_rows, key=lambda r: r["alone"])["component"]),
    "best_alone_acc": float(max(r["alone"] for r in syn_rows)),
}
print("A:", out["synthetic"]["cum80_k"], out["synthetic"]["acc_at_80"], flush=True)


# ---------------------------------------------------------------- B 実データ
def sweep(name, data, target, ks):
    Xs = StandardScaler().fit_transform(data)
    p = PCA().fit(Xs)
    Z = p.transform(Xs)
    cum = np.cumsum(p.explained_variance_ratio_)
    rows = []
    for k in ks:
        rows.append(
            {
                "k": int(k),
                "cum_ratio": float(cum[k - 1]),
                "acc": accuracy(Z[:, :k], target),
                # 元のデータをどれだけ復元できるか(相対二乗誤差)
                "recon_error": float(1.0 - cum[k - 1]),
            }
        )
        print(f"  {name} k={k} cum={cum[k-1]:.3f} acc={rows[-1]['acc']:.4f}", flush=True)
    full = accuracy(Xs, target)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    # 精度が全次元の99%に達する最小の k
    k_acc = next((r["k"] for r in rows if r["acc"] >= full * 0.99), None)
    return {
        "n_features": int(data.shape[1]),
        "n_samples": int(data.shape[0]),
        "rows": rows,
        "acc_full": full,
        "k_for_80": k80,
        "cum_at_80": float(cum[k80 - 1]),
        "acc_at_80": accuracy(Z[:, :k80], target),
        "k_for_95": k95,
        "acc_at_95": accuracy(Z[:, :k95], target),
        "k_for_99pct_acc": k_acc,
        "cum_at_99pct_acc": float(cum[k_acc - 1]) if k_acc else None,
    }


d = load_digits()
print("B digits", flush=True)
out["digits"] = sweep("digits", d.data, d.target, list(range(1, 41)) + [50, 64])

b = load_breast_cancer()
print("B breast_cancer", flush=True)
out["breast_cancer"] = sweep("breast", b.data, b.target, list(range(1, 31)))


# ------------------------------------------------- C 標準化しないとどうなるか
# 乳がんデータは列によって単位が違う(面積は3桁、なめらかさは0.1未満)。
# 標準化せずにPCAをかけると、値の大きい列だけで第1主成分が決まる。
raw = b.data
p_raw = PCA().fit(raw)
p_std = PCA().fit(StandardScaler().fit_transform(raw))
out["scaling"] = {
    "feature_names": list(b.feature_names),
    "raw_first_ratio": float(p_raw.explained_variance_ratio_[0]),
    "std_first_ratio": float(p_std.explained_variance_ratio_[0]),
    "raw_k_for_80": int(np.searchsorted(np.cumsum(p_raw.explained_variance_ratio_), 0.80) + 1),
    "std_k_for_80": int(np.searchsorted(np.cumsum(p_std.explained_variance_ratio_), 0.80) + 1),
    # 第1主成分にどの列がどれだけ効いているか(絶対値の大きい順に5つ)
    "raw_top": [
        {"name": b.feature_names[i], "weight": float(abs(p_raw.components_[0][i]))}
        for i in np.argsort(-np.abs(p_raw.components_[0]))[:5]
    ],
    "std_top": [
        {"name": b.feature_names[i], "weight": float(abs(p_std.components_[0][i]))}
        for i in np.argsort(-np.abs(p_std.components_[0]))[:5]
    ],
    "raw_acc_at_80": accuracy(p_raw.transform(raw)[:, : int(np.searchsorted(np.cumsum(p_raw.explained_variance_ratio_), 0.80) + 1)], b.target),
    "raw_ranges": [
        {"name": b.feature_names[i], "std": float(raw[:, i].std())}
        for i in np.argsort(-raw.std(axis=0))[:5]
    ],
}
print("C:", out["scaling"]["raw_first_ratio"], out["scaling"]["std_first_ratio"], flush=True)

# 図で使う散布図データ(合成データの第1主成分と、信号が乗った主成分)
best = out["synthetic"]["best_alone_component"] - 1
out["synthetic_scatter"] = {
    "y": y[:400].tolist(),
    "pc1": Z[:400, 0].tolist(),
    "pc2": Z[:400, 1].tolist(),
    "pc_best": Z[:400, best].tolist(),
    "best_index": best + 1,
}

json.dump(out, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote results.json")
