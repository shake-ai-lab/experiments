"""スイスロールに穴を開けて、Isomap/LLE/t-SNEの展開がどう壊れるかを見る。
穴はシート座標(弧長s, 幅y)上で円形にくり抜く。評価は前回記事と同じ
「埋め込み軸と巻き位置tの相関」(2軸のうち高い方の絶対値)。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

# 密にサンプルしてから穴を開け、条件間で点数を揃える
X_all, t_all = make_swiss_roll(n_samples=6000, noise=0.05, random_state=0)
y_all = X_all[:, 1]  # シートの幅方向

# 弧長 s(t): ds = sqrt(1+t^2) dt
order = np.argsort(t_all)
t_sorted = t_all[order]
s_sorted = cumulative_trapezoid(np.sqrt(1 + t_sorted**2), t_sorted, initial=0)
s_all = np.empty_like(s_sorted)
s_all[order] = s_sorted
print(f"s range: 0..{s_all.max():.1f}, y range: {y_all.min():.1f}..{y_all.max():.1f}")

s_c, y_c = s_all.max() * 0.55, (y_all.min() + y_all.max()) / 2
HOLES = {"穴なし": 0.0, "小さい穴": 4.0, "大きい穴": 8.0}


def cut(radius):
    if radius == 0:
        keep = np.ones(len(s_all), bool)
    else:
        keep = (s_all - s_c) ** 2 + (y_all - y_c) ** 2 > radius**2
    idx = np.flatnonzero(keep)
    idx = rng.choice(idx, size=1500, replace=False)
    return idx


datasets = {}
for name, R in HOLES.items():
    idx = cut(R)
    datasets[name] = dict(X=X_all[idx], t=t_all[idx], s=s_all[idx], y=y_all[idx], R=R)
    print(f"{name}: R={R}, n={len(idx)}")

np.savez(
    "data.npz",
    **{f"{k}_{f}": v[f] for k, v in datasets.items() for f in ("X", "t", "s", "y")},
    s_c=s_c, y_c=y_c,
)


def best_corr(emb, t):
    return max(abs(np.corrcoef(emb[:, i], t)[0, 1]) for i in range(emb.shape[1]))


methods = {
    "Isomap": lambda: Isomap(n_neighbors=10, n_components=2),
    "LLE": lambda: LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=0),
    "t-SNE": lambda: TSNE(n_components=2, perplexity=30, random_state=0, init="pca"),
}

fig, axes = plt.subplots(len(methods), len(datasets), figsize=(12.5, 11))
results = {}
for j, (dname, d) in enumerate(datasets.items()):
    for i, (mname, mk) in enumerate(methods.items()):
        emb = mk().fit_transform(d["X"])
        c = best_corr(emb, d["t"])
        results[(mname, dname)] = c
        ax = axes[i, j]
        ax.scatter(emb[:, 0], emb[:, 1], c=d["t"], cmap="turbo", s=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{mname} / {dname}  corr={c:.3f}", fontsize=10)
        print(f"{mname:8s} {dname}: corr {c:.4f}")
fig.suptitle("穴あきスイスロールの2次元展開 (色=巻きの位置t)", fontsize=13)
fig.tight_layout()
fig.savefig("fig2_embeddings.png", dpi=100, bbox_inches="tight")
np.savez("embed_results.npz", **{f"{m}|{d}": v for (m, d), v in results.items()})
print("saved fig2_embeddings.png")
