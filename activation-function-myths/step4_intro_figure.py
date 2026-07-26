"""検証2の導入用の図: two-moonsデータそのものと、訓練700点/テスト300点の分割を可視化する。

データはstep2_depth_sweep.pyと完全に同じ生成パイプライン(make_moons seed=0 → 標準化 →
permutation seed=0 → 先頭700点が訓練・残り300点がテスト)で決定的に再現し、
キャッシュ済みnpzのX_trと一致することを検証してから描画する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

plt.rcParams["font.family"] = "Meiryo"

OUTDIR = "C:/tmp/activation-function-myths"

N_SAMPLES = 1000
N_TRAIN = 700

# step2_depth_sweep.py と同一のデータ生成
X, y = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=0)
mu, sd = X.mean(axis=0), X.std(axis=0)
X = (X - mu) / sd
y = y.astype(np.float64)
rng0 = np.random.default_rng(0)
perm = rng0.permutation(len(X))
X, y = X[perm], y[perm]
X_tr, y_tr = X[:N_TRAIN], y[:N_TRAIN]
X_te, y_te = X[N_TRAIN:], y[N_TRAIN:]

# キャッシュ済みの実験データと同一であることを確認(違ったら描画しない)
ds = np.load(f"{OUTDIR}/depth_sweep_results.npz")
assert np.allclose(ds["X_tr"], X_tr), "X_tr mismatch vs cached npz"
assert np.allclose(ds["y_tr"], y_tr), "y_tr mismatch vs cached npz"
n0, n1 = int((y_te == 0).sum()), int((y_te == 1).sum())
print(f"test split: class0={n0}, class1={n1}, all-class1 acc={n1 / len(y_te):.3f}")

C0, C1 = "#2e6ee4", "#e4572e"  # class 0 = 青, class 1 = 赤(他の図の配色に合わせる)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)

ax = axes[0]
ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], s=14, c=C0, alpha=0.65, label="class 0 (上の三日月)")
ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], s=14, c=C1, alpha=0.65, label="class 1 (下の三日月)")
ax.set_title(f"訓練データ {N_TRAIN}点\n(モデルはこれだけを見て学習する)", fontsize=14)
ax.legend(fontsize=12, loc="lower left")

ax = axes[1]
ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], s=14, c=C0, alpha=0.65, label=f"class 0 ({n0}点)")
ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], s=14, c=C1, alpha=0.65, label=f"class 1 ({n1}点)")
ax.set_title(f"テストデータ {len(y_te)}点\n(学習には見せず、採点にだけ使う)", fontsize=14)
ax.legend(fontsize=12, loc="lower left")

for ax in axes:
    ax.set_xlabel("x1 (標準化済み)", fontsize=12)
axes[0].set_ylabel("x2 (標準化済み)", fontsize=12)

fig.suptitle("two-moons: 2つの三日月が互い違いに絡み合う2クラス分類データ (1000点, noise=0.25)", fontsize=14.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUTDIR}/fig0_two_moons.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig0_two_moons.png")
