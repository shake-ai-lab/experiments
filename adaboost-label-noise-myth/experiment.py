"""メイン実験:
1. クリーンラベル(決定的ラベル, Bayes誤差0)でのAdaBoost: 訓練誤差0後もテスト誤差が下がるか
2. ラベルノイズ 0-30% x ブースティング回数 1-1000 のテスト誤差 (5シード平均)
3. 対照群 Random Forest (同条件)
4. 誤ラベル点のサンプル重みの追跡 (重み爆発の可視化用)
結果はresults.npzに保存。
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

from adaboost import AdaBoostStumps

# ---- データ生成: 位置ノイズ付きtwo moons + 最近傍アークによる決定的ラベル ----
_t = np.linspace(0, np.pi, 2000)
_ARC0 = np.stack([np.cos(_t), np.sin(_t)], axis=1)            # 外側の月 (class 0)
_ARC1 = np.stack([1 - np.cos(_t), 0.5 - np.sin(_t)], axis=1)  # 内側の月 (class 1)


def deterministic_labels(X):
    d0 = np.min(np.sum((X[:, None, :] - _ARC0[None]) ** 2, axis=2), axis=1)
    d1 = np.min(np.sum((X[:, None, :] - _ARC1[None]) ** 2, axis=2), axis=1)
    return np.where(d1 < d0, 1.0, -1.0)


def make_data(n, seed, pos_noise=0.2):
    X, _ = make_moons(n_samples=n, noise=pos_noise, random_state=seed)
    y = deterministic_labels(X)
    return X, y


def flip_labels(y, rate, seed):
    rng = np.random.default_rng(seed)
    n_flip = int(round(rate * len(y)))
    idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[idx] *= -1
    return y_noisy, idx


N_TRAIN = 300
N_TEST = 2000
T_MAX = 1000
NOISE_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SEEDS = [0, 1, 2, 3, 4]
RF_TREES = 300

X_te, y_te = make_data(N_TEST, seed=999)

results = {}

# ---- 実験2+3: ノイズ率 x 回数 のテスト誤差 (AdaBoost) と RF ----
ada_err = np.zeros((len(SEEDS), len(NOISE_RATES), T_MAX))   # テスト誤差
ada_tr_err = np.zeros((len(SEEDS), len(NOISE_RATES), T_MAX))  # (ノイズ入り)訓練誤差
rf_err = np.zeros((len(SEEDS), len(NOISE_RATES)))

for si, seed in enumerate(SEEDS):
    X_tr, y_tr = make_data(N_TRAIN, seed=seed)
    for ni, rate in enumerate(NOISE_RATES):
        y_noisy, flip_idx = flip_labels(y_tr, rate, seed=1000 + seed)
        model = AdaBoostStumps(n_rounds=T_MAX).fit(X_tr, y_noisy)
        ada_err[si, ni] = model.staged_errors(X_te, y_te)
        ada_tr_err[si, ni] = model.staged_errors(X_tr, y_noisy)
        rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=seed, n_jobs=-1)
        rf.fit(X_tr, (y_noisy > 0).astype(int))
        rf_err[si, ni] = np.mean(rf.predict(X_te) != (y_te > 0).astype(int))
        print(f"seed={seed} noise={rate:.2f}  ada@1000={ada_err[si, ni, -1]:.4f}  "
              f"ada_min={ada_err[si, ni].min():.4f}@{ada_err[si, ni].argmin()+1}  "
              f"rf={rf_err[si, ni]:.4f}", flush=True)

results["ada_err"] = ada_err
results["ada_tr_err"] = ada_tr_err
results["rf_err"] = rf_err
results["noise_rates"] = np.array(NOISE_RATES)
results["seeds"] = np.array(SEEDS)

# ---- 実験4: 重みの追跡 (seed=0, ノイズ10%) + GIF用の学習履歴 ----
seed = 0
X_tr, y_tr = make_data(N_TRAIN, seed=seed)
y_noisy, flip_idx = flip_labels(y_tr, 0.10, seed=1000 + seed)
model_w = AdaBoostStumps(n_rounds=T_MAX).fit(X_tr, y_noisy, record_weights=True)
W = np.array(model_w.weight_history)  # (T, n)
is_flipped = np.zeros(N_TRAIN, dtype=bool)
is_flipped[flip_idx] = True
results["weights_history"] = W
results["is_flipped"] = is_flipped
results["Xtr_w"] = X_tr
results["ytr_noisy_w"] = y_noisy
results["stumps_w"] = np.array(model_w.stumps)  # (T,3): feature, thr, polarity
results["alphas_w"] = model_w.alphas

mean_w_flip = W[:, is_flipped].mean(axis=1)
mean_w_clean = W[:, ~is_flipped].mean(axis=1)
print(f"\n[重み] ノイズ10%: 誤ラベル30点の平均重み T=1: {mean_w_flip[0]:.5f} -> "
      f"T=1000: {mean_w_flip[-1]:.5f}")
print(f"[重み] クリーン270点の平均重み T=1000: {mean_w_clean[-1]:.5f}  "
      f"比率: {mean_w_flip[-1] / mean_w_clean[-1]:.1f}x")
print(f"[重み] T=1000で誤ラベル30点(10%)が占める総重み: {W[-1, is_flipped].sum():.3f}")

# ---- 実験1の要約 (クリーン, 各シード) ----
print("\n[クリーン] 訓練誤差が最初に0になる回数と、その時点/最終のテスト誤差:")
for si, seed in enumerate(SEEDS):
    tr = ada_tr_err[si, 0]
    te = ada_err[si, 0]
    zero_at = int(np.argmax(tr == 0)) + 1 if (tr == 0).any() else -1
    print(f"  seed={seed}: train0@{zero_at}  test@train0={te[zero_at-1]:.4f}  "
          f"test@1000={te[-1]:.4f}  test_min={te.min():.4f}@{te.argmin()+1}")

np.savez_compressed("results.npz", **results)
print("\nsaved results.npz")
