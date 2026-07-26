"""
Data augmentation vs regularization ablation on a small two-moons training set.
Follow-up to overfitting-regularization-life-lesson.md (which used n=15 points).
"""
import json
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_TRAIN = 15
N_TEST = 2000
NOISE = 0.25
HIDDEN = (30, 30)
MAX_ITER = 1000
N_TRIALS = 25  # independent draws of the 15-point training set

RESULTS = {}


def make_train(seed):
    X, y = make_moons(n_samples=N_TRAIN, noise=NOISE, random_state=seed)
    return X, y


def make_test(seed=999):
    X, y = make_moons(n_samples=N_TEST, noise=NOISE, random_state=seed)
    return X, y


def fit_eval(Xtr, ytr, Xte, yte, alpha, seed=0):
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, alpha=alpha, max_iter=MAX_ITER,
                         solver='lbfgs', random_state=seed)
    clf.fit(Xtr, ytr)
    train_acc = accuracy_score(ytr, clf.predict(Xtr))
    test_acc = accuracy_score(yte, clf.predict(Xte))
    return train_acc, test_acc, clf


def augment(X, y, k, jitter_sigma, rot_max_deg, rng):
    """k synthetic copies per real point: small Gaussian jitter + small random
    rotation around the training set's centroid. Both are label-preserving as
    long as the magnitude is small relative to the distance to the true
    decision boundary: a few-percent positional nudge of a point that sits
    solidly inside its moon does not move it across the S-shaped boundary."""
    Xs, ys = [X.copy()], [y.copy()]
    if k > 0:
        center = X.mean(axis=0)
        for _ in range(k):
            angles = rng.uniform(-rot_max_deg, rot_max_deg, size=len(X)) * np.pi / 180.0
            cos_a, sin_a = np.cos(angles), np.sin(angles)
            Xc = X - center
            Xrot = np.stack([
                Xc[:, 0] * cos_a - Xc[:, 1] * sin_a,
                Xc[:, 0] * sin_a + Xc[:, 1] * cos_a,
            ], axis=1) + center
            jitter = rng.normal(0, jitter_sigma, size=X.shape)
            Xs.append(Xrot + jitter)
            ys.append(y.copy())
    return np.vstack(Xs), np.concatenate(ys)


Xte, yte = make_test()

# ---------------------------------------------------------------
# (a) baseline: no augmentation, no regularization
# ---------------------------------------------------------------
print("=== (a) baseline: no aug, no reg (alpha=1e-8) ===", flush=True)
tr_accs, te_accs = [], []
for seed in range(N_TRIALS):
    Xtr, ytr = make_train(seed)
    tr, te, _ = fit_eval(Xtr, ytr, Xte, yte, alpha=1e-8, seed=seed)
    tr_accs.append(tr)
    te_accs.append(te)
tr_accs, te_accs = np.array(tr_accs), np.array(te_accs)
gap = tr_accs - te_accs
print(f"mean train_acc={tr_accs.mean():.4f} mean test_acc={te_accs.mean():.4f} mean_gap={gap.mean():.4f}", flush=True)
RESULTS['baseline'] = dict(train_acc=tr_accs.mean(), test_acc=te_accs.mean(), gap=gap.mean())

# ---------------------------------------------------------------
# (b) regularization only: sweep alpha, no augmentation
# ---------------------------------------------------------------
print("\n=== (b) regularization only: alpha sweep ===", flush=True)
alphas = np.logspace(-8, 2, 12)
alpha_results = []
for alpha in alphas:
    tr_accs, te_accs = [], []
    for seed in range(N_TRIALS):
        Xtr, ytr = make_train(seed)
        tr, te, _ = fit_eval(Xtr, ytr, Xte, yte, alpha=alpha, seed=seed)
        tr_accs.append(tr)
        te_accs.append(te)
    tr_accs, te_accs = np.array(tr_accs), np.array(te_accs)
    gap = tr_accs - te_accs
    alpha_results.append((alpha, tr_accs.mean(), te_accs.mean(), gap.mean()))
    print(f"alpha={alpha:.6f} train_acc={tr_accs.mean():.4f} test_acc={te_accs.mean():.4f} gap={gap.mean():.4f}", flush=True)

best_alpha_row = max(alpha_results, key=lambda r: r[2])
print(f"\nbest alpha={best_alpha_row[0]:.6f} test_acc={best_alpha_row[2]:.4f} gap={best_alpha_row[3]:.4f}", flush=True)
RESULTS['reg_only'] = dict(alpha=best_alpha_row[0], train_acc=best_alpha_row[1],
                            test_acc=best_alpha_row[2], gap=best_alpha_row[3])
RESULTS['alpha_sweep'] = alpha_results

# ---------------------------------------------------------------
# (c) augmentation only: sweep magnitude (jitter sigma + proportional rotation) at fixed k,
#     and sweep multiplier k at fixed magnitude. alpha stays at baseline (no reg).
# ---------------------------------------------------------------
print("\n=== (c) augmentation only: magnitude sweep (k=10 fixed) ===", flush=True)
K_FIXED = 10
magnitudes = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
mag_results = []
for m in magnitudes:
    jitter_sigma = m * 0.3       # at m=1, sigma=0.3 (comparable to data noise 0.25)
    rot_max_deg = m * 20.0       # at m=1, rotation up to +-20 degrees
    tr_accs, te_accs = [], []
    for seed in range(N_TRIALS):
        Xtr, ytr = make_train(seed)
        rng = np.random.default_rng(seed + 10000)
        Xaug, yaug = augment(Xtr, ytr, K_FIXED, jitter_sigma, rot_max_deg, rng)
        tr, te, clf = fit_eval(Xaug, yaug, Xte, yte, alpha=1e-8, seed=seed)
        tr_orig = accuracy_score(ytr, clf.predict(Xtr))
        tr_accs.append(tr_orig)
        te_accs.append(te)
    tr_accs, te_accs = np.array(tr_accs), np.array(te_accs)
    gap = tr_accs - te_accs
    mag_results.append((m, jitter_sigma, rot_max_deg, tr_accs.mean(), te_accs.mean(), gap.mean()))
    print(f"m={m:.2f} sigma={jitter_sigma:.3f} rot={rot_max_deg:.1f}deg "
          f"train_acc={tr_accs.mean():.4f} test_acc={te_accs.mean():.4f} gap={gap.mean():.4f}", flush=True)

best_mag_row = max(mag_results, key=lambda r: r[4])
print(f"\nbest magnitude m={best_mag_row[0]:.2f} test_acc={best_mag_row[4]:.4f} gap={best_mag_row[5]:.4f}", flush=True)
RESULTS['magnitude_sweep'] = mag_results

print("\n=== (c) augmentation only: multiplier k sweep (magnitude fixed at best m) ===", flush=True)
best_m = best_mag_row[0]
best_sigma = best_mag_row[1]
best_rot = best_mag_row[2]
ks = [0, 1, 2, 3, 5, 10, 20, 40]
k_results = []
for k in ks:
    tr_accs, te_accs = [], []
    for seed in range(N_TRIALS):
        Xtr, ytr = make_train(seed)
        rng = np.random.default_rng(seed + 20000)
        Xaug, yaug = augment(Xtr, ytr, k, best_sigma, best_rot, rng)
        tr, te, clf = fit_eval(Xaug, yaug, Xte, yte, alpha=1e-8, seed=seed)
        tr_orig = accuracy_score(ytr, clf.predict(Xtr))
        tr_accs.append(tr_orig)
        te_accs.append(te)
    tr_accs, te_accs = np.array(tr_accs), np.array(te_accs)
    gap = tr_accs - te_accs
    k_results.append((k, tr_accs.mean(), te_accs.mean(), gap.mean()))
    print(f"k={k:3d} train_acc={tr_accs.mean():.4f} test_acc={te_accs.mean():.4f} gap={gap.mean():.4f}", flush=True)

best_k_row = max(k_results, key=lambda r: r[2])
print(f"\nbest k={best_k_row[0]} test_acc={best_k_row[2]:.4f} gap={best_k_row[3]:.4f}", flush=True)
RESULTS['k_sweep'] = k_results
RESULTS['aug_only'] = dict(k=best_k_row[0], m=best_m, sigma=best_sigma, rot=best_rot,
                            train_acc=best_k_row[1], test_acc=best_k_row[2], gap=best_k_row[3])

# ---------------------------------------------------------------
# (d) both combined: best augmentation + alpha sweep on top
# ---------------------------------------------------------------
print("\n=== (d) both combined: best augmentation + alpha sweep ===", flush=True)
best_k = best_k_row[0]
combo_results = []
for alpha in alphas:
    tr_accs, te_accs = [], []
    for seed in range(N_TRIALS):
        Xtr, ytr = make_train(seed)
        rng = np.random.default_rng(seed + 30000)
        Xaug, yaug = augment(Xtr, ytr, best_k, best_sigma, best_rot, rng)
        tr, te, clf = fit_eval(Xaug, yaug, Xte, yte, alpha=alpha, seed=seed)
        tr_orig = accuracy_score(ytr, clf.predict(Xtr))
        tr_accs.append(tr_orig)
        te_accs.append(te)
    tr_accs, te_accs = np.array(tr_accs), np.array(te_accs)
    gap = tr_accs - te_accs
    combo_results.append((alpha, tr_accs.mean(), te_accs.mean(), gap.mean()))
    print(f"alpha={alpha:.6f} train_acc={tr_accs.mean():.4f} test_acc={te_accs.mean():.4f} gap={gap.mean():.4f}", flush=True)

best_combo_row = max(combo_results, key=lambda r: r[2])
print(f"\nbest combo alpha={best_combo_row[0]:.6f} test_acc={best_combo_row[2]:.4f} gap={best_combo_row[3]:.4f}", flush=True)
RESULTS['combo'] = dict(k=best_k, alpha=best_combo_row[0], train_acc=best_combo_row[1],
                         test_acc=best_combo_row[2], gap=best_combo_row[3])
RESULTS['combo_sweep'] = combo_results

# ---------------------------------------------------------------
# Summary of 4 conditions
# ---------------------------------------------------------------
print("\n=== SUMMARY: 4 conditions ===", flush=True)
for name in ['baseline', 'reg_only', 'aug_only', 'combo']:
    r = RESULTS[name]
    print(name, r, flush=True)

with open('results.json', 'w') as f:
    json.dump(RESULTS, f, indent=2, default=float)

print("\nsaved results.json", flush=True)
