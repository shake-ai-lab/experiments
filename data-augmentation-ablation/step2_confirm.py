"""
Confirmatory run: fixed best hyperparameters found by the sweep, but with
N_TRIALS=60 and PAIRED seeds across the 4 conditions (same 15-point training
draw is reused across baseline/reg/aug/combo within each trial), so we can
directly compare condition-by-condition on the same data and report std.
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
N_TRIALS = 60

BEST_ALPHA_REG = 0.1873817422860383
BEST_K = 40
BEST_SIGMA = 0.12
BEST_ROT = 8.0
BEST_ALPHA_COMBO = 1.5199110829529332


def make_train(seed):
    return make_moons(n_samples=N_TRAIN, noise=NOISE, random_state=seed)


def make_test(seed=999):
    return make_moons(n_samples=N_TEST, noise=NOISE, random_state=seed)


def fit_eval(Xtr, ytr, Xte, yte, alpha, seed):
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, alpha=alpha, max_iter=MAX_ITER,
                         solver='lbfgs', random_state=seed)
    clf.fit(Xtr, ytr)
    return clf


def augment(X, y, k, jitter_sigma, rot_max_deg, rng):
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

records = {name: {'train': [], 'test': []} for name in ['baseline', 'reg_only', 'aug_only', 'combo']}

for seed in range(N_TRIALS):
    Xtr, ytr = make_train(seed)

    # baseline
    clf = fit_eval(Xtr, ytr, Xte, yte, alpha=1e-8, seed=seed)
    records['baseline']['train'].append(accuracy_score(ytr, clf.predict(Xtr)))
    records['baseline']['test'].append(accuracy_score(yte, clf.predict(Xte)))

    # reg only
    clf = fit_eval(Xtr, ytr, Xte, yte, alpha=BEST_ALPHA_REG, seed=seed)
    records['reg_only']['train'].append(accuracy_score(ytr, clf.predict(Xtr)))
    records['reg_only']['test'].append(accuracy_score(yte, clf.predict(Xte)))

    # aug only
    rng = np.random.default_rng(seed + 20000)
    Xaug, yaug = augment(Xtr, ytr, BEST_K, BEST_SIGMA, BEST_ROT, rng)
    clf = fit_eval(Xaug, yaug, Xte, yte, alpha=1e-8, seed=seed)
    records['aug_only']['train'].append(accuracy_score(ytr, clf.predict(Xtr)))
    records['aug_only']['test'].append(accuracy_score(yte, clf.predict(Xte)))

    # combo
    rng = np.random.default_rng(seed + 30000)
    Xaug, yaug = augment(Xtr, ytr, BEST_K, BEST_SIGMA, BEST_ROT, rng)
    clf = fit_eval(Xaug, yaug, Xte, yte, alpha=BEST_ALPHA_COMBO, seed=seed)
    records['combo']['train'].append(accuracy_score(ytr, clf.predict(Xtr)))
    records['combo']['test'].append(accuracy_score(yte, clf.predict(Xte)))

summary = {}
print(f"N_TRIALS={N_TRIALS} (paired seeds across conditions)\n")
for name in ['baseline', 'reg_only', 'aug_only', 'combo']:
    tr = np.array(records[name]['train'])
    te = np.array(records[name]['test'])
    gap = tr - te
    summary[name] = dict(train_mean=tr.mean(), train_std=tr.std(),
                          test_mean=te.mean(), test_std=te.std(),
                          gap_mean=gap.mean(), gap_std=gap.std())
    print(f"{name:10s} train={tr.mean():.4f}+-{tr.std():.4f}  "
          f"test={te.mean():.4f}+-{te.std():.4f}  gap={gap.mean():.4f}+-{gap.std():.4f}")

# paired comparisons (same seeds)
print("\npaired differences (test_acc), mean +- std of per-seed diff:")
base_te = np.array(records['baseline']['test'])
reg_te = np.array(records['reg_only']['test'])
aug_te = np.array(records['aug_only']['test'])
combo_te = np.array(records['combo']['test'])

for name_a, arr_a, name_b, arr_b in [
    ('reg_only', reg_te, 'baseline', base_te),
    ('aug_only', aug_te, 'baseline', base_te),
    ('combo', combo_te, 'baseline', base_te),
    ('reg_only', reg_te, 'aug_only', aug_te),
    ('reg_only', reg_te, 'combo', combo_te),
    ('combo', combo_te, 'aug_only', aug_te),
]:
    diff = arr_a - arr_b
    win_rate = (diff > 0).mean()
    print(f"{name_a} - {name_b}: mean_diff={diff.mean():.4f} std={diff.std():.4f} "
          f"win_rate({name_a}>{name_b})={win_rate:.2f}")

with open('final_confirm_results.json', 'w') as f:
    json.dump(summary, f, indent=2, default=float)
print("\nsaved final_confirm_results.json")
