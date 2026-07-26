"""Dropoutは「暗黙のアンサンブル」なのか、実際に確かめる。
(1) 小さいMLP x M=10 を独立に学習した explicit ensemble
(2) 同じ総パラメータ数の大きいMLP1つを dropout で学習し、MC-dropout (確率的な複数回のforward) で予測
(3) 両者のテスト精度、予測の平均・分散のヒートマップ、点ごとの分散の相関を比較
(4) おまけ: dropout率を変えたときに相関がどう動くかのスイープ (plotlyスライダー用)
"""
import numpy as np
from sklearn.datasets import make_moons

OUTDIR = "C:/tmp/dropout-implicit-ensemble-test-experiment"
rng_global = np.random.default_rng(0)

# ---------------- data: two-moons, noisy enough for a genuinely non-trivial boundary ----------------
X, y = make_moons(n_samples=1000, noise=0.28, random_state=0)
# standardize
mu, sd = X.mean(axis=0), X.std(axis=0)
X = (X - mu) / sd
y = y.astype(np.float64)

perm = rng_global.permutation(len(X))
X, y = X[perm], y[perm]
n_train = 700
X_tr, y_tr = X[:n_train], y[:n_train]
X_te, y_te = X[n_train:], y[n_train:]
print(f"train={len(X_tr)} test={len(X_te)}")

np.save(f"{OUTDIR}/X_tr.npy", X_tr)
np.save(f"{OUTDIR}/y_tr.npy", y_tr)
np.save(f"{OUTDIR}/X_te.npy", X_te)
np.save(f"{OUTDIR}/y_te.npy", y_te)

# grid over the input space, for heatmaps + per-point variance comparison
gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 70), np.linspace(-2.6, 2.6, 70))
grid = np.column_stack([gx.ravel(), gy.ravel()])
np.save(f"{OUTDIR}/grid_x.npy", gx)
np.save(f"{OUTDIR}/grid_y.npy", gy)
np.save(f"{OUTDIR}/grid.npy", grid)

# ---------------- from-scratch MLP: 2 -> H -> 1, tanh hidden, sigmoid out, BCE loss ----------------
def init_net(H, seed):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, np.sqrt(2.0 / 2), size=(2, H))
    b1 = np.zeros(H)
    W2 = r.normal(0, np.sqrt(2.0 / H), size=(H, 1))
    b2 = np.zeros(1)
    return [W1, b1, W2, b2]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def forward(params, X, drop_p=0.0, train=False, rng=None):
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1
    h1 = np.tanh(z1)
    mask = None
    if drop_p > 0.0:
        if train:
            # inverted dropout: scale kept units so E[activation] matches eval
            mask = (rng.random(h1.shape) > drop_p).astype(np.float64) / (1 - drop_p)
        else:
            # MC-dropout: dropout STAYS ON at test time too (that's the whole point)
            mask = (rng.random(h1.shape) > drop_p).astype(np.float64) / (1 - drop_p)
        h1 = h1 * mask
    z2 = h1 @ W2 + b2
    out = sigmoid(z2).ravel()
    cache = (X, z1, h1, z2, mask)
    return out, cache


def backward(params, cache, y_true, out):
    W1, b1, W2, b2 = params
    X, z1, h1, z2, mask = cache
    n = X.shape[0]
    d_z2 = (out - y_true).reshape(-1, 1) / n  # BCE + sigmoid grad
    dW2 = h1.T @ d_z2
    db2 = d_z2.sum(axis=0)
    d_h1 = d_z2 @ W2.T
    if mask is not None:
        d_h1 = d_h1 * mask
    d_z1 = d_h1 * (1 - np.tanh(z1) ** 2)
    dW1 = X.T @ d_z1
    db1 = d_z1.sum(axis=0)
    return [dW1, db1, dW2, db2]


def make_adam(params):
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    return m, v


def adam_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for p, g, mi, vi in zip(params, grads, m, v):
        mi[:] = beta1 * mi + (1 - beta1) * g
        vi[:] = beta2 * vi + (1 - beta2) * g ** 2
        m_hat = mi / (1 - beta1 ** t)
        v_hat = vi / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def bce(y_true, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def train_net(H, seed, drop_p=0.0, epochs=4000, lr=0.03):
    params = init_net(H, seed)
    m, v = make_adam(params)
    rng = np.random.default_rng(seed + 10_000)
    for t in range(1, epochs + 1):
        out, cache = forward(params, X_tr, drop_p=drop_p, train=True, rng=rng)
        grads = backward(params, cache, y_tr, out)
        adam_step(params, grads, m, v, t, lr)
    return params


def predict_once(params, Xq, drop_p=0.0, train_time_dropout=False, rng=None):
    out, _ = forward(params, Xq, drop_p=drop_p, train=(train_time_dropout), rng=rng)
    return out


# =========================================================================
# (1) EXPLICIT ENSEMBLE: M=10 independent small MLPs, H=8 hidden units each
# =========================================================================
M = 10
H_SMALL = 8
EPOCHS = 4000
LR = 0.03

ensemble_params = []
for i in range(M):
    p = train_net(H_SMALL, seed=100 + i, drop_p=0.0, epochs=EPOCHS, lr=LR)
    ensemble_params.append(p)
    acc = ((predict_once(p, X_te) > 0.5) == y_te).mean()
    print(f"ensemble member {i}: test acc {acc:.4f}")

n_params_small = sum(p.size for p in ensemble_params[0])
print(f"params per small net: {n_params_small}, total ensemble params: {n_params_small * M}")

# ensemble predictions per member, on test set and on grid
ens_test_members = np.stack([predict_once(p, X_te) for p in ensemble_params])  # (M, n_test)
ens_grid_members = np.stack([predict_once(p, grid) for p in ensemble_params])  # (M, n_grid)

ens_test_mean = ens_test_members.mean(axis=0)
ens_test_acc = ((ens_test_mean > 0.5) == y_te).mean()
print(f"EXPLICIT ENSEMBLE test acc (averaged over {M} members): {ens_test_acc:.4f}")

# =========================================================================
# (2) SINGLE larger MLP w/ dropout, H matched to total ensemble hidden units
# =========================================================================
H_LARGE = H_SMALL * M  # equal total hidden units -> ~matched parameter budget
DROP_P_MAIN = 0.5
T_MC = M  # same number of stochastic passes as ensemble members, for a fair test-time budget too

drop_params = train_net(H_LARGE, seed=999, drop_p=DROP_P_MAIN, epochs=EPOCHS, lr=LR)
n_params_large = sum(p.size for p in drop_params)
print(f"params in single dropout net (H={H_LARGE}): {n_params_large}  "
      f"(vs ensemble total {n_params_small * M}, ratio {n_params_large / (n_params_small*M):.3f})")

mc_rng = np.random.default_rng(777)


def mc_dropout_predict(params, Xq, drop_p, T, rng):
    passes = np.stack([predict_once(params, Xq, drop_p=drop_p, train_time_dropout=True, rng=rng) for _ in range(T)])
    return passes  # (T, n)


drop_test_passes = mc_dropout_predict(drop_params, X_te, DROP_P_MAIN, T_MC, mc_rng)
drop_test_mean = drop_test_passes.mean(axis=0)
drop_test_acc = ((drop_test_mean > 0.5) == y_te).mean()
print(f"MC-DROPOUT ({T_MC} passes, p={DROP_P_MAIN}) test acc: {drop_test_acc:.4f}")

# also: deterministic (dropout off) prediction of the SAME net, for reference
det_out = predict_once(drop_params, X_te, drop_p=0.0, train_time_dropout=False)
det_acc = ((det_out > 0.5) == y_te).mean()
print(f"same net, dropout OFF (plain forward pass) test acc: {det_acc:.4f}")

drop_grid_passes = mc_dropout_predict(drop_params, grid, DROP_P_MAIN, T_MC, mc_rng)

# =========================================================================
# per-point mean / variance on the grid, for heatmaps + scatter + correlation
# =========================================================================
ens_grid_mean = ens_grid_members.mean(axis=0)
ens_grid_var = ens_grid_members.var(axis=0)
drop_grid_mean = drop_grid_passes.mean(axis=0)
drop_grid_var = drop_grid_passes.var(axis=0)

corr = np.corrcoef(ens_grid_var, drop_grid_var)[0, 1]
print(f"Pearson correlation of per-point variance (ensemble vs MC-dropout), grid, n={len(grid)}: {corr:.4f}")

# rank correlation is also informative since variances are skewed
from scipy.stats import spearmanr
rho, _ = spearmanr(ens_grid_var, drop_grid_var)
print(f"Spearman correlation of per-point variance: {rho:.4f}")

print(f"ensemble variance: mean={ens_grid_var.mean():.5f} max={ens_grid_var.max():.5f}")
print(f"MC-dropout variance: mean={drop_grid_var.mean():.5f} max={drop_grid_var.max():.5f}")

np.savez(
    f"{OUTDIR}/results_main.npz",
    gx=gx, gy=gy,
    X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
    ens_grid_mean=ens_grid_mean, ens_grid_var=ens_grid_var,
    drop_grid_mean=drop_grid_mean, drop_grid_var=drop_grid_var,
    ens_test_acc=ens_test_acc, drop_test_acc=drop_test_acc, det_acc=det_acc,
    corr=corr, rho=rho,
    n_params_small=n_params_small, n_params_large=n_params_large, M=M, H_SMALL=H_SMALL, H_LARGE=H_LARGE,
)
print("saved results_main.npz")

# save the individual member/pass grid predictions too, for the "flicker" GIF
np.savez(
    f"{OUTDIR}/individual_frames.npz",
    ens_grid_members=ens_grid_members,   # (M, n_grid)
    drop_grid_passes=drop_grid_passes,   # (T_MC, n_grid)
    gx=gx, gy=gy,
)
print("saved individual_frames.npz")

# =========================================================================
# (3) bonus: dropout-rate sweep, does the match with the ensemble get better/worse?
# =========================================================================
sweep_ps = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
sweep_results = {}
for p_drop in sweep_ps:
    params_p = train_net(H_LARGE, seed=2000 + int(p_drop * 100), drop_p=p_drop, epochs=EPOCHS, lr=LR)
    rng_p = np.random.default_rng(3000 + int(p_drop * 100))
    passes_grid = mc_dropout_predict(params_p, grid, p_drop, T_MC, rng_p)
    passes_test = mc_dropout_predict(params_p, X_te, p_drop, T_MC, rng_p)
    mean_grid = passes_grid.mean(axis=0)
    var_grid = passes_grid.var(axis=0)
    mean_test = passes_test.mean(axis=0)
    acc_p = ((mean_test > 0.5) == y_te).mean()
    corr_p = np.corrcoef(ens_grid_var, var_grid)[0, 1]
    print(f"p={p_drop}: test acc {acc_p:.4f}, variance-corr with ensemble {corr_p:.4f}, "
          f"mean var {var_grid.mean():.5f}")
    sweep_results[p_drop] = dict(mean_grid=mean_grid, var_grid=var_grid, acc=acc_p, corr=corr_p)

np.savez(
    f"{OUTDIR}/sweep_results.npz",
    ps=np.array(sweep_ps),
    means=np.stack([sweep_results[p]["mean_grid"] for p in sweep_ps]),
    vars_=np.stack([sweep_results[p]["var_grid"] for p in sweep_ps]),
    accs=np.array([sweep_results[p]["acc"] for p in sweep_ps]),
    corrs=np.array([sweep_results[p]["corr"] for p in sweep_ps]),
    gx=gx, gy=gy,
)
print("saved sweep_results.npz")
print("ALL DONE")
