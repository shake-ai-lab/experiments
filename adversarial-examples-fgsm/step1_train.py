"""FGSM (Fast Gradient Sign Method) をnumpyだけで実装し、自前のtwo-moons MLP分類器で
「見た目にはほぼ動いていない微小な摂動で、分類が本当に反転するか」を実測する。

(1) two-moonsで2->H->1のMLP(tanh, sigmoid, BCE, Adam)を高精度まで学習
(2) 損失の「入力X」に対する勾配を逆伝播で計算(重みではない点に注意)
(3) x_adv = x + eps * sign(dL/dx) で全テスト点を摂動
(4) epsilonを細かくスイープし、「もともと正解していた点のうち何%が誤分類に反転するか」を実測
(5) 個別の点を before/after で決定境界に重ねて可視化するためのデータも保存
(6) epsilonスイープをGIF用に、各epsilonでの摂動後の点群+反転数を保存
"""
import numpy as np
from sklearn.datasets import make_moons

OUTDIR = "C:/tmp/adversarial-examples-fgsm"
rng_global = np.random.default_rng(0)

# ---------------- data: two-moons, standardized ----------------
X, y = make_moons(n_samples=1000, noise=0.15, random_state=0)
mu, sd = X.mean(axis=0), X.std(axis=0)
X = (X - mu) / sd
y = y.astype(np.float64)

perm = rng_global.permutation(len(X))
X, y = X[perm], y[perm]
n_train = 700
X_tr, y_tr = X[:n_train], y[:n_train]
X_te, y_te = X[n_train:], y[n_train:]
print(f"train={len(X_tr)} test={len(X_te)}")
print(f"data scale: std per dim = {X.std(axis=0)}, overall std = {X.std():.4f}")

# grid for decision boundary visualization
gx, gy = np.meshgrid(np.linspace(-3.0, 3.0, 220), np.linspace(-3.0, 3.0, 220))
grid = np.column_stack([gx.ravel(), gy.ravel()])


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


def forward(params, X):
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1
    h1 = np.tanh(z1)
    z2 = h1 @ W2 + b2
    out = sigmoid(z2).ravel()
    cache = (X, z1, h1, z2)
    return out, cache


def backward_params(params, cache, y_true, out):
    """通常のパラメータ勾配 (学習用)"""
    W1, b1, W2, b2 = params
    X, z1, h1, z2 = cache
    n = X.shape[0]
    d_z2 = (out - y_true).reshape(-1, 1) / n
    dW2 = h1.T @ d_z2
    db2 = d_z2.sum(axis=0)
    d_h1 = d_z2 @ W2.T
    d_z1 = d_h1 * (1 - np.tanh(z1) ** 2)
    dW1 = X.T @ d_z1
    db1 = d_z1.sum(axis=0)
    return [dW1, db1, dW2, db2]


def grad_wrt_input(params, cache, y_true, out):
    """FGSMの核心: 重みではなく「入力X」についての dL/dX を計算する。
    BCE損失 L = -[y*log(p) + (1-y)*log(1-p)], p=sigmoid(z2) のとき
    dL/dz2 = (p - y) (バッチ平均ではなく、各点ごとの勾配をそのまま使う=摂動は点ごと)
    """
    W1, b1, W2, b2 = params
    X, z1, h1, z2 = cache
    d_z2 = (out - y_true).reshape(-1, 1)          # 点ごと、平均しない
    d_h1 = d_z2 @ W2.T
    d_z1 = d_h1 * (1 - np.tanh(z1) ** 2)
    d_X = d_z1 @ W1.T                               # (n, 2) dL/dX
    return d_X


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


H = 24
EPOCHS = 4000
LR = 0.03
params = init_net(H, seed=42)
m, v = make_adam(params)
for t in range(1, EPOCHS + 1):
    out, cache = forward(params, X_tr)
    grads = backward_params(params, cache, y_tr, out)
    adam_step(params, grads, m, v, t, LR)
    if t % 500 == 0:
        loss = bce(y_tr, out)
        acc = ((out > 0.5) == y_tr).mean()
        print(f"epoch {t}: train loss {loss:.4f} acc {acc:.4f}")

out_te, _ = forward(params, X_te)
test_acc = ((out_te > 0.5) == y_te).mean()
print(f"FINAL test acc: {test_acc:.4f}")

n_params = sum(p.size for p in params)
print(f"model: 2 -> {H}(tanh) -> 1(sigmoid), total params={n_params}")

# decision boundary on grid
grid_out, _ = forward(params, grid)

# =========================================================================
# FGSM on the TEST set
# =========================================================================
out_te, cache_te = forward(params, X_te)
pred_te = (out_te > 0.5).astype(np.float64)
correct_mask = (pred_te == y_te)
n_correct = correct_mask.sum()
print(f"correctly classified test points (attack pool): {int(n_correct)} / {len(X_te)}")

dX = grad_wrt_input(params, cache_te, y_te, out_te)          # (n_test, 2), points toward INCREASING loss
sign_dX = np.sign(dX)

# epsilon sweep: fine resolution, relative to standardized data scale (overall std ~1)
# range extended until flip-rate visibly saturates (found by first running 0-1.2 and seeing it hadn't converged)
epsilons = np.round(np.arange(0.0, 2.51, 0.01), 3)
flip_rates = []
mean_pred_shift = []
flip_matrix = []  # (n_eps, n_correct_pool) bool, for per-point minimal-flip-epsilon analysis
correct_idx = np.where(correct_mask)[0]
for eps in epsilons:
    X_adv = X_te + eps * sign_dX
    out_adv, _ = forward(params, X_adv)
    pred_adv = (out_adv > 0.5).astype(np.float64)
    flipped = (pred_adv[correct_mask] != y_te[correct_mask])
    flip_rate = flipped.mean()
    flip_rates.append(flip_rate)
    mean_pred_shift.append(np.mean(np.abs(out_adv - out_te)))
    flip_matrix.append(flipped)
flip_rates = np.array(flip_rates)
mean_pred_shift = np.array(mean_pred_shift)
flip_matrix = np.array(flip_matrix)  # (n_eps, n_correct)

# per-point minimal epsilon at which it FIRST flips (and stays flipped is not required, just first crossing)
min_flip_eps = np.full(flip_matrix.shape[1], np.nan)
for j in range(flip_matrix.shape[1]):
    idxs = np.where(flip_matrix[:, j])[0]
    if len(idxs) > 0:
        min_flip_eps[j] = epsilons[idxs[0]]
n_ever_flip = np.isnan(min_flip_eps).sum()
print(f"points that NEVER flip even at eps={epsilons[-1]}: {int(n_ever_flip)} / {len(min_flip_eps)}")
finite_min_eps = min_flip_eps[~np.isnan(min_flip_eps)]
print(f"of points that DO flip: min-flip-eps stats -> min={finite_min_eps.min():.3f} "
      f"median={np.median(finite_min_eps):.3f} max={finite_min_eps.max():.3f}")
for thresh in [0.01, 0.02, 0.05, 0.10]:
    frac = (finite_min_eps <= thresh).mean() if len(finite_min_eps) else float('nan')
    n_below = (finite_min_eps <= thresh).sum()
    print(f"  points flipping at eps<={thresh}: {int(n_below)} / {len(min_flip_eps)} total-pool "
          f"({n_below/len(min_flip_eps)*100:.1f}%)")

# find real threshold epsilons (first crossing), from actual measured curve, no assumptions
def first_cross(rates, eps_arr, thresh):
    idx = np.argmax(rates >= thresh)
    if rates[idx] < thresh:
        return None
    return eps_arr[idx]

eps_10 = first_cross(flip_rates, epsilons, 0.10)
eps_25 = first_cross(flip_rates, epsilons, 0.25)
eps_50 = first_cross(flip_rates, epsilons, 0.50)
eps_75 = first_cross(flip_rates, epsilons, 0.75)
eps_90 = first_cross(flip_rates, epsilons, 0.90)
print(f"flip-rate crosses 10% at eps={eps_10}")
print(f"flip-rate crosses 25% at eps={eps_25}")
print(f"flip-rate crosses 50% at eps={eps_50}")
print(f"flip-rate crosses 75% at eps={eps_75}")
print(f"flip-rate crosses 90% at eps={eps_90}")
print(f"flip-rate at eps=0.05: {flip_rates[np.argmin(np.abs(epsilons-0.05))]:.4f}")
print(f"flip-rate at eps=0.10: {flip_rates[np.argmin(np.abs(epsilons-0.10))]:.4f}")
print(f"flip-rate at eps=0.20: {flip_rates[np.argmin(np.abs(epsilons-0.20))]:.4f}")
print(f"flip-rate at eps=0.30: {flip_rates[np.argmin(np.abs(epsilons-0.30))]:.4f}")
print(f"flip-rate at max eps={epsilons[-1]}: {flip_rates[-1]:.4f}")
print(f"overall data std: {X.std():.4f}  -> eps at 50% flip is {eps_50/X.std()*100 if eps_50 else float('nan'):.1f}% of 1 std" if eps_50 else "no 50% crossing in range")

# =========================================================================
# pick a representative epsilon for the "before/after scatter" static plot:
# use the measured eps where flip-rate first crosses 10% -- still a genuinely
# small perturbation (L2 magnitude eps*sqrt(2), far below the moon-to-moon
# separation), but with a real, non-trivial fraction of points already flipping
# =========================================================================
EPS_SHOWCASE = eps_10
print(f"showcase epsilon (first 10% flip-rate crossing): {EPS_SHOWCASE}")

X_adv_showcase = X_te + EPS_SHOWCASE * sign_dX
out_adv_showcase, _ = forward(params, X_adv_showcase)
pred_adv_showcase = (out_adv_showcase > 0.5).astype(np.float64)
flipped_showcase = correct_mask & (pred_adv_showcase != y_te)
print(f"at showcase eps={EPS_SHOWCASE}: {int(flipped_showcase.sum())} / {int(n_correct)} originally-correct points flipped "
      f"({flipped_showcase.sum()/n_correct*100:.1f}%)")

# a handful of individual points to visualize before/after with arrows: pick some that flip AND some that don't
rng_pick = np.random.default_rng(1)
flip_idx_pool = np.where(flipped_showcase)[0]
stay_idx_pool = np.where(correct_mask & ~flipped_showcase)[0]
n_show_flip = min(6, len(flip_idx_pool))
n_show_stay = min(4, len(stay_idx_pool))
show_flip_idx = rng_pick.choice(flip_idx_pool, size=n_show_flip, replace=False)
show_stay_idx = rng_pick.choice(stay_idx_pool, size=n_show_stay, replace=False)
show_idx = np.concatenate([show_flip_idx, show_stay_idx])

# =========================================================================
# save everything needed for step2 visuals
# =========================================================================
np.savez(
    f"{OUTDIR}/results.npz",
    gx=gx, gy=gy, grid_out=grid_out,
    X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
    out_te=out_te, pred_te=pred_te, correct_mask=correct_mask,
    dX=dX, sign_dX=sign_dX,
    epsilons=epsilons, flip_rates=flip_rates, mean_pred_shift=mean_pred_shift,
    EPS_SHOWCASE=EPS_SHOWCASE, X_adv_showcase=X_adv_showcase, out_adv_showcase=out_adv_showcase,
    flipped_showcase=flipped_showcase,
    show_idx=show_idx, show_flip_idx=show_flip_idx, show_stay_idx=show_stay_idx,
    test_acc=test_acc, n_correct=n_correct, H=H, n_params=n_params,
    eps_10=eps_10 if eps_10 is not None else -1,
    eps_25=eps_25 if eps_25 is not None else -1,
    eps_50=eps_50 if eps_50 is not None else -1,
    eps_75=eps_75 if eps_75 is not None else -1,
    eps_90=eps_90 if eps_90 is not None else -1,
    data_std=X.std(),
    min_flip_eps=min_flip_eps, correct_idx=correct_idx,
)
print("saved results.npz")
print("ALL DONE")
