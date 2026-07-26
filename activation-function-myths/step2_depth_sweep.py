"""ReLU・sigmoid・tanh・GELUの4活性化関数について、
(A) 初期化直後の勾配の大きさが深さとともにどう変わるか(勾配消失)
(B) 実際に学習させたときのtwo-moons最終テスト精度が深さとともにどう変わるか
(C) 固定の深さでの決定境界の「形」そのもの(区分線形 vs 滑らか)
を比較する。「ReLUはsigmoidの簡略化」という俗説に反し、勾配の伝わり方も
決定境界の形も、ReLU/GELU組とsigmoid/tanh組とで全く違う振る舞いを見せることを確認する。
"""
import time
import numpy as np
from scipy.special import erf
from sklearn.datasets import make_moons

OUTDIR = "C:/tmp/activation-function-myths"
WIDTH = 32
DEPTHS = [1, 2, 4, 8, 12, 16, 24]
EPOCHS = 3000
LR = 0.01
N_SAMPLES = 1000
N_TRAIN = 700


# ---------------- activations ----------------
def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


def gelu(x):
    return x * 0.5 * (1 + erf(x / np.sqrt(2)))


def d_gelu(x):
    phi = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)  # standard normal pdf
    Phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    return Phi + x * phi


ACTIVATIONS = {
    "ReLU": (relu, d_relu, "he"),
    "GELU": (gelu, d_gelu, "he"),
    "tanh": (tanh, d_tanh, "xavier"),
    "sigmoid": (sigmoid, d_sigmoid, "xavier"),
}


def make_net(depth, width, input_dim, init_scheme, seed):
    rng = np.random.default_rng(seed)
    Ws, bs = [], []
    dims = [input_dim] + [width] * depth + [1]
    for l in range(len(dims) - 1):
        fan_in, fan_out = dims[l], dims[l + 1]
        if init_scheme == "he":
            std = np.sqrt(2.0 / fan_in)
        else:  # xavier
            std = np.sqrt(2.0 / (fan_in + fan_out))
        Ws.append(rng.normal(0, std, size=(fan_in, fan_out)))
        bs.append(np.zeros(fan_out))
    return Ws, bs


def forward(Ws, bs, X, act_fn):
    a = X
    cache = [a]
    zs = []
    for l in range(len(Ws) - 1):
        z = a @ Ws[l] + bs[l]
        a = act_fn(z)
        zs.append(z)
        cache.append(a)
    z_out = a @ Ws[-1] + bs[-1]
    out = sigmoid(z_out).ravel()
    zs.append(z_out)
    return out, zs, cache


def backward(Ws, bs, zs, cache, y_true, out, act_fn, d_act_fn):
    n = cache[0].shape[0]
    dWs = [None] * len(Ws)
    dbs = [None] * len(bs)
    d_z = (out - y_true).reshape(-1, 1) / n  # dL/dz_out (BCE + sigmoid combined grad)
    for l in range(len(Ws) - 1, -1, -1):
        a_prev = cache[l]
        dWs[l] = a_prev.T @ d_z
        dbs[l] = d_z.sum(axis=0)
        if l > 0:
            d_a_prev = d_z @ Ws[l].T
            d_z = d_a_prev * d_act_fn(zs[l - 1])
    return dWs, dbs, zs  # zs kept for first-layer grad magnitude measurement


def make_adam(Ws, bs):
    mW = [np.zeros_like(w) for w in Ws]
    vW = [np.zeros_like(w) for w in Ws]
    mb = [np.zeros_like(b) for b in bs]
    vb = [np.zeros_like(b) for b in bs]
    return mW, vW, mb, vb


def adam_step(Ws, bs, dWs, dbs, state, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    mW, vW, mb, vb = state
    for l in range(len(Ws)):
        mW[l][:] = beta1 * mW[l] + (1 - beta1) * dWs[l]
        vW[l][:] = beta2 * vW[l] + (1 - beta2) * dWs[l] ** 2
        mb[l][:] = beta1 * mb[l] + (1 - beta1) * dbs[l]
        vb[l][:] = beta2 * vb[l] + (1 - beta2) * dbs[l] ** 2
        mW_hat, vW_hat = mW[l] / (1 - beta1 ** t), vW[l] / (1 - beta2 ** t)
        mb_hat, vb_hat = mb[l] / (1 - beta1 ** t), vb[l] / (1 - beta2 ** t)
        Ws[l] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        bs[l] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


# ---------------- data ----------------
X, y = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=0)
mu, sd = X.mean(axis=0), X.std(axis=0)
X = (X - mu) / sd
y = y.astype(np.float64)
rng0 = np.random.default_rng(0)
perm = rng0.permutation(len(X))
X, y = X[perm], y[perm]
X_tr, y_tr = X[:N_TRAIN], y[:N_TRAIN]
X_te, y_te = X[N_TRAIN:], y[N_TRAIN:]

# ---------------- (A) gradient magnitude at init, vs depth, per activation ----------------
grad_results = {name: [] for name in ACTIVATIONS}
for name, (act_fn, d_act_fn, init_scheme) in ACTIVATIONS.items():
    for depth in DEPTHS:
        Ws, bs = make_net(depth, WIDTH, 2, init_scheme, seed=42)
        out, zs, cache = forward(Ws, bs, X_tr, act_fn)
        dWs, dbs, _ = backward(Ws, bs, zs, cache, y_tr, out, act_fn, d_act_fn)
        first_layer_grad_mag = np.mean(np.abs(dWs[0]))
        grad_results[name].append(first_layer_grad_mag)
    print(f"[grad@init] {name}: " + " ".join(f"{g:.2e}" for g in grad_results[name]))

# ---------------- (B) final test accuracy vs depth, per activation (actually trained) ----------------
acc_results = {name: [] for name in ACTIVATIONS}
t0 = time.time()
for name, (act_fn, d_act_fn, init_scheme) in ACTIVATIONS.items():
    for depth in DEPTHS:
        Ws, bs = make_net(depth, WIDTH, 2, init_scheme, seed=42)
        state = make_adam(Ws, bs)
        for t in range(1, EPOCHS + 1):
            out, zs, cache = forward(Ws, bs, X_tr, act_fn)
            dWs, dbs, _ = backward(Ws, bs, zs, cache, y_tr, out, act_fn, d_act_fn)
            adam_step(Ws, bs, dWs, dbs, state, t, LR)
        out_te, _, _ = forward(Ws, bs, X_te, act_fn)
        acc = ((out_te > 0.5) == y_te).mean()
        acc_results[name].append(acc)
    print(f"[test acc]  {name}: " + " ".join(f"{a:.3f}" for a in acc_results[name]))
print(f"training all configs took {time.time()-t0:.1f}s")

# ---------------- (C) decision boundaries at a fixed depth, per activation ----------------
BOUNDARY_DEPTH = 6
gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 140), np.linspace(-2.6, 2.6, 140))
grid = np.column_stack([gx.ravel(), gy.ravel()])
boundary_preds = {}
boundary_accs = {}
for name, (act_fn, d_act_fn, init_scheme) in ACTIVATIONS.items():
    Ws, bs = make_net(BOUNDARY_DEPTH, WIDTH, 2, init_scheme, seed=42)
    state = make_adam(Ws, bs)
    for t in range(1, EPOCHS + 1):
        out, zs, cache = forward(Ws, bs, X_tr, act_fn)
        dWs, dbs, _ = backward(Ws, bs, zs, cache, y_tr, out, act_fn, d_act_fn)
        adam_step(Ws, bs, dWs, dbs, state, t, LR)
    out_te, _, _ = forward(Ws, bs, X_te, act_fn)
    boundary_accs[name] = ((out_te > 0.5) == y_te).mean()
    grid_pred, _, _ = forward(Ws, bs, grid, act_fn)
    boundary_preds[name] = grid_pred.reshape(gx.shape)
    print(f"[boundary depth={BOUNDARY_DEPTH}] {name}: test_acc={boundary_accs[name]:.3f}")

np.savez(
    f"{OUTDIR}/depth_sweep_results.npz",
    depths=np.array(DEPTHS),
    names=np.array(list(ACTIVATIONS.keys())),
    grad_ReLU=np.array(grad_results["ReLU"]), grad_GELU=np.array(grad_results["GELU"]),
    grad_tanh=np.array(grad_results["tanh"]), grad_sigmoid=np.array(grad_results["sigmoid"]),
    acc_ReLU=np.array(acc_results["ReLU"]), acc_GELU=np.array(acc_results["GELU"]),
    acc_tanh=np.array(acc_results["tanh"]), acc_sigmoid=np.array(acc_results["sigmoid"]),
    gx=gx, gy=gy, X_tr=X_tr, y_tr=y_tr,
    boundary_ReLU=boundary_preds["ReLU"], boundary_GELU=boundary_preds["GELU"],
    boundary_tanh=boundary_preds["tanh"], boundary_sigmoid=boundary_preds["sigmoid"],
    boundary_acc_ReLU=boundary_accs["ReLU"], boundary_acc_GELU=boundary_accs["GELU"],
    boundary_acc_tanh=boundary_accs["tanh"], boundary_acc_sigmoid=boundary_accs["sigmoid"],
    boundary_depth=BOUNDARY_DEPTH,
)
print("saved depth_sweep_results.npz")
