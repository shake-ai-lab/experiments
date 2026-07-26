"""Normalizing Flow (RealNVP型アフィンカップリング) をnumpyだけで実装し、
diffusion・GANと全く同じtwo moonsデータ (X0.npy) で学習する。
順伝播・逆伝播とも手書き。Adamも過去記事と同じ手書き実装。"""
import numpy as np

rng = np.random.default_rng(0)
X = np.load("C:/tmp/diffusion-experiment/X0.npy")  # (4000,2) mean0 std1.6
N, D = X.shape
print(f"data: {X.shape}")

H = 64  # カップリング層内MLPの隠れユニット
L = 6  # カップリング層の数
S_SCALE = 2.0  # sの暴走防止 (s = S_SCALE*tanh(raw))


def init_layer(seed):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(0, 0.1, (1, H)),  # 入力はマスクされた1次元
        "b1": np.zeros(H),
        "W2": r.normal(0, 0.1, (H, H)),
        "b2": np.zeros(H),
        "W3": np.zeros((H, 2)),  # 出力: [raw_s, t] 初期0 -> 恒等変換から開始
        "b3": np.zeros(2),
    }


layers = [init_layer(i) for i in range(L)]
masks = [i % 2 for i in range(L)]  # 0: x0を条件にx1を変換, 1: 逆


def mlp_forward(p, a):
    h1 = np.tanh(a @ p["W1"] + p["b1"])
    h2 = np.tanh(h1 @ p["W2"] + p["b2"])
    out = h2 @ p["W3"] + p["b3"]
    raw_s, t = out[:, 0:1], out[:, 1:2]
    s = S_SCALE * np.tanh(raw_s)
    return s, t, (a, h1, h2, raw_s)


def mlp_backward(p, cache, ds, dt):
    a, h1, h2, raw_s = cache
    d_raw_s = ds * S_SCALE * (1 - np.tanh(raw_s) ** 2)
    dout = np.concatenate([d_raw_s, dt], axis=1)
    g = {}
    g["W3"] = h2.T @ dout
    g["b3"] = dout.sum(0)
    dh2 = dout @ p["W3"].T * (1 - h2**2)
    g["W2"] = h1.T @ dh2
    g["b2"] = dh2.sum(0)
    dh1 = dh2 @ p["W2"].T * (1 - h1**2)
    g["W1"] = a.T @ dh1
    g["b1"] = dh1.sum(0)
    da = dh1 @ p["W1"].T
    return g, da


def flow_forward(x):
    """x -> z。各層のキャッシュとlogdetを返す"""
    caches = []
    logdet = np.zeros((x.shape[0], 1))
    h = x
    for p, m in zip(layers, masks):
        a = h[:, m : m + 1]  # 条件側 (そのまま通す)
        b = h[:, 1 - m : 2 - m]  # 変換側
        s, t, mc = mlp_forward(p, a)
        b_new = b * np.exp(s) + t
        logdet += s
        h_new = np.empty_like(h)
        h_new[:, m : m + 1] = a
        h_new[:, 1 - m : 2 - m] = b_new
        caches.append((a, b, s, t, mc))
        h = h_new
    return h, logdet, caches


def flow_inverse(z):
    """z -> x (生成方向)"""
    h = z
    stages = [h.copy()]
    for p, m in zip(reversed(layers), reversed(masks)):
        a = h[:, m : m + 1]
        b = h[:, 1 - m : 2 - m]
        s, t, _ = mlp_forward(p, a)
        b_new = (b - t) * np.exp(-s)
        h_new = np.empty_like(h)
        h_new[:, m : m + 1] = a
        h_new[:, 1 - m : 2 - m] = b_new
        h = h_new
        stages.append(h.copy())
    return h, stages


def flow_backward(caches, dz, dlogdet):
    """dL/dz と dL/dlogdet から各層の勾配を計算"""
    grads = [None] * L
    dh = dz
    for i in reversed(range(L)):
        p, m = layers[i], masks[i]
        a, b, s, t, mc = caches[i]
        da_direct = dh[:, m : m + 1]
        db_new = dh[:, 1 - m : 2 - m]
        # b_new = b*exp(s)+t, logdet += s
        ds = db_new * b * np.exp(s) + dlogdet
        dt = db_new
        db = db_new * np.exp(s)
        g, da_mlp = mlp_backward(p, mc, ds, dt)
        grads[i] = g
        dh_prev = np.empty_like(dh)
        dh_prev[:, m : m + 1] = da_direct + da_mlp
        dh_prev[:, 1 - m : 2 - m] = db
        dh = dh_prev
    return grads


# ---- Adam (過去記事と同じ手書き実装) ----
adam_m = [{k: np.zeros_like(v) for k, v in p.items()} for p in layers]
adam_v = [{k: np.zeros_like(v) for k, v in p.items()} for p in layers]
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
step = 0


def adam_update(grads):
    global step
    step += 1
    for p, g, m_, v_ in zip(layers, grads, adam_m, adam_v):
        for k in p:
            m_[k] = B1 * m_[k] + (1 - B1) * g[k]
            v_[k] = B2 * v_[k] + (1 - B2) * g[k] ** 2
            mhat = m_[k] / (1 - B1**step)
            vhat = v_[k] / (1 - B2**step)
            p[k] -= LR * mhat / (np.sqrt(vhat) + EPS)


BATCH = 256
ITERS = 30_000
loss_hist = []
snap_at = {0, 100, 300, 1000, 3000, 10000, 30000}
sample_snaps = {}
z_fixed = rng.normal(0, 1, (1000, 2))
for it in range(ITERS + 1):
    if it in snap_at:
        xs, _ = flow_inverse(z_fixed)
        sample_snaps[it] = xs
    if it == ITERS:
        break
    idx = rng.integers(0, N, BATCH)
    x = X[idx]
    z, logdet, caches = flow_forward(x)
    # NLL = 0.5*||z||^2 + log(2pi) - logdet  (per sample)
    loss = float((0.5 * (z**2).sum(1, keepdims=True) - logdet).mean()) + np.log(2 * np.pi)
    loss_hist.append(loss)
    dz = z / BATCH
    dlogdet = -np.ones((BATCH, 1)) / BATCH
    grads = flow_backward(caches, dz, dlogdet)
    adam_update(grads)
    if it % 3000 == 0:
        print(f"iter {it:6d}: NLL {loss:.4f}")

# ---- 検証 ----
z_all, logdet_all, _ = flow_forward(X)
nll = float((0.5 * (z_all**2).sum(1, keepdims=True) - logdet_all).mean()) + np.log(2 * np.pi)
x_rec, _ = flow_inverse(z_all)
inv_err = float(np.abs(x_rec - X).max())
print(f"final NLL (train set): {nll:.4f}")
print(f"可逆性の再構成誤差 max|x - f^-1(f(x))|: {inv_err:.2e}")

np.savez(
    "flow_model.npz",
    **{f"L{i}_{k}": v for i, p in enumerate(layers) for k, v in p.items()},
    masks=np.array(masks),
    loss_hist=np.array(loss_hist),
)
np.savez("flow_snaps.npz", **{str(k): v for k, v in sample_snaps.items()}, z_fixed=z_fixed)
print("saved flow_model.npz, flow_snaps.npz")
