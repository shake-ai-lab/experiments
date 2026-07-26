import numpy as np
from sklearn.datasets import make_moons

rng = np.random.default_rng(0)

# ---------------- target data: "two moons" shape, scaled/centered ----------------
X0, _ = make_moons(n_samples=4000, noise=0.05, random_state=0)
X0 = (X0 - X0.mean(axis=0)) / X0.std(axis=0) * 1.6
np.save("C:/tmp/diffusion-experiment/X0.npy", X0)

# ---------------- DDPM noise schedule (linear beta schedule) ----------------
T = 100
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)
np.save("C:/tmp/diffusion-experiment/alpha_bars.npy", alpha_bars)
np.save("C:/tmp/diffusion-experiment/betas.npy", betas)

def q_sample(x0, t_idx, noise):
    ab = alpha_bars[t_idx][:, None]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise

# sinusoidal time embedding, same construction as the positional-encoding post
def time_embedding(t_idx, dim=8):
    t = t_idx.astype(np.float64)[:, None]
    i = np.arange(dim)[None, :]
    angle_rates = 1.0 / np.power(1000, (2 * (i // 2)) / dim)
    angles = t * angle_rates
    emb = np.zeros((len(t_idx), dim))
    emb[:, 0::2] = np.sin(angles[:, 0::2])
    emb[:, 1::2] = np.cos(angles[:, 1::2])
    return emb

TIME_DIM = 8
IN_DIM = 2 + TIME_DIM
HIDDEN = 256
OUT_DIM = 2

# ---------------- small MLP, manual forward/backward (numpy only) ----------------
init_scale1 = np.sqrt(2.0 / IN_DIM)
init_scale2 = np.sqrt(2.0 / HIDDEN)
W1 = rng.normal(0, init_scale1, size=(IN_DIM, HIDDEN))
b1 = np.zeros(HIDDEN)
W2 = rng.normal(0, init_scale2, size=(HIDDEN, OUT_DIM))
b2 = np.zeros(OUT_DIM)

def forward(x, temb):
    inp = np.concatenate([x, temb], axis=1)
    z1 = inp @ W1 + b1
    a1 = np.maximum(z1, 0)  # ReLU
    out = a1 @ W2 + b2
    cache = (inp, z1, a1)
    return out, cache

def backward(cache, d_out):
    inp, z1, a1 = cache
    dW2 = a1.T @ d_out
    db2 = d_out.sum(axis=0)
    da1 = d_out @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = inp.T @ dz1
    db1 = dz1.sum(axis=0)
    return dW1, db1, dW2, db2

# Adam optimizer state (same family used in the optimizer-race / swiss-roll posts)
params = [W1, b1, W2, b2]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
b1_, b2_, eps, lr = 0.9, 0.999, 1e-8, 1e-3

n_iters = 20000
batch_size = 256
losses = []
for it in range(1, n_iters + 1):
    idx = rng.integers(0, len(X0), size=batch_size)
    x0 = X0[idx]
    t_idx = rng.integers(0, T, size=batch_size)
    noise = rng.normal(0, 1, size=x0.shape)
    xt = q_sample(x0, t_idx, noise)
    temb = time_embedding(t_idx, TIME_DIM)

    pred_noise, cache = forward(xt, temb)
    diff = pred_noise - noise
    loss = np.mean(diff ** 2)
    d_out = 2 * diff / diff.shape[0]

    cur_lr = lr if it < n_iters * 0.6 else lr * 0.3
    grads = backward(cache, d_out)
    for p, g, mi, vi in zip(params, grads, m, v):
        mi[:] = b1_ * mi + (1 - b1_) * g
        vi[:] = b2_ * vi + (1 - b2_) * g ** 2
        m_hat = mi / (1 - b1_ ** it)
        v_hat = vi / (1 - b2_ ** it)
        p -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

    losses.append(loss)
    if it % 500 == 0:
        print(f"iter {it:5d}  loss={np.mean(losses[-500:]):.4f}")

np.save("C:/tmp/diffusion-experiment/losses.npy", np.array(losses))
np.savez("C:/tmp/diffusion-experiment/model.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("training done, final loss (last 200 iters avg):", np.mean(losses[-200:]))
