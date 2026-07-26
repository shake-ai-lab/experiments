import numpy as np
from sklearn.datasets import make_moons

rng = np.random.default_rng(0)

# ---------------- target data: same two-moons shape used in the diffusion-model post ----------------
X_real_full, _ = make_moons(n_samples=4000, noise=0.05, random_state=0)
X_real_full = (X_real_full - X_real_full.mean(axis=0)) / X_real_full.std(axis=0) * 1.6
np.save("C:/tmp/gan-experiment/X_real.npy", X_real_full)

LATENT_DIM = 4
G_HIDDEN = 64
D_HIDDEN = 64
OUT_SCALE = 3.0

def init_linear(fan_in, fan_out, scale=None):
    s = scale if scale is not None else np.sqrt(2.0 / fan_in)
    return rng.normal(0, s, size=(fan_in, fan_out)), np.zeros(fan_out)

Wg1, bg1 = init_linear(LATENT_DIM, G_HIDDEN)
Wg2, bg2 = init_linear(G_HIDDEN, 2)
Wd1, bd1 = init_linear(2, D_HIDDEN)
Wd2, bd2 = init_linear(D_HIDDEN, 1)

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def G_forward(z):
    h1 = relu(z @ Wg1 + bg1)
    z2 = h1 @ Wg2 + bg2
    out = np.tanh(z2) * OUT_SCALE
    return out, (z, h1, z2, out)

def G_backward(cache, d_out):
    z, h1, z2, out = cache
    d_z2 = d_out * OUT_SCALE * (1 - np.tanh(z2) ** 2)
    dWg2 = h1.T @ d_z2
    dbg2 = d_z2.sum(axis=0)
    d_h1 = d_z2 @ Wg2.T
    d_pre1 = d_h1 * (h1 > 0)
    dWg1 = z.T @ d_pre1
    dbg1 = d_pre1.sum(axis=0)
    return dWg1, dbg1, dWg2, dbg2

def D_forward(x):
    h1 = relu(x @ Wd1 + bd1)
    z2 = h1 @ Wd2 + bd2
    prob = sigmoid(z2)
    return prob, (x, h1, z2, prob)

def D_backward(cache, d_prob):
    x, h1, z2, prob = cache
    d_z2 = d_prob * prob * (1 - prob)  # sigmoid derivative
    dWd2 = h1.T @ d_z2
    dbd2 = d_z2.sum(axis=0)
    d_h1 = d_z2 @ Wd2.T
    d_pre1 = d_h1 * (h1 > 0)
    dWd1 = x.T @ d_pre1
    dbd1 = d_pre1.sum(axis=0)
    dx = d_pre1 @ Wd1.T
    return dWd1, dbd1, dWd2, dbd2, dx

# ---------------- Adam optimizer state, same family used in the last 3 posts ----------------
G_params = [Wg1, bg1, Wg2, bg2]
D_params = [Wd1, bd1, Wd2, bd2]
G_m = [np.zeros_like(p) for p in G_params]
G_v = [np.zeros_like(p) for p in G_params]
D_m = [np.zeros_like(p) for p in D_params]
D_v = [np.zeros_like(p) for p in D_params]
beta1, beta2, eps_adam, lr = 0.9, 0.999, 1e-8, 2e-4

def adam_step(params, grads, m, v, t, lr):
    for p, g, mi, vi in zip(params, grads, m, v):
        mi[:] = beta1 * mi + (1 - beta1) * g
        vi[:] = beta2 * vi + (1 - beta2) * g ** 2
        m_hat = mi / (1 - beta1 ** t)
        v_hat = vi / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

BATCH = 256
N_ITERS = 8000
eps = 1e-8

grid_x = np.linspace(-3.2, 3.2, 60)
grid_y = np.linspace(-3.2, 3.2, 60)
GX, GY = np.meshgrid(grid_x, grid_y)
grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

snapshots = {}
d_heatmaps = {}
d_losses, g_losses = [], []

for it in range(1, N_ITERS + 1):
    # ---- train D ----
    real = X_real_full[rng.integers(0, len(X_real_full), size=BATCH)]
    z = rng.normal(0, 1, size=(BATCH, LATENT_DIM))
    fake, _ = G_forward(z)

    prob_real, cache_real = D_forward(real)
    prob_fake, cache_fake = D_forward(fake)
    d_loss = -np.mean(np.log(prob_real + eps)) - np.mean(np.log(1 - prob_fake + eps))

    d_prob_real = (-1 / (prob_real + eps)) / BATCH
    d_prob_fake = (1 / (1 - prob_fake + eps)) / BATCH
    gr = D_backward(cache_real, d_prob_real)
    gf = D_backward(cache_fake, d_prob_fake)
    D_grads = [gr[i] + gf[i] for i in range(4)]
    adam_step(D_params, D_grads, D_m, D_v, it, lr)

    # ---- train G ----
    z = rng.normal(0, 1, size=(BATCH, LATENT_DIM))
    fake2, g_cache = G_forward(z)
    prob_fake2, cache_fake2 = D_forward(fake2)
    g_loss = -np.mean(np.log(prob_fake2 + eps))  # non-saturating generator loss

    d_prob2 = (-1 / (prob_fake2 + eps)) / BATCH
    *_, dx = D_backward(cache_fake2, d_prob2)  # gradient into fake data, D itself NOT updated here
    G_grads = G_backward(g_cache, dx)
    adam_step(G_params, G_grads, G_m, G_v, it, lr)

    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if it % 200 == 0 or it == 1:
        z_show = rng.normal(0, 1, size=(1500, LATENT_DIM))
        samples, _ = G_forward(z_show)
        snapshots[it] = samples.copy()
        grid_prob, _ = D_forward(grid_pts)
        d_heatmaps[it] = grid_prob.reshape(GX.shape).copy()
        print(f"iter {it:5d}  d_loss={np.mean(d_losses[-200:]):.4f}  g_loss={np.mean(g_losses[-200:]):.4f}")

np.savez("C:/tmp/gan-experiment/snapshots.npz", **{str(k): v for k, v in snapshots.items()})
np.savez("C:/tmp/gan-experiment/d_heatmaps.npz", **{str(k): v for k, v in d_heatmaps.items()})
np.save("C:/tmp/gan-experiment/grid_x.npy", grid_x)
np.save("C:/tmp/gan-experiment/grid_y.npy", grid_y)
np.save("C:/tmp/gan-experiment/d_losses.npy", np.array(d_losses))
np.save("C:/tmp/gan-experiment/g_losses.npy", np.array(g_losses))
np.savez("C:/tmp/gan-experiment/G_model.npz", Wg1=Wg1, bg1=bg1, Wg2=Wg2, bg2=bg2)
np.savez("C:/tmp/gan-experiment/D_model.npz", Wd1=Wd1, bd1=bd1, Wd2=Wd2, bd2=bd2)
print("training complete")
