import numpy as np

rng = np.random.default_rng(1)

X0 = np.load("C:/tmp/diffusion-experiment/X0.npy")
alpha_bars = np.load("C:/tmp/diffusion-experiment/alpha_bars.npy")
betas = np.load("C:/tmp/diffusion-experiment/betas.npy")
alphas = 1 - betas
T = len(betas)

model = np.load("C:/tmp/diffusion-experiment/model.npz")
W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
TIME_DIM = 8

def time_embedding(t_idx, dim=8):
    t = np.full(1, t_idx, dtype=np.float64)[:, None] if np.isscalar(t_idx) else t_idx.astype(np.float64)[:, None]
    i = np.arange(dim)[None, :]
    angle_rates = 1.0 / np.power(1000, (2 * (i // 2)) / dim)
    angles = t * angle_rates
    emb = np.zeros((t.shape[0], dim))
    emb[:, 0::2] = np.sin(angles[:, 0::2])
    emb[:, 1::2] = np.cos(angles[:, 1::2])
    return emb

def predict_noise(x, t_idx_scalar, n):
    temb = time_embedding(np.full(n, t_idx_scalar), TIME_DIM)
    inp = np.concatenate([x, temb], axis=1)
    a1 = np.maximum(inp @ W1 + b1, 0)
    return a1 @ W2 + b2

n_samples = 1500
x = rng.normal(0, 1, size=(n_samples, 2))  # start from pure noise

snapshots = {T: x.copy()}
for t_idx in reversed(range(T)):
    eps_theta = predict_noise(x, t_idx, n_samples)
    alpha_t = alphas[t_idx]
    alpha_bar_t = alpha_bars[t_idx]
    beta_t = betas[t_idx]
    mean = (1 / np.sqrt(alpha_t)) * (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * eps_theta)
    if t_idx > 0:
        z = rng.normal(0, 1, size=x.shape)
        sigma_t = np.sqrt(beta_t)
        x = mean + sigma_t * z
    else:
        x = mean
    snapshots[t_idx] = x.copy()

np.savez("C:/tmp/diffusion-experiment/samples.npz", **{str(k): v for k, v in snapshots.items()})
print("sampling done. final sample stats: mean=", x.mean(axis=0), "std=", x.std(axis=0))
print("target data stats:   mean=", X0.mean(axis=0), "std=", X0.std(axis=0))
