"""VAE (Variational Autoencoder) をnumpyだけで一から実装し、
diffusion・GAN・Flowと全く同じtwo moonsデータで学習する。
エンコーダ -> mu/logvar -> reparameterization trick -> デコーダ、
順伝播・逆伝播とも手書き。Adamも過去記事と同じ手書き実装。"""
import numpy as np
from sklearn.datasets import make_moons
import os

os.makedirs("C:/tmp/vae-two-moons-experiment", exist_ok=True)

rng = np.random.default_rng(0)

# ---------------- target data: 過去3記事と全く同じtwo moons ----------------
X, y = make_moons(n_samples=4000, noise=0.05, random_state=0)
X = (X - X.mean(axis=0)) / X.std(axis=0) * 1.6
N, D = X.shape
print(f"data: {X.shape}")
np.save("C:/tmp/vae-two-moons-experiment/X.npy", X)
np.save("C:/tmp/vae-two-moons-experiment/y.npy", y)

H = 64        # 隠れ層ユニット数 (GAN/Flowと同規模)
Z_DIM = 2     # 潜在次元 (可視化のため2次元)

def init_linear(fan_in, fan_out):
    s = np.sqrt(1.0 / fan_in)
    return rng.normal(0, s, size=(fan_in, fan_out)), np.zeros(fan_out)

# エンコーダ: x(2) -> h(H, tanh) -> mu(Z_DIM), logvar(Z_DIM)
W1, b1 = init_linear(D, H)
Wmu, bmu = init_linear(H, Z_DIM)
Wlv, blv = init_linear(H, Z_DIM)
# デコーダ: z(Z_DIM) -> h(H, tanh) -> xhat(2, linear)
W3, b3 = init_linear(Z_DIM, H)
W4, b4 = init_linear(H, D)

params = [W1, b1, Wmu, bmu, Wlv, blv, W3, b3, W4, b4]
names = ["W1", "b1", "Wmu", "bmu", "Wlv", "blv", "W3", "b3", "W4", "b4"]


def encode(x):
    h = np.tanh(x @ W1 + b1)
    mu = h @ Wmu + bmu
    logvar = h @ Wlv + blv
    return mu, logvar, (x, h)


def decode(z):
    h2 = np.tanh(z @ W3 + b3)
    xhat = h2 @ W4 + b4
    return xhat, (z, h2)


def forward(x, eps):
    mu, logvar, enc_cache = encode(x)
    std = np.exp(0.5 * logvar)
    z = mu + std * eps
    xhat, dec_cache = decode(z)
    return xhat, mu, logvar, z, enc_cache, dec_cache


def backward(x, xhat, mu, logvar, z, eps, enc_cache, dec_cache):
    n = x.shape[0]
    x_in, h = enc_cache
    z_in, h2 = dec_cache

    # ---- 再構成損失: 0.5*sum((x-xhat)^2), バッチ平均 ----
    d_xhat = (xhat - x) / n
    dW4 = h2.T @ d_xhat
    db4 = d_xhat.sum(0)
    d_h2 = d_xhat @ W4.T
    d_pre2 = d_h2 * (1 - h2 ** 2)
    dW3 = z_in.T @ d_pre2
    db3 = d_pre2.sum(0)
    dz_recon = d_pre2 @ W3.T  # 再構成側からzへの勾配

    # ---- KL項: -0.5*sum(1+logvar-mu^2-exp(logvar)), バッチ平均 ----
    dmu_kl = mu / n
    dlogvar_kl = 0.5 * (np.exp(logvar) - 1) / n

    # ---- reparameterization: z = mu + exp(0.5*logvar)*eps ----
    std = np.exp(0.5 * logvar)
    dmu_total = dz_recon + dmu_kl
    dlogvar_total = dz_recon * 0.5 * std * eps + dlogvar_kl

    dWmu = h.T @ dmu_total
    dbmu = dmu_total.sum(0)
    dWlv = h.T @ dlogvar_total
    dblv = dlogvar_total.sum(0)

    d_h = dmu_total @ Wmu.T + dlogvar_total @ Wlv.T
    d_pre1 = d_h * (1 - h ** 2)
    dW1 = x_in.T @ d_pre1
    db1 = d_pre1.sum(0)

    return [dW1, db1, dWmu, dbmu, dWlv, dblv, dW3, db3, dW4, db4]


# ---- Adam (過去3記事と同じ手書き実装) ----
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
step = 0


def adam_update(grads):
    global step
    step += 1
    for p, g, mi, vi in zip(params, grads, m, v):
        mi[:] = B1 * mi + (1 - B1) * g
        vi[:] = B2 * vi + (1 - B2) * g ** 2
        mhat = mi / (1 - B1 ** step)
        vhat = vi / (1 - B2 ** step)
        p -= LR * mhat / (np.sqrt(vhat) + EPS)


BATCH = 256
ITERS = 20_000
recon_hist, kl_hist = [], []

z_fixed = rng.normal(0, 1, size=(1500, Z_DIM))  # 生成の様子を追う固定ノイズ
snap_at = {0, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 13000, 16000, 20000}
gen_snapshots = {}

for it in range(ITERS + 1):
    if it in snap_at:
        xs, _ = decode(z_fixed)
        gen_snapshots[it] = xs.copy()
    if it == ITERS:
        break
    idx = rng.integers(0, N, BATCH)
    x = X[idx]
    eps = rng.normal(0, 1, size=(BATCH, Z_DIM))
    xhat, mu, logvar, z, enc_cache, dec_cache = forward(x, eps)

    recon_loss = float(0.5 * np.sum((x - xhat) ** 2, axis=1).mean())
    kl_loss = float((-0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=1)).mean())
    recon_hist.append(recon_loss)
    kl_hist.append(kl_loss)

    grads = backward(x, xhat, mu, logvar, z, eps, enc_cache, dec_cache)
    adam_update(grads)

    if it % 1000 == 0:
        print(f"iter {it:6d}  recon={np.mean(recon_hist[-1000:]) if recon_hist else recon_loss:.4f}  kl={np.mean(kl_hist[-1000:]) if kl_hist else kl_loss:.4f}")

print("training complete")

# ---------------- 検証: 再構成誤差 (訓練データ全体、muを使う=決定的) ----------------
mu_all, logvar_all, _ = encode(X)
xhat_all, _ = decode(mu_all)  # 決定的な再構成 (z=mu)
recon_mse = float(np.mean((X - xhat_all) ** 2))
recon_mae = float(np.mean(np.abs(X - xhat_all)))
kl_final = float((-0.5 * np.sum(1 + logvar_all - mu_all ** 2 - np.exp(logvar_all), axis=1)).mean())
print(f"final reconstruction MSE (z=mu): {recon_mse:.6f}")
print(f"final reconstruction MAE (z=mu): {recon_mae:.6f}")
print(f"final KL divergence (mean per sample): {kl_final:.6f}")
print(f"mu range: [{mu_all.min():.3f}, {mu_all.max():.3f}]  std of mu per-dim: {mu_all.std(axis=0)}")
print(f"mean of exp(0.5*logvar) (posterior std): {np.exp(0.5*logvar_all).mean():.4f}")

# ---------------- 生成: z ~ N(0,I) から2000点デコード ----------------
z_gen = rng.normal(0, 1, size=(2000, Z_DIM))
X_gen, _ = decode(z_gen)

np.save("C:/tmp/vae-two-moons-experiment/X_gen.npy", X_gen)
np.save("C:/tmp/vae-two-moons-experiment/X_recon.npy", xhat_all)
np.save("C:/tmp/vae-two-moons-experiment/mu_all.npy", mu_all)
np.save("C:/tmp/vae-two-moons-experiment/logvar_all.npy", logvar_all)
np.save("C:/tmp/vae-two-moons-experiment/recon_hist.npy", np.array(recon_hist))
np.save("C:/tmp/vae-two-moons-experiment/kl_hist.npy", np.array(kl_hist))
np.savez("C:/tmp/vae-two-moons-experiment/gen_snapshots.npz", **{str(k): v for k, v in gen_snapshots.items()})
np.savez(
    "C:/tmp/vae-two-moons-experiment/vae_model.npz",
    **{n: p for n, p in zip(names, params)},
)
with open("C:/tmp/vae-two-moons-experiment/summary.txt", "w") as f:
    f.write(f"recon_mse={recon_mse:.6f}\n")
    f.write(f"recon_mae={recon_mae:.6f}\n")
    f.write(f"kl_final={kl_final:.6f}\n")
    f.write(f"recon_hist_first={recon_hist[0]:.4f} recon_hist_last={np.mean(recon_hist[-500:]):.4f}\n")
    f.write(f"kl_hist_first={kl_hist[0]:.4f} kl_hist_last={np.mean(kl_hist[-500:]):.4f}\n")
    f.write(f"posterior_std_mean={float(np.exp(0.5*logvar_all).mean()):.4f}\n")
print("saved all arrays")
