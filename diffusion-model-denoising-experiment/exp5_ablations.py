"""Ablations for the paid "DDPM complete guide" note article.

Trains the same tiny numpy MLP under different noise schedules / step counts /
time-embedding settings, samples from each, and measures how close the
generated cloud is to the real two-moons data.

Configs:
  A. schedule @ T=100 : linear (blog baseline, alpha_bar_T never reaches 0)
                        / linear-scaled (alpha_bar_T ~ 1e-4) / cosine
  B. steps (cosine)   : T=10 / T=25 / T=100
  C. pitfall          : T=10 with the T=1000-style linear endpoints reused
  D. no time embedding: baseline schedule, input = coordinates only

Outputs to C:/tmp/diffusion-experiment/paid/ (figures + summary.json).
"""
import json
import time

import numpy as np
from pathlib import Path

OUT = Path("C:/tmp/diffusion-experiment/paid")
OUT.mkdir(parents=True, exist_ok=True)

X0 = np.load("C:/tmp/diffusion-experiment/X0.npy")  # 4000 pts, std 1.6, centered


# ---------------- schedules ----------------
def linear_schedule(T, b0=1e-4, b1=0.02):
    betas = np.linspace(b0, b1, T)
    return betas


def cosine_schedule(T, s=0.008):
    # Nichol & Dhariwal (2021): define alpha_bar directly, derive betas
    steps = np.arange(T + 1)
    f = np.cos(((steps / T + s) / (1 + s)) * np.pi / 2) ** 2
    abar = f / f[0]
    betas = 1 - abar[1:] / abar[:-1]
    return np.clip(betas, 1e-8, 0.999)


CONFIGS = {
    "linear_T100":        {"betas": linear_schedule(100),               "temb": True},
    "linear_scaled_T100": {"betas": linear_schedule(100, 2e-3, 0.18),   "temb": True},
    "cosine_T100":        {"betas": cosine_schedule(100),               "temb": True},
    "cosine_T25":         {"betas": cosine_schedule(25),                "temb": True},
    "cosine_T10":         {"betas": cosine_schedule(10),                "temb": True},
    "naive_T10":          {"betas": linear_schedule(10),                "temb": True},
    "no_temb_T100":       {"betas": linear_schedule(100),               "temb": False},
}


# ---------------- model pieces (identical to exp1) ----------------
def time_embedding(t_idx, dim=8):
    t = t_idx.astype(np.float64)[:, None]
    i = np.arange(dim)[None, :]
    angle_rates = 1.0 / np.power(1000, (2 * (i // 2)) / dim)
    angles = t * angle_rates
    emb = np.zeros((len(t_idx), dim))
    emb[:, 0::2] = np.sin(angles[:, 0::2])
    emb[:, 1::2] = np.cos(angles[:, 1::2])
    return emb


def train(betas, use_temb, seed=0, n_iters=20000, batch_size=256, hidden=256):
    rng = np.random.default_rng(seed)
    T = len(betas)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)

    TIME_DIM = 8 if use_temb else 0
    IN_DIM = 2 + TIME_DIM
    W1 = rng.normal(0, np.sqrt(2.0 / IN_DIM), size=(IN_DIM, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), size=(hidden, 2))
    b2 = np.zeros(2)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 1e-3

    losses = []
    for it in range(1, n_iters + 1):
        x0 = X0[rng.integers(0, len(X0), size=batch_size)]
        t_idx = rng.integers(0, T, size=batch_size)
        noise = rng.normal(0, 1, size=x0.shape)
        ab = alpha_bars[t_idx][:, None]
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise

        if use_temb:
            inp = np.concatenate([xt, time_embedding(t_idx, TIME_DIM)], axis=1)
        else:
            inp = xt
        z1 = inp @ W1 + b1
        a1 = np.maximum(z1, 0)
        pred = a1 @ W2 + b2

        diff = pred - noise
        losses.append(np.mean(diff ** 2))
        d_out = 2 * diff / diff.shape[0]

        dW2 = a1.T @ d_out
        db2 = d_out.sum(axis=0)
        da1 = d_out @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = inp.T @ dz1
        db1 = dz1.sum(axis=0)

        cur_lr = LR if it < n_iters * 0.6 else LR * 0.3
        for p, g, mi, vi in zip(params, [dW1, db1, dW2, db2], m, v):
            mi[:] = B1 * mi + (1 - B1) * g
            vi[:] = B2 * vi + (1 - B2) * g ** 2
            p -= cur_lr * (mi / (1 - B1 ** it)) / (np.sqrt(vi / (1 - B2 ** it)) + EPS)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "losses": np.array(losses),
            "betas": betas, "alphas": alphas, "alpha_bars": alpha_bars,
            "use_temb": use_temb, "TIME_DIM": TIME_DIM}


def sample(model, n_samples=1500, seed=1):
    rng = np.random.default_rng(seed)
    betas, alphas, abars = model["betas"], model["alphas"], model["alpha_bars"]
    T = len(betas)
    x = rng.normal(0, 1, size=(n_samples, 2))
    for t_idx in reversed(range(T)):
        if model["use_temb"]:
            temb = time_embedding(np.full(n_samples, t_idx), model["TIME_DIM"])
            inp = np.concatenate([x, temb], axis=1)
        else:
            inp = x
        a1 = np.maximum(inp @ model["W1"] + model["b1"], 0)
        eps_theta = a1 @ model["W2"] + model["b2"]
        mean = (1 / np.sqrt(alphas[t_idx])) * (
            x - (betas[t_idx] / np.sqrt(1 - abars[t_idx])) * eps_theta)
        if t_idx > 0:
            x = mean + np.sqrt(betas[t_idx]) * rng.normal(0, 1, size=x.shape)
        else:
            x = mean
    return x


def nn_dist(A, B):
    # mean over points in A of distance to nearest point in B
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min(axis=1)).mean())


# ---------------- run everything ----------------
summary = {}
real_ref = X0[np.random.default_rng(9).integers(0, len(X0), 1500)]
for name, cfg in CONFIGS.items():
    t0 = time.time()
    model = train(cfg["betas"], cfg["temb"])
    gen = sample(model)
    prec = nn_dist(gen, X0)        # gen -> real: are generated points on the manifold?
    cov = nn_dist(real_ref, gen)   # real -> gen: is the whole shape covered?
    summary[name] = {
        "T": len(cfg["betas"]),
        "alpha_bar_T": float(model["alpha_bars"][-1]),
        "sqrt_alpha_bar_T": float(np.sqrt(model["alpha_bars"][-1])),
        "final_loss": float(model["losses"][-200:].mean()),
        "gen_to_real": prec,
        "real_to_gen": cov,
        "seconds": round(time.time() - t0, 1),
    }
    np.save(OUT / f"gen_{name}.npy", gen)
    np.save(OUT / f"losses_{name}.npy", model["losses"])
    np.save(OUT / f"abars_{name}.npy", model["alpha_bars"])
    print(name, json.dumps(summary[name]))

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("done")
