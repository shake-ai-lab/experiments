# Lorenz attractor: Benettin estimate of the largest Lyapunov exponent, and a
# wing-switch predictor on the symbol sequence.
#
# This script produces lyap.json (read by figures.py) and the numbers quoted in
# the article: the single-run slope (0.760), the Benettin estimate (0.9057) and
# its running average, and the recall of the wing-switch predictor.
#
# Run: python -u lyap.py   (after lorenz.py has written results.json)

import json

import numpy as np

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.002


def rk4(s, dt=DT):
    """Classical RK4 on plain floats (same scheme as lorenz.py, much faster for one orbit)."""
    x, y, z = s
    h = dt / 2
    a1, b1, c1 = SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z
    x2, y2, z2 = x + h * a1, y + h * b1, z + h * c1
    a2, b2, c2 = SIGMA * (y2 - x2), x2 * (RHO - z2) - y2, x2 * y2 - BETA * z2
    x3, y3, z3 = x + h * a2, y + h * b2, z + h * c2
    a3, b3, c3 = SIGMA * (y3 - x3), x3 * (RHO - z3) - y3, x3 * y3 - BETA * z3
    x4, y4, z4 = x + dt * a3, y + dt * b3, z + dt * c3
    a4, b4, c4 = SIGMA * (y4 - x4), x4 * (RHO - z4) - y4, x4 * y4 - BETA * z4
    k = dt / 6
    return (x + k * (a1 + 2 * a2 + 2 * a3 + a4),
            y + k * (b1 + 2 * b2 + 2 * b3 + b4),
            z + k * (c1 + 2 * c2 + 2 * c3 + c4))


# Same starting point on the attractor as lorenz.py
base = (1.0, 1.0, 20.0)
for _ in range(20000):
    base = rk4(base)
print("start on the attractor:", np.round(base, 4))

# ---------------------------------------------------------------- Benettin
# Evolve a reference orbit and a nearby one; every STEPS steps, record the
# growth factor of their separation and rescale the separation back to D0.
D0 = 1e-8
STEPS = 250            # 0.5 time units per segment
NSEG = 4000            # 2000 time units in total
s = base
p = (base[0] + D0, base[1], base[2])
logs = []
for _ in range(NSEG):
    for _ in range(STEPS):
        s = rk4(s)
        p = rk4(p)
    v = (p[0] - s[0], p[1] - s[1], p[2] - s[2])
    d = float(np.linalg.norm(v))
    logs.append(np.log(d / D0))
    f = D0 / d
    p = (s[0] + v[0] * f, s[1] + v[1] * f, s[2] + v[2] * f)
logs = np.array(logs)
seg_t = STEPS * DT
running = np.cumsum(logs) / (np.arange(1, NSEG + 1) * seg_t)
lam = running[-1]
print(f"Benettin estimate: {lam:.4f}")
for n in (100, 500, 2000, NSEG):
    print(f"  after {n} segments: {running[n - 1]:.3f}")
print(f"  literature value: 0.906 ; relative gap {abs(lam - 0.906) / 0.906 * 100:.2f}%")

# Single-run slope (the 0.760 from lorenz.py, repeated here for the comparison)
R = json.load(open("results.json", encoding="utf-8"))
t = np.array(R["exp1"]["t"])
d = np.array(R["exp1"]["d"])
mask = (d > 1e-8) & (d < d.max() * 0.05)
single = float(np.polyfit(t[mask], np.log(d[mask]), 1)[0])
print(f"single-run slope: {single:.4f}  (gap to Benettin {lam - single:.3f}, "
      f"{(1 - single / 0.906) * 100:.0f}% below 0.906)")
print(f"ln10/lambda: Benettin {np.log(10) / lam:.2f}, single-run {np.log(10) / single:.2f}")

# ---------------------------------------------------------------- wing switches
# 600,000 RK4 steps, keep every 30th point -> 20,000 points
N_STEPS, THIN = 600000, 30
s = base
xs = np.empty(N_STEPS // THIN)
for i in range(N_STEPS):
    s = rk4(s)
    if (i + 1) % THIN == 0:
        xs[(i + 1) // THIN - 1] = s[0]
sym = (xs > 0).astype(int)
flip = (np.diff(sym) != 0).astype(int)      # flip[i] = 1 if the wing changes between i and i+1
rate = flip.mean()
print(f"\n{len(sym):,} points: wing switches {flip.sum()} ({rate * 100:.2f}%), "
      f"'never switch' accuracy {(1 - rate) * 100:.2f}%")

recalls = {}
for k in range(1, 17):
    ctx = {}
    for i in range(k - 1, len(sym) - 1):
        key = tuple(sym[i - k + 1:i + 1])
        ctx.setdefault(key, []).append(flip[i])
    pred_rate = {key: np.mean(v) for key, v in ctx.items()}
    hits = tp = 0
    for key, v in ctx.items():
        if pred_rate[key] > 0.5:
            tp += int(np.sum(v))
    recall = tp / flip.sum()
    n_over = sum(1 for r in pred_rate.values() if r > 0.5)
    recalls[k] = recall
    print(f"  k={k:2d}: contexts {len(ctx):6d}, contexts with switch rate > 50%: {n_over}, "
          f"recall {recall * 100:.1f}%")

with open("lyap.json", "w", encoding="utf-8") as f:
    json.dump({"lambda_benettin": float(lam), "running": running[19::20].tolist(),
               "single": single, "recall": recalls, "flip_rate": float(rate)}, f)
print("saved lyap.json")
