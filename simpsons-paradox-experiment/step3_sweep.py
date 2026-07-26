"""
Sweep the "skill gap" between the two groups (delta = advanced group's
baseline score advantage over beginners) and watch the pooled correlation
flip sign. Holding everything else fixed (hours distributions, slope,
noise), delta controls how much the advanced group's high-baseline /
low-hours cluster out-weighs the true positive within-group slope when
pooled.
"""
import numpy as np
from scipy import stats

SEED = 42
N_A, N_B = 150, 150
SLOPE = 3.5
NOISE_SD = 6.0
HOURS_A_MEAN, HOURS_A_SD = 3.0, 1.2
HOURS_B_MEAN, HOURS_B_SD = 13.0, 2.5
INTERCEPT_B = 15.0

# Fix the hours draws and noise draws once, so only `delta` changes across
# the sweep (isolates the effect of the skill-gap parameter cleanly).
rng = np.random.default_rng(SEED)
hours_a = np.clip(rng.normal(HOURS_A_MEAN, HOURS_A_SD, N_A), 0.2, None)
hours_b = np.clip(rng.normal(HOURS_B_MEAN, HOURS_B_SD, N_B), 0.2, None)
noise_a = rng.normal(0, NOISE_SD, N_A)
noise_b = rng.normal(0, NOISE_SD, N_B)

deltas = np.linspace(0, 70, 36)  # sweep 0..70 points of baseline advantage
r_pools = []
r_as = []
r_bs = []

for delta in deltas:
    intercept_a = INTERCEPT_B + delta
    score_a = np.clip(intercept_a + SLOPE * hours_a + noise_a, 0, 100)
    score_b = np.clip(INTERCEPT_B + SLOPE * hours_b + noise_b, 0, 100)

    hours_all = np.concatenate([hours_a, hours_b])
    score_all = np.concatenate([score_a, score_b])

    r_pool, _ = stats.pearsonr(hours_all, score_all)
    r_a, _ = stats.pearsonr(hours_a, score_a)
    r_b, _ = stats.pearsonr(hours_b, score_b)

    r_pools.append(r_pool)
    r_as.append(r_a)
    r_bs.append(r_b)

r_pools = np.array(r_pools)
r_as = np.array(r_as)
r_bs = np.array(r_bs)

# find zero crossing of r_pool
sign_change = np.where(np.diff(np.sign(r_pools)) != 0)[0]
if len(sign_change):
    i = sign_change[0]
    # linear interpolation for the crossing delta
    d0, d1 = deltas[i], deltas[i + 1]
    r0, r1 = r_pools[i], r_pools[i + 1]
    delta_cross = d0 + (0 - r0) * (d1 - d0) / (r1 - r0)
    print(f"pooled r crosses zero between delta={d0:.2f} (r={r0:.4f}) and delta={d1:.2f} (r={r1:.4f})")
    print(f"interpolated crossing delta ~= {delta_cross:.2f}")
else:
    print("no sign change found in swept range")

print()
print("delta, r_pool, r_A, r_B")
for d, rp, ra, rb in zip(deltas, r_pools, r_as, r_bs):
    print(f"{d:6.2f}  {rp:7.4f}  {ra:7.4f}  {rb:7.4f}")

np.savez("sweep_data.npz", deltas=deltas, r_pools=r_pools, r_as=r_as, r_bs=r_bs,
         hours_a=hours_a, hours_b=hours_b, noise_a=noise_a, noise_b=noise_b)
