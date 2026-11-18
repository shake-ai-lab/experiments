"""How much does the SA success count change with the random seed?

Same 200 start points (seed 0) and the same SA settings as experiment.py;
only the random numbers used inside SA change.
"""
import numpy as np

BOUND = 5.0


def ackley(pos):
    x, y = pos[..., 0], pos[..., 1]
    r = np.sqrt(0.5 * (x ** 2 + y ** 2))
    return -20 * np.exp(-0.2 * r) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


def run_sa_final(start, iters, step_size, T0, cooling, rng):
    pos = start.copy()
    f_pos = ackley(pos)
    T = T0
    for _ in range(iters):
        cand = np.clip(pos + rng.normal(0, step_size, size=2), -BOUND, BOUND)
        f_cand = ackley(cand)
        delta = f_cand - f_pos
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-8)):
            pos, f_pos = cand, f_cand
        T *= cooling
    return pos


starts = np.random.default_rng(0).uniform(-BOUND, BOUND, size=(200, 2))
counts = []
for seed in range(100):
    c = 0
    for i, s in enumerate(starts):
        rng = np.random.default_rng([seed, i]) if seed > 0 else np.random.default_rng(10000 + i)
        end = run_sa_final(s, 500, 0.3, 5.0, 0.985, rng)
        c += np.linalg.norm(end) < 0.5
    counts.append(c)
counts = np.array(counts)
print("seed 0 (the article's run):", counts[0], "of 200")
print("seeds 1-99: min", counts[1:].min(), "max", counts[1:].max())
print("all 100 seeds: min", counts.min(), "max", counts.max(), "mean", counts.mean(), f"({counts.mean() / 2:.1f}%)", "std", round(counts.std(), 2))
print("runs with all 200:", int((counts == 200).sum()))
