"""Momentum success count when the 200 start points are shifted by tiny amounts."""
import numpy as np

BOUND = 5.0


def ackley(pos):
    x, y = pos[..., 0], pos[..., 1]
    r = np.sqrt(0.5 * (x ** 2 + y ** 2))
    return -20 * np.exp(-0.2 * r) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


def grad(pos, h=1e-5):
    gx = (ackley(pos + np.array([h, 0])) - ackley(pos - np.array([h, 0]))) / (2 * h)
    gy = (ackley(pos + np.array([0, h])) - ackley(pos - np.array([0, h]))) / (2 * h)
    return np.array([gx, gy])


def momentum_final(start, iters=500, lr=0.05, beta=0.9):
    pos, v = start.copy(), np.zeros(2)
    for _ in range(iters):
        v = beta * v - lr * grad(pos)
        pos = np.clip(pos + v, -BOUND, BOUND)
    return pos


starts = np.random.default_rng(0).uniform(-BOUND, BOUND, size=(200, 2))
for shift in [0.0, 1e-12]:
    counts = []
    for k in range(1 if shift == 0 else 6):
        rng = np.random.default_rng(100 + k)
        s2 = starts + (rng.uniform(-1, 1, starts.shape) * shift)
        counts.append(sum(np.linalg.norm(momentum_final(s)) < 0.5 for s in s2))
    print("shift", shift, "successes of 200:", counts)
