import numpy as np

# Rosenbrock ("banana") function - classic hard test case: a long, narrow, curved valley
# with a very shallow gradient along the valley floor but steep walls on the sides.
def f(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])

start = np.array([-1.5, 2.5])
n_steps = 800

def run_sgd(lr=0.0011):
    p = start.copy()
    path = [p.copy()]
    for _ in range(n_steps):
        g = grad(*p)
        p = p - lr * g
        path.append(p.copy())
    return np.array(path)

def run_momentum(lr=0.0011, beta=0.9):
    p = start.copy()
    v = np.zeros(2)
    path = [p.copy()]
    for _ in range(n_steps):
        g = grad(*p)
        v = beta * v + g
        p = p - lr * v
        path.append(p.copy())
    return np.array(path)

def run_rmsprop(lr=0.02, decay=0.9, eps=1e-8):
    p = start.copy()
    s = np.zeros(2)
    path = [p.copy()]
    for _ in range(n_steps):
        g = grad(*p)
        s = decay * s + (1 - decay) * g ** 2
        p = p - lr * g / (np.sqrt(s) + eps)
        path.append(p.copy())
    return np.array(path)

def run_adam(lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    p = start.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    path = [p.copy()]
    for t in range(1, n_steps + 1):
        g = grad(*p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(p.copy())
    return np.array(path)

paths = {
    "SGD": run_sgd(),
    "Momentum": run_momentum(),
    "RMSprop": run_rmsprop(),
    "Adam": run_adam(),
}

target = np.array([1.0, 1.0])
print(f"start={start}, target={target}, steps={n_steps}\n")
for name, path in paths.items():
    final = path[-1]
    dist = np.linalg.norm(final - target)
    loss_final = f(*final)
    # steps to reach within 0.05 of target
    dists = np.linalg.norm(path - target, axis=1)
    reached = np.argmax(dists < 0.05) if np.any(dists < 0.05) else None
    print(f"{name:10s} final=({final[0]:+.4f},{final[1]:+.4f})  dist_to_min={dist:.4f}  "
          f"final_loss={loss_final:.6f}  steps_to_within_0.05={reached}")
    np.save(f"C:/tmp/optimizer-race-experiment/path_{name.lower()}.npy", path)
