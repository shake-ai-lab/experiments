"""LRウォームアップの効果を検証する前に、まず「不安定になるかならないかの際どいLR」を
honestに探す。深めのMLP(64->128->128->128->128->10, ReLU4層)をsklearn digitsで訓練し、
LRを1e-3から3e-1までスイープ。各LRで3シードだけ試し、
発散(loss>50 or NaN)/序盤スパイク(初期lossの3倍を超える)を記録する。
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HID = 128
N_HIDDEN = 4  # 隠れ層の数(4層、ReLU)


def load_data(seed=0):
    d = load_digits()
    X, y = d.data.astype(np.float64), d.target.astype(np.int64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    return Xtr, ytr, Xte, yte


class MLP:
    """入力64 -> 128 -> 128 -> 128 -> 128 -> 10 の全結合ReLU MLP。numpyで手書き。"""

    def __init__(self, seed, n_in=64, n_hid=HID, n_layers=N_HIDDEN, n_out=10):
        r = np.random.default_rng(seed)
        dims = [n_in] + [n_hid] * n_layers + [n_out]
        self.W = []
        self.b = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            self.W.append(r.normal(0, np.sqrt(2.0 / fan_in), (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))

    def params(self):
        return self.W + self.b

    def forward(self, X):
        zs, hs = [], [X]
        h = X
        for i in range(len(self.W) - 1):
            z = h @ self.W[i] + self.b[i]
            h = np.maximum(z, 0)
            zs.append(z)
            hs.append(h)
        logits = h @ self.W[-1] + self.b[-1]
        return logits, (zs, hs)

    def backward(self, cache, dlogits):
        zs, hs = cache
        n_layers = len(self.W)
        gW = [None] * n_layers
        gb = [None] * n_layers
        d = dlogits
        for i in reversed(range(n_layers)):
            gW[i] = hs[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (zs[i - 1] > 0)
        return gW + gb


def softmax_ce(logits, y):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1
    return loss, d / n, (p.argmax(1) == y).mean()


def train(lr, seed, steps=300, batch_size=32):
    Xtr, ytr, Xte, yte = load_data(0)  # データ分割は固定、初期化とミニバッチ順だけseed依存
    r = np.random.default_rng(seed)
    m = MLP(seed)
    ps = m.params()
    mom = [np.zeros_like(p) for p in ps]
    vel = [np.zeros_like(p) for p in ps]
    B1, B2, EPS = 0.9, 0.999, 1e-8
    n = len(Xtr)
    losses = []
    first_loss = None
    for t in range(1, steps + 1):
        idx = r.integers(0, n, batch_size)
        Xb, yb = Xtr[idx], ytr[idx]
        logits, cache = m.forward(Xb)
        loss, dl, _ = softmax_ce(logits, yb)
        if first_loss is None:
            first_loss = loss
        if not np.isfinite(loss):
            return losses, "diverged(nan)", t
        if loss > 50:
            return losses, "diverged(loss>50)", t
        grads = m.backward(cache, dl)
        n_layers = len(m.W)
        gW, gb = grads[:n_layers], grads[n_layers:]
        for i, g in enumerate(gW):
            mom[i] = B1 * mom[i] + (1 - B1) * g
            vel[i] = B2 * vel[i] + (1 - B2) * g**2
            m.W[i] -= lr * (mom[i] / (1 - B1**t)) / (np.sqrt(vel[i] / (1 - B2**t)) + EPS)
        for j, g in enumerate(gb):
            i = n_layers + j
            mom[i] = B1 * mom[i] + (1 - B1) * g
            vel[i] = B2 * vel[i] + (1 - B2) * g**2
            m.b[j] -= lr * (mom[i] / (1 - B1**t)) / (np.sqrt(vel[i] / (1 - B2**t)) + EPS)
        losses.append(loss)
    peak = max(losses)
    status = "spike" if peak > first_loss * 3 else "stable"
    return losses, status, None


if __name__ == "__main__":
    lrs = [0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3]
    seeds = [0, 1, 2]
    print(f"{'lr':>8} | " + " | ".join(f"seed{s}" for s in seeds))
    for lr in lrs:
        row = []
        for s in seeds:
            losses, status, div_t = train(lr, s)
            first = losses[0] if losses else float("nan")
            last = losses[-1] if losses else float("nan")
            peak = max(losses) if losses else float("nan")
            if div_t is not None:
                row.append(f"{status}@t{div_t}")
            else:
                row.append(f"{status}(first{first:.2f}->peak{peak:.2f}->last{last:.2f})")
        print(f"{lr:>8} | " + " | ".join(row))
