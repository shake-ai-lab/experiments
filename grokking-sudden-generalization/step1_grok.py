"""Grokking: 訓練精度100%到達後も学習を続けると、遥か後にテスト精度が突然跳ね上がる現象。
mod加算 (a+b) mod P を MLP に学習させ、weight decay あり/なし で比較する。
再現条件が繊細なので、条件を変えて複数回試す。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

P = 53  # 法
TRAIN_FRAC = 0.4
EMB, HID = 96, 256


def make_data(seed=0):
    r = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(P) for b in range(P)])
    labels = (pairs[:, 0] + pairs[:, 1]) % P
    idx = r.permutation(len(pairs))
    ntr = int(len(pairs) * TRAIN_FRAC)
    return pairs[idx[:ntr]], labels[idx[:ntr]], pairs[idx[ntr:]], labels[idx[ntr:]]


class Model:
    """埋め込み(a,b) -> concat -> MLP -> P分類。numpyで手書き"""

    def __init__(self, seed=0, scale=1.0):
        r = np.random.default_rng(seed)
        self.E = r.normal(0, scale / np.sqrt(EMB), (P, EMB))
        self.W1 = r.normal(0, scale / np.sqrt(2 * EMB), (2 * EMB, HID))
        self.b1 = np.zeros(HID)
        self.W2 = r.normal(0, scale / np.sqrt(HID), (HID, P))
        self.b2 = np.zeros(P)

    def params(self):
        return [self.E, self.W1, self.b1, self.W2, self.b2]

    def forward(self, pairs):
        ea, eb = self.E[pairs[:, 0]], self.E[pairs[:, 1]]
        h0 = np.concatenate([ea, eb], axis=1)
        z1 = h0 @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        logits = h1 @ self.W2 + self.b2
        return logits, (pairs, h0, z1, h1)

    def backward(self, cache, dlogits):
        pairs, h0, z1, h1 = cache
        gW2 = h1.T @ dlogits
        gb2 = dlogits.sum(0)
        dh1 = dlogits @ self.W2.T
        dz1 = dh1 * (z1 > 0)
        gW1 = h0.T @ dz1
        gb1 = dz1.sum(0)
        dh0 = dz1 @ self.W1.T
        gE = np.zeros_like(self.E)
        np.add.at(gE, pairs[:, 0], dh0[:, :EMB])
        np.add.at(gE, pairs[:, 1], dh0[:, EMB:])
        return [gE, gW1, gb1, gW2, gb2]


def softmax_ce(logits, y):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1
    return loss, d / n, (p.argmax(1) == y).mean()


def run(wd, steps=40_000, lr=1e-3, seed=0, scale=1.0, label=""):
    Xtr, ytr, Xte, yte = make_data(seed)
    m = Model(seed, scale)
    ps = m.params()
    mom = [np.zeros_like(p) for p in ps]
    vel = [np.zeros_like(p) for p in ps]
    B1, B2, EPS = 0.9, 0.98, 1e-8
    hist = []
    for t in range(1, steps + 1):
        logits, cache = m.forward(Xtr)
        loss, dl, acc_tr = softmax_ce(logits, ytr)
        grads = m.backward(cache, dl)
        for i, (p, g) in enumerate(zip(ps, grads)):
            g = g + wd * p  # weight decay (AdamW流)
            mom[i] = B1 * mom[i] + (1 - B1) * g
            vel[i] = B2 * vel[i] + (1 - B2) * g**2
            p -= lr * (mom[i] / (1 - B1**t)) / (np.sqrt(vel[i] / (1 - B2**t)) + EPS)
        if t % 100 == 0 or t == 1:
            lg_te, _ = m.forward(Xte)
            _, _, acc_te = softmax_ce(lg_te, yte)
            wnorm = float(np.sqrt(sum((p**2).sum() for p in ps)))
            hist.append((t, loss, acc_tr, acc_te, wnorm))
            if t % 5000 == 0 or t == 1:
                print(f"  [{label}] step {t:6d}: loss {loss:.4f} train {acc_tr:.3f} test {acc_te:.3f} |w| {wnorm:.1f}")
    return np.array(hist)


if __name__ == "__main__":
    print("=== weight decay あり (wd=1.0) ===")
    h_wd = run(wd=1.0, label="wd=1.0")
    print("=== weight decay なし ===")
    h_no = run(wd=0.0, label="wd=0")
    np.savez("hist.npz", wd=h_wd, nowd=h_no)

    def summarize(h, name):
        t, loss, tr, te, w = h.T
        i_tr = np.argmax(tr >= 0.99) if (tr >= 0.99).any() else -1
        i_te = np.argmax(te >= 0.9) if (te >= 0.9).any() else -1
        print(f"{name}: 訓練99%到達 step {int(t[i_tr]) if i_tr>=0 else '未到達'}, "
              f"テスト90%到達 step {int(t[i_te]) if i_te>=0 else '未到達'}, "
              f"最終テスト {te[-1]:.3f}")

    summarize(h_wd, "weight decay あり")
    summarize(h_no, "weight decay なし")
