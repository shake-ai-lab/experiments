"""Grokking再挑戦。前回の失敗を修正:
[失敗1] weight decayを勾配に足す「結合型(L2)」で実装 -> Adamの正規化でwd項が支配し重みが0に崩壊
[修正]  AdamWの「分離型」(更新の最後にp -= lr*wd*p)にする
複数のwdでスイープし、grokkingが起きる領域を探す。"""
import matplotlib

matplotlib.use("Agg")
import numpy as np

P_MOD = 53
TRAIN_FRAC = 0.4
EMB, HID = 96, 256


def make_data(seed=0):
    r = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(P_MOD) for b in range(P_MOD)])
    labels = (pairs[:, 0] + pairs[:, 1]) % P_MOD
    idx = r.permutation(len(pairs))
    ntr = int(len(pairs) * TRAIN_FRAC)
    return pairs[idx[:ntr]], labels[idx[:ntr]], pairs[idx[ntr:]], labels[idx[ntr:]]


class Model:
    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        self.E = r.normal(0, 1 / np.sqrt(EMB), (P_MOD, EMB))
        self.W1 = r.normal(0, 1 / np.sqrt(2 * EMB), (2 * EMB, HID))
        self.b1 = np.zeros(HID)
        self.W2 = r.normal(0, 1 / np.sqrt(HID), (HID, P_MOD))
        self.b2 = np.zeros(P_MOD)

    def params(self):
        return [self.E, self.W1, self.b1, self.W2, self.b2]

    def forward(self, pairs):
        h0 = np.concatenate([self.E[pairs[:, 0]], self.E[pairs[:, 1]]], axis=1)
        z1 = h0 @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        return h1 @ self.W2 + self.b2, (pairs, h0, z1, h1)

    def backward(self, cache, dl):
        pairs, h0, z1, h1 = cache
        gW2 = h1.T @ dl
        gb2 = dl.sum(0)
        dz1 = (dl @ self.W2.T) * (z1 > 0)
        gW1 = h0.T @ dz1
        gb1 = dz1.sum(0)
        dh0 = dz1 @ self.W1.T
        gE = np.zeros_like(self.E)
        np.add.at(gE, pairs[:, 0], dh0[:, :EMB])
        np.add.at(gE, pairs[:, 1], dh0[:, EMB:])
        return [gE, gW1, gb1, gW2, gb2]


def ce(logits, y):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1
    return loss, d / n, (p.argmax(1) == y).mean()


def run(wd, steps=60_000, lr=1e-3, seed=0, log_every=200, decoupled=True):
    Xtr, ytr, Xte, yte = make_data(seed)
    m = Model(seed)
    ps = m.params()
    mom = [np.zeros_like(p) for p in ps]
    vel = [np.zeros_like(p) for p in ps]
    B1, B2, EPS = 0.9, 0.98, 1e-8
    hist = []
    for t in range(1, steps + 1):
        logits, cache = m.forward(Xtr)
        loss, dl, acc_tr = ce(logits, ytr)
        grads = m.backward(cache, dl)
        for i, (p, g) in enumerate(zip(ps, grads)):
            if not decoupled:
                g = g + wd * p
            mom[i] = B1 * mom[i] + (1 - B1) * g
            vel[i] = B2 * vel[i] + (1 - B2) * g**2
            p -= lr * (mom[i] / (1 - B1**t)) / (np.sqrt(vel[i] / (1 - B2**t)) + EPS)
            if decoupled:
                p -= lr * wd * p  # AdamW: 分離型
        if t % log_every == 0 or t == 1:
            lg, _, _ = ce(*m.forward(Xte)[:1], yte) if False else (None, None, None)
            lgte, _ = m.forward(Xte)
            _, _, acc_te = ce(lgte, yte)
            wn = float(np.sqrt(sum((q**2).sum() for q in ps)))
            hist.append((t, loss, acc_tr, acc_te, wn))
    return np.array(hist)


def report(h, name):
    t, loss, tr, te, w = h.T
    i_tr = np.argmax(tr >= 0.99) if (tr >= 0.99).any() else -1
    i_te = np.argmax(te >= 0.90) if (te >= 0.90).any() else -1
    s_tr = int(t[i_tr]) if i_tr >= 0 else None
    s_te = int(t[i_te]) if i_te >= 0 else None
    gap = (s_te - s_tr) if (s_tr and s_te) else None
    print(f"{name}: 訓練99% @{s_tr}, テスト90% @{s_te}, 遅れ {gap}, 最終テスト {te[-1]:.3f}, |w| {w[-1]:.1f}")
    return s_tr, s_te, te[-1]


if __name__ == "__main__":
    results = {}
    for wd in [0.0, 0.03, 0.1, 0.3, 1.0]:
        h = run(wd=wd, steps=60_000)
        results[wd] = h
        report(h, f"AdamW wd={wd}")
    np.savez("hist_adamw.npz", **{f"wd{k}": v for k, v in results.items()})
    print("saved hist_adamw.npz")
