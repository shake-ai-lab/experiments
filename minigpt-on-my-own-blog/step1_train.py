"""自分のブログ全記事で、文字レベルのミニGPTをnumpyだけで学習させる。
これまでの記事で実装した部品を全部組み立てる:
- Self-Attention (QKV) … self-attention-qkv-experiment
- 位置エンコーディング … positional-encoding-order-experiment
- 手書きMLP + Adam … diffusion / GAN / flow
2層Transformer, d_model=128, 4ヘッド。順伝播・逆伝播とも手書き。"""
import numpy as np

rng = np.random.default_rng(0)

text = open("C:/tmp/bpe-experiment/corpus.txt", encoding="utf-8").read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype=np.int32)
print(f"コーパス: {len(text):,}文字, 語彙 {V}種類")

BLOCK, D, H, NL = 64, 128, 4, 2
HD = D // H
BATCH = 32
STEPS = 6000
LR = 3e-4


def init(shape, scale=None):
    s = scale or (1.0 / np.sqrt(shape[0]))
    return rng.normal(0, s, shape)


P = {
    "emb": init((V, D), 0.02),
    "pos": np.zeros((BLOCK, D)),
}
# 位置エンコーディング (sin/cos, 学習しない)
pos = np.arange(BLOCK)[:, None]
div = np.exp(np.arange(0, D, 2) * (-np.log(10000.0) / D))
P["pos"][:, 0::2] = np.sin(pos * div)
P["pos"][:, 1::2] = np.cos(pos * div)
for l in range(NL):
    P[f"Wq{l}"] = init((D, D))
    P[f"Wk{l}"] = init((D, D))
    P[f"Wv{l}"] = init((D, D))
    P[f"Wo{l}"] = init((D, D))
    P[f"W1{l}"] = init((D, 4 * D))
    P[f"b1{l}"] = np.zeros(4 * D)
    P[f"W2{l}"] = init((4 * D, D))
    P[f"b2{l}"] = np.zeros(D)
P["Wout"] = init((D, V))
P["bout"] = np.zeros(V)

TRAIN_KEYS = [k for k in P if k != "pos"]
mask = np.tril(np.ones((BLOCK, BLOCK)))


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + eps)
    return xh, (x, mu, var, xh, eps)


def layer_norm_back(cache, dout):
    x, mu, var, xh, eps = cache
    N = x.shape[-1]
    std = np.sqrt(var + eps)
    dxh = dout
    dvar = (dxh * (x - mu) * -0.5 * (var + eps) ** -1.5).sum(-1, keepdims=True)
    dmu = (-dxh / std).sum(-1, keepdims=True) + dvar * (-2 * (x - mu)).mean(-1, keepdims=True)
    return dxh / std + dvar * 2 * (x - mu) / N + dmu / N


def forward(idx):
    B, T = idx.shape
    x = P["emb"][idx] + P["pos"][:T]
    caches = []
    for l in range(NL):
        xn, ln1c = layer_norm(x)
        q = (xn @ P[f"Wq{l}"]).reshape(B, T, H, HD).transpose(0, 2, 1, 3)
        k = (xn @ P[f"Wk{l}"]).reshape(B, T, H, HD).transpose(0, 2, 1, 3)
        v = (xn @ P[f"Wv{l}"]).reshape(B, T, H, HD).transpose(0, 2, 1, 3)
        att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(HD)
        att = np.where(mask[:T, :T] == 0, -1e9, att)
        a = softmax(att)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        proj = o @ P[f"Wo{l}"]
        x = x + proj
        xn2, ln2c = layer_norm(x)
        h = np.maximum(xn2 @ P[f"W1{l}"] + P[f"b1{l}"], 0)
        ff = h @ P[f"W2{l}"] + P[f"b2{l}"]
        x = x + ff
        caches.append((ln1c, xn, q, k, v, a, o, ln2c, xn2, h))
    xf, lnfc = layer_norm(x)
    logits = xf @ P["Wout"] + P["bout"]
    return logits, (idx, caches, xf, lnfc)


def backward(cache, dlogits):
    idx, caches, xf, lnfc = cache
    B, T = idx.shape
    G = {k: np.zeros_like(P[k]) for k in TRAIN_KEYS}
    G["Wout"] = np.einsum("btd,btv->dv", xf, dlogits)
    G["bout"] = dlogits.sum((0, 1))
    dx = layer_norm_back(lnfc, dlogits @ P["Wout"].T)
    for l in reversed(range(NL)):
        ln1c, xn, q, k, v, a, o, ln2c, xn2, h = caches[l]
        # FFN branch
        dff = dx
        G[f"W2{l}"] = np.einsum("btf,btd->fd", h, dff)
        G[f"b2{l}"] = dff.sum((0, 1))
        dh = dff @ P[f"W2{l}"].T * (h > 0)
        G[f"W1{l}"] = np.einsum("btd,btf->df", xn2, dh)
        G[f"b1{l}"] = dh.sum((0, 1))
        dx = dx + layer_norm_back(ln2c, dh @ P[f"W1{l}"].T)
        # attention branch
        dproj = dx
        G[f"Wo{l}"] = np.einsum("btd,bte->de", o, dproj)
        do = dproj @ P[f"Wo{l}"].T
        do = do.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
        dv = a.transpose(0, 1, 3, 2) @ do
        da = do @ v.transpose(0, 1, 3, 2)
        datt = a * (da - (da * a).sum(-1, keepdims=True))
        datt = datt / np.sqrt(HD)
        dq = datt @ k
        dk = datt.transpose(0, 1, 3, 2) @ q
        def merge(t):
            return t.transpose(0, 2, 1, 3).reshape(B, T, D)
        G[f"Wq{l}"] = np.einsum("btd,bte->de", xn, merge(dq))
        G[f"Wk{l}"] = np.einsum("btd,bte->de", xn, merge(dk))
        G[f"Wv{l}"] = np.einsum("btd,bte->de", xn, merge(dv))
        dxn = merge(dq) @ P[f"Wq{l}"].T + merge(dk) @ P[f"Wk{l}"].T + merge(dv) @ P[f"Wv{l}"].T
        dx = dx + layer_norm_back(ln1c, dxn)
    np.add.at(G["emb"], idx, dx)
    return G


def get_batch(split_end, r):
    ix = r.integers(0, split_end - BLOCK - 1, BATCH)
    x = np.stack([data[i : i + BLOCK] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + BLOCK] for i in ix])
    return x, y


def loss_fn(logits, y):
    p = softmax(logits)
    n = y.size
    flat = p.reshape(-1, V)
    yy = y.reshape(-1)
    loss = -np.log(flat[np.arange(n), yy] + 1e-12).mean()
    d = flat.copy()
    d[np.arange(n), yy] -= 1
    return loss, (d / n).reshape(logits.shape)


def generate(n=300, temp=0.8, seed_text="この実験では", r=None):
    r = r or np.random.default_rng(1)
    ids = [stoi.get(c, 0) for c in seed_text]
    for _ in range(n):
        ctx = np.array([ids[-BLOCK:]])
        lg, _ = forward(ctx)
        p = softmax(lg[0, -1] / temp)
        ids.append(int(r.choice(V, p=p)))
    return "".join(itos[i] for i in ids)


if __name__ == "__main__":
    split = int(len(data) * 0.9)
    r = np.random.default_rng(0)
    M = {k: np.zeros_like(P[k]) for k in TRAIN_KEYS}
    Vv = {k: np.zeros_like(P[k]) for k in TRAIN_KEYS}
    B1, B2, EPS = 0.9, 0.99, 1e-8
    losses, samples = [], {}
    snap_at = {1, 200, 600, 1500, 3000, 6000}
    for t in range(1, STEPS + 1):
        x, y = get_batch(split, r)
        logits, cache = forward(x)
        loss, dl = loss_fn(logits, y)
        G = backward(cache, dl)
        for k in TRAIN_KEYS:
            M[k] = B1 * M[k] + (1 - B1) * G[k]
            Vv[k] = B2 * Vv[k] + (1 - B2) * G[k] ** 2
            P[k] -= LR * (M[k] / (1 - B1**t)) / (np.sqrt(Vv[k] / (1 - B2**t)) + EPS)
        losses.append(loss)
        if t in snap_at:
            s = generate(160, r=np.random.default_rng(7))
            samples[t] = s
            print(f"\n--- step {t} (loss {loss:.3f}) ---\n{s[:160]}")
        elif t % 500 == 0:
            print(f"step {t}: loss {loss:.4f}")
    np.savez("minigpt_model.npz", **P, losses=np.array(losses))
    import json
    json.dump(samples, open("samples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved minigpt_model.npz, samples.json")
