# -*- coding: utf-8 -*-
"""メトロポリス・ヘイスティングス法を numpy だけで実装し、two moons を
標的分布としてサンプリングする。

生成モデル四国志(拡散モデル/GAN/Flow/VAE)は「分布が分からないのでデータから
学習した」。ここでは逆に「分布の形は分かっている(正規化定数だけ不明)」場合に、
学習なしで直接サンプリングできることを確かめる。

出力: C:/tmp/mcmc-experiment/ に npz と json
"""
import json
import numpy as np
from pathlib import Path

OUT = Path("C:/tmp/mcmc-experiment")
OUT.mkdir(parents=True, exist_ok=True)

SIGMA = 0.18          # 三日月の太さ


# ---------------- 標的分布 (正規化定数は未知のままでよい) ----------------
def _dist_to_arc(p, center, r, t0, t1):
    """点群 p から、中心 center・半径 r・角度 [t0,t1] の円弧までの距離"""
    d = p - center
    ang = np.arctan2(d[:, 1], d[:, 0])
    rad = np.linalg.norm(d, axis=1)
    inside = (ang >= t0) & (ang <= t1)
    # 弧の内側なら動径方向の差、外側なら「端点までの距離」だけを使う。
    # (以前は外側でも動径方向の差を min に混ぜてしまい、半円ではなく
    #  まるい輪になっていた)
    d_end = np.minimum.reduce([
        np.linalg.norm(p - (center + r * np.array([np.cos(t), np.sin(t)])), axis=1)
        for t in (t0, t1)
    ])
    return np.where(inside, np.abs(rad - r), d_end)


def log_p_tilde(p, gap=0.0):
    """標的分布の対数(定数を除く)。gap で 2 つの三日月を引き離せる。"""
    p = np.atleast_2d(p)
    d1 = _dist_to_arc(p, np.array([0.0, gap]), 1.0, 0.0, np.pi)
    d2 = _dist_to_arc(p, np.array([1.0, -0.5 - gap]), 1.0, -np.pi, 0.0)
    a = -(d1 ** 2) / (2 * SIGMA ** 2)
    b = -(d2 ** 2) / (2 * SIGMA ** 2)
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))     # log-sum-exp


# ---------------- メトロポリス法 ----------------
def metropolis(n_steps, step, x0, seed=0, gap=0.0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, float)
    lp = log_p_tilde(x, gap)[0]
    chain = np.empty((n_steps, 2))
    acc = 0
    for i in range(n_steps):
        prop = x + rng.normal(0, step, 2)
        lp_prop = log_p_tilde(prop, gap)[0]
        # 受容確率 min(1, p~(prop)/p~(x)) — 正規化定数は比で消える
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop
            acc += 1
        chain[i] = x
    return chain, acc / n_steps


def autocorr(x, maxlag=200):
    x = x - x.mean()
    v = x.var()
    return np.array([1.0 if k == 0 else np.corrcoef(x[:-k], x[k:])[0, 1] for k in range(maxlag)]) if v > 0 else np.zeros(maxlag)


def ess(chain):
    """有効サンプルサイズ(初期正値列で打ち切る素朴な推定)"""
    n = len(chain)
    out = []
    for d in range(2):
        ac = autocorr(chain[:, d], min(500, n // 4))
        s = 0.0
        for k in range(1, len(ac)):
            if ac[k] <= 0:
                break
            s += ac[k]
        out.append(n / (1 + 2 * s))
    return float(np.mean(out))


results = {}

# ---- (1) 基本: 十分な長さのチェーン ----
N = 200_000
chain, rate = metropolis(N, step=0.35, x0=[3.0, 3.0], seed=1)
burn = 2000
results["main"] = {"steps": N, "step_size": 0.35, "acceptance": rate,
                   "ess": ess(chain[burn:]), "burn_in_used": burn}
print("main:", results["main"])
np.savez_compressed(OUT / "chain_main.npz", chain=chain[::5], burn=burn)

# ---- (2) ステップ幅スイープ ----
sweep = []
for st in [0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.8, 3.0, 6.0]:
    c, r = metropolis(60_000, step=st, x0=[0.0, 0.0], seed=2)
    e = ess(c[2000:])
    sweep.append({"step": st, "acceptance": r, "ess": e})
    print(f"  step={st:<5} acc={r:.3f} ess={e:8.1f}")
results["sweep"] = sweep

# ---- (3) バーンイン: 遠くから出発 ----
c_burn, _ = metropolis(3000, step=0.35, x0=[6.0, 6.0], seed=3)
np.savez_compressed(OUT / "chain_burnin.npz", chain=c_burn)
first_in = int(np.argmax(log_p_tilde(c_burn) > -5))
results["burn_in_steps_to_reach_target"] = first_in
print("burn-in: 標的に入るまで", first_in, "ステップ")

# ---- (4) 多峰性での失敗: 2つの三日月を引き離す ----
for gap, tag in [(0.0, "near"), (2.5, "far")]:
    start = [0.0, gap + 1.0]          # 上の三日月の頂点から出発
    c, r = metropolis(120_000, step=0.35, x0=start, seed=4, gap=gap)
    b = c[5000:]
    # どちらのモードにいるかを判定(上の弧に近いか下の弧に近いか)
    d1 = _dist_to_arc(b, np.array([0.0, gap]), 1.0, 0.0, np.pi)
    d2 = _dist_to_arc(b, np.array([1.0, -0.5 - gap]), 1.0, -np.pi, 0.0)
    frac = float((d1 < d2).mean())
    switches = int((np.diff((d1 < d2).astype(int)) != 0).sum())
    results[f"multimodal_{tag}"] = {"gap": gap, "acceptance": r,
                                    "mode1_fraction": frac, "mode_switches": switches}
    print(f"multimodal {tag}: 上の弧の割合={frac:.3f} モード往復={switches}回")
    np.savez_compressed(OUT / f"chain_gap_{tag}.npz", chain=c[::5], gap=gap)

with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nsaved ->", OUT)
