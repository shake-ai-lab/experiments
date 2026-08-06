# ウラムの螺旋: 素数を渦巻きに並べると、なぜ斜めの線が浮き出るのか
#
# 実験1: 螺旋上のすべての直線について素数密度を測り、ランダムに撒いた対照群と比べる
# 実験2: 濃い直線に対応する2次式を同定し、素数の出やすさを測る
# 実験3: オイラーの多項式 n^2+n+41 はどこまで素数を出し続けるか
#
# 実行: C:/bin/Python/python310-64/python.exe -u ulam.py

import json

import numpy as np

OUT = {}
N = 501                      # 螺旋の一辺(奇数)
LIMIT = N * N


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


IS_PRIME = sieve(LIMIT)
print(f"{LIMIT:,} までの素数: {IS_PRIME.sum():,} 個 (密度 {IS_PRIME.mean()*100:.2f}%)")


def ulam_grid(n):
    """中心から反時計回りに 1,2,3,... を並べた n x n の格子を返す"""
    g = np.zeros((n, n), dtype=np.int64)
    x = y = n // 2
    g[y, x] = 1
    val = 1
    step = 1
    dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]     # 右, 上, 左, 下
    d = 0
    while val < n * n:
        for _ in range(2):
            dy, dx = dirs[d % 4]
            for _ in range(step):
                if val >= n * n:
                    break
                y += dy
                x += dx
                val += 1
                if 0 <= y < n and 0 <= x < n:
                    g[y, x] = val
            d += 1
        step += 1
    return g


G = ulam_grid(N)
P = IS_PRIME[G]              # 素数のマス
print(f"螺旋 {N}x{N} 内の素数: {P.sum():,} 個")


def line_densities(mask):
    """水平・垂直・2方向の対角線について、各直線上の「印の割合」を返す"""
    n = mask.shape[0]
    out = {}
    out["横"] = [(mask[i].mean(), n) for i in range(n)]
    out["縦"] = [(mask[:, j].mean(), n) for j in range(n)]
    dg = [np.diagonal(mask, off) for off in range(-n + 1, n)]
    out["右下がり対角"] = [(d.mean(), len(d)) for d in dg if len(d) >= 40]
    fm = mask[:, ::-1]
    dg2 = [np.diagonal(fm, off) for off in range(-n + 1, n)]
    out["右上がり対角"] = [(d.mean(), len(d)) for d in dg2 if len(d) >= 40]
    return out


# ================================================================ 実験1
print("\n[実験1] 直線ごとの素数密度と、ランダム対照群の比較", flush=True)
dens = line_densities(P)
base = P.mean()
summary = {}
for name, lst in dens.items():
    v = np.array([d for d, ln in lst])
    summary[name] = {"max": float(v.max()), "mean": float(v.mean()),
                     "top5": sorted(v.tolist(), reverse=True)[:5]}
    print(f"  {name:<12} 最大 {v.max()*100:5.2f}% / 平均 {v.mean()*100:5.2f}% "
          f"(全体の素数密度 {base*100:.2f}%)")

rng = np.random.default_rng(0)
REPS = 200
rand_max = {k: [] for k in dens}
for _ in range(REPS):
    R = rng.random((N, N)) < base
    dd = line_densities(R)
    for k, lst in dd.items():
        rand_max[k].append(max(d for d, ln in lst))
print("\n  同じ密度でランダムに撒いた場合の「最も濃い直線」(200回)")
for k in dens:
    arr = np.array(rand_max[k])
    obs = summary[k]["max"]
    p = float((arr >= obs).mean())
    print(f"  {k:<12} ランダムの最大値 平均 {arr.mean()*100:5.2f}% / 上限 {arr.max()*100:5.2f}% "
          f"→ 実測 {obs*100:5.2f}% を超えたのは {int((arr>=obs).sum())}/{REPS} 回 (p={p:.3f})")
    summary[k]["rand_mean"] = float(arr.mean())
    summary[k]["rand_max"] = float(arr.max())
    summary[k]["p"] = p
OUT["exp1"] = {"base": float(base), "summary": summary, "reps": REPS}

# ================================================================ 実験2
print("\n[実験2] 濃い対角線に対応する2次式", flush=True)
# 右下がり対角のうち最も濃い線を取り出し、その上の数列を調べる
n = N
best = None
for off in range(-n + 1, n):
    d = np.diagonal(P, off)
    if len(d) < 100:
        continue
    if best is None or d.mean() > best[1]:
        best = (off, d.mean())
off, dm = best
vals = np.diagonal(G, off)
print(f"  最も濃い右下がり対角: オフセット {off} / 素数率 {dm*100:.2f}% / 長さ {len(vals)}")
diffs = np.diff(vals)
print(f"  その上に並ぶ数: {vals[:6].tolist()} ...  階差 {diffs[:5].tolist()}")
if len(set(np.diff(diffs).tolist())) == 1:
    a2 = np.diff(diffs)[0] // 2
    print(f"  階差の階差が一定({np.diff(diffs)[0]}) -> 2次式。先頭係数 {a2}")
OUT["exp2"] = {"offset": int(off), "density": float(dm),
               "head": vals[:8].tolist(), "second_diff": int(np.diff(diffs)[0])}

# ================================================================ 実験3
print("\n[実験3] オイラーの多項式 n^2+n+41", flush=True)
BIG = 4_000_000
big_sieve = sieve(BIG)
rows = []
for upper in [40, 100, 1000, 10000]:
    ns = np.arange(upper)
    vals = ns * ns + ns + 41
    ok = vals <= BIG
    frac = big_sieve[vals[ok]].mean()
    first_fail = int(ns[ok][~big_sieve[vals[ok]]][0]) if (~big_sieve[vals[ok]]).any() else None
    rows.append({"upper": upper, "prime_frac": float(frac), "first_composite": first_fail})
    print(f"  n=0〜{upper-1}: 素数の割合 {frac*100:5.2f}%  "
          f"(最初に素数でなくなる n = {first_fail})")
# 比較: 同じくらいの大きさの数がランダムに素数である確率
for upper in [1000, 10000]:
    mag = (upper ** 2) // 2
    print(f"  参考: {mag:,} 前後の数が素数である確率はおよそ {1/np.log(mag)*100:.2f}%")
OUT["exp3"] = {"rows": rows, "limit": BIG}

# いろいろな n^2+n+c を比べる
print("\n  n^2+n+c の c を変えて、n=0〜999 での素数率を比べる")
cands = []
for c in range(1, 1000):
    vals = np.arange(1000) ** 2 + np.arange(1000) + c
    ok = vals <= BIG
    cands.append((c, big_sieve[vals[ok]].mean()))
cands.sort(key=lambda t: -t[1])
for c, f in cands[:6]:
    print(f"    c={c:>4}: {f*100:5.2f}%")
OUT["exp3"]["top_c"] = [[int(c), float(f)] for c, f in cands[:10]]

np.save("grid.npy", G)
np.save("prime.npy", P)
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json / grid.npy / prime.npy")
