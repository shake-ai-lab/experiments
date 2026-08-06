# ローレンツアトラクタ: 「バタフライ効果」は何秒で効いてくるのか
#
# 実験1: 初期値を1e-9だけずらした2本の軌道が離れていく速さ(リアプノフ指数)を実測
# 実験2: 初期誤差を10^-3から10^-15まで振り、「予測できる時間」がどう伸びるか
# 実験3: アトラクタのフラクタル次元(相関次元)を実測し、文献値2.05前後と比べる
# 実験4: 左右どちらの羽を回るかの切り替えは予測できるか(記号列の統計)
#
# 実行: C:/bin/Python/python310-64/python.exe -u lorenz.py

import json

import numpy as np

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
OUT = {"params": {"sigma": SIGMA, "rho": RHO, "beta": BETA}}


def deriv(s):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z], axis=-1)


def integrate(s0, dt, n, keep=True):
    """古典的な4次のルンゲ・クッタ。複数の初期値を同時に流せるように配列で持つ。"""
    s = np.array(s0, dtype=np.float64)
    traj = np.empty((n + 1,) + s.shape) if keep else None
    if keep:
        traj[0] = s
    for i in range(n):
        k1 = deriv(s)
        k2 = deriv(s + dt / 2 * k1)
        k3 = deriv(s + dt / 2 * k2)
        k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if keep:
            traj[i + 1] = s
    return traj if keep else s


DT = 0.002
S0 = np.array([1.0, 1.0, 20.0])

# 過渡状態を捨てて、アトラクタの上に乗せる
base = integrate(S0, DT, 20000, keep=False)
print(f"アトラクタ上の出発点: {np.round(base, 4)}")

# ================================================================ 実験1
print("\n[実験1] 1e-9 だけずらした2本の軌道が離れていく速さ", flush=True)
EPS = 1e-9
pair = np.stack([base, base + np.array([EPS, 0.0, 0.0])])
N1 = 12000                       # 24 時間単位
tr = integrate(pair, DT, N1)
d = np.linalg.norm(tr[:, 0] - tr[:, 1], axis=-1)
t = np.arange(N1 + 1) * DT
# 指数増大している区間(飽和する前)だけで傾きを取る
sat = d.max()
mask = (d > EPS * 10) & (d < sat * 0.05)
lam = np.polyfit(t[mask], np.log(d[mask]), 1)[0]
print(f"  最大リアプノフ指数の実測: {lam:.4f} /時間単位  (文献値 約0.906)")
for tv in [0, 5, 10, 15, 20]:
    i = int(tv / DT)
    print(f"    t={tv:>3}: 2本の距離 {d[i]:.3e}")
first_visible = t[np.argmax(d > 1.0)]
print(f"  距離が 1.0 を超えたのは t={first_visible:.2f} (アトラクタの直径はおよそ {sat:.0f})")
OUT["exp1"] = {"eps": EPS, "dt": DT, "t": t[::10].tolist(), "d": d[::10].tolist(),
               "lambda": float(lam), "t_visible": float(first_visible), "sat": float(sat)}

# ================================================================ 実験2
print("\n[実験2] 初期誤差をどれだけ小さくすれば、どれだけ長く予測できるか", flush=True)
rows = []
for e in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15]:
    pair = np.stack([base, base + np.array([e, 0.0, 0.0])])
    tr2 = integrate(pair, DT, 25000)
    dd = np.linalg.norm(tr2[:, 0] - tr2[:, 1], axis=-1)
    idx = np.flatnonzero(dd > 1.0)
    th = idx[0] * DT if idx.size else np.nan
    rows.append({"eps": e, "horizon": float(th)})
    print(f"  初期誤差 {e:.0e} -> 距離1.0に達するまで {th:6.2f} 時間単位")
hz = np.array([r["horizon"] for r in rows])
ee = np.array([r["eps"] for r in rows])
slope = np.polyfit(np.log10(1 / ee), hz, 1)[0]
print(f"  精度を10倍にするごとに伸びる予測時間: {slope:.2f} 時間単位")
print(f"  (理論: ln(10)/λ = {np.log(10)/lam:.2f})")
OUT["exp2"] = {"rows": rows, "slope": float(slope), "theory": float(np.log(10) / lam)}

# ================================================================ 実験3
print("\n[実験3] アトラクタのフラクタル次元(相関次元)", flush=True)
long_tr = integrate(base, DT, 400000)[::40]      # 1万点に間引く
pts = long_tr
print(f"  軌道点 {len(pts):,} 点で相関積分を計算する", flush=True)
sub = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
D2 = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
iu = np.triu_indices(len(sub), 1)
dists = D2[iu]
rs = np.geomspace(0.05, 8.0, 24)
C = np.array([(dists < r).mean() for r in rs])
ok = (C > 1e-4) & (C < 0.1)
dim = np.polyfit(np.log(rs[ok]), np.log(C[ok]), 1)[0]
print(f"  相関次元の実測: {dim:.3f}  (文献値 約2.05)")
print(f"  フィットに使ったr: {rs[ok][0]:.3f} 〜 {rs[ok][-1]:.3f}")
OUT["exp3"] = {"r": rs.tolist(), "C": C.tolist(), "dim": float(dim),
               "fit_range": [float(rs[ok][0]), float(rs[ok][-1])]}

# ================================================================ 実験4
print("\n[実験4] 左右どちらの羽を回るかの切り替え", flush=True)
xs = long_tr[:, 0]
wing = np.sign(xs)
flips = np.flatnonzero(np.diff(wing) != 0)
runs = np.diff(np.concatenate([[0], flips]))
print(f"  40万ステップ(間引き後1万点)で羽の乗り換え {len(flips):,} 回")
print(f"  同じ羽に留まる長さ: 中央値 {np.median(runs):.0f} / 平均 {runs.mean():.1f} / 最大 {runs.max()}")
# 記号列にして、直前の履歴から次を当てられるかを見る
sym = (wing > 0).astype(int)
for k in [1, 2, 3, 5, 8]:
    ctx = {}
    for i in range(len(sym) - k):
        key = tuple(sym[i:i + k])
        ctx.setdefault(key, []).append(sym[i + k])
    acc = np.mean([max(np.mean(v), 1 - np.mean(v)) for v in ctx.values() if len(v) > 20])
    print(f"  直前{k}個の羽の並びから次を予測: 正解率 {acc*100:.1f}%")
OUT["exp4"] = {"flips": int(len(flips)), "run_median": float(np.median(runs)),
               "run_max": int(runs.max())}

np.save("traj.npy", long_tr[:6000])
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json / traj.npy")
