"""ナッシュ均衡は「存在する」のに、学習では辿り着かない。

2人のプレイヤーが指数重み更新(multiplicative weights)で相手に適応していくとき、
到達点がナッシュ均衡になるのかを3つのゲームで実測する。

  - じゃんけん      : 混合戦略ナッシュ均衡 (1/3,1/3,1/3) が唯一
  - 囚人のジレンマ  : (裏切り,裏切り) が唯一のナッシュ均衡
  - 協調ゲーム      : 純戦略のナッシュ均衡が2つ

更新則は x_i <- x_i * exp(eta * u_i) / Z。eta -> 0 の極限がレプリケータ方程式なので、
eta を小さくしていくと連続時間の挙動に近づく。
"""

import json
import numpy as np

OUT = "results.json"
rng = np.random.default_rng(20260822)

# --- ゲームの定義 -------------------------------------------------------------
# A: 行プレイヤーの利得, B: 列プレイヤーの利得
RPS_A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
RPS_B = -RPS_A  # ゼロ和

PD_A = np.array([[3.0, 0.0], [5.0, 1.0]])  # 行: 協調/裏切り
PD_B = PD_A.T

COORD_A = np.array([[2.0, 0.0], [0.0, 1.0]])
COORD_B = COORD_A.T

GAMES = {
    "rps": ("じゃんけん", RPS_A, RPS_B, np.full(3, 1 / 3), np.full(3, 1 / 3)),
    "pd": ("囚人のジレンマ", PD_A, PD_B, np.array([0.0, 1.0]), np.array([0.0, 1.0])),
    "coord": ("協調ゲーム", COORD_A, COORD_B, None, None),
}


def normalize(v):
    return v / v.sum()


def mwu(A, B, x0, y0, eta, steps):
    """指数重み更新を steps 回まわし、全軌道を返す。"""
    x, y = x0.copy(), y0.copy()
    xs = np.empty((steps + 1, len(x)))
    ys = np.empty((steps + 1, len(y)))
    xs[0], ys[0] = x, y
    for t in range(steps):
        u = A @ y  # 行プレイヤーが各手を選んだときの期待利得
        v = B.T @ x
        # exp のオーバーフロー避けに最大値を引く(正規化するので結果は変わらない)
        x = normalize(x * np.exp(eta * (u - u.max())))
        y = normalize(y * np.exp(eta * (v - v.max())))
        xs[t + 1], ys[t + 1] = x, y
    return xs, ys


def running_mean(a):
    """時間平均(経験分布)。t 行目までの平均。"""
    return np.cumsum(a, axis=0) / np.arange(1, len(a) + 1)[:, None]


def kl_to(target, p, eps=1e-300):
    """KL(target || p)。均衡からどれだけ離れているかの物差し。"""
    p = np.clip(p, eps, None)
    mask = target > 0
    return float(np.sum(target[mask] * np.log(target[mask] / p[mask])))


results = {}

# --- 実験1: じゃんけん。刻み幅を変えて、均衡に近づくか離れるかを測る ----------
x0 = normalize(np.array([0.4, 0.35, 0.25]))
y0 = normalize(np.array([0.3, 0.4, 0.3]))
nash3 = np.full(3, 1 / 3)
STEPS = 200_000

rps_runs = {}
for eta in [0.2, 0.05, 0.01, 0.002]:
    xs, ys = mwu(RPS_A, RPS_B, x0, y0, eta, STEPS)
    # Bailey & Piliouras の指摘する量。ゼロ和ゲームでは MWU でこれが増え続ける
    energy = np.array([kl_to(nash3, xs[t]) + kl_to(nash3, ys[t]) for t in range(0, STEPS + 1, 50)])
    avg = running_mean(xs)
    avg_err = np.linalg.norm(avg - nash3, axis=1)
    rps_runs[str(eta)] = {
        "energy_start": float(energy[1]),
        "energy_end": float(energy[-1]),
        "energy_ratio": float(energy[-1] / energy[1]),
        "iterate_max_prob_end": float(xs[-1].max()),
        "timeavg_err_at_1k": float(avg_err[1000]),
        "timeavg_err_at_10k": float(avg_err[10_000]),
        "timeavg_err_end": float(avg_err[-1]),
        "timeavg_end": xs[-1 * 1].tolist(),
        "timeavg_vector_end": avg[-1].tolist(),
    }
    np.save(f"rps_x_eta{eta}.npy", xs[:: max(1, STEPS // 20000)])
    np.save(f"rps_y_eta{eta}.npy", ys[:: max(1, STEPS // 20000)])
results["rps"] = rps_runs

# レプリケータ極限の保存量。標準じゃんけんでは x1*x2*x3 が保存する
xs_tiny, ys_tiny = mwu(RPS_A, RPS_B, x0, y0, 0.0005, 400_000)
prod = xs_tiny.prod(axis=1)
results["replicator_limit"] = {
    "eta": 0.0005,
    "steps": 400_000,
    "invariant_start": float(prod[0]),
    "invariant_end": float(prod[-1]),
    "invariant_drift_ratio": float(prod[-1] / prod[0]),
    "min_prob_over_run": float(xs_tiny.min()),
    "max_prob_over_run": float(xs_tiny.max()),
}
np.save("rps_x_replicator.npy", xs_tiny[::20])
np.save("rps_y_replicator.npy", ys_tiny[::20])

# --- 実験2: 囚人のジレンマと協調ゲーム ---------------------------------------
pd_x0 = normalize(np.array([0.9, 0.1]))  # ほぼ協調から始める
pd_xs, pd_ys = mwu(PD_A, PD_B, pd_x0, pd_x0.copy(), 0.05, 2000)
defect = pd_xs[:, 1]
reach = int(np.argmax(defect > 0.99)) if (defect > 0.99).any() else -1
results["pd"] = {
    "eta": 0.05,
    "start_defect_prob": float(defect[0]),
    "defect_prob_at_100": float(defect[100]),
    "defect_prob_at_500": float(defect[500]),
    "defect_prob_end": float(defect[-1]),
    "steps_to_99pct_defect": reach,
}
np.save("pd_x.npy", pd_xs)

# 協調ゲーム: 初期値を振って、どちらの均衡に落ちるかを地図にする
GRID = 121
grid = np.linspace(0.001, 0.999, GRID)
basin = np.zeros((GRID, GRID), dtype=np.int8)
for i, px in enumerate(grid):
    for j, py in enumerate(grid):
        xs_c, ys_c = mwu(COORD_A, COORD_B, np.array([px, 1 - px]), np.array([py, 1 - py]), 0.1, 1500)
        basin[i, j] = 1 if xs_c[-1][0] > 0.5 else 0
np.save("coord_basin.npy", basin)
# 理論上の境界: 混合戦略均衡は行・列とも p = 1/3 (利得2と1の協調ゲーム)
share_good = float(basin.mean())
results["coord"] = {
    "eta": 0.1,
    "grid": GRID,
    "share_converged_to_payoff2": share_good,
    "mixed_nash_p": 1 / 3,
    "predicted_share_if_boundary_at_mixed_nash": None,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
