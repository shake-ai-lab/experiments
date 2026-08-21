"""周期を実ステップ単位で測り直す。

nash.py が保存した軌道は間引いてあるので、周期の測定には使えない。
ここでは間引かずに回して、じゃんけんが一周するのに何ステップかかるかを数える。
"""

import json
import numpy as np

A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
B = -A


def normalize(v):
    return v / v.sum()


def run(eta, steps, x0, y0):
    x, y = x0.copy(), y0.copy()
    lead = np.empty(steps + 1, dtype=np.int8)
    lead[0] = x.argmax()
    for t in range(steps):
        u, v = A @ y, B.T @ x
        x = normalize(x * np.exp(eta * (u - u.max())))
        y = normalize(y * np.exp(eta * (v - v.max())))
        lead[t + 1] = x.argmax()
    return lead


x0 = normalize(np.array([0.4, 0.35, 0.25]))
y0 = normalize(np.array([0.3, 0.4, 0.3]))

out = {}
for eta in [0.2, 0.05, 0.01, 0.002]:
    lead = run(eta, 200_000, x0, y0)
    switch = np.flatnonzero(np.diff(lead) != 0)
    # じゃんけんは3手なので、一周 = 主役が3回入れ替わる
    laps = switch[::3]
    periods = np.diff(laps)
    np.save(f"periods_eta{eta}.npy", periods)  # 図は実測の周期そのものを描くのでここで保存する
    out[str(eta)] = {
        "n_laps": int(len(periods)),
        "first_period_steps": int(periods[0]) if len(periods) else None,
        "last_period_steps": int(periods[-1]) if len(periods) else None,
        "period_growth": float(periods[-1] / periods[0]) if len(periods) > 1 else None,
        "periods_head": periods[:6].tolist(),
        "periods_tail": periods[-4:].tolist(),
    }

with open("periods.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
