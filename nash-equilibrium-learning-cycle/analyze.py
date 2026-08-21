"""軌道から記事に使う量を測る。

  - exploitability: その戦略を相手に読まれたとき、どれだけ搾取されるか。
                    ナッシュ均衡なら0。じゃんけんで純戦略なら1(必ず負ける手を出される)。
  - 周期: 手が一周するのにかかるステップ数。回るたびに伸びていくかを見る。
"""

import json
import numpy as np

A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
NASH = np.full(3, 1 / 3)


def exploitability(x, y):
    """ゼロ和ゲームでの搾取されやすさ。0なら誰にも付け込まれない。"""
    return float((A @ y).max() + (-A.T @ x).max())


def running_mean(a):
    return np.cumsum(a, axis=0) / np.arange(1, len(a) + 1)[:, None]


out = {}
for eta in ["0.2", "0.05", "0.01", "0.002"]:
    xs = np.load(f"rps_x_eta{eta}.npy")
    ys = np.load(f"rps_y_eta{eta}.npy")
    ax_, ay_ = running_mean(xs), running_mean(ys)
    ex_iter = np.array([exploitability(xs[t], ys[t]) for t in range(len(xs))])
    ex_avg = np.array([exploitability(ax_[t], ay_[t]) for t in range(len(xs))])
    # 周期はここでは測らない。nash.py が保存した軌道は間引いてあるので、
    # ステップ数を数えると刻み幅ぶんずれる。周期の測定は periods.py が担当する。
    out[eta] = {
        "exploitability_iterate_last": float(ex_iter[-1]),
        "exploitability_iterate_mean_last_decile": float(ex_iter[-len(ex_iter) // 10 :].mean()),
        "exploitability_timeavg_last": float(ex_avg[-1]),
    }
    np.save(f"ex_iter_{eta}.npy", ex_iter)
    np.save(f"ex_avg_{eta}.npy", ex_avg)

# レプリケータ極限でも同じ量を測る
xr = np.load("rps_x_replicator.npy")
yr = np.load("rps_y_replicator.npy")
ex_r = np.array([exploitability(xr[t], yr[t]) for t in range(len(xr))])
out["replicator"] = {
    "exploitability_min": float(ex_r.min()),
    "exploitability_max": float(ex_r.max()),
    "exploitability_mean": float(ex_r.mean()),
}
np.save("ex_replicator.npy", ex_r)

# 均衡そのものの搾取されやすさ(0になるはず。物差しの校正)
out["nash_itself"] = {"exploitability": exploitability(NASH, NASH)}

with open("analysis.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
