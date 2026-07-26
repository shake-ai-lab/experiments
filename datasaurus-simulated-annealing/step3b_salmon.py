"""サーモンだけ作り直し: 縦向き(跳ねるサーモン)の輪郭で再実行し、npzとfig4を更新"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Meiryo"

df = pd.read_csv("datasaurus_dozen.tsv", sep="\t")
dino = df[df["dataset"] == "dino"][["x", "y"]].to_numpy()
n = len(dino)


def stats_of(pts):
    x, y = pts[:, 0], pts[:, 1]
    return np.array(
        [x.mean(), y.mean(), x.std(ddof=1), y.std(ddof=1), np.corrcoef(x, y)[0, 1]]
    )


init = stats_of(dino)
CX = 54.26

# 縦向きの魚 (頭が上、尾びれが下)。単位座標 (u=横, v=縦)
fish_unit = np.array(
    [
        [0.50, 0.98],  # 鼻先
        [0.68, 0.90], [0.80, 0.76], [0.84, 0.60], [0.80, 0.44], [0.68, 0.32],  # 右半身
        [0.62, 0.24],  # 尾の付け根右
        [0.82, 0.04],  # 尾びれ右端
        [0.50, 0.16],  # 尾びれの切れ込み
        [0.18, 0.04],  # 尾びれ左端
        [0.38, 0.24],  # 尾の付け根左
        [0.32, 0.32], [0.20, 0.44], [0.16, 0.60], [0.20, 0.76], [0.32, 0.90],  # 左半身
    ]
)
fish = np.c_[CX + (fish_unit[:, 0] - 0.5) * 58, 4 + fish_unit[:, 1] * 92]


def seg_dist_factory(verts, close=True):
    v = np.asarray(verts, float)
    if close:
        v = np.vstack([v, v[:1]])
    A, B = v[:-1], v[1:]
    AB = B - A
    L2 = (AB**2).sum(1)
    L2[L2 == 0] = 1e-12

    def dist(p):
        t = np.clip(((p - A) * AB).sum(1) / L2, 0, 1)
        proj = A + t[:, None] * AB
        return np.sqrt(((p - proj) ** 2).sum(1).min())

    return dist


salmon_dist = seg_dist_factory(fish)
TOL, ITERS, SNAP = 0.01, 1_500_000, 15_000

r = np.random.default_rng(7)
pts = dino.copy().astype(float)
dists = np.array([salmon_dist(p) for p in pts])
snaps, hist, acc = [], [], 0
for it in range(ITERS):
    if it % SNAP == 0:
        snaps.append(pts.copy())
        hist.append(dists.mean())
    frac = it / ITERS
    T = 0.6 * (0.002 / 0.6) ** frac
    sigma = 1.2 * (0.25 / 1.2) ** frac
    i = r.integers(n)
    old = pts[i].copy()
    new = old + r.normal(0, sigma, 2)
    if not (0 <= new[0] <= 105 and 0 <= new[1] <= 105):
        continue
    d_old = dists[i]
    d_new = salmon_dist(new)
    if d_new >= d_old:
        if T <= 0 or r.random() >= np.exp(-(d_new - d_old) / T):
            continue
    pts[i] = new
    if np.any(np.abs(stats_of(pts) - init) >= TOL):
        pts[i] = old
        continue
    dists[i] = d_new
    acc += 1
snaps.append(pts.copy())
hist.append(dists.mean())
print(f"salmon(vertical): final mean dist {hist[-1]:.3f}, acc {acc}, drift {np.abs(stats_of(pts)-init).max():.4f}")

# npz更新
data = dict(np.load("shape_results.npz"))
data["salmon_pts"] = pts
data["salmon_snaps"] = np.array(snaps)
data["salmon_hist"] = np.array(hist)
data["fish_outline"] = fish
np.savez("shape_results.npz", **data)

# fig4更新
names = ["circle", "star", "x_shape", "salmon"]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
axes[0].scatter(dino[:, 0], dino[:, 1], s=6, color="#333")
axes[0].set_title("start: dino", fontsize=10)
for ax, name in zip(axes[1:], names):
    p = data[f"{name}_pts"] if name != "salmon" else pts
    h = data[f"{name}_hist"] if name != "salmon" else hist
    ax.scatter(p[:, 0], p[:, 1], s=6, color="#e4572e")
    ax.set_title(f"→ {name} (dist {h[-1]:.2f})", fontsize=10)
for ax in axes:
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 110)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("全て 平均(54.26,47.83)・SD(16.77,26.94)・相関−0.06 のまま (小数2桁固定)", fontsize=11)
fig.tight_layout()
fig.savefig("fig4_final_shapes.png", dpi=110, bbox_inches="tight")
print("saved fig4")
