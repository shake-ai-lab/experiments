"""補助図: (fig5) 円ターゲットでのSA vs greedy比較
(fig6) 目標形状が小さすぎると統計量固定と両立できない、の実証"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Meiryo"

df = pd.read_csv("datasaurus_dozen.tsv", sep="\t")
dino = df[df["dataset"] == "dino"][["x", "y"]].to_numpy()
n = len(dino)
d = np.load("shape_results.npz")
SNAP = int(d["snap_every"])


def stats_of(pts):
    x, y = pts[:, 0], pts[:, 1]
    return np.array(
        [x.mean(), y.mean(), x.std(ddof=1), y.std(ddof=1), np.corrcoef(x, y)[0, 1]]
    )


init = stats_of(dino)

# ---- fig5: SA vs greedy (円) ----
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
h_sa, h_gr = d["circle_hist"], d["greedy_hist"]
axes[0].plot(np.arange(len(h_sa)) * SNAP, h_sa, color="#e4572e", label="焼きなまし")
axes[0].plot(np.arange(len(h_gr)) * SNAP, h_gr, color="#2e6ee4", label="greedy(悪化拒否)")
axes[0].set_xlabel("イテレーション")
axes[0].set_ylabel("円までの平均距離")
axes[0].legend()
axes[0].set_title("収束の比較 (目標: 円)")
for ax, pts, label, c in [
    (axes[1], d["circle_pts"], f"焼きなまし (最終 {h_sa[-1]:.2f})", "#e4572e"),
    (axes[2], d["greedy_pts"], f"greedy (最終 {h_gr[-1]:.2f})", "#2e6ee4"),
]:
    ax.scatter(pts[:, 0], pts[:, 1], s=5, color=c)
    ax.set_xlim(-5, 110)
    ax.set_ylim(-2, 108)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=10)
fig.tight_layout()
fig.savefig("fig5_sa_vs_greedy_circle.png", dpi=110, bbox_inches="tight")
print("saved fig5")


# ---- fig6: 小さすぎる魚は不可能 ----
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


fish_unit = np.array(
    [
        [0.50, 0.98], [0.68, 0.90], [0.80, 0.76], [0.84, 0.60], [0.80, 0.44],
        [0.68, 0.32], [0.62, 0.24], [0.82, 0.04], [0.50, 0.16], [0.18, 0.04],
        [0.38, 0.24], [0.32, 0.32], [0.20, 0.44], [0.16, 0.60], [0.20, 0.76],
        [0.32, 0.90],
    ]
)
CX = 54.26


def make_fish(scale):
    """scale=1が成功版と同じ大きさ。中心(CX, 50)基準で縮小する"""
    x = CX + (fish_unit[:, 0] - 0.5) * 58 * scale
    y = 50 + (4 + fish_unit[:, 1] * 92 - 50) * scale
    return np.c_[x, y]


def run(dist_fn, iters=800_000, seed=3):
    r = np.random.default_rng(seed)
    pts = dino.copy().astype(float)
    dists = np.array([dist_fn(p) for p in pts])
    acc = 0
    for it in range(iters):
        frac = it / iters
        T = 0.6 * (0.002 / 0.6) ** frac
        sigma = 1.2 * (0.25 / 1.2) ** frac
        i = r.integers(n)
        old = pts[i].copy()
        new = old + r.normal(0, sigma, 2)
        if not (0 <= new[0] <= 105 and 0 <= new[1] <= 105):
            continue
        d_old = dists[i]
        d_new = dist_fn(new)
        if d_new >= d_old:
            if T <= 0 or r.random() >= np.exp(-(d_new - d_old) / T):
                continue
        pts[i] = new
        if np.any(np.abs(stats_of(pts) - init) >= TOL):
            pts[i] = old
            continue
        dists[i] = d_new
        acc += 1
    return pts, dists.mean()


TOL = 0.01
scales = [0.45, 0.7, 1.0]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, s in zip(axes, scales):
    fish = make_fish(s)
    pts, md = run(seg_dist_factory(fish))
    ax.plot(
        np.r_[fish[:, 0], fish[0, 0]], np.r_[fish[:, 1], fish[0, 1]],
        color="#bbb", lw=1, zorder=1,
    )
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color="#e4572e", zorder=2)
    ax.set_xlim(-5, 110)
    ax.set_ylim(-2, 108)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"大きさ {int(s * 100)}% (平均距離 {md:.2f})", fontsize=10)
    print(f"scale {s}: mean dist {md:.3f}")
fig.suptitle("目標の魚が小さいと、SD固定(16.8, 26.9)と両立できず輪郭に乗れない", fontsize=11)
fig.tight_layout()
fig.savefig("fig6_size_feasibility.png", dpi=110, bbox_inches="tight")
print("saved fig6")
