"""dinoから出発し、統計量を小数2桁で固定したまま
円・星・X・サーモンの4形状へ焼きなましで変形する (原論文と同じ向き)。
目標は連続曲線(線分列)にし、点が輪郭上を滑れるようにする。"""
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

# ---- 目標形状の定義 (線分列 or 距離関数) ----
CX, CY = 54.26, 47.83


def circle_dist(p):
    return abs(np.hypot(p[0] - CX, p[1] - CY) - 33.0)


def polyline_segments(verts, close=False):
    v = np.asarray(verts, float)
    if close:
        v = np.vstack([v, v[:1]])
    return v[:-1], v[1:]


def seg_dist_factory(verts_list):
    """複数ポリラインへの最短距離関数を作る"""
    A_all, B_all = [], []
    for verts, close in verts_list:
        A, B = polyline_segments(verts, close)
        A_all.append(A)
        B_all.append(B)
    A = np.vstack(A_all)
    B = np.vstack(B_all)
    AB = B - A
    L2 = (AB**2).sum(1)
    L2[L2 == 0] = 1e-12

    def dist(p):
        t = np.clip(((p - A) * AB).sum(1) / L2, 0, 1)
        proj = A + t[:, None] * AB
        return np.sqrt(((p - proj) ** 2).sum(1).min())

    return dist


# 星 (5点星、中心CX,CY)
ang_out = np.deg2rad(90 + np.arange(5) * 72)
ang_in = np.deg2rad(126 + np.arange(5) * 72)
outer = np.c_[CX + 40 * np.cos(ang_out), CY + 40 * np.sin(ang_out)]
inner = np.c_[CX + 15 * np.cos(ang_in), CY + 15 * np.sin(ang_in)]
star_verts = np.empty((10, 2))
star_verts[0::2] = outer
star_verts[1::2] = inner
star_dist = seg_dist_factory([(star_verts, True)])

# X (2本の対角線)
x_dist = seg_dist_factory(
    [(np.array([[20, 8], [88, 88]]), False), (np.array([[20, 88], [88, 8]]), False)]
)

# サーモン (手描きシルエットのポリライン、単位座標→データ座標へスケール)
fish_unit = np.array(
    [
        [0.03, 0.50], [0.10, 0.62], [0.22, 0.72], [0.38, 0.78], [0.52, 0.76],
        [0.62, 0.68], [0.72, 0.56], [0.97, 0.78], [0.88, 0.50], [0.97, 0.22],
        [0.72, 0.44], [0.62, 0.32], [0.52, 0.24], [0.38, 0.22], [0.22, 0.28],
        [0.10, 0.38],
    ]
)
fish_y = 0.5 + (fish_unit[:, 1] - 0.5) * 1.7
fish = np.c_[8 + fish_unit[:, 0] * 88, 6 + fish_y * 88]
salmon_dist = seg_dist_factory([(fish, True)])

TARGETS = {
    "circle": circle_dist,
    "star": star_dist,
    "x_shape": x_dist,
    "salmon": salmon_dist,
}

TOL = 0.01
ITERS = 1_500_000
SNAP = 15_000


def run(dist_fn, annealing=True, seed=0):
    r = np.random.default_rng(seed)
    pts = dino.copy().astype(float)
    dists = np.array([dist_fn(p) for p in pts])
    snaps, hist, acc = [], [], 0
    for it in range(ITERS):
        if it % SNAP == 0:
            snaps.append(pts.copy())
            hist.append(dists.mean())
        frac = it / ITERS
        T = (0.6 * (0.002 / 0.6) ** frac) if annealing else 0.0
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
    snaps.append(pts.copy())
    hist.append(dists.mean())
    return pts, np.array(snaps), np.array(hist), acc


results = {}
for name, fn in TARGETS.items():
    pts, snaps, hist, acc = run(fn, annealing=True, seed=1)
    drift = np.abs(stats_of(pts) - init).max()
    print(f"{name:8s}: final mean dist {hist[-1]:.3f}, acc {acc}, max stats drift {drift:.4f}")
    results[name] = dict(pts=pts, snaps=snaps, hist=hist)

# greedy対決はcircleで
pts_g, snaps_g, hist_g, acc_g = run(circle_dist, annealing=False, seed=1)
print(f"greedy(circle): final mean dist {hist_g[-1]:.3f}, acc {acc_g}")

np.savez(
    "shape_results.npz",
    dino=dino,
    initial_stats=init,
    snap_every=SNAP,
    **{f"{k}_{f}": v[f] for k, v in results.items() for f in ("pts", "snaps", "hist")},
    greedy_pts=pts_g,
    greedy_snaps=snaps_g,
    greedy_hist=hist_g,
    fish_outline=fish,
    star_verts=star_verts,
)

# ---- fig4: 最終形状グリッド ----
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
axes[0].scatter(dino[:, 0], dino[:, 1], s=6, color="#333")
axes[0].set_title("start: dino", fontsize=10)
for ax, (name, res) in zip(axes[1:], results.items()):
    ax.scatter(res["pts"][:, 0], res["pts"][:, 1], s=6, color="#e4572e")
    ax.set_title(f"→ {name} (dist {res['hist'][-1]:.2f})", fontsize=10)
for ax in axes:
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 110)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("全て 平均(54.26,47.83)・SD(16.77,26.94)・相関−0.06 のまま (小数2桁固定)", fontsize=11)
fig.tight_layout()
fig.savefig("fig4_final_shapes.png", dpi=110, bbox_inches="tight")
print("saved fig4")
