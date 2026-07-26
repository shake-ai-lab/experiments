"""
Generate PNGs + GIF for the curse-of-dimensionality blog post.
Requires results.json / distances.npz produced by experiment.py, plus a
fresh 10-seed robustness sweep (recomputed here, small & fast).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"

with open("results.json") as f:
    results = json.load(f)

npz = np.load("distances.npz")
DIMS_REPORT = [2, 3, 5, 10, 20, 50, 100, 300]
DIMS_GIF = sorted(int(k) for k in npz.files)

by_d = {r["d"]: r for r in results}

# ---------- Figure 1: ratio & CV vs d (log-x) ----------
ds = [r["d"] for r in results]
ratios = [r["ratio_max_min"] for r in results]
cvs = [r["cv"] for r in results]

fig, ax1 = plt.subplots(figsize=(7.5, 4.6), dpi=110)
ax1.set_xscale("log")
ax1.set_yscale("log")
l1, = ax1.plot(ds, ratios, "o-", color="#d1495b", label="最大距離/最小距離の比")
ax1.set_xlabel("次元数 d (log軸)")
ax1.set_ylabel("最大距離 / 最小距離 の比", color="#d1495b")
ax1.tick_params(axis="y", labelcolor="#d1495b")
ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1)

ax2 = ax1.twinx()
l2, = ax2.plot(ds, cvs, "s-", color="#1b6ca8", label="距離の変動係数 CV")
ax2.set_ylabel("変動係数 CV (std/mean)", color="#1b6ca8")
ax2.tick_params(axis="y", labelcolor="#1b6ca8")

ax1.set_title(f"次元数を上げると「最近傍」と「最遠傍」が区別できなくなる (N={by_d[2]['N']}点)")
lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
fig.tight_layout()
fig.savefig("1_ratio_cv_vs_d.png")
plt.close(fig)
print("saved 1_ratio_cv_vs_d.png")

# ---------- Robustness sweep: 10 seeds at a few d values ----------
def stats(d, N, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, size=(N, d))
    q = np.zeros(d)
    dist = np.linalg.norm(pts - q, axis=1)
    return dist.max() / dist.min(), dist.std() / dist.mean()

robust_dims = [2, 3, 5, 10, 300]
robust_ratio = {d: [] for d in robust_dims}
robust_cv = {d: [] for d in robust_dims}
for d in robust_dims:
    for seed in range(10):
        r, cv = stats(d, 2000, seed)
        robust_ratio[d].append(r)
        robust_cv[d].append(cv)

with open("robustness.json", "w") as f:
    json.dump({"ratio": robust_ratio, "cv": robust_cv}, f, indent=2)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 4.6), dpi=110)
axL.boxplot([robust_ratio[d] for d in robust_dims], labels=[str(d) for d in robust_dims])
axL.set_yscale("log")
axL.set_xlabel("次元数 d")
axL.set_ylabel("最大/最小 比 (log軸)")
axL.set_title("最大/最小比: 乱数seedを10通り変えた場合のばらつき")

axR.boxplot([robust_cv[d] for d in robust_dims], labels=[str(d) for d in robust_dims])
axR.set_xlabel("次元数 d")
axR.set_ylabel("変動係数 CV")
axR.set_title("変動係数CV: 同じ10通りのseedでのばらつき")

fig.suptitle("同じ実験を10回繰り返すと、比の指標は低次元でかなり暴れるがCVは安定している")
fig.tight_layout()
fig.savefig("2_seed_robustness.png")
plt.close(fig)
print("saved 2_seed_robustness.png")

# ---------- Figure 3: distance histogram at low-D vs high-D, normalized by mean ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=110)
for d, color, alpha in [(2, "#d1495b", 0.55), (300, "#1b6ca8", 0.55)]:
    dist = npz[str(d)]
    normed = dist / dist.mean()
    ax.hist(normed, bins=40, color=color, alpha=alpha, density=True,
            label=f"d={d} (比の範囲 {normed.min():.2f}〜{normed.max():.2f}倍)")
ax.set_xlabel("原点からの距離 ÷ その次元での平均距離")
ax.set_ylabel("密度")
ax.set_title("原点から2000点への距離の分布: d=2は横に広いが、d=300はほぼ1点に潰れる")
ax.legend()
fig.tight_layout()
fig.savefig("3_hist_low_vs_high_d.png")
plt.close(fig)
print("saved 3_hist_low_vs_high_d.png")

# ---------- GIF: histogram of normalized distances sweeping over d ----------
gif_dims = DIMS_GIF[::2]  # stride to keep file small
if gif_dims[-1] != DIMS_GIF[-1]:
    gif_dims.append(DIMS_GIF[-1])
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=72)

def update(i):
    ax.clear()
    d = gif_dims[i]
    dist = npz[str(d)]
    normed = dist / dist.mean()
    ax.hist(normed, bins=35, range=(0.0, 2.2), color="#1b6ca8", alpha=0.75)
    ax.set_xlim(0.0, 2.2)
    ax.set_ylim(0, 500)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1)
    r = by_d.get(d, {}).get("ratio_max_min")
    cv = by_d.get(d, {}).get("cv")
    if r is None:
        rr = dist.max() / dist.min()
        rcv = dist.std() / dist.mean()
    else:
        rr, rcv = r, cv
    ax.set_title(f"次元数 d = {d:>4}   最大/最小比 = {rr:6.2f}   CV = {rcv:.3f}")
    ax.set_xlabel("原点からの距離 ÷ 平均距離")
    ax.set_ylabel("点の個数")
    return []

anim = animation.FuncAnimation(fig, update, frames=len(gif_dims), interval=350)
anim.save("sweep.gif", writer=animation.PillowWriter(fps=3))
plt.close(fig)
print("saved sweep.gif", "frames:", len(gif_dims))

# shrink further: reduce palette size via PIL
from PIL import Image
im = Image.open("sweep.gif")
frames = []
try:
    while True:
        frames.append(im.copy().convert("P", palette=Image.ADAPTIVE, colors=64))
        im.seek(im.tell() + 1)
except EOFError:
    pass
frames[0].save("sweep.gif", save_all=True, append_images=frames[1:],
               duration=350, loop=0, optimize=True)
print("recompressed sweep.gif")
