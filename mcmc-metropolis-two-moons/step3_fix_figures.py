# -*- coding: utf-8 -*-
"""step2 が出した図のうち fig4/fig5/fig6 を作り直す。

step2 版には次の問題があった:
  fig4: 密度グリッドが視野より狭く、出発点(6,6)が画面外に出ていた
  fig5: 左右で縮尺が違い、白=サンプル / オレンジ=分布 の区別も説明がなかった
  fig6: タイトルのESSがバグ修正前の古い値(3,837)のままハードコードされていて、
        本文の実測値(1,362)と食い違っていた -> results.json から読むように変更

step1 -> step2 -> step3 の順に実行すること。
"""


# ======================================================================
# fix_fig4.py
# ======================================================================
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "Meiryo"
SIGMA = 0.18
SRC = Path("C:/tmp/mcmc-experiment")
OUT = Path("C:/Users/kouta/project/forsubwork/note-assets/images/mcmc-metropolis")


def _dist_to_arc(p, center, r, t0, t1):
    d = p - center
    ang = np.arctan2(d[:, 1], d[:, 0])
    rad = np.linalg.norm(d, axis=1)
    inside = (ang >= t0) & (ang <= t1)
    d_end = np.minimum.reduce([
        np.linalg.norm(p - (center + r * np.array([np.cos(t), np.sin(t)])), axis=1)
        for t in (t0, t1)])
    return np.where(inside, np.abs(rad - r), d_end)


def log_p_tilde(p):
    p = np.atleast_2d(p)
    d1 = _dist_to_arc(p, np.array([0.0, 0.0]), 1.0, 0.0, np.pi)
    d2 = _dist_to_arc(p, np.array([1.0, -0.5]), 1.0, -np.pi, 0.0)
    a = -(d1 ** 2) / (2 * SIGMA ** 2)
    b = -(d2 ** 2) / (2 * SIGMA ** 2)
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


res = json.load(open(SRC / "results.json", encoding="utf-8"))
cb = np.load(SRC / "chain_burnin.npz")["chain"]
n_burn = res["burn_in_steps_to_reach_target"]

# 視野いっぱいに密度を描く(出発点を含める)
LO, HI = -2.6, 7.2
n = 500
g = np.linspace(LO, HI, n)
X, Y = np.meshgrid(g, g)
Z = np.exp(log_p_tilde(np.stack([X.ravel(), Y.ravel()], 1))).reshape(n, n)

fig, ax = plt.subplots(figsize=(7.0, 6.6))
ax.pcolormesh(g, g, Z, cmap="magma", shading="auto")

# 到達までの軌跡と、その後の歩き回りを描き分ける
ax.plot(cb[:n_burn + 1, 0], cb[:n_burn + 1, 1], lw=2.0, color="#ffd166",
        zorder=3, label=f"標的に降りるまでの{n_burn}歩(捨てる)")
ax.plot(cb[n_burn:600, 0], cb[n_burn:600, 1], lw=0.8, color="#7fd4ff",
        alpha=0.9, zorder=2, label="到達後の歩き回り(使う)")
ax.scatter([cb[0, 0]], [cb[0, 1]], s=150, color="#ffd166", edgecolor="k",
           lw=0.8, zorder=5)
ax.annotate("出発点 (6, 6)", xy=(cb[0, 0], cb[0, 1]), xytext=(-70, -28),
            textcoords="offset points", fontsize=13, color="white",
            arrowprops=dict(arrowstyle="->", color="white"))
ax.scatter([cb[n_burn, 0]], [cb[n_burn, 1]], s=150, color="#7CFC9B",
           edgecolor="k", lw=0.8, zorder=5)
ax.annotate(f"{n_burn}歩で標的に到達", xy=(cb[n_burn, 0], cb[n_burn, 1]),
            xytext=(30, -50), textcoords="offset points", fontsize=13,
            color="white", arrowprops=dict(arrowstyle="->", color="white"))

ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=11, loc="lower right", framealpha=0.85)
ax.set_title("見当違いの場所から始めても、勝手に標的まで降りてくる", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "fig4_burnin.png", dpi=140, bbox_inches="tight")
print("fig4 regenerated")


# ======================================================================
# fix_fig5.py
# ======================================================================
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "Meiryo"
SIGMA = 0.18
SRC = Path("C:/tmp/mcmc-experiment")
OUT = Path("C:/Users/kouta/project/forsubwork/note-assets/images/mcmc-metropolis")


def _dist_to_arc(p, center, r, t0, t1):
    d = p - center
    ang = np.arctan2(d[:, 1], d[:, 0])
    rad = np.linalg.norm(d, axis=1)
    inside = (ang >= t0) & (ang <= t1)
    d_end = np.minimum.reduce([
        np.linalg.norm(p - (center + r * np.array([np.cos(t), np.sin(t)])), axis=1)
        for t in (t0, t1)])
    return np.where(inside, np.abs(rad - r), d_end)


def log_p(p, gap):
    p = np.atleast_2d(p)
    d1 = _dist_to_arc(p, np.array([0.0, gap]), 1.0, 0.0, np.pi)
    d2 = _dist_to_arc(p, np.array([1.0, -0.5 - gap]), 1.0, -np.pi, 0.0)
    a = -(d1 ** 2) / (2 * SIGMA ** 2)
    b = -(d2 ** 2) / (2 * SIGMA ** 2)
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


res = json.load(open(SRC / "results.json", encoding="utf-8"))

# 左右で完全に同じ視野・同じ縮尺にする(そうしないと比較にならない)
XL, XH, YL, YH = -2.0, 3.0, -4.6, 4.1
nx, ny = 300, 520
gx = np.linspace(XL, XH, nx)
gy = np.linspace(YL, YH, ny)
X, Y = np.meshgrid(gx, gy)
P = np.stack([X.ravel(), Y.ravel()], 1)

fig, axes = plt.subplots(1, 2, figsize=(11.6, 8.2))
for ax, tag, ttl in zip(axes, ["near", "far"],
                        ["三日月が近い場合", "三日月を引き離した場合"]):
    d = np.load(SRC / f"chain_gap_{tag}.npz")
    gap = float(d["gap"]); c = d["chain"]
    Z = np.exp(log_p(P, gap)).reshape(ny, nx)
    ax.pcolormesh(gx, gy, Z, cmap="magma", shading="auto")
    ax.scatter(c[:, 0], c[:, 1], s=0.5, color="white", alpha=0.12, zorder=3)
    r = res[f"multimodal_{tag}"]
    ax.set_title(f"{ttl}\n受容率 {r['acceptance']*100:.0f}% / "
                 f"モード往復 {r['mode_switches']:,}回", fontsize=14)
    ax.set_xlim(XL, XH); ax.set_ylim(YL, YH)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

# 注釈: 色が何を意味するのかを図の中で言う
axes[0].annotate("どちらの三日月にも\n均等に訪れている", xy=(0.6, -1.2),
                 xytext=(0.05, 0.12), textcoords="axes fraction", fontsize=13,
                 color="white", ha="left",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.4))
axes[1].annotate("分布はあるのに\n12万歩で一度も来なかった", xy=(0.35, -3.15),
                 xytext=(0.06, 0.32), textcoords="axes fraction", fontsize=13,
                 color="#ffd166", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#ffd166", lw=1.6))
axes[1].annotate("MCMCが訪れた場所\n(白い点)", xy=(0.35, 3.0),
                 xytext=(0.5, 0.93), textcoords="axes fraction", fontsize=13,
                 color="white", ha="left",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.4))

fig.suptitle("離れた山は、12万歩あっても一度も跨げない", fontsize=17)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "fig5_multimodal.png", dpi=140, bbox_inches="tight")
print("fig5 regenerated")


# ======================================================================
# fix_fig6.py
# ======================================================================
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "Meiryo"
SIGMA = 0.18
SRC = Path("C:/tmp/mcmc-experiment")
OUT = Path("C:/Users/kouta/project/forsubwork/note-assets/images/mcmc-metropolis")


def _dist_to_arc(p, center, r, t0, t1):
    d = p - center
    ang = np.arctan2(d[:, 1], d[:, 0])
    rad = np.linalg.norm(d, axis=1)
    inside = (ang >= t0) & (ang <= t1)
    d_end = np.minimum.reduce([
        np.linalg.norm(p - (center + r * np.array([np.cos(t), np.sin(t)])), axis=1)
        for t in (t0, t1)])
    return np.where(inside, np.abs(rad - r), d_end)


def log_p_tilde(p):
    p = np.atleast_2d(p)
    d1 = _dist_to_arc(p, np.array([0.0, 0.0]), 1.0, 0.0, np.pi)
    d2 = _dist_to_arc(p, np.array([1.0, -0.5]), 1.0, -np.pi, 0.0)
    a = -(d1 ** 2) / (2 * SIGMA ** 2)
    b = -(d2 ** 2) / (2 * SIGMA ** 2)
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def metropolis(n_steps, step, x0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, float)
    lp = log_p_tilde(x)[0]
    chain = np.empty((n_steps, 2))
    for i in range(n_steps):
        prop = x + rng.normal(0, step, 2)
        lp_prop = log_p_tilde(prop)[0]
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop
        chain[i] = x
    return chain


def autocorr(x, maxlag=150):
    x = x - x.mean()
    return np.array([1.0 if k == 0 else np.corrcoef(x[:-k], x[k:])[0, 1]
                     for k in range(maxlag)])


res = json.load(open(SRC / "results.json", encoding="utf-8"))
N = res["main"]["steps"]
ESS = res["main"]["ess"]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
for st, col in [(0.05, "#8ecae6"), (0.35, "#2f6fb0"), (1.0, "#d1495b")]:
    c = metropolis(30_000, step=st, x0=[0.0, 0.0], seed=7)
    ax.plot(autocorr(c[2000:, 0]), lw=2.2, color=col, label=f"ステップ幅 {st}")
ax.axhline(0, color="0.6", lw=1)
ax.set_xlabel("何歩離れたサンプル同士か")
ax.set_ylabel("相関")
ax.set_title(f"連続するサンプルは似すぎている — "
             f"だから{N//10000}万個が{ESS:,.0f}個分にしかならない", fontsize=14)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig6_autocorr.png", dpi=140, bbox_inches="tight")
print(f"fig6 regenerated (N={N:,}, ESS={ESS:,.0f})")
