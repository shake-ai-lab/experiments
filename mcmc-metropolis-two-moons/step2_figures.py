# -*- coding: utf-8 -*-
"""note 記事用の図版と数式画像。note は LaTeX を描画できないので、
数式はすべて PNG にして貼る。

注意: fig4 / fig5 / fig6 はここで出したものに問題があり、step3_fix_figures.py で
作り直している。記事に貼ってあるのは step3 の出力の方。"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from step1_mcmc import log_p_tilde, autocorr  # noqa: E402

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 15, "axes.labelsize": 13})

SRC = Path("C:/tmp/mcmc-experiment")
OUT = Path("C:/Users/kouta/project/forsubwork/note-assets/images/mcmc-metropolis")
OUT.mkdir(parents=True, exist_ok=True)
res = json.load(open(SRC / "results.json", encoding="utf-8"))

BLUE, RED, GRAY = "#2f6fb0", "#d1495b", "0.45"


def density_grid(gap=0.0, lim=(-2.2, 3.2, -2.2, 2.2), n=400):
    xs = np.linspace(lim[0], lim[1], n)
    ys = np.linspace(lim[2], lim[3], n)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    Z = np.exp(log_p_tilde(P, gap)).reshape(n, n)
    return xs, ys, Z


# ============ 1. 標的分布 ============
xs, ys, Z = density_grid()
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.pcolormesh(xs, ys, Z, cmap="magma", shading="auto")
ax.set_title("標的分布 — 形は分かっているが、正規化定数は分からない")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "fig1_target.png", dpi=140, bbox_inches="tight")
print("fig1")

# ============ 2. チェーンの最初の歩み ============
d = np.load(SRC / "chain_main.npz")
chain = d["chain"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, k, ttl in zip(axes, [100, 2000, len(chain)],
                      ["最初の100歩", "最初の2,000歩", f"全{len(chain)*5:,}歩"]):
    ax.pcolormesh(xs, ys, Z, cmap="magma", shading="auto", alpha=0.55)
    c = chain[:k]
    if k <= 2000:
        ax.plot(c[:, 0], c[:, 1], lw=0.7, color="#7fd4ff", alpha=0.9)
        ax.scatter(c[:, 0], c[:, 1], s=6, color="white", alpha=0.8, zorder=3)
    else:
        ax.scatter(c[:, 0], c[:, 1], s=0.6, color="white", alpha=0.12)
    ax.set_title(ttl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
fig.suptitle("1本の点が歩き回るだけで、標的分布の形が塗りつぶされていく", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig2_walk.png", dpi=130, bbox_inches="tight")
print("fig2")

# ============ 3. ステップ幅スイープ ============
sw = res["sweep"]
st = [s["step"] for s in sw]; acc = [s["acceptance"] for s in sw]; e = [s["ess"] for s in sw]
fig, ax1 = plt.subplots(figsize=(7.6, 5.0))
ax1.semilogx(st, acc, "o-", color=BLUE, lw=2.4, label="受容率")
ax1.set_xlabel("提案のステップ幅"); ax1.set_ylabel("受容率", color=BLUE)
ax1.tick_params(axis="y", labelcolor=BLUE); ax1.set_ylim(0, 1.05)
ax2 = ax1.twinx()
ax2.semilogx(st, e, "s--", color=RED, lw=2.4, label="有効サンプルサイズ")
ax2.set_ylabel("有効サンプルサイズ(大きいほど良い)", color=RED)
ax2.tick_params(axis="y", labelcolor=RED)
best = max(sw, key=lambda s: s["ess"])
ax2.annotate(f"最良: ステップ幅{best['step']}\n受容率{best['acceptance']:.2f} / ESS {best['ess']:.0f}",
             xy=(best["step"], best["ess"]), xytext=(best["step"] * 1.6, best["ess"] * 0.72),
             arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=12)
ax1.set_title("小さすぎても大きすぎてもダメ — ちょうど良い歩幅がある")
fig.tight_layout(); fig.savefig(OUT / "fig3_stepsize.png", dpi=140, bbox_inches="tight")
print("fig3")

# ============ 4. バーンイン ============
cb = np.load(SRC / "chain_burnin.npz")["chain"]
fig, ax = plt.subplots(figsize=(6.6, 5.4))
ax.pcolormesh(xs, ys, Z, cmap="magma", shading="auto", alpha=0.6)
ax.plot(cb[:400, 0], cb[:400, 1], lw=1.1, color="#7fd4ff")
ax.scatter([cb[0, 0]], [cb[0, 1]], s=90, color="#ffd166", zorder=5, label="出発点 (6, 6)")
n_burn = res["burn_in_steps_to_reach_target"]
ax.scatter([cb[n_burn, 0]], [cb[n_burn, 1]], s=90, color="#7CFC9B", zorder=5,
           label=f"{n_burn}歩で標的に到達")
ax.set_xlim(-2.5, 6.6); ax.set_ylim(-2.5, 6.6)
ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=11, loc="upper left")
ax.set_title("見当違いの場所から始めても、勝手に標的まで降りてくる")
fig.tight_layout(); fig.savefig(OUT / "fig4_burnin.png", dpi=140, bbox_inches="tight")
print("fig4")

# ============ 5. 多峰性での失敗 ============
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
for ax, tag, ttl in zip(axes, ["near", "far"],
                        ["三日月が近い場合", "三日月を引き離した場合"]):
    dd = np.load(SRC / f"chain_gap_{tag}.npz")
    g = float(dd["gap"]); c = dd["chain"]
    lim = (-2.6, 3.6, -3.6 - g, 3.0 + g)
    gx, gy, gz = density_grid(g, lim, 300)
    ax.pcolormesh(gx, gy, gz, cmap="magma", shading="auto", alpha=0.55)
    ax.scatter(c[:, 0], c[:, 1], s=0.6, color="white", alpha=0.10)
    r = res[f"multimodal_{tag}"]
    ax.set_title(f"{ttl}\n上の弧に居た割合 {r['mode1_fraction']*100:.1f}% / "
                 f"モード往復 {r['mode_switches']:,}回")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("離れた山は、20万歩あっても一度も跨げない", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig5_multimodal.png", dpi=130, bbox_inches="tight")
print("fig5")

# ============ 6. 自己相関 ============
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for st_, col in [(0.05, "#8ecae6"), (0.35, BLUE), (1.0, RED)]:
    from step1_mcmc import metropolis
    c, _ = metropolis(30_000, step=st_, x0=[0.0, 0.0], seed=7)
    ac = autocorr(c[2000:, 0], 150)
    ax.plot(ac, lw=2.2, color=col, label=f"ステップ幅 {st_}")
ax.axhline(0, color="0.6", lw=1)
ax.set_xlabel("何歩離れたサンプル同士か"); ax.set_ylabel("相関")
ax.set_title("連続するサンプルは似すぎている — だから20万個が3,837個分にしかならない")
ax.legend(fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig6_autocorr.png", dpi=140, bbox_inches="tight")
print("fig6")

# ============ 数式画像 ============
FORMULAS = [
    ("f01_target", r"$p(x) = \frac{\tilde{p}(x)}{Z}, \qquad Z = \int \tilde{p}(x)\, dx \;\; (\mathrm{unknown})$"),
    ("f02_detailed_balance", r"$p(x)\, T(x \to x') \;=\; p(x')\, T(x' \to x)$"),
    ("f03_accept", r"$A(x \to x') = \min\left(1,\; \frac{p(x')\, q(x \mid x')}{p(x)\, q(x' \mid x)}\right)$"),
    ("f04_cancel", r"$\frac{p(x')}{p(x)} = \frac{\tilde{p}(x')/Z}{\tilde{p}(x)/Z} = \frac{\tilde{p}(x')}{\tilde{p}(x)}$"),
    ("f05_symmetric", r"$q(x' \mid x) = \mathcal{N}(x'; x, \sigma^2 I) \;\Rightarrow\; q(x' \mid x) = q(x \mid x')$"),
    ("f06_metropolis", r"$A(x \to x') = \min\left(1,\; \frac{\tilde{p}(x')}{\tilde{p}(x)}\right)$"),
    ("f07_log", r"$\log u < \log \tilde{p}(x') - \log \tilde{p}(x), \qquad u \sim \mathrm{Uniform}(0,1)$"),
    ("f08_ess", r"$\mathrm{ESS} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}$"),
    ("f09_target_def", r"$\tilde{p}(x) = \exp\left(-\frac{d_1(x)^2}{2\sigma^2}\right) + \exp\left(-\frac{d_2(x)^2}{2\sigma^2}\right)$"),
]
for name, tex in FORMULAS:
    fig, ax = plt.subplots(figsize=(9, 1.7))
    ax.axis("off")
    ax.text(0.5, 0.5, tex, fontsize=22, ha="center", va="center")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(name)

print("\nsaved ->", OUT)
