"""clt.py の results.json から記事用の図を作る。"""

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

D = json.load(open("results.json", encoding="utf-8"))
R, H, NS = D["results"], D["hist"], D["ns"]
INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"
NORMAL = np.exp(-np.linspace(-5, 5, 400) ** 2 / 2) / np.sqrt(2 * np.pi)
XS = np.linspace(-5, 5, 400)

ORDER = [
    "一様分布 U(0,1)",
    "コイン投げ p=0.5",
    "指数分布 Exp(1)",
    "カイ二乗分布 df=1",
    "対数正規 σ=1",
    "まれな事象 p=0.01",
    "コーシー分布",
]


def draw_hist(ax, name, n, colour=ACC):
    h = H[name][str(n)]
    edges = np.array(h["edges"])
    centres = (edges[:-1] + edges[1:]) / 2
    ax.bar(centres, h["counts"], width=edges[1] - edges[0], color=colour, alpha=0.75, lw=0)
    ax.plot(XS, NORMAL, color=INK, lw=2.0)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0, 0.62)
    ax.set_yticks([])
    ax.tick_params(labelsize=11)


# ---------------------------------------------------------------- hero
# n=30 で6分布を並べ、正規分布の曲線を重ねる。合っていないことが一目で分かる図。
print("hero", flush=True)
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
for ax, name in zip(axes.ravel(), ORDER[:6]):
    draw_hist(ax, name, 30)
    ks = next(r["ks"] for r in R[name]["rows"] if r["n"] == 30)
    ok = ks < D["ks_practical"]
    ax.set_title(
        f"{name}\nKS = {ks:.3f}" + ("  ✓" if ok else "  ✗"),
        fontsize=14,
        color=INK if ok else ACC,
        pad=9,
    )
fig.suptitle("標本サイズ n = 30 の標本平均。青線が正規分布", fontsize=18, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("hero.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig1
# n を増やすと形がどう変わるか。収束の速い分布と遅い分布を並べる。
print("fig1", flush=True)
show = ["一様分布 U(0,1)", "指数分布 Exp(1)", "対数正規 σ=1"]
cols = [1, 5, 30, 1000]
fig, axes = plt.subplots(len(show), len(cols), figsize=(13.0, 8.4))
for i, name in enumerate(show):
    for j, n in enumerate(cols):
        ax = axes[i][j]
        draw_hist(ax, name, n)
        if i == 0:
            ax.set_title(f"n = {n}", fontsize=17, pad=10)
        if j == 0:
            ax.set_ylabel(name, fontsize=14)
        ks = next(r["ks"] for r in R[name]["rows"] if r["n"] == n)
        ax.text(
            0.03,
            0.93,
            f"KS {ks:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            color=INK if ks < D["ks_practical"] else ACC,
        )
fig.tight_layout()
fig.savefig("1.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig2
# 正規分布からのズレが n とともにどう減るか。分布によって桁違いに違う。
print("fig2", flush=True)
fig, ax = plt.subplots(figsize=(11.0, 6.6))
colours = [INK, "#6a9fb5", ACC, "#c1436d", WARN, "#b5651d", "#555555"]
for name, c in zip(ORDER, colours):
    ks = [r["ks"] for r in R[name]["rows"]]
    ax.plot(NS, ks, "o-", color=c, lw=2.0, ms=5, label=name)
ax.axhline(D["ks_practical"], ls="--", color=SUB, lw=1.6)
ax.text(120, D["ks_practical"] * 1.1, "実用の目安 KS = 0.01", fontsize=12, color=SUB)
ax.axvline(30, ls=":", color=SUB, lw=1.6)
ax.text(31, 0.0016, "n = 30", fontsize=13, color=SUB)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("標本サイズ n", fontsize=14)
ax.set_ylabel("正規分布からのズレ (KS統計量)", fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12, loc="lower left")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("2.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig3
# 歪度は理論上 母集団の歪度/sqrt(n) で減る。実測がその線に乗るかを見る。
print("fig3", flush=True)
fig, ax = plt.subplots(figsize=(11.0, 6.6))
for name, c in zip(ORDER[:6], colours):
    g = R[name]["pop_skew"]
    if not g:
        continue
    sk = [abs(r["skew"]) for r in R[name]["rows"]]
    ax.plot(NS, sk, "o", color=c, ms=6, label=f"{name} (母集団の歪度 {g:.2f})")
    ax.plot(NS, [g / np.sqrt(n) for n in NS], "-", color=c, lw=1.4, alpha=0.55)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("標本サイズ n", fontsize=14)
ax.set_ylabel("標本平均の歪度 (絶対値)", fontsize=14)
ax.set_title("点が実測、線が理論値 (母集団の歪度 ÷ √n)", fontsize=15)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("3.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig4
# コーシー分布は n を増やしても形が変わらない。
print("fig4", flush=True)
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9))
for ax, n in zip(axes, [1, 30, 100, 1000]):
    draw_hist(ax, "コーシー分布", n, colour="#555555")
    ax.set_title(f"n = {n}", fontsize=16, pad=8)
fig.suptitle("コーシー分布の標本平均。何個平均しても形が変わらない", fontsize=17, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("4.png", dpi=115)
plt.close(fig)

print("done")
