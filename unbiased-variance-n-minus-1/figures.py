"""記事の図。results.json（core.py の出力）を読んで描く。"""
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from core import MU, SIGMA, c4

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
res = json.load(open("results.json", encoding="utf-8"))
NS = [int(k) for k in res["n"]]
R = {int(k): v for k, v in res["n"].items()}
BLUE, ORANGE, GRAY = "#1f77b4", "#d95f02", "#666666"
FS = 15


def fig1():
    rng = np.random.default_rng(7)
    n = 5
    fig, axes = plt.subplots(2, 1, figsize=(9, 10), gridspec_kw={"height_ratios": [1, 1.25]})
    # (a) 1 回の標本
    ax = axes[0]
    while True:
        x = rng.normal(MU, SIGMA, n)
        if abs(x.mean() - MU) > 4:
            break
    m = x.mean()
    ss_true = ((x - MU) ** 2).sum()
    ss_mean = ((x - m) ** 2).sum()
    ys = np.linspace(0.6, 2.4, n)
    ax.axvline(MU, color=BLUE, lw=2)
    ax.axvline(m, color=ORANGE, lw=2, ls="--")
    ax.text(m - 0.3, 3.05, "標本平均", color=ORANGE, ha="right", fontsize=FS, va="bottom")
    ax.text(MU + 0.3, 3.05, "真の平均", color=BLUE, ha="left", fontsize=FS, va="bottom")
    for xi, yi in zip(x, ys):
        ax.plot([MU, xi], [yi + 0.07, yi + 0.07], color=BLUE, lw=3, alpha=0.7)
        ax.plot([m, xi], [yi - 0.07, yi - 0.07], color=ORANGE, lw=3, alpha=0.9)
        ax.plot(xi, yi, "o", color="black", ms=9, zorder=5)
    ax.set_ylim(0.2, 3.7)
    ax.set_yticks([])
    ax.set_xlabel("値", fontsize=FS)
    ax.set_title(f"n=5 の標本1つ：偏差平方和は 真の平均から {ss_true:.0f}、標本平均から {ss_mean:.0f}", fontsize=FS)
    ax.tick_params(labelsize=FS - 2)
    # (b) 300 標本
    ax = axes[1]
    k = 300
    X = rng.normal(MU, SIGMA, (k, n))
    a = ((X - MU) ** 2).sum(axis=1) / n
    b = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) / n
    ax.scatter(a, b, s=18, color=GRAY, alpha=0.6)
    lim = max(a.max(), b.max()) * 1.03
    ax.plot([0, lim], [0, lim], color="black", lw=1.5)
    ax.text(lim * 0.55, lim * 0.62, "等しい線", fontsize=FS - 2, rotation=33)
    ax.plot(a.mean(), b.mean(), "o", color=ORANGE, ms=14, zorder=6)
    ax.annotate(f"平均 ({a.mean():.0f}, {b.mean():.0f})", (a.mean(), b.mean()), xytext=(a.mean() + 110, b.mean() - 45),
                fontsize=FS, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("真の平均からの偏差平方和 ÷ n", fontsize=FS)
    ax.set_ylabel("標本平均からの偏差平方和 ÷ n", fontsize=FS)
    ax.set_title(f"300 個の標本（n=5）。すべて等しい線の下側にある", fontsize=FS)
    ax.tick_params(labelsize=FS - 2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/1.png", dpi=130)
    plt.close(fig)
    print("fig1", f"n={n} ss_true={ss_true:.1f} ss_mean={ss_mean:.1f} below={(b < a).sum()}/{k} mean_a={a.mean():.1f} mean_b={b.mean():.1f}")


def fig2():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    nn = np.arange(2, 31)
    ax.plot(nn, (nn - 1) / nn, color=GRAY, lw=2, label="理論値 (n−1)/n")
    ax.axhline(1, color=BLUE, lw=2, label="n−1 で割った分散（理論 1）")
    ax.plot(NS, [R[n]["ratio_n"] for n in NS], "o", color=ORANGE, ms=11, label="n で割った分散の平均 ÷ 真の分散（実測）")
    ax.plot(NS, [R[n]["var_n1"] / SIGMA ** 2 for n in NS], "s", color=BLUE, ms=9, label="n−1 で割った分散の平均 ÷ 真の分散（実測）")
    for n in NS:
        ax.annotate(f"{R[n]['ratio_n']:.3f}", (n, R[n]["ratio_n"]), xytext=(4, -22), textcoords="offset points", fontsize=FS - 3, color=ORANGE)
    ax.set_ylim(0.4, 1.08)
    ax.set_xlabel("標本の大きさ n", fontsize=FS)
    ax.set_ylabel("真の分散に対する比", fontsize=FS)
    ax.set_title("標本を40万個×5回取った平均", fontsize=FS)
    ax.legend(fontsize=FS - 3, loc="lower right")
    ax.tick_params(labelsize=FS - 2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/2.png", dpi=130)
    plt.close(fig)


def fig3():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    nn = np.arange(2, 31)
    ax.plot(nn, [c4(int(n)) for n in nn], color=GRAY, lw=2, label="理論値 c₄(n)")
    ax.axhline(1, color=BLUE, lw=1.5, ls=":")
    ax.plot(NS, [R[n]["s_ratio"] for n in NS], "o", color=ORANGE, ms=11, label="s の平均 ÷ σ（実測）")
    for n in NS:
        ax.annotate(f"{R[n]['s_ratio']:.3f}", (n, R[n]["s_ratio"]), xytext=(4, -22), textcoords="offset points", fontsize=FS - 3, color=ORANGE)
    ax.text(30, 1.004, "σ（真の標準偏差）", ha="right", color=BLUE, fontsize=FS - 3)
    ax.set_ylim(0.76, 1.03)
    ax.set_xlabel("標本の大きさ n", fontsize=FS)
    ax.set_ylabel("標本標準偏差 s の平均 ÷ σ", fontsize=FS)
    ax.set_title("n−1 で割った分散の平方根は σ より小さく出る", fontsize=FS)
    ax.legend(fontsize=FS - 3, loc="lower right")
    ax.tick_params(labelsize=FS - 2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/3.png", dpi=130)
    plt.close(fig)


def fig4():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    w = 0.26
    x = np.arange(len(NS))
    base = [R[n]["mse_n1"] for n in NS]
    for i, (key, lab, col) in enumerate([("mse_n1", "n−1 で割る（不偏）", BLUE), ("mse_n", "n で割る", GRAY), ("mse_n2", "n+1 で割る", ORANGE)]):
        v = [R[n][key] / b for n, b in zip(NS, base)]
        bars = ax.bar(x + (i - 1) * w, v, w, color=col, label=lab)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=FS - 5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in NS], fontsize=FS - 2)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("平均二乗誤差（n−1 で割った場合を 1 とした比）", fontsize=FS - 3)
    ax.set_title("真の分散との誤差の二乗平均。小さいほど当たる", fontsize=FS)
    ax.legend(fontsize=FS - 3, loc="upper right", ncol=3)
    ax.tick_params(labelsize=FS - 2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/4.png", dpi=130)
    plt.close(fig)
    for n in NS:
        print("mse ratio", n, R[n]["mse_n"] / R[n]["mse_n1"], R[n]["mse_n2"] / R[n]["mse_n1"])


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
