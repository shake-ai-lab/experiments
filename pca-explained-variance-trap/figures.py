"""pca.py の results.json から記事用の図を作る。"""

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

D = json.load(open("results.json", encoding="utf-8"))
S, SC = D["synthetic"], D["synthetic_scatter"]
INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"

y = np.array(SC["y"])
pc1, pcb = np.array(SC["pc1"]), np.array(SC["pc_best"])
BEST = SC["best_index"]
r1 = S["rows"][0]["ratio"] * 100
rb = S["rows"][BEST - 1]["ratio"] * 100

# ---------------------------------------------------------------- hero
# 分散のいちばん大きい方向と、分類に効く方向を縦に並べる。
print("hero", flush=True)
fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.6))
for ax, v, ratio, acc, label in [
    (axes[0], pc1, r1, S["rows"][0]["alone"], "第1主成分"),
    (axes[1], pcb, rb, S["rows"][BEST - 1]["alone"], f"第{BEST}主成分"),
]:
    bins = np.linspace(v.min(), v.max(), 44)
    ax.hist(v[y == 0], bins=bins, color=INK, alpha=0.72, label="クラスA")
    ax.hist(v[y == 1], bins=bins, color=ACC, alpha=0.72, label="クラスB")
    ax.set_title(
        f"{label}   寄与率 {ratio:.2f}%   この1本だけでの分類精度 {acc:.3f}",
        fontsize=19,
        pad=12,
    )
    ax.set_yticks([])
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=14)
fig.tight_layout()
fig.savefig("hero.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig1
# 主成分ごとの寄与率と、その1本だけで分類したときの精度。
print("fig1", flush=True)
ks = [r["component"] for r in S["rows"]]
fig, ax = plt.subplots(figsize=(11.4, 6.2))
ax.bar(ks, [r["ratio"] * 100 for r in S["rows"]], color=SUB, alpha=0.85, label="寄与率")
ax.set_yscale("log")
ax.set_ylabel("寄与率 (%、対数目盛)", fontsize=14)
ax.set_xlabel("主成分の番号", fontsize=14)
ax.set_xticks(ks)
ax.tick_params(labelsize=12)
ax2 = ax.twinx()
ax2.plot(ks, [r["alone"] for r in S["rows"]], "o-", color=ACC, lw=2.2, ms=7, label="単独での精度")
ax2.axhline(0.5, ls="--", color=SUB, lw=1.4)
ax2.set_ylabel("その主成分1本だけでの分類精度", fontsize=14, color=ACC)
ax2.set_ylim(0.40, 1.02)
ax2.tick_params(labelsize=12, colors=ACC)
ax.axvspan(0.5, S["cum80_k"] + 0.5, color=INK, alpha=0.10)
ax.text(
    S["cum80_k"] + 0.7,
    50,
    f"累積寄与率80%で切ると\nここまで(k={S['cum80_k']})",
    fontsize=13,
    color=INK,
    va="center",
)
fig.tight_layout()
fig.savefig("1.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig2
# 実データ。累積寄与率と分類精度が別々に伸びる。
print("fig2", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
for ax, key, title in [
    (axes[0], "digits", "手書き数字 (64次元)"),
    (axes[1], "breast_cancer", "乳がん (30次元)"),
]:
    x = D[key]
    ks = [r["k"] for r in x["rows"]]
    ax.plot(ks, [r["cum_ratio"] for r in x["rows"]], "o-", color=INK, lw=2.2, ms=4, label="累積寄与率")
    ax.plot(ks, [r["acc"] for r in x["rows"]], "o-", color=ACC, lw=2.2, ms=4, label="分類精度")
    ax.axhline(0.80, ls="--", color=SUB, lw=1.4)
    ax.axvline(x["k_for_80"], ls=":", color=INK, lw=1.8)
    ax.text(
        x["k_for_80"] + 0.6,
        0.33,
        f"寄与率80%\nk={x['k_for_80']}",
        fontsize=12,
        color=INK,
    )
    if x["k_for_99pct_acc"]:
        ax.axvline(x["k_for_99pct_acc"], ls=":", color=ACC, lw=1.8)
        ax.text(
            x["k_for_99pct_acc"] + 0.6,
            0.16,
            f"精度が出そろう\nk={x['k_for_99pct_acc']}",
            fontsize=12,
            color=ACC,
        )
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("残した主成分の数 k", fontsize=13)
    ax.set_ylim(0.1, 1.03)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, loc="lower right")
fig.tight_layout()
fig.savefig("2.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig3
# 標準化しないと、単位の大きい列が第1主成分を占領する。
print("fig3", flush=True)
C = D["scaling"]
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
for ax, top, ratio, title in [
    (axes[0], C["raw_top"], C["raw_first_ratio"], "標準化しない"),
    (axes[1], C["std_top"], C["std_first_ratio"], "標準化する"),
]:
    names = [t["name"] for t in top][::-1]
    vals = [t["weight"] for t in top][::-1]
    ax.barh(names, vals, color=ACC if "しない" in title else INK, alpha=0.85)
    ax.set_xlim(0, 0.92)
    ax.set_title(f"{title}\n第1主成分の寄与率 {ratio*100:.1f}%", fontsize=15, pad=10)
    ax.set_xlabel("第1主成分への効き方(絶対値)", fontsize=13)
    ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig("3.png", dpi=115)
plt.close(fig)

print("done")
