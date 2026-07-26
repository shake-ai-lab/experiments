"""可視化一式:
(1) 審査員 x ライター 平均スコアのヒートマップ(3x3)
(2) 自己スタイル一致ボーナス(質問ごとのペア差、黒点で全16問を見せる)
(3) 甘口バイアス(審査員ごとの全48件平均スコア、有意差なしのANOVA結果も注記)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11, "lines.linewidth": 2.0})

OUTDIR = "C:/tmp/llm-judge-self-preference-bias"
d = np.load(f"{OUTDIR}/results.npz")

matrix = d["matrix"]
writers = list(d["writers"])
judge_names = list(d["judge_names"])
labels_jp = {"P_concise": "簡潔", "Q_friendly": "親しみ", "R_structured": "網羅性"}
axis_labels = [labels_jp[w] for w in writers]

COL_P, COL_Q, COL_R = "#2e6ee4", "#e4572e", "#2fa96b"
COLORS = {"P_concise": COL_P, "Q_friendly": COL_Q, "R_structured": COL_R}

# ---------------- (1) heatmap ----------------
fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(matrix, cmap="YlOrRd", vmin=3, vmax=9.5)
ax.set_xticks(range(3))
ax.set_xticklabels([f"ライター\n{a}" for a in axis_labels], fontsize=10.5)
ax.set_yticks(range(3))
ax.set_yticklabels([f"審査員\n{a}重視" for a in axis_labels], fontsize=10.5)
for i in range(3):
    for j in range(3):
        val = matrix[i, j]
        diag = (i == j)
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=15 if diag else 12.5, fontweight="bold" if diag else "normal",
                color="white" if val > 6.5 else "black")
        if diag:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.5))
ax.set_title("審査員 x ライター 平均スコア(1〜10点、n=16問/セル)\n対角線(黒枠)=審査基準とライターのスタイルが一致するセル", fontsize=11.5)
fig.colorbar(im, ax=ax, shrink=0.8, label="平均スコア")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig1_heatmap.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig1_heatmap.png")

# ---------------- (2) self-style-alignment bonus, per judge, per-question dots ----------------
fig, ax = plt.subplots(figsize=(8.5, 5.4))
diffs_list = [d["p_diffs"], d["q_diffs"], d["r_diffs"]]
tvals = [float(d["p_t"]), float(d["q_t"]), float(d["r_t"])]
pvals = [float(d["p_p"]), float(d["q_p"]), float(d["r_p"])]
xpos = [0, 1, 2]

ax.axhline(0, color="#999", lw=1, ls="--")
for xi, jname, diffs, c in zip(xpos, judge_names, diffs_list, [COL_P, COL_Q, COL_R]):
    ax.bar(xi, diffs.mean(), yerr=diffs.std(), width=0.6, color=c, alpha=0.55, edgecolor=c, capsize=5)
    rng = np.random.default_rng(hash(jname) % 1000)
    ax.scatter(np.full(len(diffs), xi) + rng.uniform(-0.15, 0.15, len(diffs)), diffs,
               s=22, color="black", alpha=0.55, zorder=5)

ax.set_xticks(xpos)
ax.set_xticklabels([f"審査員{labels_jp[j]}重視\nt={t:.1f}, p={p:.4f}" for j, t, p in zip(judge_names, tvals, pvals)],
                    fontsize=9.5)
ax.set_ylabel("自己スタイル一致ボーナス\n(自分の基準に合うライター − 他2ライター平均)")
ax.set_title("どの審査員も、自分の評価基準に一致するスタイルのライターを一貫して高く採点した\n(黒点は16問それぞれの実測値、対応のあるt検定)", fontsize=11.3)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig2_alignment_bonus.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig2_alignment_bonus.png")

# ---------------- (3) leniency comparison ----------------
fig, ax = plt.subplots(figsize=(6.8, 5.0))
means = [float(d["p_lenient_mean"]), float(d["q_lenient_mean"]), float(d["r_lenient_mean"])]
stds = [float(d["p_lenient_std"]), float(d["q_lenient_std"]), float(d["r_lenient_std"])]
colors = [COL_P, COL_Q, COL_R]
for xi, jname, m, s, c in zip(xpos, judge_names, means, stds, colors):
    ax.bar(xi, m, yerr=s, width=0.55, color=c, alpha=0.6, edgecolor=c, capsize=5)
    ax.text(xi, m + s + 0.15, f"{m:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels([f"審査員{labels_jp[j]}重視" for j in judge_names], fontsize=10.5)
ax.set_ylabel("48件全体の平均スコア(採点の甘さ)")
ax.set_ylim(0, 10)
fstat, pval_anova = float(d["fstat"]), float(d["pval_anova"])
ax.set_title(f"審査員間で「全体的な採点の甘さ」に有意差はなかった\n(一元配置ANOVA F={fstat:.2f}, p={pval_anova:.3f})", fontsize=11.5)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig3_leniency.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig3_leniency.png")

print("ALL VISUALS DONE")
