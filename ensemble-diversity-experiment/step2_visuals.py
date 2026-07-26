"""可視化一式:
(0) セットアップ図解 (A: seed-only vs B: bagging+width)
(1) 精度ゲイン vs メンバー間不一致率の散布図 (A/Bで色分け、8試行×2条件=16点)
(2) 集計棒グラフ (個々のメンバー精度・アンサンブル精度・ゲイン・不一致率、A/B比較、試行ごとの点も重ねてノイズを見せる)
(3) 決定境界オーバーレイGIF (A/B並び、メンバーが1人ずつ重なっていき最終アンサンブル境界に収束)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11, "lines.linewidth": 2.0})

OUTDIR = "C:/tmp/ensemble-diversity-experiment"

agg = np.load(f"{OUTDIR}/results_aggregate.npz")
a_gains, b_gains = agg["a_gains"], agg["b_gains"]
a_disagree, b_disagree = agg["a_disagree"], agg["b_disagree"]
a_ens, b_ens = agg["a_ens"], agg["b_ens"]
a_mean_member, b_mean_member = agg["a_mean_member"], agg["b_mean_member"]
corr_dg = float(agg["corr_dg"])
R, M, H_SMALL = int(agg["R"]), int(agg["M"]), int(agg["H_SMALL"])
H_LIST_B = agg["H_LIST_B"]

rep = np.load(f"{OUTDIR}/rep_trial.npz")
X_tr, y_tr = rep["X_tr"], rep["y_tr"]
gx, gy = rep["gx"], rep["gy"]
ens_grid_a, ens_grid_b = rep["ens_grid_a"], rep["ens_grid_b"]  # (M, n_grid) each
a_ens_acc_rep, b_ens_acc_rep = float(rep["a_ens_acc"]), float(rep["b_ens_acc"])

COL_A = "#2e6ee4"
COL_B = "#e4572e"

print(f"A(seed-only): mean_member={a_mean_member.mean():.4f}+-{a_mean_member.std():.4f} "
      f"ens={a_ens.mean():.4f}+-{a_ens.std():.4f} gain={a_gains.mean():+.4f}+-{a_gains.std():.4f} "
      f"disagree={a_disagree.mean():.4f}+-{a_disagree.std():.4f}")
print(f"B(bagging+width): mean_member={b_mean_member.mean():.4f}+-{b_mean_member.std():.4f} "
      f"ens={b_ens.mean():.4f}+-{b_ens.std():.4f} gain={b_gains.mean():+.4f}+-{b_gains.std():.4f} "
      f"disagree={b_disagree.mean():.4f}+-{b_disagree.std():.4f}")
print(f"pooled corr(disagree, gain) = {corr_dg:.4f}")

# ---------------- (0) setup diagram ----------------
# Stacked (A on top, B below), full-width panels — reads cleanly even when
# scaled down to a narrow (mobile) column, unlike a cramped side-by-side layout.
fig, ax = plt.subplots(figsize=(8, 9))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.5, 0.985, "総パラメータ数はどちらも330 (=4H+1の合計)\n「同じ予算」でのA/B対決",
        ha="center", va="top", fontsize=13)

# --- Panel A (top half) ---
ax.text(0.5, 0.90, "A: shallow diversity (seedのみ)", ha="center", fontsize=15, fontweight="bold", color=COL_A)
n_a = M  # one box per member: A and B both actually have M=10 members
box_w, gap = 0.06, 0.075
start_x = 0.5 - (n_a - 1) * gap / 2 - box_w / 2
for i in range(n_a):
    cx = start_x + i * gap
    ax.add_patch(plt.Rectangle((cx, 0.76), box_w, 0.10, fc="#dfe8f7", ec=COL_A, lw=1.3))
ax.text(0.5, 0.685, f"全メンバー 隠れ{H_SMALL}ユニット固定\n同じ訓練データ700点をそのまま使用\n乱数初期化seedだけを変える (M={M})",
        ha="center", va="top", fontsize=12.5, linespacing=1.6)
ax.annotate("", xy=(0.5, 0.545), xytext=(0.5, 0.595), arrowprops=dict(arrowstyle="->", lw=1.8, color=COL_A))
ax.text(0.5, 0.515, "予測を平均", ha="center", fontsize=12.5, color=COL_A)

ax.plot([0.03, 0.97], [0.485, 0.485], color="#999", lw=1, ls="--")

# --- Panel B (bottom half) ---
ax.text(0.5, 0.44, "B: real diversity (bagging+幅)", ha="center", fontsize=15, fontweight="bold", color=COL_B)
widths = H_LIST_B
n_b = len(widths)
gap_b = 0.075
start_x_b = 0.5 - (n_b - 1) * gap_b / 2 - box_w / 2
base_y = 0.30
for i, h in enumerate(widths):
    cx = start_x_b + i * gap_b
    hgt = 0.03 + 0.011 * h
    ax.add_patch(plt.Rectangle((cx, base_y), box_w, hgt, fc="#fbe3dc", ec=COL_B, lw=1.3))
ax.text(0.5, 0.225, f"メンバーごとに隠れユニット数を変える (4〜12, 合計80=Aと同じ)\n"
                    "各メンバーは訓練データのブートストラップ再抽出(700点、重複あり)で学習",
        ha="center", va="top", fontsize=12.5, linespacing=1.6)
ax.annotate("", xy=(0.5, 0.075), xytext=(0.5, 0.125), arrowprops=dict(arrowstyle="->", lw=1.8, color=COL_B))
ax.text(0.5, 0.045, "予測を平均", ha="center", fontsize=12.5, color=COL_B)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig0_setup.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("saved fig0_setup.png (stacked layout)")
print("saved fig0_setup.png")

# ---------------- (1) scatter: gain vs disagreement, colored by condition ----------------
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.axhline(0, color="#999", lw=1, ls="--")
ax.scatter(a_disagree, a_gains, s=70, color=COL_A, alpha=0.85, edgecolor="white", linewidth=0.6,
           label=f"A: shallow diversity (seedのみ, n={R})", zorder=3)
ax.scatter(b_disagree, b_gains, s=70, color=COL_B, alpha=0.85, marker="^", edgecolor="white", linewidth=0.6,
           label=f"B: real diversity (bagging+幅, n={R})", zorder=3)

# simple linear fit across pooled points, just to show the trend visually
all_d = np.concatenate([a_disagree, b_disagree])
all_g = np.concatenate([a_gains, b_gains])
coef = np.polyfit(all_d, all_g, 1)
xs = np.linspace(all_d.min(), all_d.max(), 50)
ax.plot(xs, np.polyval(coef, xs), color="#666", lw=1.3, ls=":", zorder=2, label="全体の線形フィット")

ax.set_xlabel("メンバー間の予測不一致率 (pairwise disagreement, test set)")
ax.set_ylabel("アンサンブルの精度ゲイン (ensemble acc − 平均メンバーacc)")
ax.set_title(f"不一致率が高いほどゲインも大きい傾向 (プールした相関 r={corr_dg:.2f})", fontsize=12.5)
ax.legend(fontsize=9.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig1_scatter.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig1_scatter.png")

# ---------------- (2) summary bars with per-trial dots (noise floor) ----------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

# panel 1: member acc vs ensemble acc
ax = axes[0]
xpos = np.array([0, 1, 3, 4])
means = [a_mean_member.mean(), a_ens.mean(), b_mean_member.mean(), b_ens.mean()]
stds = [a_mean_member.std(), a_ens.std(), b_mean_member.std(), b_ens.std()]
colors = [COL_A, COL_A, COL_B, COL_B]
hatches = [None, "//", None, "//"]
for xp, mn, sd, c, h in zip(xpos, means, stds, colors, hatches):
    ax.bar(xp, mn, yerr=sd, width=0.8, color=c, alpha=0.55 if h is None else 0.85, hatch=h,
           edgecolor=c, capsize=4)
for xi, arr in zip([0, 1, 3, 4], [a_mean_member, a_ens, b_mean_member, b_ens]):
    ax.scatter(np.full_like(arr, xi, dtype=float) + np.random.default_rng(0).uniform(-0.12, 0.12, len(arr)),
               arr, s=14, color="black", alpha=0.5, zorder=5)
ax.set_xticks(xpos)
ax.set_xticklabels(["A\n個々の\nメンバー", "A\nアンサンブル", "B\n個々の\nメンバー", "B\nアンサンブル"], fontsize=9)
ax.set_ylabel("test accuracy")
ax.set_title("精度: 個々のメンバー vs アンサンブル", fontsize=11.5)
ax.set_ylim(0.85, 0.96)

# panel 2: gain
ax = axes[1]
for xi, arr, c in zip([0, 1], [a_gains, b_gains], [COL_A, COL_B]):
    ax.bar(xi, arr.mean(), yerr=arr.std(), width=0.6, color=c, alpha=0.6, edgecolor=c, capsize=4)
    ax.scatter(np.full_like(arr, xi, dtype=float) + np.random.default_rng(1).uniform(-0.10, 0.10, len(arr)),
               arr, s=16, color="black", alpha=0.55, zorder=5)
ax.axhline(0, color="#999", lw=1, ls="--")
ax.set_xticks([0, 1])
ax.set_xticklabels(["A: seedのみ", "B: bagging+幅"], fontsize=10)
ax.set_ylabel("精度ゲイン (ensemble − 平均member)")
ax.set_title(f"アンサンブルの精度ゲイン (R={R}試行)", fontsize=11.5)

# panel 3: disagreement
ax = axes[2]
for xi, arr, c in zip([0, 1], [a_disagree, b_disagree], [COL_A, COL_B]):
    ax.bar(xi, arr.mean(), yerr=arr.std(), width=0.6, color=c, alpha=0.6, edgecolor=c, capsize=4)
    ax.scatter(np.full_like(arr, xi, dtype=float) + np.random.default_rng(2).uniform(-0.10, 0.10, len(arr)),
               arr, s=16, color="black", alpha=0.55, zorder=5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["A: seedのみ", "B: bagging+幅"], fontsize=10)
ax.set_ylabel("メンバー間の予測不一致率")
ax.set_title("多様性の実測値 (disagreement rate)", fontsize=11.5)

fig.suptitle(f"黒点は{R}試行それぞれの実測値 (ノイズの大きさをそのまま見せている)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUTDIR}/fig2_summary_bars.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig2_summary_bars.png")

# ---------------- (3) decision-boundary overlay GIF, A vs B side by side ----------------
frames = []
fig, axes = plt.subplots(1, 2, figsize=(7, 4.2), dpi=80)


def draw_panel(ax, grid_members, i_show, title, color, ens_acc, final):
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(X_tr[:, 0], X_tr[:, 1], s=2, color="k", alpha=0.10)
    for i in range(i_show):
        ax.contour(gx, gy, grid_members[i].reshape(gx.shape), levels=[0.5],
                   colors=[color], alpha=0.40, linewidths=1.1)
    if final:
        ens_mean = grid_members[:i_show].mean(axis=0)
        ax.contour(gx, gy, ens_mean.reshape(gx.shape), levels=[0.5],
                   colors=["black"], linewidths=2.4)
        ax.set_title(f"{title}\n{i_show}メンバー→アンサンブル境界 (acc {ens_acc:.3f})", fontsize=8.6)
    else:
        ax.set_title(f"{title}\n{i_show}/{grid_members.shape[0]} メンバーの境界を重ね描き", fontsize=8.6)


n_show_final_hold = 4
durations = []
for i in range(1, M + 1):
    draw_panel(axes[0], ens_grid_a, i, "A: shallow diversity", COL_A, a_ens_acc_rep, final=(i == M))
    draw_panel(axes[1], ens_grid_b, i, "B: real diversity", COL_B, b_ens_acc_rep, final=(i == M))
    fig.suptitle("メンバーの決定境界(半透明)を1人ずつ重ねると、最終的にどちらもアンサンブル境界(黒太線)に収束する", fontsize=8.8)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(buf.copy())
    durations.append(0.45 if i < M else 1.0)

# hold on the final ensemble frame a bit longer
for _ in range(n_show_final_hold):
    frames.append(frames[-1])
    durations.append(0.9)

plt.close(fig)
imageio.mimsave(f"{OUTDIR}/boundary.gif", frames, duration=durations)
print("saved boundary.gif")

import os
sz = os.path.getsize(f"{OUTDIR}/boundary.gif") / 1024
print(f"boundary.gif size: {sz:.1f} KB")

print("ALL VISUALS DONE")
