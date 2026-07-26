import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "Meiryo"

RED = "#d62728"
BLUE = "#1f77b4"
GRAY = "#888888"

d = np.load("trajectory.npz")
steps_no, a_acc_no, b_acc_no = d["steps_no"], d["a_acc_no"], d["b_acc_no"]
steps_re, a_acc_re, b_acc_re = d["steps_re"], d["a_acc_re"], d["b_acc_re"]
a_start = float(d["a_test_acc_start"])

with open("summary.json") as f:
    summary = json.load(f)

# ============================================================ GIF ========
# symlog x-axis (0..3000) to show both the fast early collapse/dip-recover
# and the long flat plateau, growing reveal animation.
# dpi=100 -> 700x420 px: rendered at ~720 px column width, so text stays
# legible (the first published version was 297x179 and unreadable).
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=100)

max_step = 3000
ax.set_xscale("symlog", linthresh=10)
ax.set_xlim(0, max_step)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Task Bのみで学習を続けたステップ数", fontsize=11)
ax.set_ylabel("Task A テスト精度", fontsize=11)
ax.tick_params(labelsize=10)
ax.axhline(a_start, color=GRAY, linestyle=":", linewidth=1, label=f"開始時点 ({a_start:.3f})")
(line_no,) = ax.plot([], [], color=RED, linewidth=2, label="リハーサルなし")
(line_re,) = ax.plot([], [], color=BLUE, linewidth=2, label="10%リハーサルあり")
# set a representative title BEFORE tight_layout so vertical space is
# reserved for it (an empty title here gets clipped in every frame)
title = ax.set_title("step=0000   A精度: なし=0.000  あり=0.000", fontsize=12)
ax.legend(loc="center right", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()

# frame schedule: dense over the collapse/dip (steps 1-40), medium over the
# recovery (to ~step 150), sparse over the plateau; hold the final frame ~2 s.
n_points = len(steps_no)
frame_idx = list(range(0, 9)) + list(range(9, 30, 2)) + list(range(30, n_points, 30))
if frame_idx[-1] != n_points - 1:
    frame_idx.append(n_points - 1)
frame_idx = frame_idx + [frame_idx[-1]] * 20


def update(i):
    k = frame_idx[i]
    line_no.set_data(steps_no[: k + 1], a_acc_no[: k + 1])
    line_re.set_data(steps_re[: k + 1], a_acc_re[: k + 1])
    cur_step = steps_no[k]
    title.set_text(f"step={cur_step}   A精度: なし={a_acc_no[k]:.3f}  あり={a_acc_re[k]:.3f}")
    return line_no, line_re, title


anim = FuncAnimation(fig, update, frames=len(frame_idx), blit=False)
anim.save("collapse.gif", writer=PillowWriter(fps=10))
plt.close(fig)
print("saved collapse.gif")


# ============================================================ PNG 1 ======
# final accuracy comparison (multi-seed mean, 15 seeds)
import subprocess  # noqa
# multiseed numbers are hardcoded here (computed and printed by multiseed.py)
labels = ["Task A\n開始時", "Task A\nリハーサルなし\n(最終)", "Task A\nリハーサルあり\n(最終)",
          "Task B\nリハーサルなし\n(最終)", "Task B\nリハーサルあり\n(最終)"]
means = [0.9948, 0.0011, 0.9676, 0.9822, 0.9752]
stds = [0.0059, 0.0041, 0.0129, 0.0074, 0.0090]
colors = [GRAY, RED, BLUE, RED, BLUE]

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=110)
bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.08)
ax.set_ylabel("テスト精度 (15 seed平均)")
ax.set_title("Task Bのみで再学習したときのTask A / Task B精度")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("1_final_comparison.png")
plt.close(fig)
print("saved 1_final_comparison.png")


# ============================================================ PNG 2 ======
# mechanism: mean logit for task-A classes vs task-B classes over first 60 steps
mech_steps = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60]
logit_a = [0.700, -0.229, -1.078, -2.503, -4.245, -5.192, -6.781, -7.834, -8.905, -9.572, -10.463]
logit_b = [-8.630, -7.448, -6.328, -4.269, -1.762, -0.512, 1.318, 2.083, 2.749, 2.842, 2.668]

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=110)
ax.plot(mech_steps, logit_a, "o-", color=RED, label="Task Aクラス(0-4)の平均logit")
ax.plot(mech_steps, logit_b, "o-", color=BLUE, label="Task Bクラス(5-9)の平均logit")
ax.axhline(0, color=GRAY, linewidth=0.8)
ax.axvline(15, color=GRAY, linestyle=":", linewidth=1)
ax.text(16, -2.6, "A精度50%割れ (step=15)", fontsize=10, color="#555555")
ax.set_xlabel("Task Bのみで学習を続けたステップ数(リハーサルなし)")
ax.set_ylabel("出力層の平均logit(Task Aテストデータに対する)")
ax.set_title("わずか数十ステップでlogitの大小が逆転する")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("2_logit_crossover.png")
plt.close(fig)
print("saved 2_logit_crossover.png")


# ============================================================ PNG 3 ======
# prediction remap: for each true task-A digit, distribution over predicted class (after collapse)
dist = {
    0: {5: 5, 6: 30, 7: 1},
    1: {6: 4, 8: 28, 9: 5},
    2: {5: 1, 6: 1, 7: 3, 8: 19, 9: 11},
    3: {5: 2, 7: 3, 8: 7, 9: 25},
    4: {5: 2, 6: 24, 7: 8, 8: 1, 9: 1},
}
task_b_classes = [5, 6, 7, 8, 9]
mat = np.zeros((5, 5))
for i, true_cls in enumerate(range(5)):
    total = sum(dist[true_cls].values())
    for j, pred_cls in enumerate(task_b_classes):
        mat[i, j] = dist[true_cls].get(pred_cls, 0) / total

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=110)
im = ax.imshow(mat, cmap="Oranges", vmin=0, vmax=1)
ax.set_xticks(range(5))
ax.set_xticklabels([str(c) for c in task_b_classes])
ax.set_yticks(range(5))
ax.set_yticklabels([str(c) for c in range(5)])
ax.set_xlabel("崩壊後の予測クラス(Task B側の数字)")
ax.set_ylabel("本当の数字(Task A、0-4)")
ax.set_title("忘れた後、古い数字は「新しいどれか」に化ける")
for i in range(5):
    for j in range(5):
        v = mat[i, j]
        if v > 0.01:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.5 else "black", fontsize=9)
fig.colorbar(im, ax=ax, label="予測された割合")
fig.tight_layout()
fig.savefig("3_prediction_remap.png")
plt.close(fig)
print("saved 3_prediction_remap.png")

import os
for f in ["collapse.gif", "1_final_comparison.png", "2_logit_crossover.png", "3_prediction_remap.png"]:
    print(f, os.path.getsize(f), "bytes")
