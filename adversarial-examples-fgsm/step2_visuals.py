"""FGSM実験の可視化一式:
(1) two-moonsデータセット + 学習済み決定境界
(2) epsilonに対するflip-rate(反転率)のカーブ (0〜2.5、実測した頭打ちまで)
(3) 個別の点のbefore/afterを決定境界に重ねた散布図 (showcase epsilon)
(4) epsilonを0から連続的に上げていくと点が境界を越えていく様子のGIF
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11, "lines.linewidth": 2.0})

OUTDIR = "C:/tmp/adversarial-examples-fgsm"
d = np.load(f"{OUTDIR}/results.npz")

gx, gy, grid_out = d["gx"], d["gy"], d["grid_out"]
X_tr, y_tr, X_te, y_te = d["X_tr"], d["y_tr"], d["X_te"], d["y_te"]
out_te, pred_te, correct_mask = d["out_te"], d["pred_te"], d["correct_mask"]
dX, sign_dX = d["dX"], d["sign_dX"]
epsilons, flip_rates = d["epsilons"], d["flip_rates"]
EPS_SHOWCASE = float(d["EPS_SHOWCASE"])
X_adv_showcase, flipped_showcase = d["X_adv_showcase"], d["flipped_showcase"]
show_flip_idx, show_stay_idx = d["show_flip_idx"], d["show_stay_idx"]
test_acc, n_correct, H, n_params = float(d["test_acc"]), int(d["n_correct"]), int(d["H"]), int(d["n_params"])
eps_10, eps_25 = float(d["eps_10"]), float(d["eps_25"])
min_flip_eps = d["min_flip_eps"]

CLASS_COLORS = ["#3b6fb6", "#d1495b"]  # class0=blue, class1=red系

# ---------------- (1) dataset + decision boundary ----------------
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.contourf(gx, gy, grid_out.reshape(gx.shape), levels=np.linspace(0, 1, 21),
            cmap="RdBu_r", alpha=0.55, vmin=0, vmax=1)
ax.contour(gx, gy, grid_out.reshape(gx.shape), levels=[0.5], colors="black", linewidths=1.6)
for c in [0, 1]:
    m = y_tr == c
    ax.scatter(X_tr[m, 0], X_tr[m, 1], s=10, color=CLASS_COLORS[c], alpha=0.55,
               label=f"class {c} (train)", edgecolors="none")
ax.set_title(f"two-moons: 学習済みMLP(隠れ{H}ユニット, パラメータ{n_params}個)の決定境界\nテスト精度 {test_acc:.4f}")
ax.set_xlabel("x1 (標準化後)")
ax.set_ylabel("x2 (標準化後)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/1_data_boundary.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved 1_data_boundary.png")

# ---------------- (2) flip-rate vs epsilon curve ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(epsilons, flip_rates * 100, color="#d1495b", linewidth=2.2)
ax.axhline(10, color="gray", linestyle=":", linewidth=1)
ax.axhline(25, color="gray", linestyle=":", linewidth=1)
ax.axvline(eps_10, color="#3b6fb6", linestyle="--", linewidth=1.3)
ax.axvline(eps_25, color="#3b6fb6", linestyle="--", linewidth=1.3)
# 注釈はε=0.41の破線と重ならないよう、カーブ下の空白域(右下)に置く
ax.annotate(f"flip率10%\nε={eps_10:.2f}", xy=(eps_10, 10), xytext=(eps_10 + 0.22, 6),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
ax.annotate(f"flip率25%\nε={eps_25:.2f}", xy=(eps_25, 25), xytext=(eps_25 + 0.15, 31),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
final_rate = flip_rates[-1] * 100
ax.annotate(f"ε={epsilons[-1]:.1f}でも\n頭打ち({final_rate:.1f}%)", xy=(epsilons[-1], final_rate),
            xytext=(epsilons[-1] - 1.0, final_rate - 14), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("摂動の大きさ ε (標準化後データのstd=1.0が基準)")
ax.set_ylabel("反転率 (もともと正解していた点のうち\n誤分類に変わった割合) [%]")
ax.set_title("FGSM摂動の強さ ε と分類の反転率の実測カーブ")
ax.set_xlim(0, epsilons[-1])
ax.set_ylim(0, 55)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/2_flip_curve.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved 2_flip_curve.png")

# ---------------- (3) before/after scatter for showcase epsilon ----------------
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.contourf(gx, gy, grid_out.reshape(gx.shape), levels=np.linspace(0, 1, 21),
            cmap="RdBu_r", alpha=0.35, vmin=0, vmax=1)
ax.contour(gx, gy, grid_out.reshape(gx.shape), levels=[0.5], colors="black", linewidths=1.6)
ax.scatter(X_te[correct_mask, 0], X_te[correct_mask, 1], s=8, color="#999999", alpha=0.35,
           label="テスト点(正解, 背景)")

show_idx_all = np.concatenate([show_flip_idx, show_stay_idx])
for i in show_idx_all:
    x0, y0 = X_te[i]
    x1, y1 = X_adv_showcase[i]
    flipped = bool(flipped_showcase[i])
    color = "#d1495b" if flipped else "#3b6fb6"
    ax.scatter([x0], [y0], marker="o", s=70, facecolors="white", edgecolors=color, linewidths=2, zorder=5)
    ax.scatter([x1], [y1], marker="X", s=90, color=color, zorder=6)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1.6, alpha=0.85))

ax.scatter([], [], marker="o", s=70, facecolors="white", edgecolors="#d1495b", linewidths=2, label="摂動前(反転する点)")
ax.scatter([], [], marker="X", s=90, color="#d1495b", label="摂動後(反転した)")
ax.scatter([], [], marker="o", s=70, facecolors="white", edgecolors="#3b6fb6", linewidths=2, label="摂動前(反転しない点)")
ax.scatter([], [], marker="X", s=90, color="#3b6fb6", label="摂動後(反転しなかった)")
n_flipped_show = int(flipped_showcase.sum())
ax.set_title(f"ε={EPS_SHOWCASE:.2f}でのFGSM摂動: 全体では{n_correct}点中{n_flipped_show}点が反転({n_flipped_show/n_correct*100:.1f}%)\n"
             f"矢印は個別に選んだ点の摂動前→摂動後の移動")
ax.set_xlabel("x1 (標準化後)")
ax.set_ylabel("x2 (標準化後)")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/3_before_after.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved 3_before_after.png")

# ---------------- (4) GIF: epsilon sweep, points crossing the boundary ----------------
from PIL import Image

gif_eps = np.round(np.arange(0.0, 1.02, 0.06), 3)  # stride for a lighter GIF
# background boundary grid, coarser for a lighter GIF
gxs, gys = np.meshgrid(np.linspace(-3.0, 3.0, 55), np.linspace(-3.0, 3.0, 55))
step_gx = gx.shape[0] // gxs.shape[0]
gx_small = gx[::step_gx, ::step_gx][: gxs.shape[0], : gxs.shape[1]]
gy_small = gy[::step_gx, ::step_gx][: gxs.shape[0], : gxs.shape[1]]
grid_out_small = grid_out.reshape(gx.shape)[::step_gx, ::step_gx][: gxs.shape[0], : gxs.shape[1]]

fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=68)
frames = []
for eps in gif_eps:
    ax.clear()
    ax.contourf(gx_small, gy_small, grid_out_small, levels=np.linspace(0, 1, 5),
                cmap="RdBu_r", alpha=0.45, vmin=0, vmax=1)
    ax.contour(gx_small, gy_small, grid_out_small, levels=[0.5], colors="black", linewidths=1.2)
    X_now = X_te[correct_mask] + eps * sign_dX[correct_mask]
    y_now = y_te[correct_mask]
    idx_eps = np.argmin(np.abs(epsilons - eps))
    rate_now = flip_rates[idx_eps] * 100
    for c in [0, 1]:
        m = y_now == c
        ax.scatter(X_now[m, 0], X_now[m, 1], s=9, color=CLASS_COLORS[c], alpha=0.85, edgecolors="none")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"ε = {eps:.2f}  |  反転率 {rate_now:.1f}%", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(Image.fromarray(buf).convert("RGB"))
plt.close(fig)

durations_ms = [900] + [140] * (len(frames) - 2) + [1600]
pal_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
pal_frames[0].save(
    f"{OUTDIR}/sweep.gif", save_all=True, append_images=pal_frames[1:],
    duration=durations_ms, loop=0, optimize=True, disposal=2,
)
print(f"saved sweep.gif, {len(frames)} frames")

import os
sz = os.path.getsize(f"{OUTDIR}/sweep.gif")
print(f"sweep.gif size: {sz/1024:.1f} KB")

print("ALL DONE")
