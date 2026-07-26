"""静止画3枚 + GIF1本を生成する。
1.png: LRスイープ(崩壊率がLRの関数としてどこで立ち上がるか、選んだLR=0.018に矢印)
2.png: 3条件(warmupなし/linear/cosine)の多シード損失曲線(20本の細い線+太い平均線)
3.png: warmupは序盤のピークだけ抑える、後半のピークは変わらない、を示す棒グラフ
warmup.gif: 2.pngのアニメーション版(20本の線が伸びていく)
"""
import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11.5, "font.size": 11})

OUT = "C:/tmp/learning-rate-warmup-ablation-experiment"

COND_NAMES = {"none": "warmupなし", "linear": "linear warmup", "cosine": "cosine warmup"}
COND_COLOR = {"none": "#6b7280", "linear": "#2563eb", "cosine": "#16a34a"}
FAIL_COLOR = "#dc2626"
OK_COLOR = "#2563eb"
FAIL_THRESH = 0.5
TARGET_LR = 0.018
WARMUP_STEPS = 150  # steps=1500, warmup_frac=0.1

# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------
sweep = np.load(f"{OUT}/sweep_results.npz")
res = np.load(f"{OUT}/results.npz")
import json

with open(f"{OUT}/summary.json", encoding="utf-8") as f:
    summary = json.load(f)

conds = ["none", "linear", "cosine"]
loss = {c: res[f"loss_{c}"] for c in conds}  # (N_SEEDS, STEPS)
accs = {c: np.array(summary[c]["accs"]) for c in conds}
n_seeds = loss["none"].shape[0]
steps_n = loss["none"].shape[1]

# ---------------------------------------------------------------------------
# 図1: LRスイープ。崩壊率(test_acc<0.5の割合)がLRに対してどこで立ち上がるか
# ---------------------------------------------------------------------------
lrs = sweep["lrs"]
fail_rate = sweep["fail_rate"] * 100
mean_acc = sweep["mean_acc"] * 100

fig, ax = plt.subplots(figsize=(8.6, 4.6))
bar_colors = ["#dc2626" if abs(lr - TARGET_LR) < 1e-9 else "#94a3b8" for lr in lrs]
x = np.arange(len(lrs))
bars = ax.bar(x, fail_rate, color=bar_colors, width=0.62, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([f"{lr:g}" for lr in lrs], rotation=45, ha="right")
ax.set_ylabel("崩壊率(test_acc<0.5になったシードの割合、%)")
ax.set_xlabel("学習率(Adam, warmupなし、8シード、1500ステップ)")
ax.set_ylim(0, 108)
ax.grid(axis="y", alpha=0.3, zorder=0)
for xi, (fr, ma) in enumerate(zip(fail_rate, mean_acc)):
    ax.text(xi, fr + 3, f"acc\n{ma:.0f}%", ha="center", fontsize=8.2, color="#374151")
i_target = list(lrs).index(TARGET_LR)
ax.annotate(
    "本実験で使うLR=0.018\n(8シード中4シードが崩壊=五分五分)",
    xy=(i_target, fail_rate[i_target]),
    xytext=(i_target - 3.6, 78),
    fontsize=10.5,
    color="#dc2626",
    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8),
)
ax.set_title("崩壊率はLR=0.012→0.018の間で0%から50%へ急に立ち上がる", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 1.png")

# ---------------------------------------------------------------------------
# 図2: 3条件、多シードの損失曲線
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
for ax, c in zip(axes, conds):
    L = loss[c]
    a = accs[c]
    for i in range(n_seeds):
        color = FAIL_COLOR if a[i] < FAIL_THRESH else OK_COLOR
        ax.plot(L[i], color=color, linewidth=0.7, alpha=0.55)
    mean_curve = np.nanmean(L, axis=0)
    ax.plot(mean_curve, color="black", linewidth=2.2, label="平均")
    if c != "none":
        ax.axvspan(0, WARMUP_STEPS, color="#fde68a", alpha=0.45, zorder=0)
    n_fail = int((a < FAIL_THRESH).sum())
    ax.set_title(f"{COND_NAMES[c]}\n崩壊 {n_fail}/{n_seeds}シード", fontsize=12.5)
    ax.set_xlabel("ステップ")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("損失(対数軸)")
handles = [
    plt.Line2D([0], [0], color=OK_COLOR, lw=1.5, label="最終test_acc≥0.5"),
    plt.Line2D([0], [0], color=FAIL_COLOR, lw=1.5, label="最終test_acc<0.5(崩壊)"),
    plt.Line2D([0], [0], color="black", lw=2.2, label="平均"),
    plt.Rectangle((0, 0), 1, 1, color="#fde68a", alpha=0.45, label="warmup区間(先頭10%)"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, fontsize=9.5, frameon=False)
fig.suptitle(
    "warmupは序盤の暴れを鎮めるが、最終的な崩壊シード数はほぼ変わらない",
    fontsize=14,
    y=1.14,
)
fig.tight_layout()
fig.savefig(f"{OUT}/2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 2.png")

# ---------------------------------------------------------------------------
# 図3: 序盤(<=200step)と後半(>200step)のピーク損失比較
# ---------------------------------------------------------------------------
early_peaks = {c: np.nanmax(loss[c][:, :200], axis=1) for c in conds}
late_peaks = {c: np.nanmax(loss[c][:, 200:], axis=1) for c in conds}

fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(conds))
w = 0.32
early_means = [early_peaks[c].mean() for c in conds]
late_means = [late_peaks[c].mean() for c in conds]
b1 = ax.bar(x - w / 2, early_means, width=w, color="#93c5fd", label="序盤(1〜200step)の最大損失(平均)", zorder=3)
b2 = ax.bar(x + w / 2, late_means, width=w, color="#1e3a8a", label="後半(201〜1500step)の最大損失(平均)", zorder=3)
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.08, f"{h:.2f}", ha="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85), zorder=5)
ax.set_xticks(x)
ax.set_xticklabels([COND_NAMES[c] for c in conds], fontsize=11.5)
ax.set_ylabel("損失のピーク値(平均)")
ax.set_ylim(0, 8.6)
ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.annotate(
    "序盤のピークは下がる\n(4.7→3.1〜3.4)",
    xy=(1 - w / 2, early_means[1]),
    xytext=(0.55, 7.7),
    fontsize=10,
    color="#1d4ed8",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.6, shrinkB=10),
)
ax.annotate(
    "後半のピークはむしろ変わらない\nor悪化(4.4→4.3〜4.6)",
    xy=(1 + w / 2, late_means[1]),
    xytext=(1.75, 7.7),
    fontsize=10,
    color="#991b1b",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.6, shrinkB=10),
)
ax.set_title("warmupが下げるのは序盤のピークだけ。後半の暴れには効かない", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/3.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 3.png")
print("early_means", early_means, "late_means", late_means)

# ---------------------------------------------------------------------------
# GIF: 図2のアニメーション版(3パネル、20本の線が伸びていく)
# ---------------------------------------------------------------------------
from PIL import Image

stride = 160
frame_idxs = list(range(1, steps_n, stride)) + [steps_n - 1]
frame_idxs = [frame_idxs[0]] * 4 + frame_idxs + [frame_idxs[-1]] * 6  # 最初と最後を長く保持

frames = []
fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.3), sharey=True, dpi=75)
fig.subplots_adjust(top=0.78, bottom=0.18, left=0.09, right=0.98, wspace=0.08)

for upto in frame_idxs:
    for ax, c in zip(axes, conds):
        ax.clear()
        L = loss[c]
        a = accs[c]
        for i in range(n_seeds):
            color = FAIL_COLOR if a[i] < FAIL_THRESH else OK_COLOR
            ax.plot(L[i, :upto], color=color, linewidth=0.7, alpha=0.55)
        if c != "none":
            ax.axvspan(0, WARMUP_STEPS, color="#fde68a", alpha=0.45, zorder=0)
        ax.set_title(COND_NAMES[c], fontsize=10)
        ax.set_xlim(0, steps_n)
        ax.set_ylim(0.01, 30)
        ax.set_yscale("log")
        ax.set_xlabel("ステップ", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("損失(対数軸)", fontsize=9)
    fig.suptitle(f"学習率0.018での20シード同時学習(step {upto}/{steps_n})", fontsize=11.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[:, :, :3]
    pal_img = Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=16, dither=Image.NONE)
    frames.append(pal_img)

plt.close(fig)
frames[0].save(
    f"{OUT}/warmup.gif",
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
    optimize=True,
)
print("saved warmup.gif")
