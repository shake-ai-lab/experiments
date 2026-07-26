"""記事の導入用の補足図2枚を生成する。
0_digits_sample.png: sklearn digitsの中身(8x8ピクセルの手書き数字)を0〜9で1枚ずつ表示。
    読者が「モデルが何を見て何を当てるのか」を具体的にイメージできるようにする。
0b_lr_schedule.png: warmupなし/linear/cosineの3条件で、実際にAdamに渡される学習率が
    ステップに対してどう立ち上がるかを描く。コードだけだと分かりづらい
    「cosineはゆっくり立ち上がってから加速する」を目で確認できるようにする。
訓練の再実行は不要(データセット表示とスケジュール式の描画だけ)。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11.5, "font.size": 11})

OUT = "C:/tmp/learning-rate-warmup-ablation-experiment"
BLOG_IMG = "C:/Users/kouta/project/forsubwork/src/content/blog/images/learning-rate-warmup-ablation"

# ---------------------------------------------------------------------------
# 図0: digitsのサンプル画像(0〜9を1枚ずつ)
# ---------------------------------------------------------------------------
d = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(8.6, 4.4), constrained_layout=True)
for cls, ax in enumerate(axes.ravel()):
    idx = int(np.where(d.target == cls)[0][0])
    ax.imshow(d.images[idx], cmap="gray_r", interpolation="nearest")
    ax.set_title(f"正解ラベル: {cls}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("sklearn digitsの中身: 8×8ピクセルの手書き数字(0〜9の10クラス)", fontsize=14)
for dst in (OUT, BLOG_IMG):
    fig.savefig(f"{dst}/0_digits_sample.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 0_digits_sample.png")

# ---------------------------------------------------------------------------
# 図0b: 3条件の学習率スケジュール(先頭400ステップ)
# ---------------------------------------------------------------------------
LR = 0.018
STEPS_SHOWN = 400
WARMUP_STEPS = 150  # 全1500ステップの先頭10%

t = np.arange(1, STEPS_SHOWN + 1)


def schedule(kind):
    cur = np.full_like(t, LR, dtype=float)
    w = t <= WARMUP_STEPS
    if kind == "linear":
        cur[w] = LR * t[w] / WARMUP_STEPS
    elif kind == "cosine":
        frac = t[w] / WARMUP_STEPS
        cur[w] = LR * 0.5 * (1 - np.cos(np.pi * frac))
    return cur


fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.axvspan(0, WARMUP_STEPS, color="#fde68a", alpha=0.45, zorder=0, label="warmup区間(先頭150ステップ)")
ax.plot(t, schedule("none"), color="#6b7280", lw=2.4, label="warmupなし")
ax.plot(t, schedule("linear"), color="#2563eb", lw=2.4, label="linear warmup")
ax.plot(t, schedule("cosine"), color="#16a34a", lw=2.4, label="cosine warmup")
ax.axhline(LR, color="#9ca3af", lw=0.8, ls="--", zorder=1)
ax.text(STEPS_SHOWN - 6, LR + 0.0006, "目標LR=0.018(以降1500ステップまでこのまま)", ha="right", fontsize=10.5, color="#374151")
ax.set_xlim(0, STEPS_SHOWN)
ax.set_ylim(0, 0.0215)
ax.set_xlabel("ステップ(先頭400ステップのみ表示)")
ax.set_ylabel("実際にAdamに渡す学習率")
ax.grid(alpha=0.3, zorder=0)
ax.legend(fontsize=10.5, loc="lower right", frameon=True)
ax.set_title("3条件の学習率スケジュール: warmup区間150ステップの立ち上がり方だけが違う", fontsize=13)
fig.tight_layout()
for dst in (OUT, BLOG_IMG):
    fig.savefig(f"{dst}/0b_lr_schedule.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 0b_lr_schedule.png")
