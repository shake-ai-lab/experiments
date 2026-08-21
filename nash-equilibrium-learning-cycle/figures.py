"""記事に載せる図を作る。

模式的なA/B比較は横並びではなく縦スタックにして、文字は大きめにする
(そうしないとスマホで読めない)。
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

DARK = "#3d4753"
BLUE = "#2f6f9e"
RED = "#ef476f"
AMBER = "#e8a33d"
GRAY = "#8d99ae"
GREEN = "#06a77d"
LIGHT = "#c0c6cc"

results = json.load(open("results.json", encoding="utf-8"))
analysis = json.load(open("analysis.json", encoding="utf-8"))
periods = json.load(open("periods.json", encoding="utf-8"))


def to_xy(p):
    """3つの確率を正三角形の座標に落とす(シンプレックス)。"""
    r, pa, s = p[..., 0], p[..., 1], p[..., 2]
    return 0.5 * (2 * pa + s) / (r + pa + s), (np.sqrt(3) / 2) * s / (r + pa + s)


def draw_simplex(ax, label_size=13):
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color=LIGHT, lw=1.4, zorder=1)
    nx, ny = to_xy(np.array([1 / 3, 1 / 3, 1 / 3]))
    ax.plot([nx], [ny], marker="*", ms=17, color=DARK, zorder=6)
    ax.text(-0.02, -0.06, "グー", ha="center", fontsize=label_size, color=DARK)
    ax.text(1.02, -0.06, "パー", ha="center", fontsize=label_size, color=DARK)
    ax.text(0.5, np.sqrt(3) / 2 + 0.035, "チョキ", ha="center", fontsize=label_size, color=DARK)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.14, 1.02)
    ax.set_aspect("equal")
    # 箱の縦横比をデータの縦横比に合わせる。こうしないと箱の中に余白が残り、
    # タイトルが三角形から遠く離れた位置に浮いてしまう。
    ax.set_box_aspect(1.16 / 1.24)
    ax.axis("off")


# --- 図1: 刻み幅を変えたときの軌道(small multiples、2x2) ---------------------
# 200,000手を全部重ねると線がつぶれて螺旋が見えないので、周回数をそろえて切る。
# 色は時間で変え、どちらへ向かって進んでいるのかを線自体に語らせる。
LAPS = 14


def cut_at_laps(xs, laps=LAPS):
    lead = xs.argmax(axis=1)
    switch = np.flatnonzero(np.diff(lead) != 0)
    if len(switch) < laps * 3 + 1:
        return len(xs)
    return int(switch[laps * 3]) + 1


fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.0))
etas = ["0.002", "0.01", "0.05", "0.2"]
caps = [
    "刻み幅 0.002 — ほぼ同じ輪をなぞる",
    "刻み幅 0.01 — 少しずつ外へ",
    "刻み幅 0.05 — はっきり外向きの螺旋",
    "刻み幅 0.2 — 数周で角に届く",
]
for ax, eta, cap in zip(axes.ravel(), etas, caps):
    xs = np.load(f"rps_x_eta{eta}.npy")[: cut_at_laps(np.load(f"rps_x_eta{eta}.npy"))]
    x, y = to_xy(xs)
    seg = np.stack([np.c_[x[:-1], y[:-1]], np.c_[x[1:], y[1:]]], axis=1)
    lc = matplotlib.collections.LineCollection(
        seg, cmap="YlGnBu", array=np.linspace(0, 1, len(seg)), lw=1.5, alpha=0.95, zorder=3
    )
    lc.set_clim(-0.35, 1.0)  # 淡すぎる端を使わない
    ax.add_collection(lc)
    ax.plot([x[0]], [y[0]], marker="o", ms=9, color=RED, zorder=7)
    draw_simplex(ax)
    ax.set_title(cap, fontsize=15, pad=14, color=DARK)
fig.suptitle(
    "同じじゃんけん、同じ出発点。刻み幅だけを変えて14周ぶん\n"
    "★がナッシュ均衡(1/3ずつ)、●が出発点。線は薄い→濃いの順に進む",
    fontsize=17,
)
fig.subplots_adjust(hspace=0.16, wspace=0.02, top=0.86)
fig.savefig("fig1_simplex.png", dpi=135, bbox_inches="tight", facecolor="white")
plt.close(fig)

# --- 図2: 「今の手」と「時間平均」の読まれやすさ(縦スタック) -----------------
fig, axes = plt.subplots(2, 1, figsize=(10, 10.5))
for ax, eta, ttl in [
    (axes[0], "0.05", "刻み幅 0.05"),
    (axes[1], "0.002", "刻み幅 0.002（ゆっくり学習）"),
]:
    ex_i = np.load(f"ex_iter_{eta}.npy")
    ex_a = np.load(f"ex_avg_{eta}.npy")
    stride = 200_000 // (len(ex_i) - 1)
    t = np.arange(len(ex_i)) * stride
    ax.plot(t[1:], ex_i[1:], color=RED, lw=1.6, label="いま出している手")
    ax.plot(t[1:], ex_a[1:], color=BLUE, lw=2.4, label="ここまでの手の平均")
    ax.axhline(2.0, color=GRAY, ls=":", lw=1.4)
    # 注釈は軸の内側に置く。データ座標の右端に置くと枠からはみ出す。
    ax.text(0.985, 2.06, "読まれたら必ず負ける", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=12, color=GRAY)
    ax.axhline(0.0, color=DARK, ls="--", lw=1.4)
    ax.text(0.015, 0.05, "ナッシュ均衡(誰にも付け込まれない)", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=12, color=DARK)
    ax.set_xscale("log")
    ax.set_ylim(-0.12, 2.35)
    ax.set_xlabel("手数", fontsize=14)
    ax.set_ylabel("読まれたときの損(0が無敵)", fontsize=14)
    ax.set_title(ttl, fontsize=16, pad=10, color=DARK)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, loc="center left")
    ax.grid(alpha=0.25)
fig.suptitle("学習した「今の手」は読まれる。読まれないのは、その平均だけ", fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig2_exploitability.png", dpi=135, bbox_inches="tight", facecolor="white")
plt.close(fig)

# --- 図3: 周期が伸びていく ---------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.6))
for eta, col in [("0.2", RED), ("0.05", AMBER), ("0.01", BLUE), ("0.002", GREEN)]:
    per = np.load(f"periods_eta{eta}.npy")  # 実測した1周ぶんの手数。補間はしない
    ax.plot(np.arange(1, len(per) + 1), per, color=col, lw=2.0, label=f"刻み幅 {eta}")
ax.set_xlabel("何周目か", fontsize=14)
ax.set_ylabel("1周にかかる手数", fontsize=14)
ax.set_yscale("log")
ax.set_title("外へ広がるほど、一周が遅くなる", fontsize=17, pad=10, color=DARK)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig3_period.png", dpi=135, bbox_inches="tight", facecolor="white")
plt.close(fig)

# --- 図4: 3つのゲームの行き先(縦スタック) ------------------------------------
# 注釈は必ず軸の内側に置くこと。データ座標で軸の外に文字を出すと
# bbox_inches="tight" がその文字まで含めて余白を広げ、図が横に間延びする。
fig, axes = plt.subplots(3, 1, figsize=(10, 14.5))

xs = np.load("rps_x_eta0.05.npy")
stride = 200_000 // (len(xs) - 1)
t = np.arange(len(xs)) * stride
hi = 3000 // stride
for i, (name, col) in enumerate([("グー", BLUE), ("パー", AMBER), ("チョキ", RED)]):
    axes[0].plot(t[:hi], xs[:hi, i], color=col, lw=2.2, label=name)
axes[0].axhline(1 / 3, color=DARK, ls="--", lw=1.5)
axes[0].text(0.985, 0.9, "破線 = ナッシュ均衡 1/3", transform=axes[0].transAxes,
             ha="right", va="top", fontsize=13, color=DARK)
axes[0].set_ylim(-0.03, 1.15)
axes[0].set_title("じゃんけん — 均衡を通り過ぎ続ける。振れ幅はむしろ育つ",
                  fontsize=17, pad=12, color=DARK)
axes[0].legend(fontsize=13, ncol=3, loc="upper left")

pd_xs = np.load("pd_x.npy")
axes[1].plot(np.arange(len(pd_xs)), pd_xs[:, 1], color=RED, lw=2.6, label="裏切りを選ぶ確率")
axes[1].axhline(1.0, color=DARK, ls="--", lw=1.5)
axes[1].text(0.985, 0.78, "破線 = ナッシュ均衡(裏切り)", transform=axes[1].transAxes,
             ha="right", va="top", fontsize=13, color=DARK)
axes[1].axvline(results["pd"]["steps_to_99pct_defect"], color=GRAY, ls=":", lw=1.6)
axes[1].text(results["pd"]["steps_to_99pct_defect"] + 15, 0.45, "107手で99%",
             fontsize=14, color=GRAY)
axes[1].set_xlim(0, 800)
axes[1].set_ylim(-0.03, 1.15)
axes[1].set_title("囚人のジレンマ — まっすぐ均衡へ", fontsize=17, pad=12, color=DARK)
axes[1].legend(fontsize=13, loc="center right")

for ax in axes[:2]:
    ax.set_xlabel("手数", fontsize=14)
    ax.set_ylabel("確率", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)

basin = np.load("coord_basin.npy")
axes[2].imshow(basin.T, origin="lower", extent=[0, 1, 0, 1],
               cmap=matplotlib.colors.ListedColormap([LIGHT, BLUE]),
               aspect="auto", interpolation="nearest")
axes[2].plot([1 / 3], [1 / 3], marker="*", ms=20, color=DARK, zorder=5)
axes[2].annotate("混合戦略の均衡", xy=(1 / 3, 1 / 3), xytext=(0.46, 0.13),
                 fontsize=13, color=DARK,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="-", color=DARK, lw=1.2))
axes[2].set_xlabel("プレイヤー1が最初に「大きい方」を選ぶ確率", fontsize=14)
axes[2].set_ylabel("プレイヤー2の同じ確率", fontsize=14)
axes[2].set_title("協調ゲーム — 出発点で行き先が決まる", fontsize=17, pad=12, color=DARK)
axes[2].tick_params(labelsize=12)
axes[2].legend(
    handles=[
        Line2D([], [], marker="s", ls="", ms=13, color=BLUE, label="利得2の均衡へ (73.3%)"),
        Line2D([], [], marker="s", ls="", ms=13, color=LIGHT, label="利得1の均衡へ (26.7%)"),
    ],
    fontsize=12, loc="upper right", framealpha=0.92,
)
fig.suptitle("同じ学習則を、3つのゲームに当てた結果", fontsize=19, y=0.985)
fig.subplots_adjust(hspace=0.34, top=0.925, left=0.09, right=0.97, bottom=0.05)
fig.savefig("fig4_three_games.png", dpi=135, facecolor="white")
plt.close(fig)

print("figures done")
