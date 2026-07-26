# -*- coding: utf-8 -*-
"""results_grid.json / results_dist.json から記事用の図を生成する。"""
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

C_LEAK = "#D55E00"   # 特徴選択リーク
C_PROPER = "#0072B2"  # 正しいCV
C_SCALE = "#009E73"  # スケーリングリーク

OUT = r"C:\Users\kouta\project\forsubwork\src\content\blog\images\feature-selection-leakage"

with open("results_grid.json") as f:
    grid = json.load(f)
with open("results_dist.json") as f:
    dist = json.load(f)

P_LIST = sorted({r["p"] for r in grid})
K_LIST = sorted({r["k"] for r in grid})


def agg(cond, p, k):
    v = [r[cond] for r in grid if r["p"] == p and r["k"] == k]
    return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)


# ---------------------------------------------------------------- fig 1
# k=10 固定、p を横軸に3条件の精度
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=110)
k = 10
for cond, color, label, ls in [
    ("leak_select", C_LEAK, "特徴選択を全データで(リーク)", "-"),
    ("proper", C_PROPER, "すべてfold内で(正しいCV)", "-"),
    ("leak_scale", C_SCALE, "標準化だけ全データで(リーク)", "--"),
]:
    means = np.array([agg(cond, p, k)[0] for p in P_LIST])
    stds = np.array([agg(cond, p, k)[1] for p in P_LIST])
    ax.plot(P_LIST, means, ls, marker="o", color=color, label=label, lw=2, ms=5)
    if cond != "leak_scale":
        ax.fill_between(P_LIST, means - stds, means + stds, color=color, alpha=0.15, lw=0)
ax.axhline(0.5, color="#888888", ls=":", lw=1)
ax.text(P_LIST[-1], 0.472, "偶然の水準 (0.5)", color="#666666", fontsize=9, ha="right")
ax.set_xscale("log")
ax.set_xlabel("特徴数 p(ログスケール)")
ax.set_ylabel("5-fold 交差検証精度")
ax.set_title(f"純粋なノイズ + ランダムラベル(n=100, 上位k={k}特徴を選択)", fontsize=11)
ax.set_ylim(0.3, 1.0)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(f"{OUT}/1.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 2
# p x k ヒートマップ(上: リーク、下: 正しいCV)を縦スタック
fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.2), dpi=110)
for ax, cond, title in [
    (axes[0], "leak_select", "(A) 特徴選択を全データでやってから交差検証(リークあり)"),
    (axes[1], "proper", "(B) 特徴選択をfold内の訓練データだけでやる(正しいCV)"),
]:
    M = np.full((len(K_LIST), len(P_LIST)), np.nan)
    for i, k in enumerate(K_LIST):
        for j, p in enumerate(P_LIST):
            if k <= p:
                M[i, j] = agg(cond, p, k)[0]
    im = ax.imshow(M, cmap="Oranges", vmin=0.45, vmax=1.0, aspect="auto", origin="lower")
    for i in range(len(K_LIST)):
        for j in range(len(P_LIST)):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if v > 0.80 else "#333333")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="#aaaaaa")
    ax.set_xticks(range(len(P_LIST)))
    ax.set_xticklabels([f"{p:,}" for p in P_LIST], fontsize=9)
    ax.set_yticks(range(len(K_LIST)))
    ax.set_yticklabels([f"{k:,}" for k in K_LIST], fontsize=9)
    ax.set_xlabel("特徴数 p")
    ax.set_ylabel("選択する特徴数 k")
    ax.set_title(title, fontsize=11.5)
fig.suptitle("5-fold 交差検証精度の平均(20シード)— データは純粋な乱数", fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}/2.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 3
# リークの種類による害の違い: 正しいCVとの散布図(縦スタック)
fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), dpi=110)
props = [r["proper"] for r in grid]
sel = [r["leak_select"] for r in grid]
scl = [r["leak_scale"] for r in grid]
for ax, ys, color, title in [
    (axes[0], sel, C_LEAK, "特徴選択リーク vs 正しいCV"),
    (axes[1], scl, C_SCALE, "標準化リーク vs 正しいCV"),
]:
    ax.scatter(props, ys, s=14, alpha=0.35, color=color, lw=0)
    ax.plot([0.2, 1.0], [0.2, 1.0], color="#888888", ls="--", lw=1)
    ax.text(0.83, 0.86, "y = x", color="#666666", fontsize=9, rotation=38)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("正しいCVの精度(同じデータ・同じ設定)")
    ax.set_ylabel("リークありCVの精度")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)
fig.suptitle("全条件(p×k×20シード)の1点ずつを比較", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(f"{OUT}/3.png")
plt.close(fig)

# ---------------------------------------------------------------- GIF
# p を増やしながら、リークあり/正しいCVの精度分布(60シード)が動く様子
p_frames = dist["p_list"]
data = dist["data"]
bins = np.arange(0.275, 1.03, 0.05)

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=80)


def draw(frame):
    p = p_frames[frame]
    ax.clear()
    ax.hist(data[str(p)]["proper"], bins=bins, color=C_PROPER, alpha=0.75,
            label="正しいCV(fold内で選択)")
    ax.hist(data[str(p)]["leak"], bins=bins, color=C_LEAK, alpha=0.75,
            label="リークあり(全データで選択)")
    ax.axvline(0.5, color="#888888", ls="--", lw=1)
    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(0, 42)
    ax.set_xlabel("5-fold 交差検証精度")
    ax.set_ylabel("シード数(全60)")
    leak_mean = np.mean(data[str(p)]["leak"])
    ax.set_title(f"特徴数 p = {p:,}(k=10選択)  リークあり平均 {leak_mean:.3f}", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)


frames = list(range(len(p_frames))) + [len(p_frames) - 1] * 3  # 最終フレームを長めに
anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames))
anim.save(f"{OUT}/sweep.gif", writer=PillowWriter(fps=1.4))
plt.close(fig)

print("figures done")
