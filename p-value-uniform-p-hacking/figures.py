"""図とGIFを作る。出力先は引数（既定は記事の画像フォルダ）。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from core import simulate_p
from step2_optional_stopping import p_paths, N0

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\kouta\project\forsubwork\src\content\blog\images\p-value-uniform-p-hacking")
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, RED, GRAY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"
SIMS = 100_000


def hist_panel(ax, p, color, title, bins=20, ymax=None):
    ax.hist(p, bins=bins, range=(0, 1), color=color, density=True, edgecolor="white")
    ax.axhline(1, color=GRAY, ls=":", lw=1.2)
    ax.axvline(0.05, color=RED, ls="--", lw=1.4)
    ax.set_xlim(0, 1)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=15, loc="left")
    ax.tick_params(labelsize=12)
    ax.set_ylabel("密度", fontsize=13)


# ---- 図1: p値の分布 3段
p_null = simulate_p(np.random.default_rng(100), SIMS, 20, 0.0)
p_a20 = simulate_p(np.random.default_rng(7), SIMS, 20, 0.5)
p_a64 = simulate_p(np.random.default_rng(8), SIMS, 64, 0.5)
fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
hist_panel(axs[0], p_null, BLUE, f"本当は差なし (d=0, 各群20人): p<0.05は{(p_null<.05).mean()*100:.1f}%", ymax=4.2)
hist_panel(axs[1], p_a20, ORANGE, f"本当に差あり (d=0.5, 各群20人): p<0.05は{(p_a20<.05).mean()*100:.1f}%", ymax=7)
hist_panel(axs[2], p_a64, GREEN, f"本当に差あり (d=0.5, 各群64人): p<0.05は{(p_a64<.05).mean()*100:.1f}%", ymax=18)
axs[2].set_xlabel("p値 (赤い破線は0.05、灰色の点線は完全に一様なときの高さ)", fontsize=13)
fig.suptitle("同じ t 検定を10万回ずつ行ったときの p値の分布", fontsize=16, y=0.995)
fig.tight_layout()
fig.savefig(OUT / "1.png", dpi=110)
plt.close(fig)
print("1.png: null<0.05 %.4f, a20 %.4f, a64 %.4f" % ((p_null < .05).mean(), (p_a20 < .05).mean(), (p_a64 < .05).mean()))

# ---- GIF: 積み上がるヒストグラム (差なし/差あり d=0.5 各群20人)
counts = np.unique(np.round(np.geomspace(5, 20000, 36)).astype(int))
counts = np.concatenate([counts, [20000] * 6])
fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)


def draw(i):
    m = counts[i]
    for ax, p, col, ttl in [(axs[0], p_null, BLUE, "本当は差なし (d=0)"), (axs[1], p_a20, ORANGE, "本当に差あり (d=0.5)")]:
        ax.clear()
        q = p[:m]
        ax.hist(q, bins=20, range=(0, 1), color=col, density=True, edgecolor="white")
        ax.axhline(1, color=GRAY, ls=":", lw=1.2)
        ax.axvline(0.05, color=RED, ls="--", lw=1.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 4.2 if ax is axs[0] else 7)
        ax.set_title(f"{ttl}: {m:,}回の検定  p<0.05は{(q<.05).mean()*100:.1f}%", fontsize=14, loc="left")
        ax.tick_params(labelsize=11)
        ax.set_ylabel("密度", fontsize=12)
    axs[1].set_xlabel("p値 (各群20人。赤い破線は0.05)", fontsize=12)
    fig.tight_layout()


anim = FuncAnimation(fig, draw, frames=len(counts), interval=250)
anim.save(OUT / "pvalue_pile.gif", writer=PillowWriter(fps=5), dpi=70)
plt.close(fig)
print("pvalue_pile.gif frames", len(counts))

# ---- 図2: 差が無い2群に1人ずつ足して検定し直したときの p値の経路
rng = np.random.default_rng(3)
P, _, _ = p_paths(rng, 12000)
Pm = P[:, :291]  # 各群 N0..300 人
n = np.arange(N0, 301)
fig, axs = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [3, 2]})
ax = axs[0]
hit_idx = []
for i in range(12):
    p = Pm[i]
    hit = np.where(p < 0.05)[0]
    col = RED if len(hit) else GRAY
    ax.plot(n, p, color=col, lw=1.1, alpha=0.8)
    if len(hit):
        ax.plot(n[hit[0]], p[hit[0]], "o", color=RED, ms=7)
        hit_idx.append(hit[0])
ax.axhline(0.05, color="k", ls="--", lw=1.4)
ax.set_yscale("log")
ax.set_ylim(1e-4, 1.5)
ax.set_xlabel("各群の人数 (1人ずつ足して、そのたびに検定し直す)", fontsize=13)
ax.set_ylabel("p値 (対数軸)", fontsize=13)
ax.set_title(f"本当は差が無い2群の p値の経路12本: {len(hit_idx)}本が途中で0.05を下回った (赤)", fontsize=14, loc="left")
ax.tick_params(labelsize=12)
frac = (Pm < 0.05).any(1).mean()
frac_end = (Pm[:, -1] < 0.05).mean()
ax = axs[1]
cum = np.array([(Pm[:, : k + 1] < 0.05).any(1).mean() for k in range(Pm.shape[1])]) * 100
ax.plot(n, cum, color=RED, lw=2.2, label="途中で一度でも p<0.05 になった割合")
ax.axhline(5, color="k", ls="--", lw=1.2)
ax.text(N0 + 140, 2.0, "5% (検定を1回だけ行うとき)", fontsize=12)
ax.set_ylim(0, 45)
ax.set_xlabel("各群の人数の上限", fontsize=13)
ax.set_ylabel("偽陽性率 (%)", fontsize=13)
ax.legend(fontsize=12, loc="center right")
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(OUT / "2.png", dpi=110)
plt.close(fig)
print("2.png: 12本中", len(hit_idx), "本が交差; 1.2万回で300人上限の偽陽性率 %.4f, 300人時点の単発 %.4f" % (frac, frac_end))

# ---- 図3: 上限人数 x 検定の頻度
res = json.load(open("optional_stopping.json"))
L = [int(k) for k in res["every1"]]
a = [np.mean(res["every1"][str(k)]) * 100 for k in L]
b = [np.mean(res["every10"][str(k)]) * 100 for k in L]
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(L, a, "o-", color=RED, lw=2.2, ms=8, label="1人ずつ足して毎回検定")
ax.plot(L, b, "s-", color=ORANGE, lw=2.2, ms=8, label="10人ずつ足して毎回検定")
ax.axhline(5, color="k", ls="--", lw=1.2)
ax.text(150, 2.0, "5% (最後に1回だけ検定)", fontsize=12)
for x_, y_ in zip(L, a):
    ax.annotate(f"{y_:.0f}%", (x_, y_), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=11, color=RED)
for x_, y_ in zip(L, b):
    ax.annotate(f"{y_:.0f}%", (x_, y_), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=11, color=ORANGE)
ax.set_xscale("log")
ax.set_xticks(L)
ax.set_xticklabels([str(v) for v in L])
ax.set_ylim(0, 47)
ax.set_xlabel("上限の人数 (各群、10人から開始。この人数まで足しても有意にならなければ諦める)", fontsize=12)
ax.set_ylabel("偽陽性率 (%)  本当は差が無いのに「有意」", fontsize=12)
ax.set_title("有意になるまで足す: 偽陽性率は上限を増やすほど上がる", fontsize=14, loc="left")
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(OUT / "3.png", dpi=110)
plt.close(fig)

# ---- 図4: 指標の数 k
mo = json.load(open("multiple_outcomes.json"))
ks = [int(k) for k in mo]
fig, ax = plt.subplots(figsize=(9, 6))
kk = np.arange(1, 21)
ax.plot(kk, (1 - 0.95 ** kk) * 100, color=GRAY, lw=2, label="理論値 1 - 0.95^k")
ax.plot(ks, [mo[str(k)] * 100 for k in ks], "o", color=RED, ms=10, label="実測 (各10万回 x 5種の平均)")
for k in ks:
    ax.annotate(f"{mo[str(k)]*100:.1f}%", (k, mo[str(k)] * 100), textcoords="offset points", xytext=(6, -16), fontsize=11, color=RED)
ax.axhline(5, color="k", ls="--", lw=1.2)
ax.set_xlabel("測った指標の数 k (全部、本当は差が無い・互いに独立)", fontsize=12)
ax.set_ylabel("一番小さい p が 0.05 未満になる割合 (%)", fontsize=12)
ax.set_title("都合のよい指標だけ報告すると", fontsize=14, loc="left")
ax.set_ylim(0, 75)
ax.set_xticks([1, 2, 3, 5, 10, 15, 20])
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(OUT / "4.png", dpi=110)
plt.close(fig)
print("done")
