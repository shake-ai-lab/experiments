"""sorts.py の results.json から記事用の図とGIFを作る。"""

import json
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

D = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"

ALGOS = list(D["comparisons"]["ランダム"].keys())
KINDS = list(D["comparisons"].keys())

# ---------------------------------------------------------------- fig0 (本文冒頭)
# 入力の種類ごとの比較回数。同じアルゴリズムでも入力で桁が変わることを見せる。
print("fig0", flush=True)
M = np.array([[D["comparisons"][k][a] for k in KINDS] for a in ALGOS], float)
fig, ax = plt.subplots(figsize=(11.6, 6.4))
im = ax.imshow(np.log10(M), cmap="RdYlBu_r", aspect="auto")
ax.set_xticks(range(len(KINDS)), KINDS, fontsize=13)
ax.set_yticks(range(len(ALGOS)), ALGOS, fontsize=12)
for i in range(len(ALGOS)):
    for j in range(len(KINDS)):
        v = M[i, j]
        ax.text(
            j,
            i,
            f"{v/1000:,.0f}千" if v >= 10000 else f"{v:,.0f}",
            ha="center",
            va="center",
            fontsize=11.5,
            color="white" if np.log10(v) > 5.3 else "#222",
        )
ax.set_title("2000個を並べ替えるのに必要な比較回数(少ないほど良い)", fontsize=16, pad=14)
fig.tight_layout()
fig.savefig("0.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig1
# n を変えたときの伸び方。傾きでオーダーが見える。
print("fig1", flush=True)
fig, ax = plt.subplots(figsize=(11.0, 6.6))
colours = {
    "バブルソート": "#b5651d",
    "選択ソート": WARN,
    "挿入ソート": "#c1436d",
    "マージソート": INK,
    "ヒープソート": "#6a9fb5",
    "クイックソート(先頭ピボット)": ACC,
    "クイックソート(乱択ピボット)": "#3f7d3f",
    "クイックソート(3分割)": "#555555",
}
for a in ALGOS:
    ns = sorted(int(n) for n in D["scaling"][a])
    ys = [D["scaling"][a][str(n)] for n in ns]
    ax.plot(ns, ys, "o-", color=colours[a], lw=2.0, ms=5, label=a)
ns = np.array([100, 6400])
ax.plot(ns, ns * np.log2(ns) * 1.0, "--", color="black", lw=1.3, alpha=0.5)
ax.text(6600, 6400 * np.log2(6400) * 1.0, "n log n", fontsize=12, va="center")
ax.plot(ns, ns**2 / 2, ":", color="black", lw=1.3, alpha=0.5)
ax.text(3400, 3200**2 / 2 * 1.15, "n² / 2", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("要素数 n (ランダムな入力)", fontsize=14)
ax.set_ylabel("比較回数", fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, loc="upper left")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("1.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig2
# 最悪ケース。ソート済み入力に先頭ピボットを食わせると n(n-1)/2 にぴたり乗る。
print("fig2", flush=True)
ns = sorted(int(n) for n in D["worst_case"])
got = [D["worst_case"][str(n)] for n in ns]
theory = [n * (n - 1) // 2 for n in ns]
rand = [D["scaling"]["クイックソート(先頭ピボット)"][str(n)] for n in ns if str(n) in D["scaling"]["クイックソート(先頭ピボット)"]]
fig, ax = plt.subplots(figsize=(10.6, 6.2))
ax.plot(ns, theory, "-", color="black", lw=6, alpha=0.22, label="理論値 n(n-1)/2")
ax.plot(ns, got, "o-", color=ACC, lw=2.2, ms=8, label="ソート済みを入れたとき")
ax.plot(ns[: len(rand)], rand, "s-", color=INK, lw=2.2, ms=7, label="ランダムを入れたとき")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("要素数 n", fontsize=14)
ax.set_ylabel("比較回数", fontsize=14)
ax.set_title("先頭をピボットにしたクイックソート", fontsize=16, pad=12)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("2.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- fig3
# 小さい n で挿入ソートが勝つか。比較回数では勝たない。
print("fig3", flush=True)
fig, ax = plt.subplots(figsize=(10.6, 6.0))
for name, col in [
    ("挿入ソート", "#c1436d"),
    ("マージソート", INK),
    ("クイックソート(乱択ピボット)", "#3f7d3f"),
]:
    ns = sorted(int(n) for n in D["crossover"][name])
    ax.plot(ns, [D["crossover"][name][str(n)] for n in ns], "o-", color=col, lw=2.2, ms=5, label=name)
ax.set_xlabel("要素数 n", fontsize=14)
ax.set_ylabel("比較回数(200回の平均)", fontsize=14)
ax.set_title("小さい配列でも、比較回数では挿入ソートは勝たない", fontsize=16, pad=12)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("3.png", dpi=115)
plt.close(fig)

# ---------------------------------------------------------------- GIF
# 並べ替わっていく様子。**比較1回につき1コマ**記録する。
# 最初はアルゴリズムごとに都合のいい単位(外側ループ1回、交換1回…)で記録していたが、
# それだと粒度がそろわず、選択ソート49コマ・挿入ソート662コマのように
# 実際の仕事量と逆の印象になった。比較を単位にすれば横並びで比べられる。
print("gif", flush=True)
N = 48
random.seed(7)
BASE = list(range(1, N + 1))
random.shuffle(BASE)


def rec_insertion(a):
    a = a[:]
    frames = []
    for i in range(1, len(a)):
        v, j = a[i], i - 1
        while True:
            frames.append((a[:], max(j, 0)))  # 比較1回ぶん
            if not (j >= 0 and v < a[j]):
                break
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = v
    frames.append((a[:], -1))
    return frames


def rec_selection(a):
    a = a[:]
    frames = []
    for i in range(len(a)):
        m = i
        for j in range(i + 1, len(a)):
            frames.append((a[:], j))
            if a[j] < a[m]:
                m = j
        a[i], a[m] = a[m], a[i]
    frames.append((a[:], -1))
    return frames


def rec_quick(a):
    """その場で分割するクイックソート(末尾ピボット)"""
    a = a[:]
    frames = []

    def part(lo, hi):
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            frames.append((a[:], j))
            if a[j] < pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[hi] = a[hi], a[i]
        return i

    def go(lo, hi):
        if lo < hi:
            p = part(lo, hi)
            go(lo, p - 1)
            go(p + 1, hi)

    go(0, len(a) - 1)
    frames.append((a[:], -1))
    return frames


def rec_heap(a):
    a = a[:]
    frames = []
    n = len(a)

    def sift(root, end):
        while True:
            ch = 2 * root + 1
            if ch >= end:
                return
            if ch + 1 < end:
                frames.append((a[:], ch))
                if a[ch] < a[ch + 1]:
                    ch += 1
            frames.append((a[:], ch))
            if not a[root] < a[ch]:
                return
            a[root], a[ch] = a[ch], a[root]
            root = ch

    for i in range(n // 2 - 1, -1, -1):
        sift(i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift(0, end)
    frames.append((a[:], -1))
    return frames


runs = [
    ("挿入ソート", rec_insertion(BASE)),
    ("選択ソート", rec_selection(BASE)),
    ("クイックソート", rec_quick(BASE)),
    ("ヒープソート", rec_heap(BASE)),
]
LONGEST = max(len(f) for _, f in runs)
STEPS = 95
idx = np.linspace(0, LONGEST - 1, STEPS).astype(int)

fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6))
bars = []
for ax, (name, frames) in zip(axes.ravel(), runs):
    b = ax.bar(range(N), frames[0][0], color=SUB, width=0.86)
    # 比較回数はタイトルに固定で出す。コマごとに数字を書き換えると差分が大きくなり、
    # GIFが1.7MBまで膨らんだ。途中経過は棒の動きと、終わったパネルが青くなることで分かる。
    ax.set_title(f"{name}   比較 {len(frames) - 1:,} 回", fontsize=15, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, N + 2)
    bars.append(b)
fig.suptitle("48個を並べ替える。比較1回を1コマにして横並びに", fontsize=16, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.94))


def draw(t):
    for b, (name, frames) in zip(bars, runs):
        f = min(idx[t], len(frames) - 1)
        state, active = frames[f]
        done = f == len(frames) - 1
        for k, rect in enumerate(b):
            rect.set_height(state[k])
            rect.set_color(INK if done else (ACC if k == active else SUB))
    return [r for b in bars for r in b]


anim = FuncAnimation(fig, draw, frames=STEPS, interval=45, blit=False)
anim.save("sort.gif", writer=PillowWriter(fps=9), dpi=58)
plt.close(fig)

print("done")
