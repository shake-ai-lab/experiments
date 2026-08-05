import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"

# ------------------------------------------------------------------ fig1
print("fig1", flush=True)
labels = ["変更しない\n(司会者は中身を知っている)", "変更する\n(司会者は中身を知っている)",
          "変更する\n(司会者は中身を知らない)"]
vals = [R["classic"]["stay"], R["classic"]["switch"], R["ignorant_host"]["switch"]]
theo = [1 / 3, 2 / 3, 1 / 2]
fig, ax = plt.subplots(figsize=(8.6, 5.2))
bars = ax.bar(labels, vals, color=[SUB, ACC, INK], width=0.62)
for b, v, t in zip(bars, vals, theo):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.016, f"{v:.4f}", ha="center",
            fontsize=14, fontweight="bold", color="#3d4753")
    ax.hlines(t, b.get_x(), b.get_x() + b.get_width(), color="#3d4753", ls="--", lw=1.6)
ax.text(2.34, 0.5, "理論値", color="#3d4753", fontsize=10, va="center")
ax.set_ylim(0, 0.79)
ax.set_ylabel("車を当てた割合")
ax.set_title("100万回まわした結果。右2本は「ヤギのドアが1枚開いた」という見た目が同じ",
             fontsize=13)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig1_three.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig2
print("fig2", flush=True)
cum = np.array(R["convergence"])
x = np.arange(1, cum.shape[1] + 1) * 5
fig, ax = plt.subplots(figsize=(8.4, 5.0))
for row in cum:
    ax.plot(x, row, color=ACC, alpha=0.16, lw=1.1)
ax.plot(x, cum.mean(axis=0), color=ACC, lw=2.6, label="40本の平均")
ax.axhline(2 / 3, ls="--", color="#3d4753", lw=1.6, label="2/3")
ax.axhline(0.5, ls=":", color=SUB, lw=1.6, label="1/2（直感）")
ax.set_xscale("log")
ax.set_xlim(5, x[-1])
ax.set_ylim(0.2, 1.0)
ax.set_xlabel("試行回数（対数）")
ax.set_ylabel("そこまでの累積勝率（変更する戦略）")
ax.set_title("40人が同時に試したら、勝率はどう見えるか", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig2_convergence.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig3
print("fig3", flush=True)
rows = R["general"]
Ns = sorted(set(r["N"] for r in rows))
k1 = [next(r for r in rows if r["N"] == n and r["k"] == 1) for n in Ns]
kmax = [next(r for r in rows if r["N"] == n and r["k"] == n - 2) for n in Ns]
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(Ns, [r["switch"] for r in kmax], "o-", color=ACC, lw=2.4, ms=8,
        label="司会者が「残り1枚」になるまで開ける (k=N-2)")
ax.plot(Ns, [r["switch"] for r in k1], "o-", color=INK, lw=2.4, ms=8,
        label="司会者が1枚だけ開ける (k=1)")
ax.plot(Ns, [r["stay"] for r in k1], "o--", color=SUB, lw=2.0, ms=6,
        label="変更しない (1/N)")
ax.set_xscale("log")
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel("ドアの枚数 N")
ax.set_ylabel("車を当てた割合")
ax.set_title("「変更が得」なのは、司会者が何枚を消してくれるかで決まる", fontsize=13)
ax.annotate("N=100で98枚開けてもらえば 0.990", (100, 0.99), textcoords="offset points",
            xytext=(-12, -26), ha="right", color=ACC, fontsize=10.5, fontweight="bold")
ax.annotate("N=100で1枚だけなら 0.0099。\n変更しない場合(0.0097)とほぼ変わらない",
            (100, 0.01), textcoords="offset points", xytext=(-12, 26), ha="right",
            color=INK, fontsize=10.5, fontweight="bold")
ax.legend(fontsize=10, loc="center left")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig3_general.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig4
print("fig4", flush=True)
need = np.array(R["power"]["need"])
need = need[need <= R["power"]["maxn"]]
fig, ax = plt.subplots(figsize=(8.4, 4.8))
ax.hist(need, bins=np.arange(0, 160, 4), color=INK, alpha=0.85)
for v, lab, col in [(R["power"]["median"], "中央値", ACC), (R["power"]["q90"], "90%点", WARN)]:
    ax.axvline(v, color=col, lw=2.2, ls="--")
    ax.text(v + 2, ax.get_ylim()[1] * 0.88, f"{lab} {v:.0f}回", color=col,
            fontsize=11.5, fontweight="bold")
ax.set_xlabel("「変更が有利」と5%水準で言い切れるまでにかかった試行回数")
ax.set_ylabel("回数（2,000回の再現実験）")
ax.set_title("自分で試して納得するのに、何回まわせばいいか", fontsize=13)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig4_power.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ GIF
print("gif: 100枚のドア", flush=True)
N = 100
COLS = 10
PICK = 37
CAR = 61
order = [d for d in range(N) if d not in (PICK, CAR)]
rng = np.random.default_rng(11)
rng.shuffle(order)
FRAMES = 36
per = int(np.ceil(len(order) / (FRAMES - 6)))

fig, ax = plt.subplots(figsize=(5.2, 5.6))
ax.set_xlim(-0.5, COLS - 0.5)
ax.set_ylim(-0.5, N // COLS - 0.5 + 1.4)
ax.set_aspect("equal")
ax.axis("off")
patches = []
for d in range(N):
    r, c = divmod(d, COLS)
    p = Rectangle((c - 0.42, (N // COLS - 1 - r) - 0.42), 0.84, 0.84,
                  facecolor="#c9d3dd", edgecolor="white", lw=1.4)
    ax.add_patch(p)
    patches.append(p)
title = ax.text(4.5, N // COLS - 0.5 + 0.7, "", ha="center", fontsize=12.5,
                fontweight="bold", color="#3d4753")


def frame(k):
    opened = order[: max(0, (k - 3)) * per] if k >= 3 else []
    for d, p in enumerate(patches):
        if d == PICK:
            p.set_facecolor(INK)
        elif d in opened:
            p.set_facecolor("#f1f3f5")
            p.set_edgecolor("#dfe4ea")
        else:
            p.set_facecolor("#c9d3dd")
            p.set_edgecolor("white")
    if k < 3:
        title.set_text("100枚から1枚選んだ（当たる確率 1/100）")
    elif len(opened) < len(order):
        title.set_text(f"司会者がヤギのドアを開けていく… 残り {N - 1 - len(opened)} 枚")
    else:
        patches[CAR].set_facecolor(ACC)
        title.set_text("最後まで残ったのはこの2枚。どちらに変えるか")
    return patches


FuncAnimation(fig, frame, frames=FRAMES, blit=False).save(
    "doors.gif", writer=PillowWriter(fps=4), dpi=82)
plt.close(fig)

from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im = Image.open("doors.gif")
fr = [f.copy().convert("RGB").quantize(colors=32) for f in ImageSequence.Iterator(im)]
fr[0].save("doors.gif", save_all=True, append_images=fr[1:], loop=0, duration=300, optimize=True)
print(f"  doors.gif {os.path.getsize('doors.gif')/1024:.0f}KB / {len(fr)}フレーム / {fr[0].size}")

# ------------------------------------------------------------------ interactive
print("3d/heatmap", flush=True)
import plotly.graph_objects as go  # noqa: E402

Nv = np.arange(3, 41)
Z, X, Y = [], [], []
for n in Nv:
    row = []
    for k in range(0, 39):
        row.append((1 - 1 / n) / (n - 1 - k) if k <= n - 2 else np.nan)
    Z.append(row)
Z = np.array(Z)
fig = go.Figure(go.Heatmap(
    z=Z, x=np.arange(0, 39), y=Nv,
    colorscale=[[0, "#eef2f6"], [0.25, "#9ec5dd"], [0.6, "#2f6f9e"], [1, "#ef476f"]],
    colorbar=dict(title="変更したときの勝率"),
    hovertemplate="ドア %{y} 枚 / 司会者が %{x} 枚開ける<br>変更したときの勝率 %{z:.3f}<extra></extra>",
))
fig.update_layout(
    title=dict(text="ドアの枚数と、司会者が開ける枚数で、変更の得は決まる", font=dict(size=14)),
    xaxis_title="司会者が開けるドアの枚数 k", yaxis_title="ドアの総数 N",
    height=560, margin=dict(l=60, r=20, t=50, b=50), paper_bgcolor="rgba(0,0,0,0)")
fig.write_html("monty-heatmap.html", include_plotlyjs="cdn")
print(f"  monty-heatmap.html {os.path.getsize('monty-heatmap.html')/1024:.0f}KB")
print("完了")
