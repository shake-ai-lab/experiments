# 記事用の図・GIF・3D。
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from core import step, run

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

R1 = json.load(open("results.json", encoding="utf-8"))
R2 = json.load(open("results2.json", encoding="utf-8"))

INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"

# ---------------------------------------------------------------- fig1
print("fig1: 初期密度 vs 最終密度", flush=True)
rows = R1["density_sweep"]["rows"]
p = np.array([r["p"] for r in rows])
fin = np.array([r["final_mean"] for r in rows])
sd = np.array([r["final_sd"] for r in rows])
st = {r["p"]: (r["t_half_mean"], r["t_half_sd"]) for r in R2["settle"]}

fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.0, 8.2), sharex=True)
a1.errorbar(p, fin, yerr=sd, fmt="o-", color=INK, lw=2.2, ms=7, capsize=3)
a1.axhspan(0.033, 0.038, color=ACC, alpha=0.13)
a1.annotate("初期密度が9倍違っても\nこの帯に入る", (0.45, 0.0355), color=ACC, fontsize=11,
            ha="center", va="center", fontweight="bold")
a1.annotate("薄すぎると\n絶滅", (0.03, 0.006), color=SUB, fontsize=10.5, ha="left")
a1.annotate("濃すぎても\n絶滅", (0.93, 0.006), color=SUB, fontsize=10.5, ha="right")
a1.set_ylabel("2000世代後の生存密度")
a1.set_title("どの初期密度から始めても、行き着く密度はほぼ同じ", fontsize=13.5)
a1.grid(alpha=0.3)

pp = np.array([r["p"] for r in R2["settle"]])
tt = np.array([st[x][0] for x in pp])
te = np.array([st[x][1] for x in pp])
a2.errorbar(pp, tt, yerr=te, fmt="o-", color=WARN, lw=2.2, ms=7, capsize=3)
imax = int(np.argmax(tt))
a2.annotate(f"最も長く暴れるのは初期密度 {pp[imax]}\n({tt[imax]:.0f}世代)",
            (pp[imax], tt[imax]), textcoords="offset points", xytext=(10, -34),
            color=WARN, fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=WARN))
a2.set_xlabel("初期密度")
a2.set_ylabel("最終密度の2倍を下回るまでの世代数")
a2.set_title("違うのは「行き先」ではなく「落ち着くまでの時間」", fontsize=13.5)
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig1_density.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig2
print("fig2: 10000世代の減り方", flush=True)
d = np.array(R2["long_run"]["density_every_10"])
t = np.arange(len(d)) * 10
fig, ax = plt.subplots(figsize=(8.0, 5.0))
ax.plot(np.maximum(t, 1), d, color=INK, lw=2.2)
ax.set_xscale("log")
for g, lab in [(100, None), (1000, None), (10000, None)]:
    ax.axvline(g, ls=":", color=SUB, lw=1)
ax.annotate(f"10,000世代でも {d[-1]:.5f}\nまだゆっくり減り続けている",
            (10000, d[-1]), textcoords="offset points", xytext=(-16, 42),
            ha="right", color=ACC, fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ACC))
ax.set_xlabel("世代（対数）")
ax.set_ylabel("生存密度")
ax.set_title("初期密度0.375から10,000世代。急に落ちたあと、いつまでも下げ止まらない", fontsize=13)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig2_decay.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig3
print("fig3: 灰の組成", flush=True)
top = R1["census"]["top"][:7]
total = R1["census"]["total"]
names = [t[0] for t in top][::-1]
vals = [t[1] / total * 100 for t in top][::-1]
kind = R1["census"]["kind"]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [2, 1]})
bars = b1.barh(names, vals, color=[ACC if v == max(vals) else INK for v in vals])
for y, v in enumerate(vals):
    b1.text(v + 0.4, y, f"{v:.1f}%", va="center", fontsize=10.5, color="#3d4753")
b1.set_xlabel("灰に含まれる割合（個数ベース）")
b1.set_title(f"燃え尽きたあとに残るもの（{total:,}個を分類）", fontsize=13)
b1.set_xlim(0, max(vals) * 1.22)
b1.grid(alpha=0.3, axis="x")

order = ["静止物", "周期2の振動子", "その他"]
kv = [kind.get(k, 0) for k in order]
b2.pie(kv, labels=order, autopct="%1.1f%%", colors=[INK, WARN, SUB],
       textprops={"fontsize": 11}, startangle=90)
b2.set_title("動かないか、2世代で戻るか", fontsize=13)
fig.tight_layout()
fig.savefig("fig3_ash.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- fig4
print("fig4: 光円錐", flush=True)
rad = np.array(R2["lightcone"]["radius"])
cnt = np.array(R2["lightcone"]["diff_cells"])
gen = np.arange(1, len(rad) + 1)
fig, (c1, c2) = plt.subplots(1, 2, figsize=(11.4, 4.9))
c1.plot(gen, gen, ls="--", color=SUB, lw=1.8, label="理論上限 1セル/世代（ライフゲームの光速c）")
c1.plot(gen, rad, color=ACC, lw=2.4, label=f"実測（回帰の傾き {R2['lightcone']['speed']:.3f}）")
c1.set_xlabel("世代")
c1.set_ylabel("違いが届いた距離（セル）")
c1.set_title("1セルの違いが広がる速さ", fontsize=13)
c1.legend(fontsize=9.5, loc="upper left")
c1.grid(alpha=0.3)
c2.plot(gen, cnt, color=INK, lw=2.2)
c2.set_xlabel("世代")
c2.set_ylabel("2つの宇宙で食い違うセルの数")
c2.set_title(f"{len(rad)}世代後には {cnt[-1]:,} セルが違っている", fontsize=13)
c2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig4_lightcone.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif: ランダムから灰へ", flush=True)
NG = 220
rng = np.random.default_rng(31415)
g = (rng.random((NG, NG)) < 0.375).astype(np.uint8)
frames_at = sorted(set(np.unique(np.round(np.geomspace(1, 900, 46)).astype(int)).tolist()))
snaps, labels = [], []
cur, nxt = g.copy(), 0
for t in range(1, max(frames_at) + 1):
    cur = step(cur)
    if t in frames_at:
        snaps.append(cur.copy())
        labels.append(t)
snaps = [g] + snaps
labels = [0] + labels

fig, ax = plt.subplots(figsize=(4.7, 4.7))
ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(snaps[0], cmap="binary", interpolation="nearest", vmin=0, vmax=1)
txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=11.5,
              fontweight="bold", color="#ef476f")


def frame(k):
    im.set_data(snaps[k])
    txt.set_text(f"{labels[k]:,} 世代   密度 {snaps[k].mean():.4f}")
    return im, txt


FuncAnimation(fig, frame, frames=len(snaps), blit=False).save(
    "life.gif", writer=PillowWriter(fps=4), dpi=80)
plt.close(fig)

from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im2 = Image.open("life.gif")
fr = [f.copy().convert("RGB").quantize(colors=16) for f in ImageSequence.Iterator(im2)]
fr[0].save("life.gif", save_all=True, append_images=fr[1:], loop=0, duration=280, optimize=True)
print(f"  life.gif {os.path.getsize('life.gif')/1024:.0f}KB / {len(fr)}フレーム / {fr[0].size}")

# ---------------------------------------------------------------- 3D 時空図
print("3d: 時空図", flush=True)
import plotly.graph_objects as go  # noqa: E402

NS, TS = 38, 56
rng = np.random.default_rng(2718)
gg = (rng.random((NS, NS)) < 0.375).astype(np.uint8)
xs, ys, zs = [], [], []
for t in range(TS):
    yy, xx = np.nonzero(gg)
    xs.append(xx); ys.append(yy); zs.append(np.full(xx.size, t))
    gg = step(gg)
X = np.concatenate(xs); Y = np.concatenate(ys); Z = np.concatenate(zs)
print(f"  点の数 {X.size:,}")
fig3d = go.Figure(go.Scatter3d(
    x=X, y=Y, z=Z, mode="markers",
    marker=dict(size=2.2, color=Z, colorscale=[[0, "#ef476f"], [0.25, "#e8a33d"],
                                               [0.6, "#2f6f9e"], [1, "#0b1622"]],
                opacity=0.85, showscale=False),
    hoverinfo="skip",
))
fig3d.update_layout(
    title=dict(text="ライフゲームの時空図：高さが世代。柱＝静止物、斜めの筋＝グライダー", font=dict(size=14)),
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="世代",
               camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
               aspectratio=dict(x=1, y=1, z=1.15)),
    margin=dict(l=0, r=0, t=40, b=0), height=580, paper_bgcolor="rgba(0,0,0,0)")
fig3d.write_html("life-3d.html", include_plotlyjs="cdn")
print(f"  life-3d.html {os.path.getsize('life-3d.html')/1024:.0f}KB")
print("完了")
