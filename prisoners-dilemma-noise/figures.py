import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
E = json.load(open("echo.json", encoding="utf-8"))
SP = json.load(open("selfplay.json", encoding="utf-8"))
NAMES, SHORT = R["names"], R["short"]
INK, ACC, WARN, SUB, GOOD = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae", "#06a77d"

# ------------------------------------------------------------------ fig1
print("fig1", flush=True)
t0 = np.array(R["no_noise"]["total"])
t1 = np.array(R["noise_1pct"]["total"])
o = np.argsort(t0)
y = np.arange(len(o))
fig, ax = plt.subplots(figsize=(9.2, 5.8))
ax.barh(y - 0.2, t0[o], height=0.38, color=INK, label="手違いなし")
ax.barh(y + 0.2, t1[o], height=0.38, color=ACC, label="手違い1%")
ax.set_yticks(y)
ax.set_yticklabels([NAMES[i] for i in o], fontsize=10.5)
for k, i in enumerate(o):
    ax.text(t0[i] + 0.02, k - 0.2, f"{t0[i]:.3f}", va="center", fontsize=9, color="#3d4753")
    ax.text(t1[i] + 0.02, k + 0.2, f"{t1[i]:.3f}", va="center", fontsize=9, color="#3d4753")
ax.set_xlim(0, 3.05)
ax.axvline(3.0, ls=":", color=SUB)
ax.text(3.0, -0.9, "全員が協調し続けたときの3.000", ha="right", fontsize=9.5, color=SUB)
ax.set_xlabel("1手あたりの平均得点（全戦略との総当たり）")
ax.set_title("10戦略の総当たり戦。しっぺ返しは4位だった", fontsize=13.5)
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig("fig1_ranking.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig2 エコー
print("fig2", flush=True)
fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.2))
for ax, key, title, score in [
    (axes[0], "tft", "しっぺ返し × しっぺ返し", E["tft_score"]),
    (axes[1], "gtft", "寛容なしっぺ返し × 寛容なしっぺ返し", E["gtft_score"]),
]:
    h = np.array(E[key])
    ax.imshow(h, cmap=matplotlib.colors.ListedColormap([GOOD, ACC]), aspect="auto",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["A", "B"], fontsize=13)
    ax.set_xticks(range(0, 40, 5))
    ax.set_xticklabels(range(0, 40, 5), fontsize=11)
    ax.set_title(f"{title}   →  {score[0]} 点 vs {score[1]} 点", fontsize=14, pad=8)
    for x in range(41):
        ax.axvline(x - 0.5, color="white", lw=0.8)
    ax.axhline(0.5, color="white", lw=1.6)
axes[1].set_xlabel("手番（同じ乱数、手違い率2%）", fontsize=12)
fig.suptitle("緑＝協調、赤＝裏切り。同じ「手違い」から始まって、片方だけ立て直せた",
             fontsize=14.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig2_echo.png", dpi=125)
plt.close(fig)

# ------------------------------------------------------------------ fig3 sweep
print("fig3", flush=True)
noises = np.array(R["sweep"]["noises"]) * 100
sc = np.array(R["sweep"]["scores"])
pick = ["TFT", "GTFT", "TF2T", "GRUDGER", "ALLD", "PAVLOV"]
cols = {"TFT": INK, "GTFT": GOOD, "TF2T": WARN, "GRUDGER": "#7b5ea7", "ALLD": ACC,
        "PAVLOV": "#3d4753"}
fig, ax = plt.subplots(figsize=(9.0, 5.6))
for s in pick:
    i = SHORT.index(s)
    ax.plot(noises, sc[:, i], "o-", lw=2.4, ms=6, color=cols[s], label=NAMES[i])
top = [SHORT[int(np.argmax(row))] for row in sc]
prev = None
for x, w in zip(noises, top):
    if w != prev:
        ax.axvline(x, ls=":", color=SUB, lw=1.2)
        ax.text(x + 0.15, 1.72, f"ここから\n{w}が1位", fontsize=9.5, color="#3d4753")
        prev = w
ax.set_xlabel("手違いが起きる確率（%）")
ax.set_ylabel("1手あたりの平均得点")
ax.set_title("手違いが増えるほど、「許す」戦略の価値は消えていく", fontsize=13.5)
ax.legend(fontsize=9.5, ncol=2, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig3_noise_sweep.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig4 evolution
print("fig4", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
for ax, key, title in [(axes[0], "evolve_0", "手違いなし"), (axes[1], "evolve_1", "手違い1%")]:
    h = np.array(R[key])
    for i in range(h.shape[1]):
        if h[-1, i] > 0.01 or h[:, i].max() > 0.12:
            ax.plot(h[:, i], lw=2.4, label=NAMES[i],
                    color=cols.get(SHORT[i], None))
        else:
            ax.plot(h[:, i], lw=1.0, color="#dfe4ea")
    ax.set_xlabel("世代")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
axes[0].set_ylabel("集団に占める割合")
fig.suptitle("得点の高い戦略が増えていく進化シミュレーション", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig4_evolution.png", dpi=125)
plt.close(fig)

# ------------------------------------------------------------------ GIF
print("gif", flush=True)
fig, ax = plt.subplots(figsize=(7.4, 5.0))


def frame(k):
    ax.clear()
    v = sc[k]
    o = np.argsort(v)
    ax.barh(np.arange(len(o)), v[o],
            color=[ACC if j == o[-1] else INK for j in o])
    ax.set_yticks(np.arange(len(o)))
    ax.set_yticklabels([NAMES[i] for i in o], fontsize=9.5)
    ax.set_xlim(0, 2.85)
    ax.set_xlabel("1手あたりの平均得点")
    ax.set_title(f"手違いが起きる確率 {noises[k]:.1f}%   1位: {SHORT[int(np.argmax(v))]}",
                 fontsize=13.5)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()


FuncAnimation(fig, frame, frames=len(noises), blit=False).save(
    "sweep.gif", writer=PillowWriter(fps=1.3), dpi=84)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im = Image.open("sweep.gif")
fr = [f.copy().convert("RGB").quantize(colors=32) for f in ImageSequence.Iterator(im)]
fr[0].save("sweep.gif", save_all=True, append_images=fr[1:], loop=0, duration=1100, optimize=True)
print(f"  sweep.gif {os.path.getsize('sweep.gif')/1024:.0f}KB / {len(fr)}フレーム")

# ------------------------------------------------------------------ interactive
print("heatmap", flush=True)
import plotly.graph_objects as go  # noqa: E402

M = np.array(R["noise_1pct"]["matrix"])
fig = go.Figure(go.Heatmap(
    z=M, x=SHORT, y=SHORT,
    colorscale=[[0, "#ef476f"], [0.45, "#f1f3f5"], [1, "#06a77d"]],
    zmid=2.0, colorbar=dict(title="1手あたり得点"),
    hovertemplate="%{y} が %{x} と対戦したときの得点: %{z:.3f}<extra></extra>"))
fig.update_layout(
    title=dict(text="誰が誰に強いか（手違い1%、縦が自分・横が相手、1手あたり得点）",
               font=dict(size=14)),
    xaxis=dict(title="相手の戦略", side="bottom"), yaxis=dict(title="自分の戦略", autorange="reversed"),
    height=600, margin=dict(l=90, r=20, t=50, b=80), paper_bgcolor="rgba(0,0,0,0)")
fig.write_html("pd-heatmap.html", include_plotlyjs="cdn")
print(f"  pd-heatmap.html {os.path.getsize('pd-heatmap.html')/1024:.0f}KB")
print("完了")
