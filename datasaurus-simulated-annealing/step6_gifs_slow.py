"""GIF作り直し: 開始の恐竜と最終形をしっかり見せ、変化の速い序盤をゆっくり再生する。
元は開始フレームが0.1秒しか映らず、何から始まったのか分からなかった。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"

d = np.load("shape_results.npz")
dino = d["dino"]
SNAP = int(d["snap_every"])
names = [("circle", "円"), ("star", "星"), ("x_shape", "X"), ("salmon", "サーモン")]

FPS = 12
HOLD_START = int(FPS * 2.2)  # 恐竜のまま2.2秒静止
HOLD_END = int(FPS * 2.5)  # 完成形で2.5秒静止


def build_schedule(n_snaps):
    """序盤は1枚ずつ、後半は間引く。前後に静止フレームを足す"""
    early = list(range(0, min(30, n_snaps)))  # 序盤30枚は全部使う(変化が速い)
    late = list(range(30, n_snaps, 3))  # 以降は3枚に1枚
    seq = early + late
    if seq[-1] != n_snaps - 1:
        seq.append(n_snaps - 1)
    return [0] * HOLD_START + seq + [n_snaps - 1] * HOLD_END


def stats_of(pts):
    x, y = pts[:, 0], pts[:, 1]
    return (x.mean(), y.mean(), x.std(ddof=1), y.std(ddof=1), np.corrcoef(x, y)[0, 1])


# ---- GIF1: 4形状同時モーフィング ----
n_snaps = d["circle_snaps"].shape[0]
sched = build_schedule(n_snaps)
fig, axes = plt.subplots(2, 2, figsize=(6.6, 7.0))
scatters = []
for ax, (key, label) in zip(axes.flat, names):
    sc = ax.scatter([], [], s=5.5, color="#e4572e")
    ax.set_xlim(-5, 110)
    ax.set_ylim(-2, 108)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"恐竜 → {label}", fontsize=11)
    scatters.append(sc)
title = fig.suptitle("", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])


def update(fi):
    f = sched[fi]
    for sc, (key, _) in zip(scatters, names):
        snaps = d[f"{key}_snaps"]
        sc.set_offsets(snaps[min(f, len(snaps) - 1)])
    if fi < HOLD_START:
        title.set_text("出発点: Datasaurusの恐竜")
    elif fi >= len(sched) - HOLD_END:
        title.set_text("完成。平均・分散・相関は最初から一度も動いていない")
    else:
        title.set_text(f"変形中  (iter {min(f * SNAP, 1_500_000):,})")
    return scatters


ani = animation.FuncAnimation(fig, update, frames=len(sched), interval=1000 // FPS)
ani.save("morph4.gif", writer="pillow", fps=FPS, dpi=78)
print(f"saved morph4.gif ({len(sched)}フレーム, {len(sched)/FPS:.1f}秒)")
plt.close(fig)

# ---- GIF2: サーモン変形 + 統計量ライブ表示 ----
snaps = d["salmon_snaps"]
sched2 = build_schedule(snaps.shape[0])
fig = plt.figure(figsize=(7.4, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
ax = fig.add_subplot(gs[0])
ax_t = fig.add_subplot(gs[1])
ax_t.axis("off")
sc = ax.scatter([], [], s=6.5, color="#e4572e")
ax.set_xlim(-5, 110)
ax.set_ylim(-2, 108)
ax.set_xticks([])
ax.set_yticks([])
label = ax.set_title("", fontsize=12)
txt = ax_t.text(0.02, 0.5, "", fontsize=12.5, family="Meiryo", va="center", linespacing=2.1)


def update2(fi):
    f = min(sched2[fi], len(snaps) - 1)
    pts = snaps[f]
    sc.set_offsets(pts)
    mx, my, sx, sy, r = stats_of(pts)
    if fi < HOLD_START:
        label.set_text("出発点: Datasaurusの恐竜")
    elif fi >= len(sched2) - HOLD_END:
        label.set_text("完成: サーモン")
    else:
        label.set_text("変形中")
    txt.set_text(
        f"iter {min(f * SNAP, 1_500_000):>9,}\n\n"
        f"平均 x  = {mx:6.2f}\n平均 y  = {my:6.2f}\n"
        f"SD x    = {sx:6.2f}\nSD y    = {sy:6.2f}\n相関    = {r:6.2f}"
    )
    return sc, txt


ani2 = animation.FuncAnimation(fig, update2, frames=len(sched2), interval=1000 // FPS)
ani2.save("salmon_stats.gif", writer="pillow", fps=FPS, dpi=78)
print(f"saved salmon_stats.gif ({len(sched2)}フレーム, {len(sched2)/FPS:.1f}秒)")
