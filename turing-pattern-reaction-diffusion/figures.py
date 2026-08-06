import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import ndimage

from core import simulate, describe  # noqa: F401

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB, GOOD = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae", "#06a77d"
CM = "turbo"


def classify(d):
    """面積比で3つに分ける。100%は「模様」ではなく一様に埋まった状態。"""
    if d["frac"] > 0.90:
        return "一様（全面が高濃度）"
    if d["frac"] < 0.02:
        return "消滅"
    return "模様あり"


def wavelength(v):
    """動径平均スペクトルのピークを放物線補間して、格子単位の主波長を返す"""
    N = v.shape[0]
    f = np.abs(np.fft.fftshift(np.fft.fft2(v - v.mean())))
    c = N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(yy - c, xx - c).astype(int)
    prof = np.bincount(rr.ravel(), f.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    prof[:3] = 0
    i = int(np.argmax(prof[: N // 2]))
    if 1 <= i < N // 2 - 1:
        y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
        denom = (y0 - 2 * y1 + y2)
        d = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        d = 0.0
    return N / (i + d) if (i + d) > 0 else np.nan


# ---------------------------------------------------------------- 分類し直し
FS, KS = R["phase"]["F"], R["phase"]["k"]
counts = {"模様あり": 0, "一様（全面が高濃度）": 0, "消滅": 0}
table = {}
for F in FS:
    for k in KS:
        d = R["phase"]["grid"][f"{F:.3f}_{k:.3f}"]
        c = classify(d)
        table[(F, k)] = c
        counts[c] += 1
print("相図の分類:", counts, flush=True)

# ---------------------------------------------------------------- fig1 相図
print("fig1", flush=True)
fig, axes = plt.subplots(len(FS), len(KS), figsize=(11.2, 12.6))
for i, F in enumerate(FS):
    for j, k in enumerate(KS):
        _, v, _ = simulate(F, k, 9000, N=110, seed=7)
        ax = axes[i][j]
        ax.imshow(v, cmap=CM, vmin=0, vmax=0.5, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        c = table[(F, k)]
        col = {"模様あり": ACC, "一様（全面が高濃度）": SUB, "消滅": "#c0c6cc"}[c]
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(2.4)
        if i == 0:
            ax.set_title(f"k={k:.3f}", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"F={F:.3f}", fontsize=11)
fig.suptitle(f"供給率Fと除去率kを変えただけ（赤枠＝模様あり {counts['模様あり']}通り／"
             f"灰枠＝一様 {counts['一様（全面が高濃度）']}／薄枠＝消滅 {counts['消滅']}）", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("fig1_phase.png", dpi=100)
plt.close(fig)

# ---------------------------------------------------------------- fig2 代表例と波長
print("fig2", flush=True)
cases = [(0.030, 0.055), (0.022, 0.051), (0.038, 0.061), (0.046, 0.062), (0.026, 0.058),
         (0.034, 0.065)]
fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.8))
rows2 = []
for ax, (F, k) in zip(axes.ravel(), cases):
    _, v, _ = simulate(F, k, 14000, N=256, seed=3)
    wl = wavelength(v)
    m = v > 0.25
    _, n = ndimage.label(m)
    rows2.append({"F": F, "k": k, "wavelength": float(wl), "blobs": int(n),
                  "frac": float(m.mean())})
    ax.imshow(v, cmap=CM, vmin=0, vmax=0.5, interpolation="bilinear")
    ax.set_title(f"F={F:.3f}  k={k:.3f}\n主波長 {wl:.1f} 格子 ／ 塊 {n:,} 個 ／ 面積 {m.mean()*100:.1f}%",
                 fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"  F={F:.3f} k={k:.3f}: 波長 {wl:.2f} / 塊 {n} / 面積 {m.mean()*100:.1f}%", flush=True)
fig.suptitle("同じ式のまま、2つの数字を変えただけ（256×256、14,000ステップ）", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("fig2_cases.png", dpi=112)
plt.close(fig)

# ---------------------------------------------------------------- fig3 拡散比
print("fig3", flush=True)
rows3 = []
for r in [1.0, 1.1, 1.2, 1.35, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]:
    _, v, _ = simulate(0.030, 0.055, 9000, N=160, seed=11, du=0.16, dv=0.16 / r)
    m = v > 0.25
    rows3.append({"ratio": r, "frac": float(m.mean()), "wl": float(wavelength(v))})
    print(f"  Du/Dv={r:.2f}: 面積 {m.mean()*100:5.2f}% / 波長 {wavelength(v):.1f}", flush=True)
rr = np.array([x["ratio"] for x in rows3])
ff = np.array([x["frac"] for x in rows3]) * 100
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.0))
a1.plot(rr, ff, "o-", color=ACC, lw=2.4, ms=8)
a1.axvline(1.0, ls=":", color=SUB)
a1.annotate("拡散の速さが同じだと\n模様はまったくできない（0.00%）", (1.0, 1),
            textcoords="offset points", xytext=(16, 30), fontsize=11, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK))
a1.set_xlabel("拡散係数の比 Du/Dv（活性化因子より抑制因子が速く広がる度合い）")
a1.set_ylabel("模様が占める面積（%）")
a1.set_title("速さの差がないと、模様は生まれない", fontsize=13)
a1.grid(alpha=0.3)
for r, ax_pos in [(1.0, 0), (1.5, 1), (3.0, 2)]:
    pass
imgs = []
for r in [1.0, 1.5, 3.0]:
    _, v, _ = simulate(0.030, 0.055, 9000, N=160, seed=11, du=0.16, dv=0.16 / r)
    imgs.append((r, v))
a2.axis("off")
for i, (r, v) in enumerate(imgs):
    sub = fig.add_axes([0.55 + i * 0.145, 0.30, 0.135, 0.42])
    sub.imshow(v, cmap=CM, vmin=0, vmax=0.5, interpolation="bilinear")
    sub.set_xticks([]); sub.set_yticks([])
    sub.set_title(f"Du/Dv={r:g}", fontsize=11.5)
fig.text(0.75, 0.15, "同じF・k・同じ初期値。拡散の速さの比だけを変えた", ha="center", fontsize=11,
         color="#3d4753")
fig.savefig("fig3_ratio.png", dpi=118, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
FR = 44
marks = np.unique(np.round(np.geomspace(40, 16000, FR)).astype(int))
_, _, snaps = simulate(0.030, 0.055, int(marks[-1]), N=200, seed=5, snap_at=set(marks.tolist()))
keys = sorted(snaps)
fig, ax = plt.subplots(figsize=(4.8, 4.8))
ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(snaps[keys[0]], cmap=CM, vmin=0, vmax=0.5, interpolation="bilinear")
txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=12,
              fontweight="bold", color="white")


def fr(i):
    im.set_data(snaps[keys[i]])
    txt.set_text(f"{keys[i]:,} ステップ")
    return im, txt


FuncAnimation(fig, fr, frames=len(keys), blit=False).save(
    "turing.gif", writer=PillowWriter(fps=6), dpi=78)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im2 = Image.open("turing.gif")
fl = [f.copy().convert("RGB").quantize(colors=64) for f in ImageSequence.Iterator(im2)]
fl[0].save("turing.gif", save_all=True, append_images=fl[1:], loop=0, duration=180, optimize=True)
print(f"  turing.gif {os.path.getsize('turing.gif')/1024:.0f}KB / {len(fl)}フレーム")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

sel = [(0.022, 0.051), (0.026, 0.055), (0.030, 0.055), (0.034, 0.060),
       (0.038, 0.061), (0.046, 0.062)]
frames = []
for F, k in sel:
    _, v, _ = simulate(F, k, 12000, N=128, seed=3)
    z = np.rint(np.clip(v, 0, 0.5) * 400).astype(int)
    frames.append(go.Frame(data=[go.Heatmap(z=z.tolist(), zmin=0, zmax=200,
                                            colorscale="Turbo", showscale=False)],
                           name=f"F={F:.3f}, k={k:.3f}"))
fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(
    title=dict(text="供給率Fと除去率kを変えると模様はどう変わるか（下のスライダー）", font=dict(size=14)),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    height=560, margin=dict(l=10, r=10, t=48, b=70), paper_bgcolor="rgba(0,0,0,0)",
    sliders=[dict(active=0, currentvalue=dict(prefix=""),
                  steps=[dict(method="animate", label=f.name,
                              args=[[f.name], dict(mode="immediate",
                                                   frame=dict(duration=0, redraw=True))])
                         for f in frames])])
fig.write_html("turing-slider.html", include_plotlyjs="cdn")
print(f"  turing-slider.html {os.path.getsize('turing-slider.html')/1024:.0f}KB")

# ---------------------------------------------------------------- hero
print("hero", flush=True)
_, v, _ = simulate(0.030, 0.055, 20000, N=520, seed=21)
h = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:1024, :1024]
img = (plt.get_cmap(CM)(np.clip(h / 0.45, 0, 1))[:, :, :3] * 255).astype(np.uint8)
canvas = np.zeros((1024, 1536, 3), dtype=np.uint8)
wide = np.repeat(np.repeat(v, 2, axis=0), 3, axis=1)
wide = wide[:1024, :1536]
img = (plt.get_cmap(CM)(np.clip(wide / 0.45, 0, 1))[:, :, :3] * 255).astype(np.uint8)
Image.fromarray(img).save("hero.png")
print(f"  hero.png {os.path.getsize('hero.png')/1024:.0f}KB {img.shape}")
json.dump({"cases": rows2, "ratio": rows3, "counts": counts},
          open("analysis.json", "w", encoding="utf-8"), ensure_ascii=False)
print("完了")
