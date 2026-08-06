import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

from core import julia_escape

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("results.json", encoding="utf-8"))
INK, ACC, WARN, SUB = "#2f6f9e", "#ef476f", "#e8a33d", "#8d99ae"
CMAP = LinearSegmentedColormap.from_list(
    "shake", ["#050a12", "#0d2340", "#1b4a78", "#2f81b5", "#7cc0e0", "#e3f0f8", "#ffffff"])


def escape_counts(c, n, maxiter, rad=1.6):
    """色付け用に、脱出までの反復回数を返す(0=脱出しなかった)"""
    x = np.linspace(-rad, rad, n)
    Z = (x[None, :] + 1j * x[:, None]).ravel()
    idx = np.arange(Z.size)
    Zc = Z.copy()
    out = np.zeros(Z.size, dtype=np.int32)
    for i in range(maxiter):
        Zc = Zc * Zc + c
        esc = Zc.real ** 2 + Zc.imag ** 2 > 4.0
        if esc.any():
            out[idx[esc]] = i + 1
            k = ~esc
            idx, Zc = idx[k], Zc[k]
            if idx.size == 0:
                break
    return out.reshape(n, n)


def colorize(e, maxiter):
    v = e.astype(float)
    v[e == 0] = np.nan
    v = np.log1p(v) / np.log1p(maxiter)
    img = CMAP(np.clip(v * 1.8, 0, 1))
    img[np.isnan(v)] = (0.02, 0.03, 0.05, 1.0)
    return img


G = R["gridA"]
nc = np.array(G["ncomp"])
md = np.array(G["mand"])
ext = [G["cx"][0], G["cx"][1], G["cy"][0], G["cy"][1]]

# ---------------------------------------------------------------- fig1
print("fig1", flush=True)
pred = nc == 1
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8))
axes[0].imshow(pred, cmap="binary", origin="lower", extent=ext, interpolation="nearest")
axes[0].set_title("ジュリア集合の連結成分が\nちょうど1個だった c", fontsize=12.5)
axes[1].imshow(md, cmap="binary", origin="lower", extent=ext, interpolation="nearest")
axes[1].set_title("定義どおりに描いた\nマンデルブロ集合", fontsize=12.5)
diff = np.zeros(pred.shape + (3,))
diff[...] = 1.0
diff[pred & md] = (0.15, 0.18, 0.22)
diff[~pred & md] = (0.937, 0.278, 0.435)     # 取りこぼし
diff[pred & ~md] = (0.117, 0.654, 0.490)     # 誤検出
axes[2].imshow(diff, origin="lower", extent=ext, interpolation="nearest")
axes[2].set_title(f"食い違い（赤={int((~pred & md).sum())}点の取りこぼし、\n"
                  f"緑={int((pred & ~md).sum())}点の誤検出）", fontsize=12.5)
for a in axes:
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle(f"c を {nc.shape[0]}×{nc.shape[0]} に振り、各ジュリア集合の連結成分を数えただけ"
             f"（一致率 {(pred == md).mean()*100:.2f}%）", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig1_reconstruct.png", dpi=125)
plt.close(fig)

# ---------------------------------------------------------------- fig2 ギャラリー
print("fig2", flush=True)
samples = [
    (complex(-0.123, 0.745), "ダグラス・ラビット\nc = -0.123+0.745i", "内側"),
    (complex(-1.0, 0.0), "バジリカ\nc = -1", "内側"),
    (complex(0.0, 1.0), "デンドライト\nc = i", "境界上"),
    (complex(-0.75, 0.11), "しっぽの生えた形\nc = -0.75+0.11i", "内側"),
    (complex(0.3, 0.5), "砂に散る\nc = 0.3+0.5i", "外側"),
    (complex(-0.8, 0.4), "砂に散る\nc = -0.8+0.4i", "外側"),
]
fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.6))
for ax, (c, name, where) in zip(axes.ravel(), samples):
    e = escape_counts(c, 460, 400, rad=1.55)
    ax.imshow(colorize(e, 400), origin="lower", interpolation="nearest")
    K = julia_escape(c, -2, 2, -2, 2, 260, 300)
    _, n = ndimage.label(K, structure=np.ones((3, 3)))
    col = ACC if where == "外側" else ("#e8a33d" if where == "境界上" else "#7cc0e0")
    ax.set_title(f"{name}\nMの{where} ／ 連結成分 {n:,} 個", fontsize=11.5, color=col)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("同じ z→z²+c で、c を変えただけ", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("fig2_gallery.png", dpi=118)
plt.close(fig)

# ---------------------------------------------------------------- fig3 次元
print("fig3", flush=True)
rows = R["dim"]["rows"]
cv = np.array([r["c"] for r in rows])
dc = np.array([r["D_corrected"] for r in rows])
ru = np.array([r["ruelle"] for r in rows])
o = np.argsort(cv)
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(cv[o], ru[o], "--", color=SUB, lw=2.0, label="Ruelleの近似式  1 + |c|²/(4 ln2)")
ax.plot(cv[o], dc[o], "o-", color=ACC, lw=2.4, ms=8, label="box-countingの実測（円の系統誤差を補正）")
ax.axhline(1.0, ls=":", color="#c0c6cc")
ax.annotate("c=0 は単位円そのもの（次元1）", (0.0, 1.0), textcoords="offset points",
            xytext=(8, -22), fontsize=10, color="#5c6672")
ax.annotate("c=-0.75 はカージオイドの先端\n（Mの境界上）", (-0.75, dc[o][0]),
            textcoords="offset points", xytext=(14, -4), fontsize=10, color=ACC)
ax.set_xlabel("c（実軸上）")
ax.set_ylabel("ジュリア集合の境界次元")
ax.set_title("cが0から離れるほど、ジュリア集合はほつれていく", fontsize=13.5)
ax.legend(fontsize=10, loc="upper center")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig3_dimension.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- GIF
print("gif", flush=True)
FR, NG = 40, 330
th = np.linspace(0, 2 * np.pi, FR, endpoint=False)
# メインカージオイドのわずかに内側をなぞる
path = 0.985 * (np.exp(1j * th) / 2 - np.exp(2j * th) / 4)
fig, ax = plt.subplots(figsize=(4.7, 4.7))
ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(np.zeros((NG, NG, 4)), origin="lower")
txt = ax.text(0.03, 0.965, "", transform=ax.transAxes, va="top", color="white",
              fontsize=11.5, fontweight="bold")


def frame(k):
    c = complex(path[k])
    im.set_data(colorize(escape_counts(c, NG, 260, rad=1.5), 260))
    txt.set_text(f"c = {c.real:+.3f} {c.imag:+.3f}i")
    return im, txt


FuncAnimation(fig, frame, frames=FR, blit=False).save(
    "julia.gif", writer=PillowWriter(fps=6), dpi=78)
plt.close(fig)
from PIL import Image, ImageSequence  # noqa: E402
import os  # noqa: E402

im2 = Image.open("julia.gif")
fr = [f.copy().convert("RGB").quantize(colors=64) for f in ImageSequence.Iterator(im2)]
fr[0].save("julia.gif", save_all=True, append_images=fr[1:], loop=0, duration=170, optimize=True)
print(f"  julia.gif {os.path.getsize('julia.gif')/1024:.0f}KB / {len(fr)}フレーム / {fr[0].size}")

# ---------------------------------------------------------------- interactive
print("interactive", flush=True)
import plotly.graph_objects as go  # noqa: E402

CS = [complex(-0.123, 0.745), complex(-0.4, 0.6), complex(-0.7269, 0.1889),
      complex(-1.0, 0.0), complex(0.285, 0.01), complex(0.0, 0.8),
      complex(-0.8, 0.156), complex(0.35, 0.35)]
NS = 150
frames, buttons = [], []
for i, c in enumerate(CS):
    e = escape_counts(c, NS, 200, rad=1.55)
    z = np.log1p(e.astype(float))
    z[e == 0] = 0.0
    frames.append(go.Frame(data=[go.Heatmap(z=z, showscale=False,
                                            colorscale=[[0, "#050a12"], [0.3, "#1b4a78"],
                                                        [0.65, "#2f81b5"], [1, "#e3f0f8"]])],
                           name=f"{c.real:+.3f}{c.imag:+.3f}i"))
fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(
    title=dict(text="cを変えるとジュリア集合はどう変わるか（スライダーを動かしてみてください）",
               font=dict(size=14)),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    height=580, margin=dict(l=10, r=10, t=50, b=70), paper_bgcolor="rgba(0,0,0,0)",
    sliders=[dict(active=0, currentvalue=dict(prefix="c = "),
                  steps=[dict(method="animate", label=f.name,
                              args=[[f.name], dict(mode="immediate",
                                                   frame=dict(duration=0, redraw=True))])
                         for f in frames])])
fig.write_html("julia-slider.html", include_plotlyjs="cdn")
print(f"  julia-slider.html {os.path.getsize('julia-slider.html')/1024:.0f}KB")

# ---------------------------------------------------------------- hero
print("hero", flush=True)
W, H = 1536, 1024
c = complex(-0.7269, 0.1889)     # 渦を巻いたデンドライト
x = np.linspace(-1.55 * W / H, 1.55 * W / H, W)
y = np.linspace(-1.55, 1.55, H)
Z = (x[None, :] + 1j * y[:, None]).ravel()
idx = np.arange(Z.size)
Zc = Z.copy()
val = np.zeros(Z.size)
MI = 600
for i in range(MI):
    Zc = Zc * Zc + c
    m2 = Zc.real ** 2 + Zc.imag ** 2
    esc = m2 > 256.0
    if esc.any():
        val[idx[esc]] = i + 1 - np.log2(np.log(np.sqrt(m2[esc])))
        k = ~esc
        idx, Zc = idx[k], Zc[k]
        if idx.size == 0:
            break
val = val.reshape(H, W)
t = np.log1p(val) / np.log1p(val.max())
img = (CMAP(np.clip(t * 2.1, 0, 1))[:, :, :3] * 255).astype(np.uint8)
img[val == 0] = (5, 8, 14)
Image.fromarray(np.flipud(img)).save("hero.png")
print(f"  hero.png {os.path.getsize('hero.png')/1024:.0f}KB")
print("完了")
