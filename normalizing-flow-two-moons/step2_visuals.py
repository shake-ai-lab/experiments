"""Flow記事の可視化一式 (見せ方標準v2適用):
(0) セットアップ図解 (データ⇄カップリング層⇄ガウス、両方向矢印)
(1) レイヤー変形モーフ mp4 + GIF (punchline)
(2) 厳密な密度マップ (主張文タイトル+注釈)
(3) real/diffusion/GAN/flow の2x2比較 (モバイル対応)
(4) スライダー付きplotly (層を手で進める)"""
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12, "lines.linewidth": 2.2})

# ---- モデル読み込み ----
d = np.load("flow_model.npz")
L, H, S_SCALE = 6, 64, 2.0
masks = d["masks"].tolist()
layers = [
    {k: d[f"L{i}_{k}"] for k in ("W1", "b1", "W2", "b2", "W3", "b3")} for i in range(L)
]
X = np.load("C:/tmp/diffusion-experiment/X0.npy")
rng = np.random.default_rng(3)


def mlp_forward(p, a):
    h1 = np.tanh(a @ p["W1"] + p["b1"])
    h2 = np.tanh(h1 @ p["W2"] + p["b2"])
    out = h2 @ p["W3"] + p["b3"]
    return S_SCALE * np.tanh(out[:, 0:1]), out[:, 1:2]


def flow_forward(x):
    logdet = np.zeros((x.shape[0], 1))
    h = x
    for p, m in zip(layers, masks):
        a, b = h[:, m : m + 1], h[:, 1 - m : 2 - m]
        s, t = mlp_forward(p, a)
        hn = np.empty_like(h)
        hn[:, m : m + 1] = a
        hn[:, 1 - m : 2 - m] = b * np.exp(s) + t
        logdet += s
        h = hn
    return h, logdet


def flow_inverse_stages(z):
    h = z
    stages = [h.copy()]
    for p, m in zip(reversed(layers), reversed(masks)):
        a, b = h[:, m : m + 1], h[:, 1 - m : 2 - m]
        s, t = mlp_forward(p, a)
        hn = np.empty_like(h)
        hn[:, m : m + 1] = a
        hn[:, 1 - m : 2 - m] = (b - t) * np.exp(-s)
        h = hn
        stages.append(h.copy())
    return stages


# ---- (0) セットアップ図解 ----
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")
boxes = [("ガウス雲\nz ~ N(0, I)", 0.06, "#dfe8f7")] + [
    (f"層{i+1}", 0.235 + i * 0.089, "#fbe3dc") for i in range(6)
] + [("two moons\nデータ x", 0.905, "#e0f2e4")]
for label, cx, col in boxes:
    w = 0.115 if "\n" in label else 0.065
    ax.add_patch(plt.Rectangle((cx - w / 2, 0.35), w, 0.34, fc=col, ec="#555", lw=1.4))
    ax.text(cx, 0.52, label, ha="center", va="center", fontsize=11)
ax.annotate("", xy=(0.88, 0.76), xytext=(0.12, 0.76),
            arrowprops=dict(arrowstyle="->", lw=2, color="#e4572e"))
ax.text(0.5, 0.85, "生成: z を6つの層で順に変形 (逆変換)", ha="center", fontsize=12, color="#e4572e")
ax.annotate("", xy=(0.12, 0.24), xytext=(0.88, 0.24),
            arrowprops=dict(arrowstyle="->", lw=2, color="#2e6ee4"))
ax.text(0.5, 0.13, "学習・尤度計算: x を z に戻す (順変換)。全ステップが可逆なので log p(x) が厳密に求まる",
        ha="center", fontsize=12, color="#2e6ee4")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Normalizing Flowの構造: 可逆な変換の鎖 (どちらの向きにも流せる)", fontsize=14)
fig.tight_layout()
fig.savefig("fig0_setup.png", dpi=110, bbox_inches="tight")
print("saved fig0_setup.png")
plt.close(fig)

# ---- (1) レイヤー変形モーフ (mp4 + GIF) ----
z = rng.normal(0, 1, (1200, 2))
stages = flow_inverse_stages(z)  # 7 stages
angle = np.arctan2(z[:, 1], z[:, 0])
PAUSE, TRANS = 6, 10
frames = []
for i in range(len(stages) - 1):
    for _ in range(PAUSE if i == 0 else 2):
        frames.append((stages[i], i, 0.0))
    for k in range(1, TRANS + 1):
        a = k / TRANS
        frames.append(((1 - a) * stages[i] + a * stages[i + 1], i, a))
for _ in range(PAUSE * 2):
    frames.append((stages[-1], len(stages) - 1, 0.0))

fig, ax = plt.subplots(figsize=(5.6, 5.6))


def update(fi):
    pts, si, a = frames[fi]
    ax.clear()
    ax.scatter(pts[:, 0], pts[:, 1], s=5, c=angle, cmap="hsv", alpha=0.85)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    label = "ガウス雲 (出発点)" if si == 0 and a == 0 else (
        "two moons (完成)" if si >= 6 else f"層 {si + (1 if a > 0 else 0)}/6 を通過中"
    )
    ax.set_title(f"可逆変換の鎖: {label}", fontsize=14)
    return []


ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=90)
ani.save("flow_morph.mp4", writer=animation.FFMpegWriter(fps=12, bitrate=1800), dpi=100)
print("saved flow_morph.mp4")
ani.save("flow_morph.gif", writer="pillow", fps=10, dpi=72)
print("saved flow_morph.gif")
plt.close(fig)

# ---- (2) 厳密な密度マップ ----
g = np.linspace(-3.2, 3.2, 220)
gx, gy = np.meshgrid(g, g)
grid = np.c_[gx.ravel(), gy.ravel()]
zg, logdet = flow_forward(grid)
logp = -0.5 * (zg**2).sum(1) - np.log(2 * np.pi) + logdet.ravel()
P = np.exp(logp).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(7, 5.6))
im = ax.pcolormesh(gx, gy, P, cmap="magma", shading="auto")
fig.colorbar(im, ax=ax, label="p(x) (厳密な確率密度)")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Flowだけが「この点の確率密度は?」に即答できる", fontsize=14)
ax.annotate("密度が薄く残った\n三日月間のブリッジ", xy=(-0.6, 0.5), xytext=(-3.0, 2.6),
            color="w", fontsize=11,
            arrowprops=dict(arrowstyle="->", color="w", lw=1.5))
fig.tight_layout()
fig.savefig("fig2_density.png", dpi=110, bbox_inches="tight")
print("saved fig2_density.png")
plt.close(fig)

# ---- (3) 2x2三つ巴比較 ----
diff_samples = np.load("C:/tmp/diffusion-experiment/samples.npz")["0"]
gan_snaps = np.load("C:/tmp/gan-experiment/snapshots.npz")
gan_final = gan_snaps[str(max(int(k) for k in gan_snaps.files))]
flow_final = flow_inverse_stages(rng.normal(0, 1, (1000, 2)))[-1]
panels = [
    ("本物のデータ", X[:1000], "#555"),
    ("拡散モデル (前々回)", diff_samples[:1000], "#3aa655"),
    ("GAN (前回)", gan_final[:1000], "#2e6ee4"),
    ("Flow (今回)", flow_final, "#e4572e"),
]
fig, axes = plt.subplots(2, 2, figsize=(8.6, 8))
for ax, (name, pts, col) in zip(axes.flat, panels):
    ax.scatter(pts[:, 0], pts[:, 1], s=5, color=col, alpha=0.8)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=13)
fig.suptitle("同じtwo moons・同規模の手書きMLPで、生成モデル3流派を並べる", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig3_three_families.png", dpi=110, bbox_inches="tight")
print("saved fig3_three_families.png")
plt.close(fig)

# ---- (4) スライダー付きplotly ----
z2 = rng.normal(0, 1, (900, 2))
stages2 = flow_inverse_stages(z2)
ang2 = np.arctan2(z2[:, 1], z2[:, 0])
figp = go.Figure()
for i, st in enumerate(stages2):
    figp.add_trace(
        go.Scatter(
            x=st[:, 0], y=st[:, 1], mode="markers", visible=(i == 0),
            marker=dict(size=4.5, color=ang2, colorscale="HSV"), hoverinfo="skip",
        )
    )
steps = []
labels = ["ガウス雲 z"] + [f"層{i}を通過後" for i in range(1, 7)]
for i, lb in enumerate(labels):
    steps.append(
        dict(method="update", label=lb,
             args=[{"visible": [j == i for j in range(len(stages2))]}])
    )
figp.update_layout(
    sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="表示中: "))],
    title="スライダーで6つの層を1枚ずつ進める (ガウス雲→two moons)。点の色は出発時の角度",
    xaxis=dict(range=[-3.2, 3.2], visible=False),
    yaxis=dict(range=[-3.2, 3.2], visible=False, scaleanchor="x"),
    margin=dict(l=10, r=10, t=60, b=10), height=620,
)
figp.write_html("flow-layers-slider.html", include_plotlyjs="cdn", full_html=True)
print("saved flow-layers-slider.html")
