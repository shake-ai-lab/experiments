"""GIF: ノイズ10%でブースティングが進むと決定境界が誤ラベル点に食いつき、
その点のサンプル重み(=マーカーサイズ)が膨張していく様子"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "Meiryo"

R = np.load("results.npz")
W = R["weights_history"]      # (1000, 300)
is_flipped = R["is_flipped"]
Xtr = R["Xtr_w"]
y = R["ytr_noisy_w"]
stumps = R["stumps_w"]        # (1000, 3)
alphas = R["alphas_w"]
test_err = R["ada_err"][0, 2]  # seed=0, noise=10% (重み追跡モデルと同一の学習)

C_BLUE = "#4477AA"
C_ORANGE = "#EE7733"
C_RED = "#CC3311"

# グリッド上のスコアを逐次計算
pad = 0.35
x_min, x_max = Xtr[:, 0].min() - pad, Xtr[:, 0].max() + pad
y_min, y_max = Xtr[:, 1].min() - pad, Xtr[:, 1].max() + pad
gx, gy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
G = np.column_stack([gx.ravel(), gy.ravel()])

FRAMES = [1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 140, 210, 320, 480, 700, 1000]
score = np.zeros(len(G))
frame_scores = {}
for t in range(1, 1001):
    j, thr, pol = stumps[t - 1]
    j = int(j)
    score += alphas[t - 1] * pol * np.where(G[:, j] > thr, 1.0, -1.0)
    if t in FRAMES:
        frame_scores[t] = score.copy()

images = []
for t in FRAMES:
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=80)
    Z = frame_scores[t].reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-1e9, 0, 1e9], colors=["#c9d7e8", "#f6dcc8"])
    ax.contour(gx, gy, Z, levels=[0], colors="k", linewidths=1.2)
    w = W[t - 1]
    s = np.clip(w * 300 * 30, 6, 1200)  # 重み1/300(初期値)で30pt
    for cls, col in [(-1, C_BLUE), (1, C_ORANGE)]:
        m = (y == cls) & ~is_flipped
        ax.scatter(Xtr[m, 0], Xtr[m, 1], s=s[m], c=col, alpha=0.8, linewidths=0)
        mf = (y == cls) & is_flipped
        ax.scatter(Xtr[mf, 0], Xtr[mf, 1], s=s[mf], c=col, alpha=0.9,
                   edgecolors=C_RED, linewidths=1.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"ラウンド {t}   テスト誤差 {test_err[t-1]:.3f}\n"
                 f"(赤縁=誤ラベル点, 大きさ=サンプル重み)", fontsize=11)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))

durations = [700] * len(images)
durations[-1] = 2500
images[0].save("figs/boost.gif", save_all=True, append_images=images[1:],
               duration=durations, loop=0, optimize=True)
import os
print(f"boost.gif: {os.path.getsize('figs/boost.gif') / 1024:.0f} KB, {len(images)} frames")
