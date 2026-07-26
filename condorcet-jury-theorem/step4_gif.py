"""pを0.30から0.90へ動かしながら「正解率 vs N」曲線を掃引するGIF。
p=0.5を境に、曲線が下(0)へ向かう形から上(1)へ向かう形へひっくり返る瞬間を見せる。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

plt.rcParams["font.family"] = "Meiryo"

data = np.load("grid_results.npz")
N_VALUES, P_VALUES, acc_grid = data["N_VALUES"], data["P_VALUES"], data["acc_grid"]

frame_idx = list(range(0, len(P_VALUES), 2))  # p=0.30,0.32,...,0.90 相当のストライド
frames = []

for k, j in enumerate(frame_idx):
    p = P_VALUES[j]
    curve = acc_grid[:, j]
    color = "#1f77b4" if p > 0.5 else ("#7f7f7f" if abs(p - 0.5) < 1e-9 else "#d62728")

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=80)
    ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=1.2, zorder=0)
    ax.axvline(0, color="white")
    ax.plot(N_VALUES, curve, color=color, linewidth=3)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("投票者の人数 N", fontsize=12)
    ax.set_ylabel("多数決の正解率", fontsize=12)
    status = "賢くなる (p>0.5)" if p > 0.501 else ("境界 (p=0.50)" if abs(p - 0.5) < 1e-9 else "悪くなる (p<0.5)")
    ax.set_title(f"p = {p:.2f}   →  {status}", fontsize=13.5, color=color if p != 0.5 else "#555555")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frames.append(frame)
    plt.close(fig)

# 最初と、p=0.5をまたぐ瞬間、最後を長めに見せる
hold_start = [frames[0]] * 6
j50_in_stride = min(range(len(frame_idx)), key=lambda k: abs(P_VALUES[frame_idx[k]] - 0.5))
hold_mid = [frames[j50_in_stride]] * 6
hold_end = [frames[-1]] * 8

sequence = hold_start + frames[1:j50_in_stride] + hold_mid + frames[j50_in_stride + 1:] + hold_end

imageio.mimsave("sweep.gif", sequence, duration=0.12)
import os
print("saved sweep.gif, size(KB):", round(os.path.getsize("sweep.gif") / 1024, 1))
