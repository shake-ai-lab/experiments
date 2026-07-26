import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

def sinusoidal_pe(max_pos, d_model):
    pos = np.arange(max_pos)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates
    pe = np.zeros((max_pos, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

d_model = 64
max_pos = 100
PE_plot = sinusoidal_pe(max_pos, d_model)
PE = sinusoidal_pe(500, d_model)  # larger range saved for later experiments (exp3, exp4)
print("PE shape:", PE.shape)

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(PE_plot.T, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
ax.set_xlabel("位置 pos")
ax.set_ylabel("次元 i")
ax.set_title(f"sin/cos位置エンコーディング (d_model={d_model}, pos=0〜{max_pos-1})")
plt.colorbar(im, ax=ax, label="値")
plt.savefig("C:/tmp/posenc-experiment/fig2_pe_heatmap.png", dpi=150, bbox_inches="tight")
print("saved fig2_pe_heatmap.png")

# also plot a few individual dimension waveforms to show frequency differs by dimension
fig2, ax2 = plt.subplots(figsize=(9, 4))
for dim in [0, 1, 8, 9, 32, 33, 62, 63]:
    ax2.plot(np.arange(max_pos), PE_plot[:, dim], label=f"dim={dim}")
ax2.set_xlabel("位置 pos")
ax2.set_ylabel("値")
ax2.set_title("次元ごとの波形(次元番号が大きいほど低周波=ゆっくり変化)")
ax2.legend(fontsize=8, ncol=4)
plt.savefig("C:/tmp/posenc-experiment/fig2b_pe_waveforms.png", dpi=150, bbox_inches="tight")
print("saved fig2b_pe_waveforms.png")

np.save("C:/tmp/posenc-experiment/PE.npy", PE)
