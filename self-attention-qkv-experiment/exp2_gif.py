import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

seq_len = 16
d_ks = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# fix Q,K generation per d_k with same underlying "identity" of the query token (index 0)
# by reusing a single big random matrix and slicing, so growing d_k feels like "adding more dimensions"
# to the same vectors rather than redrawing unrelated ones.
max_d = max(d_ks)
Q_full = rng.normal(0, 1, size=(seq_len, max_d))
K_full = rng.normal(0, 1, size=(seq_len, max_d))

query_idx = 0
frames_data = []
for d_k in d_ks:
    Q = Q_full[:, :d_k]
    K = K_full[:, :d_k]
    scores = Q @ K.T
    attn_noscale = softmax(scores)[query_idx]
    attn_scaled = softmax(scores / np.sqrt(d_k))[query_idx]
    frames_data.append((d_k, attn_noscale, attn_scaled))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
x = np.arange(seq_len)
bars0 = axes[0].bar(x, frames_data[0][1], color="crimson")
bars1 = axes[1].bar(x, frames_data[0][2], color="steelblue")
for ax, title in zip(axes, ["スケーリングなし", "1/√d_k でスケーリング"]):
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Key位置(0=Query自身)")
    ax.set_title(title)
axes[0].set_ylabel("attention重み(Query位置0)")
suptitle = fig.suptitle("", fontsize=14)

def update(frame_idx):
    d_k, attn_ns, attn_s = frames_data[frame_idx]
    for bar, h in zip(bars0, attn_ns):
        bar.set_height(h)
    for bar, h in zip(bars1, attn_s):
        bar.set_height(h)
    suptitle.set_text(f"d_k = {d_k}")
    return list(bars0) + list(bars1) + [suptitle]

# hold the last frame a bit longer by repeating it
frame_order = list(range(len(frames_data))) + [len(frames_data) - 1] * 6

ani = animation.FuncAnimation(fig, update, frames=frame_order, interval=700, blit=False)
ani.save("C:/tmp/attention-experiment/exp2_scaling.gif", writer=animation.PillowWriter(fps=1.4))
print("saved exp2_scaling.gif")
