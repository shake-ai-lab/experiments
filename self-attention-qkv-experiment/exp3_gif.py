import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(1)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

seq_len = 10
d_k = 32
Q = rng.normal(0, 1, size=(seq_len, d_k))
K = rng.normal(0, 1, size=(seq_len, d_k))
scores = Q @ K.T / np.sqrt(d_k)

# frame t: only rows 0..t have been "generated" (as in autoregressive decoding),
# and each of those rows can only attend to columns <= its own row index.
attn_frames = []
for t in range(seq_len):
    full_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    scores_masked = np.where(full_mask, -np.inf, scores)
    attn = softmax(scores_masked)
    display = np.full((seq_len, seq_len), np.nan)
    display[: t + 1, :] = attn[: t + 1, :]
    attn_frames.append(display)

fig, ax = plt.subplots(figsize=(6, 5.5))
cmap = plt.cm.viridis.copy()
cmap.set_bad("#e0e0e0")
im = ax.imshow(attn_frames[0], cmap=cmap, vmin=0, vmax=0.6)
ax.set_xlabel("Key位置(参照先)")
ax.set_ylabel("Query位置(参照元)")
title = ax.set_title("生成済みトークン数 = 1 / 10")
plt.colorbar(im, ax=ax, fraction=0.046)

def update(t):
    im.set_data(attn_frames[t])
    title.set_text(f"生成済みトークン数 = {t + 1} / {seq_len}")
    return [im, title]

frame_order = list(range(seq_len)) + [seq_len - 1] * 6
ani = animation.FuncAnimation(fig, update, frames=frame_order, interval=500, blit=False)
ani.save("C:/tmp/attention-experiment/exp3_causal.gif", writer=animation.PillowWriter(fps=2))
print("saved exp3_causal.gif")
