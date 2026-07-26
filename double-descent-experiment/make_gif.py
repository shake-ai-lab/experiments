import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "Meiryo"

d = np.load("results.npz")
degrees = d["degrees"]
params = d["params"]
acc_test = d["acc_test"]
x_grid = d["x_grid"]
y_grid_true = d["y_grid_true"]
preds_grid = d["preds_grid"]
gif_degrees = d["gif_degrees"]
x_tr = d["x_tr"]
y_tr = d["y_tr"]
n_train = int(d["n_train"])

OUT = "C:/Users/kouta/project/forsubwork/src/content/blog/images/double-descent-experiment"

# fine range for the error-curve panel (0..80)
fine_mask = degrees <= 80
params_fine = params[fine_mask]
test_fine = acc_test[fine_mask]

# frame selection: dense near the interesting zone (0-42), sparser tail (42-80)
frame_degrees = list(range(0, 30, 3)) + list(range(30, 42, 2)) + list(range(42, 81, 8))
frame_idx = [np.where(gif_degrees == fd)[0][0] for fd in frame_degrees]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.6), dpi=70)

def draw(i):
    ax1.clear()
    ax2.clear()
    idx = frame_idx[i]
    dg = frame_degrees[i]
    p = dg + 1

    ax1.plot(x_grid, y_grid_true, color="gray", lw=1.0, ls="--", antialiased=False)
    ax1.plot(x_grid, preds_grid[idx], color="#2980b9", lw=1.6, antialiased=False)
    ax1.scatter(x_tr, y_tr, color="#c0392b", s=10, zorder=5)
    ax1.set_ylim(-3, 3)
    ax1.set_xlim(-1, 1)
    ax1.set_title(f"次数={dg} (p={p})", fontsize=8)
    ax1.tick_params(labelsize=6)

    ax2.plot(params_fine, test_fine, color="#c0392b", lw=1.3, antialiased=False)
    ax2.axvline(n_train, color="gray", ls="--", lw=0.7)
    cur_test = np.interp(p, params_fine, test_fine)
    ax2.scatter([p], [cur_test], color="#111111", s=22, zorder=6)
    ax2.set_yscale("log")
    ax2.set_xlabel("パラメータ数 p", fontsize=7)
    ax2.set_ylabel("テスト誤差", fontsize=7)
    ax2.set_title("テスト誤差 vs 容量", fontsize=8)
    ax2.tick_params(labelsize=6)

    fig.suptitle("容量(次数)を上げていくと何が起きるか", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return []

anim = animation.FuncAnimation(fig, draw, frames=len(frame_idx), interval=180)
writer = animation.PillowWriter(fps=5)
anim.save(f"{OUT}/sweep_raw.gif", writer=writer)
print("saved sweep_raw.gif, frames:", len(frame_idx))

# post-process: reduce palette / optimize for smaller file size
from PIL import Image
im = Image.open(f"{OUT}/sweep_raw.gif")
frames = []
durations = []
try:
    while True:
        frame = im.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=32)
        frames.append(frame)
        durations.append(im.info.get("duration", 180))
        im.seek(im.tell() + 1)
except EOFError:
    pass

frames[0].save(
    f"{OUT}/sweep.gif",
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=True,
)
print("saved optimized sweep.gif")
