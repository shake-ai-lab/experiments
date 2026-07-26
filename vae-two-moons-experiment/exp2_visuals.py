import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

plt.rcParams["font.family"] = "Meiryo"

D = "C:/tmp/vae-two-moons-experiment/"
OUT = "C:/tmp/vae-two-moons-experiment/"

X = np.load(D + "X.npy")
y = np.load(D + "y.npy")
X_gen = np.load(D + "X_gen.npy")
X_recon = np.load(D + "X_recon.npy")
mu_all = np.load(D + "mu_all.npy")
logvar_all = np.load(D + "logvar_all.npy")
recon_hist = np.load(D + "recon_hist.npy")
kl_hist = np.load(D + "kl_hist.npy")
gen_snaps = np.load(D + "gen_snapshots.npz")

LIM = 3.2

# ---------------- 0. 導入用: データそのものの可視化 (moonラベルで色分け) ----------------
# 潜在空間の図(4_latent.png, coolwarm)と同じ色対応: moon 0=青, moon 1=赤
fig, ax = plt.subplots(figsize=(6, 5.0))
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=5, alpha=0.5, c="#3b4cc0", label="moon 0 (上の三日月)")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=5, alpha=0.5, c="#b40426", label="moon 1 (下の三日月)")
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
ax.set_aspect("equal")
ax.set_title("学習に使うtwo moonsデータ (4000点)")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT + "0_data.png", dpi=130)
plt.close()
print("saved 0_data.png")

# ---------------- 1. 学習曲線: 再構成損失とKL項 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(recon_hist, color="#2b6cb0", lw=0.7)
axes[0].set_title("再構成損失 (0.5*Σ(x-x̂)²)")
axes[0].set_xlabel("イテレーション")
axes[0].set_ylabel("損失")
axes[1].plot(kl_hist, color="#c53030", lw=0.7)
axes[1].set_title("KLダイバージェンス項")
axes[1].set_xlabel("イテレーション")
axes[1].set_ylabel("KL (nats/サンプル)")
plt.tight_layout()
plt.savefig(OUT + "1_loss_curves.png", dpi=130)
plt.close()
print("saved 1_loss_curves.png")

# ---------------- 2. 実データ vs 再構成 ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
axes[0].scatter(X[:, 0], X[:, 1], s=4, alpha=0.5, c="#555555")
axes[0].set_title("本物のtwo moons")
axes[0].set_xlim(-LIM, LIM); axes[0].set_ylim(-LIM, LIM)
axes[0].set_aspect("equal")
axes[1].scatter(X_recon[:, 0], X_recon[:, 1], s=4, alpha=0.5, c="#38a169")
axes[1].set_title("VAEの再構成 (z=μ)")
axes[1].set_xlim(-LIM, LIM); axes[1].set_ylim(-LIM, LIM)
axes[1].set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT + "2_recon.png", dpi=130)
plt.close()
print("saved 2_recon.png")

# ---------------- 3. 本物 vs 事前分布からの生成 ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
axes[0].scatter(X[:, 0], X[:, 1], s=4, alpha=0.5, c="#555555")
axes[0].set_title("本物のtwo moons")
axes[0].set_xlim(-LIM, LIM); axes[0].set_ylim(-LIM, LIM)
axes[0].set_aspect("equal")
axes[1].scatter(X_gen[:, 0], X_gen[:, 1], s=4, alpha=0.5, c="#805ad5")
axes[1].set_title("VAE生成 (z~N(0,I)からデコード)")
axes[1].set_xlim(-LIM, LIM); axes[1].set_ylim(-LIM, LIM)
axes[1].set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT + "3_generated.png", dpi=130)
plt.close()
print("saved 3_generated.png")

# ---------------- 4. 潜在空間の散布図 (moonラベルで色分け) ----------------
fig, ax = plt.subplots(figsize=(6, 5.5))
sc = ax.scatter(mu_all[:, 0], mu_all[:, 1], c=y, cmap="coolwarm", s=5, alpha=0.6)
ax.set_title("潜在空間 μ の散布図 (色=どちらの三日月か)")
ax.set_xlabel("z1"); ax.set_ylabel("z2")
plt.colorbar(sc, ax=ax, label="moon label")
plt.tight_layout()
plt.savefig(OUT + "4_latent.png", dpi=130)
plt.close()
print("saved 4_latent.png")

# ---------------- 5. GIF: 学習中に生成サンプルがどう変わるか ----------------
all_keys = sorted(gen_snaps.files, key=lambda s: int(s))
keys = [k for i, k in enumerate(all_keys) if i % 2 == 0 or k == all_keys[-1]]  # 間引き
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=70)

def update(i):
    ax.clear()
    it = keys[i]
    pts = gen_snaps[it]
    ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.25, c="#aaaaaa", label="本物")
    ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.6, c="#805ad5", label="生成 (z~N(0,I))")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_title(f"イテレーション {it}")
    ax.legend(loc="upper right", fontsize=7)

anim_writer = PillowWriter(fps=1.5)
from matplotlib.animation import FuncAnimation
anim = FuncAnimation(fig, update, frames=len(keys))
anim.save(OUT + "gen_evolution.gif", writer=anim_writer)
plt.close()
print("saved gen_evolution.gif")

import os
from PIL import Image
print("gif size KB (raw):", os.path.getsize(OUT + "gen_evolution.gif") / 1024)

# パレット量子化で軽量化
im = Image.open(OUT + "gen_evolution.gif")
frames = []
durations = []
try:
    while True:
        frames.append(im.convert("RGB").quantize(colors=64, method=Image.MEDIANCUT))
        durations.append(im.info.get("duration", 500))
        im.seek(im.tell() + 1)
except EOFError:
    pass
frames[0].save(
    OUT + "gen_evolution.gif",
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=True,
)
print("gif size KB (quantized):", os.path.getsize(OUT + "gen_evolution.gif") / 1024)
