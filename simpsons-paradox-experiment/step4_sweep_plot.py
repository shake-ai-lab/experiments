import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

d = np.load("sweep_data.npz")
deltas, r_pools, r_as, r_bs = d["deltas"], d["r_pools"], d["r_as"], d["r_bs"]

fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

ax.axhline(0, color="gray", lw=1, linestyle=":")
ax.plot(deltas, r_pools, color="#111827", lw=2.5, marker="o", ms=4, label="全体をプールした相関 r_pool")
ax.plot(deltas, r_as, color="#1e3a8a", lw=2, linestyle="--", label="上級者グループ内の相関 r_A")
ax.plot(deltas, r_bs, color="#c2410c", lw=2, linestyle="--", label="初級者グループ内の相関 r_B")

# zero crossing
delta_cross = 39.80
ax.axvline(delta_cross, color="#dc2626", lw=1.5, linestyle="-.")
ax.annotate(f"r_pool=0\nΔ≈{delta_cross:.1f}点", xy=(delta_cross, 0), xytext=(delta_cross + 4, 0.35),
            fontsize=10, color="#dc2626",
            arrowprops=dict(arrowstyle="->", color="#dc2626"))

ax.set_xlabel("上級者グループの基礎スコア優位 Δ (点)")
ax.set_ylabel("Pearsonの相関係数 r")
ax.set_title("グループ間の基礎スコア差Δを動かすと、プールした相関の符号が反転する")
ax.legend(loc="upper right", fontsize=9, framealpha=1.0, facecolor="white", edgecolor="#888888")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("2.png", dpi=140)
print("saved 2.png")
