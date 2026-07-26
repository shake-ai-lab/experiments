"""継ぎ目で何が起きているのか。
反転を重なりなしで実現するには、帯を「潰す」しかないはず。
展開図上での帯の幅を角度ごとに測り、反転位置でくびれているかを確かめる。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

d = np.load("bands.npz")
CASES = [(0, "ひねり0回 (ただの帯)", "#2e6ee4"), (1, "ひねり1回 (メビウスの輪)", "#e4572e")]
NB = 48

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
res = {}
for ht, cname, col in CASES:
    X, u, v = d[f"{ht}_X"], d[f"{ht}_u"], d[f"{ht}_v"]
    emb = Isomap(n_neighbors=12, n_components=2).fit_transform(X)
    emb = emb / np.abs(emb).max()  # スケールを揃える
    c = emb - emb.mean(0)
    theta = np.arctan2(c[:, 1], c[:, 0])
    radius = np.hypot(c[:, 0], c[:, 1])
    bins = np.linspace(-np.pi, np.pi, NB + 1)
    widths, agrees = [], []
    for b in range(NB):
        m = (theta >= bins[b]) & (theta < bins[b + 1])
        if m.sum() < 8:
            widths.append(np.nan)
            agrees.append(np.nan)
            continue
        widths.append(np.percentile(radius[m], 95) - np.percentile(radius[m], 5))
        side = np.sign(radius[m] - np.median(radius[m]))
        agrees.append(np.mean(side * np.sign(v[m])))
    widths, agrees = np.array(widths), np.array(agrees)
    centers = (bins[:-1] + bins[1:]) / 2
    res[cname] = (centers, widths, agrees)
    rel = widths / np.nanmedian(widths)
    print(f"{cname}: 幅の中央値比 最小 {np.nanmin(rel):.2f}, 最大 {np.nanmax(rel):.2f}")
    axes[0].plot(centers, rel, "-o", ms=4, color=col, label=cname)

axes[0].set_xlabel("展開図上での角度")
axes[0].set_ylabel("帯の幅 (中央値=1に正規化)")
axes[0].set_title("メビウスだけ、2箇所で帯がくびれている")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1.55)
cen, w, ag = res["ひねり1回 (メビウスの輪)"]
rel = w / np.nanmedian(w)
imin = int(np.nanargmin(rel))
axes[0].annotate(f"幅が{rel[imin]:.0%}まで潰れる\n(=ここで表裏が入れ替わる)",
                 xy=(cen[imin], rel[imin]), xytext=(cen[imin] + 0.5, 0.42),
                 fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.6))

# 幅と一致度を重ねて、くびれ位置と反転位置が一致することを見せる
ax2 = axes[1]
ax2.plot(cen, rel, "-o", ms=4, color="#e4572e", label="帯の幅 (正規化)")
ax2.plot(cen, ag, "-s", ms=4, color="#333", label="表裏の一致度")
ax2.axhline(0, color="#aaa", lw=1)
ax2.set_xlabel("展開図上での角度")
ax2.set_title("くびれる場所と、表裏が反転する場所は同じ")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
fig.suptitle("向き付け不可能な面を平面に押し込むと、どこかを潰すしかない", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig4_pinch.png", dpi=110, bbox_inches="tight")
print("saved fig4_pinch.png")
