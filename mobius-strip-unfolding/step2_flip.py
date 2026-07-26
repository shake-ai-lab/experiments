"""向き付け不可能性の定量化。
埋め込み平面上で、輪の中心線から見て「外側/内側」のどちらに来るかを調べ、
それが元の帯の幅方向vの符号と一致するかを角度ごとに測る。
・ただの帯(向き付け可能)なら、一致度は全周で+1のまま
・メビウス(向き付け不可能)なら、どこかで必ず-1に反転する = 「継ぎ目」ができる
さらに継ぎ目で何が起きているか(重なりか裂け目か)も測る。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

d = np.load("bands.npz")
CASES = [(0, "ひねり0回 (ただの帯)", "#2e6ee4"), (1, "ひねり1回 (メビウスの輪)", "#e4572e")]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
summary = {}
for row, (ht, cname, col) in enumerate(CASES):
    X, u, v = d[f"{ht}_X"], d[f"{ht}_u"], d[f"{ht}_v"]
    emb = Isomap(n_neighbors=12, n_components=2).fit_transform(X)
    c = emb - emb.mean(0)
    theta = np.arctan2(c[:, 1], c[:, 0])
    radius = np.hypot(c[:, 0], c[:, 1])

    # 角度ビンごとに、中心線(そのビンの半径中央値)より外か内かを判定
    NB = 48
    bins = np.linspace(-np.pi, np.pi, NB + 1)
    agree = np.full(NB, np.nan)
    for b in range(NB):
        m = (theta >= bins[b]) & (theta < bins[b + 1])
        if m.sum() < 8:
            continue
        side = np.sign(radius[m] - np.median(radius[m]))
        agree[b] = np.mean(side * np.sign(v[m]))
    valid = ~np.isnan(agree)
    # 反転の有無: 符号が変わる境界の数
    sgn = np.sign(agree[valid])
    flips = int((sgn[:-1] * sgn[1:] < 0).sum() + (sgn[-1] * sgn[0] < 0))
    summary[cname] = dict(flips=flips, mean_abs=float(np.nanmean(np.abs(agree))))
    print(f"{cname}: 符号反転の回数 {flips}, |一致度|の平均 {summary[cname]['mean_abs']:.3f}")

    ax = axes[row, 0]
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=v, cmap="RdBu", s=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Isomapの展開: {cname}", fontsize=13)
    ax2 = axes[row, 1]
    centers = (bins[:-1] + bins[1:]) / 2
    ax2.plot(centers[valid], agree[valid], "-o", ms=4, color=col)
    ax2.axhline(0, color="#888", lw=1)
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_xlabel("展開図上での角度")
    ax2.set_ylabel("表裏の一致度")
    ax2.set_title(f"一致度が反転した回数: {flips}回", fontsize=13)
    ax2.grid(alpha=0.3)
    if flips > 0:
        # 反転点に注釈
        idx = np.flatnonzero(sgn[:-1] * sgn[1:] < 0)
        if len(idx):
            xf = centers[valid][idx[0]]
            ax2.annotate("ここで表裏が\n入れ替わる", xy=(xf, 0), xytext=(xf - 2.2, 0.55),
                         fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.6))
    else:
        ax2.annotate("全周で一致 (+1)\n=表裏が保たれている", xy=(0, agree[valid].mean()),
                     xytext=(-2.8, -0.6), fontsize=11,
                     arrowprops=dict(arrowstyle="->", lw=1.6))
fig.suptitle("平面に広げると、メビウスの輪だけ「表裏が入れ替わる継ぎ目」ができる", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig3_flip.png", dpi=110, bbox_inches="tight")
print("saved fig3_flip.png")

# ---- 継ぎ目での「重なり」を測る: 埋め込みで近いのに元では遠いペア ----
from scipy.spatial import cKDTree

print("\n=== 偽の近傍(埋め込みで近いのに元の帯では遠い)の割合 ===")
false_rates = {}
for ht, cname, _ in CASES:
    X, u, v = d[f"{ht}_X"], d[f"{ht}_u"], d[f"{ht}_v"]
    emb = Isomap(n_neighbors=12, n_components=2).fit_transform(X)
    emb_s = (emb - emb.mean(0)) / emb.std(0)
    tree = cKDTree(emb_s)
    _, nn = tree.query(emb_s, k=9)
    du = np.abs(u[nn[:, 1:]] - u[:, None])
    du = np.minimum(du, 2 * np.pi - du)
    rate = float((du > 0.35).mean())  # 帯に沿って十分離れている近傍の割合
    false_rates[cname] = rate
    print(f"  {cname}: {rate*100:.1f}%")
np.savez("flip_summary.npz",
         flips=np.array([summary[c]["flips"] for _, c, _ in CASES]),
         false_rates=np.array([false_rates[c] for _, c, _ in CASES]))
