"""穴の反対側同士の測地線が、穴をどう迂回するかを可視化・定量化する。
kNNグラフ上の最短路(=Isomapが使う測地線近似)を、展開座標(s,y)上で描く。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph

plt.rcParams["font.family"] = "Meiryo"

d = np.load("data.npz")
s_c, y_c = float(d["s_c"]), float(d["y_c"])

conds = ["穴なし", "大きい穴"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6))
inflations = {}
for ax, cond in zip(axes, conds):
    X, s, y = d[f"{cond}_X"], d[f"{cond}_s"], d[f"{cond}_y"]
    G = kneighbors_graph(X, n_neighbors=10, mode="distance")
    # 穴の左右で、幅方向の真ん中に近い2点を選ぶ
    src = np.argmin((s - (s_c - 14)) ** 2 + (y - y_c) ** 2)
    dst = np.argmin((s - (s_c + 14)) ** 2 + (y - y_c) ** 2)
    dist, pred = shortest_path(G, indices=src, return_predecessors=True, directed=False)
    path = [dst]
    while path[-1] != src:
        path.append(pred[path[-1]])
    path = path[::-1]
    geo = dist[dst]
    straight = np.hypot(s[dst] - s[src], y[dst] - y[src])
    inflations[cond] = geo / straight
    print(f"{cond}: 測地線 {geo:.1f} vs 直線 {straight:.1f} -> {geo/straight:.2f}倍")

    ax.scatter(s, y, s=4, color="#ccc")
    ax.plot(s[path], y[path], color="#e4572e", lw=2, zorder=3, label="グラフ上の最短路")
    ax.plot([s[src], s[dst]], [y[src], y[dst]], "--", color="#2e6ee4", lw=1.5, label="シート上の直線距離")
    ax.scatter([s[src], s[dst]], [y[src], y[dst]], s=50, color="#111", zorder=4)
    ax.set_title(f"{cond}: 測地線 {geo:.1f} / 直線 {straight:.1f} = {geo/straight:.2f}倍", fontsize=11)
    ax.set_xlabel("シート上の位置 (弧長 s)")
    ax.set_ylabel("幅 y")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(s_c - 35, s_c + 35)
fig.suptitle("Isomapの「距離」= kNNグラフの最短路。穴があると迂回した分だけ距離が水増しされる", fontsize=12)
fig.tight_layout()
fig.savefig("fig3_geodesic.png", dpi=110, bbox_inches="tight")
print("saved fig3_geodesic.png")

# 穴をまたぐ多数ペアで水増し率の分布を測る
X, s, y = d["大きい穴_X"], d["大きい穴_s"], d["大きい穴_y"]
G = kneighbors_graph(X, n_neighbors=10, mode="distance")
left = np.flatnonzero((s < s_c - 10) & (s > s_c - 25))
right = np.flatnonzero((s > s_c + 10) & (s < s_c + 25))
li = np.random.default_rng(1).choice(left, 60, replace=False)
dist = shortest_path(G, indices=li, directed=False)
ratios = []
for k, i in enumerate(li):
    for j in np.random.default_rng(k).choice(right, 20, replace=False):
        straight = np.hypot(s[j] - s[i], y[j] - y[i])
        if straight > 1:
            ratios.append(dist[k, j] / straight)
ratios = np.array(ratios)
print(f"穴をまたぐ{len(ratios)}ペアの水増し率: 中央値 {np.median(ratios):.2f}倍, 最大 {ratios.max():.2f}倍")
np.save("inflation_ratios.npy", ratios)
