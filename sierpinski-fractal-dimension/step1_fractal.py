"""シェルピンスキーの三角形は何次元か。
(1) カオスゲーム(ランダム中点移動)で三角形を生成
(2) box-counting法で4つの図形(直線・塗りつぶし正方形・コッホ曲線・シェルピンスキー)の
    フラクタル次元を実測し、理論値と比較する"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

# ---- カオスゲーム ----
verts = np.array([[0.05, 0.05], [0.95, 0.05], [0.5, 0.05 + 0.9 * np.sqrt(3) / 2]])
N = 2_000_000
pts = np.empty((N, 2))
p = np.array([0.3, 0.3])
choices = rng.integers(0, 3, N)
for i in range(N):
    p = (p + verts[choices[i]]) / 2
    pts[i] = p
pts = pts[100:]  # 初期の過渡を捨てる
np.save("sierpinski_pts.npy", pts[::4].astype(np.float32))
print(f"chaos game: {len(pts):,} points")


# ---- 比較用図形 ----
def koch_curve(levels=6):
    seg = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    for _ in range(levels):
        new = []
        for a, b in zip(seg[:-1], seg[1:]):
            d = b - a
            p1 = a + d / 3
            p2 = a + d * 2 / 3
            rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
            peak = p1 + rot @ (d / 3)
            new += [a, p1, peak, p2]
        new.append(seg[-1])
        seg = new
    seg = np.array(seg)
    # 線分を点で密にサンプル
    dense = []
    for a, b in zip(seg[:-1], seg[1:]):
        for t in np.linspace(0, 1, 8, endpoint=False):
            dense.append(a + t * (b - a))
    d = np.array(dense)
    d[:, 1] += 0.3  # 見やすい位置へ
    return d


line = np.c_[np.linspace(0.05, 0.95, 200_000), np.full(200_000, 0.5)]
square = rng.uniform(0.05, 0.95, (2_000_000, 2))
koch = koch_curve()
shapes = {
    "直線 (理論1.0)": (line, 1.0),
    "塗りつぶし正方形 (理論2.0)": (square, 2.0),
    "コッホ曲線 (理論1.262)": (koch, np.log(4) / np.log(3)),
    "シェルピンスキー (理論1.585)": (pts, np.log(3) / np.log(2)),
}


def box_count(P, k):
    """2^k x 2^k のグリッドで占有ボックス数を数える"""
    H, *_ = np.histogram2d(P[:, 0], P[:, 1], bins=2**k, range=[[0, 1], [0, 1]])
    return int((H > 0).sum())


ks = np.arange(2, 9)  # 4x4 .. 256x256
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
colors = ["#999", "#2e6ee4", "#3aa655", "#e4572e"]
results = {}
for (name, (P, theo)), col in zip(shapes.items(), colors):
    counts = [box_count(P, k) for k in ks]
    slope, _ = np.polyfit(ks * np.log(2), np.log(counts), 1)
    results[name] = (slope, theo)
    axes[0].plot(ks * np.log(2), np.log(counts), "o-", color=col, ms=4,
                 label=f"{name} → 実測 {slope:.3f}")
    print(f"{name}: 実測 {slope:.4f} (理論 {theo:.4f}, 誤差 {abs(slope-theo):.4f})")
axes[0].set_xlabel("log(1/ボックスサイズ)")
axes[0].set_ylabel("log(占有ボックス数)")
axes[0].set_title("box-counting: この直線の傾きが「次元」")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

names = list(results.keys())
x = np.arange(len(names))
axes[1].bar(x - 0.18, [results[n][0] for n in names], 0.36, label="実測", color="#e4572e")
axes[1].bar(x + 0.18, [results[n][1] for n in names], 0.36, label="理論", color="#bbb")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["直線", "正方形", "コッホ", "シェルピンスキー"], fontsize=9)
axes[1].set_ylabel("次元")
axes[1].set_title("実測次元 vs 理論値")
axes[1].legend()
axes[1].grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig2_boxcount.png", dpi=110, bbox_inches="tight")
print("saved fig2_boxcount.png")

# ---- fig1: 4図形 ----
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (name, (P, theo)), col in zip(axes, shapes.items(), colors):
    sub = P[:: max(1, len(P) // 60_000)]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.2, color=col)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=10)
fig.tight_layout()
fig.savefig("fig1_shapes.png", dpi=110, bbox_inches="tight")
print("saved fig1_shapes.png")
