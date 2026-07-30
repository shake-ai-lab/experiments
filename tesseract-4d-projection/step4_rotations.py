"""4次元の回転は6種類(xy, yz, zx, wx, wy, wz の各平面)。
このうち w を含む3種だけが、3次元への射影後に「剛体の回転」に見えず
「伸び縮み」に見えることを定量化する。
- 各平面で回転角を0→360°まで振り、平行投影後の32辺の長さを記録
- 指標: 各辺の長さの変動幅(max-min)の全辺平均・最大、全辺長合計の変動
"""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 15, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12})


def hypercube(d):
    V = np.array(list(itertools.product([-1, 1], repeat=d)), dtype=float)
    E = [(i, j) for i in range(len(V)) for j in range(i + 1, len(V))
         if np.sum(V[i] != V[j]) == 1]
    return V, E


def rot4(a, b, th):
    R = np.eye(4)
    R[a, a] = R[b, b] = np.cos(th)
    R[a, b] = -np.sin(th)
    R[b, a] = np.sin(th)
    return R


V4, E4 = hypercube(4)
AX = {"x": 0, "y": 1, "z": 2, "w": 3}
PLANES = ["xy", "yz", "zx", "wx", "wy", "wz"]
thetas = np.linspace(0, 2 * np.pi, 361)

print("=== 平面ごとの、平行投影後の辺長の変動 ===")
print(f"{'平面':>4s} {'辺長変動幅(全辺平均)':>16s} {'同(最大の辺)':>12s} {'全長合計 min-max':>20s}")
curves = {}
for pl in PLANES:
    a, b = AX[pl[0]], AX[pl[1]]
    lens = np.empty((len(thetas), len(E4)))
    for t, th in enumerate(thetas):
        P = (V4 @ rot4(a, b, th).T)[:, :3]  # 平行投影
        lens[t] = [np.linalg.norm(P[i] - P[j]) for i, j in E4]
    per_edge_range = lens.max(0) - lens.min(0)
    total = lens.sum(1)
    curves[pl] = total
    print(f"{pl:>4s} {per_edge_range.mean():16.4f} {per_edge_range.max():12.4f} "
        f"   {total.min():8.3f} - {total.max():8.3f}")

# ---- 図4: 縦スタック2段 ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 10.5), sharex=True)
deg = np.degrees(thetas)
for pl, col in zip(["xy", "yz", "zx"], ["#2e6ee4", "#3aa76d", "#7a5cd6"]):
    ax1.plot(deg, curves[pl], lw=2.4, color=col, label=f"{pl}平面")
ax1.set_ylabel("投影後の全辺長の合計")
ax1.set_title("wを含まない回転 (xy, yz, zx):\n投影後も剛体の回転。全辺の長さが1本も変わらない", fontsize=14)
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(44, 58)

for pl, col in zip(["wx", "wy", "wz"], ["#e4572e", "#e49b2e", "#c23b6e"]):
    ax2.plot(deg, curves[pl], lw=2.4, color=col, label=f"{pl}平面")
ax2.set_xlabel("回転角 (度)")
ax2.set_ylabel("投影後の全辺長の合計")
ax2.set_title("wを含む回転 (wx, wy, wz):\n同じ「回転」なのに、3次元では伸び縮みに見える", fontsize=14)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(44, 58)
fig.tight_layout()
fig.savefig("fig4_rotations.png", dpi=110, bbox_inches="tight")
print("\nsaved fig4_rotations.png")

# 参考: wx回転45°時点の辺長分布(記事の言及用)
P45 = (V4 @ rot4(3, 0, np.pi / 4).T)[:, :3]
L45 = np.array([np.linalg.norm(P45[i] - P45[j]) for i, j in E4])
uniq, cnt = np.unique(np.round(L45, 4), return_counts=True)
print("\nwx回転45°での投影後辺長:")
for u, c in zip(uniq, cnt):
    print(f"  {u:.4f} x {c}本")
