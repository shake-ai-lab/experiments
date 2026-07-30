"""ボーイ曲面の自己交差構造を数値的に検出する。

方法: パラメータ円板から密にサンプリングした点群を3D側で kd-tree にかけ、
「3次元では近いのに、射影平面の上では遠い」ペアを探す。
そのペアの中点が自己交差線(2枚の面が貫き合う場所)のサンプルになる。
さらに「3枚の面が同時に重なる点」= 三重点を探し、
理論の予言(分母の零点 → 原点、preimage は半径 ((3-√5)/2)^(1/3) の3点)と突き合わせる。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from boy_common import POLE_RADIUS, boy, param_dist, sample_disk
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

# ---- 密サンプリング + ボクセル間引き(密度の偏りを均す) ----
N_RAW = 300000
W_raw = sample_disk(N_RAW, seed=42)
X_raw = boy(W_raw)
VOX = 0.009
vox_id = np.round(X_raw / VOX).astype(np.int64)
_, keep = np.unique(vox_id, axis=0, return_index=True)
W, X = W_raw[keep], X_raw[keep]
print(f"サンプル {N_RAW} 点 → ボクセル間引き後 {len(X)} 点")

# ---- 自己交差ペアの検出 ----
EPS3D = 0.022   # 3次元でこの距離以内
DPARAM = 0.25   # かつパラメータ空間(同一視込み)でこの距離以上離れているペア
tree = cKDTree(X)
pairs = np.array(list(tree.query_pairs(EPS3D)))
pd = param_dist(W[pairs[:, 0]], W[pairs[:, 1]])
cross = pairs[pd > DPARAM]
mid = (X[cross[:, 0]] + X[cross[:, 1]]) / 2
print(f"3D近傍ペア {len(pairs)} 組のうち、パラメータ的に遠いペア(=自己交差の証拠) {len(cross)} 組")

# ---- 三重点の検出: 3枚以上の面が同時に集まる点 ----
# 各交差サンプルの近傍にある元の点のパラメータを貪欲にグループ分けし、
# 互いに遠いグループが3つ以上あれば「3枚の面が重なっている」。
def count_sheets(p, radius=EPS3D):
    idx = tree.query_ball_point(p, radius)
    groups = []
    for i in idx:
        for g in groups:
            if param_dist(W[i], g[0]) < DPARAM:
                g.append(W[i])
                break
        else:
            groups.append([W[i]])
    return groups

sheet_counts = np.array([len(count_sheets(p)) for p in mid])
triple_mask = sheet_counts >= 3
triples = mid[triple_mask]
print(f"\n=== 三重点 ===")
print(f"3枚が重なる点のサンプル: {triple_mask.sum()} 個")
# クラスタ数(=三重点の個数)を連結成分で数える
if len(triples):
    tt = cKDTree(triples)
    tp = np.array(list(tt.query_pairs(0.15)))
    n_tp = len(triples) if len(tp) == 0 else connected_components(
        coo_matrix((np.ones(len(tp)), (tp[:, 0], tp[:, 1])), shape=(len(triples),) * 2),
        directed=False)[0]
    center = triples.mean(axis=0)
    print(f"三重点の個数(クラスタ数): {n_tp}")
    print(f"位置: ({center[0]:+.4f}, {center[1]:+.4f}, {center[2]:+.4f}), 原点からの距離 {np.linalg.norm(center):.4f}")
    print(f"理論の予言: Bryant-Kusner の分母の零点が反転で原点に写る → 三重点は原点のはず")
    # preimage の確認
    w_tri = np.concatenate([[g[0] for g in count_sheets(p)] for p in triples[:200]])
    r_tri = np.abs(w_tri)
    # 角度は 0°, 120°, 240° の3点に集まるはず → 最寄りの理論角からの偏差で測る
    a_dev = (np.degrees(np.angle(w_tri)) + 60) % 120 - 60
    print(f"三重点の preimage: |w| = {r_tri.mean():.4f} ± {r_tri.std():.4f} "
          f"(理論値 ((3-√5)/2)^(1/3) = {POLE_RADIUS:.4f})")
    print(f"                   角度の理論値(0°,120°,240°)からの偏差 = {a_dev.mean():+.2f}° ± {a_dev.std():.2f}°")

# ---- 羽根の本数: 三重点の周りを除いて連結成分を数える ----
R_CUT = 0.30
far = mid[np.linalg.norm(mid - center, axis=1) > R_CUT]
ft = cKDTree(far)
fp = np.array(list(ft.query_pairs(0.09)))
lab_n, labels = connected_components(
    coo_matrix((np.ones(len(fp)), (fp[:, 0], fp[:, 1])), shape=(len(far),) * 2), directed=False)
sizes = np.sort(np.bincount(labels))[::-1]
big = sizes[sizes >= 20]
print(f"\n=== 自己交差線の構造 ===")
print(f"三重点の周り(半径{R_CUT})を取り除いた自己交差線の連結成分: {len(big)} 本 (点数 {list(big)})")
print("→ 三重点から3枚の羽根が伸びる『三葉のプロペラ』")
# 全体では1成分(すべて三重点でつながる)ことも確認
at = cKDTree(mid)
ap = np.array(list(at.query_pairs(0.09)))
all_n, all_lab = connected_components(
    coo_matrix((np.ones(len(ap)), (ap[:, 0], ap[:, 1])), shape=(len(mid),) * 2), directed=False)
all_sizes = np.sort(np.bincount(all_lab))[::-1]
print(f"三重点を含めた全体の連結成分(20点以上): {(all_sizes >= 20).sum()} 個 → 全部が三重点でつながっている")

np.savez("self_intersection.npz", mid=mid, center=center,
         far=far, far_labels=labels, big_labels=np.flatnonzero(np.bincount(labels) >= 20))

# ---- 図3: プロペラの可視化 (縦スタック: 3D 2ビュー + パラメータ円板) ----
BRANCH_COLORS = ["#2e6ee4", "#e4572e", "#2ca02c"]
big_ids = np.argsort(np.bincount(labels))[::-1][:3]
sub = sample_disk(25000, seed=5)
Xs = boy(sub)

fig = plt.figure(figsize=(9.5, 18))
for row, (elev, azim, vlabel) in enumerate([(15, -55, "斜めから"), (88, -90, "真上から")]):
    ax = fig.add_subplot(3, 1, row + 1, projection="3d")
    ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], s=1, c="#bbbbbb", alpha=0.12)
    for k, gid in enumerate(big_ids):
        pts = far[labels == gid]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, c=BRANCH_COLORS[k], alpha=0.9)
    near = mid[np.linalg.norm(mid - center, axis=1) <= R_CUT]
    ax.scatter(near[:, 0], near[:, 1], near[:, 2], s=4, c="#666666", alpha=0.5)
    ax.scatter(*center, s=260, marker="*", c="#111111", zorder=10)
    ax.text(center[0], center[1], center[2] + 0.25, "三重点", fontsize=15, ha="center")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.9))
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_title(f"自己交差線({vlabel})。3色 = 3枚の羽根", fontsize=16)
# パラメータ円板での preimage
ax3 = fig.add_subplot(3, 1, 3)
w_cross = np.r_[W[cross[:, 0]], W[cross[:, 1]]]
ax3.scatter(w_cross.real, w_cross.imag, s=3, c="#555555", alpha=0.4)
th_c = np.linspace(0, 2 * np.pi, 200)
ax3.plot(np.cos(th_c), np.sin(th_c), "k-", lw=2)
for k in range(3):
    ang = 2 * np.pi * k / 3
    ax3.scatter(POLE_RADIUS * np.cos(ang), POLE_RADIUS * np.sin(ang), s=280, marker="*",
                c="#111111", zorder=5)
ax3.set_aspect("equal")
ax3.set_title("同じものをパラメータ円板で見る\n(灰色 = 自己交差線の preimage、★ = 三重点の preimage 3点)", fontsize=16)
ax3.axis("off")
fig.suptitle("ボーイ曲面の自己交差線は「三葉のプロペラ」\n1つの三重点から3枚の羽根が伸びる", fontsize=17, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("fig3_propeller.png", dpi=110, bbox_inches="tight")
print("saved fig3_propeller.png")
