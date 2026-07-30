"""「メビウスの帯 + 円板 = 実射影平面」を座標で実際に作る。

パラメータ円板を半径 RHO で2つに分ける:
  - 内側 |w| <= RHO      : ただの円板
  - 外側 RHO <= |w| <= 1 : 縁 |w|=1 の対蹠点が貼り合わさっているので、
                           実は「幅方向に開いたメビウスの帯」になっている
外側パーツをメビウスの帯の標準座標 (σ, t) で貼り直し、
  (a) 帯の中心線(t=0)が縁 |w|=1 を通って連続につながること
  (b) 帯が一周すると幅方向が反転して閉じること(=ひねり1回)
  (c) 中心線を一周すると法線が反転すること(=向き付け不可能)
  (d) 帯の境界(t=±1)と円板の境界が同じ円に乗ること(=貼り合わせの連続性)
をすべて数値で確認する。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from boy_common import boy, boy_normal

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

RHO = 0.65  # 円板パートと帯パートの境目の半径


def strip_to_disk(sigma, t):
    """メビウスの帯の標準座標 (σ∈[0,π), t∈[-1,1]) → パラメータ円板の複素数 w。
    t>=0 は角度 σ 側、t<0 は角度 σ+π 側の半径方向に伸びる。t=0 が帯の中心線(縁 |w|=1)。"""
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)
    r = 1 - (1 - RHO) * np.abs(t)
    ang = np.where(t >= 0, sigma, sigma + np.pi)
    return r * np.exp(1j * ang)


print(f"=== メビウスの帯 + 円板 = 射影平面 (境目の半径 RHO={RHO}) ===\n")

# ---- (a) 帯の中心線が縁を通って連続か: t=+ε と t=-ε の3D距離 ----
sig = np.linspace(0, np.pi, 400, endpoint=False)
print("(a) 帯の中心線(t=0, 縁|w|=1)をまたぐ連続性: ε→0 で距離が0に収束するか")
for eps in (1e-2, 1e-3, 1e-4):
    gap = np.linalg.norm(boy(strip_to_disk(sig, eps)) - boy(strip_to_disk(sig, -eps)), axis=1)
    print(f"    ε={eps:.0e}: 3D距離の最大 {gap.max():.2e}")

# ---- (b) 帯が一周して閉じるとき幅方向が反転しているか ----
t_arr = np.linspace(-1, 1, 201)
gap_close = np.linalg.norm(boy(strip_to_disk(np.pi - 1e-9, t_arr)) - boy(strip_to_disk(0.0, -t_arr)), axis=1)
gap_noflip = np.linalg.norm(boy(strip_to_disk(np.pi - 1e-9, t_arr)) - boy(strip_to_disk(0.0, t_arr)), axis=1)
print(f"\n(b) 帯を一周した継ぎ目: 幅方向を反転して合わせると 最大距離 {gap_close.max():.2e}")
print(f"    反転せずに合わせると 最大距離 {gap_noflip.max():.3f} (合わない = ひねりが入っている)")

# ---- (c) 中心線一周での法線の反転 ----
# 中心線 = 縁 |w|=1。法線は境界のわずかに内側で評価し、σ を 0→π まで連続に追跡する。
sig_line = np.linspace(0, np.pi, 1001)
w_line = (1 - 1e-4) * np.exp(1j * sig_line)  # 縁のすぐ内側
normals = boy_normal(w_line)
# 連続性の確認(隣り合う法線のなす角が小さいこと)
step_dots = np.sum(normals[:-1] * normals[1:], axis=1)
print(f"\n(c) 中心線に沿った法線の連続追跡: 隣接ステップの内積の最小 {step_dots.min():.6f} (1に近い=連続)")
d_endpoints = float(np.dot(normals[0], normals[-1]))
p_start, p_end = boy(w_line[0]), boy(w_line[-1])
print(f"    出発点と到着点の3D距離: {np.linalg.norm(p_start - p_end):.2e} (同じ点に戻る)")
print(f"    出発時と到着時の法線の内積: {d_endpoints:.6f} (-1 = 裏返って戻ってくる)")
# 対照: 貼り合わせを通らないループ(円板パート内の円周)では法線は反転しない
th_full = np.linspace(0, 2 * np.pi, 2001)
w_inner = 0.5 * np.exp(1j * th_full)
n_inner = boy_normal(w_inner)
print(f"    対照(貼り合わせを通らない円 |w|=0.5 の一周): 法線の内積 {np.dot(n_inner[0], n_inner[-1]):.6f} (+1 = 反転しない)")

# ---- (d) 帯の境界と円板の境界の貼り合わせ ----
# 円板側と帯側を互いに無関係な角度でサンプリングし、
# 「円板の縁の点が、帯の縁の点列(密)からどれだけ離れているか」を測る。
# 同じ閉曲線に乗っているなら、距離は帯側のサンプル間隔の半分以下に収まるはず。
print(f"\n(d) 帯の境界(t=±1)と円板の境界(|w|={RHO})の継ぎ目:")
from scipy.spatial import cKDTree

rng_d = np.random.default_rng(7)
disk_edge = boy(RHO * np.exp(1j * rng_d.uniform(0, 2 * np.pi, 800)))  # 円板側: ランダム角度
sig_dense = np.linspace(0, np.pi, 10000, endpoint=False)  # 帯側: 等間隔で密に
band_edge = np.r_[boy(strip_to_disk(sig_dense, np.ones_like(sig_dense))),
                  boy(strip_to_disk(sig_dense, -np.ones_like(sig_dense)))]
spacing = max(np.linalg.norm(np.diff(band_edge[:10000], axis=0), axis=1).max(),
              np.linalg.norm(np.diff(band_edge[10000:], axis=0), axis=1).max())
d_seam, _ = cKDTree(band_edge).query(disk_edge)
print(f"    円板の縁(ランダム800点)から帯の縁(密1万点)への距離: 最大 {d_seam.max():.2e}")
print(f"    帯側のサンプル間隔(最大 {spacing:.2e})以下に収まる → 同じ1本の閉曲線に乗っている")
# 対照: わずかに違う円 (|w|=RHO+0.05) だと距離は桁違いに大きい
disk_off = boy((RHO + 0.05) * np.exp(1j * rng_d.uniform(0, 2 * np.pi, 800)))
d_off, _ = cKDTree(band_edge).query(disk_off)
print(f"    対照(半径を0.05ずらした円): 距離の最大 {d_off.max():.3f} (桁違いに大きい)")

# ---- 図2: 分解の模式図(縦スタック: 上=パラメータ円板、下=3Dでの対応) ----
rng = np.random.default_rng(1)
n_pts = 30000
r_s = np.sqrt(rng.uniform(0, 1, n_pts))
a_s = rng.uniform(0, 2 * np.pi, n_pts)
W_s = r_s * np.exp(1j * a_s)
is_band = np.abs(W_s) >= RHO
X_s = boy(W_s)
C_DISK, C_BAND = "#e4572e", "#2e6ee4"

fig = plt.figure(figsize=(9.5, 13.5))
# 上段: パラメータ円板
ax1 = fig.add_subplot(2, 1, 1)
ax1.scatter(W_s.real[~is_band], W_s.imag[~is_band], s=2, c=C_DISK, alpha=0.4)
ax1.scatter(W_s.real[is_band], W_s.imag[is_band], s=2, c=C_BAND, alpha=0.4)
th_c = np.linspace(0, 2 * np.pi, 200)
ax1.plot(np.cos(th_c), np.sin(th_c), "k-", lw=2.5)
ax1.plot(RHO * np.cos(th_c), RHO * np.sin(th_c), "k--", lw=1.8)
# 対蹠点同一視の矢印
for ang, mk in [(0.55, "★"), (2.1, "▲")]:
    for s_ in (0, np.pi):
        ax1.annotate(mk, xy=(1.13 * np.cos(ang + s_), 1.13 * np.sin(ang + s_)),
                     ha="center", va="center", fontsize=17, color="#111")
ax1.text(0, 0, "円板パート\n(ただの円板)", ha="center", va="center", fontsize=16,
         color="white", fontweight="bold",
         bbox=dict(boxstyle="round", fc=C_DISK, ec="none", alpha=0.9))
ax1.text(0, -0.85, "帯パート", ha="center", va="center", fontsize=16, color="white",
         fontweight="bold", bbox=dict(boxstyle="round", fc=C_BAND, ec="none", alpha=0.9))
ax1.text(0, 1.32, "縁の対蹠点(★と★、▲と▲)は同じ点として貼り合わされている\n→ 帯パートは縁を通って反対側とつながり、メビウスの帯になる",
         ha="center", fontsize=14)
ax1.set_xlim(-1.75, 1.75)
ax1.set_ylim(-1.55, 1.62)
ax1.set_aspect("equal")
ax1.axis("off")
ax1.set_title("パラメータ円板 = 実射影平面の設計図", fontsize=17, pad=30)
# 下段: 3D での対応
ax2 = fig.add_subplot(2, 1, 2, projection="3d")
ax2.scatter(X_s[~is_band, 0], X_s[~is_band, 1], X_s[~is_band, 2], s=1.5, c=C_DISK, alpha=0.5)
ax2.scatter(X_s[is_band, 0], X_s[is_band, 1], X_s[is_band, 2], s=1.5, c=C_BAND, alpha=0.25)
seam = boy(RHO * np.exp(1j * th_c))
ax2.plot(seam[:, 0], seam[:, 1], seam[:, 2], "k-", lw=3)
ax2.view_init(elev=12, azim=-55)
ax2.set_box_aspect((1, 1, 0.9))
ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_zticks([])
ax2.set_title("同じ分解をボーイ曲面の上で見る\n(黒い線=貼り合わせの継ぎ目、1本の閉曲線)", fontsize=17)
fig.suptitle("実射影平面 = メビウスの帯(青) + 円板(赤) を1本の継ぎ目で貼ったもの", fontsize=17, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("fig2_decomposition.png", dpi=110, bbox_inches="tight")
print("\nsaved fig2_decomposition.png")
np.savez("decomposition.npz", W=W_s, X=X_s, is_band=is_band, rho=RHO)
