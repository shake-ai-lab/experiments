"""射影と次元削減を同じ土俵(stress-1 / 距離相関)で比較する。
- 4D→3D の平行投影・透視投影の stress を測る
  (「見た目が良い」透視投影の方が距離構造の保存では悪い、を確認する)
- 4D→2D のランダム直交射影200本の stress 分布を出し、
  PCA(固有値が完全縮退)がその分布のどこに落ちるかを見る
"""
import itertools

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

V = np.array(list(itertools.product([-1, 1], repeat=4)), dtype=float)
D4 = pdist(V)


def stress1(D_hi, D_lo):
    b = np.dot(D_hi, D_lo) / np.dot(D_lo, D_lo)
    return np.sqrt(np.sum((D_hi - b * D_lo) ** 2) / np.sum(D_hi ** 2))


print("=== 4D→3D 射影の距離保存 ===")
P_par = V[:, :3]  # 平行投影
s = 3.0 / (3.0 - V[:, 3])
P_per = V[:, :3] * s[:, None]  # 透視投影(c=3)
for name, P in [("平行投影", P_par), ("透視投影(c=3)", P_per)]:
    Dl = pdist(P)
    print(f"  {name}: stress-1 {stress1(D4, Dl):.4f} / Spearman {spearmanr(D4, Dl)[0]:.4f}"
          f" / 完全に重なる頂点ペア {int((Dl < 1e-9).sum())}")

print("\n=== 4D→2D ランダム直交射影 200本 vs PCA ===")
rng = np.random.default_rng(0)
ss = np.array([stress1(D4, pdist(V @ np.linalg.qr(rng.normal(size=(4, 2)))[0]))
               for _ in range(200)])
print(f"  ランダム射影の stress-1: 最小 {ss.min():.4f} / 平均 {ss.mean():.4f} / 最大 {ss.max():.4f}")

from sklearn.decomposition import PCA

e = PCA(2, random_state=0).fit_transform(V)
sp = stress1(D4, pdist(e))
print(f"  PCA(2) の stress-1: {sp:.4f} → ランダム射影200本のうち "
      f"{int((ss < sp).sum())}本がPCAより良い")
uniq = np.unique(np.round(e, 6), axis=0)
print(f"  PCAの2D埋め込みで異なる位置の数: {len(uniq)} (16頂点が重なって{len(uniq)}点に見える)")
