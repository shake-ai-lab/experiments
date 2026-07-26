import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.spatial import cKDTree

D = "C:/tmp/vae-two-moons-experiment/"
X = np.load(D + "X.npy")
y = np.load(D + "y.npy")
mu_all = np.load(D + "mu_all.npy")
logvar_all = np.load(D + "logvar_all.npy")
X_gen = np.load(D + "X_gen.npy")

# 潜在空間でどれだけ2つのmoonが線形分離できるか
clf = LogisticRegression()
clf.fit(mu_all, y)
acc = clf.score(mu_all, y)
print(f"latent space linear separability (logistic regression acc): {acc:.4f}")

# 各潜在次元ごとのKL寄与 (posterior collapseチェック)
kl_per_dim = -0.5 * (1 + logvar_all - mu_all**2 - np.exp(logvar_all))
print(f"KL per dim (mean): z1={kl_per_dim[:,0].mean():.4f}  z2={kl_per_dim[:,1].mean():.4f}")
print(f"mu std per dim: z1={mu_all[:,0].std():.4f}  z2={mu_all[:,1].std():.4f}")

# 生成サンプルが本物データにどれだけ近いか (最近傍距離)
tree_real = cKDTree(X)
d_gen_to_real, _ = tree_real.query(X_gen, k=1)
print(f"generated->real nearest-neighbor dist: mean={d_gen_to_real.mean():.4f} median={np.median(d_gen_to_real):.4f}")

# 本物同士の最近傍距離 (ベースライン、ノイズスケール感覚)
d_real_self, _ = tree_real.query(X, k=2)  # k=1は自分自身なのでk=2の2列目
print(f"real->real nearest-neighbor dist (baseline): mean={d_real_self[:,1].mean():.4f}")
