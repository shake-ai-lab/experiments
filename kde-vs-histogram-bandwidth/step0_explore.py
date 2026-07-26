"""同じ1次元サンプルが、ヒストグラムのビン幅やKDEのバンド幅次第で
「単峰」にも「双峰」にも「ノイズだらけ」にも見えることを確かめる下調べ。
まず真の混合分布と、実際に引いた有限サンプルの様子を見る。"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)

# 真の分布: 2つの正規分布の等重み混合 (ギャップは「適度」を狙う)
MU1, SD1, MU2, SD2, W = -2.0, 1.0, 2.2, 1.0, 0.5
N = 220

def sample(n, seed):
    r = np.random.default_rng(seed)
    z = r.random(n) < W
    x = np.where(z, r.normal(MU1, SD1, n), r.normal(MU2, SD2, n))
    return x

def true_pdf(x):
    return W * stats.norm.pdf(x, MU1, SD1) + (1 - W) * stats.norm.pdf(x, MU2, SD2)

# 真の密度の谷の深さを確認
g = np.linspace(-6, 6, 2000)
p = true_pdf(g)
peak1 = p[g < 0].max()
valley = p[(g > MU1) & (g < MU2)].min()
peak2 = p[g > 0].max()
print(f"真の密度: 山1={peak1:.4f}, 谷={valley:.4f}, 山2={peak2:.4f}, 谷/山比={valley/min(peak1,peak2):.3f}")

x = sample(N, seed=42)
print(f"サンプル N={N}, range=[{x.min():.2f}, {x.max():.2f}], mean={x.mean():.2f}, std={x.std():.2f}")
np.save("x_sample.npy", x)

# ざっくり複数seedで試して「谷が浅くて有限サンプルで消えたり暴れたりしやすい」ものを選ぶ
for seed in range(10):
    xs = sample(N, seed)
    print(seed, xs.min().round(2), xs.max().round(2))
