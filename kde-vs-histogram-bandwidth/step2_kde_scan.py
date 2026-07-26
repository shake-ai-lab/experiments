"""KDEのbw_method(factor)を変えたときに、密度推定の山の数がどう変わるか走査する。
scipy.stats.gaussian_kdeのbw_methodはスケール因子(factor)で、
実際のガウスカーネルの帯域幅(標準偏差)は factor * サンプルの標準偏差 になる。"""
import numpy as np
from scipy import stats

x = np.load("x_sample.npy")
g = np.linspace(x.min() - 2, x.max() + 2, 2000)

def count_peaks(y, min_prom=0.002):
    peaks = 0
    for i in range(1, len(y) - 1):
        if y[i] > y[i-1] and y[i] >= y[i+1]:
            # 山の高さ(左右の谷との差)がmin_prom以上のものだけ数える
            peaks += 1
    return peaks

print("factor, bandwidth(=factor*std), #peaks(粗)")
for factor in [0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]:
    kde = stats.gaussian_kde(x, bw_method=factor)
    y = kde(g)
    print(f"{factor:.2f}  bw={factor*x.std():.3f}  peaks={count_peaks(y)}")

print()
print("scott factor:", stats.gaussian_kde(x, bw_method='scott').factor)
print("silverman factor:", stats.gaussian_kde(x, bw_method='silverman').factor)
