"""d と相関係数 r の換算（等人数の2群）を、シミュレーションの点双列相関と突き合わせる。"""
import numpy as np

rng = np.random.default_rng(3)
N = 1_000_000
for d in (0.2, 0.5, 0.8):
    a = rng.normal(d, 1, N)
    b = rng.normal(0, 1, N)
    x = np.concatenate([a, b])
    g = np.concatenate([np.ones(N), np.zeros(N)])
    r_sim = np.corrcoef(x, g)[0, 1]
    r_th = d / np.sqrt(d * d + 4)
    print(f"d={d}: r 式 {r_th:.4f} 実測 {r_sim:.4f}  説明される分散 r^2 = {r_th**2*100:.2f}%")
