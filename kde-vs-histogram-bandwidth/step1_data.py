"""実験全体の土台: 「適度なギャップ」を持つ2峰性混合分布から有限サンプルを1つ引く。
このサンプルを、以降すべてのヒストグラム/KDEスクリプトで使い回す
(同じデータに対して見方だけを変える、というのがこの記事の主旨のため)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.family"] = "Meiryo"

MU1, SD1 = -2.0, 1.0
MU2, SD2 = 2.2, 1.0
W = 0.5
N = 220
SEED = 42

def true_pdf(g):
    return W * stats.norm.pdf(g, MU1, SD1) + (1 - W) * stats.norm.pdf(g, MU2, SD2)

rng = np.random.default_rng(SEED)
is_comp1 = rng.random(N) < W
x = np.where(is_comp1, rng.normal(MU1, SD1, N), rng.normal(MU2, SD2, N))
np.save("x_sample.npy", x)

g = np.linspace(-6.5, 6.5, 2000)
p = true_pdf(g)
peak1, peak2 = p[g < 0].max(), p[g > 0].max()
valley = p[(g > MU1) & (g < MU2)].min()
print(f"真の分布: N({MU1},{SD1}^2)*{W} + N({MU2},{SD2}^2)*{1-W}")
print(f"真の密度の谷/山比 = {valley/min(peak1,peak2):.3f} (1.0=谷なし単峰, 0=完全に0まで落ちる)")
print(f"サンプル N={N}: range=[{x.min():.2f},{x.max():.2f}], mean={x.mean():.2f}, std={x.std():.2f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(g, p, color="#e4572e", lw=2.2, label="真の密度 (2つの正規分布の混合)")
ax.plot(x, np.full_like(x, -0.01), "|", color="#333", markersize=10, alpha=0.6, label=f"実際に引いたサンプル (N={N})")
ax.set_ylim(-0.03, None)
ax.set_yticks([])
ax.set_xlabel("x")
ax.legend(fontsize=10, loc="upper left")
ax.set_title("真の分布は明確に2峰。だが220点のサンプルだけを渡されたら、あなたは双峰だと言い切れるか?", fontsize=12.5)
fig.tight_layout()
fig.savefig("fig0_setup.png", dpi=110, bbox_inches="tight")
print("saved fig0_setup.png")
