import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

plt.rcParams["font.family"] = "Meiryo"

# Illustrative Beta-Binomial model: "impression" as a probability estimate updated by observed evidence.
# Competence impression: many strong prior observations already accumulated (well-established track record).
# Warmth/relatability impression: very few prior observations (largely unknown, close to uninformative).
comp_a, comp_b = 46, 4     # ~50 pseudo-observations, strongly skewed positive (established competence)
warm_a, warm_b = 1, 1      # uninformative prior (barely anything known about "warmth" yet)

# one new data point: a minor, humanizing flaw (e.g. a clumsy slide deck)
# -> counts as one "negative" observation for the competence axis,
#    but simultaneously one "positive/relatable" observation for the warmth axis.
comp_a2, comp_b2 = comp_a, comp_b + 1
warm_a2, warm_b2 = warm_a + 1, warm_b

comp_mean_before = comp_a / (comp_a + comp_b)
comp_mean_after = comp_a2 / (comp_a2 + comp_b2)
warm_mean_before = warm_a / (warm_a + warm_b)
warm_mean_after = warm_a2 / (warm_a2 + warm_b2)

print(f"competence impression: {comp_mean_before:.4f} -> {comp_mean_after:.4f}  (Δ={comp_mean_after-comp_mean_before:+.4f})")
print(f"warmth impression:     {warm_mean_before:.4f} -> {warm_mean_after:.4f}  (Δ={warm_mean_after-warm_mean_before:+.4f})")

x = np.linspace(0, 1, 500)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(x, beta.pdf(x, comp_a, comp_b), color="steelblue", label="観測前(すでに実績が多い)")
axes[0].plot(x, beta.pdf(x, comp_a2, comp_b2), color="crimson", linestyle="--", label="小さな失敗を1回観測した後")
axes[0].set_title(f"「有能さ」の印象\n平均 {comp_mean_before:.3f} → {comp_mean_after:.3f}(Δ{comp_mean_after-comp_mean_before:+.3f})")
axes[0].set_xlabel("有能さの推定確率")
axes[0].legend(fontsize=8)

axes[1].plot(x, beta.pdf(x, warm_a, warm_b), color="steelblue", label="観測前(まだ何も分からない)")
axes[1].plot(x, beta.pdf(x, warm_a2, warm_b2), color="crimson", linestyle="--", label="親しみの持てる一面を1回観測した後")
axes[1].set_title(f"「親しみやすさ」の印象\n平均 {warm_mean_before:.3f} → {warm_mean_after:.3f}(Δ{warm_mean_after-warm_mean_before:+.3f})")
axes[1].set_xlabel("親しみやすさの推定確率")
axes[1].legend(fontsize=8)

fig.suptitle("同じ「1回の出来事」でも、それまでの蓄積が薄い軸ほど、印象は大きく動く(ベイズ更新のイメージ)")
plt.tight_layout()
plt.savefig("C:/tmp/pratfall-experiment/fig1_bayes.png", dpi=150, bbox_inches="tight")
print("saved fig1_bayes.png")
