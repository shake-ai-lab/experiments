import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

seq_len = 16
d_ks = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

var_scores_noscale = []
max_attn_noscale = []
max_attn_scaled = []

n_trials = 50
for d_k in d_ks:
    trial_var = []
    trial_max_ns = []
    trial_max_s = []
    for _ in range(n_trials):
        Q = rng.normal(0, 1, size=(seq_len, d_k))
        K = rng.normal(0, 1, size=(seq_len, d_k))
        scores = Q @ K.T
        trial_var.append(scores.var())
        trial_max_ns.append(softmax(scores).max(axis=-1).mean())
        trial_max_s.append(softmax(scores / np.sqrt(d_k)).max(axis=-1).mean())
    var_scores_noscale.append(np.mean(trial_var))
    max_attn_noscale.append(np.mean(trial_max_ns))
    max_attn_scaled.append(np.mean(trial_max_s))

print("d_k    var(QK^T)   理論値d_k   softmax最大重み(無scale)  softmax最大重み(scale後)")
for d_k, v, mns, ms in zip(d_ks, var_scores_noscale, max_attn_noscale, max_attn_scaled):
    print(f"{d_k:5d}  {v:9.2f}   {d_k:8d}   {mns:.4f}                  {ms:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(d_ks, var_scores_noscale, "o-", label="実測 var(Q·K)")
axes[0].plot(d_ks, d_ks, "--", color="gray", label="理論値 d_k")
axes[0].set_xscale("log", base=2)
axes[0].set_yscale("log", base=2)
axes[0].set_xlabel("d_k(QとKの次元数)")
axes[0].set_ylabel("内積のスコアの分散")
axes[0].set_title("次元数と内積スコアの分散")
axes[0].legend()

axes[1].plot(d_ks, max_attn_noscale, "o-", color="crimson", label="スケーリングなし")
axes[1].plot(d_ks, max_attn_scaled, "o-", color="steelblue", label="1/√d_k でスケーリング")
axes[1].axhline(1 / seq_len, ls=":", color="gray", label=f"完全一様(1/{seq_len})")
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("d_k(QとKの次元数)")
axes[1].set_ylabel("softmax後の最大attention重み(平均)")
axes[1].set_title("スケーリングの有無と重みの偏り")
axes[1].legend()

plt.tight_layout()
plt.savefig("C:/tmp/attention-experiment/fig2_scaling.png", dpi=150, bbox_inches="tight")
print("\nsaved fig2_scaling.png")
