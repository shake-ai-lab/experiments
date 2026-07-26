import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

d = np.load("results.npz")
degrees = d["degrees"]
params = d["params"]
acc_train = d["acc_train"]
acc_test = d["acc_test"]
median_test = d["median_test"]
raw_test = d["raw_test"]
n_train = int(d["n_train"])
noise_std = float(d["noise_std"])
lam = float(d["lam"])

OUT = "C:/Users/kouta/project/forsubwork/src/content/blog/images/double-descent-experiment"

# -------------------------------------------------------------------
# 1.png: raw ridgeless (no stabilization) vs tiny-ridge stabilized, near threshold
# -------------------------------------------------------------------
raw_degrees = degrees[degrees <= 80]
raw_params = raw_degrees + 1
fine_mask = degrees <= 80

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=110)
ax.plot(raw_params, raw_test, color="#c0392b", lw=1.6, marker="o", ms=3, label="正則化なし(厳密なridgeless最小ノルム解)")
ax.plot(params[fine_mask], acc_test[fine_mask], color="#2980b9", lw=2.0, marker="o", ms=3, label=f"tiny ridge (λ={lam:g}) で安定化")
ax.axvline(n_train, color="gray", ls="--", lw=1, alpha=0.8)
ax.text(n_train * 1.03, ax.get_ylim()[1], f"補間閾値 p=n={n_train}", va="top", fontsize=9, color="gray")
ax.set_yscale("log")
ax.set_xlabel("パラメータ数 p (=次数+1)")
ax.set_ylabel("テスト誤差 (MSE, 対数軸)")
ax.set_title("正則化なしだと閾値付近でテスト誤差が桁違いに爆発する")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/1.png")
plt.close(fig)
print("saved 1.png")

# -------------------------------------------------------------------
# 2.png: fitted curve grid at 4 representative capacities
# -------------------------------------------------------------------
x_grid = d["x_grid"]
y_grid_true = d["y_grid_true"]
preds_grid = d["preds_grid"]
gif_degrees = d["gif_degrees"]
x_tr = d["x_tr"]
y_tr = d["y_tr"]

show_degrees = [2, 5, 18, 80]
labels = ["過小適合\n(次数2, p=3)", "ちょうど良い\n(次数5, p=6)", "危険地帯=ピーク付近\n(次数18, p=19)", "過剰パラメータ化\n(次数80, p=81)"]

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), dpi=110, sharey=True)
for ax, dg, lbl in zip(axes, show_degrees, labels):
    idx = np.where(gif_degrees == dg)[0][0]
    ax.plot(x_grid, y_grid_true, color="gray", lw=1.2, ls="--", label="真の関数")
    ax.plot(x_grid, preds_grid[idx], color="#2980b9", lw=1.8, label="フィット")
    ax.scatter(x_tr, y_tr, color="#c0392b", s=14, zorder=5, label="訓練データ(25点)")
    ax.set_ylim(-3, 3)
    ax.set_title(lbl, fontsize=10)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, loc="lower left")
axes[0].set_ylabel("y")
for ax in axes:
    ax.set_xlabel("x")
fig.suptitle("同じ25点の訓練データに対する、容量(次数)を変えたときのフィット", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/2.png")
plt.close(fig)
print("saved 2.png")

# -------------------------------------------------------------------
# 3.png: the main double-descent curve, full range including extended tail
# -------------------------------------------------------------------
best_under_idx = np.argmin(acc_test[:degrees.tolist().index(n_train - 1) + 1])
peak_idx = np.argmax(acc_test)
final_idx = -1

fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=110)
ax.plot(params, acc_test, color="#c0392b", lw=2.0, marker="o", ms=3, label="テスト誤差 (500試行平均)")
ax.plot(params, acc_train, color="#2980b9", lw=1.6, marker="o", ms=2.5, label="訓練誤差 (500試行平均)")
ax.axvline(n_train, color="gray", ls="--", lw=1.2)
ax.axhline(acc_test[best_under_idx], color="#27ae60", ls=":", lw=1.2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("パラメータ数 p (対数軸)")
ax.set_ylabel("MSE (対数軸)")
ax.set_title("二重降下: 補間閾値でピーク、その後は下がるが最良点には届かない")

ax.annotate(f"underfit側の最良点\np={params[best_under_idx]}, test={acc_test[best_under_idx]:.3f}",
            xy=(params[best_under_idx], acc_test[best_under_idx]), xytext=(1.3, 0.09),
            fontsize=8, color="#27ae60", arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1))
ax.annotate(f"ピーク\np={params[peak_idx]}, test={acc_test[peak_idx]:.2f}",
            xy=(params[peak_idx], acc_test[peak_idx]), xytext=(2.0, 3.0),
            fontsize=8, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
ax.annotate(f"p=1001での値\ntest={acc_test[final_idx]:.3f}\n(U字の底の{acc_test[final_idx]/acc_test[best_under_idx]:.2f}倍でまだ上)",
            xy=(params[final_idx], acc_test[final_idx]), xytext=(params[final_idx]*0.12, acc_test[final_idx]*4.0),
            fontsize=8, color="#8e44ad", arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1))
ax.text(n_train * 1.05, ax.get_ylim()[0]*1.3, f"補間閾値\np=n={n_train}", fontsize=8, color="gray")

ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/3.png")
plt.close(fig)
print("saved 3.png")

print("\nsummary numbers:")
print(f"  n_train={n_train} noise_std={noise_std}")
print(f"  best underparam: p={params[best_under_idx]} test={acc_test[best_under_idx]:.5f}")
print(f"  peak: p={params[peak_idx]} test={acc_test[peak_idx]:.5f}  (ratio to best={acc_test[peak_idx]/acc_test[best_under_idx]:.2f}x)")
print(f"  final p={params[final_idx]} test={acc_test[final_idx]:.5f} (ratio to best={acc_test[final_idx]/acc_test[best_under_idx]:.2f}x, ratio to peak={acc_test[peak_idx]/acc_test[final_idx]:.2f}x reduction)")
