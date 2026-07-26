"""可視化一式:
(1) sigmoid vs 最良フィットのscaled-ReLU(俗説「ReLUはsigmoidの簡略化」の検証)、
    フィット窓[-8,8]と、その外挿[-30,30]の2パネル
(2) 初期化直後の1層目の勾配の大きさ、深さ×4活性化関数(logスケール、sigmoidの崩壊が主役)
(3) 実際に学習させた場合のtwo-moonsテスト精度、深さ×4活性化関数
(4) 深さ6での決定境界を4活性化関数で並べる(区分線形 vs 滑らかの見た目比較)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11, "lines.linewidth": 2.0})

OUTDIR = "C:/tmp/activation-function-myths"

COL = {"ReLU": "#2e6ee4", "GELU": "#2fa96b", "tanh": "#9b59b6", "sigmoid": "#e4572e"}

# ---------------- (1) curve fit myth ----------------
cf = np.load(f"{OUTDIR}/curve_fit_results.npz")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
ax.plot(cf["xs"], cf["ys_sigmoid"], color="#e4572e", lw=2.3, label="sigmoid(x)")
ax.plot(cf["xs"], cf["ys_fit"], color="#2e6ee4", lw=2.0, ls="--",
        label=f"最良フィットの scale・ReLU(x+shift)\n(R²={float(cf['r2']):.3f}, フィット窓内)")
ax.set_xlim(-8, 8)
ax.set_title("フィット窓 x∈[-8,8] では、それなりに近く見える", fontsize=11.5)
ax.set_xlabel("x")
ax.legend(fontsize=9.5, loc="upper left")

ax = axes[1]
ax.plot(cf["xs_wide"], cf["ys_sigmoid_wide"], color="#e4572e", lw=2.3, label="sigmoid(x)  (0〜1に飽和)")
ax.plot(cf["xs_wide"], cf["ys_fit_wide"], color="#2e6ee4", lw=2.0, ls="--",
        label="同じ係数のscaled ReLU\n(飽和せず無限に伸び続ける)")
ax.axvspan(-8, 8, color="gray", alpha=0.12, label="フィット窓")
ax.set_title(f"窓の外(x∈[-30,30])に広げると、最大誤差{float(cf['max_abs_err_wide']):.2f}まで乖離", fontsize=11.5)
ax.set_xlabel("x")
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("俗説「ReLUはsigmoidの簡略化」を検証: 同じ係数の直線でも、窓の外では全く別物になる", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUTDIR}/fig1_curve_fit_myth.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig1_curve_fit_myth.png")

# ---------------- (2) gradient magnitude at init vs depth ----------------
ds = np.load(f"{OUTDIR}/depth_sweep_results.npz")
depths = ds["depths"]

fig, ax = plt.subplots(figsize=(8.5, 5.6))
for name in ["ReLU", "GELU", "tanh", "sigmoid"]:
    ax.plot(depths, ds[f"grad_{name}"], "o-", color=COL[name], lw=2.2, ms=6, label=name)
ax.set_yscale("log")
ax.set_xlabel("隠れ層の深さ")
ax.set_ylabel("初期化直後の1層目の勾配の大きさ (|dW|平均, logスケール)")
ax.set_title("sigmoidだけが深さとともに勾配が指数的に消えていく\n(depth=1の1.1e-2からdepth=24で2.1e-17まで、15桁の崩壊)", fontsize=11.8)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig2_gradient_vs_depth.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig2_gradient_vs_depth.png")

# ---------------- (3) test accuracy vs depth ----------------
fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.axhline(0.5, color="#999", lw=1.2, ls="--", label="チャンスレベル(2値分類の当てずっぽう)")
for name in ["ReLU", "GELU", "tanh", "sigmoid"]:
    ax.plot(depths, ds[f"acc_{name}"], "o-", color=COL[name], lw=2.2, ms=6, label=name)
ax.set_xlabel("隠れ層の深さ")
ax.set_ylabel("two-moons テスト精度")
ax.set_title("sigmoidは深さ16以降、実質チャンスレベル(0.49)まで学習が崩壊した", fontsize=12)
ax.legend(fontsize=10, loc="lower left")
ax.set_ylim(0.4, 1.0)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig3_accuracy_vs_depth.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig3_accuracy_vs_depth.png")

# ---------------- (4) decision boundaries at fixed depth ----------------
gx, gy = ds["gx"], ds["gy"]
X_tr, y_tr = ds["X_tr"], ds["y_tr"]
boundary_depth = int(ds["boundary_depth"])

fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
for ax, name in zip(axes, ["ReLU", "GELU", "tanh", "sigmoid"]):
    pred = ds[f"boundary_{name}"]
    acc = float(ds[f"boundary_acc_{name}"])
    ax.contourf(gx, gy, pred, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.75)
    ax.contour(gx, gy, pred, levels=[0.5], colors="black", linewidths=1.8)
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="RdBu_r", s=6, edgecolor="white", linewidth=0.15, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name}  (acc={acc:.3f})", fontsize=11.5, color=COL[name])
fig.suptitle(f"深さ{boundary_depth}層での決定境界(色=予測確率): sigmoidだけ境界の周りに幅広く滑らかな確率勾配を作り、\nReLU/GELU/tanhはどれも境界のすぐ外でほぼ確信度100%に切り替わる", fontsize=12.0)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(f"{OUTDIR}/fig4_decision_boundaries.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved fig4_decision_boundaries.png")

print("ALL VISUALS DONE")
