"""ドーナツ型(同心円)データで、カーネルトリックを「目で見る」。
(1) 2Dでは線形SVMが分離できない(acc~0.5)、RBFカーネルなら分離できる
(2) z = x^2+y^2 の3次元目を明示的に足すと、線形の平面1枚で切れる
(3) 二重らせんでは2次の持ち上げでは足りず、カーネルの選択が効く
(4) RBFのgammaを振ると過学習の顔が出る"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})
rng = np.random.default_rng(0)


def make_spirals(n=600, noise=0.06, seed=1):
    r = np.random.default_rng(seed)
    n2 = n // 2
    t = r.uniform(0.25, 1.0, n2) * 3 * np.pi
    x1 = np.c_[t * np.cos(t), t * np.sin(t)] / (3 * np.pi)
    x2 = np.c_[t * np.cos(t + np.pi), t * np.sin(t + np.pi)] / (3 * np.pi)
    X = np.vstack([x1, x2]) + r.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n2, int), np.ones(n2, int)]
    return X, y


Xc, yc = make_circles(n_samples=600, factor=0.45, noise=0.06, random_state=0)
Xs, ys = make_spirals()
Xc_te, yc_te = make_circles(n_samples=600, factor=0.45, noise=0.06, random_state=9)
Xs_te, ys_te = make_spirals(seed=7)
np.savez("data.npz", Xc=Xc, yc=yc, Xs=Xs, ys=ys)


def boundary_plot(ax, model, X, y, title):
    g = np.linspace(X[:, 0].min() - 0.25, X[:, 0].max() + 0.25, 300)
    h = np.linspace(X[:, 1].min() - 0.25, X[:, 1].max() + 0.25, 300)
    gx, gy = np.meshgrid(g, h)
    Z = model.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, Z > 0, levels=1, cmap="coolwarm", alpha=0.25)
    ax.contour(gx, gy, Z, levels=[0], colors="k", linewidths=1.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)


# ---- (1)(3) 2x2: 線形 vs RBF × ドーナツ vs らせん ----
fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
results = {}
for j, (X, y, Xte, yte, dname) in enumerate(
    [(Xc, yc, Xc_te, yc_te, "ドーナツ"), (Xs, ys, Xs_te, ys_te, "二重らせん")]
):
    for i, (kname, model) in enumerate(
        [("線形SVM", SVC(kernel="linear")), ("RBFカーネルSVM", SVC(kernel="rbf", gamma=20))]
    ):
        model.fit(X, y)
        acc = model.score(Xte, yte)
        results[(kname, dname)] = acc
        boundary_plot(axes[i, j], model, X, y, f"{kname} / {dname}  acc {acc:.2f}")
        print(f"{kname} {dname}: {acc:.3f}")
axes[0, 0].annotate("直線1本ではどう引いても半々", xy=(0, 0), xytext=(-1.45, 1.25),
                    fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.5))
fig.suptitle("同じSVMでも、カーネル次第で見える世界が変わる", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig1_2x2.png", dpi=110, bbox_inches="tight")
print("saved fig1_2x2.png")

# ---- (2) 明示的な持ち上げ: 3D線形分離 ----
z_lift = (Xc**2).sum(1)
X3 = np.c_[Xc, z_lift]
from sklearn.svm import LinearSVC

lin3 = SVC(kernel="linear").fit(X3, yc)
z_lift_te = (Xc_te**2).sum(1)
acc3 = lin3.score(np.c_[Xc_te, z_lift_te], yc_te)
w = lin3.coef_[0]
b = lin3.intercept_[0]
print(f"持ち上げ後の線形SVM: acc {acc3:.3f}, 平面 z~{-b/w[2]:.3f} (w={np.round(w,3)})")

# らせんに同じ持ち上げをしても無駄なことも確認
zs = (Xs**2).sum(1)
lin3s = SVC(kernel="linear").fit(np.c_[Xs, zs], ys)
acc3s = lin3s.score(np.c_[Xs_te, (Xs_te**2).sum(1)], ys_te)
print(f"らせんに2次の持ち上げ: acc {acc3s:.3f} (効かない)")

# ---- (4) gammaスイープ ----
gammas = np.logspace(-1, 3.2, 18)
tr_acc, te_acc = [], []
for gm in gammas:
    m = SVC(kernel="rbf", gamma=gm).fit(Xs, ys)
    tr_acc.append(m.score(Xs, ys))
    te_acc.append(m.score(Xs_te, ys_te))
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.semilogx(gammas, tr_acc, "-o", ms=5, color="#2e6ee4", label="訓練データ")
ax.semilogx(gammas, te_acc, "-o", ms=5, color="#e4572e", label="テストデータ")
best = int(np.argmax(te_acc))
ax.annotate(f"ちょうど良い柔らかさ\n(test {te_acc[best]:.2f})",
            xy=(gammas[best], te_acc[best]), xytext=(gammas[best] * 0.04, 0.72),
            fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("訓練は満点のまま\nテストだけ落ちる=丸暗記", xy=(gammas[-2], te_acc[-2]),
            xytext=(30, 0.6), fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.5))
ax.set_xlabel("gamma (大きいほど境界が細かく曲がる)")
ax.set_ylabel("正解率")
ax.set_title("カーネルの柔らかさにも「ちょうど良さ」がある", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig3_gamma.png", dpi=110, bbox_inches="tight")
print("saved fig3_gamma.png")
np.savez("res.npz", gammas=gammas, tr=tr_acc, te=te_acc, plane_z=-b / w[2])
