"""陰陽(Yin-Yang)模様の2クラス分類。
(1) ロジスティック回帰と隠れユニット数2/4/8/32のMLPで最終決定境界を比較
(2) 32ユニットMLPの学習過程で決定境界がどう変形するかをGIF化
(3) 「本体の渦」と「目玉」の正解率を別々に追跡し、どちらが先に学ばれるかを測る"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

plt.rcParams["font.family"] = "Meiryo"
rng = np.random.default_rng(0)

R = 1.0
EYE_R = 0.14


def yinyang_label(X):
    """陰陽の定義: 大円内で、S字境界(半径R/2の2つの半円)の左右+目玉の反転"""
    x, y = X[:, 0], X[:, 1]
    up = np.hypot(x, y - R / 2)  # 上の小円中心からの距離
    dn = np.hypot(x, y + R / 2)  # 下の小円中心からの距離
    # 基本: x>0側がクラス1、ただしS字の膨らみで入れ替わる
    label = (x > 0).astype(int)
    label[(up < R / 2) & (x <= 0)] = 1
    label[(up < R / 2) & (x > 0)] = 1
    label[(dn < R / 2)] = 0
    # 正しいS字: 上半分の膨らみはクラス1に、下半分の膨らみはクラス0に
    label = np.where(up < R / 2, 1, np.where(dn < R / 2, 0, (x > 0).astype(int)))
    # 目玉 (所属クラスの中に相手クラスの点)
    eye_up = np.hypot(x, y - R / 2) < EYE_R
    eye_dn = np.hypot(x, y + R / 2) < EYE_R
    label = np.where(eye_up, 0, np.where(eye_dn, 1, label))
    return label, eye_up | eye_dn


def sample(n, seed):
    r = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        cand = r.uniform(-R, R, (n * 2, 2))
        cand = cand[np.hypot(cand[:, 0], cand[:, 1]) < R]
        pts.extend(cand.tolist())
    X = np.array(pts[:n])
    y, is_eye = yinyang_label(X)
    return X, y, is_eye


X_tr, y_tr, eye_tr = sample(2000, 0)
X_te, y_te, eye_te = sample(2000, 1)
print(f"train: {len(X_tr)}, eyes: {eye_tr.sum()} ({eye_tr.mean()*100:.1f}%)")

# グリッド (決定境界表示用)
gx, gy = np.meshgrid(np.linspace(-1.05, 1.05, 220), np.linspace(-1.05, 1.05, 220))
grid = np.c_[gx.ravel(), gy.ravel()]
mask_circle = np.hypot(grid[:, 0], grid[:, 1]) < R

# ---- fig1: データセット ----
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="coolwarm", s=5)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("陰陽データセット (2000点、目玉は相手クラス)", fontsize=11)
fig.tight_layout()
fig.savefig("fig1_data.png", dpi=110, bbox_inches="tight")
print("saved fig1_data.png")


def plot_boundary(ax, model, title):
    p = model.predict_proba(grid)[:, 1].reshape(gx.shape)
    p_masked = np.where(mask_circle.reshape(gx.shape), p, np.nan)
    ax.contourf(gx, gy, p_masked, levels=20, cmap="coolwarm", alpha=0.85)
    ax.contour(gx, gy, np.nan_to_num(p_masked, nan=0.5), levels=[0.5], colors="k", linewidths=1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


# ---- (1) ユニット数スイープ ----
models = {"ロジスティック回帰": LogisticRegression()}
for h in [2, 4, 8, 32]:
    models[f"MLP {h}ユニット"] = MLPClassifier(
        (h,), activation="tanh", max_iter=8000, random_state=0, learning_rate_init=0.01
    )
fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
for ax, (name, m) in zip(axes, models.items()):
    m.fit(X_tr, y_tr)
    acc = m.score(X_te, y_te)
    plot_boundary(ax, m, f"{name}\n test acc {acc:.3f}")
    print(f"{name}: acc {acc:.4f}")
fig.suptitle("モデルの表現力と決定境界 (黒線=境界)", fontsize=12)
fig.tight_layout()
fig.savefig("fig2_capacity.png", dpi=110, bbox_inches="tight")
print("saved fig2_capacity.png")

# ---- (2)(3) 学習過程の追跡 (32ユニット) ----
net = MLPClassifier(
    (32,), activation="tanh", max_iter=1, warm_start=True, random_state=0, learning_rate_init=0.01
)
epochs, frames, acc_body, acc_eye, acc_all = [], [], [], [], []
EPOCHS = 3000
record_at = set(
    list(range(0, 100, 5)) + list(range(100, 500, 25)) + list(range(500, EPOCHS + 1, 100))
)
import warnings

warnings.filterwarnings("ignore")
for ep in range(1, EPOCHS + 1):
    net.fit(X_tr, y_tr)
    if ep in record_at:
        pred = net.predict(X_te)
        epochs.append(ep)
        acc_all.append((pred == y_te).mean())
        acc_body.append((pred[~eye_te] == y_te[~eye_te]).mean())
        acc_eye.append((pred[eye_te] == y_te[eye_te]).mean())
        p = net.predict_proba(grid)[:, 1].reshape(gx.shape)
        frames.append((ep, np.where(mask_circle.reshape(gx.shape), p, np.nan)))
print(f"final: all {acc_all[-1]:.3f}, body {acc_body[-1]:.3f}, eye {acc_eye[-1]:.3f}")
np.savez("training_track.npz", epochs=epochs, acc_all=acc_all, acc_body=acc_body, acc_eye=acc_eye)

# 目玉 vs 本体の学習曲線
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(epochs, acc_body, label="本体の渦 (目玉以外)", color="#2e6ee4")
ax.plot(epochs, acc_eye, label="目玉だけ", color="#e4572e")
ax.set_xscale("log")
ax.set_xlabel("エポック (対数)")
ax.set_ylabel("テスト正解率")
ax.set_title("大きな構造(渦)が先、例外(目玉)は後")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig3_eyes_last.png", dpi=110, bbox_inches="tight")
print("saved fig3_eyes_last.png")

# 境界進化GIF
sel = [f for f in frames if f[0] in record_at]
fig, ax = plt.subplots(figsize=(5.4, 5.6))


def update(fi):
    ep, p = frames[fi]
    ax.clear()
    ax.contourf(gx, gy, p, levels=20, cmap="coolwarm", alpha=0.85)
    ax.contour(gx, gy, np.nan_to_num(p, nan=0.5), levels=[0.5], colors="k", linewidths=1.2)
    i = epochs.index(ep)
    ax.set_title(f"epoch {ep}   全体 {acc_all[i]:.2f} / 目玉 {acc_eye[i]:.2f}", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return []


ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=160)
ani.save("boundary_evolution.gif", writer="pillow", fps=6, dpi=80)
print("saved boundary_evolution.gif")
