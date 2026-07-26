import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"

rng = np.random.default_rng(0)


def decide(w, b, X):
    # 学習時の閾値判定(>=0)と完全に同じ規則で予測する。np.signは0をどちらのクラスにも
    # 倒さないため、学習ループの判定基準とズレて見かけの精度がおかしくなるので使わない。
    return np.where(X @ w + b >= 0, 1.0, -1.0)


def make_data(n_pos, n_neg, seed):
    local_rng = np.random.default_rng(seed)
    pos = local_rng.normal(loc=[2, 2], scale=1.0, size=(n_pos, 2))
    neg = local_rng.normal(loc=[-2, -2], scale=1.0, size=(n_neg, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
    perm = local_rng.permutation(len(X))
    return X[perm], y[perm]


def perceptron_train(X, y, epochs, lr=0.1, shuffle_seed=123):
    w = np.zeros(X.shape[1])
    b = 0.0
    acc_history = []
    norm_history = []
    shuffle_rng = np.random.default_rng(shuffle_seed)
    for epoch in range(epochs):
        order = shuffle_rng.permutation(len(X))    # 固定順の周回だと振動して収束しないことがあるため、毎エポック並べ替える
        for xi, yi in zip(X[order], y[order]):
            pred = 1 if (w @ xi + b) >= 0 else -1
            if pred != yi:                      # 誤分類したときだけ更新(誤差信号あり)
                w += lr * yi * xi
                b += lr * yi
        acc_history.append((decide(w, b, X) == y).mean())
        norm_history.append(np.linalg.norm(w))
    return w, b, acc_history, norm_history


def hebbian_train(X, y, epochs, lr=0.1, shuffle_seed=123):
    w = np.zeros(X.shape[1])
    b = 0.0
    acc_history = []
    norm_history = []
    shuffle_rng = np.random.default_rng(shuffle_seed)
    for epoch in range(epochs):
        order = shuffle_rng.permutation(len(X))
        for xi, yi in zip(X[order], y[order]):
            w += lr * yi * xi                   # 発火の相関だけで常に更新(誤差信号なし)
            b += lr * yi
        acc_history.append((decide(w, b, X) == y).mean())
        norm_history.append(np.linalg.norm(w))
    return w, b, acc_history, norm_history


EPOCHS = 40

print("=== 実験1: 同じ精度でも、重みの中身は何が起きているか ===")
X_bal, y_bal = make_data(500, 500, seed=1)
w_p, b_p, acc_p, norm_p = perceptron_train(X_bal, y_bal, EPOCHS)
w_h, b_h, acc_h, norm_h = hebbian_train(X_bal, y_bal, EPOCHS)
print(f"perceptron: final_acc={acc_p[-1]:.4f}  final_||w||={norm_p[-1]:.2f}")
print(f"hebbian:    final_acc={acc_h[-1]:.4f}  final_||w||={norm_h[-1]:.2f}")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(acc_p, color="tab:blue", label="perceptron学習則: 精度")
ax1.plot(acc_h, color="tab:orange", label="ヘブ則: 精度")
ax1.set_xlabel("エポック")
ax1.set_ylabel("精度")
ax1.set_ylim(0.9, 1.01)
ax2 = ax1.twinx()
ax2.plot(norm_p, color="tab:blue", linestyle="--", label="perceptron学習則: ||w||")
ax2.plot(norm_h, color="tab:orange", linestyle="--", label="ヘブ則: ||w||")
ax2.set_ylabel("重みベクトルのノルム ||w||")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
plt.title("精度はほぼ同じでも、重みの大きさは全く違う動きをする")
plt.tight_layout()
plt.savefig("weight_norm.png", dpi=150)
plt.close()

print("\n=== 実験2: XOR問題 ===")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([-1, 1, 1, -1], dtype=float)

X_xor_rep = np.tile(X_xor, (200, 1))
y_xor_rep = np.tile(y_xor, 200)
perm = rng.permutation(len(X_xor_rep))
X_xor_rep, y_xor_rep = X_xor_rep[perm], y_xor_rep[perm]

w_x, b_x, acc_x, _ = perceptron_train(X_xor_rep, y_xor_rep, epochs=60, lr=0.1)
print(f"single perceptron on XOR: final_acc over epochs (last 10) = {[round(a,3) for a in acc_x[-10:]]}")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mlp_train(X, y01, hidden=4, epochs=20000, lr=1.0, seed=3):
    local_rng = np.random.default_rng(seed)
    n_in = X.shape[1]
    W1 = local_rng.normal(0, 1, size=(n_in, hidden))
    b1 = np.zeros(hidden)
    W2 = local_rng.normal(0, 1, size=(hidden, 1))
    b2 = np.zeros(1)

    loss_history = []
    for epoch in range(epochs):
        h = sigmoid(X @ W1 + b1)
        out = sigmoid(h @ W2 + b2).ravel()
        loss = np.mean((out - y01) ** 2)
        loss_history.append(loss)

        d_out = (out - y01) * out * (1 - out)
        d_W2 = h.T @ d_out.reshape(-1, 1) / len(X)
        d_b2 = d_out.mean()

        d_h = np.outer(d_out, W2.ravel()) * h * (1 - h)
        d_W1 = X.T @ d_h / len(X)
        d_b1 = d_h.mean(axis=0)

        W2 -= lr * d_W2
        b2 -= lr * d_b2
        W1 -= lr * d_W1
        b1 -= lr * d_b1

    return (W1, b1, W2, b2), loss_history


y_xor01 = (y_xor + 1) / 2
(W1, b1, W2, b2), loss_hist = mlp_train(X_xor, y_xor01, hidden=4, epochs=20000, lr=1.0)
final_pred = sigmoid(sigmoid(X_xor @ W1 + b1) @ W2 + b2).ravel()
print(f"2-layer MLP on XOR: final predictions = {np.round(final_pred, 4)}  targets = {y_xor01}")
print(f"2-layer MLP final loss = {loss_hist[-1]:.6f}")

plt.figure(figsize=(9, 5))
plt.plot(acc_x, label="単層パーセプトロン(隠れ層なし)の精度")
plt.xlabel("エポック")
plt.ylabel("XORデータでの精度")
plt.title("単層パーセプトロンはXORを解けず、精度が頭打ちになる")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("xor_perceptron_stuck.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(loss_hist)
plt.xlabel("エポック")
plt.ylabel("損失(MSE)")
plt.title("隠れ層を1つ足したMLPはXORを解いて損失がゼロに近づく")
plt.tight_layout()
plt.savefig("xor_mlp_loss.png", dpi=150)
plt.close()

print("\n=== 実験3: 層を足す代わりに、特徴量を1個足すだけで解けるか ===")
X_xor_aug = np.column_stack([X_xor[:, 0], X_xor[:, 1], X_xor[:, 0] * X_xor[:, 1]])
X_xor_aug_rep = np.tile(X_xor_aug, (200, 1))
y_xor_aug_rep = np.tile(y_xor, 200)   # y_xorから独立に作り直す(実験2で既に並べ替え済みのy_xor_repを使い回すと対応がズレる)

w_aug, b_aug, acc_aug, _ = perceptron_train(X_xor_aug_rep, y_xor_aug_rep, epochs=200, lr=0.1)
print(f"x1*x2を足した単層パーセプトロン on XOR: final_acc = {acc_aug[-1]:.4f}")
print(f"weights (x1, x2, x1*x2) = {np.round(w_aug, 3)}, bias = {round(b_aug, 3)}")

plt.figure(figsize=(9, 5))
plt.plot(acc_x, label="素のx1, x2だけ(層を足さない)")
plt.plot(acc_aug, label="x1*x2を足しただけ(層を足さない)")
plt.xlabel("エポック")
plt.ylabel("XORデータでの精度")
plt.title("層を足さなくても、掛け算という特徴量を1つ足すだけでXORは解ける")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("xor_feature_trick.png", dpi=150)
plt.close()
