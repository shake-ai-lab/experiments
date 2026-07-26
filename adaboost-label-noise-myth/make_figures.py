"""results.npzから記事用の図を生成する"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

OUT = r"C:\tmp\adaboost-label-noise-myth-experiment\figs"
import os
os.makedirs(OUT, exist_ok=True)

R = np.load("results.npz")
ada_err = R["ada_err"]        # (5, 7, 1000)
ada_tr_err = R["ada_tr_err"]
rf_err = R["rf_err"]          # (5, 7)
noise_rates = R["noise_rates"]
W = R["weights_history"]      # (1000, 300)
is_flipped = R["is_flipped"]
Xtr = R["Xtr_w"]
ytr_noisy = R["ytr_noisy_w"]

C_BLUE = "#4477AA"   # class 0 / AdaBoost
C_ORANGE = "#EE7733" # class 1 / RF
C_RED = "#CC3311"    # 誤ラベル強調
C_TEAL = "#009988"

rounds = np.arange(1, 1001)

# ------- 図1: データセット (クリーン vs 10%ノイズ) 縦スタック -------
X0 = Xtr
y_noisy2 = ytr_noisy
flip_idx = np.where(is_flipped)[0]
y0 = y_noisy2.copy()
y0[flip_idx] *= -1  # フリップを戻すとクリーンラベル
y_noisy = y_noisy2

fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6))
for ax, y, title in [
    (axes[0], y0, "クリーンラベル(ラベルは座標から決定的に決まる)"),
    (axes[1], y_noisy, "ラベルノイズ10%(赤丸=フリップされた30点)"),
]:
    ax.scatter(X0[y < 0, 0], X0[y < 0, 1], c=C_BLUE, s=28, alpha=0.85, label="class −1")
    ax.scatter(X0[y > 0, 0], X0[y > 0, 1], c=C_ORANGE, s=28, alpha=0.85, label="class +1")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
axes[1].scatter(X0[flip_idx, 0], X0[flip_idx, 1], facecolors="none",
                edgecolors=C_RED, s=180, linewidths=2.2, label="誤ラベル")
axes[1].legend(fontsize=11, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/1.png", dpi=150)
plt.close(fig)

# ------- 図2: クリーンでの訓練/テスト誤差 -------
tr_mean = ada_tr_err[:, 0].mean(axis=0)
te_mean = ada_err[:, 0].mean(axis=0)
te_lo = ada_err[:, 0].min(axis=0)
te_hi = ada_err[:, 0].max(axis=0)
zero_rounds = [int(np.argmax(ada_tr_err[s, 0] == 0)) + 1 for s in range(5)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(rounds, te_lo, te_hi, color=C_ORANGE, alpha=0.18, lw=0)
ax.plot(rounds, tr_mean, color=C_BLUE, lw=2, label="訓練誤差(5シード平均)")
ax.plot(rounds, te_mean, color=C_ORANGE, lw=2, label="テスト誤差(5シード平均)")
med_zero = int(np.median(zero_rounds))
ax.axvline(med_zero, color="gray", ls=":", lw=1.5)
ax.annotate(f"訓練誤差が0になる\n(中央値: {med_zero}回)", xy=(med_zero, 0.10),
            xytext=(med_zero * 2.2, 0.125), fontsize=11,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xscale("log")
ax.set_xlabel("ブースティング回数(log)", fontsize=12)
ax.set_ylabel("誤分類率", fontsize=12)
ax.set_title("クリーンラベル: 訓練誤差0のあと1,000回まで回してもテスト誤差は壊れない", fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(0, 0.19)
fig.tight_layout()
fig.savefig(f"{OUT}/2.png", dpi=150)
plt.close(fig)

# ------- 図3: ノイズ別テスト誤差カーブ + RF水平線 -------
show_noise = [0.0, 0.10, 0.20, 0.30]
cmap = plt.get_cmap("plasma")
fig, ax = plt.subplots(figsize=(8, 5.4))
for rate in show_noise:
    ni = int(np.argmin(np.abs(noise_rates - rate)))
    col = cmap(0.1 + 0.75 * rate / 0.30)
    m = ada_err[:, ni].mean(axis=0)
    ax.plot(rounds, m, color=col, lw=2.2, label=f"AdaBoost ノイズ{int(rate*100)}%")
    ax.axhline(rf_err[:, ni].mean(), color=col, ls="--", lw=1.4, alpha=0.85)
ax.plot([], [], color="gray", ls="--", lw=1.4, label="Random Forest(同ノイズ, 破線)")
ax.set_xscale("log")
ax.set_xlabel("ブースティング回数(log)", fontsize=12)
ax.set_ylabel("テスト誤差", fontsize=12)
ax.set_title("ラベルノイズがあると回すほど悪化する(破線=同ノイズのRandom Forest)", fontsize=13)
ax.legend(fontsize=10.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/3.png", dpi=150)
plt.close(fig)

# ------- 図4: ヒートマップ (回数 x ノイズ率) -------
mean_err = ada_err.mean(axis=0)  # (7, 1000)
fig, ax = plt.subplots(figsize=(8, 5))
x_edges = np.concatenate([[0.9], (rounds[:-1] + rounds[1:]) / 2, [1050]])
y_edges = np.arange(len(noise_rates) + 1)
pc = ax.pcolormesh(x_edges, y_edges, mean_err, cmap="viridis",
                   norm=Normalize(0, mean_err.max()), shading="flat")
ax.set_xscale("log")
ax.set_yticks(np.arange(len(noise_rates)) + 0.5)
ax.set_yticklabels([f"{int(r*100)}%" for r in noise_rates])
best_rounds = mean_err.argmin(axis=1) + 1
ax.scatter(best_rounds, np.arange(len(noise_rates)) + 0.5, color="white",
           marker="*", s=130, edgecolors="black", linewidths=0.6, zorder=3,
           label="その行の最小テスト誤差")
ax.set_xlabel("ブースティング回数(log)", fontsize=12)
ax.set_ylabel("ラベルノイズ率", fontsize=12)
ax.set_title("テスト誤差ヒートマップ(5シード平均): 谷は常に20〜50回付近にある", fontsize=13)
cb = fig.colorbar(pc, ax=ax)
cb.set_label("テスト誤差", fontsize=11)
ax.legend(fontsize=10, loc="lower left", framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/4.png", dpi=150)
plt.close(fig)

# ------- 図5: ノイズ率 vs テスト誤差 (ada@1000 / ada最良 / RF) -------
ada_1000 = ada_err[:, :, -1]           # (5,7)
ada_best = ada_err.min(axis=2)          # (5,7) オラクル早期停止
fig, ax = plt.subplots(figsize=(8, 5))
xs = noise_rates * 100
for arr, col, ls, label in [
    (ada_1000, C_RED, "-", "AdaBoost 1,000回"),
    (rf_err, C_ORANGE, "-", "Random Forest(300本)"),
    (ada_best, C_BLUE, "-", "AdaBoost 最良の回数で止めた場合(オラクル)"),
]:
    m, sd = arr.mean(axis=0), arr.std(axis=0)
    ax.plot(xs, m, color=col, ls=ls, lw=2.2, marker="o", ms=6, label=label)
    ax.fill_between(xs, m - sd, m + sd, color=col, alpha=0.15, lw=0)
ax.set_xlabel("ラベルノイズ率 [%]", fontsize=12)
ax.set_ylabel("テスト誤差", fontsize=12)
ax.set_title("ノイズ耐性の比較(帯は5シードの±1σ)", fontsize=13)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/5.png", dpi=150)
plt.close(fig)

# ------- 図6: 重みの爆発 (2段スタック) -------
mean_flip = W[:, is_flipped].mean(axis=1)
mean_clean = W[:, ~is_flipped].mean(axis=1)
share_flip = W[:, is_flipped].sum(axis=1)
fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.4), sharex=True)
ax = axes[0]
ax.plot(rounds, mean_flip, color=C_RED, lw=2.2, label="誤ラベル30点の平均重み")
ax.plot(rounds, mean_clean, color=C_BLUE, lw=2.2, label="クリーン270点の平均重み")
ax.axhline(1 / 300, color="gray", ls=":", lw=1.2)
ax.text(1.1, 1 / 300 * 1.15, "初期値 1/300", fontsize=10, color="gray")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("サンプル重み(log)", fontsize=12)
ax.set_title("ノイズ10%: 誤ラベル点の重みだけが釣り上がる", fontsize=13)
ax.legend(fontsize=11)
ax = axes[1]
ax.plot(rounds, share_flip * 100, color=C_RED, lw=2.2)
ax.axhline(10, color="gray", ls=":", lw=1.2)
ax.text(1.1, 11, "点の数の割合(10%)", fontsize=10, color="gray")
ax.set_xscale("log")
ax.set_xlabel("ブースティング回数(log)", fontsize=12)
ax.set_ylabel("誤ラベル点が占める総重みの割合 [%]", fontsize=12)
ax.set_title(f"データの10%しかない誤ラベル点が、最終的に総重みの{share_flip[-1]*100:.0f}%を占める", fontsize=13)
ax.set_ylim(0, 60)
fig.tight_layout()
fig.savefig(f"{OUT}/6.png", dpi=150)
plt.close(fig)

print("figures saved")
print(f"clean: median train0 = {med_zero}, test@1000 mean = {te_mean[-1]:.4f}, "
      f"test min mean = {ada_err[:,0].min(axis=1).mean():.4f}")
print(f"share_flip: start {share_flip[0]*100:.1f}% -> end {share_flip[-1]*100:.1f}%")
