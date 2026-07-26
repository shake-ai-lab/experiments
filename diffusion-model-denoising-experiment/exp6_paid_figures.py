"""Figures + rendered formula images for the paid DDPM note article.

note.com cannot render LaTeX, so every formula ships as a PNG (large font,
white background). Layout follows the vertical-stack / big-font rule for
schematic comparisons.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "Meiryo"
IN = Path("C:/tmp/diffusion-experiment")
OUT = IN / "paid"

X0 = np.load(IN / "X0.npy")
summary = json.load(open(OUT / "summary.json"))

BLUE, RED, GRAY = "steelblue", "crimson", "0.45"


def panel(ax, pts, title, color=RED, sub=""):
    ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, color=color)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title + ("\n" + sub if sub else ""), fontsize=15)


# ============ 1. alpha_bar curves ============
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for name, label, color, ls in [
    ("linear_T100", "linear(元の実験) " + r"$\bar{\alpha}_T$" + "=0.36", BLUE, "-"),
    ("linear_scaled_T100", "linear 強め " + r"$\bar{\alpha}_T \approx 10^{-4}$", "darkorange", "-"),
    ("cosine_T100", "cosine " + r"$\bar{\alpha}_T \approx 2 \times 10^{-7}$", "seagreen", "-"),
]:
    ab = np.load(OUT / f"abars_{name}.npy")
    ax.plot(np.arange(1, len(ab) + 1), ab, ls, color=color, lw=2.8, label=label)
ax.axhline(0, color="0.6", lw=1)
ax.set_xlabel("時刻 t", fontsize=15)
ax.set_ylabel(r"$\bar{\alpha}_t$" + "  残っている信号の割合", fontsize=15)
ax.set_title("ノイズスケジュール別の " + r"$\bar{\alpha}_t$" + chr(10) + "元の実験は t=100 でも信号が36%残っていた", fontsize=16)
ax.tick_params(labelsize=13)
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig_abar_curves.png", dpi=150, bbox_inches="tight")
print("fig_abar_curves.png")

# ============ 2. schedule comparison samples (2x2, vertical-ish grid) ============
fig, axes = plt.subplots(2, 2, figsize=(9, 9.6))
panel(axes[0, 0], X0[:1500], "本物のデータ", color=BLUE)
for ax, name, title in [
    (axes[0, 1], "linear_T100", "linear(元の実験)"),
    (axes[1, 0], "linear_scaled_T100", "linear 強め"),
    (axes[1, 1], "cosine_T100", "cosine"),
]:
    s = summary[name]
    panel(ax, np.load(OUT / f"gen_{name}.npy"), title,
          sub=rf"$\bar{{\alpha}}_T$" + f"={s['alpha_bar_T']:.2g} / 実データまでの平均距離 {s['gen_to_real']:.3f}")
fig.suptitle("ノイズスケジュールを変えて生成した結果(T=100)", fontsize=17)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig_schedule_samples.png", dpi=150, bbox_inches="tight")
print("fig_schedule_samples.png")

# ============ 3. step count comparison (2x2) ============
fig, axes = plt.subplots(2, 2, figsize=(9, 9.6))
for ax, name, title in [
    (axes[0, 0], "cosine_T100", "cosine T=100"),
    (axes[0, 1], "cosine_T25", "cosine T=25"),
    (axes[1, 0], "cosine_T10", "cosine T=10"),
    (axes[1, 1], "naive_T10", "linear T=10(端点を使い回した失敗例)"),
]:
    s = summary[name]
    panel(ax, np.load(OUT / f"gen_{name}.npy"), title,
          sub=rf"$\bar{{\alpha}}_T$" + f"={s['alpha_bar_T']:.2g} / 平均距離 {s['gen_to_real']:.3f}")
fig.suptitle("ステップ数Tを減らすとどうなるか", fontsize=17)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig_T_samples.png", dpi=150, bbox_inches="tight")
print("fig_T_samples.png")

# ============ 4. time embedding ablation (vertical stack) ============
fig, axes = plt.subplots(2, 1, figsize=(5.4, 10.2))
s1, s2 = summary["linear_T100"], summary["no_temb_T100"]
panel(axes[0], np.load(OUT / "gen_linear_T100.npy"),
      "時刻埋め込みあり(通常のDDPM)", sub=f"平均距離 {s1['gen_to_real']:.3f}")
panel(axes[1], np.load(OUT / "gen_no_temb_T100.npy"),
      "時刻埋め込みなし(tを教えない)", sub=f"平均距離 {s2['gen_to_real']:.3f}", color="darkorange")
fig.suptitle("ネットワークに「今何時か」を教えないと\nどうなるか", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig_temb.png", dpi=150, bbox_inches="tight")
print("fig_temb.png")

# ============ 5. baseline loss curve ============
losses = np.load(IN / "losses.npy")
w = 200
smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(losses, color="0.85", lw=0.5)
ax.plot(np.arange(w - 1, len(losses)), smooth, color=BLUE, lw=2.5)
ax.axhline(1.0, color=RED, lw=1.8, ls="--")
ax.text(500, 1.02, "1.0 = 何も学習していない状態(ノイズの分散)", fontsize=13, color=RED)
ax.set_xlabel("イテレーション", fontsize=14)
ax.set_ylabel("損失(ノイズ予測の平均二乗誤差)", fontsize=14)
ax.set_title("学習曲線: 0.93 → 0.68", fontsize=16)
ax.tick_params(labelsize=12)
ax.set_ylim(0.5, 1.15)
fig.tight_layout()
fig.savefig(OUT / "fig_loss_curve.png", dpi=150, bbox_inches="tight")
print("fig_loss_curve.png")

# ============ 6. concept diagram (vertical stack, big fonts) ============
abars_lin = np.load(OUT / "abars_linear_T100.npy")
rng = np.random.default_rng(3)
noise_fixed = rng.normal(0, 1, size=X0.shape)
show = [0, 50, 99]
fig, axes = plt.subplots(2, 3, figsize=(10.2, 8.2))
for j, t in enumerate(show):
    ab = abars_lin[t]
    xt = np.sqrt(ab) * X0 + np.sqrt(1 - ab) * noise_fixed
    panel(axes[0, j], xt[:1500], f"t={t}", color=BLUE)
data = np.load(IN / "samples.npz")
for j, t in enumerate([100, 50, 0]):
    panel(axes[1, j], data[str(t)], f"t={t}", color=RED)
axes[0, 0].set_ylabel("前向き過程\n(ノイズを混ぜるだけ)", fontsize=16)
axes[1, 0].set_ylabel("逆向き過程\n(学習したモデルで復元)", fontsize=16)
for axrow, x0f, x1f in [(axes[0], 0.30, 0.44), (axes[1], 0.30, 0.44)]:
    pass
fig.text(0.5, 0.965, "拡散モデルの全体像: 壊す方は簡単、戻す方を学習する", fontsize=17, ha="center")
fig.text(0.5, 0.515, "→ 進む方向 →", fontsize=15, ha="center", color=BLUE)
fig.text(0.5, 0.06, "→ 進む方向 →", fontsize=15, ha="center", color=RED)
fig.tight_layout(rect=[0, 0.07, 1, 0.95])
fig.savefig(OUT / "fig_concept.png", dpi=150, bbox_inches="tight")
print("fig_concept.png")

# ============ 7. network schematic (vertical stack, big font) ============
fig, ax = plt.subplots(figsize=(7, 8.5))
ax.axis("off")
boxes = [
    ("入力\n座標 x (2次元) + 時刻埋め込み (8次元) = 10次元", "#dbe9f6"),
    ("全結合層 10 → 256\n+ ReLU", "#f6e7db"),
    ("全結合層 256 → 2", "#f6e7db"),
    ("出力\n「混ざっているノイズ」の予測 (2次元)", "#e3f6db"),
]
y = 0.92
for text, color in boxes:
    ax.text(0.5, y, text, fontsize=17, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color, edgecolor="0.4"))
    if y > 0.2:
        ax.annotate("", xy=(0.5, y - 0.155), xytext=(0.5, y - 0.075),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="0.3"))
    y -= 0.23
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("今回使うネットワークの全体像(これだけ)", fontsize=18)
fig.savefig(OUT / "fig_network.png", dpi=150, bbox_inches="tight")
print("fig_network.png")

# ============ 8. formula images (mathtext -> png) ============
FORMULAS = [
    ("f01_forward_step",
     r"$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)$"),
    ("f02_alpha_def",
     r"$\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \alpha_1 \alpha_2 \cdots \alpha_t$"),
    ("f03_closed_form",
     r"$x_t = \sqrt{\bar{\alpha}_t}\; x_0 + \sqrt{1-\bar{\alpha}_t}\;\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$"),
    ("f04_two_step",
     r"$x_2 = \sqrt{\alpha_2}\left(\sqrt{\alpha_1}\,x_0 + \sqrt{1-\alpha_1}\,\epsilon_1\right) + \sqrt{1-\alpha_2}\,\epsilon_2$"),
    ("f05_merge_noise",
     r"$\sqrt{\alpha_2(1-\alpha_1)}\,\epsilon_1 + \sqrt{1-\alpha_2}\,\epsilon_2 \;=\; \sqrt{1-\alpha_1\alpha_2}\,\epsilon$"),
    ("f06_loss",
     r"$L(\theta) = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\; \left\| \epsilon - \epsilon_\theta(x_t,\ t) \right\|^2 \;\right]$"),
    ("f07_reverse_mean",
     r"$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\;\epsilon_\theta(x_t, t)\right)$"),
    ("f08_reverse_step",
     r"$x_{t-1} = \mu_\theta(x_t, t) + \sqrt{\beta_t}\; z, \qquad z \sim \mathcal{N}(0, I)$"),
    ("f09_terminal_check",
     r"$x_{100} = \sqrt{0.36}\; x_0 + \sqrt{0.64}\;\epsilon = 0.60\, x_0 + 0.80\, \epsilon$"),
    ("f10_x0_estimate",
     r"$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\;\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$"),
]
for name, tex in FORMULAS:
    fig, ax = plt.subplots(figsize=(9, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.5, tex, fontsize=23, ha="center", va="center")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(f"{name}.png")

print("all figures done")
