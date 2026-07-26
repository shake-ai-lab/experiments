"""ロジスティック写像: バタフライ効果の本体実験 (見せ方標準v2適用)
(1) r=3.9(カオス) と r=2.8(安定) で、x0とx0+1e-8の2軌道の乖離をlogスケールで比較
(2) 乖離の指数関数的成長(または非成長)に直線をフィットして経験的な成長率を推定
(3) 直接計算したリアプノフ指数と比較
(4) カオス側で2軌道が重なった状態から剥がれていく様子のGIF"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12, "lines.linewidth": 2.0})

OUT = "."
EPS = 1e-8
X0 = 0.2


def logistic_iterate(r, x0, n_steps):
    xs = np.empty(n_steps + 1)
    xs[0] = x0
    x = x0
    for i in range(n_steps):
        x = r * x * (1 - x)
        xs[i + 1] = x
    return xs


def direct_lyapunov(r, x0, n_steps):
    """lambda = (1/n) * sum(ln|r(1-2x_i)|) : 写像の微分の対数の時間平均"""
    x = x0
    total = 0.0
    for _ in range(n_steps):
        deriv = abs(r * (1 - 2 * x))
        total += np.log(max(deriv, 1e-300))
        x = r * x * (1 - x)
    return total / n_steps


# ---------------------------------------------------------------------------
# (1)+(2) 乖離の時系列 + 指数フィット: カオス(r=3.9) vs 安定(r=2.8)
# ---------------------------------------------------------------------------
N_STEPS = 60
cases = [
    (3.9, "#C44E52", "r = 3.9 (カオス域)"),
    (2.8, "#4C72B0", "r = 2.8 (安定な1点への収束域)"),
]

results = {}
for r, color, label in cases:
    xs_a = logistic_iterate(r, X0, N_STEPS)
    xs_b = logistic_iterate(r, X0 + EPS, N_STEPS)
    diff = np.abs(xs_a - xs_b)
    diff_safe = np.maximum(diff, 1e-300)
    results[r] = dict(xs_a=xs_a, xs_b=xs_b, diff=diff, color=color, label=label)

    lyap_direct = direct_lyapunov(r, X0, N_STEPS)
    # 長時間平均(transient除去後10万ステップ)で、より正確なリアプノフ指数も計算
    x_settle = X0
    for _ in range(1000):
        x_settle = r * x_settle * (1 - x_settle)
    lyap_long = direct_lyapunov(r, x_settle, 100_000)
    print(f"--- r={r} ---")
    print("diff[0..10]:", diff[:10])
    print("diff[-5:]:", diff[-5:])
    print(f"直接計算したリアプノフ指数(60ステップ平均): {lyap_direct:.5f}")
    print(f"直接計算したリアプノフ指数(transient除去後10万ステップ平均): {lyap_long:.5f}")
    results[r]["lyap_direct"] = lyap_direct
    results[r]["lyap_long"] = lyap_long

# 指数フィット: カオス側は乖離が飽和(x in [0,1]なので上限がある)する前の
# 区間だけを使う。diffが1e-8(初期epsilon)〜0.1の範囲を線形領域とみなす。
r_chaos = 3.9
diff_c = results[r_chaos]["diff"]
mask = (diff_c > EPS * 0.5) & (diff_c < 0.1)
n_fit = np.arange(N_STEPS + 1)[mask]
log_diff_fit = np.log(diff_c[mask])
slope, intercept = np.polyfit(n_fit, log_diff_fit, 1)
print(f"\nカオス域(r={r_chaos})のフィット区間: n={n_fit.min()}〜{n_fit.max()} ({mask.sum()}点)")
print(f"フィットで得た経験的成長率(自然対数/step) = {slope:.5f}")
results[r_chaos]["fit_slope"] = slope
results[r_chaos]["fit_intercept"] = intercept
results[r_chaos]["fit_mask"] = mask

# 安定側(r=2.8)も同じやり方でフィット(縮小方向)。0になる直前まで使う。
r_stable = 2.8
diff_s = results[r_stable]["diff"]
mask_s = diff_s > 1e-14
n_fit_s = np.arange(N_STEPS + 1)[mask_s]
log_diff_fit_s = np.log(diff_s[mask_s])
slope_s, intercept_s = np.polyfit(n_fit_s, log_diff_fit_s, 1)
print(f"\n安定域(r={r_stable})のフィット区間: n={n_fit_s.min()}〜{n_fit_s.max()} ({mask_s.sum()}点)")
print(f"フィットで得た経験的成長率(自然対数/step) = {slope_s:.5f}")
results[r_stable]["fit_slope"] = slope_s
results[r_stable]["fit_intercept"] = intercept_s
results[r_stable]["fit_mask"] = mask_s

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, r in zip(axes, [r_chaos, r_stable]):
    d = results[r]
    diff_plot = np.maximum(d["diff"], 1e-16)
    ax.plot(range(N_STEPS + 1), diff_plot, "o-", color=d["color"], markersize=3.5)
    fit_n = np.arange(N_STEPS + 1)
    fit_line = np.exp(d["fit_intercept"] + d["fit_slope"] * fit_n)
    ax.plot(fit_n, fit_line, "--", color="black", linewidth=1.4,
            label=f"フィット直線 (成長率={d['fit_slope']:.3f}/step)")
    ax.set_yscale("log")
    ax.set_xlabel("反復回数 n")
    ax.set_ylabel("|差| = |x_n - x_n'|" if r == r_chaos else "")
    ax.set_ylim(1e-10, 2)
    ax.legend(fontsize=9, loc="lower right" if r == r_chaos else "upper right")
    verb = "指数関数的に拡大する" if d["fit_slope"] > 0 else "拡大せず縮んでいく"
    ax.set_title(f"{d['label']}: 差が{verb}", fontsize=12.5)
fig.suptitle(f"1e-8の初期差が、rが違うだけで運命が正反対になる", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/3_divergence.png", dpi=160)
plt.close(fig)
print("\nsaved 3_divergence.png")

# ---------------------------------------------------------------------------
# (4) GIF: カオス側で2軌道が重なった状態から剥がれていく
# (先頭フレームを長くホールドして「最初は重なっている」をきちんと見せる)
# ---------------------------------------------------------------------------
import imageio
from PIL import Image

d = results[r_chaos]
n_gif = N_STEPS + 1
# 間引いて枚数を抑える(先頭の重なり区間は細かく、後半はやや粗く)
frame_idxs = sorted(set(list(range(0, 16, 3)) + list(range(16, n_gif, 4)) + [n_gif - 1]))

# タイトルは1行だと右端で切れるので2行に分ける。凡例は軌道が全域を暴れ回って
# どこに置いても重なるため、プロットの外(下)に出す。tight_layoutはフレームごとに
# レイアウトが微妙に動くので使わず、余白を固定する。
frames_dir_imgs = []
fig2, ax2 = plt.subplots(figsize=(7, 4.5), dpi=100)
fig2.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.26)
for frame in frame_idxs:
    ax2.clear()
    upto = frame + 1
    ax2.plot(range(upto), d["xs_a"][:upto], color="#333333", linewidth=2, label="x0 = 0.2")
    ax2.plot(range(upto), d["xs_b"][:upto], color="#C44E52", linewidth=2, linestyle="--",
             label="x0 = 0.2 + 1e-8")
    ax2.set_xlim(0, N_STEPS)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("反復回数 n")
    ax2.set_ylabel("x_n")
    if upto < 15:
        stage = "最初はほぼ完全に重なっている"
    elif upto < 30:
        stage = "少しずつずれ始める"
    else:
        stage = "もはや別の軌道にしか見えない"
    ax2.set_title(f"r=3.9: 1e-8だけ違う2つの初期値が引き裂かれていく\n({stage})",
                  fontsize=12.5)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=11,
               frameon=False)
    fig2.canvas.draw()
    buf = np.asarray(fig2.canvas.buffer_rgba())
    frames_dir_imgs.append(buf[:, :, :3].copy())
plt.close(fig2)

# 全フレーム共通のパレットで量子化する(フレームごとに別パレットだと配色が壊れる)
stacked = np.stack(frames_dir_imgs, axis=0)  # (n_frames, H, W, 3)
h, w = stacked.shape[1], stacked.shape[2]
combined = Image.fromarray(stacked.reshape(-1, w, 3))
combined_q = combined.quantize(colors=24, method=Image.MEDIANCUT)
palette = combined_q.getpalette()
quantized_frames = []
for i in range(len(frames_dir_imgs)):
    frame_q = combined_q.crop((0, i * h, w, (i + 1) * h))
    quantized_frames.append(frame_q)

n_frames = len(quantized_frames)
per_frame_duration_ms = [500] * 4 + [90] * (n_frames - 5) + [1300]
quantized_frames[0].save(
    f"{OUT}/diverge.gif", save_all=True, append_images=quantized_frames[1:],
    duration=per_frame_duration_ms, loop=0, disposal=2,
)
size_kb = __import__("os").path.getsize(f"{OUT}/diverge.gif") / 1024
print(f"saved diverge.gif ({size_kb:.0f} KB, {n_frames} frames)")

print("done step2")
