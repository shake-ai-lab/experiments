import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image

plt.rcParams["font.family"] = "Meiryo"

d = np.load("sweep_data.npz")
hours_a, hours_b = d["hours_a"], d["hours_b"]
noise_a, noise_b = d["noise_a"], d["noise_b"]

SLOPE = 3.5
INTERCEPT_B = 15.0

deltas = np.linspace(0, 70, 14)  # fewer frames -> smaller file

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=80)

hours_all = np.concatenate([hours_a, hours_b])
xs_a = np.linspace(hours_a.min(), hours_a.max(), 30)
xs_b = np.linspace(hours_b.min(), hours_b.max(), 30)
xs_all = np.linspace(hours_all.min(), hours_all.max(), 30)


def compute(delta):
    intercept_a = INTERCEPT_B + delta
    score_a = np.clip(intercept_a + SLOPE * hours_a + noise_a, 0, 100)
    score_b = np.clip(INTERCEPT_B + SLOPE * hours_b + noise_b, 0, 100)
    score_all = np.concatenate([score_a, score_b])
    r_pool, _ = stats.pearsonr(hours_all, score_all)
    slope_pool, intercept_pool, *_ = stats.linregress(hours_all, score_all)
    slope_a, intercept_a_fit, *_ = stats.linregress(hours_a, score_a)
    slope_b, intercept_b_fit, *_ = stats.linregress(hours_b, score_b)
    return dict(score_a=score_a, score_b=score_b, r_pool=r_pool, slope_pool=slope_pool,
                intercept_pool=intercept_pool, slope_a=slope_a, intercept_a_fit=intercept_a_fit,
                slope_b=slope_b, intercept_b_fit=intercept_b_fit)


frames = []
for delta in deltas:
    ax.clear()
    res = compute(delta)

    ax.scatter(hours_a, res["score_a"], s=14, alpha=0.6, color="#2563eb", label="上級者")
    ax.scatter(hours_b, res["score_b"], s=14, alpha=0.6, color="#f97316", label="初級者")
    ax.plot(xs_a, res["intercept_a_fit"] + res["slope_a"] * xs_a, color="#1e3a8a", lw=2)
    ax.plot(xs_b, res["intercept_b_fit"] + res["slope_b"] * xs_b, color="#c2410c", lw=2)

    pool_color = "#dc2626" if res["r_pool"] < 0 else "#111827"
    ax.plot(xs_all, res["intercept_pool"] + res["slope_pool"] * xs_all,
            color=pool_color, lw=2.5, linestyle="--", label="全体プール")

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    ax.set_xlabel("週あたり利用時間 (時間)")
    ax.set_ylabel("到達テストのスコア")
    status = "パラドックス発生 (r<0)" if res["r_pool"] < 0 else "まだパラドックスなし (r>0)"
    ax.set_title(f"Δ={delta:5.1f}点   r_pool={res['r_pool']:+.3f}   {status}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf, "RGBA").convert("RGB")
    img = img.quantize(colors=32, method=Image.MEDIANCUT)
    frames.append(img)

# hold last frame a bit longer so viewers can read the final state
durations = [220] * (len(frames) - 1) + [900]

frames[0].save(
    "sweep.gif",
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=True,
)
print("saved sweep.gif")
