import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from core import load_subjects, moments, hensachi_threshold_row
from step2_selected_and_small_class import SELECTED

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
OUT = r"C:\Users\kouta\project\forsubwork\src\content\blog\images\hensachi-top-percent-real-scores"
os.makedirs(OUT, exist_ok=True)
subj = load_subjects()
C_BAR = "#3b6fb6"
C_NORM = "#d9534f"
C_GREY = "#666666"

# ---------- 図1: 分布と正規分布、偏差値60/65/70の得点 ----------
fig, axes = plt.subplots(2, 1, figsize=(9, 9))
for ax, nm, title, ymax in zip(axes, ["英語（リーディング）", "化学"], ["英語(リーディング)", "化学"], (2.4, 3.0)):
    d = subj[nm]
    c = d["counts"]
    n, m, sd = moments(c)
    s_ = np.arange(len(c))
    ax.bar(s_, c / n * 100, width=1.0, color=C_BAR, alpha=0.85, label="実際の受験者(1点刻み)")
    xs = np.linspace(-20, 130, 600)
    ax.plot(xs, norm.pdf(xs, m, sd) * 100, color=C_NORM, lw=2.5, label="同じ平均・SDの正規分布")
    ax.axvline(len(c) - 1, color="black", lw=2, label="満点(100点)")
    r = hensachi_threshold_row(c)
    ax.set_ylim(0, ymax)
    for h, col, yy in zip((60, 65, 70), ("#e08a1e", "#c2571a", "#8a1c1c"), (0.98, 0.80, 0.62)):
        thr = r[h]["thr"]
        ax.axvline(thr, color=col, ls="--", lw=2)
        if thr > len(c) - 1:
            txt = f"偏差値{h} = {thr:.0f}点\n満点超え 実測0%\n(正規: {r[h]['normal']*100:.1f}%)"
            ax.text(thr + 1, ymax * 0.62, txt, color=col, fontsize=11.5, va="top", ha="left",
                    bbox=dict(fc="white", ec=col, alpha=0.9))
        else:
            txt = f"偏差値{h} = {thr:.0f}点\n実測 {r[h]['actual']*100:.1f}% (正規: {r[h]['normal']*100:.1f}%)"
            ax.text(thr - 1, ymax * yy, txt, color=col, fontsize=11.5, va="top", ha="right",
                    bbox=dict(fc="white", ec=col, alpha=0.9))
    ax.set_xlim(-8, 128)
    ax.set_title(f"{title}  (2025年本試験、受験者 {n:,} 人、平均 {m:.1f}、SD {sd:.1f})", fontsize=14)
    ax.set_xlabel("得点", fontsize=13)
    ax.set_ylabel("受験者の割合 (%/1点)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper left" if nm.startswith("英語") else "upper right", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "1.png"), dpi=130)
plt.close(fig)

# ---------- 図2: 16科目の実測 vs 正規分布(縦に3段) ----------
fig, axes = plt.subplots(3, 1, figsize=(9, 15), sharex=False)
labels = [nm.replace("，", "・") for nm in SELECTED]
y = np.arange(len(SELECTED))[::-1]
for ax, h, xmax in zip(axes, (60, 65, 70), (24, 11, 6)):
    vals = []
    for nm in SELECTED:
        r = hensachi_threshold_row(subj[nm]["counts"])[h]
        vals.append((r["actual"] * 100, r["k"] > subj[nm]["full"]))
    norm_v = hensachi_threshold_row(subj[SELECTED[0]]["counts"])[h]["normal"] * 100
    ax.barh(y, [v for v, _ in vals], color=[("#999999" if cap else C_BAR) for _, cap in vals], height=0.7)
    ax.axvline(norm_v, color=C_NORM, lw=2.5)
    ax.text(norm_v, len(SELECTED) - 0.1, f" 正規分布なら {norm_v:.1f}%", color=C_NORM, fontsize=13, va="bottom")
    for yy, (v, cap) in zip(y, vals):
        ax.text(v + xmax * 0.01, yy, ("満点でも届かない 0%" if cap else f"{v:.1f}%"), va="center", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, len(SELECTED) + 1.0)
    ax.set_title(f"偏差値{h}の得点以上を取った人の実際の割合", fontsize=14)
    ax.set_xlabel("受験者に占める割合 (%)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "2.png"), dpi=130)
plt.close(fig)

# ---------- 図3: 少人数クラスで測った偏差値のばらつき ----------
pop = np.repeat(np.arange(101), subj["英語（リーディング）"]["counts"])
n_pop, m_pop, sd_pop = moments(subj["英語（リーディング）"]["counts"])
x0 = int(round(m_pop + sd_pop))
REPS = 20000
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
for ax, n in zip(axes, (10, 40, 200)):
    rng = np.random.default_rng(0)
    S = pop[rng.integers(0, len(pop), size=(REPS, n))].astype(float)
    h = 50 + 10 * (x0 - S.mean(axis=1)) / S.std(axis=1)
    lo, hi = np.percentile(h, [5, 95])
    ax.hist(h, bins=np.arange(48, 80, 0.5), color=C_BAR, alpha=0.85)
    ax.axvline(60.16, color=C_NORM, lw=2.5)
    ax.axvspan(lo, hi, color="#e08a1e", alpha=0.18)
    ax.set_title(f"{n}人のクラスで測ると: 90%の範囲 {lo:.1f}〜{hi:.1f}", fontsize=14)
    ax.set_ylabel("クラスの数", fontsize=12)
    ax.tick_params(labelsize=11)
axes[-1].set_xlabel(f"同じ素点({x0}点、全国の偏差値60.2)の偏差値  ※赤線=全国の値", fontsize=12)
axes[-1].set_xlim(48, 78)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "3.png"), dpi=130)
plt.close(fig)
print("saved")
