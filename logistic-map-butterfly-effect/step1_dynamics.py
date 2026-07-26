"""ロジスティック写像: 基本挙動 + 分岐図 (見せ方標準v2適用)
(1) r=2.8/3.2/3.9 の代表軌道 (small multiples, 結論を書いたタイトル)
(2) 分岐図 (周期倍分岐カスケード・カオス突入を図中に直接注釈)
(3) 分岐図のインタラクティブ版 (rをスライダーで動かして軌道を見る)"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12, "lines.linewidth": 2.0})

OUT = "."


def logistic_iterate(r, x0, n_steps):
    xs = np.empty(n_steps + 1)
    xs[0] = x0
    x = x0
    for i in range(n_steps):
        x = r * x * (1 - x)
        xs[i + 1] = x
    return xs


# ---------------------------------------------------------------------------
# (1) 代表的なrでの軌道 (small multiples)
# ---------------------------------------------------------------------------
reps = [
    (2.8, "r = 2.8: 1点に収束する", "#4C72B0"),
    (3.2, "r = 3.2: 2周期振動に落ち着く", "#55A868"),
    (3.9, "r = 3.9: 収束せず、カオス的に暴れ続ける", "#C44E52"),
]
N_STEPS_REP = 60
X0_REP = 0.2

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (r, title, color) in zip(axes, reps):
    xs = logistic_iterate(r, X0_REP, N_STEPS_REP)
    ax.plot(range(N_STEPS_REP + 1), xs, color=color, marker=".", markersize=3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("反復回数 n")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("x_n" if ax is axes[0] else "")
fig.suptitle("同じ式・同じ初期値でも、rが変わるだけで挙動がまったく別物になる", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/1_trajectories.png", dpi=150)
plt.close(fig)
print("saved 1_trajectories.png")

for r, title, _ in reps:
    xs = logistic_iterate(r, X0_REP, N_STEPS_REP)
    print(f"r={r}: last 8 values = {np.round(xs[-8:], 5)}")

# ---------------------------------------------------------------------------
# (2) 分岐図 (静的、注釈つき)
# ---------------------------------------------------------------------------
R_MIN, R_MAX = 2.4, 4.0
N_R = 2000
TRANSIENT = 500
KEEP = 200

r_values = np.linspace(R_MIN, R_MAX, N_R)
r_plot = []
x_plot = []
for r in r_values:
    x = 0.5
    for _ in range(TRANSIENT):
        x = r * x * (1 - x)
    for _ in range(KEEP):
        x = r * x * (1 - x)
        r_plot.append(r)
        x_plot.append(x)
r_plot = np.array(r_plot)
x_plot = np.array(x_plot)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(r_plot, x_plot, ",", color="#333333", alpha=0.35, markersize=0.6)
ax.set_xlabel("パラメータ r")
ax.set_ylabel("長期的なxの値 (最初の500回を捨てた後の200点)")
ax.set_title("rを増やすと、1点への収束が周期倍分岐を経てカオスへ崩れていく", fontsize=14)

# 注釈: 最初の分岐 (r=3.0) -- xyは実際のアトラクタ値(1-1/r=0.667)に合わせる
ax.annotate("r≈3.0: 1周期→2周期\n最初の分岐", xy=(3.0, 0.667), xytext=(2.55, 0.85),
            fontsize=10.5, color="#1a5fb4",
            arrowprops=dict(arrowstyle="->", color="#1a5fb4", lw=1.6))
# 注釈: 周期倍カスケード領域 -- xyは実際にr=3.45で下側ブランチが通る値(約0.44)に合わせる
ax.annotate("周期倍分岐のカスケード\n(2→4→8→16…周期)", xy=(3.45, 0.44), xytext=(3.02, 0.05),
            fontsize=10.5, color="#c64600",
            arrowprops=dict(arrowstyle="->", color="#c64600", lw=1.6))
# 注釈: カオス突入 -- xyは実際にr=3.7で点群が存在する範囲内(実測max0.925)に合わせる
ax.axvline(3.569946, color="#c01c28", linestyle="--", linewidth=1.3, alpha=0.8)
label_bbox = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85)
ax.annotate("r≈3.570でカオスに突入\n(以降は白いすじ=窓を除き全域に散る)", xy=(3.7, 0.90),
            xytext=(3.66, 0.62), fontsize=10.5, color="#c01c28", bbox=label_bbox,
            arrowprops=dict(arrowstyle="->", color="#c01c28", lw=1.6, shrinkB=6))
# 注釈: 周期3の窓
ax.annotate("r≈3.83: 周期3の窓\n(カオスの中に規則性が戻る)", xy=(3.83, 0.5), xytext=(3.72, 0.15),
            fontsize=10.5, color="#2d7d46", bbox=label_bbox,
            arrowprops=dict(arrowstyle="->", color="#2d7d46", lw=1.6, shrinkB=6))

ax.axvline(2.8, color="#4C72B0", linestyle=":", linewidth=1.2)
ax.axvline(3.2, color="#55A868", linestyle=":", linewidth=1.2)
ax.axvline(3.9, color="#C44E52", linestyle=":", linewidth=1.2)
# 注: 軸の目盛りラベル(2.8/3.2など)と重なるので、色付きラベルは軸の下ではなく
# プロット内側の下端に置く(白バックで点群より手前に)
r_label_bbox = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85)
ax.text(2.8, 0.015, "2.8", ha="center", va="bottom", fontsize=10, color="#4C72B0", bbox=r_label_bbox)
ax.text(3.2, 0.015, "3.2", ha="center", va="bottom", fontsize=10, color="#55A868", bbox=r_label_bbox)
ax.text(3.9, 0.015, "3.9", ha="center", va="bottom", fontsize=10, color="#C44E52", bbox=r_label_bbox)

ax.set_xlim(R_MIN, R_MAX)
ax.set_ylim(-0.03, 1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/2_bifurcation.png", dpi=160)
plt.close(fig)
print("saved 2_bifurcation.png")

# ---------------------------------------------------------------------------
# (3) スライダー付きplotly: rを動かして軌道を見る
# ---------------------------------------------------------------------------
slider_rs = np.round(np.arange(2.4, 4.001, 0.05), 3)
N_STEPS_SLIDER = 80
X0_SLIDER = 0.2

figp = go.Figure()
for i, r in enumerate(slider_rs):
    xs = logistic_iterate(r, X0_SLIDER, N_STEPS_SLIDER)
    figp.add_trace(
        go.Scatter(
            x=list(range(N_STEPS_SLIDER + 1)), y=xs, mode="lines+markers",
            visible=(i == 0), line=dict(color="#C44E52", width=2), marker=dict(size=3),
            hoverinfo="skip",
        )
    )
steps = []
for i, r in enumerate(slider_rs):
    steps.append(
        dict(method="update", label=f"{r:.2f}",
             args=[{"visible": [j == i for j in range(len(slider_rs))]},
                    {"title": f"r = {r:.2f} のときの軌道 (x0 = {X0_SLIDER})"}])
    )
figp.update_layout(
    sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="r = "))],
    title=f"r = {slider_rs[0]:.2f} のときの軌道 (x0 = {X0_SLIDER})",
    xaxis=dict(title="反復回数 n", range=[0, N_STEPS_SLIDER]),
    yaxis=dict(title="x_n", range=[-0.02, 1.02]),
    margin=dict(l=50, r=20, t=60, b=40), height=560,
)
figp.write_html(f"{OUT}/logistic-map-sweep.html", include_plotlyjs="cdn", full_html=True)
print("saved logistic-map-sweep.html")

print("done step1")
