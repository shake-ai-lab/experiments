"""可視化一式 (見せ方標準v2適用):
(0) セットアップ図解 (explicit ensemble vs dropout+MC-dropout)
(1) データセット (two moons, noise=0.28)
(2) 平均予測+分散のヒートマップ 2x2 (ensemble / dropout)
(3) 点ごとの分散の散布図 (相関係数を明記)
(4) 個々のメンバー/パスがどれだけ「ばらつくか」を見せるGIF
(5) dropout率スイープのplotlyスライダー
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
from scipy.spatial import cKDTree

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11, "lines.linewidth": 2.0})

OUTDIR = "C:/tmp/dropout-implicit-ensemble-test-experiment"
d = np.load(f"{OUTDIR}/results_main.npz")
gx, gy = d["gx"], d["gy"]
X_tr, y_tr, X_te, y_te = d["X_tr"], d["y_tr"], d["X_te"], d["y_te"]
ens_mean, ens_var = d["ens_grid_mean"].reshape(gx.shape), d["ens_grid_var"].reshape(gx.shape)
drop_mean, drop_var = d["drop_grid_mean"].reshape(gx.shape), d["drop_grid_var"].reshape(gx.shape)
ens_test_acc, drop_test_acc, det_acc = float(d["ens_test_acc"]), float(d["drop_test_acc"]), float(d["det_acc"])
corr, rho = float(d["corr"]), float(d["rho"])
M, H_SMALL, H_LARGE = int(d["M"]), int(d["H_SMALL"]), int(d["H_LARGE"])
n_small, n_large = int(d["n_params_small"]), int(d["n_params_large"])

print(f"loaded: ens_acc={ens_test_acc:.4f} drop_acc={drop_test_acc:.4f} det_acc={det_acc:.4f} corr={corr:.4f} rho={rho:.4f}")

# distance from each grid point to the nearest training point -> near-data vs far-from-data (extrapolation) split
grid_flat = np.column_stack([gx.ravel(), gy.ravel()])
tree = cKDTree(X_tr)
dist_to_data, _ = tree.query(grid_flat, k=1)
near_mask = dist_to_data < 0.3
ev_flat, dv_flat = d["ens_grid_var"], d["drop_grid_var"]
corr_near = np.corrcoef(ev_flat[near_mask], dv_flat[near_mask])[0, 1]
corr_far = np.corrcoef(ev_flat[~near_mask], dv_flat[~near_mask])[0, 1]
ens_near_mean, ens_far_mean = ev_flat[near_mask].mean(), ev_flat[~near_mask].mean()
drop_near_mean, drop_far_mean = dv_flat[near_mask].mean(), dv_flat[~near_mask].mean()
print(f"near-data (n={near_mask.sum()}): corr={corr_near:.3f}  ens_var_mean={ens_near_mean:.5f}  drop_var_mean={drop_near_mean:.5f}")
print(f"far-from-data (n={(~near_mask).sum()}): corr={corr_far:.3f}  ens_var_mean={ens_far_mean:.5f}  drop_var_mean={drop_far_mean:.5f}")
print(f"ensemble var far/near ratio: {ens_far_mean/ens_near_mean:.2f}x   MC-dropout var far/near ratio: {drop_far_mean/drop_near_mean:.2f}x")

# ---------------- (0) setup diagram ----------------
# 720px幅の本文カラムで読めるよう、左右並びではなく上下スタック+大きめフォントで描く
fig, ax = plt.subplots(figsize=(7.4, 8.2))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.5, 0.975, f"総パラメータ数はほぼ同じ ({n_small*M} vs {n_large})\n── 「同じ予算」での対決 ──",
        ha="center", va="top", fontsize=14, fontweight="bold")

# top: explicit ensemble
ax.text(0.5, 0.885, f"explicit ensemble (M={M})", ha="center", va="top", fontsize=15,
        fontweight="bold", color="#2e6ee4")
for i in range(M):
    cx = 0.075 + i * 0.087
    ax.add_patch(plt.Rectangle((cx, 0.70), 0.062, 0.145, fc="#dfe8f7", ec="#2e6ee4", lw=1.4))
ax.text(0.5, 0.675, f"独立に初期化・独立に学習 ── 各 隠れ{H_SMALL}ユニット (計{n_small*M}params)",
        ha="center", va="top", fontsize=12)
ax.annotate("", xy=(0.5, 0.575), xytext=(0.5, 0.640),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#2e6ee4"))
ax.text(0.5, 0.565, f"{M}個の予測を平均 → test acc {ens_test_acc:.3f}",
        ha="center", va="top", fontsize=13, color="#2e6ee4")

ax.plot([0.03, 0.97], [0.505, 0.505], color="#999", lw=1, ls="--")

# bottom: single dropout net + MC-dropout
ax.text(0.5, 0.475, "dropout付きMLP 1個 + MC-dropout", ha="center", va="top", fontsize=15,
        fontweight="bold", color="#e4572e")
ax.add_patch(plt.Rectangle((0.25, 0.225), 0.5, 0.185, fc="#fbe3dc", ec="#e4572e", lw=1.6))
ax.text(0.5, 0.3175, f"隠れ{H_LARGE}ユニット ({n_large}params)\np=0.5でdropout学習",
        ha="center", va="center", fontsize=12.5)
ax.annotate("", xy=(0.5, 0.135), xytext=(0.5, 0.215),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#e4572e"))
ax.text(0.5, 0.125, f"推論時もdropoutをONのまま{M}回forwardして平均 (MC-dropout)\n→ test acc {drop_test_acc:.3f}",
        ha="center", va="top", fontsize=13, color="#e4572e")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig0_setup.png", dpi=110, bbox_inches="tight")
print("saved fig0_setup.png")
plt.close(fig)

# ---------------- (1) dataset ----------------
fig, ax = plt.subplots(figsize=(5.4, 5.0))
ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], s=10, color="#2e6ee4", alpha=0.7, label="class 0")
ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], s=10, color="#e4572e", alpha=0.7, label="class 1")
ax.set_title(f"two moons (noise=0.28): 重なりが大きく、単純な境界では解けない", fontsize=12)
ax.legend(fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig1_data.png", dpi=110, bbox_inches="tight")
print("saved fig1_data.png")
plt.close(fig)

# ---------------- (2) 2x2 heatmap: mean / var x ensemble / dropout ----------------
diff = ens_var - drop_var
iy, ix = np.unravel_index(np.argmax(diff), diff.shape)
hot_x, hot_y = gx[iy, ix], gy[iy, ix]

fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.2))
vmax_var = max(ens_var.max(), drop_var.max())

im0 = axes[0, 0].pcolormesh(gx, gy, ens_mean, cmap="coolwarm", vmin=0, vmax=1, shading="auto")
axes[0, 0].set_title("ensemble: 平均予測(クラス1の確率)", fontsize=12)
fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].pcolormesh(gx, gy, drop_mean, cmap="coolwarm", vmin=0, vmax=1, shading="auto")
axes[0, 1].set_title("MC-dropout: 平均予測(クラス1の確率)", fontsize=12)
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

im2 = axes[1, 0].pcolormesh(gx, gy, ens_var, cmap="viridis", vmin=0, vmax=vmax_var, shading="auto")
axes[1, 0].set_title(f"ensemble: 分散(不確実性) 最大{ens_var.max():.3f}", fontsize=12)
fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

im3 = axes[1, 1].pcolormesh(gx, gy, drop_var, cmap="viridis", vmin=0, vmax=vmax_var, shading="auto")
axes[1, 1].set_title(f"MC-dropout: 分散(不確実性) 最大{drop_var.max():.3f}", fontsize=12)
fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

# annotate the hottest mismatch point on BOTH variance panels (data-outside region: ensemble explodes, dropout doesn't)
for axp in (axes[1, 0], axes[1, 1]):
    axp.annotate("データが疎な外挿域\n(ここでensembleだけ急上昇)",
                 xy=(hot_x, hot_y), xytext=(0.02, 0.03), textcoords="axes fraction",
                 fontsize=9, color="white",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.4))

for axp in axes.flat:
    axp.scatter(X_tr[:, 0], X_tr[:, 1], s=2, color="k", alpha=0.15)
    axp.set_xticks([])
    axp.set_yticks([])

fig.suptitle(f"平均予測はほぼ一致 (acc {ens_test_acc:.3f} vs {drop_test_acc:.3f})、"
             f"分散の形は食い違う (相関 r={corr:.2f})", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUTDIR}/fig2_heatmap.png", dpi=110, bbox_inches="tight")
print("saved fig2_heatmap.png")
plt.close(fig)

# ---------------- (3) scatter: per-point variance, ensemble vs mc-dropout, colored by distance from training data ----------------
ev, dv = d["ens_grid_var"], d["drop_grid_var"]
fig, ax = plt.subplots(figsize=(6.8, 6.2))
sca = ax.scatter(ev, dv, s=5, alpha=0.5, c=dist_to_data, cmap="magma_r", vmax=1.2)
cb = fig.colorbar(sca, ax=ax, fraction=0.046)
cb.set_label("最寄りの訓練点までの距離")
lim = max(ev.max(), dv.max()) * 1.05
ax.plot([0, lim], [0, lim], color="#999", ls="--", lw=1.3, label="y = x (完全一致ならこの線に乗る)")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("explicit ensembleの分散 (grid各点)")
ax.set_ylabel("MC-dropoutの分散 (grid各点)")
ax.set_title(f"全体では弱い相関 (Pearson r={corr:.3f}, Spearman ρ={rho:.3f})", fontsize=12.5)
ax.annotate(f"データ疎な領域(色が濃い点)ほど\nensembleの分散だけが伸びる\n"
            f"近傍(dist<0.3): r={corr_near:.3f}  /  遠方: r={corr_far:.3f}",
            xy=(lim * 0.80, lim * 0.30), xytext=(lim * 0.28, lim * 0.70),
            fontsize=9.5, color="#e4572e",
            arrowprops=dict(arrowstyle="->", color="#e4572e", lw=1.5))
# 凡例は右下だと外挿域の点(x大・y小の暗色クラスタ)に被るので左上に置く
ax.legend(fontsize=10, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig3_scatter.png", dpi=110, bbox_inches="tight")
print("saved fig3_scatter.png")
plt.close(fig)

r_check, _ = pearsonr(ev, dv)
print(f"sanity check pearsonr: {r_check:.4f} (matches saved corr {corr:.4f})")

# ---------------- (4) GIF: individual members/passes flashing ----------------
fr = np.load(f"{OUTDIR}/individual_frames.npz")
ens_members = fr["ens_grid_members"]   # (M, n_grid)
drop_passes = fr["drop_grid_passes"]   # (T, n_grid)
n_show = min(ens_members.shape[0], drop_passes.shape[0])

cmap = "coolwarm"
import imageio.v2 as imageio
frames_png = []
fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.15), dpi=78)
for i in range(n_show):
    for ax in axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(X_tr[:, 0], X_tr[:, 1], s=1.2, color="k", alpha=0.12)
    axes[0].pcolormesh(gx, gy, ens_members[i].reshape(gx.shape), cmap=cmap, vmin=0, vmax=1, shading="auto")
    axes[0].set_title(f"ensemble メンバー {i+1}/{n_show}", fontsize=9.5)
    axes[1].pcolormesh(gx, gy, drop_passes[i].reshape(gx.shape), cmap=cmap, vmin=0, vmax=1, shading="auto")
    axes[1].set_title(f"MC-dropout パス {i+1}/{n_show}", fontsize=9.5)
    fig.suptitle("独立に学習した10個 vs 同じ網の10回のdropoutサンプル", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames_png.append(buf.copy())
plt.close(fig)
durations = [1.6] + [0.55] * (n_show - 1)
imageio.mimsave(f"{OUTDIR}/flicker.gif", frames_png, duration=durations)
print("saved flicker.gif (custom durations)")

# ---------------- (5) plotly slider: dropout rate sweep ----------------
sw = np.load(f"{OUTDIR}/sweep_results.npz")
ps, means, vars_, accs, corrs = sw["ps"], sw["means"], sw["vars_"], sw["accs"], sw["corrs"]
sgx, sgy = sw["gx"], sw["gy"]
vmax_sweep = max(vars_.max(), ens_var.max())

figp = go.Figure()
# trace 0: fixed ensemble variance reference (always visible)
figp.add_trace(go.Heatmap(z=ens_var, x=sgx[0], y=sgy[:, 0], zmin=0, zmax=vmax_sweep,
                           colorscale="Viridis", showscale=True, colorbar=dict(x=0.46, title="分散"),
                           xaxis="x", yaxis="y"))
# traces for each p (dropout variance), only one visible at a time
n_p = len(ps)
for i, p in enumerate(ps):
    figp.add_trace(go.Heatmap(z=vars_[i], x=sgx[0], y=sgy[:, 0], zmin=0, zmax=vmax_sweep,
                               colorscale="Viridis", showscale=True, colorbar=dict(x=1.0, title="分散"),
                               xaxis="x2", yaxis="y2", visible=(i == 0)))

steps = []
for i, p in enumerate(ps):
    vis = [True] + [j == i for j in range(n_p)]
    steps.append(dict(
        method="update",
        args=[{"visible": vis},
              {"annotations": [
                  dict(text=f"dropout p={p:.1f}: test acc {accs[i]:.3f}, ensembleとの分散相関 r={corrs[i]:.3f}",
                       x=0.5, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=14))
              ]}],
        label=f"p={p:.1f}",
    ))

figp.update_layout(
    grid=dict(rows=1, columns=2, pattern="independent"),
    xaxis=dict(domain=[0, 0.46], visible=False), yaxis=dict(visible=False, scaleanchor="x"),
    xaxis2=dict(domain=[0.54, 1.0], visible=False), yaxis2=dict(visible=False, scaleanchor="x2"),
    sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="dropout率: "))],
    annotations=[dict(text=f"dropout p=0.1: test acc {accs[0]:.3f}, ensembleとの分散相関 r={corrs[0]:.3f}",
                       x=0.5, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=14))],
    title=dict(text="左=explicit ensembleの分散(固定基準) / 右=MC-dropoutの分散(スライダーでp変化)", x=0.5),
    margin=dict(l=10, r=10, t=110, b=10), height=560, width=1000,
)
figp.write_html(f"{OUTDIR}/dropout-rate-sweep-slider.html", include_plotlyjs="cdn", full_html=True)
print("saved dropout-rate-sweep-slider.html")
print("ALL VISUALS DONE")
