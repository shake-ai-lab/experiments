"""繧ｯ繝ｩ繧､繝ｳ縺ｮ螢ｺ縺ｮ荳ｭ縺ｫ貎懊・繝｡繝薙え繧ｹ縺ｮ霈ｪ繧貞叙繧雁・縺吶・繝√Η繝ｼ繝門捉繧翫・蠎ｧ讓・v 縺・0 莉倩ｿ代・邏ｰ縺・ｸｯ (|v| < 0.7) 繧貞・繧雁・縺吶→縲・繧ｯ繝ｩ繧､繝ｳ縺ｮ螢ｺ縺ｧ縺ｯ u 縺御ｸ蜻ｨ縺励◆縺ｨ縺・v 縺ｮ隨ｦ蜿ｷ縺悟渚霆｢縺励※雋ｼ繧雁粋繧上＆繧九・縺ｧ縲・縺薙・蟶ｯ縺ｯ繝｡繝薙え繧ｹ縺ｮ霈ｪ縺ｫ縺ｪ縺｣縺ｦ縺・ｋ縲ゅヨ繝ｼ繝ｩ繧ｹ縺九ｉ蜷後§蟶ｯ繧貞・繧九→縺溘□縺ｮ蟶ｯ(蜀・ｭ・縲・蜑榊屓(mobius-strip-unfolding)縺ｨ蜷後§縲瑚｡ｨ陬上・荳閾ｴ蠎ｦ縲阪〒Isomap螻暮幕縺ｮ蜿崎ｻ｢繧呈ｸｬ繧九・"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

R, r = 2.0, 1.0
N = 1600
DELTA = 0.7  # 蟶ｯ縺ｮ蜊雁ｹ・(v縺ｮ遽・峇)


def band(surface, n=N, seed=3, noise=0.01):
    rg = np.random.default_rng(seed)
    u = rg.uniform(0, 2 * np.pi, n)
    v = rg.uniform(-DELTA, DELTA, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    if surface == "klein":
        z = r * np.sin(v) * np.cos(u / 2)
        w = r * np.sin(v) * np.sin(u / 2)
        X = np.c_[x, y, z, w]
    else:  # torus
        z = r * np.sin(v)
        X = np.c_[x, y, z]
    return X + rg.normal(0, noise, X.shape), u, v


CASES = [("torus", "繝医・繝ｩ繧ｹ縺九ｉ蛻・▲縺溷ｸｯ", "#2e6ee4"),
         ("klein", "繧ｯ繝ｩ繧､繝ｳ縺ｮ螢ｺ縺九ｉ蛻・▲縺溷ｸｯ", "#e4572e")]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
summary = {}
for row, (sname, cname, col) in enumerate(CASES):
    X, u, v = band(sname)
    emb = Isomap(n_neighbors=12, n_components=2).fit_transform(X)
    c = emb - emb.mean(0)
    theta = np.arctan2(c[:, 1], c[:, 0])
    radius = np.hypot(c[:, 0], c[:, 1])
    NB = 48
    bins = np.linspace(-np.pi, np.pi, NB + 1)
    agree = np.full(NB, np.nan)
    for b in range(NB):
        m = (theta >= bins[b]) & (theta < bins[b + 1])
        if m.sum() < 8:
            continue
        side = np.sign(radius[m] - np.median(radius[m]))
        agree[b] = np.mean(side * np.sign(v[m]))
    valid = ~np.isnan(agree)
    sgn = np.sign(agree[valid])
    flips = int((sgn[:-1] * sgn[1:] < 0).sum() + (sgn[-1] * sgn[0] < 0))
    summary[cname] = flips
    print(f"{cname}: 隨ｦ蜿ｷ蜿崎ｻ｢縺ｮ蝗樊焚 {flips}, |荳閾ｴ蠎ｦ|縺ｮ蟷ｳ蝮・{np.nanmean(np.abs(agree)):.3f}")

    ax = axes[row, 0]
    ax.scatter(emb[:, 0], emb[:, 1], c=v, cmap="RdBu", s=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Isomap縺ｮ螻暮幕: {cname}", fontsize=13)
    ax2 = axes[row, 1]
    centers = (bins[:-1] + bins[1:]) / 2
    ax2.plot(centers[valid], agree[valid], "-o", ms=4, color=col)
    ax2.axhline(0, color="#888", lw=1)
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_xlabel("螻暮幕蝗ｳ荳翫〒縺ｮ隗貞ｺｦ")
    ax2.set_ylabel("陦ｨ陬上・荳閾ｴ蠎ｦ")
    ax2.set_title(f"荳閾ｴ蠎ｦ縺悟渚霆｢縺励◆蝗樊焚: {flips}蝗・, fontsize=13)
    ax2.grid(alpha=0.3)
    if flips > 0:
        idx = np.flatnonzero(sgn[:-1] * sgn[1:] < 0)
        if len(idx):
            xf = centers[valid][idx[0]]
            ax2.annotate("縺薙％縺ｧ陦ｨ陬上′\n蜈･繧梧崛繧上ｋ", xy=(xf, 0), xytext=(xf - 2.2, 0.55),
                         fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.6))
    else:
        ax2.annotate("蜈ｨ蜻ｨ縺ｧ荳閾ｴ\n=縺溘□縺ｮ蟶ｯ", xy=(0, np.nanmean(agree)),
                     xytext=(-2.8, -0.6), fontsize=11,
                     arrowprops=dict(arrowstyle="->", lw=1.6))
fig.suptitle("繧ｯ繝ｩ繧､繝ｳ縺ｮ螢ｺ縺九ｉ蛻・ｊ蜃ｺ縺励◆蟶ｯ縺ｯ縲√Γ繝薙え繧ｹ縺ｮ霈ｪ縺ｫ縺ｪ縺｣縺ｦ縺・ｋ", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig4_band_attempt.png", dpi=110, bbox_inches="tight")
print("saved fig4_band_attempt.png")
