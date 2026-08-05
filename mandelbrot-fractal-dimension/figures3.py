# fig5: 場所別・拡大率別の次元。
# 予想は外れた。「メインカージオイドの縁はなめらかな解析曲線だから次元1のはず」と思って
# 測ったら 1.85 まで上がった。カージオイドの縁には、あらゆる有理角に小さな球が
# びっしり生えているので、Mの境界としては全くなめらかではない。
# そして、ある深さから先は「集合内の画素が0%」になって測定自体が壊れる。

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

d3 = json.load(open("dimension3.json", encoding="utf-8"))
LABEL = {
    "なめらかな縁(メインカージオイド上)": ("カージオイドの縁", "#ffd166"),
    "Misiurewicz点 (-0.1011+0.9563i)": ("Misiurewicz点", "#ef476f"),
    "シーホースバレー入口 (-0.75+0.01i)": ("シーホースバレー", "#06d6a0"),
}

fig, ax = plt.subplots(figsize=(8.0, 5.6))
for name, rec in d3["centers"].items():
    label, col = LABEL[name]
    w = np.array([r["half_width"] for r in rec])
    D = np.array([r["D"] if r["D"] == r["D"] else np.nan for r in rec])
    frac = np.array([r["inside_frac"] for r in rec])
    # 集合内の画素が消えかけている(1%未満)測定は信用できないので分けて描く
    good = (frac > 0.01) & (frac < 0.999) & ~np.isnan(D)
    ax.plot(w[good], D[good], "o-", lw=2.4, ms=8, color=col, label=label)
    bad = ~good
    if bad.any():
        ax.plot(w[bad], np.nan_to_num(D[bad], nan=0.0), "x", ms=10, mew=2.4,
                color=col, alpha=0.85)

ax.axhline(2.0, ls="--", color="#8d99ae", lw=1.6)
ax.text(1.1e-1, 2.03, "理論値 = 2 (Shishikura 1998)", color="#5c6672", fontsize=10)
ax.axhline(1.0904, ls=":", color="#5c6672", lw=1.6)
ax.text(1.1e-1, 1.11, "集合全体をまとめて測ったときの値 1.090", color="#5c6672", fontsize=10)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_ylim(-0.1, 2.25)
ax.set_xlabel("見ている窓の半幅（右へ行くほど拡大）")
ax.set_ylabel("box-counting で測った次元")
ax.set_title("境界の一点を拡大していくと、次元推定は 2 に近づく。\nただし、ある深さで集合が画面から消えて測定が壊れる（×印）",
             fontsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=10.5, loc="center left")
fig.tight_layout()
fig.savefig("fig5_by_location.png", dpi=130)
plt.close(fig)
print("saved fig5_by_location.png")

# --- fig1 のラベルを実測に合わせて修正して作り直す
from common import escape_time  # noqa: E402
from figures import colorize  # noqa: E402

XMIN, XMAX, YMIN, YMAX = -2.0, 0.6, -1.3, 1.3
R = 2000
esc = escape_time(XMIN, XMAX, YMIN, YMAX, R, R, 800)
fig, ax = plt.subplots(figsize=(7.2, 7.2))
ax.imshow(colorize(esc, 800), extent=[XMIN, XMAX, YMIN, YMAX], origin="lower")
theta = 2.0
smooth = np.exp(1j * theta) / 2 - np.exp(2j * theta) / 4
for x, y, label, col, off in [
    (smooth.real, smooth.imag, "カージオイドの縁", "#ffd166", (14, 8)),
    (-0.10109636384562, 0.95628651080914, "Misiurewicz点", "#ef476f", (12, 10)),
    (-0.75, 0.01, "シーホースバレー", "#06d6a0", (-160, 12)),
]:
    ax.plot(x, y, "o", ms=9, mfc="none", mec=col, mew=2.2)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=off,
                color=col, fontsize=11, fontweight="bold")
ax.set_xlabel("Re(c)")
ax.set_ylabel("Im(c)")
ax.set_title("マンデルブロ集合と、境界のどこを拡大して測るか", fontsize=13)
fig.tight_layout()
fig.savefig("fig1_overview.png", dpi=130)
plt.close(fig)
print("saved fig1_overview.png")
