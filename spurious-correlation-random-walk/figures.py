# -*- coding: utf-8 -*-
"""記事用の図版を生成する(experiment.py の出力を使用)。"""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy import stats

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
IMG = Path(r"c:\Users\kouta\project\forsubwork\src\content\blog\images"
           r"\spurious-correlation-random-walk")
IMG.mkdir(parents=True, exist_ok=True)

# パレット(validate済みのカテゴリカル3色+チャートクローム)
C_IID = "#2a78d6"   # 青
C_AR1 = "#1baf7a"   # アクア
C_RW = "#eb6834"    # オレンジ
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

with open(HERE / "results.json", encoding="utf-8") as f:
    RES = json.load(f)
RHIST = np.load(HERE / "rhist.npz")
GIFP = np.load(HERE / "gif_pairs.npz")


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


# ---- 図1: r分布ヒストグラム(縦3段スタック) ------------------------------
def fig1():
    T = 200
    df = T - 2
    tcrit = stats.t.ppf(0.975, df)
    r_crit = tcrit / np.sqrt(tcrit**2 + df)
    bins = np.linspace(-1, 1, 81)
    panels = [
        ("iid", C_IID, "iidガウスノイズ", RES["A_iid"]),
        ("ar1", C_AR1, "AR(1) φ=0.8(定常・自己相関あり)", RES["A_ar1"]),
        ("rw", C_RW, "ランダムウォーク(単位根)", RES["A_rw"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7, 8.2), sharex=True)
    for ax, (key, color, label, st) in zip(axes, panels):
        style_ax(ax)
        ax.hist(RHIST[key], bins=bins, density=True, color=color,
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvspan(-1, -r_crit, color="#d03b3b", alpha=0.06)
        ax.axvspan(r_crit, 1, color="#d03b3b", alpha=0.06)
        ax.axvline(-r_crit, color="#d03b3b", linewidth=1, linestyle="--",
                   alpha=0.6)
        ax.axvline(r_crit, color="#d03b3b", linewidth=1, linestyle="--",
                   alpha=0.6)
        ax.set_title(label, fontsize=13, color=INK, loc="left", pad=8)
        ax.text(0.985, 0.86, f"偽有意率 {st['false_sig_rate']*100:.1f}%",
                transform=ax.transAxes, ha="right", fontsize=12.5,
                color=INK, fontweight="bold")
        ax.text(0.985, 0.70, f"|r|の中央値 {st['median_abs_r']:.3f}",
                transform=ax.transAxes, ha="right", fontsize=10.5,
                color=INK2)
        ax.set_ylabel("密度", fontsize=11, color=INK2)
    axes[0].text(r_crit + 0.02, axes[0].get_ylim()[1] * 0.45,
                 "← ここから外は p<0.05\n(|r|>0.139)", fontsize=9.5,
                 color="#d03b3b")
    axes[-1].set_xlabel("独立な2系列間の相関係数 r", fontsize=12, color=INK)
    axes[-1].set_xlim(-1, 1)
    fig.suptitle("完全に独立な2系列1万ペアの相関係数の分布(系列長T=200)",
                 fontsize=13.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(IMG / "1.png", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"fig1 saved (r_crit={r_crit:.4f})")


# ---- 図2: 偽有意率 vs 系列長 ------------------------------------------------
def fig2():
    sw = RES["B_sweep"]
    T = sw["T"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    style_ax(ax)
    ax.axhline(5, color=MUTED, linewidth=1, linestyle=":")
    series = [
        ("rw", "ランダムウォーク", C_RW, "-", "o"),
        ("ar1", "AR(1) φ=0.8", C_AR1, "-", "s"),
        ("rw_diff", "ランダムウォークの差分", C_RW, "--", "^"),
        ("iid", "iidノイズ", C_IID, "-", "D"),
    ]
    for key, label, color, ls, marker in series:
        y = np.array(sw[key]) * 100
        ax.plot(T, y, ls, color=color, linewidth=2, marker=marker,
                markersize=6, label=label,
                markerfacecolor="white" if ls == "--" else color,
                markeredgecolor=color)
    ax.set_xscale("log")
    ax.set_xticks(T)
    ax.set_xticklabels([str(t) for t in T])
    ax.set_ylim(0, 104)
    ax.set_xlabel("系列長 n(対数軸)", fontsize=12, color=INK)
    ax.set_ylabel("偽有意率(p<0.05になった割合, %)", fontsize=11.5,
                  color=INK)
    ax.set_title("独立な2系列が「有意に相関」と誤判定される率 vs 系列長\n"
                 "(各点1万ペア)", fontsize=13, color=INK, loc="left", pad=10)
    # 直接ラベル
    ax.annotate("ランダムウォーク\n96.9%まで上昇", (T[-1], 96.85),
                xytext=(-8, -30), textcoords="offset points", ha="right",
                fontsize=10.5, color=C_RW, fontweight="bold")
    ax.annotate("AR(1) φ=0.8: 36%前後で頭打ち", (T[2], 35.5),
                xytext=(10, 10), textcoords="offset points",
                fontsize=10, color=C_AR1)
    ax.annotate("差分を取ると5%に戻る", (T[4], 4.9),
                xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=10, color=C_RW)
    ax.annotate("iid: ほぼ5%(教科書通り)", (T[0], 5.1),
                xytext=(2, 12), textcoords="offset points",
                fontsize=10, color=C_IID)
    ax.legend(loc="center left", fontsize=9.5, frameon=False,
              bbox_to_anchor=(0.02, 0.55))
    fig.tight_layout()
    fig.savefig(IMG / "2.png", dpi=110, facecolor="white")
    plt.close(fig)
    print("fig2 saved")


# ---- 図3: ヒートマップ(φ × T) --------------------------------------------
def fig3():
    heat = RES["C_heat"]
    rate = np.array(heat["rate"])
    phis = heat["phi"]
    Ts = heat["T"]
    cmap = LinearSegmentedColormap.from_list("refblue", [
        "#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
        "#256abf", "#1c5cab", "#104281", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(Ts)))
    ax.set_xticklabels([str(t) for t in Ts], fontsize=10.5)
    labels = [f"{p:g}" for p in phis]
    labels[-1] = "1.0\n(ランダム\nウォーク)"
    ax.set_yticks(range(len(phis)))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("系列長 n", fontsize=12, color=INK)
    ax.set_ylabel("自己相関の強さ φ", fontsize=12, color=INK)
    ax.set_title("偽有意率ヒートマップ: 自己相関の強さ × 系列長\n"
                 "(独立な2系列がp<0.05になる割合、各セル3,000ペア)",
                 fontsize=12.5, color=INK, loc="left", pad=10)
    for i in range(len(phis)):
        for j in range(len(Ts)):
            v = rate[i, j]
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                    fontsize=10.5,
                    color="white" if v > 0.42 else INK,
                    fontweight="bold" if v > 0.8 else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("偽有意率", fontsize=10.5, color=INK2)
    cbar.ax.tick_params(labelsize=9, colors=INK2)
    cbar.outline.set_visible(False)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    fig.savefig(IMG / "heatmap.png", dpi=110, facecolor="white")
    plt.close(fig)
    print("fig3 saved")


# ---- 図4: 差分を取ると偽相関が消える(縦2段) ------------------------------
def fig4():
    # pair 0 (r=0.977) は「最大rで選んだ」選択効果で差分にも弱い相関が残る
    # (diff p=0.033) ため、代表例としてはpair 1を使う。この点は記事本文で言及。
    x = GIFP["x"][1]
    y = GIFP["y"][1]
    r_level = float(GIFP["r"][1])
    dx, dy = np.diff(x), np.diff(y)
    r_diff, p_diff = stats.pearsonr(dx, dy)
    t = np.arange(len(x))
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.6))
    ax = axes[0]
    style_ax(ax)
    ax.plot(t, x, color=C_IID, linewidth=1.8, label="系列X")
    ax.plot(t, y, color=C_RW, linewidth=1.8, label="系列Y")
    ax.set_title(f"水準のまま: r = {r_level:.3f}(完全に独立なのに)",
                 fontsize=13.5, color=INK, loc="left", pad=8)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_ylabel("値", fontsize=11, color=INK2)
    ax = axes[1]
    style_ax(ax)
    ax.plot(t[1:], dx, color=C_IID, linewidth=0.9, alpha=0.85)
    ax.plot(t[1:], dy, color=C_RW, linewidth=0.9, alpha=0.85)
    ax.set_title(f"差分を取ると: r = {r_diff:.3f} (p = {p_diff:.2f})",
                 fontsize=13.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("時点 t", fontsize=12, color=INK)
    ax.set_ylabel("1階差分", fontsize=11, color=INK2)
    fig.suptitle("同じランダムウォークのペアでも、差分を取れば偽相関は消える",
                 fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(IMG / "3.png", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"fig4 saved (r_level={r_level:.4f}, r_diff={r_diff:.4f}, "
          f"p_diff={p_diff:.4f})")


# ---- GIF: 「それっぽく連動して見える」独立ペアを次々表示 --------------------
def make_gif():
    frames = []
    n_show = 8
    for i in range(n_show):
        x, y, r = GIFP["x"][i], GIFP["y"][i], float(GIFP["r"][i])
        fig, ax = plt.subplots(figsize=(5, 3.1), dpi=80)
        style_ax(ax)
        t = np.arange(len(x))
        ax.plot(t, x, color=C_IID, linewidth=1.6)
        ax.plot(t, y, color=C_RW, linewidth=1.6)
        ax.set_title("完全に独立な2本のランダムウォーク",
                     fontsize=11.5, color=INK, loc="left")
        ax.text(0.98, 0.04, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", fontsize=17, color=INK, fontweight="bold")
        ax.text(0.02, 0.04, f"ペア {i+1}/{n_show}", transform=ax.transAxes,
                fontsize=9, color=MUTED)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE,
                                              colors=64))
    frames[0].save(IMG / "pairs.gif", save_all=True,
                   append_images=frames[1:], duration=1500, loop=0,
                   optimize=True)
    size = (IMG / "pairs.gif").stat().st_size
    print(f"gif saved: {size/1024:.0f} KB, {n_show} frames")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    make_gif()
