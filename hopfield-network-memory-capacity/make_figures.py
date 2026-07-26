# -*- coding: utf-8 -*-
"""実験結果から記事用の図とGIFを生成する。

出力:
  fig/1.png  記憶させた文字パターンの例
  fig/2.png  想起成功率 vs 記憶枚数(3条件)
  fig/3.png  文字10枚(ヘブ則)で全手がかりが同じ幽霊アトラクタに収束する様子
  fig/4.png  ランダムパターンの成功時/失敗時のオーバーラップ分布
  fig/5.png  (記憶枚数 x ノイズ量) 想起成功率ヒートマップ
  fig/6.png  射影ルール容量超過時の「混ざった偽記憶」
  fig/recall.gif  想起ダイナミクス(成功 vs 幽霊化)の対比
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import ImageFont

from hopfield_common import (make_char_patterns, make_random_patterns,
                             render_char, train, train_projection, add_noise,
                             occlude, recall, overlap, N, N_SIDE, FONT_PATH)

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("fig", exist_ok=True)

C_BLUE = "#2563eb"
C_RED = "#dc2626"
C_GREEN = "#059669"

char_pats, chars = make_char_pats = make_char_patterns(300)
rand_pats = make_random_patterns(300, seed=0)
rng = np.random.default_rng(123)


def show_pat(ax, p, title=None, fontsize=11):
    ax.imshow(p.reshape(N_SIDE, N_SIDE), cmap="gray_r", vmin=-1, vmax=1,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize)


# ---------------------------------------------------------------- fig 1
def fig1():
    sample_idx = [0, 4, 9, 20, 40, 60, 85, 100, 120, 150, 170, 200, 220, 250,
                  270, 290]
    fig, axes = plt.subplots(2, 8, figsize=(9.6, 3.4))
    for ax, i in zip(axes.ravel(), sample_idx):
        show_pat(ax, char_pats[i], f"「{chars[i]}」", fontsize=11)
    fig.suptitle("記憶させる32×32の白黒パターンの例(全300枚から抜粋)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=2.5)
    fig.savefig("fig/1.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig2():
    d = np.load("capacity_results.npz")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    specs = [
        ("random_hebb", C_BLUE, "ランダムパターン+ヘブ則"),
        ("chars_projection", C_GREEN, "文字パターン+射影ルール"),
        ("chars_hebb", C_RED, "文字パターン+ヘブ則"),
    ]
    for key, c, label in specs:
        P = d[f"{key}_P"]
        r = d[f"{key}_rate"]
        ax.plot(P, r, "-o", color=c, label=label, lw=2, ms=4.5)
    ax.axvline(141.3, color="gray", ls="--", lw=1.5)
    ax.text(145, 0.55, "理論容量 0.138N ≒ 141枚", color="gray", fontsize=11,
            rotation=90, va="center")
    ax.set_xlabel("記憶させた枚数 P", fontsize=13)
    ax.set_ylabel("想起成功率(ノイズ10%の手がかり)", fontsize=13)
    ax.set_title("Hopfieldネットワーク(1,024ニューロン)の想起成功率", fontsize=14)
    ax.legend(fontsize=11, loc="center right")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("fig/2.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
DIVERSE = list("アヨ山0ネ8キツ日ミ")


def fig3():
    font = ImageFont.truetype(FONT_PATH, 30)
    dpats = np.stack([render_char(c, font) for c in DIVERSE])
    W = train(dpats)
    finals = []
    cues = []
    for i in range(5):
        cue = add_noise(dpats[i], 0.25, rng)
        s, _, _ = recall(W, cue, rng)
        cues.append(cue)
        finals.append(s)
    # 全収束先が一致しているか確認
    same = all(np.array_equal(finals[0], f) for f in finals[1:])
    ink = (finals[0] > 0).mean()
    ms = [overlap(finals[0], p) for p in dpats]
    print(f"[fig3] all-identical={same}, ghost ink={ink:.3f}, "
          f"overlaps with stored={np.round(ms, 2)}")

    fig, axes = plt.subplots(2, 5, figsize=(8.0, 4.3))
    for i in range(5):
        show_pat(axes[0, i], cues[i], f"手がかり「{DIVERSE[i]}」", fontsize=11)
        show_pat(axes[1, i], finals[i], "収束先", fontsize=11)
    fig.suptitle("形の違う10文字を記憶させたヘブ則ネット:\nどの文字を思い出させても、ほぼ同じ「幽霊」に収束する",
                 fontsize=13)
    fig.subplots_adjust(top=0.76, bottom=0.03, left=0.03, right=0.97,
                        hspace=0.35)
    fig.savefig("fig/3.png", dpi=150)
    plt.close(fig)
    return same, ink, ms


# ---------------------------------------------------------------- fig 4
def fig4():
    # 成功例: P=100, 失敗例: P=200
    out = {}
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4))
    for ax, P, label in [(axes[0], 100, "記憶100枚: 成功"),
                         (axes[1], 200, "記憶200枚: 失敗(偽アトラクタ)")]:
        W = train(rand_pats[:P])
        # 失敗例が出るまで(P=100は1回で成功するはず)
        for attempt in range(50):
            idx = int(rng.integers(P))
            cue = add_noise(rand_pats[idx], 0.10, rng)
            s, _, _ = recall(W, cue, rng)
            m = overlap(s, rand_pats[idx])
            if (P == 100 and m >= 0.95) or (P == 200 and m < 0.9):
                break
        ms = rand_pats[:P] @ s / N
        order = np.argsort(-np.abs(ms))
        top3 = order[:3]
        mix = np.sign(sum(np.sign(ms[k]) * rand_pats[k] for k in top3))
        mix[mix == 0] = 1
        agree = float((mix == s).mean())
        out[P] = dict(idx=idx, m=m, top3=[(int(k), float(ms[k])) for k in top3],
                      mix_agree=agree)
        print(f"[fig4] P={P}: cue idx={idx}, m_cue={m:.3f}, "
              f"top3={[(int(k), round(float(ms[k]), 2)) for k in top3]}, "
              f"top3混合との一致率={agree:.3f}")
        colors = [C_RED if k == idx else "#94a3b8" for k in range(P)]
        ax.bar(np.arange(P), np.abs(ms), color=colors, width=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("|オーバーラップ|", fontsize=12)
        ax.set_title(f"{label}(赤=思い出させたかった記憶)", fontsize=13)
        ax.grid(alpha=0.25, axis="y")
    axes[1].set_xlabel("記憶のインデックス", fontsize=12)
    fig.suptitle("収束した状態は各記憶とどれだけ似ているか(ランダムパターン+ヘブ則)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("fig/4.png", dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- fig 5
def fig5():
    d = np.load("heatmap_results.npz")
    grid, P_LIST, NOISE_LIST = d["grid"], d["P_LIST"], d["NOISE_LIST"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(P_LIST)))
    ax.set_xticklabels(P_LIST, fontsize=10)
    ax.set_yticks(range(len(NOISE_LIST)))
    ax.set_yticklabels([f"{int(z*100)}%" for z in NOISE_LIST], fontsize=10)
    ax.set_xlabel("記憶させた枚数 P", fontsize=13)
    ax.set_ylabel("手がかりのノイズ量(反転画素の割合)", fontsize=13)
    ax.set_title("想起成功率ヒートマップ(ランダムパターン+ヘブ則)", fontsize=14)
    # 理論容量の位置
    x141 = np.interp(141.3, P_LIST, np.arange(len(P_LIST)))
    ax.axvline(x141, color=C_RED, ls="--", lw=2)
    ax.text(x141 + 0.15, len(NOISE_LIST) - 1.3, "0.138N", color=C_RED,
            fontsize=12)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("想起成功率", fontsize=12)
    fig.tight_layout()
    fig.savefig("fig/5.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 6
def fig6():
    P = 300
    rng6 = np.random.default_rng(777)
    W = train_projection(char_pats[:P])
    examples = []
    for attempt in range(200):
        idx = int(rng6.integers(P))
        cue = add_noise(char_pats[idx], 0.25, rng6)
        s, _, _ = recall(W, cue, rng6)
        m = overlap(s, char_pats[idx])
        if m < 0.9:
            ms = char_pats[:P] @ s / N
            order = np.argsort(-np.abs(ms))
            examples.append((idx, cue, s, ms, order))
            if len(examples) == 2:
                break
    fig, axes = plt.subplots(2, 5, figsize=(9.2, 5.0))
    for r, (idx, cue, s, ms, order) in enumerate(examples):
        show_pat(axes[r, 0], cue, f"手がかり「{chars[idx]}」+ノイズ", fontsize=10)
        show_pat(axes[r, 1], s, "収束した状態", fontsize=10)
        for k in range(3):
            j = order[k]
            show_pat(axes[r, 2 + k],
                     char_pats[j] * np.sign(ms[j]),
                     f"近い記憶{k+1}位「{chars[j]}」\nm={ms[j]:+.2f}", fontsize=9)
        print(f"[fig6] cue={chars[idx]}, m_cue={overlap(s, char_pats[idx]):.3f}, "
              f"top3={[(chars[order[k]], round(float(ms[order[k]]), 2)) for k in range(3)]}")
    fig.suptitle("容量超過(射影ルール・300文字)で現れた「混ざった偽記憶」\n"
                 "収束状態は、どの記憶とも一致しない複数記憶の混合になる", fontsize=13)
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.03, right=0.97,
                        hspace=0.45)
    fig.savefig("fig/6.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- GIF
def make_gif():
    font = ImageFont.truetype(FONT_PATH, 30)
    dpats = np.stack([render_char(c, font) for c in DIVERSE])

    cue = add_noise(char_pats[0], 0.25, np.random.default_rng(2024))  # 「ア」

    W_ok = train_projection(char_pats[:100])   # 成功する側: 射影・100文字
    W_ng = train(dpats)                        # 幽霊化する側: ヘブ則・10文字

    r1 = np.random.default_rng(5)
    r2 = np.random.default_rng(5)
    s_ok, _, snaps_ok = recall(W_ok, cue, r1, record_every=64)
    s_ng, _, snaps_ng = recall(W_ng, cue, r2, record_every=64)
    m_ok = overlap(s_ok, char_pats[0])
    m_ng = overlap(s_ng, char_pats[0])
    print(f"[gif] success m={m_ok:.3f} ({len(snaps_ok)} snaps), "
          f"ghost m={m_ng:.3f} ({len(snaps_ng)} snaps)")

    n = max(len(snaps_ok), len(snaps_ng))
    pad = 10  # 最後の状態で静止
    frames = n + pad

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 3.4))
    fig.suptitle("同じ手がかり「ア」(ノイズ25%)からの想起", fontsize=12)
    ims = []
    for ax, title in [(axes[0], "射影ルール\n100文字を記憶"),
                      (axes[1], "ヘブ則\n10文字を記憶")]:
        im = ax.imshow(np.zeros((N_SIDE, N_SIDE)), cmap="gray_r", vmin=-1,
                       vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ims.append(im)
    step_txt = fig.text(0.5, 0.03, "", ha="center", fontsize=10)

    def update(f):
        k = min(f, n - 1)
        a = snaps_ok[min(k, len(snaps_ok) - 1)]
        b = snaps_ng[min(k, len(snaps_ng) - 1)]
        ims[0].set_data(a.reshape(N_SIDE, N_SIDE))
        ims[1].set_data(b.reshape(N_SIDE, N_SIDE))
        if f >= n - 1:
            step_txt.set_text(
                f"収束: 左は「ア」に復元 (m={m_ok:.2f}) / 右は幽霊 (m={m_ng:.2f})")
        else:
            step_txt.set_text(f"更新 {k * 64} ニューロン")
        return ims + [step_txt]

    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    ani = FuncAnimation(fig, update, frames=frames)
    ani.save("fig/recall.gif", writer=PillowWriter(fps=7), dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- 射影ルール・強ノイズ(25%)の統計 (本文用)
def projection_hard_noise_stats():
    for P in [100, 150, 200, 250, 300]:
        W = train_projection(char_pats[:P])
        ok = 0
        T = 30
        ms = []
        for t in range(T):
            idx = int(rng.integers(P))
            cue = add_noise(char_pats[idx], 0.25, rng)
            s, _, _ = recall(W, cue, rng)
            m = overlap(s, char_pats[idx])
            ms.append(m)
            if m >= 0.95:
                ok += 1
        print(f"[proj-25%noise] P={P}: success {ok}/{T}, mean m={np.mean(ms):.3f}")


# ---------------------------------------------------------------- 欠損手がかりの確認 (記事の本文用の数値)
def occlusion_stats():
    W = train_projection(char_pats[:100])
    ok = 0
    T = 30
    for t in range(T):
        idx = int(rng.integers(100))
        cue = occlude(char_pats[idx], 0.5)
        s, _, _ = recall(W, cue, rng)
        if overlap(s, char_pats[idx]) >= 0.95:
            ok += 1
    print(f"[occlusion] projection-100, 下半分欠損: {ok}/{T}")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    make_gif()
    projection_hard_noise_stats()
    occlusion_stats()
    print("all figures done")
