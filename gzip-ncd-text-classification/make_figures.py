"""記事用の図を生成する。
1.png: 記事×記事のNCD距離行列ヒートマップ(series/トラック順に並べ替え)
2.png: k=2タイブレークのバグ有無での精度比較(タスクA/B 縦スタック)
3.png: タスクAの手法別精度比較(ベースライン込み)
4.png: 日本語UTF-8の圧縮特性(文字数 vs バイト数)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.edgecolor"] = "#c3c2b7"
plt.rcParams["axes.labelcolor"] = "#0b0b0b"
plt.rcParams["xtick.color"] = "#52514e"
plt.rcParams["ytick.color"] = "#52514e"

EXP_DIR = Path(r"C:\tmp\gzip-ncd-text-classification-experiment")
OUT = Path(r"C:\Users\kouta\project\forsubwork\src\content\blog\images\gzip-ncd-text-classification")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRID = "#e1e0d9"

records = json.loads((EXP_DIR / "corpus.json").read_text(encoding="utf-8"))
results = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))
ncd = np.load(EXP_DIR / "ncd_matrix.npy")
n = len(records)
ncd_sym = (ncd + ncd.T) / 2


def track(r):
    tags = set(r["tags"])
    if tags & {"実践記", "機械学習"}:
        return "実験系"
    if tags & {"Claude入門", "比較"}:
        return "入門系"
    return "その他"


# ---------- 図1: NCDヒートマップ ----------
SERIES_ORDER = [
    ("画像処理の周波数分析", "周波数"),
    ("生成モデルを一から実装する", "生成モデル"),
    ("Self-Attentionを実装する", "Self-Attn"),
    ("多様体学習の実験", "多様体"),
    ("探索と活用のアルゴリズム", "探索と活用"),
    ("最適化のライフレッスン", "最適化"),
]
order = []
blocks = []  # (start, end, label)
pos = 0
for s, short in SERIES_ORDER:
    ii = [i for i, r in enumerate(records) if r["series"] == s]
    order += ii
    blocks.append((pos, pos + len(ii), short))
    pos += len(ii)
for name, pred in [
    ("実験系(単発)", lambda r: r["series"] is None and track(r) == "実験系"),
    ("入門系", lambda r: track(r) == "入門系"),
    ("その他", lambda r: track(r) == "その他"),
]:
    ii = [i for i, r in enumerate(records) if pred(r)]
    order += ii
    blocks.append((pos, pos + len(ii), name))
    pos += len(ii)
assert pos == n

M = ncd_sym[np.ix_(order, order)].copy()
np.fill_diagonal(M, np.nan)  # 対角(自分自身)は表示から外す

# 青の単色ランプ(低NCD=濃い=似ている)
ramp = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#f2f7fd"]
cmap = LinearSegmentedColormap.from_list("blue_seq", ramp)
cmap.set_bad("#fcfcfb")

fig, ax = plt.subplots(figsize=(9.2, 8.4), dpi=150)
im = ax.imshow(M, cmap=cmap, vmin=0.80, vmax=0.98, interpolation="nearest")
for s0, s1, lab in blocks:
    ax.axhline(s0 - 0.5, color="#fcfcfb", lw=1.6)
    ax.axvline(s0 - 0.5, color="#fcfcfb", lw=1.6)
    ax.axhline(s0 - 0.5, color=MUTED, lw=0.7)
    ax.axvline(s0 - 0.5, color=MUTED, lw=0.7)
ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -0.5)
ax.set_xticks([(s0 + s1) / 2 - 0.5 for s0, s1, _ in blocks])
ax.set_xticklabels([lab for _, _, lab in blocks], rotation=45, ha="right", fontsize=11)
ax.set_yticks([(s0 + s1) / 2 - 0.5 for s0, s1, _ in blocks])
ax.set_yticklabels([lab for _, _, lab in blocks], fontsize=11)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("NCD(正規化圧縮距離)  濃い=似ている", fontsize=11)
cb.outline.set_visible(False)
ax.set_title("全73記事 × 73記事のNCD距離行列(series・トラック順)", fontsize=14, pad=12)
fig.tight_layout()
fig.savefig(OUT / "1.png", facecolor="#fcfcfb")
plt.close(fig)
print("1.png saved")

# ---------- 図2: タイブレークバグ比較(縦スタック) ----------
ta, tb = results["task_a"], results["task_b"]
fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), dpi=150)
panels = [
    (
        axes[0],
        "タスクA: 実験系 vs 入門系(65記事・2クラス、破線=多数決81.5%)",
        [
            ("k=2 正しい評価\n(タイはランダム)", ta["ncd_knn_k2_randomtie"], BLUE),
            ("k=2 バグ入り評価\n(実質top-2精度)", ta["ncd_knn_k2_top2bug"], ORANGE),
        ],
        ta["majority_baseline"],
    ),
    (
        axes[1],
        "タスクB: series当て(22記事・6クラス、破線=多数決27.3%)",
        [
            ("k=2 正しい評価\n(タイはランダム)", 0.7727, BLUE),  # k=2タイ→実質1NN側の値は下で上書き
            ("k=2 バグ入り評価\n(実質top-2精度)", tb["ncd_knn_k2_top2bug"], ORANGE),
        ],
        tb["majority_baseline"],
    ),
]
# タスクBの正しい評価はk=1と同じ扱いにせず、実測のk=1値(=タイ時最近傍)を使う
panels[1][2][0] = ("k=2 正しい評価\n(タイは最近傍)", tb["ncd_knn_k1"], BLUE)

for ax, title, bars, base in panels:
    labels = [b[0] for b in bars]
    vals = [b[1] * 100 for b in bars]
    colors = [b[2] for b in bars]
    y = np.arange(len(bars))[::-1]
    ax.barh(y, vals, height=0.55, color=colors, edgecolor="none")
    ax.axvline(base * 100, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    for yi, v in zip(y, vals):
        ax.text(v - 1.0, yi, f"{v:.1f}%", va="center", ha="right",
                fontsize=12, color="#ffffff", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=12.5, loc="left", pad=8)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
axes[1].set_xlabel("leave-one-out 精度(%)", fontsize=11)
fig.suptitle("NCD+kNN(k=2): タイブレークのバグで精度はどれだけ盛られるか", fontsize=13.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "2.png", facecolor="white")
plt.close(fig)
print("2.png saved")

# ---------- 図3: タスクA 手法別比較 ----------
methods = [
    ("多数決(全部「実験系」と答える)", ta["majority_baseline"], MUTED),
    ("本文の文字数だけ(閾値)", 0.8923, MUTED),
    ("TF-IDF+ロジスティック回帰(balanced)", ta["tfidf_logreg_balanced"], ORANGE),
    ("gzip NCD + 1NN", ta["ncd_knn_k1"], BLUE),
    ("gzip NCD + kNN(k=5)", ta["ncd_knn_k5"], BLUE),
    ("TF-IDF + 1NN(コサイン距離)", ta["tfidf_knn_k1"], ORANGE),
]
fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=150)
y = np.arange(len(methods))[::-1]
vals = [m[1] * 100 for m in methods]
ax.barh(y, vals, height=0.6, color=[m[2] for m in methods])
for yi, v in zip(y, vals):
    ax.text(v - 1.0, yi, f"{v:.1f}%", va="center", ha="right",
            fontsize=12, color="#ffffff", fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([m[0] for m in methods], fontsize=11)
ax.set_xlim(0, 100)
ax.set_xlabel("leave-one-out 精度(%)", fontsize=11)
ax.set_title("タスクA(実験系53 vs 入門系12)の手法別精度\n灰=ベースライン / 青=gzip NCD / 橙=TF-IDF", fontsize=13, loc="left", pad=10)
ax.tick_params(length=0)
ax.grid(axis="x", color=GRID, lw=0.7)
ax.set_axisbelow(True)
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "3.png", facecolor="white")
plt.close(fig)
print("3.png saved")

# ---------- 図4: UTF-8圧縮特性 ----------
char_lens = np.array([len(r["body"]) for r in records])
byte_lens = np.array([len(r["body"].encode("utf-8")) for r in records])
import zlib
comp_lens = np.array([len(zlib.compress(r["body"].encode("utf-8"), 9)) for r in records])

fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=150)
ax.scatter(char_lens, byte_lens, s=34, color=BLUE, alpha=0.85, label="UTF-8バイト数(圧縮前)", edgecolors="white", linewidths=0.5)
ax.scatter(char_lens, comp_lens, s=34, color=ORANGE, alpha=0.85, label="zlib圧縮後バイト数", edgecolors="white", linewidths=0.5)
xs = np.array([0, char_lens.max() * 1.05])
b_per_char = float(np.sum(byte_lens) / np.sum(char_lens))
c_per_char = float(np.sum(comp_lens) / np.sum(char_lens))
ax.plot(xs, xs * b_per_char, color=BLUE, lw=1.2, ls=(0, (4, 3)))
ax.plot(xs, xs * c_per_char, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
ax.text(xs[1] * 0.99, xs[1] * b_per_char * 0.99, f"約{b_per_char:.2f}バイト/文字", fontsize=11, color=BLUE, ha="right", va="bottom")
ax.text(xs[1] * 0.99, xs[1] * c_per_char * 1.02, f"約{c_per_char:.2f}バイト/文字", fontsize=11, color=ORANGE, ha="right", va="bottom")
ax.set_xlabel("本文の文字数", fontsize=11)
ax.set_ylabel("バイト数", fontsize=11)
ax.set_title("日本語記事のUTF-8バイト数と圧縮後サイズ(73記事)", fontsize=13, loc="left", pad=10)
ax.legend(fontsize=10.5, frameon=False, loc="upper left")
ax.grid(color=GRID, lw=0.7)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "4.png", facecolor="white")
plt.close(fig)
print("4.png saved")
print(f"bytes/char raw={b_per_char:.3f} compressed={c_per_char:.3f}")
