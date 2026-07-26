"""BPE(Byte Pair Encoding)を一から実装し、日本語がなぜトークン代が高いのかを測る。
(1) おもちゃコーパスでマージ過程を追う
(2) 自分のブログ49記事(日本語)と英語テキストで、語彙サイズ vs 圧縮率を比較
(3) UTF-8バイト数の非対称性を確認 (日本語3バイト/字 vs 英語1バイト/字)
(4) 実際のトークナイザ(tiktoken系の挙動)と近い傾向が出るか"""
import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

# ---- BPE本体 ----
def get_stats(ids):
    counts = collections.Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text, num_merges, record_every=None):
    ids = list(text.encode("utf-8"))
    orig_len = len(ids)
    vocab = {i: bytes([i]) for i in range(256)}
    merges, history = {}, []
    for k in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair, freq = stats.most_common(1)[0]
        new_id = 256 + k
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if record_every and (k + 1) % record_every == 0:
            history.append((k + 1, len(ids), orig_len / len(ids)))
    return ids, merges, vocab, history, orig_len


# ---- (1) おもちゃコーパスでマージの中身を見せる ----
toy = "低い 低い 高い 高い 高い 高低差 高低差 " * 40
ids, merges, vocab, _, orig = train_bpe(toy, 12)
print("=== おもちゃコーパスで最初の12マージ ===")
merge_log = []
for (a, b), nid in merges.items():
    try:
        s = vocab[nid].decode("utf-8")
    except UnicodeDecodeError:
        s = repr(vocab[nid])
    merge_log.append((nid - 256 + 1, s))
    print(f"  マージ{nid-256+1:2d}: -> {s!r}")

# ---- (2) 日本語(自ブログ) vs 英語(不思議の国のアリス) ----
ja = open("corpus.txt", encoding="utf-8").read()
en_raw = open("alice.txt", encoding="utf-8").read()
en_raw = en_raw[en_raw.index("CHAPTER I."):]  # ヘッダ除去
en = en_raw[: len(ja)]  # 文字数を日本語と揃える
print("")
print(f"日本語コーパス: {len(ja):,}文字 / {len(ja.encode('utf-8')):,}バイト "
      f"({len(ja.encode('utf-8'))/len(ja):.2f}バイト/文字)")
print(f"英語コーパス  : {len(en):,}文字 / {len(en.encode('utf-8')):,}バイト "
      f"({len(en.encode('utf-8'))/len(en):.2f}バイト/文字)")

VOCABS = [0, 100, 300, 600, 1000, 1500, 2000, 3000]
res = {}
for name, text in [("日本語(自ブログ49記事)", ja), ("英語(アリス)", en)]:
    chars = len(text)
    row = []
    prev = None
    for v in VOCABS:
        if v == 0:
            ntok = len(text.encode("utf-8"))
        else:
            ids, _, _, _, _ = train_bpe(text, v)
            ntok = len(ids)
        row.append((v, ntok, chars / ntok))
        print(f"  {name} vocab+{v:5d}: {ntok:7,}トークン  {chars/ntok:.2f}文字/トークン")
    res[name] = row

json.dump(
    {k: [[int(a), int(b), float(c)] for a, b, c in v] for k, v in res.items()},
    open("results.json", "w", encoding="utf-8"), ensure_ascii=False,
)

# ---- 図1: 語彙サイズ vs 圧縮率 ----
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
cols = {"日本語(自ブログ49記事)": "#e4572e", "英語(アリス)": "#2e6ee4"}
for name, row in res.items():
    vs = [r[0] for r in row]
    cpt = [r[2] for r in row]
    axes[0].plot(vs, cpt, "-o", ms=5, color=cols[name], label=name)
axes[0].set_xlabel("追加した語彙数 (マージ回数)")
axes[0].set_ylabel("1トークンあたりの文字数 (高いほど得)")
axes[0].set_title("語彙を増やすほど「1トークンで運べる文字数」が増える")
axes[0].legend()
axes[0].grid(alpha=0.3)
ja_row, en_row = res["日本語(自ブログ49記事)"], res["英語(アリス)"]
axes[0].annotate(
    f"語彙なし(バイト単位)では\n日本語は1トークン={ja_row[0][2]:.2f}文字",
    xy=(0, ja_row[0][2]), xytext=(300, 0.55), fontsize=11,
    arrowprops=dict(arrowstyle="->", lw=1.5))

# 比率
ratio = [en_row[i][2] / ja_row[i][2] for i in range(len(VOCABS))]
axes[1].plot(VOCABS, ratio, "-o", ms=5, color="#333")
axes[1].axhline(1, color="#888", ls="--")
axes[1].set_xlabel("追加した語彙数")
axes[1].set_ylabel("英語の効率 ÷ 日本語の効率")
axes[1].set_title("同じ文字数なら日本語は何倍のトークンを食うか")
axes[1].annotate(f"語彙なしでは英語が{ratio[0]:.1f}倍効率的",
                 xy=(0, ratio[0]), xytext=(500, ratio[0] * 0.95), fontsize=11,
                 arrowprops=dict(arrowstyle="->", lw=1.5))
axes[1].annotate(f"語彙3000でも{ratio[-1]:.1f}倍の差が残る",
                 xy=(3000, ratio[-1]), xytext=(1200, ratio[-1] * 1.15), fontsize=11,
                 arrowprops=dict(arrowstyle="->", lw=1.5))
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig2_vocab.png", dpi=110, bbox_inches="tight")
print("saved fig2_vocab.png")

# ---- 図2: バイト数の非対称 ----
samples = [("あ", "a"), ("日本語", "abc"), ("拡散モデル", "diffusion")]
fig, ax = plt.subplots(figsize=(8.4, 4.0))
labels, ja_bytes, en_bytes = [], [], []
for j, e in samples:
    labels.append(f"「{j}」 vs \"{e}\"")
    ja_bytes.append(len(j.encode("utf-8")))
    en_bytes.append(len(e.encode("utf-8")))
x = np.arange(len(labels))
ax.barh(x + 0.18, ja_bytes, 0.36, color="#e4572e", label="日本語のUTF-8バイト数")
ax.barh(x - 0.18, en_bytes, 0.36, color="#2e6ee4", label="英語のUTF-8バイト数")
for i, (jb, eb) in enumerate(zip(ja_bytes, en_bytes)):
    ax.text(jb + 0.3, i + 0.18, f"{jb}B", va="center", fontsize=11)
    ax.text(eb + 0.3, i - 0.18, f"{eb}B", va="center", fontsize=11)
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("UTF-8でのバイト数")
ax.set_title("出発点からハンデ: 日本語1文字=3バイト、英語1文字=1バイト")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("fig1_bytes.png", dpi=110, bbox_inches="tight")
print("saved fig1_bytes.png")
json.dump(merge_log, open("merge_log.json", "w", encoding="utf-8"), ensure_ascii=False)
