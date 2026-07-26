"""実際のLLMトークナイザ(tiktoken o200k_base = GPT-4o系)で、
日本語と英語の実料金差を測る。自前BPEの結論が本物でも成り立つかの検証。"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

enc = tiktoken.get_encoding("o200k_base")  # GPT-4o系
enc_old = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4系

ja = open("corpus.txt", encoding="utf-8").read()
en_raw = open("alice.txt", encoding="utf-8").read()
en = en_raw[en_raw.index("CHAPTER I.") :][: len(ja)]

print("=== 実トークナイザでの比較 (同じ文字数) ===")
rows = []
for name, text in [("日本語(自ブログ)", ja), ("英語(アリス)", en)]:
    n_new = len(enc.encode(text))
    n_old = len(enc_old.encode(text))
    rows.append((name, len(text), n_new, n_old, len(text) / n_new, len(text) / n_old))
    print(f"{name}: {len(text):,}文字 -> o200k {n_new:,}tok ({len(text)/n_new:.2f}字/tok), "
          f"cl100k {n_old:,}tok ({len(text)/n_old:.2f}字/tok)")

ratio_new = rows[1][4] / rows[0][4]
ratio_old = rows[1][5] / rows[0][5]
print(f"英語/日本語の効率比: o200k {ratio_new:.2f}倍, cl100k {ratio_old:.2f}倍")

# 同じ内容の対訳ペアでの比較 (より公平)
pairs = [
    ("こんにちは、世界。", "Hello, world."),
    ("拡散モデルをnumpyで一から実装しました。",
     "I implemented a diffusion model from scratch in numpy."),
    ("この記事では、機械学習の基礎を実験しながら確かめていきます。",
     "In this article, we verify the fundamentals of machine learning through experiments."),
    ("要約統計量が同じでも、散布図は全く違う形になることがあります。",
     "Even when the summary statistics are identical, the scatter plots can look completely different."),
]
print("\n=== 同じ意味の対訳での比較 (o200k_base) ===")
pair_rows = []
for j, e in pairs:
    nj, ne = len(enc.encode(j)), len(enc.encode(e))
    pair_rows.append((j, e, nj, ne))
    print(f"  日{nj:3d}tok / 英{ne:3d}tok  ({nj/ne:.2f}倍)  「{j[:20]}...」")
avg = np.mean([r[2] / r[3] for r in pair_rows])
print(f"対訳4組の平均: 日本語は英語の {avg:.2f}倍のトークンを消費")

# 料金換算 (Claude Sonnet 4.5相当の入力$3/M tokensで試算)
PRICE = 3.0  # USD per 1M input tokens
for name, chars, n_new, *_ in rows:
    print(f"{name} 12万字の入力コスト: ${n_new/1e6*PRICE:.4f}")

json.dump(
    dict(rows=[[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows],
         pairs=[[p[0], p[1], p[2], p[3]] for p in pair_rows],
         ratio_new=ratio_new, ratio_old=ratio_old, avg_pair=float(avg)),
    open("real_tok.json", "w", encoding="utf-8"), ensure_ascii=False,
)

# ---- 図: 対訳ペアのトークン数比較 ----
fig, ax = plt.subplots(figsize=(9.6, 4.4))
labels = [f"例{i+1}" for i in range(len(pair_rows))]
x = np.arange(len(labels))
nj = [r[2] for r in pair_rows]
ne = [r[3] for r in pair_rows]
ax.bar(x - 0.19, nj, 0.38, color="#e4572e", label="日本語")
ax.bar(x + 0.19, ne, 0.38, color="#2e6ee4", label="英語(同じ意味)")
for i, (a, b) in enumerate(zip(nj, ne)):
    ax.text(i - 0.19, a + 0.6, str(a), ha="center", fontsize=11)
    ax.text(i + 0.19, b + 0.6, str(b), ha="center", fontsize=11)
    ax.text(i, max(a, b) + 3.5, f"{a/b:.1f}倍", ha="center", fontsize=12, color="#333")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("トークン数 (o200k_base)")
ax.set_title(f"同じ内容を伝えるのに、日本語は平均{avg:.1f}倍のトークンを使う", fontsize=14)
ax.legend()
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig("fig3_real_pairs.png", dpi=110, bbox_inches="tight")
print("saved fig3_real_pairs.png")

# ---- 図: 実際のトークン分割を色分けして見せる (長めの文で差を出す) ----
show = [
    (pairs[2][0], "日本語"),
    (pairs[2][1], "英語 (同じ意味)"),
]
fig, axes = plt.subplots(2, 1, figsize=(12, 4.6))
cmap = plt.get_cmap("tab20")
for ax, (text, title) in zip(axes, show):
    toks = [enc.decode([t]) for t in enc.encode(text)]
    ax.axis("off")
    xpos, ypos = 0.005, 0.62
    for i, t in enumerate(toks):
        disp = t.replace(chr(10), "").replace(" ", "␣")
        wide = sum(1.9 if ord(c) > 127 else 1.0 for c in disp)
        w = 0.0135 * max(1.7, wide)
        if xpos + w > 0.99:
            xpos, ypos = 0.005, ypos - 0.42
        ax.add_patch(plt.Rectangle((xpos, ypos - 0.16), w, 0.32, fc=cmap(i % 20),
                                   alpha=0.55, ec="#555", lw=1))
        ax.text(xpos + w / 2, ypos, disp, ha="center", va="center", fontsize=11)
        xpos += w + 0.004
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.set_title(f"{title}: {len(toks)}トークン", fontsize=13, loc="left")
fig.suptitle("同じ内容でも分割の粒度が違う (色=1トークン)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig4_split.png", dpi=110, bbox_inches="tight")
print("saved fig4_split.png")
