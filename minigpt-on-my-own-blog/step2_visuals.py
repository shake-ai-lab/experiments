"""ミニGPT記事の可視化:
(1) 学習が進むにつれ生成文がまともになる過程 (テキストの進化を見せる主役ビジュアル)
(2) 損失曲線
(3) attention可視化 (どの文字がどこを見ているか)
(4) 温度による生成の違い"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

src = open("step1_train.py", encoding="utf-8").read().split("if __name__")[0]
exec(src)

d = np.load("minigpt_model.npz")
for k in P:
    if k in d:
        P[k] = d[k]
losses = d["losses"]
samples = json.load(open("samples.json", encoding="utf-8"))

# ---- (2) 損失曲線 ----
fig, ax = plt.subplots(figsize=(7.6, 4.2))
w = 50
ax.plot(losses, color="#e4572e", alpha=0.25, lw=1)
ax.plot(np.arange(w - 1, len(losses)), np.convolve(losses, np.ones(w) / w, mode="valid"),
        color="#e4572e", lw=2.4)
ax.set_xlabel("学習ステップ")
ax.set_ylabel("損失 (次の1文字の予測)")
ax.set_title(f"損失 {losses[0]:.2f} → {losses[-1000:].mean():.2f}")
ax.annotate(f"完全なランダムなら\nln(語彙{V}) = {np.log(V):.2f}",
            xy=(0, np.log(V)), xytext=(len(losses) * 0.25, np.log(V) * 0.85),
            fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.5))
ax.axhline(np.log(V), color="#888", ls="--", lw=1)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig1_loss.png", dpi=110, bbox_inches="tight")
print("saved fig1_loss.png")

# ---- (1) 生成文の進化 (テキストパネル) ----
steps = sorted(int(k) for k in samples)
fig, axes = plt.subplots(len(steps), 1, figsize=(11.5, 1.15 * len(steps)))
for ax, st in zip(axes, steps):
    ax.axis("off")
    txt = samples[str(st)][:78].replace("\n", " ")
    ax.text(0.005, 0.72, f"step {st:>5,}", fontsize=11, color="#e4572e", family="Meiryo")
    ax.text(0.005, 0.22, txt, fontsize=12.5, family="Meiryo")
fig.suptitle("学習が進むと、生成文はどう変わるか (先頭78文字)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig2_evolution.png", dpi=110, bbox_inches="tight")
print("saved fig2_evolution.png")

# ---- (4) 温度スイープ ----
temps = [0.4, 0.8, 1.2]
gen_by_temp = {}
for tp in temps:
    gen_by_temp[tp] = generate(220, temp=tp, seed_text="この実験では", r=np.random.default_rng(3))
    print(f"temp {tp}: {gen_by_temp[tp][:60]}")
json.dump(gen_by_temp, open("temp_samples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- (3) attention可視化 ----
probe = "拡散モデルを実装した結果、"
ids = np.array([[stoi.get(c, 0) for c in probe]])
B, T = ids.shape
x = P["emb"][ids] + P["pos"][:T]
atts = []
for l in range(NL):
    xn, _ = layer_norm(x)
    q = (xn @ P[f"Wq{l}"]).reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    k = (xn @ P[f"Wk{l}"]).reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    v = (xn @ P[f"Wv{l}"]).reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(HD)
    att = np.where(mask[:T, :T] == 0, -1e9, att)
    a = softmax(att)
    atts.append(a[0])
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    x = x + o @ P[f"Wo{l}"]
    xn2, _ = layer_norm(x)
    h = np.maximum(xn2 @ P[f"W1{l}"] + P[f"b1{l}"], 0)
    x = x + h @ P[f"W2{l}"] + P[f"b2{l}"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
show = [(0, 0), (1, 0), (1, 2)]
for ax, (l, hd) in zip(axes, show):
    im = ax.imshow(atts[l][hd], cmap="magma", vmin=0)
    ax.set_xticks(range(T))
    ax.set_xticklabels(list(probe), fontsize=10)
    ax.set_yticks(range(T))
    ax.set_yticklabels(list(probe), fontsize=10)
    ax.set_title(f"層{l+1} ヘッド{hd+1}", fontsize=12)
axes[0].set_ylabel("この文字が (query)")
axes[1].set_xlabel("どこを見ているか (key)")
fig.suptitle("学習後のattention: 直前の文字への注目と、離れた語への注目が分業している", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig3_attention.png", dpi=110, bbox_inches="tight")
print("saved fig3_attention.png")
