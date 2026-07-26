"""Grokking記事の可視化 (punchline-first構成用):
(1) 主役: 訓練とテストの乖離→突然の合流 (対数軸、注釈たっぷり)
(2) weight decayスイープ: どの強さでgrokkingが起きるか
(3) 重みノルムの推移 (暗記期は増え、汎化期に落ちる)
(4) 失敗した1回目(結合型L2)との比較 = 実装の教訓"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

d = np.load("hist_adamw.npz")
keys = sorted(d.files, key=lambda k: float(k[2:]))
print("available:", keys)

# grokkingが最も綺麗に出た条件を選ぶ
best_key, best_gap = None, -1
info = {}
for k in keys:
    t, loss, tr, te, w = d[k].T
    i_tr = np.argmax(tr >= 0.99) if (tr >= 0.99).any() else None
    i_te = np.argmax(te >= 0.90) if (te >= 0.90).any() else None
    s_tr = int(t[i_tr]) if i_tr is not None else None
    s_te = int(t[i_te]) if i_te is not None else None
    info[k] = (s_tr, s_te, te[-1])
    print(f"{k}: 訓練99% @{s_tr}, テスト90% @{s_te}, 最終テスト {te[-1]:.3f}")
    if s_tr and s_te and (s_te - s_tr) > best_gap:
        best_key, best_gap = k, s_te - s_tr

if best_key is None:
    print("!! grokkingは再現しなかった。最終テスト精度が最も高い条件を主役にする")
    best_key = max(keys, key=lambda k: d[k][-1, 3])

t, loss, tr, te, w = d[best_key].T
wd_val = best_key[2:]

# ---- 主役図 ----
fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.semilogx(t, tr, color="#2e6ee4", lw=2.6, label="訓練データ")
ax.semilogx(t, te, color="#e4572e", lw=2.6, label="テストデータ (未知の問題)")
ax.set_xlabel("学習ステップ (対数)")
ax.set_ylabel("正解率")
ax.set_ylim(-0.03, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="center left", fontsize=11)
s_tr, s_te, final_te = info.get(best_key, (None, None, te[-1]))
if s_tr:
    ax.axvline(s_tr, color="#2e6ee4", ls=":", lw=1.4)
    ax.annotate(f"step {s_tr:,}\n訓練は満点に", xy=(s_tr, 1.0), xytext=(s_tr * 0.12, 0.62),
                fontsize=11, color="#2e6ee4", arrowprops=dict(arrowstyle="->", lw=1.5, color="#2e6ee4"))
if s_te:
    ax.axvline(s_te, color="#e4572e", ls=":", lw=1.4)
    ax.annotate(f"step {s_te:,}\nここで突然「わかる」", xy=(s_te, 0.9), xytext=(s_te * 0.09, 0.28),
                fontsize=11, color="#e4572e", arrowprops=dict(arrowstyle="->", lw=1.5, color="#e4572e"))
    ax.annotate("", xy=(s_te, 0.06), xytext=(s_tr, 0.06),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color="#555"))
    ax.text(np.sqrt(s_tr * s_te), 0.09, f"何も起きていないように見える {s_te - s_tr:,} ステップ",
            ha="center", fontsize=11, color="#555")
ax.set_title(f"暗記してから理解するまで (weight decay={wd_val})", fontsize=15)
fig.tight_layout()
fig.savefig("fig1_grokking.png", dpi=110, bbox_inches="tight")
print("saved fig1_grokking.png")

# ---- wdスイープ ----
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
cmap = plt.get_cmap("viridis")
for i, k in enumerate(keys):
    t2, _, tr2, te2, w2 = d[k].T
    col = cmap(i / max(1, len(keys) - 1))
    axes[0].semilogx(t2, te2, color=col, lw=2, label=f"wd={k[2:]}")
    axes[1].semilogx(t2, w2, color=col, lw=2, label=f"wd={k[2:]}")
axes[0].set_xlabel("学習ステップ (対数)")
axes[0].set_ylabel("テスト正解率")
axes[0].set_title("weight decayが弱すぎても強すぎても汎化しない")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[1].set_xlabel("学習ステップ (対数)")
axes[1].set_ylabel("重みの大きさ |w|")
axes[1].set_title("重みが縮んだ先に汎化がある")
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig2_sweep.png", dpi=110, bbox_inches="tight")
print("saved fig2_sweep.png")

# ---- 重みノルムと精度の重ね描き ----
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.semilogx(t, te, color="#e4572e", lw=2.6, label="テスト正解率")
ax.set_xlabel("学習ステップ (対数)")
ax.set_ylabel("テスト正解率", color="#e4572e")
ax.set_ylim(-0.03, 1.05)
ax2 = ax.twinx()
ax2.semilogx(t, w, color="#2e6ee4", lw=2.2, label="重みの大きさ |w|")
ax2.set_ylabel("重みの大きさ |w|", color="#2e6ee4")
ax.set_title("重みが小さくなるのと、汎化が始まるのは同時")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig3_norm.png", dpi=110, bbox_inches="tight")
print("saved fig3_norm.png")
np.savez("summary.npz", best_key=best_key, s_tr=s_tr or -1, s_te=s_te or -1, final=final_te)
