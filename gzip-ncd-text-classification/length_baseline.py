"""交絡チェック: 「本文の長さ」だけでタスクA(実験系vs入門系)を
どこまで分類できるかを測る。NCDやTF-IDFの数字を解釈するための下限線。"""
import json
from pathlib import Path
import numpy as np

EXP_DIR = Path(r"C:\tmp\gzip-ncd-text-classification-experiment")
records = json.loads((EXP_DIR / "corpus.json").read_text(encoding="utf-8"))


def track(r):
    tags = set(r["tags"])
    if tags & {"実践記", "機械学習"}:
        return "実験系"
    if tags & {"Claude入門", "比較"}:
        return "入門系"
    return "その他"


data = [(len(r["body"]), track(r)) for r in records if track(r) != "その他"]
lens = np.array([d[0] for d in data], dtype=float)
labs = [d[1] for d in data]
m = len(data)

# 1NN on |len_i - len_j| (leave-one-out)
correct = 0
for i in range(m):
    j = min((j for j in range(m) if j != i), key=lambda j: abs(lens[i] - lens[j]))
    correct += labs[j] == labs[i]
print(f"長さのみ1NN (LOO): {correct / m:.4f}  (n={m})")

# best single threshold (LOO: threshold fit on the other 64 each time)
correct = 0
for i in range(m):
    tr = [j for j in range(m) if j != i]
    # 実験系のほうが長い前提で、閾値を全候補から選ぶ
    cands = sorted(set(lens[tr]))
    best_acc, best_t = -1, None
    for t in cands:
        acc = np.mean([
            (labs[j] == "実験系") == (lens[j] >= t) for j in tr
        ])
        if acc > best_acc:
            best_acc, best_t = acc, t
    pred = "実験系" if lens[i] >= best_t else "入門系"
    correct += pred == labs[i]
print(f"長さの閾値のみ (LOO): {correct / m:.4f}")

by = {}
for l, t in data:
    by.setdefault(t, []).append(l)
for t, v in by.items():
    print(t, f"n={len(v)} mean={np.mean(v):.0f} min={min(v)} max={max(v)}")
