"""実データ(共通テスト2025本試験)で、偏差値60/65/70を超えた人の実際の割合を測る。"""
import sys
import numpy as np
from core import load_subjects, moments, hensachi_threshold_row, skewness

sys.stdout.reconfigure(encoding="utf-8")
subj = load_subjects()
print(f"科目数 {len(subj)}")
rows = []
for name, d in subj.items():
    n, m, sd = moments(d["counts"])
    # CSV冒頭の平均・標準偏差と自分で計算した値が一致するか
    ok = abs(m - d["mean_csv"]) < 0.01 and abs(sd - d["sd_csv"]) < 0.01 and n == d["n_csv"]
    rows.append((name, d, n, m, sd, ok))
    if not ok:
        print(f"  [不一致] {name}: 自計算 n={n} mean={m:.2f} sd={sd:.2f} / CSV n={d['n_csv']} mean={d['mean_csv']} sd={d['sd_csv']}")
print("CSVの平均・SDと自計算が一致した科目:", sum(r[5] for r in rows), "/", len(rows))
print()
print(f"{'科目':<22}{'満点':>4}{'人数':>8}{'平均':>7}{'SD':>7}{'歪度':>6} | "
      f"{'60:実':>6}{'正':>6} | {'65:実':>6}{'正':>6} | {'70:実':>6}{'正':>6}")
for name, d, n, m, sd, ok in rows:
    if n < 1000:
        continue
    r = hensachi_threshold_row(d["counts"])
    print(f"{name:<22}{d['full']:>4}{n:>8}{m:>7.2f}{sd:>7.2f}{skewness(d['counts']):>6.2f} | "
          f"{r[60]['actual']*100:>5.1f}%{r[60]['normal']*100:>5.1f}% | "
          f"{r[65]['actual']*100:>5.1f}%{r[65]['normal']*100:>5.1f}% | "
          f"{r[70]['actual']*100:>5.1f}%{r[70]['normal']*100:>5.1f}%")

# ---- 集計: 受験者1万人以上の科目だけ(教科の合算表も含む。重複を避けたい場合は下の個別表示を参照)
big = [(nm, d) for nm, d in subj.items() if d["counts"].sum() >= 10000]
print()
print(f"受験者1万人以上: {len(big)} 表")
for h in (60, 65, 70):
    reach = 0
    over = 0
    ratios = []
    for nm, d in big:
        r = hensachi_threshold_row(d["counts"])[h]
        if r["k"] > d["full"]:
            pass
        else:
            reach += 1
        ratios.append(r["actual"] / r["normal"])
    print(f"偏差値{h}: 満点以内に必要得点がある表 {reach}/{len(big)}、実際/正規 の中央値 {np.median(ratios):.2f}、範囲 {min(ratios):.2f}〜{max(ratios):.2f}")
print()
print("必要得点(偏差値70)が満点を超える表:")
for nm, d in big:
    r = hensachi_threshold_row(d["counts"])[70]
    if r["k"] > d["full"]:
        print(f"  {nm}(満点{d['full']}): 必要 {r['thr']:.1f} 点")
print()
print("代表科目の必要得点と実測")
for nm in ["国語", "数学①", "数学②", "英語（リーディング）", "リスニング", "情報", "化学", "地学基礎", "物理"]:
    d = subj[nm]
    r = hensachi_threshold_row(d["counts"])
    n, m, sd = moments(d["counts"])
    print(nm, f"満点{d['full']}", " ".join(f"偏差値{h}:必要{r[h]['thr']:.1f}点→実{r[h]['actual']*100:.2f}%/正{r[h]['normal']*100:.2f}%" for h in (60, 65, 70)))
