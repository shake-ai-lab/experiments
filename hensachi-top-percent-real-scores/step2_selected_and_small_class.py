"""(a) 代表16科目の表  (b) 40人のクラスで偏差値を出すとどれくらい揺れるか"""
import sys
import numpy as np
from scipy.stats import zscore
from core import load_subjects, moments, hensachi_threshold_row

sys.stdout.reconfigure(encoding="utf-8")
subj = load_subjects()

SELECTED = [
    "国語", "英語（リーディング）", "リスニング", "情報Ⅰ",
    "数学Ⅰ，数学Ａ", "数学Ⅱ，数学Ｂ，数学Ｃ",
    "物理", "化学", "生物", "化学基礎", "生物基礎",
    "地理総合，地理探究", "歴史総合，日本史探究", "歴史総合，世界史探究",
    "公共，政治・経済", "公共，倫理",
]

if __name__ == "__main__":
    print("== (a) 代表16科目 ==")
    cnt70 = 0
    ratios60 = []
    for nm in SELECTED:
        d = subj[nm]
        r = hensachi_threshold_row(d["counts"])
        n, m, sd = moments(d["counts"])
        reach70 = r[70]["k"] <= d["full"]
        cnt70 += not reach70
        ratios60.append(r[60]["actual"])
        print(f"{nm:<14}満点{d['full']:>3} n={n:>7} 平均{m:6.2f} SD{sd:6.2f} 必要得点 60:{r[60]['thr']:6.1f} 65:{r[65]['thr']:6.1f} 70:{r[70]['thr']:6.1f}"
              f" | 実測% 60:{r[60]['actual']*100:5.2f} 65:{r[65]['actual']*100:5.2f} 70:{r[70]['actual']*100:5.2f}")
    print(f"偏差値70の得点が満点を超える科目: {cnt70}/{len(SELECTED)}")
    print(f"偏差値60以上の割合: 最小 {min(ratios60)*100:.1f}% 最大 {max(ratios60)*100:.1f}% 中央値 {np.median(ratios60)*100:.1f}% (正規分布なら15.87%)")

    # 偏差値の計算の自己検証: scipy.stats.zscore と一致するか
    rng = np.random.default_rng(0)
    pop = np.repeat(np.arange(101), subj["英語（リーディング）"]["counts"])
    x = rng.choice(pop, 40, replace=False).astype(float)
    mine = 50 + 10 * (x - x.mean()) / x.std()
    ref = 50 + 10 * zscore(x, ddof=0)
    print("自作の偏差値と scipy.stats.zscore(ddof=0) の最大差:", np.abs(mine - ref).max())

    # ---- (b) 少人数クラス
    print()
    print("== (b) 英語リーディングの全受験者(母集団)から n 人を無作為に取り、同じ素点の偏差値を測る ==")
    n_pop, m_pop, sd_pop = moments(subj["英語（リーディング）"]["counts"])
    x0 = int(round(m_pop + sd_pop))  # 全国で偏差値約60になる素点
    print(f"母集団: 平均 {m_pop:.2f} SD {sd_pop:.2f}、素点 {x0} の全国偏差値 = {50 + 10 * (x0 - m_pop) / sd_pop:.2f}")
    REPS = 20000
    for n in (10, 40, 200, 1000):
        res = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(pop), size=(REPS, n))  # 非復元でなく復元抽出(母集団が45万人なので差は無視できる)
            S = pop[idx].astype(float)
            m = S.mean(axis=1)
            s = S.std(axis=1)
            h = 50 + 10 * (x0 - m) / s
            res.append((np.percentile(h, 5), np.percentile(h, 50), np.percentile(h, 95), h.std()))
        res = np.array(res)
        print(f"n={n:>4}: 5%点 {res[:,0].mean():5.1f}({res[:,0].min():.1f}〜{res[:,0].max():.1f})  中央値 {res[:,1].mean():5.1f}  "
              f"95%点 {res[:,2].mean():5.1f}({res[:,2].min():.1f}〜{res[:,2].max():.1f})  標準偏差 {res[:,3].mean():.2f}")

    # 40人のクラスで「偏差値60以上」になる人数
    print()
    print("== 40人のクラスで、そのクラス内の偏差値が60以上になる人数(英語リーディングの本物の分布から抽出) ==")
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        idx = rng.integers(0, len(pop), size=(REPS, 40))
        S = pop[idx].astype(float)
        H = 50 + 10 * (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
        k = (H >= 60).sum(axis=1)
        print(f"  種{100+seed}: 平均 {k.mean():.2f} 人、5%〜95% = {np.percentile(k,5):.0f}〜{np.percentile(k,95):.0f} 人、0人のクラス {np.mean(k==0)*100:.1f}%")
