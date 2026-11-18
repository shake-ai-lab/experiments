"""手順2: 「有意になるまで足して検定し直す」と、本当は差が無いのに偽陽性率はどこまで上がるか。
各群10人から始め、1人ずつ（または10人ずつ）足して検定し直し、p<0.05 になった時点で止める。
上限の人数まで行っても有意にならなければ「差なし」で終わる。
5つの種 x 各1万回 = 5万回で測り、種ごとの範囲も出す。"""
import numpy as np
from scipy import stats
import json

N0 = 10
NMAX = 500
SEEDS = [11, 12, 13, 14, 15]
SIMS = 10_000
LIMITS = [10, 20, 30, 50, 100, 200, 500]


def p_paths(rng, sims):
    """各シミュレーションで n=N0..NMAX 人目までの p値の経路。形 (sims, NMAX-N0+1)。差は無い(d=0)。"""
    x = rng.standard_normal((sims, NMAX))
    y = rng.standard_normal((sims, NMAX))
    n = np.arange(N0, NMAX + 1)
    cx, cy = np.cumsum(x, 1)[:, N0 - 1:], np.cumsum(y, 1)[:, N0 - 1:]
    cx2, cy2 = np.cumsum(x ** 2, 1)[:, N0 - 1:], np.cumsum(y ** 2, 1)[:, N0 - 1:]
    mx, my = cx / n, cy / n
    vx, vy = (cx2 - n * mx ** 2) / (n - 1), (cy2 - n * my ** 2) / (n - 1)
    t = (mx - my) / np.sqrt((vx + vy) / n)  # 同じ人数なので sp2*(2/n) = (vx+vy)/n
    return 2 * stats.t.sf(np.abs(t), 2 * n - 2), x, y


def main():
    n = np.arange(N0, NMAX + 1)
    res = {"every1": {L: [] for L in LIMITS}, "every10": {L: [] for L in LIMITS}}
    stop_n = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        P, x, y = p_paths(rng, SIMS)
        if seed == SEEDS[0]:
            # 自作の累積計算が scipy と一致するか（経路上の全点から200本x6点を無作為に）
            chk = np.random.default_rng(0)
            md = 0.0
            for i in chk.integers(0, SIMS, 200):
                for k in chk.integers(0, len(n), 6):
                    m = n[k]
                    pr = stats.ttest_ind(x[i, :m], y[i, :m]).pvalue
                    md = max(md, abs(pr - P[i, k]))
            print(f"累積計算 vs scipy.ttest_ind: 1200点での p の最大絶対差 = {md:.2e}")
        sig = P < 0.05
        for L in LIMITS:
            cols = n <= L
            res["every1"][L].append(sig[:, cols].any(1).mean())
            cols10 = cols & ((n - N0) % 10 == 0)
            res["every10"][L].append(sig[:, cols10].any(1).mean())
        # 500人まで足したとき、止まった時点の人数
        first = np.where(sig.any(1), sig.argmax(1) + N0, 0)
        stop_n[seed] = first
    print(f"\n各群 {N0} 人から開始。{len(SEEDS)}種 x {SIMS}回 = {len(SEEDS)*SIMS} 回。偽陽性率(%) 平均 [種ごとの最小-最大]")
    print("上限人数(各群) | 1人ずつ検定 | 10人ずつ検定")
    for L in LIMITS:
        a, b = np.array(res["every1"][L]) * 100, np.array(res["every10"][L]) * 100
        print(f"  {L:>4} | {a.mean():5.1f} [{a.min():.1f}-{a.max():.1f}] | {b.mean():5.1f} [{b.min():.1f}-{b.max():.1f}]")
    allfirst = np.concatenate(list(stop_n.values()))
    hit = allfirst[allfirst > 0]
    print(f"\n上限500のとき、有意が出て止まった割合 {len(hit)/len(allfirst)*100:.1f}%。"
          f"止まった人のうち、最初の10人目で止まった={np.mean(hit == N0)*100:.1f}%, "
          f"中央値={np.median(hit):.0f}人, 50人以内で止まった={np.mean(hit <= 50)*100:.1f}%, 100人以内={np.mean(hit <= 100)*100:.1f}%")
    # 一回だけ（上限なし・人数固定で1回検定）の参照値
    p_fixed = P[:, n == 100][:, 0]
    print(f"参照: 100人で1回だけ検定したときの偽陽性率(最後の種) = {(p_fixed < 0.05).mean()*100:.2f}%")
    json.dump({k: {str(L): v for L, v in d.items()} for k, d in res.items()}, open("optional_stopping.json", "w"))


if __name__ == "__main__":
    main()
