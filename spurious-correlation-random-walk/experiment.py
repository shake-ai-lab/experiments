# -*- coding: utf-8 -*-
"""見せかけの回帰(spurious regression)の実測実験。

完全に独立な2系列のペアを大量に生成し、
  (a) iidガウスノイズ
  (b) AR(1) (phi=0.8, 定常)
  (c) ランダムウォーク(単位根)
の3種で相関係数 r と t検定の p値の分布を比較する。

さらに系列長 T を 50→5000 で振って「p<0.05 で有意と誤判定される率
(偽有意率)」の変化を測り、ランダムウォークは差分を取ると
偽相関が消えることも確認する。
"""
import json
from pathlib import Path

import numpy as np
from scipy import signal, stats

OUT = Path(__file__).parent
rng = np.random.default_rng(42)

N_PAIRS = 10_000          # 実験A/Bのペア数
N_PAIRS_HEAT = 3_000      # ヒートマップのセルあたりペア数
T_MAIN = 200              # 実験Aの系列長
T_LIST = [50, 100, 200, 500, 1000, 2000, 5000]
PHI_LIST = [0.0, 0.3, 0.6, 0.9, 0.95, 0.99, 1.0]  # 1.0 = ランダムウォーク
CHUNK = 2_000


def gen_series(kind, npairs, T, rng, phi=0.8):
    """独立な系列を (npairs, T) で生成する。"""
    e = rng.standard_normal((npairs, T))
    if kind == "iid":
        return e
    if kind == "rw":
        return np.cumsum(e, axis=1)
    if kind == "ar1":
        if phi >= 1.0:
            return np.cumsum(e, axis=1)
        if phi == 0.0:
            return e
        # 定常分布 N(0, 1/(1-phi^2)) から初期値を引いた厳密な定常AR(1)
        x0 = rng.standard_normal(npairs) * np.sqrt(1.0 / (1.0 - phi**2))
        y, _ = signal.lfilter([1.0], [1.0, -phi], e, axis=1,
                              zi=(phi * x0)[:, None])
        return y
    raise ValueError(kind)


def pearson_rows(x, y):
    """行ごとのPearson相関係数。"""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    r = (xc * yc).sum(axis=1) / np.sqrt(
        (xc**2).sum(axis=1) * (yc**2).sum(axis=1))
    return np.clip(r, -1.0, 1.0)


def pvalues(r, T):
    """通常のt検定(scipy.stats.pearsonrと同じ式)による両側p値。"""
    with np.errstate(divide="ignore"):
        t = r * np.sqrt((T - 2) / (1.0 - r**2))
    return 2.0 * stats.t.sf(np.abs(t), T - 2)


def run_batch(kind, npairs, T, rng, phi=0.8, diff=False):
    """独立ペアを生成して r と p を返す(チャンク処理)。"""
    rs = []
    done = 0
    while done < npairs:
        n = min(CHUNK, npairs - done)
        x = gen_series(kind, n, T, rng, phi)
        y = gen_series(kind, n, T, rng, phi)
        if diff:
            x = np.diff(x, axis=1)
            y = np.diff(y, axis=1)
        rs.append(pearson_rows(x, y))
        done += n
    r = np.concatenate(rs)
    T_eff = T - 1 if diff else T
    p = pvalues(r, T_eff)
    return r, p


def sanity_check():
    """自作のr/p計算がscipy.stats.pearsonrと一致するか確認。"""
    x = rng.standard_normal((5, 100))
    y = rng.standard_normal((5, 100))
    r_mine = pearson_rows(x, y)
    p_mine = pvalues(r_mine, 100)
    for i in range(5):
        r_sp, p_sp = stats.pearsonr(x[i], y[i])
        assert abs(r_sp - r_mine[i]) < 1e-12, (r_sp, r_mine[i])
        assert abs(p_sp - p_mine[i]) < 1e-9, (p_sp, p_mine[i])
    print("sanity check OK: pearson_rows/pvalues == scipy.stats.pearsonr")


def main():
    sanity_check()
    results = {}

    # ---- 実験A: T=200 で3種の r 分布を比較 --------------------------------
    print(f"\n=== 実験A: T={T_MAIN}, {N_PAIRS}ペア, r と p の分布 ===")
    hist = {}
    for label, kind, phi in [("iid", "iid", 0.0),
                             ("ar1", "ar1", 0.8),
                             ("rw", "rw", 0.0)]:
        r, p = run_batch(kind, N_PAIRS, T_MAIN, rng, phi)
        hist[label] = r
        stats_d = {
            "false_sig_rate": float((p < 0.05).mean()),
            "median_abs_r": float(np.median(np.abs(r))),
            "share_abs_r_gt_05": float((np.abs(r) > 0.5).mean()),
            "share_abs_r_gt_09": float((np.abs(r) > 0.9).mean()),
            "max_abs_r": float(np.abs(r).max()),
            "sd_r": float(r.std()),
        }
        results[f"A_{label}"] = stats_d
        print(f"{label:4s}: 偽有意率={stats_d['false_sig_rate']:.4f}  "
              f"|r|中央値={stats_d['median_abs_r']:.4f}  "
              f"|r|>0.5率={stats_d['share_abs_r_gt_05']:.4f}  "
              f"|r|>0.9率={stats_d['share_abs_r_gt_09']:.4f}  "
              f"max|r|={stats_d['max_abs_r']:.4f}")
    np.savez_compressed(OUT / "rhist.npz", **hist)

    # ---- 実験B: 系列長スイープ(差分系列も含む) ---------------------------
    print(f"\n=== 実験B: 偽有意率 vs 系列長 ({N_PAIRS}ペア/点) ===")
    sweep = {"T": T_LIST}
    for label, kind, phi, diff in [("iid", "iid", 0.0, False),
                                   ("ar1", "ar1", 0.8, False),
                                   ("rw", "rw", 0.0, False),
                                   ("rw_diff", "rw", 0.0, True)]:
        rates = []
        med_abs_r = []
        for T in T_LIST:
            r, p = run_batch(kind, N_PAIRS, T, rng, phi, diff=diff)
            rates.append(float((p < 0.05).mean()))
            med_abs_r.append(float(np.median(np.abs(r))))
        sweep[label] = rates
        sweep[label + "_med_abs_r"] = med_abs_r
        print(f"{label:8s}: " + "  ".join(
            f"T={T}:{v:.4f}" for T, v in zip(T_LIST, rates)))
    results["B_sweep"] = sweep

    # ---- 実験C: 自己相関の強さ phi × 系列長 T のヒートマップ ---------------
    print(f"\n=== 実験C: 偽有意率ヒートマップ phi×T ({N_PAIRS_HEAT}ペア/セル) ===")
    heat = np.zeros((len(PHI_LIST), len(T_LIST)))
    for i, phi in enumerate(PHI_LIST):
        for j, T in enumerate(T_LIST):
            r, p = run_batch("ar1", N_PAIRS_HEAT, T, rng, phi)
            heat[i, j] = (p < 0.05).mean()
        print(f"phi={phi:4.2f}: " + "  ".join(
            f"{v:.3f}" for v in heat[i]))
    results["C_heat"] = {"phi": PHI_LIST, "T": T_LIST,
                         "rate": heat.tolist()}

    # ---- 実験D: GIF/例示用の「それっぽく連動する」ランダムウォーク採集 -----
    print("\n=== 実験D: 高|r|なランダムウォークペアの採集 (T=300) ===")
    T_gif = 300
    x = gen_series("rw", 40_000, T_gif, rng)
    y = gen_series("rw", 40_000, T_gif, rng)
    r = pearson_rows(x, y)
    p = pvalues(r, T_gif)
    order = np.argsort(-r)[:10]  # 正の高相関ペア上位10
    np.savez_compressed(OUT / "gif_pairs.npz",
                        x=x[order], y=y[order], r=r[order], p=p[order])
    results["D_gif"] = {
        "share_r_gt_09_of_40000": float((np.abs(r) > 0.9).mean()),
        "top_r": [float(v) for v in r[order]],
        "top_p": [float(v) for v in p[order]],
    }
    print("採集した上位10ペアの r:",
          "  ".join(f"{v:.3f}" for v in r[order]))
    print(f"40,000ペア中 |r|>0.9 の割合: {(np.abs(r) > 0.9).mean():.4f}")

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nsaved:", OUT / "results.json")


if __name__ == "__main__":
    main()
