# 全体で測ると次元1.09にしかならず、解像度を上げても動かなかった。
# ならば「境界のどこを測るか」で変わるのではないか。
#
# マンデルブロ集合の境界には、なめらかな部分(メインカージオイドの縁は解析曲線)と、
# いくらでも入り組んだ部分(Misiurewicz点のまわり)がある。
# Shishikura(1998)の証明も「Misiurewicz点の近傍で次元が2に近づく」という形をしている。
# 同じ box-counting コードで、場所ごとに・拡大しながら測ってみる。

import json
import time

import numpy as np

from common import escape_time

FIT_SIZES = [4, 8, 16, 32, 64]
R = 1024
# 反復回数は窓の大きさに合わせる。広い窓をいきなり2万回で回すと、
# カージオイドの縁のすぐ外にある「なかなか逃げない点」が全部生き残って終わらない。
# 全体像の実験で、次元推定は maxiter 2000 以上でほぼ動かなくなることを確認済み。
def maxiter_for(w):
    return 2000 if w >= 1e-3 else 8000


def boundary_pixels(inside):
    b = np.zeros_like(inside)
    b[1:, :] |= inside[1:, :] & ~inside[:-1, :]
    b[:-1, :] |= inside[:-1, :] & ~inside[1:, :]
    b[:, 1:] |= inside[:, 1:] & ~inside[:, :-1]
    b[:, :-1] |= inside[:, :-1] & ~inside[:, 1:]
    return b


def dimension(mask, extent, sizes=FIT_SIZES):
    RR = mask.shape[0]
    counts = np.array(
        [int(mask[: (RR // b) * b, : (RR // b) * b]
             .reshape(RR // b, b, RR // b, b)
             .any(axis=(1, 3)).sum()) for b in sizes]
    )
    eps = np.array(sizes, dtype=float) * extent / RR
    ok = counts > 0
    if ok.sum() < 3:
        return float("nan"), counts
    D = np.polyfit(np.log(1 / eps[ok]), np.log(counts[ok]), 1)[0]
    return float(D), counts


# メインカージオイド上の点 c(θ) = e^{iθ}/2 - e^{2iθ}/4 (θ=2.0rad)。ここは解析曲線なので次元1のはず
theta = 2.0
smooth = np.exp(1j * theta) / 2 - np.exp(2j * theta) / 4

CENTERS = {
    "なめらかな縁(メインカージオイド上)": complex(smooth),
    "Misiurewicz点 (-0.1011+0.9563i)": complex(-0.10109636384562, 0.95628651080914),
    "シーホースバレー入口 (-0.75+0.01i)": complex(-0.75, 0.01),
}
WINDOWS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

out = {"resolution": R, "fit_sizes_px": FIT_SIZES, "centers": {}}
print(f"解像度 {R}x{R} / フィット範囲 {FIT_SIZES} 画素\n")
for name, c0 in CENTERS.items():
    print(f"=== {name}  c = {c0.real:.10f} {c0.imag:+.10f}i ===", flush=True)
    rec = []
    for w in WINDOWS:
        t0 = time.time()
        mi = maxiter_for(w)
        esc = escape_time(c0.real - w, c0.real + w, c0.imag - w, c0.imag + w, R, R, mi)
        inside = esc == 0
        frac = inside.mean()
        bp = boundary_pixels(inside)
        D, counts = dimension(bp, 2 * w)
        rec.append({"half_width": w, "maxiter": mi, "D": D, "inside_frac": float(frac),
                    "boundary_px": int(bp.sum()), "counts": counts.tolist()})
        print(f"  半幅 {w:>7.0e}  D={D:6.4f}  集合内割合={frac*100:5.2f}%  "
              f"境界画素={bp.sum():>8,}  ({time.time()-t0:.0f}秒)", flush=True)
    out["centers"][name] = rec

with open("dimension3.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n保存: dimension3.json")
