# ジュリア集合: 「連結か、砂に散るか」だけでマンデルブロ集合を描き直せるか
#
# マンデルブロ集合の定義は「z -> z^2 + c を 0 から反復して発散しない c の集合」だが、
# もう一つの顔がある。「ジュリア集合 J_c が連結になる c の集合」でもある(Fatou/Julia)。
# 後者は前者を一切使わずに定義できるので、実際にジュリア集合の連結成分を数えるだけで
# マンデルブロ集合が出てくるかを確かめる。
#
# 実験A: c を格子状に振り、それぞれのジュリア集合の連結成分数を数える
# 実験B: 小さい |c| でジュリア集合の境界次元を測り、Ruelleの公式と比べる
#
# 実行: C:/bin/Python/python310-64/python.exe -u julia.py

import json
import time

import numpy as np
from scipy import ndimage

OUT = {}


from core import julia_escape


# ================================================================ 実験A
GRID = 90          # c の格子(片側)
JN = 120           # 各ジュリア集合のラスタ解像度
JITER = 90
CX0, CX1 = -2.05, 0.6
CY0, CY1 = -1.25, 1.25
RAD = 2.0          # ジュリア集合を描く窓(|z|<=2 の外は必ず発散する)

print(f"[実験A] c を {GRID}x{GRID} に振り、各ジュリア集合({JN}x{JN})の連結成分を数える", flush=True)
cx = np.linspace(CX0, CX1, GRID)
cy = np.linspace(CY0, CY1, GRID)
ncomp = np.zeros((GRID, GRID), dtype=np.int32)
filled = np.zeros((GRID, GRID), dtype=np.float64)
t0 = time.time()
for i, yy in enumerate(cy):
    for j, xx in enumerate(cx):
        c = complex(xx, yy)
        K = julia_escape(c, -RAD, RAD, -RAD, RAD, JN, JITER)
        filled[i, j] = K.mean()
        lab, n = ndimage.label(K, structure=np.ones((3, 3)))
        ncomp[i, j] = n
    if i % 15 == 0:
        print(f"  {i}/{GRID} 行 ({time.time()-t0:.0f}秒)", flush=True)
print(f"  完了 {time.time()-t0:.0f}秒", flush=True)

# 「連結成分が少ない」= 連結とみなす。閾値を変えたときの一致率を見る
mand_ref = np.zeros((GRID, GRID), dtype=bool)   # 定義どおりのマンデルブロ集合(答え合わせ用)
for i, yy in enumerate(cy):
    for j, xx in enumerate(cx):
        c = complex(xx, yy)
        z = 0j
        ok = True
        for _ in range(400):
            z = z * z + c
            if z.real * z.real + z.imag * z.imag > 4:
                ok = False
                break
        mand_ref[i, j] = ok

print("\n  連結成分数の閾値ごとに、定義どおりのマンデルブロ集合とどれだけ一致するか")
best = None
for th in [1, 2, 3, 5, 10, 20, 50]:
    pred = ncomp <= th
    agree = (pred == mand_ref).mean()
    tp = (pred & mand_ref).sum()
    fp = (pred & ~mand_ref).sum()
    fn = (~pred & mand_ref).sum()
    print(f"    成分数<={th:>3} を連結とみなす: 一致率 {agree*100:5.2f}%  "
          f"(取りこぼし {fn:>4} / 誤検出 {fp:>4})")
    if best is None or agree > best[1]:
        best = (th, agree)
print(f"  最良の閾値: 成分数<={best[0]} で一致率 {best[1]*100:.2f}%")

inside = mand_ref
print(f"\n  マンデルブロ集合の内側 {inside.sum():,} 点: 連結成分数の中央値 "
      f"{np.median(ncomp[inside]):.0f} / 最大 {ncomp[inside].max()}")
print(f"  外側 {(~inside).sum():,} 点: 連結成分数の中央値 "
      f"{np.median(ncomp[~inside]):.0f} / 最大 {ncomp[~inside].max():,}")

OUT["gridA"] = {"grid": GRID, "jn": JN, "jiter": JITER,
                "cx": [CX0, CX1], "cy": [CY0, CY1],
                "ncomp": ncomp.tolist(), "mand": mand_ref.tolist(),
                "filled": filled.tolist(), "best_th": int(best[0]),
                "best_agree": float(best[1])}

# ================================================================ 実験B
print("\n[実験B] 小さい |c| でジュリア集合の境界次元を測り、Ruelleの公式と比べる", flush=True)
# Ruelle(1982): |c| が小さいとき dim(J_c) = 1 + |c|^2/(4 log 2) + O(|c|^3)
FIT_SIZES = [4, 8, 16, 32, 64]
RB = 2048
BITER = 3000


def boundary_pixels(m):
    b = np.zeros_like(m)
    b[1:, :] |= m[1:, :] & ~m[:-1, :]
    b[:-1, :] |= m[:-1, :] & ~m[1:, :]
    b[:, 1:] |= m[:, 1:] & ~m[:, :-1]
    b[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    return b


def boxdim(mask, extent):
    R = mask.shape[0]
    counts = np.array([int(mask[: (R // b) * b, : (R // b) * b]
                           .reshape(R // b, b, R // b, b).any(axis=(1, 3)).sum())
                       for b in FIT_SIZES])
    eps = np.array(FIT_SIZES, float) * extent / R
    ok = counts > 0
    return float(np.polyfit(np.log(1 / eps[ok]), np.log(counts[ok]), 1)[0]), counts


# 系統誤差の基準: 同じコードで円を測るといくつになるか(マンデルブロの回では0.976だった)
t = np.linspace(-2, 2, RB)
X, Y = np.meshgrid(t, t)
circle = X * X + Y * Y <= 1.0
d_circle, _ = boxdim(boundary_pixels(circle), 4.0)
print(f"  基準: 同じコードで円を測ると {d_circle:.4f} (理論値 1.0000)")

rows = []
for cv in [0.0, 0.05, 0.10, 0.15, 0.20, 0.24, -0.20, -0.40, -0.60, -0.75]:
    c = complex(cv, 0.0)
    K = julia_escape(c, -2, 2, -2, 2, RB, BITER)
    D, counts = boxdim(boundary_pixels(K), 4.0)
    ruelle = 1 + abs(c) ** 2 / (4 * np.log(2))
    rows.append({"c": cv, "D": D, "D_corrected": D + (1.0 - d_circle),
                 "ruelle": ruelle, "filled": float(K.mean())})
    print(f"  c={cv:>6.2f}: 実測 {D:.4f} / 円の誤差を補正 {D + (1.0 - d_circle):.4f} / "
          f"Ruelleの近似 {ruelle:.4f}", flush=True)
OUT["dim"] = {"circle": d_circle, "rows": rows, "R": RB, "maxiter": BITER}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json")
