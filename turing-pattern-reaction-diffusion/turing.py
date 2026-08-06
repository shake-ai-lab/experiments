# チューリングパターン: 拡散するだけの2物質から、なぜ縞や斑点ができるのか
#
# Gray-Scottモデル:
#   du/dt = Du∇²u - u v² + F(1-u)
#   dv/dt = Dv∇²v + u v² - (F+k)v
# 反応と拡散しかない。どこにも「縞を作れ」とは書いていない。
#
# 実験1: 供給率Fと除去率kを格子状に振り、出てくる模様を分類する(相図)
# 実験2: 縞の波長をFFTで測る
# 実験3: チューリングの主張「拡散が不安定を生む」の確認。
#         2物質の拡散係数の比を1に近づけると模様は消えるのか
#
# 実行: C:/bin/Python/python310-64/python.exe -u turing.py

import json

import numpy as np
from scipy import ndimage

OUT = {}
DU, DV = 0.16, 0.08
DT = 1.0


from core import simulate, describe, laplacian, DU, DV, DT


# ================================================================ 実験1
print("[実験1] 供給率Fと除去率kを振って、出てくる模様を分類する", flush=True)
FS = [0.014, 0.022, 0.030, 0.038, 0.046, 0.054, 0.062]
KS = [0.045, 0.050, 0.055, 0.060, 0.062, 0.065]
STEPS = 9000
grid = {}
for F in FS:
    line = []
    for k in KS:
        _, v, _ = simulate(F, k, STEPS, seed=7)
        d = describe(v)
        line.append(d)
        grid[f"{F:.3f}_{k:.3f}"] = d
    print("  F=%.3f: " % F + " / ".join(
        f"k={kk:.3f} 塊{d['blobs']:>4}個 面積{d['frac']*100:4.1f}%" for kk, d in zip(KS, line)),
        flush=True)
OUT["phase"] = {"F": FS, "k": KS, "steps": STEPS, "grid": grid}

alive = [(f, kk) for f in FS for kk in KS if grid[f"{f:.3f}_{kk:.3f}"]["frac"] > 0.02]
print(f"\n  模様が残ったのは {len(alive)} / {len(FS)*len(KS)} 通り")

# ================================================================ 実験2
print("\n[実験2] 縞の波長を測る", flush=True)
cases = [(0.030, 0.055, "迷路"), (0.022, 0.051, "水玉"), (0.038, 0.061, "細かい斑点"),
         (0.014, 0.045, "分裂して増える"), (0.046, 0.062, "虫食い")]
rows = []
for F, k, name in cases:
    _, v, _ = simulate(F, k, 12000, N=200, seed=3)
    d = describe(v)
    rows.append({"F": F, "k": k, "name": name, **d})
    print(f"  F={F:.3f} k={k:.3f} ({name}): 主波長 {d['wavelength']:.1f} 格子 / "
          f"塊 {d['blobs']:,} 個 / 面積 {d['frac']*100:.1f}%")
OUT["wavelength"] = rows

# ================================================================ 実験3
print("\n[実験3] 2物質の拡散係数の比を変える(チューリング不安定性の条件)", flush=True)
ratios = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]
rows3 = []
for r in ratios:
    dv = DU / r
    _, v, _ = simulate(0.030, 0.055, 9000, N=140, seed=11, du=DU, dv=dv)
    d = describe(v)
    rows3.append({"ratio": r, "dv": dv, **d})
    print(f"  Du/Dv = {r:.1f}: 面積 {d['frac']*100:5.2f}% / 塊 {d['blobs']:>4} 個 / "
          f"主波長 {d['wavelength']:.1f}")
OUT["ratio"] = rows3

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json")
