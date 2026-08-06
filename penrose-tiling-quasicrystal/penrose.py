# ペンローズ・タイル: 2種類のひし形で平面を埋めると、なぜ周期が現れないのか
#
# ロビンソン三角形の細分(deflation)で生成する。
#
# 実験1: 細分のたびに、太いひし形と細いひし形の個数比が黄金比に近づくか
# 実験2: フーリエ変換して回折パターンを見る。5回/10回対称のスポットが出るか
# 実験3: 周期があるかどうかを自己相関で確かめる(周期タイリングとの比較)
#
# 実行: C:/bin/Python/python310-64/python.exe -u penrose.py

import json

import numpy as np

from penrose_core import PHI, subdivide, initial_wheel

OUT = {"phi": PHI}

# ================================================================ 実験1
print("[実験1] 細分するたびに、2種類のひし形の個数比はどうなるか", flush=True)
tris = initial_wheel()
rows = []
for gen in range(1, 11):
    tris = subdivide(tris)
    n0 = sum(1 for t in tris if t[0] == 0)
    n1 = sum(1 for t in tris if t[0] == 1)
    ratio = n0 / n1 if n1 else float("nan")
    rows.append({"gen": gen, "thick": n0, "thin": n1, "ratio": ratio,
                 "err": abs(ratio - PHI)})
    print(f"  {gen:>2}回目: 太い {n0:>7,} / 細い {n1:>7,} / 比 {ratio:.8f} "
          f"(黄金比との差 {abs(ratio-PHI):.2e})")
print(f"  黄金比 φ = {PHI:.8f}")
OUT["exp1"] = rows

# ================================================================ 実験2
print("\n[実験2] 頂点の配置をフーリエ変換して回折パターンを見る", flush=True)
pts = []
for kind, A, B, C in tris:
    pts += [A, B, C]
P = np.array(pts)
P = np.unique(np.round(P, 9))
P = P[np.abs(P) < 0.92]
print(f"  頂点 {len(P):,} 個（重複を除き、外周を切り落としたもの）")

G = 1024
img = np.zeros((G, G))
xi = ((P.real + 1) / 2 * (G - 1)).astype(int)
yi = ((P.imag + 1) / 2 * (G - 1)).astype(int)
np.add.at(img, (yi, xi), 1.0)
F = np.abs(np.fft.fftshift(np.fft.fft2(img)))
F[G // 2, G // 2] = 0
# 角度ごとのピーク強度を見て、何回対称かを調べる
cy = cx = G // 2
yy, xx = np.mgrid[0:G, 0:G]
rr = np.hypot(yy - cy, xx - cx)
th = np.arctan2(yy - cy, xx - cx)
ring = (rr > 30) & (rr < 220)
bins = 720
prof = np.zeros(bins)
tb = ((th[ring] + np.pi) / (2 * np.pi) * bins).astype(int) % bins
np.add.at(prof, tb, F[ring])
# 角度プロファイルのフーリエ係数で対称性の次数を測る
spec = np.abs(np.fft.rfft(prof - prof.mean()))
top = np.argsort(spec[1:40])[::-1][:5] + 1
print(f"  角度方向のスペクトルで強い成分: {top.tolist()} 次")
print("  （10次が強ければ10回対称＝準結晶の特徴）")
OUT["exp2"] = {"n_points": int(len(P)), "top_orders": top.tolist(),
               "spec": spec[:41].tolist()}
np.save("diffraction.npy", F[cy - 260:cy + 260, cx - 260:cx + 260])

# ================================================================ 実験3
print("\n[実験3] 周期があるか(自己相関で確かめる)", flush=True)
# ペンローズと、比較用の周期的な三角格子で同じことをやる
def autocorr_peaks(points, G=512, span=0.9):
    im = np.zeros((G, G))
    ok = np.abs(points) < span
    q = points[ok]
    xi = ((q.real / span + 1) / 2 * (G - 1)).astype(int)
    yi = ((q.imag / span + 1) / 2 * (G - 1)).astype(int)
    np.add.at(im, (yi, xi), 1.0)
    f = np.fft.fft2(im)
    ac = np.fft.fftshift(np.abs(np.fft.ifft2(f * np.conj(f))))
    ac[G // 2, G // 2] = 0
    return ac


ac_p = autocorr_peaks(P)
# 周期格子(三角格子)
a = 0.03
gy, gx = np.mgrid[-40:41, -40:41]
lat = (gx * a + gy * a * 0.5) + 1j * (gy * a * np.sqrt(3) / 2)
lat = lat.ravel()
ac_l = autocorr_peaks(lat)
for name, ac in [("ペンローズ", ac_p), ("周期的な三角格子", ac_l)]:
    m = ac.max()
    strong = int((ac > m * 0.5).sum())
    print(f"  {name:<16} 自己相関の最大値に対し50%を超えるピーク {strong:>5} 個")
OUT["exp3"] = {"penrose_peaks": int((ac_p > ac_p.max() * 0.5).sum()),
               "lattice_peaks": int((ac_l > ac_l.max() * 0.5).sum())}

np.save("tris.npy", np.array([[t[0], t[1].real, t[1].imag, t[2].real, t[2].imag,
                               t[3].real, t[3].imag] for t in tris]))
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json")
