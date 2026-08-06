# ニュートンフラクタル: 3つの根が「どの国境も3か国に接している」ことを実測する
#
# z^3 - 1 = 0 をニュートン法で解くとき、どの初期値がどの根に落ちるかを色分けすると
# 境界がフラクタルになる。しかもこの境界には「Wadaの湖」という性質があるとされる:
# 境界上のどの点も、3つの領域すべてに接している。
#
# 実験1: 3根の吸引域を描き、境界のフラクタル次元を測る
# 実験2: 境界画素の近傍に何色が現れるかを数える(Wada性質の実測)
# 実験3: 根の数を変える(z^5-1)、および収束しない初期値がある例(z^3-2z+2)
#
# 実行: C:/bin/Python/python310-64/python.exe -u newton.py

import json

import numpy as np

OUT = {}


from newton_core import newton_basins, boundary_mask, boxdim


# ================================================================ 実験1
R = 2048
roots3 = [np.exp(2j * np.pi * k / 3) for k in range(3)]
print(f"[実験1] z^3-1=0 の吸引域を {R}x{R} で描く", flush=True)
w3, it3 = newton_basins(lambda z: z ** 3 - 1, lambda z: 3 * z ** 2, roots3, R)
for r in range(3):
    print(f"  根 {r} ({roots3[r]:.3f}) の吸引域: {(w3==r).mean()*100:.2f}%")
print(f"  どの根にも落ちなかった画素: {(w3==-1).mean()*100:.4f}%")

b3 = boundary_mask(w3)
D3, counts = boxdim(b3, 4.0)
# 較正: 同じコードで円の境界を測る
t = np.linspace(-2, 2, R)
X, Y = np.meshgrid(t, t)
circ = (X * X + Y * Y <= 1.0).astype(np.int8)
Dc, _ = boxdim(boundary_mask(circ), 4.0)
print(f"  境界のbox-counting次元: {D3:.4f}  (同じコードで円は {Dc:.4f}、理論値1)")
print(f"  補正後: {D3 + (1.0 - Dc):.4f}")
OUT["exp1"] = {"R": R, "D": D3, "D_circle": Dc, "D_corrected": D3 + (1.0 - Dc),
               "counts": counts.tolist(),
               "areas": [float((w3 == r).mean()) for r in range(3)],
               "none": float((w3 == -1).mean())}

# ================================================================ 実験2
print("\n[実験2] 境界のどの点が、3つの領域すべてに接しているか", flush=True)
rows = []
for RR in [512, 1024, 2048]:
    ww, _ = newton_basins(lambda z: z ** 3 - 1, lambda z: 3 * z ** 2, roots3, RR)
    bb = boundary_mask(ww)
    # 各境界画素の半径rの窓に何色あるか
    for rad in [1, 2, 4]:
        cols = np.zeros((RR, RR), dtype=np.int8)
        for c in range(3):
            m = (ww == c).astype(np.uint8)
            # 半径radの箱でmaxフィルタ
            acc = np.zeros_like(m)
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    acc |= np.roll(np.roll(m, dy, 0), dx, 1)
            cols += acc.astype(np.int8)
        sel = bb
        frac3 = float((cols[sel] == 3).mean())
        rows.append({"R": RR, "rad": rad, "frac3": frac3,
                     "n_boundary": int(sel.sum())})
        print(f"  解像度 {RR:>5}、半径 {rad} 画素の窓: 3色すべてが見える境界画素 "
              f"{frac3*100:5.2f}%  (境界画素 {sel.sum():,})")
OUT["exp2"] = rows

# 対照: 3つの円で平面を分けた場合(普通の境界)は2色しか見えないはず
print("\n  対照: ふつうに3等分した境界だとどうなるか", flush=True)
RR = 1024
ang = np.arctan2(*np.meshgrid(np.linspace(-2, 2, RR), np.linspace(-2, 2, RR))[::-1])
ww2 = ((ang + np.pi) / (2 * np.pi / 3)).astype(np.int8) % 3
bb2 = boundary_mask(ww2)
cols = np.zeros((RR, RR), dtype=np.int8)
for c in range(3):
    m = (ww2 == c).astype(np.uint8)
    acc = np.zeros_like(m)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            acc |= np.roll(np.roll(m, dy, 0), dx, 1)
    cols += acc.astype(np.int8)
print(f"  3等分の境界で3色すべてが見える割合: {(cols[bb2]==3).mean()*100:.2f}%")
OUT["exp2_control"] = float((cols[bb2] == 3).mean())

# ================================================================ 実験3
print("\n[実験3] 根の数を変える / 収束しない初期値がある例", flush=True)
roots5 = [np.exp(2j * np.pi * k / 5) for k in range(5)]
w5, _ = newton_basins(lambda z: z ** 5 - 1, lambda z: 5 * z ** 4, roots5, 1200)
b5 = boundary_mask(w5)
D5, _ = boxdim(b5, 4.0)
print(f"  z^5-1: 境界次元 {D5:.4f} / どの根にも落ちない画素 {(w5==-1).mean()*100:.3f}%")

# z^3-2z+2 はニュートン法が周期2の軌道に落ち込む初期値を持つ
r2 = np.roots([1, 0, -2, 2])
w2, _ = newton_basins(lambda z: z ** 3 - 2 * z + 2, lambda z: 3 * z ** 2 - 2,
                      list(r2), 1200, maxiter=200)
print(f"  z^3-2z+2: どの根にも落ちなかった画素 {(w2==-1).mean()*100:.3f}% "
      f"(200回まわしても収束しない領域がある)")
OUT["exp3"] = {"D5": D5, "none5": float((w5 == -1).mean()),
               "none_trap": float((w2 == -1).mean())}

np.save("w3.npy", w3)
np.save("it3.npy", it3)
np.save("w5.npy", w5)
np.save("w2.npy", w2)
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json")
