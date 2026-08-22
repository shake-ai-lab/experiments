"""実測1.568と理論1.585の0.017差は、どこから来たのか。

step1 では「点の数もマス目の細かさも有限だから」と説明していたが、それを検証する。
候補を4つ立てて1つずつ潰す:
  (A) カオスゲームの乱数のブレ        → シードを変えて振れ幅を見る
  (B) 点の数が足りない                → 2万/20万/400万点で比べる
  (C) マス目が粗い                    → 256から4096まで細かくする
  (D) 三角形と方眼のズレ(格子非整列)  → 三角形を格子に揃えて置き直す

結論だけ先に書いておくと、効くのは (D) だけだった。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Meiryo"
THEORY = np.log(3) / np.log(2)  # 1.584963...

# 記事(step1)で使った正三角形。単位正方形の中に少し余白をとって置いてある
TRI_EQUILATERAL = np.array([[0.05, 0.05], [0.95, 0.05], [0.5, 0.05 + 0.9 * np.sqrt(3) / 2]])
# 直角二等辺三角形。頂点が単位正方形の角にあるので、2^k のマス目と辺がぴたり重なる
TRI_RIGHT = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
# 形は正三角形のまま、位置だけ変えたもの(ズレの向きが配置で変わることの確認用)
TRI_EQ_MOVED = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def chaos_game(verts, n, seed=0):
    """頂点をランダムに選び、今の点とその頂点の中点へ移り続ける"""
    rng = np.random.default_rng(seed)
    p = verts.mean(axis=0).copy()
    out = np.empty((n, 2))
    picks = rng.integers(0, 3, n)
    for i in range(n):
        p = (p + verts[picks[i]]) / 2
        out[i] = p
    return out[1000:]  # 出発点の影響が残る最初のぶんを捨てる


def occupied_boxes(points, k):
    """2^k x 2^k のマス目に重ねたとき、点が1つでも入るマスの数"""
    idx = np.floor(points * (2**k)).astype(np.int64)
    idx = np.clip(idx, 0, 2**k - 1)
    return len(np.unique(idx[:, 0] * (2**k) + idx[:, 1]))


def box_dimension(points, ks):
    counts = [occupied_boxes(points, k) for k in ks]
    slope, _ = np.polyfit(ks * np.log(2), np.log(counts), 1)
    return slope, counts


N = 4_000_000
KS_DEFAULT = np.arange(2, 9)  # 4x4 .. 256x256 (step1と同じ範囲)

print(f"理論値 log3/log2 = {THEORY:.6f}\n")

# ---- (A) 乱数のブレか ----
print("(A) シードを変えると値はどれだけ動くか (30万点, 4x4..256x256)")
seed_dims = [box_dimension(chaos_game(TRI_EQUILATERAL, 300_000, s), KS_DEFAULT)[0] for s in range(10)]
print(f"    10シード: 平均 {np.mean(seed_dims):.4f} / 標準偏差 {np.std(seed_dims):.5f} "
      f"/ 幅 {min(seed_dims):.4f}..{max(seed_dims):.4f}")
print(f"    → 理論との差 {abs(np.mean(seed_dims) - THEORY):.4f} に対しブレは2桁小さい。乱数のせいではない\n")

# ---- (B) 点の数か ----
print("(B) 点の数を増やすと理論値に近づくか (4x4..256x256)")
for n in (20_000, 200_000, 4_000_000):
    d, _ = box_dimension(chaos_game(TRI_EQUILATERAL, n), KS_DEFAULT)
    print(f"    {n:>9,}点: {d:.4f} (誤差 {abs(d - THEORY):.4f})")
print("    → 20万点でほぼ頭打ち。200倍にしても0.017は残る\n")

# ---- (C) マス目の細かさか ----
pts_eq = chaos_game(TRI_EQUILATERAL, N)
print("(C) マス目を細かくすると理論値に近づくか (400万点)")
for hi in (9, 11, 13):
    d, _ = box_dimension(pts_eq, np.arange(2, hi))
    print(f"    4x4..{2**(hi-1)}x{2**(hi-1)}: {d:.4f} (誤差 {abs(d - THEORY):.4f})")
print("    → 16倍細かくしても動かない\n")

# ---- (D) 三角形と方眼のズレか ----
print("(D) 三角形を格子に揃えて置き直す (400万点)")
runs = {}
for label, verts in (("正三角形(step1の配置)", TRI_EQUILATERAL),
                     ("正三角形(位置だけ変更)", TRI_EQ_MOVED),
                     ("直角二等辺(格子に整列)", TRI_RIGHT)):
    pts = chaos_game(verts, N)
    runs[label] = pts
    for hi in (9, 11, 13):
        d, _ = box_dimension(pts, np.arange(2, hi))
        print(f"    {label:<22} 4x4..{2**(hi-1):>4}: {d:.4f} (誤差 {abs(d - THEORY):.4f})")
    print()

print("占有マス数そのもの (400万点)")
print(f"    {'マス目':<10}" + "".join(f"{2**k:>7}" for k in KS_DEFAULT))
print(f"    {'理論 3^k':<10}" + "".join(f"{3**k:>7}" for k in KS_DEFAULT))
for label in ("正三角形(step1の配置)", "直角二等辺(格子に整列)"):
    counts = [occupied_boxes(runs[label], k) for k in KS_DEFAULT]
    print(f"    {label:<10}" + "".join(f"{c:>7}" for c in counts))


# ---- (E) 揃えれば必ず合うのか。マス目の「刻み」を1/2から1/3に変えてみる ----
def occupied_boxes_m(points, m):
    """m x m のマス目(mは2のべきでなくてよい)"""
    idx = np.clip(np.floor(points * m).astype(np.int64), 0, m - 1)
    return len(np.unique(idx[:, 0] * m + idx[:, 1]))


def dimension_from(points, ms):
    counts = [occupied_boxes_m(points, m) for m in ms]
    slope, _ = np.polyfit(np.log(ms), np.log(counts), 1)
    return slope, counts


print("\n(E) 同じ図形(直角二等辺・格子に整列)を、マス目の刻みだけ変えて測る")
pts_rt = runs["直角二等辺(格子に整列)"]
base2 = [2**k for k in range(2, 9)]
base3 = [3**k for k in range(2, 7)]
d2, c2 = dimension_from(pts_rt, base2)
d3, c3 = dimension_from(pts_rt, base3)
print(f"    1/2刻み {base2[0]}..{base2[-1]}: {d2:.4f} (誤差 {abs(d2 - THEORY):.4f}) 塗れたマス {c2}")
print(f"    1/3刻み {base3[0]}..{base3[-1]}: {d3:.4f} (誤差 {abs(d3 - THEORY):.4f}) 塗れたマス {c3}")
print("    → 図形も置き方も同じなのに0.05ずれる。噛み合っているのは向きではなく『刻み』のほう")

# ---- (F) 較正用の直線・正方形も、同じ理由でずれていたのか ----
# step1 では4図形すべてを 0.05..0.95 の範囲に置いていた。方眼の端と合っていない
print("\n(F) 較正用の図形(直線・正方形)も置き直してみる")
rng = np.random.default_rng(0)
for name, shape_art, shape_fit, theo in (
    ("直線 (理論1.0)",
     np.c_[np.linspace(0.05, 0.95, 200_000), np.full(200_000, 0.5)],
     np.c_[np.linspace(0.0, 1.0, 200_000), np.full(200_000, 0.5)], 1.0),
    ("正方形 (理論2.0)",
     rng.uniform(0.05, 0.95, (2_000_000, 2)),
     rng.uniform(0.0, 1.0, (2_000_000, 2)), 2.0),
):
    d_art, c_art = box_dimension(shape_art, KS_DEFAULT)
    d_fit, c_fit = box_dimension(shape_fit, KS_DEFAULT)
    print(f"    {name}")
    print(f"      0.05..0.95 (step1): {d_art:.4f} (誤差 {abs(d_art - theo):.4f}) 塗れたマス {c_art}")
    print(f"      0..1 (方眼に整列)  : {d_fit:.4f} (誤差 {abs(d_fit - theo):.4f}) 塗れたマス {c_fit}")
print("    → 直線・正方形・シェルピンスキーの3つは同じ原因。端を方眼に合わせるだけで誤差が消える")
print("      (残るコッホ曲線だけは事情が違う。次の (G) を見ること)")


# ---- (G) コッホ曲線は、刻みを1/3に合わせれば揃うのか ----
def koch_curve(levels=7):
    seg = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    for _ in range(levels):
        new = []
        for a, b in zip(seg[:-1], seg[1:]):
            d = b - a
            p1, p2 = a + d / 3, a + d * 2 / 3
            new += [a, p1, p1 + rot @ (d / 3), p2]
        new.append(seg[-1])
        seg = new
    seg = np.array(seg)
    dense = [a + t * (b - a) for a, b in zip(seg[:-1], seg[1:])
             for t in np.linspace(0, 1, 12, endpoint=False)]
    return np.array(dense)


print("\n(G) コッホ曲線(自己相似比 1/3)を、1/2刻みと1/3刻みの方眼で測る")
koch_theory = np.log(4) / np.log(3)
K = koch_curve()
for label, ms in (("1/2刻み 4..256", [2**k for k in range(2, 9)]),
                  ("1/3刻み 3..729", [3**k for k in range(1, 7)])):
    d, c = dimension_from(K, ms)
    print(f"    {label}: {d:.4f} (誤差 {abs(d - koch_theory):.4f}) 塗れたマス {c}")
print(f"    参考 4^k = {[4**k for k in range(1, 7)]}")
print("    → 刻みを1/3に合わせても揃わない。線分が60度に走っていて、どの方眼とも平行にならないため")

# ---- 図3: 方眼に重ねて、踏んでいるマスを塗る ----
K_SHOW = 4  # 16x16
fig, axes = plt.subplots(2, 1, figsize=(7.2, 13.0))
for ax, (label, verts, color) in zip(axes, [
    ("正三角形 — step1で測っていた置き方", TRI_EQUILATERAL, "#e4572e"),
    ("直角二等辺 — 方眼に辺を揃えた置き方", TRI_RIGHT, "#2e6ee4"),
]):
    pts = chaos_game(verts, 600_000)
    m = 2**K_SHOW
    idx = np.clip(np.floor(pts * m).astype(int), 0, m - 1)
    filled = set(map(tuple, idx))
    for gx, gy in filled:
        ax.add_patch(Rectangle((gx / m, gy / m), 1 / m, 1 / m,
                               facecolor=color, alpha=0.22, edgecolor="none"))
    sub = pts[:: max(1, len(pts) // 40_000)]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.12, color=color)
    for i in range(m + 1):
        ax.axhline(i / m, color="#999", lw=0.5, alpha=0.55)
        ax.axvline(i / m, color="#999", lw=0.5, alpha=0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    verdict = "ぴったり" if len(filled) == 3**K_SHOW else f"{len(filled) - 3**K_SHOW}マス多い"
    ax.set_title(f"{label}\n16×16の方眼で塗れたマス {len(filled)} / 理論 {3**K_SHOW}  → {verdict}",
                 fontsize=15, pad=14)
fig.tight_layout(h_pad=2.4)
fig.savefig("fig3_alignment.png", dpi=110, bbox_inches="tight")
print("\nsaved fig3_alignment.png")

# ---- 図4: 理論3^kからのズレを、マス目の細かさごとに ----
ks_long = np.arange(2, 12)
fig, axes = plt.subplots(2, 1, figsize=(7.6, 10.4))
styles = {"正三角形(step1の配置)": ("#e4572e", "o-"),
          "正三角形(位置だけ変更)": ("#c9a227", "s--"),
          "直角二等辺(格子に整列)": ("#2e6ee4", "^-")}
for label, pts in runs.items():
    color, style = styles[label]
    counts = np.array([occupied_boxes(pts, k) for k in ks_long])
    axes[0].plot(ks_long * np.log(2), np.log(counts), style, color=color, ms=5, label=label)
    axes[1].plot(2**ks_long, counts / 3.0**ks_long, style, color=color, ms=5, label=label)
axes[0].set_xlabel("log(1 / マス目の大きさ)", fontsize=13)
axes[0].set_ylabel("log(塗れたマス数)", fontsize=13)
axes[0].set_title("この直線の傾きが「次元」", fontsize=15, pad=12)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[1].axhline(1.0, color="#333", lw=1.2, ls=":")
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("マス目の細かさ (2^k)", fontsize=13)
axes[1].set_ylabel("塗れたマス数 ÷ 理論の 3^k", fontsize=13)
axes[1].set_title("1.0 に貼りついていれば「理論どおり」", fontsize=15, pad=12)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)
fig.tight_layout(h_pad=2.4)
fig.savefig("fig4_ratio.png", dpi=110, bbox_inches="tight")
print("saved fig4_ratio.png")
