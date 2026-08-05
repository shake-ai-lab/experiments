# マンデルブロ集合の実験で共通に使う関数群。
# escape_time: 脱出時間の計算 / circle_mask, koch_mask: box-counting の検証用図形

import numpy as np

# ---------------------------------------------------------------- 逃避時間
def escape_time(xmin, xmax, ymin, ymax, w, h, maxiter, chunk=256):
    """脱出時間(0 = maxiter回でも脱出しなかった=集合内とみなす)を返す。
    メモリ節約のため行方向にチャンク分割して計算する。"""
    xs = np.linspace(xmin, xmax, w, dtype=np.float64)
    ys = np.linspace(ymin, ymax, h, dtype=np.float64)
    out = np.zeros((h, w), dtype=np.int32)
    for r0 in range(0, h, chunk):
        r1 = min(r0 + chunk, h)
        C = (xs[None, :] + 1j * ys[r0:r1, None]).ravel()
        n = np.zeros(C.size, dtype=np.int32)

        # メインカージオイドと周期2の球は解析的に内部と分かるので、最初から除外する。
        # (これを入れないと、集合内部の点が毎回maxiter回まで回り続けて非常に遅い)
        q = (C.real - 0.25) ** 2 + C.imag**2
        in_cardioid = q * (q + (C.real - 0.25)) <= 0.25 * C.imag**2
        in_bulb = (C.real + 1.0) ** 2 + C.imag**2 <= 0.0625

        # 生き残っている点だけを詰め直しながら回す(毎回フル配列をマスクすると遅い)
        idx = np.flatnonzero(~(in_cardioid | in_bulb))
        Cc = C[idx]
        Zc = np.zeros(idx.size, dtype=np.complex128)
        for i in range(maxiter):
            Zc = Zc * Zc + Cc
            esc = Zc.real * Zc.real + Zc.imag * Zc.imag > 4.0
            if esc.any():
                n[idx[esc]] = i + 1
                keep = ~esc
                idx = idx[keep]
                Zc = Zc[keep]
                Cc = Cc[keep]
                if idx.size == 0:
                    break
        out[r0:r1] = n.reshape(r1 - r0, w)
    return out


# ---------------------------------------------------------------- box counting
def boundary_box_count(inside, sizes):
    """inside: bool配列(集合内=True)。
    「内側の画素と外側の画素が両方入っている箱」の個数を数える = 境界を覆う箱の数。"""
    R = inside.shape[0]
    counts = []
    for b in sizes:
        k = R // b
        blk = inside[: k * b, : k * b].reshape(k, b, k, b)
        has_in = blk.any(axis=(1, 3))
        has_out = (~blk).any(axis=(1, 3))
        counts.append(int((has_in & has_out).sum()))
    return np.array(counts)


def fit_dimension(sizes, counts, extent, R):
    """log N = -D log ε の傾きから次元を出す。εは実座標での箱の一辺。"""
    eps = np.array(sizes, dtype=float) * extent / R
    ok = counts > 0
    D, intercept = np.polyfit(np.log(1 / eps[ok]), np.log(counts[ok]), 1)
    return D, eps, counts


# ---------------------------------------------------------------- 検証用の図形
def circle_mask(R, extent=2.6):
    t = np.linspace(-extent / 2, extent / 2, R)
    X, Y = np.meshgrid(t, t)
    return X**2 + Y**2 <= 1.0**2


def koch_snowflake_vertices(order):
    """コッホ雪片の頂点列。境界のフラクタル次元は log4/log3 = 1.26186"""
    def koch(p0, p1, depth):
        if depth == 0:
            return [p0]
        p0, p1 = np.array(p0), np.array(p1)
        d = (p1 - p0) / 3.0
        a = p0 + d
        b = p0 + 2 * d
        # 60度回転して尖りを作る
        rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
        peak = a + rot @ d
        return (
            koch(p0, a, depth - 1)
            + koch(a, peak, depth - 1)
            + koch(peak, b, depth - 1)
            + koch(b, p1, depth - 1)
        )

    tri = [
        (np.cos(np.pi / 2), np.sin(np.pi / 2)),
        (np.cos(np.pi / 2 + 2 * np.pi / 3), np.sin(np.pi / 2 + 2 * np.pi / 3)),
        (np.cos(np.pi / 2 + 4 * np.pi / 3), np.sin(np.pi / 2 + 4 * np.pi / 3)),
    ]
    pts = []
    for i in range(3):
        pts += koch(tri[i], tri[(i + 1) % 3], order)
    return np.array(pts)


def koch_mask(R, order=6, extent=2.6):
    """PILのポリゴン塗りつぶしでラスタライズする。
    matplotlibのPath.contains_pointsは 4096^2 画素 x 12288頂点 で現実的な時間に終わらない。"""
    from PIL import Image, ImageDraw

    verts = koch_snowflake_vertices(order)
    # 実座標 -> 画素座標
    px = (verts[:, 0] + extent / 2) / extent * (R - 1)
    py = (verts[:, 1] + extent / 2) / extent * (R - 1)
    img = Image.new("1", (R, R), 0)
    ImageDraw.Draw(img).polygon(list(zip(px.tolist(), py.tolist())), fill=1)
    return np.array(img, dtype=bool)


