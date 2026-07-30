"""ボーイ曲面(実射影平面 RP^2 の3次元への immersion)の共通ユーティリティ。

Bryant-Kusner のパラメータ表示: 単位円板 |w| <= 1 上で
    g1 = -3/2 Im[ w(1-w^4) / (w^6 + sqrt(5) w^3 - 1) ]
    g2 = -3/2 Re[ w(1+w^4) / (w^6 + sqrt(5) w^3 - 1) ]
    g3 = Im[ (1+w^6) / (w^6 + sqrt(5) w^3 - 1) ] - 1/2
    (x, y, z) = (g1, g2, g3) / (g1^2 + g2^2 + g3^2)

境界 |w|=1 では f(w) = f(-w) が成り立つ(step1で数値確認)。
つまり「円板の縁の対蹠点を貼り合わせる」という実射影平面の定義そのものが
このパラメータ表示に組み込まれている。
"""
import numpy as np

SQRT5 = np.sqrt(5.0)
# 分母 w^6 + sqrt(5) w^3 - 1 の零点: w^3 = (3-sqrt(5))/2。
# ここで g が発散し、反転 (g/|g|^2) で原点に写る = 三重点の予言。
POLE_RADIUS = ((3 - SQRT5) / 2) ** (1 / 3)
POLE_ANGLES = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])


def boy(w):
    """Bryant-Kusner 表示。w: complex ndarray (|w|<=1) -> (…,3) 実座標。"""
    w = np.asarray(w, dtype=complex)
    d = w**6 + SQRT5 * w**3 - 1
    g1 = -1.5 * np.imag(w * (1 - w**4) / d)
    g2 = -1.5 * np.real(w * (1 + w**4) / d)
    g3 = np.imag((1 + w**6) / d) - 0.5
    s = g1**2 + g2**2 + g3**2
    return np.stack([g1 / s, g2 / s, g3 / s], axis=-1)


def boy_normal(w, h=1e-6):
    """有限差分による単位法線ベクトル (∂f/∂Re(w) × ∂f/∂Im(w))。"""
    w = np.asarray(w, dtype=complex)
    du = (boy(w + h) - boy(w - h)) / (2 * h)
    dv = (boy(w + 1j * h) - boy(w - 1j * h)) / (2 * h)
    n = np.cross(du, dv)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def sample_disk(n, seed=0, r_max=1.0):
    """パラメータ円板から一様サンプリング。"""
    rng = np.random.default_rng(seed)
    r = r_max * np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    return r * np.exp(1j * t)


def antipode(w):
    """円板モデルでの同一視相手 w -> -1/conj(w) (境界上では -w に一致)。"""
    return -1.0 / np.conj(w)


def param_dist(w1, w2):
    """同一視を考慮したパラメータ空間での距離(RP^2 上の距離の代理)。"""
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    return np.minimum(np.abs(w1 - w2),
                      np.minimum(np.abs(w1 - antipode(w2)),
                                 np.abs(antipode(w1) - w2)))
