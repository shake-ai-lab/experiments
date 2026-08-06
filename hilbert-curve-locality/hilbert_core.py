# ヒルベルト曲線・Z階数・行優先の索引を作る

import numpy as np


def hilbert_d2xy(order, d):
    """曲線上の位置dから座標(x,y)へ。orderは2^order x 2^order の格子。"""
    n = 1 << order
    x = np.zeros_like(d)
    y = np.zeros_like(d)
    t = d.copy()
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        # 回転
        swap = ry == 0
        x_, y_ = x.copy(), y.copy()
        flip = swap & (rx == 1)
        x_[flip] = s - 1 - x[flip]
        y_[flip] = s - 1 - y[flip]
        x[swap], y[swap] = y_[swap], x_[swap]
        x = x + s * rx
        y = y + s * ry
        t = t // 4
        s <<= 1
    return x, y


def hilbert_index(order):
    """各格子点に「曲線上の順番」を振った 2^order x 2^order の配列"""
    n = 1 << order
    d = np.arange(n * n)
    x, y = hilbert_d2xy(order, d)
    idx = np.zeros((n, n), dtype=np.int64)
    idx[y, x] = d
    return idx


def morton_index(order):
    n = 1 << order
    yy, xx = np.mgrid[0:n, 0:n]
    z = np.zeros((n, n), dtype=np.int64)
    for b in range(order):
        z |= ((xx >> b) & 1) << (2 * b)
        z |= ((yy >> b) & 1) << (2 * b + 1)
    return z


def row_index(order):
    n = 1 << order
    return np.arange(n * n).reshape(n, n)


