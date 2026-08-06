# ジュリア集合の計算コア

import numpy as np


def julia_escape(c, xmin, xmax, ymin, ymax, n, maxiter):
    """パラメータ c のジュリア集合。各画素の座標そのものを z0 として反復する。
    脱出しなかった画素(=充填ジュリア集合 K_c)を True で返す。"""
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    Z = (x[None, :] + 1j * y[:, None]).ravel()
    idx = np.arange(Z.size)
    Zc = Z.copy()
    alive = np.ones(Z.size, dtype=bool)
    for _ in range(maxiter):
        Zc = Zc * Zc + c
        esc = Zc.real * Zc.real + Zc.imag * Zc.imag > 4.0
        if esc.any():
            alive[idx[esc]] = False
            keep = ~esc
            idx = idx[keep]
            Zc = Zc[keep]
            if idx.size == 0:
                break
    return alive.reshape(n, n)


