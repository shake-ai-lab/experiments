# ペンローズ・タイルをロビンソン三角形の細分で作る

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
OUT = {"phi": PHI}


def subdivide(triangles):
    """ロビンソン三角形の細分。各三角形は (種類, A, B, C) の複素数3点。
    種類0=鋭い三角形(太いひし形の半分), 種類1=鈍い三角形(細いひし形の半分)"""
    out = []
    for kind, A, B, C in triangles:
        if kind == 0:
            P = A + (B - A) / PHI
            out.append((0, C, P, B))
            out.append((1, P, C, A))
        else:
            Q = B + (A - B) / PHI
            Rp = B + (C - B) / PHI
            out.append((1, Rp, C, A))
            out.append((1, Q, Rp, B))
            out.append((0, Rp, Q, A))
    return out


def initial_wheel():
    tris = []
    for i in range(10):
        B = np.exp(1j * (2 * i - 1) * np.pi / 10)
        C = np.exp(1j * (2 * i + 1) * np.pi / 10)
        if i % 2 == 0:
            B, C = C, B
        tris.append((0, 0j, B, C))
    return tris


