# ニュートン法の吸引域を計算するコア

import numpy as np


def newton_basins(f, df, roots, R, extent=2.0, maxiter=60, tol=1e-6):
    """各画素の初期値からニュートン法を回し、どの根に落ちたか(-1=どれでもない)と
    反復回数を返す。"""
    x = np.linspace(-extent, extent, R)
    Z = (x[None, :] + 1j * x[:, None]).ravel()
    which = np.full(Z.size, -1, dtype=np.int8)
    iters = np.full(Z.size, maxiter, dtype=np.int16)
    idx = np.arange(Z.size)
    for i in range(maxiter):
        d = df(Z)
        d[d == 0] = 1e-18
        Z = Z - f(Z) / d
        done = np.zeros(Z.size, dtype=bool)
        for r, root in enumerate(roots):
            hit = (np.abs(Z - root) < tol) & ~done
            which[idx[hit]] = r
            iters[idx[hit]] = i + 1
            done |= hit
        if done.any():
            keep = ~done
            idx = idx[keep]
            Z = Z[keep]
            if idx.size == 0:
                break
    return which.reshape(R, R), iters.reshape(R, R)


def boundary_mask(w):
    b = np.zeros_like(w, dtype=bool)
    b[1:, :] |= w[1:, :] != w[:-1, :]
    b[:-1, :] |= w[:-1, :] != w[1:, :]
    b[:, 1:] |= w[:, 1:] != w[:, :-1]
    b[:, :-1] |= w[:, :-1] != w[:, 1:]
    return b


def boxdim(mask, extent, sizes=(4, 8, 16, 32, 64)):
    R = mask.shape[0]
    counts = np.array([int(mask[: (R // b) * b, : (R // b) * b]
                           .reshape(R // b, b, R // b, b).any(axis=(1, 3)).sum())
                       for b in sizes])
    eps = np.array(sizes, float) * extent / R
    ok = counts > 0
    return float(np.polyfit(np.log(1 / eps[ok]), np.log(counts[ok]), 1)[0]), counts


