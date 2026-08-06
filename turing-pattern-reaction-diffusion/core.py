# Gray-Scottモデルの計算コア

import numpy as np
from scipy import ndimage

DU, DV = 0.16, 0.08
DT = 1.0


def laplacian(a):
    return (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)
            - 4 * a)


def simulate(F, k, steps, N=140, seed=0, du=DU, dv=DV, snap_at=()):
    rng = np.random.default_rng(seed)
    u = np.ones((N, N))
    v = np.zeros((N, N))
    # 中央に種を置き、わずかな乱れを乗せる
    s = N // 2
    r = N // 14
    u[s - r:s + r, s - r:s + r] = 0.50
    v[s - r:s + r, s - r:s + r] = 0.25
    u += 0.02 * rng.standard_normal((N, N))
    v += 0.02 * rng.standard_normal((N, N))
    snaps = {}
    for t in range(steps):
        uvv = u * v * v
        u += DT * (du * laplacian(u) - uvv + F * (1 - u))
        v += DT * (dv * laplacian(v) + uvv - (F + k) * v)
        np.clip(u, 0, 1, out=u)
        np.clip(v, 0, 1, out=v)
        if t + 1 in snap_at:
            snaps[t + 1] = v.copy()
    return u, v, snaps


def describe(v):
    """模様を数値で表す: 高濃度領域の個数・面積比・FFTの主波長"""
    m = v > 0.25
    lab, n = ndimage.label(m)
    frac = float(m.mean())
    # 空間FFTの主波長(直流成分を除いた動径平均のピーク)
    f = np.abs(np.fft.fftshift(np.fft.fft2(v - v.mean())))
    N = v.shape[0]
    cy = cx = N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(yy - cy, xx - cx).astype(int)
    prof = np.bincount(rr.ravel(), f.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    prof[:2] = 0
    kpk = int(np.argmax(prof[:N // 2]))
    wl = N / kpk if kpk > 0 else np.nan
    return {"blobs": int(n), "frac": frac, "wavelength": float(wl)}


