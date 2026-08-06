# ラングトンのアリの計算コア

import numpy as np


DIRS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])   # 右回りに 上/右/下/左
OUT = {}


def run_ant(steps, N=520, rule="RL", init=None, seed=0, record_every=0):
    """rule は各色でどちらに曲がるかの文字列。'RL' が古典的なラングトンのアリ。
    盤は N x N のトーラス(はみ出さない大きさを取る)。"""
    k = len(rule)
    grid = np.zeros((N, N), dtype=np.uint8)
    if init is not None:
        grid = init.copy()
    pos = np.array([N // 2, N // 2])
    d = 0
    turns = np.array([1 if ch == "R" else -1 for ch in rule])
    traj = np.empty((steps, 2), dtype=np.int32)
    black = np.empty(steps, dtype=np.int32) if record_every else None
    snaps = {}
    nb = int((grid > 0).sum())
    for t in range(steps):
        traj[t] = pos
        c = grid[pos[0], pos[1]]
        d = (d + turns[c]) % 4
        nc = (c + 1) % k
        nb += (1 if nc else 0) - (1 if c else 0)
        grid[pos[0], pos[1]] = nc
        pos = (pos + DIRS[d]) % N
        if record_every:
            black[t] = nb
            if t in record_every if isinstance(record_every, set) else False:
                snaps[t] = grid.copy()
    return grid, traj, black, snaps


def highway_start(traj, period=104, confirm=6):
    """周期 period ごとの変位が一定になり、それが confirm 回続く最初の時刻を返す。"""
    n = len(traj)
    if n < period * (confirm + 2):
        return None, None
    dif = traj[period:] - traj[:-period]
    for t in range(n - period * (confirm + 1)):
        d0 = dif[t]
        if abs(d0[0]) + abs(d0[1]) == 0:
            continue
        if np.all(dif[t:t + period * confirm] == d0):
            return t, tuple(int(v) for v in d0)
    return None, None


