# 繰り返し囚人のジレンマ: しっぺ返し(Tit for Tat)は本当に強いのか
#
# 実験1: ノイズなしの総当たり戦(Axelrodの第1回大会の再現)
# 実験2: 手が1%の確率で意図と逆になる「手違いのある世界」で同じことをやる
# 実験3: 手違い率を0%から20%まで振り、順位がどう入れ替わるかを見る
# 実験4: 勝った戦略が増える進化シミュレーション
#
# 利得: 両方協調=3点ずつ / 両方裏切り=1点ずつ / 裏切った側5点・協調した側0点

import json

import numpy as np

C, D = 0, 1                      # 0 = 協調(Cooperate), 1 = 裏切り(Defect)
PAY = np.array([[[3, 3], [0, 5]],
                [[5, 0], [1, 1]]])  # PAY[自分の手][相手の手] = (自分の点, 相手の点)


# ----------------------------------------------------------------- 戦略
def allc(me, opp, rng):
    return C


def alld(me, opp, rng):
    return D


def tft(me, opp, rng):
    return C if not opp else opp[-1]


def stft(me, opp, rng):          # 疑心暗鬼のしっぺ返し: 初手だけ裏切る
    return D if not opp else opp[-1]


def tf2t(me, opp, rng):          # 二度続けて裏切られたら仕返す
    return D if len(opp) >= 2 and opp[-1] == D and opp[-2] == D else C


def grudger(me, opp, rng):       # 一度でも裏切られたら以後ずっと裏切る
    return D if D in opp else C


def random_p(me, opp, rng):
    return int(rng.random() < 0.5)


def pavlov(me, opp, rng):        # 勝てばそのまま、負ければ変える
    if not me:
        return C
    good = PAY[me[-1]][opp[-1]][0] >= 3
    return me[-1] if good else 1 - me[-1]


def joss(me, opp, rng):          # しっぺ返しだが10%は抜き打ちで裏切る
    if not opp:
        return C
    return D if rng.random() < 0.1 else opp[-1]


def gtft(me, opp, rng):          # 寛容なしっぺ返し: 裏切られても10%は水に流す
    if not opp:
        return C
    if opp[-1] == D and rng.random() < 0.1:
        return C
    return opp[-1]


STRATEGIES = [
    ("お人好し (ALLC)", allc),
    ("裏切り者 (ALLD)", alld),
    ("しっぺ返し (TFT)", tft),
    ("疑心暗鬼のしっぺ返し (STFT)", stft),
    ("二度までは許す (TF2T)", tf2t),
    ("根に持つ (GRUDGER)", grudger),
    ("気まぐれ (RANDOM)", random_p),
    ("パブロフ (PAVLOV)", pavlov),
    ("ジョス (JOSS)", joss),
    ("寛容なしっぺ返し (GTFT)", gtft),
]
NAMES = [s[0] for s in STRATEGIES]
SHORT = [n.split("(")[1].rstrip(")") for n in NAMES]


def play(f1, f2, rounds, noise, rng):
    h1, h2 = [], []
    s1 = s2 = 0
    for _ in range(rounds):
        a = f1(h1, h2, rng)
        b = f2(h2, h1, rng)
        if noise:                       # 手違い: 意図と逆の手が出てしまう
            if rng.random() < noise:
                a = 1 - a
            if rng.random() < noise:
                b = 1 - b
        p, q = PAY[a][b]
        s1 += p
        s2 += q
        h1.append(a)
        h2.append(b)
    return s1, s2, h1, h2


def tournament(noise, rounds=200, reps=40, seed=0):
    """総当たり(自分自身とも対戦する)。1試合あたりの平均得点の行列を返す。"""
    rng = np.random.default_rng(seed)
    n = len(STRATEGIES)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            tot = 0.0
            for _ in range(reps):
                s1, _, _, _ = play(STRATEGIES[i][1], STRATEGIES[j][1], rounds, noise, rng)
                tot += s1
            M[i, j] = tot / reps / rounds     # 1手あたりの平均得点
    return M


OUT = {"names": NAMES, "short": SHORT}

# ================================================================ 実験1
print("[実験1] 手違いなしの総当たり戦 (200手 x 40回 x 全ペア)", flush=True)
M0 = tournament(0.0, seed=1)
tot0 = M0.mean(axis=1)
order0 = np.argsort(-tot0)
for r, i in enumerate(order0, 1):
    print(f"  {r:>2}位 {NAMES[i]:<28} 1手あたり {tot0[i]:.4f}")
OUT["no_noise"] = {"matrix": M0.tolist(), "total": tot0.tolist()}

# ================================================================ 実験2
print("\n[実験2] 1%の確率で手が意図と逆になる世界", flush=True)
M1 = tournament(0.01, seed=2)
tot1 = M1.mean(axis=1)
order1 = np.argsort(-tot1)
for r, i in enumerate(order1, 1):
    prev = int(np.where(order0 == i)[0][0]) + 1
    arrow = "→" if prev == r else ("↑" if prev > r else "↓")
    print(f"  {r:>2}位 {NAMES[i]:<28} {tot1[i]:.4f}   (手違いなしでは{prev}位 {arrow})")
OUT["noise_1pct"] = {"matrix": M1.tolist(), "total": tot1.tolist()}

# TFT同士がノイズ下でどうなるか
rng = np.random.default_rng(7)
s1, s2, h1, h2 = play(tft, tft, 60, 0.01, rng)
print(f"\n  参考: しっぺ返し同士を60手(手違い1%)戦わせた1例 -> {s1}点 vs {s2}点 "
      f"(手違いが無ければ180点ずつ)")
rng = np.random.default_rng(7)
_, _, g1, g2 = play(gtft, gtft, 60, 0.01, rng)
OUT["echo"] = {"tft": [h1, h2], "gtft": [g1, g2]}

# ================================================================ 実験3
print("\n[実験3] 手違い率を0%から20%まで振る", flush=True)
noises = [0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20]
sweep = []
for nz in noises:
    M = tournament(nz, reps=25, seed=100 + int(nz * 1000))
    t = M.mean(axis=1)
    sweep.append(t.tolist())
    win = int(np.argmax(t))
    print(f"  手違い {nz*100:>5.1f}%: 1位 {NAMES[win]:<28} ({t[win]:.4f})", flush=True)
OUT["sweep"] = {"noises": noises, "scores": sweep}

# ================================================================ 実験4
print("\n[実験4] 進化シミュレーション(得点の高い戦略が次世代で増える)", flush=True)


def evolve(M, gens=120, start=None):
    n = M.shape[0]
    x = np.ones(n) / n if start is None else start.copy()
    hist = [x.copy()]
    for _ in range(gens):
        fit = M @ x                       # 現在の人口構成に対する平均得点
        x = x * fit / (x @ fit)           # レプリケータ動態(離散版)
        x = np.maximum(x, 0)
        x /= x.sum()
        hist.append(x.copy())
    return np.array(hist)


for label, M in [("手違いなし", M0), ("手違い1%", M1)]:
    h = evolve(M)
    final = h[-1]
    top = np.argsort(-final)[:3]
    print(f"  {label}: 120世代後 " +
          " / ".join(f"{SHORT[i]} {final[i]*100:.1f}%" for i in top))
    OUT[f"evolve_{'0' if label == '手違いなし' else '1'}"] = h.tolist()

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json")
