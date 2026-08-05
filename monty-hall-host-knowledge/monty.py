# モンティ・ホール問題を実測する。
#
# 実験1: 変更する/しない を100万回
# 実験2: 司会者が中身を知らずに開けて、たまたまヤギだった場合(見た目は同じなのに確率が違うはず)
# 実験3: ドアをN枚に一般化し、司会者がk枚開ける場合の勝率
# 実験4: 「変更したほうが得だ」と5%水準で言い切るには何回試せばいいか
#
# 実行: C:/bin/Python/python310-64/python.exe -u monty.py

import json

import numpy as np

rng = np.random.default_rng(20260805)
OUT = {}
TRIALS = 1_000_000

# ================================================================ 実験1・2
print(f"[実験1] 3枚のドア、{TRIALS:,}回", flush=True)
car = rng.integers(0, 3, TRIALS)
pick = rng.integers(0, 3, TRIALS)

# 司会者(中身を知っている)は、プレイヤーの選択でもなく車でもないドアを開ける
r = rng.random(TRIALS)
host = np.empty(TRIALS, dtype=np.int64)
for i in range(TRIALS // 500_000 + 1):
    pass
# 3枚しかないので、候補は最大2枚。ベクトル化して選ぶ
cand = np.stack([(pick + 1) % 3, (pick + 2) % 3], axis=1)   # プレイヤーが選んでいない2枚
is_car = cand == car[:, None]
only_one = is_car.any(axis=1)          # 片方が車 -> もう片方を開けるしかない
host = np.where(only_one,
                np.where(is_car[:, 0], cand[:, 1], cand[:, 0]),
                np.where(r < 0.5, cand[:, 0], cand[:, 1]))   # どちらもヤギなら等確率

stay_win = (pick == car)
switch = 3 - pick - host                # 残り1枚
switch_win = (switch == car)

p_stay = stay_win.mean()
p_switch = switch_win.mean()
print(f"  変更しない: {p_stay:.5f}  (理論 1/3 = 0.33333)")
print(f"  変更する  : {p_switch:.5f}  (理論 2/3 = 0.66667)")
OUT["classic"] = {"trials": TRIALS, "stay": float(p_stay), "switch": float(p_switch)}

print(f"\n[実験2] 司会者が中身を知らない場合", flush=True)
# 司会者はプレイヤーが選ばなかった2枚から等確率に開ける。車を開けてしまった回は「その回は無効」
open_idx = np.where(r < 0.5, cand[:, 0], cand[:, 1])
revealed_car = open_idx == car
valid = ~revealed_car
sw2 = 3 - pick - open_idx
p_switch_ignorant = (sw2[valid] == car[valid]).mean()
p_stay_ignorant = (pick[valid] == car[valid]).mean()
print(f"  司会者が車を開けてしまった回: {revealed_car.mean()*100:.2f}% (これは問題として成立しないので除外)")
print(f"  残った回で変更しない: {p_stay_ignorant:.5f}")
print(f"  残った回で変更する  : {p_switch_ignorant:.5f}  (理論 1/2 = 0.50000)")
OUT["ignorant_host"] = {"invalid_frac": float(revealed_car.mean()),
                        "stay": float(p_stay_ignorant), "switch": float(p_switch_ignorant)}

# ================================================================ 実験3
print(f"\n[実験3] ドアN枚、司会者がk枚開ける", flush=True)
T3 = 200_000
rows = []
for N in [3, 4, 5, 10, 20, 50, 100]:
    for k in sorted(set([1, max(1, N // 2), N - 2])):
        if k > N - 2:
            continue
        c = rng.integers(0, N, T3)
        pk = rng.integers(0, N, T3)
        # 司会者が k枚のヤギを開けたあと、残るのは (N - 1 - k) 枚。
        # プレイヤーが車を引いていれば残り全部ヤギ、引いていなければ残りに必ず車が1枚ある。
        stay = (pk == c)
        # 変更先は残りから等確率に1枚
        win_switch = np.where(stay, False, rng.random(T3) < 1.0 / (N - 1 - k))
        theo_stay = 1 / N
        theo_switch = (1 - 1 / N) / (N - 1 - k)
        rows.append({"N": N, "k": k, "stay": float(stay.mean()),
                     "switch": float(win_switch.mean()),
                     "theo_stay": theo_stay, "theo_switch": theo_switch})
        print(f"  N={N:>3} k={k:>3}: 変更しない {stay.mean():.4f} (理論 {theo_stay:.4f}) / "
              f"変更する {win_switch.mean():.4f} (理論 {theo_switch:.4f})")
OUT["general"] = rows

# ================================================================ 実験4
print(f"\n[実験4] 「変更が有利」と5%水準で言い切るのに必要な試行回数", flush=True)
# 二項検定(帰無仮説 p=1/2)を逐次的に見て、初めて有意になった試行数を記録する
from scipy import stats  # noqa: E402

REPS = 2000
MAXN = 400
need = []
for rep in range(REPS):
    wins = rng.random(MAXN) < 2 / 3
    cw = np.cumsum(wins)
    n = np.arange(1, MAXN + 1)
    # 片側二項検定のp値
    pv = stats.binom.sf(cw - 1, n, 0.5)
    hit = np.flatnonzero(pv < 0.05)
    need.append(int(n[hit[0]]) if hit.size else MAXN + 1)
need = np.array(need)
q = np.percentile(need[need <= MAXN], [25, 50, 75, 90])
print(f"  中央値 {np.median(need[need <= MAXN]):.0f}回 / 四分位 {q[0]:.0f}, {q[2]:.0f} / 90%点 {q[3]:.0f}回")
print(f"  {MAXN}回やっても有意にならなかった割合: {(need > MAXN).mean()*100:.2f}%")
OUT["power"] = {"reps": REPS, "maxn": MAXN, "need": need.tolist(),
                "median": float(np.median(need[need <= MAXN])),
                "q25": float(q[0]), "q75": float(q[2]), "q90": float(q[3]),
                "fail_frac": float((need > MAXN).mean())}

# 収束のようす(並走ラン)
RUNS, LEN = 40, 3000
w = rng.random((RUNS, LEN)) < 2 / 3
cum = np.cumsum(w, axis=1) / np.arange(1, LEN + 1)
OUT["convergence"] = cum[:, ::5].tolist()

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False)
print("\n保存: results.json")
