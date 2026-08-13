"""「クイックソートが最速」を入力の種類を変えて実測する。

計測は**比較回数**で行う。実時間だとPython標準の sorted() だけがC実装なので、
自作アルゴリズムと比べても実装言語の差を見ているだけになる。比較回数なら
アルゴリズムそのものの仕事量を比べられる。実時間は別枠で測る。

結果は results.json に落とす。図は figures.py。
"""

import json
import random
import sys
import time

sys.setrecursionlimit(100000)


class Counter:
    """比較のたびに数える。a < b の判定を全部ここに通す"""

    def __init__(self):
        self.n = 0

    def lt(self, a, b):
        self.n += 1
        return a < b


# ------------------------------------------------------------------ 各アルゴリズム
def bubble(a, c):
    a = a[:]
    for i in range(len(a)):
        swapped = False
        for j in range(len(a) - i - 1):
            if c.lt(a[j + 1], a[j]):
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:  # 一度も交換がなければ整列済み
            break
    return a


def insertion(a, c):
    a = a[:]
    for i in range(1, len(a)):
        v = a[i]
        j = i - 1
        while j >= 0 and c.lt(v, a[j]):
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = v
    return a


def selection(a, c):
    a = a[:]
    for i in range(len(a)):
        m = i
        for j in range(i + 1, len(a)):
            if c.lt(a[j], a[m]):
                m = j
        a[i], a[m] = a[m], a[i]
    return a


def merge_sort(a, c):
    if len(a) <= 1:
        return a[:]
    mid = len(a) // 2
    left, right = merge_sort(a[:mid], c), merge_sort(a[mid:], c)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if c.lt(right[j], left[i]):
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def quick_naive(a, c):
    """教科書どおりの素朴なクイックソート。先頭をピボットにする"""
    if len(a) <= 1:
        return a[:]
    pivot, rest = a[0], a[1:]
    # 内包表記を2本書くと各要素を2回比較してしまい、比較回数が倍に出る。
    # 他のアルゴリズムと比べるので、1要素あたり1回で振り分ける。
    lo, hi = [], []
    for x in rest:
        (lo if c.lt(x, pivot) else hi).append(x)
    return quick_naive(lo, c) + [pivot] + quick_naive(hi, c)


def quick_random(a, c):
    """ピボットを乱択する。最悪ケースを狙い撃ちされないための定番の対処"""
    if len(a) <= 1:
        return a[:]
    k = random.randrange(len(a))
    a = a[:]
    a[0], a[k] = a[k], a[0]
    pivot, rest = a[0], a[1:]
    lo, hi = [], []
    for x in rest:
        (lo if c.lt(x, pivot) else hi).append(x)
    return quick_random(lo, c) + [pivot] + quick_random(hi, c)


def quick_three_way(a, c):
    """ピボットと等しい要素をまとめる3分割。重複の多い入力に効く"""
    if len(a) <= 1:
        return a[:]
    k = random.randrange(len(a))
    pivot = a[k]
    lo, eq, hi = [], [], []
    for x in a:
        if c.lt(x, pivot):
            lo.append(x)
        elif c.lt(pivot, x):
            hi.append(x)
        else:
            eq.append(x)
    return quick_three_way(lo, c) + eq + quick_three_way(hi, c)


def heap_sort(a, c):
    a = a[:]
    n = len(a)

    def sift(root, end):
        while True:
            child = 2 * root + 1
            if child >= end:
                return
            if child + 1 < end and c.lt(a[child], a[child + 1]):
                child += 1
            if not c.lt(a[root], a[child]):
                return
            a[root], a[child] = a[child], a[root]
            root = child

    for i in range(n // 2 - 1, -1, -1):
        sift(i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift(0, end)
    return a


ALGOS = {
    "バブルソート": bubble,
    "選択ソート": selection,
    "挿入ソート": insertion,
    "マージソート": merge_sort,
    "ヒープソート": heap_sort,
    "クイックソート(先頭ピボット)": quick_naive,
    "クイックソート(乱択ピボット)": quick_random,
    "クイックソート(3分割)": quick_three_way,
}

# 二乗オーダーのものは大きい n で現実的な時間に終わらないので上限を分ける
QUADRATIC = {"バブルソート", "選択ソート", "挿入ソート", "クイックソート(先頭ピボット)"}


# ------------------------------------------------------------------ 入力の種類
def make_input(kind, n, rng):
    if kind == "ランダム":
        a = list(range(n))
        rng.shuffle(a)
        return a
    if kind == "ソート済み":
        return list(range(n))
    if kind == "逆順":
        return list(range(n - 1, -1, -1))
    if kind == "ほぼ整列済み":  # 1%だけ入れ替える
        a = list(range(n))
        for _ in range(max(1, n // 100)):
            i, j = rng.randrange(n), rng.randrange(n)
            a[i], a[j] = a[j], a[i]
        return a
    if kind == "重複だらけ":  # 値が10種類しかない
        return [rng.randrange(10) for _ in range(n)]
    raise ValueError(kind)


KINDS = ["ランダム", "ソート済み", "逆順", "ほぼ整列済み", "重複だらけ"]

out = {"comparisons": {}, "scaling": {}, "crossover": {}, "wall": {}}

# ------------------------------------------------- 1. 入力の種類ごとの比較回数
N = 2000
rng = random.Random(20260814)
for kind in KINDS:
    data = make_input(kind, N, rng)
    out["comparisons"][kind] = {}
    for name, fn in ALGOS.items():
        random.seed(20260814)  # 乱択ピボットを再現可能にする
        c = Counter()
        got = fn(data, c)
        assert got == sorted(data), f"{name} が {kind} で壊れている"
        out["comparisons"][kind][name] = c.n
        print(f"{kind:8s} {name:24s} {c.n:>12,}", flush=True)

# ------------------------------------------------- 2. n を変えたときの伸び方
SIZES = [100, 200, 400, 800, 1600, 3200, 6400]
for name, fn in ALGOS.items():
    out["scaling"][name] = {}
    for n in SIZES:
        if name in QUADRATIC and n > 3200:
            continue
        data = make_input("ランダム", n, rng)
        random.seed(20260814)
        c = Counter()
        fn(data, c)
        out["scaling"][name][n] = c.n
    print(f"scaling {name} done", flush=True)

# 先頭ピボットのクイックソートを、ソート済み入力で n を変えて回す(最悪ケース)
out["worst_case"] = {}
for n in [100, 200, 400, 800, 1600]:
    data = make_input("ソート済み", n, rng)
    c = Counter()
    quick_naive(data, c)
    out["worst_case"][n] = c.n
    # 比較のため理論値も置く
    print(f"worst n={n} {c.n:,} (n(n-1)/2 = {n*(n-1)//2:,})", flush=True)

# ------------------------------------------------- 3. 小さい n での逆転
out["crossover"] = {"挿入ソート": {}, "マージソート": {}, "クイックソート(乱択ピボット)": {}}
for n in range(2, 65, 2):
    for name in out["crossover"]:
        total = 0
        for t in range(200):  # 小さい n はばらつくので平均を取る
            data = make_input("ランダム", n, rng)
            random.seed(t)
            c = Counter()
            ALGOS[name](data, c)
            total += c.n
        out["crossover"][name][n] = total / 200
print("crossover done", flush=True)

# ------------------------------------------------- 4. 実時間(参考)
# Python標準の sorted() はC実装のTimsortなので、自作と実時間で比べても
# 実装言語の差しか見えない。それでも「実務でどれを使うか」の答えにはなるので測る。
out["wall"] = {}
for kind in KINDS:
    data = make_input(kind, 200_000, rng)
    t0 = time.perf_counter()
    sorted(data)
    out["wall"][kind] = time.perf_counter() - t0
    print(f"sorted() {kind:8s} {out['wall'][kind]*1000:.1f} ms", flush=True)

json.dump(out, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote results.json")
