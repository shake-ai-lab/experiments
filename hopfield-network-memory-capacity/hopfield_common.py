# -*- coding: utf-8 -*-
"""Hopfieldネットワーク記憶容量実験の共通モジュール。

32x32(=1024ニューロン)の±1パターンを扱う。
- 文字パターン: PILでMeiryoフォントのカタカナ/ひらがな/英数字/漢字を描画して二値化
- ランダムパターン: 各画素を等確率で±1
- 学習: ヘブ則 W = (1/N) Σ ξξ^T (対角0)
- 想起: 非同期更新(ランダム順)。局所場を差分更新して高速化
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

N_SIDE = 32
N = N_SIDE * N_SIDE
FONT_PATH = r"C:\Windows\Fonts\meiryo.ttc"


def build_charset(limit=300):
    """カタカナ→ひらがな→英数字→常用漢字の順で最大limit文字を返す。"""
    chars = [chr(c) for c in range(0x30A2, 0x30F7)]      # カタカナ(ァ小文字含む一部)
    chars += [chr(c) for c in range(0x3042, 0x3094)]     # ひらがな
    chars += list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    kanji = ("日月火水木金土山川田人口目耳手足力男女子学校先生年時分間国語算数"
             "理科社会音楽図工体育春夏秋冬雨雪風空海森林花草虫魚鳥犬猫牛馬米茶"
             "肉家門戸窓部屋台所電気車道橋駅店町村市都県島岸波光色形声音味"
             "東西南北左右上下中外内前後大小多少長短高低新古赤青白黒円玉本文字")
    chars += list(kanji)
    chars = list(dict.fromkeys(chars))
    assert len(chars) >= limit, f"charset too small: {len(chars)} < {limit}"
    return chars[:limit]


def render_char(ch, font):
    """1文字を32x32に描画し、±1のフラットなパターンにする。"""
    img = Image.new("L", (N_SIDE, N_SIDE), 0)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((N_SIDE - w) / 2 - bbox[0], (N_SIDE - h) / 2 - bbox[1]),
           ch, fill=255, font=font)
    a = np.asarray(img)
    return np.where(a > 96, 1, -1).astype(np.float32).ravel()


def make_char_patterns(n=300, font_size=30):
    font = ImageFont.truetype(FONT_PATH, font_size)
    chars = build_charset(n)
    pats = np.stack([render_char(c, font) for c in chars])
    return pats, chars


def make_random_patterns(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n, N)).astype(np.float32)


def train(patterns):
    """ヘブ則。patterns: (P, N)。"""
    X = patterns.astype(np.float32)
    W = X.T @ X / N
    np.fill_diagonal(W, 0.0)
    return W


def train_projection(patterns):
    """射影(擬似逆行列)ルール。相関のあるパターンでも記憶できる。"""
    X = patterns.astype(np.float64)
    G = X @ X.T / N
    W = X.T @ np.linalg.solve(G, X) / N
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def add_noise(p, frac, rng):
    """画素をfracの割合で反転させた手がかりを作る。"""
    s = p.copy()
    mask = rng.random(N) < frac
    s[mask] *= -1
    return s


def occlude(p, frac=0.5):
    """下からfracの割合を背景(-1)で塗りつぶした手がかりを作る。"""
    s = p.reshape(N_SIDE, N_SIDE).copy()
    cut = int(N_SIDE * (1 - frac))
    s[cut:, :] = -1
    return s.ravel()


def recall(W, s0, rng, max_sweeps=15, record_every=None):
    """非同期更新(ランダム順)。局所場hを差分更新する。

    record_every を与えると record_every 回の更新ごとの状態スナップショットも返す。
    返り値: (最終状態, スイープ数, スナップショットのリスト or None)
    """
    s = s0.astype(np.float32).copy()
    h = W @ s
    snaps = None
    if record_every is not None:
        snaps = [s.copy()]
    n_updates = 0
    sweeps_done = 0
    for sweep in range(max_sweeps):
        flips = 0
        for i in rng.permutation(N):
            new = 1.0 if h[i] >= 0 else -1.0
            if new != s[i]:
                h += W[:, i] * (2.0 * new)
                s[i] = new
                flips += 1
            n_updates += 1
            if record_every is not None and n_updates % record_every == 0:
                snaps.append(s.copy())
        sweeps_done = sweep + 1
        if flips == 0:
            break
    if record_every is not None and not np.array_equal(snaps[-1], s):
        snaps.append(s.copy())
    return s, sweeps_done, snaps


def overlap(a, b):
    return float(a @ b) / N


def classify_result(s, patterns, cue_idx, success_th=0.95, spurious_th=0.9):
    """想起結果を分類する。

    returns (label, m_cue, best_idx, best_abs_m)
      label: 'success' | 'other_memory' | 'spurious'
    """
    m_cue = overlap(s, patterns[cue_idx])
    ms = patterns @ s / N
    best_idx = int(np.argmax(np.abs(ms)))
    best_abs = float(abs(ms[best_idx]))
    if m_cue >= success_th:
        return "success", m_cue, best_idx, best_abs
    if best_abs >= spurious_th:
        return "other_memory", m_cue, best_idx, best_abs
    return "spurious", m_cue, best_idx, best_abs
