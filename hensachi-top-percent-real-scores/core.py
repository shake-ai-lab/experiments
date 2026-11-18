"""大学入試センター 令和7年度 共通テスト本試験 科目別成績分布(CSV)の読み込みと、
偏差値ごとの「実際に上回った人の割合」と正規分布の予測の比較。"""
import csv
import io
import os
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "dist_raw.csv")


def load_subjects(path=RAW):
    """科目名 -> dict(満点, 平均, 標準偏差, 受験者数, counts[0..満点])"""
    if not os.path.exists(path):
        # 元データはこのリポジトリに入れていない（大学入試センターの公開データ）。
        raise FileNotFoundError(
            "data/dist_raw.csv がありません。大学入試センター「令和7年度 試験情報データ（本試験）」\n"
            "https://www.dnc.ac.jp/kyotsu/hyouka/r7_hyouka/r7_data.html\n"
            "の「02 科目別成績分布」CSV をダウンロードし、data/dist_raw.csv として保存してください（文字コードは cp932 のまま）。")
    rows =list(csv.reader(io.StringIO(open(path, encoding="cp932").read())))
    out = {}
    i = 0
    while i < len(rows) - 2:
        r = rows[i]
        nxt = rows[i + 1]
        # 科目ヘッダ行: 次の行が「得点,人数,比率」
        if r[0] and nxt[0] == "点数" and len(r) > 6 and r[1].replace(".", "").isdigit():
            name = r[0].replace("　", "").strip()
            full = int(r[1])
            counts = np.zeros(full + 1, dtype=np.int64)
            j = i + 2
            while j < len(rows):
                c0 = rows[j][0]
                if c0.startswith("令和") or c0 == "":
                    break
                try:
                    s = int(c0)
                except ValueError:
                    j += 1  # 「１段階」などの小計行は読み飛ばす
                    continue
                counts[s] = int(rows[j][1])
                j += 1
            if name in out:
                if out[name]["full"] == full:
                    # 同名・同満点が2回出てくる場合は、中身が同じことを確かめて最初の方を使う
                    assert (out[name]["counts"] == counts).all(), name
                    i = j
                    continue
                name = f"{name}(満点{full})"
            out[name] = dict(
                full=full,
                mean_csv=float(r[4]),
                sd_csv=float(r[5]),
                n_csv=int(r[6]),
                counts=counts,
            )
            i = j
        else:
            i += 1
    return out


def moments(counts):
    s = np.arange(len(counts))
    n = counts.sum()
    m = (s * counts).sum() / n
    sd = np.sqrt((((s - m) ** 2) * counts).sum() / n)  # 母標準偏差(全受験者)
    return n, m, sd


def hensachi_threshold_row(counts, hs=(60, 65, 70)):
    """偏差値 h に必要な得点(= mean + (h-50)/10*sd)以上の人の実際の割合と、正規分布の予測。
    「ちょうど閾値の人」を含む/含まないを揃えるため「得点 >= 閾値(切り上げ)」を使う。"""
    n, m, sd = moments(counts)
    s = np.arange(len(counts))
    res = {}
    for h in hs:
        z = (h - 50) / 10
        thr = m + z * sd
        k = int(np.ceil(thr - 1e-12))
        if k > len(counts) - 1:
            actual = 0.0
        else:
            actual = counts[max(k, 0):].sum() / n
        res[h] = dict(thr=thr, k=k, actual=actual, normal=1 - norm.cdf(z))
    return res


def skewness(counts):
    s = np.arange(len(counts))
    n, m, sd = moments(counts)
    return (((s - m) ** 3) * counts).sum() / n / sd**3
