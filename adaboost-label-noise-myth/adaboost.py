"""numpy自作AdaBoost (決定株ベース, 離散SAMME/元祖AdaBoost.M1相当の2クラス版)"""
import numpy as np


def train_stump(X, y, w):
    """重み付き0-1誤差を最小化する決定株を全探索で学習する。

    h(x) = polarity * sign(x[feature] - threshold)  (sign(0)=-1側: x>t で+1)
    戻り値: (feature, threshold, polarity, weighted_error)
    """
    n, d = X.shape
    best = None
    best_err = np.inf
    for j in range(d):
        xs = X[:, j]
        order = np.argsort(xs, kind="stable")
        x_sorted = xs[order]
        y_sorted = y[order]
        w_sorted = w[order]
        wp = np.cumsum(np.where(y_sorted > 0, w_sorted, 0.0))  # 左側(<=t)の正例重み
        wn = np.cumsum(np.where(y_sorted < 0, w_sorted, 0.0))  # 左側(<=t)の負例重み
        Wp, Wn = wp[-1], wn[-1]
        # 閾値をインデックスiとi+1の間に置く (左側 = 0..i)
        # polarity=+1 (x>t → +1): 誤り = 左側の正例 + 右側の負例
        err_pos = wp + (Wn - wn)
        # polarity=-1 (x>t → -1): 誤り = 左側の負例 + 右側の正例
        err_neg = wn + (Wp - wp)
        # 同じ値の間には閾値を置けない
        valid = np.ones(n, dtype=bool)
        valid[:-1] = x_sorted[:-1] < x_sorted[1:]
        valid[-1] = False  # 全部左側 = 定数分類器はpolarityだけで表現できるので除外
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        for errs, pol in ((err_pos, 1), (err_neg, -1)):
            i_best = idx[np.argmin(errs[idx])]
            e = errs[i_best]
            if e < best_err:
                thr = 0.5 * (x_sorted[i_best] + x_sorted[i_best + 1])
                best_err = e
                best = (j, thr, pol)
    j, thr, pol = best
    return j, thr, pol, best_err


def stump_predict(X, feature, threshold, polarity):
    return polarity * np.where(X[:, feature] > threshold, 1.0, -1.0)


class AdaBoostStumps:
    """決定株のAdaBoost。yは{-1,+1}。"""

    def __init__(self, n_rounds=1000):
        self.n_rounds = n_rounds
        self.stumps = []   # (feature, threshold, polarity)
        self.alphas = []

    def fit(self, X, y, record_weights=False):
        n = len(y)
        w = np.full(n, 1.0 / n)
        self.weight_history = [] if record_weights else None
        for t in range(self.n_rounds):
            j, thr, pol, err = train_stump(X, y, w)
            err = np.clip(err, 1e-12, 1 - 1e-12)
            alpha = 0.5 * np.log((1 - err) / err)
            h = stump_predict(X, j, thr, pol)
            w = w * np.exp(-alpha * y * h)
            w /= w.sum()
            self.stumps.append((j, thr, pol))
            self.alphas.append(alpha)
            if record_weights:
                self.weight_history.append(w.copy())
        self.alphas = np.array(self.alphas)
        return self

    def decision_function(self, X, n_rounds=None):
        T = len(self.stumps) if n_rounds is None else n_rounds
        score = np.zeros(len(X))
        for (j, thr, pol), a in zip(self.stumps[:T], self.alphas[:T]):
            score += a * stump_predict(X, j, thr, pol)
        return score

    def staged_scores(self, X):
        """各ラウンド終了時点のスコア(累積)をジェネレータで返す。"""
        score = np.zeros(len(X))
        for (j, thr, pol), a in zip(self.stumps, self.alphas):
            score += a * stump_predict(X, j, thr, pol)
            yield score

    def staged_errors(self, X, y):
        """各ラウンド終了時点の0-1誤差の配列。"""
        errs = np.empty(len(self.stumps))
        for t, s in enumerate(self.staged_scores(X)):
            pred = np.where(s >= 0, 1.0, -1.0)
            errs[t] = np.mean(pred != y)
        return errs
