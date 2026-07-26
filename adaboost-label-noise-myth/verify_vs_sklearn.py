"""自作AdaBoostとsklearnのAdaBoostClassifier(SAMME, 決定株)の照合"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from adaboost import AdaBoostStumps

rng = np.random.default_rng(0)
X_tr, y_tr01 = make_moons(n_samples=300, noise=0.15, random_state=0)
X_te, y_te01 = make_moons(n_samples=2000, noise=0.15, random_state=1)
y_tr = 2.0 * y_tr01 - 1
y_te = 2.0 * y_te01 - 1

T = 200
mine = AdaBoostStumps(n_rounds=T).fit(X_tr, y_tr)
skl = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1), n_estimators=T, random_state=0
).fit(X_tr, y_tr01)

mine_tr = mine.staged_errors(X_tr, y_tr)
mine_te = mine.staged_errors(X_te, y_te)
skl_tr = np.array([np.mean(p != y_tr01) for p in skl.staged_predict(X_tr)])
skl_te = np.array([np.mean(p != y_te01) for p in skl.staged_predict(X_te)])

print("round | mine train | skl train | mine test | skl test")
for t in [0, 4, 9, 19, 49, 99, 199]:
    print(f"{t+1:5d} | {mine_tr[t]:.4f} | {skl_tr[t]:.4f} | {mine_te[t]:.4f} | {skl_te[t]:.4f}")

pred_mine = np.where(mine.decision_function(X_te) >= 0, 1, 0)
pred_skl = skl.predict(X_te)
agree = np.mean(pred_mine == pred_skl)
print(f"\nT={T} でのテスト予測一致率: {agree:.4f}")
print(f"最終テスト誤差 mine={mine_te[-1]:.4f} sklearn={skl_te[-1]:.4f}")
print(f"最大テスト誤差差分(全ラウンド): {np.max(np.abs(mine_te - skl_te)):.4f}")
