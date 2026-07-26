"""ensemble diversityは本当に効くのか。
dropout-implicit-ensemble-test の続き: 今回はdropoutを使わず、
明示的アンサンブル同士で「多様性の質」を変えて比較する。

同じtwo-moonsタスク・同じ手書きnumpy MLP(tanh隠れ層+sigmoid出力、Adam)を使い、
M=10メンバー・総隠れユニット数80(≒総パラメータ数330)という同じ予算のもとで、

  (a) shallow diversity: 全メンバーが同じ隠れ8ユニットの構造で、同じ訓練データ全体
      (700点)を使い、乱数初期化seedだけを変える
  (b) real diversity: 訓練データをブートストラップ再抽出(bagging)し、
      かつメンバーごとに隠れユニット数を変える(4,5,6,7,8,9,10,11,12,8 = 合計80で(a)と同じ予算)

を比較する。個々のメンバー精度・アンサンブル精度・メンバー間の予測不一致率(disagreement)
を測り、「多様性が大きいほどアンサンブルの精度ゲインも大きいか」を実測する。
run-to-run noiseを見るため、データ生成seedを変えてR回繰り返す。
"""
import time
import numpy as np
from sklearn.datasets import make_moons

OUTDIR = "C:/tmp/ensemble-diversity-experiment"
M = 10
H_SMALL = 8  # condition (a): fixed width for all members
H_LIST_B = [4, 5, 6, 7, 8, 9, 10, 11, 12, 8]  # condition (b): varied width, sum=80 (same total as a: 8*10=80)
assert sum(H_LIST_B) == H_SMALL * M
EPOCHS = 4000
LR = 0.03
R = 8  # number of repeated trials (different data draws / seeds) to gauge the noise floor


# ---------------- from-scratch MLP: 2 -> H -> 1, tanh hidden, sigmoid out, BCE loss ----------------
# (same architecture/training code as experiments/dropout-implicit-ensemble-test/step1_train.py,
#  reused here for genuine continuity between the two posts)
def init_net(H, seed):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, np.sqrt(2.0 / 2), size=(2, H))
    b1 = np.zeros(H)
    W2 = r.normal(0, np.sqrt(2.0 / H), size=(H, 1))
    b2 = np.zeros(1)
    return [W1, b1, W2, b2]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def forward(params, X):
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1
    h1 = np.tanh(z1)
    z2 = h1 @ W2 + b2
    out = sigmoid(z2).ravel()
    cache = (X, z1, h1, z2)
    return out, cache


def backward(params, cache, y_true, out):
    W1, b1, W2, b2 = params
    X, z1, h1, z2 = cache
    n = X.shape[0]
    d_z2 = (out - y_true).reshape(-1, 1) / n
    dW2 = h1.T @ d_z2
    db2 = d_z2.sum(axis=0)
    d_h1 = d_z2 @ W2.T
    d_z1 = d_h1 * (1 - np.tanh(z1) ** 2)
    dW1 = X.T @ d_z1
    db1 = d_z1.sum(axis=0)
    return [dW1, db1, dW2, db2]


def make_adam(params):
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    return m, v


def adam_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for p, g, mi, vi in zip(params, grads, m, v):
        mi[:] = beta1 * mi + (1 - beta1) * g
        vi[:] = beta2 * vi + (1 - beta2) * g ** 2
        m_hat = mi / (1 - beta1 ** t)
        v_hat = vi / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_net(H, seed, X_tr, y_tr, epochs=EPOCHS, lr=LR):
    params = init_net(H, seed)
    m, v = make_adam(params)
    for t in range(1, epochs + 1):
        out, cache = forward(params, X_tr)
        grads = backward(params, cache, y_tr, out)
        adam_step(params, grads, m, v, t, lr)
    return params


def predict(params, Xq):
    out, _ = forward(params, Xq)
    return out


def pairwise_disagreement(bin_preds):
    """bin_preds: (M, n) boolean array of member predictions. Average pairwise disagreement rate."""
    M_ = bin_preds.shape[0]
    total = 0.0
    n_pairs = 0
    for i in range(M_):
        for j in range(i + 1, M_):
            total += (bin_preds[i] != bin_preds[j]).mean()
            n_pairs += 1
    return total / n_pairs


def run_condition_a(X_tr, y_tr, X_te, y_te, trial_seed):
    """shallow diversity: same arch (H=8), same full data, seed-only."""
    members = []
    accs = []
    test_preds = []
    for i in range(M):
        seed = trial_seed * 10_000 + 100 + i
        p = train_net(H_SMALL, seed, X_tr, y_tr)
        members.append(p)
        out = predict(p, X_te)
        accs.append(((out > 0.5) == y_te).mean())
        test_preds.append(out)
    test_preds = np.stack(test_preds)  # (M, n_test)
    ens_mean = test_preds.mean(axis=0)
    ens_acc = ((ens_mean > 0.5) == y_te).mean()
    bin_preds = test_preds > 0.5
    disagree = pairwise_disagreement(bin_preds)
    n_params = sum(H_SMALL * 4 + 1 for _ in range(M))  # 4H+1 params per member (2H+H+H+1)
    return dict(members=members, accs=np.array(accs), ens_acc=ens_acc,
                disagree=disagree, n_params=n_params, test_preds=test_preds)


def run_condition_b(X_tr, y_tr, X_te, y_te, trial_seed):
    """real diversity: bootstrap-resampled training data per member + varied hidden width."""
    n = len(X_tr)
    members = []
    accs = []
    test_preds = []
    for i, H in enumerate(H_LIST_B):
        resample_seed = trial_seed * 10_000 + 200 + i
        init_seed = trial_seed * 10_000 + 300 + i
        r = np.random.default_rng(resample_seed)
        idx = r.integers(0, n, size=n)  # bootstrap resample, with replacement
        Xb, yb = X_tr[idx], y_tr[idx]
        p = train_net(H, init_seed, Xb, yb)
        members.append(p)
        out = predict(p, X_te)
        accs.append(((out > 0.5) == y_te).mean())
        test_preds.append(out)
    test_preds = np.stack(test_preds)
    ens_mean = test_preds.mean(axis=0)
    ens_acc = ((ens_mean > 0.5) == y_te).mean()
    bin_preds = test_preds > 0.5
    disagree = pairwise_disagreement(bin_preds)
    n_params = sum(H * 4 + 1 for H in H_LIST_B)
    return dict(members=members, accs=np.array(accs), ens_acc=ens_acc,
                disagree=disagree, n_params=n_params, test_preds=test_preds)


def make_grid():
    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 70), np.linspace(-2.6, 2.6, 70))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    return gx, gy, grid


results = []
gx, gy, grid = make_grid()
rep_trial_data = None  # save one representative trial's full detail for the boundary GIF

t0 = time.time()
for trial in range(R):
    X, y = make_moons(n_samples=1000, noise=0.28, random_state=trial)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    X = (X - mu) / sd
    y = y.astype(np.float64)
    rng = np.random.default_rng(trial)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_train = 700
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    res_a = run_condition_a(X_tr, y_tr, X_te, y_te, trial)
    res_b = run_condition_b(X_tr, y_tr, X_te, y_te, trial)

    gain_a = res_a["ens_acc"] - res_a["accs"].mean()
    gain_b = res_b["ens_acc"] - res_b["accs"].mean()

    print(f"[trial {trial}] "
          f"A(seed-only): mean_member={res_a['accs'].mean():.4f} ens={res_a['ens_acc']:.4f} "
          f"gain={gain_a:+.4f} disagree={res_a['disagree']:.4f} n_params={res_a['n_params']}  |  "
          f"B(bagging+width): mean_member={res_b['accs'].mean():.4f} ens={res_b['ens_acc']:.4f} "
          f"gain={gain_b:+.4f} disagree={res_b['disagree']:.4f} n_params={res_b['n_params']}")

    results.append(dict(
        trial=trial,
        a_mean_member=res_a["accs"].mean(), a_ens=res_a["ens_acc"], a_gain=gain_a, a_disagree=res_a["disagree"],
        b_mean_member=res_b["accs"].mean(), b_ens=res_b["ens_acc"], b_gain=gain_b, b_disagree=res_b["disagree"],
    ))

    if trial == 0:
        # save full detail (grid predictions per member) for the decision-boundary GIF
        ens_grid_a = np.stack([predict(p, grid) for p in res_a["members"]])
        ens_grid_b = np.stack([predict(p, grid) for p in res_b["members"]])
        rep_trial_data = dict(
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            gx=gx, gy=gy,
            ens_grid_a=ens_grid_a, ens_grid_b=ens_grid_b,
            a_ens_acc=res_a["ens_acc"], b_ens_acc=res_b["ens_acc"],
            a_member_accs=res_a["accs"], b_member_accs=res_b["accs"],
        )

print(f"total time: {time.time()-t0:.1f}s")

# ---------------- aggregate across trials ----------------
import json

a_gains = np.array([r["a_gain"] for r in results])
b_gains = np.array([r["b_gain"] for r in results])
a_disagree = np.array([r["a_disagree"] for r in results])
b_disagree = np.array([r["b_disagree"] for r in results])
a_ens = np.array([r["a_ens"] for r in results])
b_ens = np.array([r["b_ens"] for r in results])
a_mean_member = np.array([r["a_mean_member"] for r in results])
b_mean_member = np.array([r["b_mean_member"] for r in results])

print("\n=== AGGREGATE over", R, "trials ===")
print(f"A (seed-only):     mean_member_acc={a_mean_member.mean():.4f}±{a_mean_member.std():.4f}  "
      f"ens_acc={a_ens.mean():.4f}±{a_ens.std():.4f}  gain={a_gains.mean():+.4f}±{a_gains.std():.4f}  "
      f"disagree={a_disagree.mean():.4f}±{a_disagree.std():.4f}")
print(f"B (bagging+width): mean_member_acc={b_mean_member.mean():.4f}±{b_mean_member.std():.4f}  "
      f"ens_acc={b_ens.mean():.4f}±{b_ens.std():.4f}  gain={b_gains.mean():+.4f}±{b_gains.std():.4f}  "
      f"disagree={b_disagree.mean():.4f}±{b_disagree.std():.4f}")

diff_gain = b_gains.mean() - a_gains.mean()
# paired difference per trial, since same data draw used for both conditions within a trial
paired_diff = b_gains - a_gains
print(f"\ngain difference (B - A): {diff_gain:+.4f}")
print(f"paired per-trial gain diff: mean={paired_diff.mean():+.4f} std={paired_diff.std():.4f} "
      f"min={paired_diff.min():+.4f} max={paired_diff.max():+.4f}")
from scipy import stats
tstat, pval = stats.ttest_rel(b_gains, a_gains)
print(f"paired t-test on gain (B vs A): t={tstat:.3f} p={pval:.4f}")

diff_disagree = b_disagree.mean() - a_disagree.mean()
print(f"\ndisagreement difference (B - A): {diff_disagree:+.4f}")
tstat2, pval2 = stats.ttest_rel(b_disagree, a_disagree)
print(f"paired t-test on disagreement (B vs A): t={tstat2:.3f} p={pval2:.4f}")

# correlation between disagreement and gain, pooling both conditions across all trials
all_disagree = np.concatenate([a_disagree, b_disagree])
all_gain = np.concatenate([a_gains, b_gains])
corr_dg = np.corrcoef(all_disagree, all_gain)[0, 1]
print(f"\npooled correlation (disagreement rate vs accuracy gain), n={len(all_disagree)}: r={corr_dg:.4f}")

np.savez(
    f"{OUTDIR}/results_aggregate.npz",
    a_gains=a_gains, b_gains=b_gains, a_disagree=a_disagree, b_disagree=b_disagree,
    a_ens=a_ens, b_ens=b_ens, a_mean_member=a_mean_member, b_mean_member=b_mean_member,
    corr_dg=corr_dg, R=R, M=M, H_SMALL=H_SMALL, H_LIST_B=np.array(H_LIST_B),
)
print("saved results_aggregate.npz")

np.savez(
    f"{OUTDIR}/rep_trial.npz",
    **{k: v for k, v in rep_trial_data.items()},
)
print("saved rep_trial.npz")
print("ALL DONE")
