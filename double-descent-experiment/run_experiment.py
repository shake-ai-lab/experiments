import numpy as np
import json
from numpy.polynomial import legendre

N_TRAIN = 25
NOISE_STD = 0.3
LAM = 1e-4          # tiny stabilizing ridge (numerical proxy for the ridgeless/min-norm limit)
DEGREES = list(range(0, 81)) + [90, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]
N_TRIALS = 500
N_TEST = 1000
SEED = 42


def true_fn(x):
    return np.sin(1.5 * np.pi * x) + 0.5 * x


def legendre_features(x, degree):
    return legendre.legvander(x, degree)


def ridge_solve(Phi, y, lam):
    n, p = Phi.shape
    if p <= n:
        A = Phi.T @ Phi + lam * np.eye(p)
        coef = np.linalg.solve(A, Phi.T @ y)
    else:
        A = Phi @ Phi.T + lam * np.eye(n)
        coef = Phi.T @ np.linalg.solve(A, y)
    return coef


def raw_min_norm_solve(Phi, y):
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return coef


def one_trial(rng, degrees, lam, n_train=N_TRAIN, noise_std=NOISE_STD, n_test=N_TEST):
    x_tr = rng.uniform(-1, 1, size=n_train)
    y_tr = true_fn(x_tr) + rng.normal(0, noise_std, size=n_train)
    x_te = rng.uniform(-1, 1, size=n_test)
    y_te = true_fn(x_te) + rng.normal(0, noise_std, size=n_test)

    train_mses, test_mses, coef_norms = [], [], []
    for d in degrees:
        Phi_tr = legendre_features(x_tr, d)
        Phi_te = legendre_features(x_te, d)
        coef = ridge_solve(Phi_tr, y_tr, lam)
        pred_tr = Phi_tr @ coef
        pred_te = Phi_te @ coef
        train_mses.append(np.mean((pred_tr - y_tr) ** 2))
        test_mses.append(np.mean((pred_te - y_te) ** 2))
        coef_norms.append(np.linalg.norm(coef))
    return np.array(train_mses), np.array(test_mses), np.array(coef_norms)


def main():
    master = np.random.default_rng(SEED)

    # --- 1) raw ridgeless (rcond=None lstsq) single trial, to document the numerical blow-up ---
    rng_raw = np.random.default_rng(SEED)
    x_tr = rng_raw.uniform(-1, 1, size=N_TRAIN)
    y_tr = true_fn(x_tr) + rng_raw.normal(0, NOISE_STD, size=N_TRAIN)
    x_te = rng_raw.uniform(-1, 1, size=N_TEST)
    y_te = true_fn(x_te) + rng_raw.normal(0, NOISE_STD, size=N_TEST)
    raw_test = []
    raw_degrees = [d for d in DEGREES if d <= 80]
    for d in raw_degrees:
        Phi_tr = legendre_features(x_tr, d)
        Phi_te = legendre_features(x_te, d)
        coef = raw_min_norm_solve(Phi_tr, y_tr)
        pred_te = Phi_te @ coef
        raw_test.append(np.mean((pred_te - y_te) ** 2))
    raw_test = np.array(raw_test)
    print("raw ridgeless (no stabilization) test MSE, degree 20-30:")
    for d in range(20, 31):
        print(f"  degree={d} p={d+1} test_mse={raw_test[d]:.6g}")
    print(f"  max raw test mse over sweep: {raw_test.max():.6g} at degree {raw_degrees[np.argmax(raw_test)]}")

    # --- 2) single trial with stabilized ridge (for the GIF / "one real run" narrative) ---
    rng_single = np.random.default_rng(SEED)
    single_train, single_test, single_coefnorm = one_trial(rng_single, DEGREES, LAM)

    # also stash the single-trial fitted predictions on a dense grid, for the GIF
    x_grid = np.linspace(-1, 1, 400)
    y_grid_true = true_fn(x_grid)
    rng_single2 = np.random.default_rng(SEED)
    x_tr2 = rng_single2.uniform(-1, 1, size=N_TRAIN)
    y_tr2 = true_fn(x_tr2) + rng_single2.normal(0, NOISE_STD, size=N_TRAIN)
    gif_degrees = [d for d in DEGREES if d <= 80]
    preds_grid = []
    for d in gif_degrees:
        Phi_tr2 = legendre_features(x_tr2, d)
        Phi_grid = legendre_features(x_grid, d)
        coef = ridge_solve(Phi_tr2, y_tr2, LAM)
        preds_grid.append(Phi_grid @ coef)
    preds_grid = np.array(preds_grid)

    assert np.allclose(x_tr2, x_tr) and np.allclose(y_tr2, y_tr)

    # --- 3) averaged sweep over many trials (the headline curve) ---
    acc_train = np.zeros(len(DEGREES))
    acc_test = np.zeros(len(DEGREES))
    acc_coef = np.zeros(len(DEGREES))
    all_test = np.zeros((N_TRIALS, len(DEGREES)))
    for t in range(N_TRIALS):
        rng = np.random.default_rng(master.integers(0, 2**31 - 1))
        tr, te, cn = one_trial(rng, DEGREES, LAM)
        acc_train += tr
        acc_test += te
        acc_coef += cn
        all_test[t] = te
    acc_train /= N_TRIALS
    acc_test /= N_TRIALS
    acc_coef /= N_TRIALS
    median_test = np.median(all_test, axis=0)
    q25 = np.percentile(all_test, 25, axis=0)
    q75 = np.percentile(all_test, 75, axis=0)

    print(f"\n=== averaged over {N_TRIALS} trials (lam={LAM}) ===")
    for i, d in enumerate(DEGREES):
        p = d + 1
        flag = " <-- threshold" if p == N_TRAIN else ""
        print(f"deg={d:3d} p={p:3d} train={acc_train[i]:9.5f} test_mean={acc_test[i]:10.5f} test_median={median_test[i]:10.5f} coefnorm={acc_coef[i]:9.2f}{flag}")

    argmin_under = np.argmin(acc_test[:N_TRAIN-1])  # best point before threshold region
    argmax_all = np.argmax(acc_test)
    print(f"\nbest underparam point (degree<{N_TRAIN-1}): degree={DEGREES[argmin_under]} p={DEGREES[argmin_under]+1} test={acc_test[argmin_under]:.5f}")
    print(f"global peak: degree={DEGREES[argmax_all]} p={DEGREES[argmax_all]+1} test={acc_test[argmax_all]:.5f}")
    print(f"test at final degree {DEGREES[-1]} (p={DEGREES[-1]+1}): {acc_test[-1]:.5f}")
    # best point in the overparam regime, well past threshold
    over_start = N_TRAIN + 5
    over_idx = np.argmin(acc_test[over_start:]) + over_start
    print(f"best overparam point (p>{over_start}): degree={DEGREES[over_idx]} p={DEGREES[over_idx]+1} test={acc_test[over_idx]:.5f}")

    np.savez(
        "results.npz",
        degrees=np.array(DEGREES),
        params=np.array(DEGREES) + 1,
        acc_train=acc_train,
        acc_test=acc_test,
        acc_coef=acc_coef,
        median_test=median_test,
        q25=q25,
        q75=q75,
        raw_test=raw_test,
        single_train=single_train,
        single_test=single_test,
        single_coefnorm=single_coefnorm,
        x_grid=x_grid,
        y_grid_true=y_grid_true,
        preds_grid=preds_grid,
        gif_degrees=np.array(gif_degrees),
        x_tr=x_tr2,
        y_tr=y_tr2,
        n_train=N_TRAIN,
        noise_std=NOISE_STD,
        lam=LAM,
    )
    print("\nsaved results.npz")


if __name__ == "__main__":
    main()
