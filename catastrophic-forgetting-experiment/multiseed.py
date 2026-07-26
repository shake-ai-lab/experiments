"""Multi-seed robustness check: does A-accuracy really collapse to ~0 without
rehearsal, and does rehearsal really preserve most of it, across several
random seeds (different data splits + different init + different batch order)?"""
import numpy as np
from experiment import (
    load_split, init_params, forward, Adam, onehot, accuracy,
    train_full_batch, clone_params, train_phase2, LR, PHASE1_EPOCHS,
    BATCH_SIZE, REHEARSAL_FRAC,
)

N_SEEDS = 15
PHASE2_STEPS = 1500  # shorter than main run; plateau is reached well before 3000

results = []
for seed in range(N_SEEDS):
    (A_Xtr, A_ytr, A_Xte, A_yte), (B_Xtr, B_ytr, B_Xte, B_yte) = load_split(seed=seed)

    params0 = init_params(seed=seed)
    opt0 = Adam(params0, lr=LR)
    params_after_a = train_full_batch(params0, opt0, A_Xtr, A_ytr, PHASE1_EPOCHS)
    a_start = accuracy(params_after_a, A_Xte, A_yte)
    checkpoint = clone_params(params_after_a)

    rng_no = np.random.default_rng(seed + 1000)
    p_no = clone_params(checkpoint)
    opt_no = Adam(p_no, lr=LR)
    p_no, s_no, a_no, b_no = train_phase2(
        p_no, opt_no, B_Xtr, B_ytr, A_Xtr, A_ytr, A_Xte, A_yte, B_Xte, B_yte,
        steps=PHASE2_STEPS, batch_size=BATCH_SIZE, rehearsal_frac=0.0, rng=rng_no,
    )

    rng_re = np.random.default_rng(seed + 1000)
    p_re = clone_params(checkpoint)
    opt_re = Adam(p_re, lr=LR)
    p_re, s_re, a_re, b_re = train_phase2(
        p_re, opt_re, B_Xtr, B_ytr, A_Xtr, A_ytr, A_Xte, A_yte, B_Xte, B_yte,
        steps=PHASE2_STEPS, batch_size=BATCH_SIZE, rehearsal_frac=REHEARSAL_FRAC, rng=rng_re,
    )

    results.append({
        "seed": seed,
        "a_start": a_start,
        "a_final_no": a_no[-1],
        "a_final_re": a_re[-1],
        "b_final_no": b_no[-1],
        "b_final_re": b_re[-1],
    })
    print(f"seed={seed:2d}  a_start={a_start:.4f}  a_final_no={a_no[-1]:.4f}  a_final_re={a_re[-1]:.4f}  "
          f"b_final_no={b_no[-1]:.4f}  b_final_re={b_re[-1]:.4f}")

arr = lambda k: np.array([r[k] for r in results])
print("\n--- mean over {} seeds ---".format(N_SEEDS))
for k in ["a_start", "a_final_no", "a_final_re", "b_final_no", "b_final_re"]:
    v = arr(k)
    print(f"{k}: mean={v.mean():.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}")
