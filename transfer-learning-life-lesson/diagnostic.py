import numpy as np
import io

src = io.open("experiment.py", encoding="utf-8").read()
prefix = src.split("PRETRAINED_W1 = source_params")[0]
exec(compile(prefix, "experiment.py", "exec"))

# recompute pretrained W1/b1 exactly like the main script (same rng flow)
PRETRAINED_W1 = source_params["W1"].copy()
PRETRAINED_B1 = source_params["b1"].copy()

X_tgt, y_tgt = subset(TARGET_CLASSES)
rng1 = np.random.default_rng(1)
pool_idx, test_idx = split_train_pool_test(X_tgt, y_tgt, N_TEST_PER_CLASS, rng1)
X_tgt_test, y_tgt_test = X_tgt[test_idx], y_tgt[test_idx]

z1_pre = X_tgt @ PRETRAINED_W1 + PRETRAINED_B1
dead_pre = np.mean(np.all(z1_pre <= 0, axis=0))
active_pre = np.mean(z1_pre > 0)

rng_r = np.random.default_rng(999)
random_init = init_params(rng_r)
z1_rand = X_tgt @ random_init["W1"] + random_init["b1"]
dead_rand = np.mean(np.all(z1_rand <= 0, axis=0))
active_rand = np.mean(z1_rand > 0)

print("=== dead-unit diagnostic (400-epoch pretrained source) ===")
print(f"pretrained: abs_mean={np.abs(PRETRAINED_W1).mean():.4f} dead_frac={dead_pre:.4f} active_rate={active_pre:.4f}")
print(f"random:     abs_mean={np.abs(random_init['W1']).mean():.4f} dead_frac={dead_rand:.4f} active_rate={active_rand:.4f}")

# ---------------------------------------------------------------------------
# does stopping source pretraining earlier improve downstream transfer?
# ---------------------------------------------------------------------------

STOP_EPOCHS = [20, 50, 100, 200, 400]
N_PER_CLASS_TEST = [1, 3, 10]
N_SEEDS_DIAG = 20

print("\n=== effect of source pretraining length on transfer accuracy ===")
for stop_epoch in STOP_EPOCHS:
	rng0b = np.random.default_rng(0)
	perm_b = rng0b.permutation(len(y_src))
	Xs, ys = X_src[perm_b], y_src[perm_b]
	Xs_test, ys_test = Xs[:n_src_test], ys[:n_src_test]
	Xs_train, ys_train = Xs[n_src_test:], ys[n_src_test:]
	sp = init_params(rng0b)
	sp, hist = train(
		sp, Xs_train, ys_train, trainable=["W1", "b1", "W2", "b2"], epochs=stop_epoch, lr=0.02,
		X_test=Xs_test, y_test=ys_test, record_every=stop_epoch,
	)
	w1, b1 = sp["W1"], sp["b1"]
	z1 = X_tgt @ w1 + b1
	dead = np.mean(np.all(z1 <= 0, axis=0))

	line = f"stop_epoch={stop_epoch:4d}  src_test_acc={hist[-1]['test_acc']:.3f}  dead_frac={dead:.4f}  "
	accs_by_n = {}
	for n_per_class in N_PER_CLASS_TEST:
		accs = []
		for seed in range(N_SEEDS_DIAG):
			rng = np.random.default_rng(2000 + seed)
			train_idx = draw_small_set(pool_idx, y_tgt, n_per_class, rng)
			Xtr, ytr = X_tgt[train_idx], y_tgt[train_idx]
			p_ft = init_params(rng)
			p_ft["W1"], p_ft["b1"] = w1.copy(), b1.copy()
			p_ft, _ = train(p_ft, Xtr, ytr, trainable=["W1", "b1", "W2", "b2"], epochs=300, lr=0.02)
			accs.append(accuracy(p_ft, X_tgt_test, y_tgt_test))
		accs_by_n[n_per_class] = np.mean(accs)
	line += "  ".join(f"n={n}:acc={accs_by_n[n]:.4f}" for n in N_PER_CLASS_TEST)
	print(line)
