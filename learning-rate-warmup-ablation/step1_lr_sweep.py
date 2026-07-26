"""高LR側の「不安定になるかならないか」の境目を探すための、正式なLRスイープ。
64->128->128->128->128->10のReLU MLP(隠れ層4層)をsklearn digitsで1500ステップ訓練し、
warmupなし・Adamで、LRを0.001から0.05まで動かして各8シードのtest_accを記録する。
test_acc<0.5を「崩壊」とみなし、崩壊率をLRの関数としてプロットするための元データを作る。
"""
import numpy as np

from exp1_train import train_run

LRS = [0.001, 0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03, 0.05]
SEEDS = list(range(8))
STEPS = 1500

if __name__ == "__main__":
    fail_rate = []
    mean_acc = []
    all_accs = {}
    all_losses_repseed = {}  # 代表シード(seed=0)の損失曲線をLRごとに保存(スライダー用)
    for lr in LRS:
        accs = []
        for s in SEEDS:
            losses, acc = train_run(lr, s, "none", steps=STEPS)
            accs.append(acc)
            if s == 0:
                all_losses_repseed[lr] = losses
        all_accs[lr] = accs
        fails = sum(1 for a in accs if a < 0.5)
        fail_rate.append(fails / len(SEEDS))
        mean_acc.append(float(np.mean(accs)))
        print(f"lr={lr:<6} fails={fails}/{len(SEEDS)} mean_acc={np.mean(accs):.3f} accs={[round(a,3) for a in accs]}")

    np.savez(
        "sweep_results.npz",
        lrs=np.array(LRS),
        fail_rate=np.array(fail_rate),
        mean_acc=np.array(mean_acc),
        **{f"repcurve_lr{lr}": all_losses_repseed[lr] for lr in LRS},
    )
    print("saved sweep_results.npz")
