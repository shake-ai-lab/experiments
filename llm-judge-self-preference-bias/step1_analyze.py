"""LLM-as-judge の「自己スタイル贔屓」バイアス実験、集計スクリプト。

セットアップ:
  - 3人の「ライター」ペルソナ(Claude Sonnet 5、同一モデルにスタイル指示のみ変えて生成)
    P_concise: 簡潔・無駄なし / Q_friendly: 親しみやすい・比喩多用 / R_structured: 構造的・網羅的
  - 同じ16問にそれぞれ回答させ、48件の回答を匿名化・シャッフル
  - 3人の「審査員」ペルソナ(同じくClaude Sonnet 5、評価基準の重み付けだけ変える)
    が48件全てを1-10点で採点。審査員の重視基準はライターの美徳と1対1で対応させてある
    (P=簡潔さ重視, Q=分かりやすさ重視, R=網羅性重視)

見たいもの:
  1. 審査員 x ライター の平均スコア行列(3x3) → ヒートマップ
  2. 「自己スタイル一致ボーナス」: 各審査員について、質問ごとに
     (対応するライターへのスコア) - (他2ライターへのスコアの平均) を計算し、
     16問分のペア差の平均が0からずれているかを対応のあるt検定で確認
  3. 「甘口」バイアス: 審査員ごとの全48件に対する平均スコア(採点の甘さ/辛さ)
"""
import json
import numpy as np
from scipy import stats

BASE = "experiments/llm-judge-self-preference-bias/raw"

with open(f"{BASE}/anonymized_mapping.json", encoding="utf-8") as f:
    mapping = json.load(f)  # label -> {writer, qid}

judges = {
    "P_concise": json.load(open(f"{BASE}/judge_P_concise_scores.json", encoding="utf-8")),
    "Q_friendly": json.load(open(f"{BASE}/judge_Q_friendly_scores.json", encoding="utf-8")),
    "R_structured": json.load(open(f"{BASE}/judge_R_structured_scores.json", encoding="utf-8")),
}

writers = ["P_concise", "Q_friendly", "R_structured"]
judge_names = ["P_concise", "Q_friendly", "R_structured"]
n_questions = 16

# ---------------- (1) judge x writer mean score matrix ----------------
matrix = np.zeros((3, 3))  # rows=judge, cols=writer
counts = np.zeros((3, 3))
for label, info in mapping.items():
    w_idx = writers.index(info["writer"])
    for j_idx, jname in enumerate(judge_names):
        score = judges[jname][label]
        matrix[j_idx, w_idx] += score
        counts[j_idx, w_idx] += 1
matrix /= counts

print("=== judge x writer mean score matrix ===")
print("rows=judge, cols=writer:", writers)
for j_idx, jname in enumerate(judge_names):
    print(f"{jname}: " + " ".join(f"{matrix[j_idx, w_idx]:.2f}" for w_idx in range(3)))

# ---------------- (2) self-style-alignment bonus, paired per question ----------------
# build qid -> {writer: label} lookup
qid_writer_label = {}
for label, info in mapping.items():
    qid_writer_label.setdefault(info["qid"], {})[info["writer"]] = label

alignment_results = {}
for jname in judge_names:
    matched_writer = jname  # judge X is aligned with writer X's virtue
    other_writers = [w for w in writers if w != matched_writer]
    diffs = []
    for qid in range(1, n_questions + 1):
        labels_for_q = qid_writer_label[qid]
        own_score = judges[jname][labels_for_q[matched_writer]]
        other_scores = [judges[jname][labels_for_q[w]] for w in other_writers]
        diffs.append(own_score - np.mean(other_scores))
    diffs = np.array(diffs, dtype=float)
    tstat, pval = stats.ttest_1samp(diffs, 0.0)
    alignment_results[jname] = dict(diffs=diffs, mean=diffs.mean(), std=diffs.std(), t=tstat, p=pval)
    print(f"\n[{jname}] self-style-alignment bonus (own - mean(other 2)), n={n_questions}")
    print(f"  mean={diffs.mean():+.3f} std={diffs.std():.3f} t={tstat:.3f} p={pval:.4f}")
    print(f"  per-question diffs: {np.round(diffs, 2).tolist()}")

# ---------------- (3) leniency: overall mean score per judge across all 48 answers ----------------
print("\n=== leniency (overall mean score across all 48 answers, per judge) ===")
overall_means = {}
for jname in judge_names:
    vals = np.array(list(judges[jname].values()), dtype=float)
    overall_means[jname] = dict(mean=vals.mean(), std=vals.std())
    print(f"{jname}: mean={vals.mean():.3f} std={vals.std():.3f}")

# one-way comparison across judges' overall score distributions (are some judges systematically more lenient?)
all_vals = [np.array(list(judges[j].values()), dtype=float) for j in judge_names]
fstat, pval_anova = stats.f_oneway(*all_vals)
print(f"\none-way ANOVA across judges' overall scores: F={fstat:.3f} p={pval_anova:.4f}")

# pooled self-alignment bonus across all 3 judges (n=48 paired diffs total)
pooled_diffs = np.concatenate([alignment_results[j]["diffs"] for j in judge_names])
t_pooled, p_pooled = stats.ttest_1samp(pooled_diffs, 0.0)
print(f"\npooled self-style-alignment bonus across all 3 judges (n={len(pooled_diffs)}): "
      f"mean={pooled_diffs.mean():+.3f} std={pooled_diffs.std():.3f} t={t_pooled:.3f} p={p_pooled:.4f}")

np.savez(
    "C:/tmp/llm-judge-self-preference-bias/results.npz",
    matrix=matrix,
    writers=np.array(writers),
    judge_names=np.array(judge_names),
    p_diffs=alignment_results["P_concise"]["diffs"],
    q_diffs=alignment_results["Q_friendly"]["diffs"],
    r_diffs=alignment_results["R_structured"]["diffs"],
    p_t=alignment_results["P_concise"]["t"], p_p=alignment_results["P_concise"]["p"],
    q_t=alignment_results["Q_friendly"]["t"], q_p=alignment_results["Q_friendly"]["p"],
    r_t=alignment_results["R_structured"]["t"], r_p=alignment_results["R_structured"]["p"],
    p_lenient_mean=overall_means["P_concise"]["mean"], p_lenient_std=overall_means["P_concise"]["std"],
    q_lenient_mean=overall_means["Q_friendly"]["mean"], q_lenient_std=overall_means["Q_friendly"]["std"],
    r_lenient_mean=overall_means["R_structured"]["mean"], r_lenient_std=overall_means["R_structured"]["std"],
    fstat=fstat, pval_anova=pval_anova,
    t_pooled=t_pooled, p_pooled=p_pooled, pooled_diffs=pooled_diffs,
)
print("\nsaved results.npz")
