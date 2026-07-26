"""
Simpson's paradox experiment: main simulation.

Scenario: a learning app tracks two user cohorts.
- Group A ("上級者" / advanced users): already skilled, use the app only a
  little (few hours/week), and score high on a reach test.
- Group B ("初級者" / beginners): use the app a lot (many hours/week,
  because they're grinding to catch up), but score lower overall since
  they're starting from a lower base.

Within EACH group, more hours/week -> higher score (true positive effect
of practice, slope > 0). But because advanced users cluster at
(few hours, high score) and beginners cluster at (many hours, lower score),
pooling the two groups together can flip the sign of the hours-vs-score
relationship.
"""
import numpy as np
from scipy import stats

SEED = 42
N_A = 150  # advanced users
N_B = 150  # beginners

SLOPE = 3.5       # true within-group effect of 1 more hour/week on score
NOISE_SD = 6.0

# Group A: advanced, few hours, high baseline
HOURS_A_MEAN, HOURS_A_SD = 3.0, 1.2
INTERCEPT_A = 60.0

# Group B: beginners, many hours, low baseline
HOURS_B_MEAN, HOURS_B_SD = 13.0, 2.5
INTERCEPT_B = 15.0


def simulate(seed=SEED):
    rng = np.random.default_rng(seed)

    hours_a = np.clip(rng.normal(HOURS_A_MEAN, HOURS_A_SD, N_A), 0.2, None)
    score_a = INTERCEPT_A + SLOPE * hours_a + rng.normal(0, NOISE_SD, N_A)
    score_a = np.clip(score_a, 0, 100)

    hours_b = np.clip(rng.normal(HOURS_B_MEAN, HOURS_B_SD, N_B), 0.2, None)
    score_b = INTERCEPT_B + SLOPE * hours_b + rng.normal(0, NOISE_SD, N_B)
    score_b = np.clip(score_b, 0, 100)

    return hours_a, score_a, hours_b, score_b


def report(hours_a, score_a, hours_b, score_b):
    hours_all = np.concatenate([hours_a, hours_b])
    score_all = np.concatenate([score_a, score_b])

    r_a, p_a = stats.pearsonr(hours_a, score_a)
    r_b, p_b = stats.pearsonr(hours_b, score_b)
    r_pool, p_pool = stats.pearsonr(hours_all, score_all)

    slope_a, intercept_a, *_ = stats.linregress(hours_a, score_a)
    slope_b, intercept_b, *_ = stats.linregress(hours_b, score_b)
    slope_pool, intercept_pool, *_ = stats.linregress(hours_all, score_all)

    print(f"N_A={N_A} N_B={N_B} N_total={N_A+N_B}")
    print(f"Group A (advanced): hours mean={hours_a.mean():.2f} sd={hours_a.std():.2f}  "
          f"score mean={score_a.mean():.2f} sd={score_a.std():.2f}")
    print(f"Group B (beginner): hours mean={hours_b.mean():.2f} sd={hours_b.std():.2f}  "
          f"score mean={score_b.mean():.2f} sd={score_b.std():.2f}")
    print()
    print(f"Group A regression: score = {intercept_a:.2f} + {slope_a:.3f} * hours   r={r_a:.4f} (p={p_a:.2e})")
    print(f"Group B regression: score = {intercept_b:.2f} + {slope_b:.3f} * hours   r={r_b:.4f} (p={p_b:.2e})")
    print(f"Pooled  regression: score = {intercept_pool:.2f} + {slope_pool:.3f} * hours   r={r_pool:.4f} (p={p_pool:.2e})")
    print()
    print(f"Paradox check: group slopes positive ({slope_a:.3f}, {slope_b:.3f}) "
          f"but pooled slope {'negative' if slope_pool < 0 else 'positive'} ({slope_pool:.3f})")

    return {
        "hours_a": hours_a, "score_a": score_a,
        "hours_b": hours_b, "score_b": score_b,
        "r_a": r_a, "r_b": r_b, "r_pool": r_pool,
        "slope_a": slope_a, "slope_b": slope_b, "slope_pool": slope_pool,
        "intercept_a": intercept_a, "intercept_b": intercept_b, "intercept_pool": intercept_pool,
    }


if __name__ == "__main__":
    hours_a, score_a, hours_b, score_b = simulate()
    result = report(hours_a, score_a, hours_b, score_b)
