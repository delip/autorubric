# Distribution Metrics

Statistical functions for comparing score distributions between predicted and ground truth.

## Overview

These metrics go beyond point estimates (like accuracy) to compare the full distribution of scores. This is important because high correlation can mask systematic biases in the judge's behavior.

!!! tip "Research Background"

    He et al. (2025) emphasize that correlation alone can mask systematic bias. Distribution-aware comparisons like Earth Mover's Distance reveal systematic deviations that point metrics miss.

## Quick Example

```python
from autorubric import earth_movers_distance, ks_test, score_distribution, systematic_bias

predicted_scores = [0.8, 0.7, 0.9, 0.6, 0.85]
ground_truth_scores = [0.75, 0.72, 0.88, 0.65, 0.80]

# Earth Mover's Distance (lower = more similar distributions).
# EMDResult.emd is `float | None` (None for an empty distribution); guard before formatting.
emd_result = earth_movers_distance(predicted_scores, ground_truth_scores)
print(f"EMD: {emd_result.emd:.4f}" if emd_result.emd is not None else "EMD: n/a")

# Kolmogorov-Smirnov test (statistic / p_value are always floats)
ks = ks_test(predicted_scores, ground_truth_scores)
print(f"KS statistic: {ks.statistic:.4f}, p-value: {ks.p_value:.4f}")

# Score distribution statistics. DistributionResult.mean is `float | None` (None at n=0)
# and .std is `float | None` (None at n < 2); guard before formatting.
pred_dist = score_distribution(predicted_scores)
mean_str = f"{pred_dist.mean:.3f}" if pred_dist.mean is not None else "n/a"
std_str = f"{pred_dist.std:.3f}" if pred_dist.std is not None else "n/a"
print(f"Mean: {mean_str}, Std: {std_str}")

# Systematic bias. bias.mean_bias is `float | None` (None at n=0); guard before
# formatting and comparison.
bias = systematic_bias(predicted_scores, ground_truth_scores)
if bias.mean_bias is not None:
    direction = "higher" if bias.mean_bias > 0 else "lower"
    print(f"Bias: {bias.mean_bias:+.4f} (predicted tends to be {direction})")
else:
    print("Bias: n/a (need at least 1 paired sample)")
```

---

## earth_movers_distance

Compute Earth Mover's Distance (Wasserstein-1) between two score distributions.

::: autorubric.earth_movers_distance
    options:
      show_source: true

---

## wasserstein_distance

Alias for `earth_movers_distance`.

::: autorubric.wasserstein_distance
    options:
      show_source: true

---

## ks_test

Perform Kolmogorov-Smirnov test comparing two distributions.

::: autorubric.ks_test
    options:
      show_source: true

---

## score_distribution

Compute distribution statistics for a set of scores.

::: autorubric.score_distribution
    options:
      show_source: true

---

## systematic_bias

Analyze systematic bias between predicted and ground truth scores.

::: autorubric.systematic_bias
    options:
      show_source: true

---

## Result Types

!!! note "`None` means genuinely undefined, never a fabricated `0.0`"

    The numeric fields on these result types are typed `float | None`. A field is
    `None` when the statistic is genuinely undefined for the data at hand (it is
    **never** silently reported as a fake `0.0`). In particular, `BiasResult.mean_bias`
    is `None` at n=0 and `BiasResult.std_bias` is `None` for n < 2; the per-criterion
    `CorrelationResult.coefficient` (Pearson/Spearman/Kendall) is `None` for a constant
    array or fewer than 3 samples. Guard before formatting (e.g.
    `f"{x:+.4f}" if x is not None else "n/a"`).

### EMDResult

::: autorubric.EMDResult
    options:
      show_source: true
      members_order: source

### KSTestResult

::: autorubric.KSTestResult
    options:
      show_source: true
      members_order: source

### DistributionResult

::: autorubric.DistributionResult
    options:
      show_source: true
      members_order: source

### BiasResult

::: autorubric.BiasResult
    options:
      show_source: true
      members_order: source

---

## References

He, J., Shi, J., Zhuo, T. Y., Treude, C., Sun, J., Xing, Z., Du, X., and Lo, D. (2025). LLM-as-a-Judge for Software Engineering: Literature Review, Vision, and the Road Ahead. arXiv:2510.24367.
