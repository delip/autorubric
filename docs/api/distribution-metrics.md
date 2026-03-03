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

# Earth Mover's Distance (lower = more similar distributions)
emd = earth_movers_distance(predicted_scores, ground_truth_scores)
print(f"EMD: {emd.distance:.4f}")

# Kolmogorov-Smirnov test
ks = ks_test(predicted_scores, ground_truth_scores)
print(f"KS statistic: {ks.statistic:.4f}, p-value: {ks.p_value:.4f}")

# Score distribution statistics
pred_dist = score_distribution(predicted_scores)
print(f"Mean: {pred_dist.mean:.3f}, Std: {pred_dist.std:.3f}")

# Systematic bias
bias = systematic_bias(predicted_scores, ground_truth_scores)
print(f"Bias: {bias.mean_bias:+.4f} (predicted tends to be {'higher' if bias.mean_bias > 0 else 'lower'})")
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
