# Metrics

Agreement and correlation metrics for validating LLM judges against ground truth.

## Overview

When your dataset includes ground truth labels, `compute_metrics()` measures how well your LLM judge agrees with human annotations. Metrics include accuracy, precision, recall, F1, Cohen's kappa, correlations, and systematic bias analysis.

For ensemble (multi-judge) evaluations, each per-criterion metrics object also reports **inter-judge agreement** (judges vs. each other, independent of ground truth). The recommended statistic is **Krippendorff's alpha** (`krippendorff_alpha`) — it handles unequal/missing raters and is level-aware (nominal vs. ordinal). **Fleiss' kappa** (`fleiss_kappa`) is also reported as the classic fixed-rater nominal measure, computed complete-case. Both are populated only with an ensemble of ≥2 judges and ≥2 items, and are `None` otherwise.

!!! tip "Research Background"

    Casabianca et al. (2025) recommend agreement metrics including ICC, Krippendorff's alpha, and quadratic-weighted kappa (QWK), with iterative refinement until agreement with human-labeled subsets is acceptable. He et al. (2025) emphasize that correlation alone can mask systematic bias.

## Quick Example

```python
from autorubric import RubricDataset, LLMConfig, evaluate
from autorubric.graders import CriterionGrader

dataset = RubricDataset.from_file("data_with_ground_truth.json")
grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4.1-mini"))

result = await evaluate(dataset, grader, show_progress=True)

# Compute metrics
metrics = result.compute_metrics(dataset)

# Formatted summary
print(metrics.summary())

# Export options
df = metrics.to_dataframe()
metrics.to_file("metrics.json")
```

## Bootstrap Confidence Intervals

```python
metrics = result.compute_metrics(
    dataset,
    bootstrap=True,
    n_bootstrap=1000,
    confidence_level=0.95,
    seed=42,
)

print(metrics.summary())
# Bootstrap CIs (95%):
#   Accuracy: [85.2%, 92.1%]
#   Kappa:    [0.712, 0.845]
```

## Per-Judge Metrics (Ensemble)

```python
metrics = result.compute_metrics(
    dataset,
    per_judge=True,
)

for judge_id, jm in metrics.per_judge.items():
    print(f"{judge_id}: Accuracy={jm.criterion_accuracy:.1%}, RMSE={jm.score_rmse:.4f}")
```

## Metric Fields

| Field | Description |
|-------|-------------|
| `criterion_accuracy` | Overall accuracy across all criteria |
| `criterion_precision` | Precision for MET class |
| `criterion_recall` | Recall for MET class |
| `criterion_f1` | F1 score for MET class |
| `mean_kappa` | Mean Cohen's kappa across criteria |
| `per_criterion` | Per-criterion metrics breakdown (polymorphic: `CriterionMetrics`, `OrdinalCriterionMetrics`, `NominalCriterionMetrics`) |
| `score_rmse` | RMSE of cumulative scores |
| `score_mae` | MAE of cumulative scores |
| `score_spearman` | Spearman rank correlation |
| `score_kendall` | Kendall tau correlation |
| `score_pearson` | Pearson correlation |
| `bias` | Systematic bias analysis (`BiasResult`) |
| `bootstrap` | Bootstrap confidence intervals (`BootstrapResults`, if enabled) |
| `per_judge` | Per-judge metrics for ensemble (`dict[str, JudgeMetrics]`, if enabled) |
| `n_items` | Number of items used in computation |
| `n_criteria` | Number of criteria |
| `n_binary_criteria` | Number of binary criteria |
| `n_ordinal_criteria` | Number of ordinal multi-choice criteria |
| `n_nominal_criteria` | Number of nominal multi-choice criteria |
| `na_stats` | Statistics for NA handling in multi-choice criteria (`NAStats`) |
| `warnings` | Any warnings generated during computation |

---

## compute_metrics

Compute agreement metrics between predictions and ground truth.

::: autorubric.compute_metrics
    options:
      show_source: true

---

## MetricsResult

Complete metrics result with aggregate and per-criterion breakdowns.

::: autorubric.MetricsResult
    options:
      show_source: true
      members_order: source

---

## CriterionMetrics

Per-criterion binary metrics.

::: autorubric.CriterionMetrics
    options:
      show_source: true
      members_order: source

---

## CorrelationResult

Correlation statistics between predicted and ground truth scores.

::: autorubric.CorrelationResult
    options:
      show_source: true
      members_order: source

---

## BootstrapResults

Bootstrap confidence intervals for key metrics.

::: autorubric.BootstrapResults
    options:
      show_source: true
      members_order: source

---

## BootstrapResult

Single bootstrap result with confidence interval.

::: autorubric.BootstrapResult
    options:
      show_source: true
      members_order: source

---

## ConfidenceInterval

Confidence interval bounds.

::: autorubric.ConfidenceInterval
    options:
      show_source: true
      members_order: source

---

## JudgeMetrics

Per-judge metrics for ensemble evaluations.

::: autorubric.JudgeMetrics
    options:
      show_source: true
      members_order: source

---

## BiasResult

Systematic bias analysis between predicted and ground truth scores.

::: autorubric.BiasResult
    options:
      show_source: true
      members_order: source

---

## OrdinalCriterionMetrics

Per-criterion metrics for ordinal multi-choice criteria.

::: autorubric.metrics.OrdinalCriterionMetrics
    options:
      show_source: true
      members_order: source

---

## NominalCriterionMetrics

Per-criterion metrics for nominal multi-choice criteria.

::: autorubric.metrics.NominalCriterionMetrics
    options:
      show_source: true
      members_order: source

---

## NAStats

Statistics for NA (not applicable) handling in multi-choice criteria.

::: autorubric.metrics.NAStats
    options:
      show_source: true
      members_order: source

---

## References

Casabianca, J., McCaffrey, D. F., Johnson, M. S., Alper, N., and Zubenko, V. (2025). Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications. arXiv:2501.02334.

He, J., Shi, J., Zhuo, T. Y., Treude, C., Sun, J., Xing, Z., Du, X., and Lo, D. (2025). LLM-as-a-Judge for Software Engineering: Literature Review, Vision, and the Road Ahead. arXiv:2510.24367.
