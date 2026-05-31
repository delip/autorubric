# Metrics

Agreement and correlation metrics for validating LLM judges against ground truth.

## Overview

When your dataset includes ground truth labels, `compute_metrics()` measures how well your LLM judge agrees with human annotations. Metrics include accuracy, precision, recall, F1, Cohen's kappa, correlations, and systematic bias analysis.

For ensemble (multi-judge) evaluations, each per-criterion metrics object also reports **inter-judge agreement** (judges vs. each other, independent of ground truth). The recommended statistic is **Krippendorff's alpha** (`krippendorff_alpha`) — it handles unequal/missing raters and is level-aware (nominal vs. ordinal). **Fleiss' kappa** (`fleiss_kappa`) is also computed as the classic fixed-rater nominal measure, complete-case. Both are populated only with an ensemble of ≥2 judges and ≥2 items, and are `None` otherwise.

!!! note "One inter-judge statistic on binary/nominal data"

    On binary and nominal data Krippendorff's nominal α and Fleiss' κ coincide up to a
    finite-sample correction `(1 − κ_F)/(N·R)` — they are one statistic, not corroborating
    evidence. `summary()` therefore reports **α as the single primary** inter-judge column
    for binary/nominal criteria and drops the bare Fleiss column (a note explains the
    omission); `to_dataframe()` leaves the binary/nominal `fleiss_kappa` value `None`. On
    **ordinal** data α is distance-aware while Fleiss is nominal (different geometry), so
    both are kept with a distinguishing note.

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

# Formatted summary. The header names the handling modes
# (CANNOT_ASSESS / NA estimands), the criterion-level scalars carry their
# aggregation level (micro vs macro), and binary criteria show φ + FP/FN/FPR/FNR.
print(metrics.summary())

# verbose=True additionally prints the per-judge RMSE/Spearman columns and each
# judge's confusion matrix (the default per-judge line leads with accuracy + kappa + φ).
print(metrics.summary(verbose=True))

# Export options. to_dataframe() uses level-labelled aggregate keys
# (accuracy_micro / accuracy_macro / mean_kappa_macro / kappa_micro / phi_micro / ...)
# and round-trips the handling modes + coverage columns.
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
    # jm.criterion_accuracy is `float | None` (None when undefined); score_rmse is always a float.
    acc = f"{jm.criterion_accuracy:.1%}" if jm.criterion_accuracy is not None else "n/a"
    print(f"{judge_id}: Accuracy={acc}, RMSE={jm.score_rmse:.4f}")
```

## Metric Fields

!!! note "`None` means genuinely undefined, never a fabricated `0.0`"

    The numeric metric fields below are typed `float | None`. A field is `None`
    when the metric is genuinely undefined for the data at hand — it is **never**
    silently reported as a fake `0.0`. Always guard the format spec (e.g.
    `f"{x:.2f}" if x is not None else "n/a"`) before printing these.

| Field | Description |
|-------|-------------|
| `criterion_accuracy` | Overall accuracy across all criteria. `float | None` — `None` when undefined (e.g. no paired predictions). |
| `criterion_precision` | Precision for the binary MET class. `float | None` — `None` when not applicable, e.g. a **multi-choice-only** rubric (no binary MET class). |
| `criterion_recall` | Recall for the binary MET class. `float | None` — `None` when not applicable (multi-choice-only rubric). |
| `criterion_f1` | F1 for the binary MET class. `float | None` — `None` when not applicable (multi-choice-only rubric). |
| `mean_kappa` | Mean Cohen's kappa across criteria (**macro** — unweighted mean over criteria). `float | None` — `None` when undefined (e.g. degenerate single-class). |
| `macro_accuracy` | Unweighted mean of the per-criterion accuracies (**macro**). `float | None`. |
| `micro_kappa` | Cohen's kappa pooled across criteria (**micro**, distinct from the macro `mean_kappa`). `float | None`. |
| `criterion_phi` | Matthews correlation coefficient (φ) pooled over the binary MET-vs-rest flats (**micro**). `float | None` — `None` for a multi-choice-only rubric or on single-class data. φ = Pearson = Spearman = Kendall = MCC on binary data; the κ − φ gap is the judge's positive-rate drift. |
| `mean_krippendorff_alpha` | Macro mean of the per-criterion Krippendorff's α (inter-judge). `float | None`. |
| `cannot_assess_mode` / `na_mode` | How CANNOT_ASSESS / NA were handled when the metrics were computed (`exclude` / `as_unmet` / `as_category`). Frozen on the result and round-tripped by `to_file` so a serialized number is never ambiguous among the estimands. |
| `n_samples` | Total paired observations contributing to the aggregate metrics. `int | None`. |
| `coverage_stats` | Under the `exclude` mode, how much of the raw paired sample survived abstention/error exclusion (`CoverageStats | None`). Counts `n_total` (raw pre-exclusion denominator), `n_covered` (== per-criterion `n_samples`), and `n_errored`; rates `coverage`, `judge_abstain_rate`, `gt_abstain_rate`, `union_exclusion_rate`, `error_rate` are each `float | None` (`None` when `n_total == 0`). |
| `per_criterion` | Per-criterion metrics breakdown (polymorphic: `CriterionMetrics`, `OrdinalCriterionMetrics`, `NominalCriterionMetrics`). Their per-criterion numeric fields (`accuracy`, `precision`, `recall`, `f1`, `kappa`, `weighted_kappa`, `adjacent_accuracy`, per-option metrics) are likewise `float | None` when undefined. |
| `score_rmse` | RMSE of cumulative scores (always a `float`). |
| `score_mae` | MAE of cumulative scores (always a `float`). |
| `score_spearman` | Spearman rank correlation (`CorrelationResult`). Its `.coefficient` is `float | None` — `None` for a constant array or fewer than 3 samples. |
| `score_kendall` | Kendall tau correlation (`CorrelationResult`). `.coefficient` is `float | None` (`None` for a constant array or < 3 samples). |
| `score_pearson` | Pearson correlation (`CorrelationResult`). `.coefficient` is `float | None` (`None` for a constant array or < 3 samples). |
| `bias` | Systematic bias analysis (`BiasResult`). Its `.mean_bias` / `.std_bias` are `float | None` — `mean_bias` is `None` at n=0 and `std_bias` is `None` for n < 2. |
| `bootstrap` | Bootstrap confidence intervals (`BootstrapResults`, if enabled) |
| `per_judge` | Per-judge metrics for ensemble (`dict[str, JudgeMetrics]`, if enabled) |
| `n_items` | Number of items used in computation |
| `n_criteria` | Number of criteria |
| `n_binary_criteria` | Number of binary criteria |
| `n_ordinal_criteria` | Number of ordinal multi-choice criteria |
| `n_nominal_criteria` | Number of nominal multi-choice criteria |
| `na_stats` | Statistics for NA handling in multi-choice criteria (`NAStats`): `na_count_true` / `na_count_pred` counts, `na_kappa` (`float | None`) on the {NA, not-NA} dichotomy, and `na_false_positive` / `na_false_negative`. |
| `cannot_assess_stats` | Statistics for CANNOT_ASSESS handling in binary criteria (`CannotAssessStats`) — the binary parallel of `na_stats` (a **distinct** kind of abstention; see below): `ca_count_true` / `ca_count_pred` counts, `ca_kappa` (`float | None`) on the {CANNOT_ASSESS, not-CANNOT_ASSESS} dichotomy, and `ca_false_positive` / `ca_false_negative`. |
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

## CannotAssessStats

Statistics for CANNOT_ASSESS handling in binary criteria — the binary parallel of `NAStats`. Both are abstentions that flow through the same SKIP scoring path and get a dichotomized Cohen's-kappa diagnostic, but they are tracked as **distinct types**: CANNOT_ASSESS is an *epistemic* abstention on a yes/no decision ("I cannot determine MET vs. UNMET"), while multi-choice NA is "no applicable option" (a statement about the option space). Its fields are `ca_`-prefixed: `ca_count_true`, `ca_count_pred`, `ca_kappa` (`float | None`), `ca_kappa_interpretation`, `ca_false_positive`, `ca_false_negative`.

::: autorubric.metrics.CannotAssessStats
    options:
      show_source: true
      members_order: source

---

## CoverageStats

How much of the raw paired sample survived abstention/error exclusion. Built only under the `exclude` handling mode (under `as_unmet` / `as_category` no observation is dropped, so coverage would be trivially `1.0` and these stats are left `None`). `n_total` is the raw pre-exclusion denominator and `n_covered` equals the per-criterion `n_samples`; every rate (`coverage`, `judge_abstain_rate`, `gt_abstain_rate`, `union_exclusion_rate`, `error_rate`) is `float | None`, `None` when its denominator is zero.

::: autorubric.metrics.CoverageStats
    options:
      show_source: true
      members_order: source

---

## References

Casabianca, J., McCaffrey, D. F., Johnson, M. S., Alper, N., and Zubenko, V. (2025). Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications. arXiv:2501.02334.

He, J., Shi, J., Zhuo, T. Y., Treude, C., Sun, J., Xing, Z., Du, X., and Lo, D. (2025). LLM-as-a-Judge for Software Engineering: Literature Review, Vision, and the Road Ahead. arXiv:2510.24367.
