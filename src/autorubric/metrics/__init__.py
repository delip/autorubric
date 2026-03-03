"""Evaluation metrics for autorubric.

This module provides metrics for evaluating LLM judge performance.
The primary interface is `EvalResult.compute_metrics()` which computes
comprehensive metrics from an evaluation run.

Example:
    >>> from autorubric import evaluate, RubricDataset
    >>> from autorubric.graders import CriterionGrader
    >>>
    >>> dataset = RubricDataset.from_file("data.json")
    >>> result = await evaluate(dataset, grader)
    >>>
    >>> # Compute metrics
    >>> metrics = result.compute_metrics(dataset)
    >>> print(metrics.summary())
    >>> print(f"Accuracy: {metrics.criterion_accuracy:.1%}")
    >>>
    >>> # Export to DataFrame
    >>> df = metrics.to_dataframe()
    >>>
    >>> # With bootstrap CIs (expensive)
    >>> metrics = result.compute_metrics(dataset, bootstrap=True)
    >>> print(f"Accuracy 95% CI: {metrics.bootstrap.accuracy_ci}")
    >>>
    >>> # With per-judge breakdown (for ensemble)
    >>> metrics = result.compute_metrics(dataset, per_judge=True)
    >>> for judge_id, jm in metrics.per_judge.items():
    ...     print(f"{judge_id}: RMSE={jm.score_rmse:.4f}")
"""

# Result types
from ._types import (
    BiasResult,
    BinaryCriterionMetrics,
    BootstrapResult,
    BootstrapResults,
    CannotAssessMode,
    ConfidenceInterval,
    CorrelationResult,
    CriterionMetrics,
    CriterionMetricsUnion,
    CriterionType,
    DistributionResult,
    EMDResult,
    JudgeMetrics,
    KSTestResult,
    MetricsResult,
    NAStats,
    NominalCriterionMetrics,
    OptionMetrics,
    OrdinalCriterionMetrics,
)

# Distribution metrics (unique value-add, not in sklearn)
from .distribution import (
    earth_movers_distance,
    ks_test,
    score_distribution,
    systematic_bias,
    wasserstein_distance,
)

# Helper functions (for advanced use cases)
from ._helpers import (
    classify_criteria,
    classify_criterion,
    extract_all_verdicts_from_report,
    extract_verdicts_from_report,
    filter_cannot_assess,
    filter_na_multi_choice,
    get_option_value,
    is_na_option,
    resolve_ground_truth,
    verdict_to_binary,
    verdict_to_string,
)

# Main compute function (also accessible via EvalResult.compute_metrics)
from ._compute import compute_metrics

__all__ = [
    # Main interface
    "compute_metrics",
    # Result types
    "BiasResult",
    "BinaryCriterionMetrics",
    "BootstrapResult",
    "BootstrapResults",
    "CannotAssessMode",
    "ConfidenceInterval",
    "CorrelationResult",
    "CriterionMetrics",
    "CriterionMetricsUnion",
    "CriterionType",
    "DistributionResult",
    "EMDResult",
    "JudgeMetrics",
    "KSTestResult",
    "MetricsResult",
    "NAStats",
    "NominalCriterionMetrics",
    "OptionMetrics",
    "OrdinalCriterionMetrics",
    # Distribution metrics (unique to autorubric)
    "earth_movers_distance",
    "wasserstein_distance",
    "ks_test",
    "score_distribution",
    "systematic_bias",
    # Helpers
    "classify_criteria",
    "classify_criterion",
    "extract_all_verdicts_from_report",
    "extract_verdicts_from_report",
    "filter_cannot_assess",
    "filter_na_multi_choice",
    "get_option_value",
    "is_na_option",
    "resolve_ground_truth",
    "verdict_to_binary",
    "verdict_to_string",
]
