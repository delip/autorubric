"""Result types for evaluation metrics.

This module defines Pydantic models and type aliases for metric results.
All models are frozen (immutable) for consistency with the rest of autorubric.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pandas as pd


def _fmt_opt(value: float | None, spec: str, width: int = 0) -> str:
    """Format an optional float, rendering ``None`` as a right-aligned ``'n/a'``.

    A single None-safe formatter shared by every place a metric may be ``None``
    (genuinely undefined / not applicable) so the rendering of "n/a" never drifts.

    Args:
        value: The value to format, or ``None``.
        spec: A :func:`format` spec applied when ``value`` is not None (e.g. ``".3f"``).
        width: When non-zero, the ``'n/a'`` placeholder is right-aligned to this width
            (matching the column width the formatted value would occupy).

    Returns:
        ``format(value, spec)`` when ``value`` is not None; otherwise ``'n/a'`` (right
        aligned to ``width`` when ``width`` is given).
    """
    if value is not None:
        return format(value, spec)
    return f"{'n/a':>{width}}" if width else "n/a"


# Type alias for CANNOT_ASSESS handling in metrics
CannotAssessMode = Literal["exclude", "as_unmet", "as_category"]
"""How to handle CANNOT_ASSESS verdicts in metric calculations.

- "exclude": Skip items with CANNOT_ASSESS verdicts from metric calculation (default)
- "as_unmet": Treat CANNOT_ASSESS as UNMET for agreement calculation
- "as_category": Treat CANNOT_ASSESS as a distinct third category (3-class classification)
"""

# Type alias for NA-option handling in multi-choice metrics (mirrors CannotAssessMode).
NAMode = Literal["exclude", "as_unmet", "as_category"]
"""How to handle NA options in multi-choice metric calculations.

Mirrors :data:`CannotAssessMode` — NA on multi-choice is the structural analog
of CANNOT_ASSESS on binary.

- "exclude": Skip pairs where either is NA (default)
- "as_unmet": Remap NA → the score-minimizing non-NA option, weight-sign aware
  (mirrors binary ``as_unmet`` which collapses CANNOT_ASSESS → UNMET, the
  worst-scoring valid verdict). Shares ``Criterion.worst_scored_option()`` with
  the grader's ``unknown``-error worst-case path.
- "as_category": Keep NA as a distinct categorical column. Refused for ordinal
  criteria that define an NA option (NA has no ordinal position; quadratic
  weighted kappa would assign NA a meaningless geometric distance).
"""


class ConfidenceInterval(BaseModel):
    """Confidence interval for a statistic.

    Attributes:
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        confidence: Confidence level (default 0.95 for 95% CI).
        method: Method used to compute the interval.
    """

    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    confidence: float = 0.95
    method: str = "normal"

    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper - self.lower


class KappaResult(BaseModel):
    """Result from Cohen's Kappa calculation.

    Cohen's kappa measures agreement while accounting for chance agreement.
    Values range from -1 (systematic disagreement) to 1 (perfect agreement),
    with 0 indicating chance-level agreement.

    Attributes:
        kappa: The kappa coefficient (-1 to 1).
        observed_agreement: Proportion of exact agreements (0 to 1).
        expected_agreement: Expected agreement by chance (0 to 1).
        standard_error: Standard error of kappa estimate.
        ci: Optional confidence interval.
        interpretation: Human-readable interpretation of kappa value.
        n_samples: Number of samples used in calculation.
    """

    model_config = ConfigDict(frozen=True)

    kappa: float
    observed_agreement: float
    expected_agreement: float
    standard_error: float | None = None
    ci: ConfidenceInterval | None = None
    interpretation: str
    n_samples: int

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """Return human-readable interpretation of kappa value.

        Based on Landis & Koch (1977) guidelines.
        """
        if kappa < 0:
            return "poor (worse than chance)"
        elif kappa < 0.21:
            return "slight"
        elif kappa < 0.41:
            return "fair"
        elif kappa < 0.61:
            return "moderate"
        elif kappa < 0.81:
            return "substantial"
        else:
            return "almost perfect"


class CorrelationResult(BaseModel):
    """Result from correlation calculation (Spearman, Kendall, Pearson).

    Attributes:
        coefficient: The correlation coefficient (-1 to 1). ``None`` when the correlation
            is genuinely undefined — a constant input array (zero variance → NaN) or fewer
            than 3 samples. A ``0.0`` ("no correlation") would be a lie in those cases.
        p_value: P-value for testing the null hypothesis of no correlation. ``None`` when
            the coefficient is undefined (see ``coefficient``).
        ci: Optional confidence interval for the coefficient.
        interpretation: Human-readable interpretation ("undefined" for a constant/NaN
            array, "insufficient data" for <3 samples).
        n_samples: Number of samples used in calculation.
        method: Correlation method used (e.g., "spearman", "kendall", "pearson").
    """

    model_config = ConfigDict(frozen=True)

    coefficient: float | None
    p_value: float | None = None
    ci: ConfidenceInterval | None = None
    interpretation: str
    n_samples: int
    method: str = "spearman"

    @staticmethod
    def interpret_correlation(r: float) -> str:
        """Return human-readable interpretation of correlation coefficient."""
        abs_r = abs(r)
        if abs_r >= 0.9:
            strength = "very strong"
        elif abs_r >= 0.7:
            strength = "strong"
        elif abs_r >= 0.5:
            strength = "moderate"
        elif abs_r >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"

        direction = "positive" if r >= 0 else "negative"
        return f"{strength} {direction}"


class BiasResult(BaseModel):
    """Result from systematic bias analysis.

    Systematic bias occurs when one rater consistently scores higher or lower
    than another, independent of the item being rated.

    A statistic is ``None`` when it is genuinely undefined for the sample size, never a
    fake ``0.0``. ``mean_bias`` is the single pred−true difference at n=1 (computable) and
    is ``None`` only at n=0. ``std_bias`` is ``None`` when undefined (n<2). ``effect_size``
    (Cohen's d) is ``None`` when ``std_bias`` is 0 or undefined.

    Attributes:
        mean_bias: Mean difference (predictions - actuals). ``None`` only at n=0.
        std_bias: Standard deviation of differences. ``None`` when undefined (n < 2).
        is_significant: Whether the bias is statistically significant (p < 0.05).
        p_value: P-value from t-test.
        direction: Direction of bias ("positive" if predictions > actuals).
        effect_size: Cohen's d effect size. ``None`` when undefined (std_bias 0 or
            undefined).
        ci: Confidence interval for mean bias.
        n_samples: Number of samples.
    """

    model_config = ConfigDict(frozen=True)

    mean_bias: float | None
    std_bias: float | None
    is_significant: bool
    p_value: float | None = None
    direction: Literal["positive", "negative", "none"]
    effect_size: float | None = None
    ci: ConfidenceInterval | None = None
    n_samples: int

    @staticmethod
    def interpret_effect_size(d: float) -> str:
        """Interpret effect size using Cohen's guidelines."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


class ClassificationReport(BaseModel):
    """Per-class classification metrics.

    Attributes:
        accuracy: Overall accuracy (correct / total).
        per_class: Dict mapping label to metrics dict with precision, recall, f1, support.
        macro_avg: Unweighted mean of per-class metrics.
        weighted_avg: Weighted mean of per-class metrics (by support).
        confusion_matrix: 2D confusion matrix as list of lists.
        labels: List of class labels in order.
        n_samples: Total number of samples.
    """

    model_config = ConfigDict(frozen=True)

    accuracy: float
    per_class: dict[str, dict[str, float]]
    macro_avg: dict[str, float]
    weighted_avg: dict[str, float]
    confusion_matrix: list[list[int]]
    labels: list[str]
    n_samples: int


# Type alias for criterion type classification
CriterionType = Literal["binary", "ordinal", "nominal"]
"""Classification of criterion type for metrics computation.

- "binary": Traditional MET/UNMET criteria
- "ordinal": Multi-choice with ordered options (e.g., 1-4 satisfaction scale)
- "nominal": Multi-choice with unordered categories (e.g., "too few", "just right", "too many")
"""


class CriterionMetrics(BaseModel):
    """Metrics for a single binary criterion.

    Attributes:
        name: Name of the criterion.
        index: Index of the criterion in the rubric.
        criterion_type: Type of criterion ("binary" for this class).
        n_samples: Number of samples used for this criterion.
        accuracy: Binary accuracy (proportion of exact matches). None when undefined / no
            samples.
        precision: Precision for MET class. None when undefined / no samples.
        recall: Recall for MET class. None when undefined / no samples.
        f1: F1 score for MET class. None when undefined / no samples.
        kappa: Cohen's kappa coefficient. None when undefined (degenerate single-class) /
            no samples.
        kappa_interpretation: Human-readable interpretation of kappa ("undefined" when
            kappa is None).
        krippendorff_alpha: Krippendorff's alpha — the general, recommended inter-judge
            agreement statistic. It natively handles unequal/missing raters (errored or
            excluded votes) and is level-aware (nominal for binary criteria). None unless
            this is an ensemble with >=2 judges and >=2 items.
        fleiss_kappa: Fleiss' kappa — the classic fixed-rater nominal inter-judge agreement
            measure, computed complete-case (only items where every judge cast a genuine
            counted vote contribute). Prefer ``krippendorff_alpha`` as the general statistic.
            None unless ensemble with >=2 judges and >=2 complete-case items.
        support_true: Count of MET in ground truth.
        support_pred: Count of MET in predictions.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    index: int
    criterion_type: Literal["binary"] = "binary"
    n_samples: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    kappa: float | None
    kappa_interpretation: str
    krippendorff_alpha: float | None = None
    fleiss_kappa: float | None = None
    support_true: int
    support_pred: int


# Alias for backwards compatibility
BinaryCriterionMetrics = CriterionMetrics


class OptionMetrics(BaseModel):
    """Metrics for a single option in a multi-choice criterion.

    Provides precision/recall/F1 breakdown per option, enabling analysis
    of which options are well-predicted vs confused.

    Attributes:
        label: Label text of the option.
        index: Zero-based index of the option.
        precision: Precision for this option class. None when undefined / no samples.
        recall: Recall for this option class. None when undefined / no samples.
        f1: F1 score for this option class. None when undefined / no samples.
        support_true: Count of this option in ground truth.
        support_pred: Count of this option in predictions.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    index: int
    precision: float | None
    recall: float | None
    f1: float | None
    support_true: int
    support_pred: int


class NAStats(BaseModel):
    """Statistics for NA (not applicable) handling in multi-choice criteria.

    Tracks how the prediction and ground truth agree on the dichotomized
    {NA, not-NA} decision per item, similar to how CANNOT_ASSESS is handled for
    binary criteria.

    Attributes:
        na_count_true: Number of NA selections in ground truth.
        na_count_pred: Number of NA selections in predictions.
        na_kappa: Cohen's kappa on the {NA, not-NA} dichotomy (pred vs truth).
            Range [-1, 1]; 1.0 is perfect agreement, 0 is chance-level, negative is
            worse than chance. None when undefined (no paired NA observations,
            single class, or NaN). The framework reports prediction-vs-ground-truth
            categorical agreement as Cohen's kappa across the board (binary `kappa`,
            ordinal `weighted_kappa`, nominal `kappa`); na_kappa is the dichotomized
            kappa for the orthogonal abstain decision. Readers who want a raw
            proportion can derive `A / (A + fp + fn)` from the counts below.
        na_kappa_interpretation: Landis & Koch interpretation of `na_kappa` via
            `KappaResult.interpret_kappa`. None when na_kappa is None.
        na_false_positive: Count where prediction was NA but ground truth was not.
        na_false_negative: Count where ground truth was NA but prediction was not.
    """

    model_config = ConfigDict(frozen=True)

    na_count_true: int
    na_count_pred: int
    na_kappa: float | None = None
    na_kappa_interpretation: str | None = None
    na_false_positive: int
    na_false_negative: int


class CannotAssessStats(BaseModel):
    """Statistics for CANNOT_ASSESS handling in binary criteria.

    The binary parallel of :class:`NAStats`: tracks how the prediction and ground truth
    agree on the dichotomized {CANNOT_ASSESS, not-CANNOT_ASSESS} decision per item.

    Both CANNOT_ASSESS (binary) and NA (multi-choice) are *abstentions* that flow through
    the same SKIP scoring path (``score_reports``), and both get a parallel dichotomized
    Cohen's-kappa diagnostic block. They are nonetheless **distinct kinds of abstention**,
    which is exactly why they are tracked by separate stats types rather than merged:

    - Binary **CANNOT_ASSESS** is the judge being unable to determine MET-vs-UNMET — an
      *epistemic* abstention on a yes/no question ("I cannot decide whether this
      requirement is met").
    - Multi-choice **NA** is "not applicable / cannot pick an applicable option" —
      abstaining because no scored category fits, a statement about the *option space*
      rather than a yes/no decision.

    Keeping them separate (and prefixing these fields ``ca_``) makes the semantic
    distinction explicit in the data model while preserving the structural analogy.

    Attributes:
        ca_count_true: Number of CANNOT_ASSESS verdicts in ground truth.
        ca_count_pred: Number of CANNOT_ASSESS verdicts in predictions.
        ca_kappa: Cohen's kappa on the {CANNOT_ASSESS, not-CANNOT_ASSESS} dichotomy
            (pred vs truth). Range [-1, 1]; 1.0 is perfect agreement, 0 is chance-level,
            negative is worse than chance. None when undefined (no paired CANNOT_ASSESS
            observations, single class, or NaN). The framework reports
            prediction-vs-ground-truth categorical agreement as Cohen's kappa across the
            board (binary `kappa`, ordinal `weighted_kappa`, nominal `kappa`, and NA's
            `na_kappa`); ca_kappa is the dichotomized kappa for the orthogonal binary
            abstain decision. Readers who want a raw proportion can derive
            `A / (A + ca_fp + ca_fn)` from the counts below.
        ca_kappa_interpretation: Landis & Koch interpretation of `ca_kappa` via
            `KappaResult.interpret_kappa`. None when ca_kappa is None.
        ca_false_positive: Count where prediction was CANNOT_ASSESS but ground truth
            was not.
        ca_false_negative: Count where ground truth was CANNOT_ASSESS but prediction
            was not.
    """

    model_config = ConfigDict(frozen=True)

    ca_count_true: int
    ca_count_pred: int
    ca_kappa: float | None = None
    ca_kappa_interpretation: str | None = None
    ca_false_positive: int
    ca_false_negative: int


class OrdinalCriterionMetrics(BaseModel):
    """Metrics for an ordinal multi-choice criterion.

    Ordinal criteria have options with inherent ordering (e.g., satisfaction 1-4).
    This enables additional metrics like weighted kappa and rank correlations.

    Attributes:
        name: Name of the criterion.
        index: Index of the criterion in the rubric.
        criterion_type: Type of criterion ("ordinal" for this class).
        n_samples: Number of samples used in computation.
        n_options: Number of options in this criterion.
        exact_accuracy: Proportion of exact index matches. None when undefined / no samples.
        adjacent_accuracy: Proportion within +/-1 position. None when undefined / no samples.
        weighted_kappa: Quadratic-weighted Cohen's kappa (accounts for distance). None when
            undefined (degenerate single-class) / no samples.
        kappa_interpretation: Human-readable interpretation of kappa ("undefined" when
            weighted_kappa is None).
        krippendorff_alpha: Krippendorff's alpha — the general, recommended inter-judge
            agreement statistic. Computed with ``level_of_measurement="ordinal"`` so it
            is distance-aware (near-miss disagreements penalized less than far-miss), and
            it natively handles unequal/missing raters. None unless ensemble with >=2
            judges and >=2 items.
        fleiss_kappa: Fleiss' kappa — the classic fixed-rater *nominal* measure (ignores
            ordering), computed complete-case. Prefer ``krippendorff_alpha`` for ordinal
            criteria. None unless ensemble with >=2 judges and >=2 complete-case items.
        spearman: Spearman rank correlation result.
        kendall: Kendall tau correlation result.
        rmse: RMSE on option values (0-1 scale). None when undefined / no samples.
        mae: MAE on option values (0-1 scale). None when undefined / no samples.
        per_option: Per-option precision/recall/F1 breakdown.
        confusion_matrix: N×N confusion matrix (rows=true, cols=pred).
        option_labels: Labels for confusion matrix axes.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    index: int
    criterion_type: Literal["ordinal"] = "ordinal"
    n_samples: int
    n_options: int
    exact_accuracy: float | None
    adjacent_accuracy: float | None
    weighted_kappa: float | None
    kappa_interpretation: str
    krippendorff_alpha: float | None = None
    fleiss_kappa: float | None = None
    spearman: CorrelationResult
    kendall: CorrelationResult
    rmse: float | None
    mae: float | None
    per_option: list[OptionMetrics]
    confusion_matrix: list[list[int]]
    option_labels: list[str]


class NominalCriterionMetrics(BaseModel):
    """Metrics for a nominal multi-choice criterion.

    Nominal criteria have unordered categories (e.g., "too few", "just right", "too many").
    Distance between options is not meaningful, so only exact matches matter.

    Attributes:
        name: Name of the criterion.
        index: Index of the criterion in the rubric.
        criterion_type: Type of criterion ("nominal" for this class).
        n_samples: Number of samples used in computation.
        n_options: Number of options in this criterion.
        exact_accuracy: Proportion of exact index matches. None when undefined / no samples.
        kappa: Unweighted Cohen's kappa (N×N). None when undefined (degenerate
            single-class) / no samples.
        kappa_interpretation: Human-readable interpretation of kappa ("undefined" when
            kappa is None).
        krippendorff_alpha: Krippendorff's alpha — the general, recommended inter-judge
            agreement statistic. Computed with ``level_of_measurement="nominal"`` and
            natively handles unequal/missing raters. None unless ensemble with >=2 judges
            and >=2 items.
        fleiss_kappa: Fleiss' kappa — the classic fixed-rater nominal measure, computed
            complete-case. Prefer ``krippendorff_alpha`` as the general statistic. None
            unless ensemble with >=2 judges and >=2 complete-case items.
        per_option: Per-option precision/recall/F1 breakdown.
        confusion_matrix: N×N confusion matrix (rows=true, cols=pred).
        option_labels: Labels for confusion matrix axes.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    index: int
    criterion_type: Literal["nominal"] = "nominal"
    n_samples: int
    n_options: int
    exact_accuracy: float | None
    kappa: float | None
    kappa_interpretation: str
    krippendorff_alpha: float | None = None
    fleiss_kappa: float | None = None
    per_option: list[OptionMetrics]
    confusion_matrix: list[list[int]]
    option_labels: list[str]


# Union type for polymorphic per-criterion metrics
CriterionMetricsUnion = CriterionMetrics | OrdinalCriterionMetrics | NominalCriterionMetrics
"""Discriminated union of criterion metrics types.

Use the `criterion_type` field to determine which type:
- "binary": CriterionMetrics (BinaryCriterionMetrics)
- "ordinal": OrdinalCriterionMetrics
- "nominal": NominalCriterionMetrics
"""


class ScoreCorrelationResult(BaseModel):
    """Correlation between predicted and actual scores.

    Attributes:
        spearman: Spearman rank correlation result.
        kendall: Kendall tau correlation result.
        pearson: Pearson correlation result.
        rmse: Root mean square error.
        mae: Mean absolute error.
        n_samples: Number of samples.
    """

    model_config = ConfigDict(frozen=True)

    spearman: CorrelationResult
    kendall: CorrelationResult
    pearson: CorrelationResult
    rmse: float
    mae: float
    n_samples: int


class AgreementSummary(BaseModel):
    """Summary of agreement between predictions and ground truth.

    This is the main result type for Level 2 agreement computation,
    aggregating per-criterion and score-level metrics.

    Attributes:
        overall_accuracy: Overall criterion-level accuracy.
        mean_kappa: Mean Cohen's kappa across criteria.
        per_criterion: Per-criterion metrics breakdown.
        score_rmse: RMSE of cumulative scores.
        score_mae: MAE of cumulative scores.
        score_correlation: Score correlation results.
        n_items: Number of items evaluated.
        n_criteria: Number of criteria.
        cannot_assess_mode: How CANNOT_ASSESS was handled.
    """

    model_config = ConfigDict(frozen=True)

    overall_accuracy: float
    mean_kappa: float
    per_criterion: list[CriterionMetrics]
    score_rmse: float
    score_mae: float
    score_correlation: ScoreCorrelationResult
    n_items: int
    n_criteria: int
    cannot_assess_mode: CannotAssessMode = "exclude"


class DistributionResult(BaseModel):
    """Score distribution statistics.

    A statistic is ``None`` when it is genuinely undefined for the sample size, never a
    fake ``0.0``. At n=0 every stat is ``None``. A single point still has a defined
    mean/min/max/median/q25/q75 (and iqr = q75 − q25 = 0.0, the true IQR of one point);
    ``std``/``variance`` need ≥2 points, ``skewness`` ≥3, ``kurtosis`` ≥4 — each ``None``
    below its threshold. ``n`` is always the real count.

    Attributes:
        n: Number of samples.
        mean: Mean score. ``None`` when undefined (n = 0).
        std: Standard deviation. ``None`` when undefined (n < 2).
        variance: Variance. ``None`` when undefined (n < 2).
        min: Minimum score. ``None`` when undefined (n = 0).
        max: Maximum score. ``None`` when undefined (n = 0).
        median: Median score. ``None`` when undefined (n = 0).
        q25: 25th percentile. ``None`` when undefined (n = 0).
        q75: 75th percentile. ``None`` when undefined (n = 0).
        iqr: Interquartile range. ``None`` when undefined (n = 0); 0.0 for a single point.
        skewness: Skewness (measure of asymmetry). ``None`` when undefined (n < 3).
        kurtosis: Kurtosis (measure of tail heaviness). ``None`` when undefined (n < 4).
        histogram: Tuple of (counts, bin_edges).
    """

    model_config = ConfigDict(frozen=True)

    n: int
    mean: float | None
    std: float | None
    variance: float | None
    min: float | None
    max: float | None
    median: float | None
    q25: float | None
    q75: float | None
    iqr: float | None
    skewness: float | None
    kurtosis: float | None
    histogram: tuple[list[float], list[float]] | None = None


class EMDResult(BaseModel):
    """Result of Earth Mover's Distance computation.

    EMD measures the minimum "work" required to transform one distribution
    into another. Unlike correlation, it captures both shift (systematic bias)
    and shape differences (variance, skew).

    A statistic is ``None`` when it is genuinely undefined, never a fake ``0.0``. With an
    empty distribution on either side (no data to transport) ``emd``/``mean_diff``/
    ``std_diff``/``bias_magnitude`` are all ``None``; ``bias_direction`` stays ``"none"``
    and ``interpretation`` ``"insufficient data"``.

    Attributes:
        emd: Earth Mover's Distance (0 to ~1 if normalized). ``None`` for empty input.
        mean_diff: Difference in means (dist2 - dist1). ``None`` for empty input.
        std_diff: Difference in standard deviations. ``None`` for empty input.
        bias_direction: Whether dist1 tends higher, lower, or same.
        bias_magnitude: Absolute mean difference. ``None`` for empty input.
        interpretation: Human-readable interpretation.
    """

    model_config = ConfigDict(frozen=True)

    emd: float | None
    mean_diff: float | None
    std_diff: float | None
    bias_direction: Literal["higher", "lower", "none"]
    bias_magnitude: float | None
    interpretation: str

    @staticmethod
    def interpret_emd(emd: float) -> str:
        """Human-readable interpretation of EMD value."""
        if emd < 0.05:
            return "very similar"
        elif emd < 0.10:
            return "minor differences"
        elif emd < 0.20:
            return "moderate differences"
        else:
            return "substantial differences"


class KSTestResult(BaseModel):
    """Kolmogorov-Smirnov test result.

    The KS test compares two distributions and tests whether they
    come from the same underlying distribution.

    Attributes:
        statistic: KS test statistic.
        p_value: P-value for the test.
        is_significant: Whether the difference is significant (p < 0.05).
    """

    model_config = ConfigDict(frozen=True)

    statistic: float
    p_value: float
    is_significant: bool


class BootstrapResult(BaseModel):
    """Bootstrap confidence interval result.

    Attributes:
        estimate: Point estimate of the statistic.
        ci: Confidence interval from bootstrap.
        standard_error: Bootstrap standard error.
        n_bootstrap: Number of bootstrap samples used.
        bootstrap_distribution: Optional array of bootstrap estimates.
    """

    model_config = ConfigDict(frozen=True)

    estimate: float
    ci: ConfidenceInterval
    standard_error: float
    n_bootstrap: int
    bootstrap_distribution: list[float] | None = None


class MetricWithCI(BaseModel):
    """Any metric value with confidence interval.

    Attributes:
        metric_name: Name of the metric.
        value: Point estimate.
        ci: Confidence interval.
        n_samples: Number of samples used.
    """

    model_config = ConfigDict(frozen=True)

    metric_name: str
    value: float
    ci: ConfidenceInterval
    n_samples: int


class BootstrapResults(BaseModel):
    """Bootstrap confidence interval results.

    The three CIs are MARGINAL — bootstrapped on two independent item-level resample axes (a
    verdict-item axis for accuracy/kappa, an independent scored-item axis for RMSE), so they
    reflect each statistic's own sampling distribution, not their joint covariance. Covers any
    rubric type (binary / multi-choice / mixed).

    Attributes:
        accuracy_ci: 95% CI for ``criterion_accuracy``. None when undefined / no samples.
        kappa_ci: 95% CI for ``mean_kappa`` (ordinal contributes quadratic-weighted kappa).
            Each replicate's mean conditions on which criteria were non-degenerate in that
            resample. None when undefined (kappa never defined across resamples) / no samples.
        rmse_ci: 95% CI for ``score_rmse`` over the scored-item subset. None when no samples
            (a single scored item yields a degenerate ``(v, v)`` interval, not None).
        n_bootstrap: Number of bootstrap samples used.
        confidence_level: Confidence level (default 0.95).
    """

    model_config = ConfigDict(frozen=True)

    accuracy_ci: tuple[float, float] | None
    kappa_ci: tuple[float, float] | None
    rmse_ci: tuple[float, float] | None
    n_bootstrap: int
    confidence_level: float = 0.95


class JudgeMetrics(BaseModel):
    """Metrics for a single judge in an ensemble.

    Mirrors the aggregate's type handling field-for-field: precision/recall/f1 are the
    binary MET-vs-rest metric → ``None`` for a multi-choice-only rubric (no MET class),
    and accuracy/mean_kappa generalize but are ``None`` when undefined.

    Attributes:
        judge_id: Identifier for this judge.
        criterion_accuracy: Overall criterion-level accuracy (binary label and/or
            multi-choice exact-match). ``None`` when undefined.
        criterion_precision: Overall precision for the binary MET class. ``None`` when
            not applicable (multi-choice-only — no MET class).
        criterion_recall: Overall recall for the binary MET class. ``None`` when not
            applicable (see ``criterion_precision``).
        criterion_f1: Overall F1 for the binary MET class. ``None`` when not applicable
            (see ``criterion_precision``).
        mean_kappa: Mean Cohen's kappa across criteria. ``None`` when undefined.
        score_rmse: RMSE of cumulative scores.
        score_mae: MAE of cumulative scores.
        score_spearman: Spearman correlation result.
        score_kendall: Kendall tau correlation result.
        score_pearson: Pearson correlation result.
        bias: Systematic bias analysis result.
    """

    model_config = ConfigDict(frozen=True)

    judge_id: str
    criterion_accuracy: float | None
    criterion_precision: float | None
    criterion_recall: float | None
    criterion_f1: float | None
    mean_kappa: float | None
    score_rmse: float
    score_mae: float
    score_spearman: CorrelationResult
    score_kendall: CorrelationResult
    score_pearson: CorrelationResult
    bias: BiasResult


class MetricsResult(BaseModel):
    """Complete metrics result from compute_metrics().

    This is the main result type returned by EvalResult.compute_metrics().
    It provides a comprehensive view of evaluation quality including:
    - Criterion-level agreement metrics
    - Score-level correlation and error metrics
    - Per-criterion breakdown (supports binary, ordinal, and nominal criteria)
    - Optional bootstrap confidence intervals
    - Optional per-judge metrics for ensemble evaluations

    Attributes:
        criterion_accuracy: Overall accuracy across all criteria (binary label accuracy
            and/or multi-choice exact-match). ``None`` when undefined (no comparable
            pairs at all).
        criterion_precision: Overall precision for the binary MET class. ``None`` when
            not applicable — multi-choice-only rubrics have no MET class (the per-option
            precision/recall/f1 story lives in each criterion's ``per_option``).
        criterion_recall: Overall recall for the binary MET class. ``None`` when not
            applicable (see ``criterion_precision``).
        criterion_f1: Overall F1 for the binary MET class. ``None`` when not applicable
            (see ``criterion_precision``).
        mean_kappa: Mean kappa across criteria (weighted for ordinal, unweighted for
            binary/nominal). ``None`` when undefined (no criterion contributed a kappa).
        per_criterion: Per-criterion metrics breakdown (polymorphic union type).
        score_rmse: RMSE of cumulative scores.
        score_mae: MAE of cumulative scores.
        score_spearman: Spearman correlation result.
        score_kendall: Kendall tau correlation result.
        score_pearson: Pearson correlation result.
        bias: Systematic bias analysis.
        bootstrap: Optional bootstrap confidence intervals.
        per_judge: Optional per-judge metrics for ensemble.
        n_items: Number of items used in computation.
        n_criteria: Number of criteria.
        n_binary_criteria: Number of binary criteria (default 0 for backwards compat).
        n_ordinal_criteria: Number of ordinal multi-choice criteria.
        n_nominal_criteria: Number of nominal multi-choice criteria.
        na_stats: Statistics for NA handling in multi-choice criteria.
        cannot_assess_stats: Statistics for CANNOT_ASSESS handling in binary criteria —
            the binary parallel to ``na_stats`` (a distinct kind of abstention; see
            CannotAssessStats).
        warnings: Any warnings generated during computation.
    """

    model_config = ConfigDict(frozen=True)

    # Aggregate criterion-level metrics. precision/recall/f1 are the binary MET-vs-rest
    # metric → None for multi-choice-only rubrics (no MET class). accuracy/mean_kappa
    # generalize across types but are None when genuinely undefined (no comparable pairs).
    criterion_accuracy: float | None
    criterion_precision: float | None
    criterion_recall: float | None
    criterion_f1: float | None
    mean_kappa: float | None

    # Per-criterion breakdown (supports union type)
    per_criterion: list[CriterionMetricsUnion]

    # Score-level metrics
    score_rmse: float
    score_mae: float
    score_spearman: CorrelationResult
    score_kendall: CorrelationResult
    score_pearson: CorrelationResult

    # Bias analysis
    bias: BiasResult

    # Optional results
    bootstrap: BootstrapResults | None = None
    per_judge: dict[str, JudgeMetrics] | None = None

    # Metadata
    n_items: int
    n_criteria: int
    n_binary_criteria: int = 0
    n_ordinal_criteria: int = 0
    n_nominal_criteria: int = 0
    na_stats: NAStats | None = None
    cannot_assess_stats: CannotAssessStats | None = None
    warnings: list[str] = []

    def summary(self) -> str:
        """Return formatted text summary of metrics."""
        lines = []
        lines.append("=" * 60)
        lines.append("METRICS SUMMARY")
        lines.append("=" * 60)

        # Show criteria type breakdown if mixed
        criteria_info = f"Items: {self.n_items}, Criteria: {self.n_criteria}"
        if self.n_ordinal_criteria > 0 or self.n_nominal_criteria > 0:
            type_parts = []
            if self.n_binary_criteria > 0:
                type_parts.append(f"{self.n_binary_criteria} binary")
            if self.n_ordinal_criteria > 0:
                type_parts.append(f"{self.n_ordinal_criteria} ordinal")
            if self.n_nominal_criteria > 0:
                type_parts.append(f"{self.n_nominal_criteria} nominal")
            criteria_info += f" ({', '.join(type_parts)})"
        lines.append(criteria_info)

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")

        lines.append("")
        lines.append("Criterion-Level Metrics:")
        lines.append(f"  Accuracy:   {_fmt_opt(self.criterion_accuracy, '.1%')}")
        if self.n_binary_criteria > 0:
            # Guaranteed non-None here, but render via _fmt_opt so ty is satisfied.
            lines.append(f"  Precision:  {_fmt_opt(self.criterion_precision, '.2f')}")
            lines.append(f"  Recall:     {_fmt_opt(self.criterion_recall, '.2f')}")
            lines.append(f"  F1:         {_fmt_opt(self.criterion_f1, '.2f')}")
        lines.append(f"  Mean Kappa: {_fmt_opt(self.mean_kappa, '.3f')}")

        lines.append("")
        lines.append("Score-Level Metrics:")
        lines.append(f"  RMSE:     {self.score_rmse:.4f}")
        lines.append(f"  MAE:      {self.score_mae:.4f}")
        lines.append(
            f"  Spearman: {_fmt_opt(self.score_spearman.coefficient, '.4f')} "
            f"({self.score_spearman.interpretation})"
        )
        lines.append(
            f"  Kendall:  {_fmt_opt(self.score_kendall.coefficient, '.4f')} "
            f"({self.score_kendall.interpretation})"
        )
        lines.append(
            f"  Pearson:  {_fmt_opt(self.score_pearson.coefficient, '.4f')} "
            f"({self.score_pearson.interpretation})"
        )

        lines.append("")
        lines.append("Bias Analysis:")
        lines.append(
            f"  Mean Bias:   {_fmt_opt(self.bias.mean_bias, '+.4f')} ({self.bias.direction})"
        )
        lines.append(f"  Significant: {'Yes' if self.bias.is_significant else 'No'}")

        # NA stats for multi-choice
        if self.na_stats:
            lines.append("")
            lines.append("NA Handling:")
            lines.append(f"  NA in Ground Truth: {self.na_stats.na_count_true}")
            lines.append(f"  NA in Predictions:  {self.na_stats.na_count_pred}")
            if self.na_stats.na_kappa is not None:
                interp = self.na_stats.na_kappa_interpretation or ""
                lines.append(f"  NA Kappa:           {self.na_stats.na_kappa:.3f} ({interp})")
            if self.na_stats.na_false_positive > 0 or self.na_stats.na_false_negative > 0:
                lines.append(
                    f"  NA FP/FN:           {self.na_stats.na_false_positive} / "
                    f"{self.na_stats.na_false_negative}"
                )

        # CANNOT_ASSESS stats for binary criteria (parallel to NA Handling above; a
        # distinct kind of abstention — epistemic MET/UNMET rather than "no option").
        if self.cannot_assess_stats:
            ca = self.cannot_assess_stats
            lines.append("")
            lines.append("CANNOT_ASSESS Handling:")
            lines.append(f"  CA in Ground Truth: {ca.ca_count_true}")
            lines.append(f"  CA in Predictions:  {ca.ca_count_pred}")
            if ca.ca_kappa is not None:
                interp = ca.ca_kappa_interpretation or ""
                lines.append(f"  CA Kappa:           {ca.ca_kappa:.3f} ({interp})")
            if ca.ca_false_positive > 0 or ca.ca_false_negative > 0:
                lines.append(
                    f"  CA FP/FN:           {ca.ca_false_positive} / {ca.ca_false_negative}"
                )

        if self.bootstrap:
            lines.append("")
            lines.append(f"Bootstrap CIs ({self.bootstrap.confidence_level:.0%}):")
            # Each CI may be None (genuinely undefined / no samples) — render "n/a".
            acc_ci = self.bootstrap.accuracy_ci
            kappa_ci = self.bootstrap.kappa_ci
            rmse_ci = self.bootstrap.rmse_ci
            lines.append(
                "  Accuracy: "
                + (f"[{acc_ci[0]:.1%}, {acc_ci[1]:.1%}]" if acc_ci is not None else "n/a")
            )
            lines.append(
                "  Kappa:    "
                + (f"[{kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}]" if kappa_ci is not None else "n/a")
            )
            lines.append(
                "  RMSE:     "
                + (f"[{rmse_ci[0]:.4f}, {rmse_ci[1]:.4f}]" if rmse_ci is not None else "n/a")
            )

        if self.per_judge:
            lines.append("")
            lines.append("Per-Judge Metrics:")
            for judge_id, jm in sorted(self.per_judge.items()):
                lines.append(
                    f"  {judge_id}: RMSE={jm.score_rmse:.4f}, "
                    f"Spearman={_fmt_opt(jm.score_spearman.coefficient, '.4f')}"
                )

        lines.append("")
        lines.append("Per-Criterion Breakdown:")

        # Inter-judge agreement (Krippendorff's alpha + Fleiss' kappa) is only populated for
        # ensembles with >=2 judges; append those columns to a group's table only when at
        # least one criterion in the group carries them. Krippendorff's alpha is the
        # recommended statistic, so it leads.
        def _has_agreement(criteria: list) -> bool:
            return any(
                cm.krippendorff_alpha is not None or cm.fleiss_kappa is not None for cm in criteria
            )

        def _agreement_header() -> str:
            return f" {'Kripp-α':>9} {'Fleiss':>8}"

        def _agreement_cells(cm) -> str:
            return (
                f" {_fmt_opt(cm.krippendorff_alpha, '>9.3f', 9)}"
                f" {_fmt_opt(cm.fleiss_kappa, '>8.3f', 8)}"
            )

        # Separate display by criterion type
        binary_criteria = [cm for cm in self.per_criterion if cm.criterion_type == "binary"]
        ordinal_criteria = [cm for cm in self.per_criterion if cm.criterion_type == "ordinal"]
        nominal_criteria = [cm for cm in self.per_criterion if cm.criterion_type == "nominal"]

        if binary_criteria:
            if ordinal_criteria or nominal_criteria:
                lines.append("\nBinary Criteria:")
            show_agreement = _has_agreement(binary_criteria)
            header = f"{'Criterion':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Kappa':>8}"
            if show_agreement:
                header += _agreement_header()
            lines.append(header)
            lines.append("-" * len(header))
            for cm in binary_criteria:
                row = (
                    f"{cm.name:<20} {_fmt_opt(cm.accuracy, '>8.1%', 8)} "
                    f"{_fmt_opt(cm.precision, '>8.2f', 8)} "
                    f"{_fmt_opt(cm.recall, '>8.2f', 8)} {_fmt_opt(cm.f1, '>8.2f', 8)} "
                    f"{_fmt_opt(cm.kappa, '>8.3f', 8)}"
                )
                if show_agreement:
                    row += _agreement_cells(cm)
                lines.append(row)

        if ordinal_criteria:
            lines.append("\nOrdinal Criteria:")
            show_agreement = _has_agreement(ordinal_criteria)
            header = (
                f"{'Criterion':<20} {'Exact':>8} {'Adj':>8} "
                f"{'WKappa':>8} {'Spearman':>10} {'RMSE':>8}"
            )
            if show_agreement:
                header += _agreement_header()
            lines.append(header)
            lines.append("-" * len(header))
            for cm in ordinal_criteria:
                row = (
                    f"{cm.name:<20} {_fmt_opt(cm.exact_accuracy, '>8.1%', 8)} "
                    f"{_fmt_opt(cm.adjacent_accuracy, '>8.1%', 8)} "
                    f"{_fmt_opt(cm.weighted_kappa, '>8.3f', 8)} "
                    f"{_fmt_opt(cm.spearman.coefficient, '>10.4f', 10)} "
                    f"{_fmt_opt(cm.rmse, '>8.4f', 8)}"
                )
                if show_agreement:
                    row += _agreement_cells(cm)
                lines.append(row)

        if nominal_criteria:
            lines.append("\nNominal Criteria:")
            show_agreement = _has_agreement(nominal_criteria)
            header = f"{'Criterion':<20} {'Accuracy':>10} {'Kappa':>8} {'Interpretation':<20}"
            if show_agreement:
                header += _agreement_header()
            lines.append(header)
            lines.append("-" * len(header))
            for cm in nominal_criteria:
                row = (
                    f"{cm.name:<20} {_fmt_opt(cm.exact_accuracy, '>10.1%', 10)} "
                    f"{_fmt_opt(cm.kappa, '>8.3f', 8)} {cm.kappa_interpretation:<20}"
                )
                if show_agreement:
                    row += _agreement_cells(cm)
                lines.append(row)

        return "\n".join(lines)

    def to_dataframe(self) -> "pd.DataFrame":
        """Export metrics to pandas DataFrame.

        Returns a flat DataFrame with a 'level' column indicating:
        - 'aggregate': Overall metrics
        - 'criterion': Per-criterion metrics (binary)
        - 'criterion_ordinal': Per-criterion metrics (ordinal)
        - 'criterion_nominal': Per-criterion metrics (nominal)
        - 'judge': Per-judge metrics (if available)
        """
        import pandas as pd

        rows = []

        # Aggregate row
        rows.append(
            {
                "level": "aggregate",
                "name": "overall",
                "criterion_type": "all",
                "accuracy": self.criterion_accuracy,
                "precision": self.criterion_precision,
                "recall": self.criterion_recall,
                "f1": self.criterion_f1,
                "kappa": self.mean_kappa,
                "rmse": self.score_rmse,
                "mae": self.score_mae,
                "spearman": self.score_spearman.coefficient,
                "kendall": self.score_kendall.coefficient,
                "pearson": self.score_pearson.coefficient,
                "bias": self.bias.mean_bias,
                "adjacent_accuracy": None,
                "weighted_kappa": None,
                "krippendorff_alpha": None,
                "fleiss_kappa": None,
            }
        )

        # Per-criterion rows (handle different types)
        for cm in self.per_criterion:
            if cm.criterion_type == "binary":
                rows.append(
                    {
                        "level": "criterion",
                        "name": cm.name,
                        "criterion_type": "binary",
                        "accuracy": cm.accuracy,
                        "precision": cm.precision,
                        "recall": cm.recall,
                        "f1": cm.f1,
                        "kappa": cm.kappa,
                        "rmse": None,
                        "mae": None,
                        "spearman": None,
                        "kendall": None,
                        "pearson": None,
                        "bias": None,
                        "adjacent_accuracy": None,
                        "weighted_kappa": None,
                        "krippendorff_alpha": cm.krippendorff_alpha,
                        "fleiss_kappa": cm.fleiss_kappa,
                    }
                )
            elif cm.criterion_type == "ordinal":
                rows.append(
                    {
                        "level": "criterion",
                        "name": cm.name,
                        "criterion_type": "ordinal",
                        "accuracy": cm.exact_accuracy,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "kappa": cm.weighted_kappa,
                        "rmse": cm.rmse,
                        "mae": cm.mae,
                        "spearman": cm.spearman.coefficient,
                        "kendall": cm.kendall.coefficient,
                        "pearson": None,
                        "bias": None,
                        "adjacent_accuracy": cm.adjacent_accuracy,
                        "weighted_kappa": cm.weighted_kappa,
                        "krippendorff_alpha": cm.krippendorff_alpha,
                        "fleiss_kappa": cm.fleiss_kappa,
                    }
                )
            else:  # nominal
                rows.append(
                    {
                        "level": "criterion",
                        "name": cm.name,
                        "criterion_type": "nominal",
                        "accuracy": cm.exact_accuracy,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "kappa": cm.kappa,
                        "rmse": None,
                        "mae": None,
                        "spearman": None,
                        "kendall": None,
                        "pearson": None,
                        "bias": None,
                        "adjacent_accuracy": None,
                        "weighted_kappa": None,
                        "krippendorff_alpha": cm.krippendorff_alpha,
                        "fleiss_kappa": cm.fleiss_kappa,
                    }
                )

        # Per-judge rows (if available)
        if self.per_judge:
            for judge_id, jm in self.per_judge.items():
                rows.append(
                    {
                        "level": "judge",
                        "name": judge_id,
                        "criterion_type": "all",
                        "accuracy": jm.criterion_accuracy,
                        "precision": jm.criterion_precision,
                        "recall": jm.criterion_recall,
                        "f1": jm.criterion_f1,
                        "kappa": jm.mean_kappa,
                        "rmse": jm.score_rmse,
                        "mae": jm.score_mae,
                        "spearman": jm.score_spearman.coefficient,
                        "kendall": jm.score_kendall.coefficient,
                        "pearson": jm.score_pearson.coefficient,
                        "bias": jm.bias.mean_bias,
                        "adjacent_accuracy": None,
                        "weighted_kappa": None,
                        "krippendorff_alpha": None,
                        "fleiss_kappa": None,
                    }
                )

        return pd.DataFrame(rows)

    def to_file(self, path: str | Path) -> None:
        """Save metrics to a JSON file.

        Args:
            path: Path to the output JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
