"""Core compute_metrics implementation.

This module provides the main compute_metrics function that computes
comprehensive evaluation metrics from an EvalResult and RubricDataset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from ..types import Criterion, CriterionVerdict
from ._helpers import (
    classify_criteria,
    extract_all_verdicts_from_report,
    filter_na_multi_choice,
    get_option_value,
    prepare_binary_metric_inputs,
    resolve_ground_truth,
)
from ._types import (
    BootstrapResults,
    CannotAssessMode,
    CorrelationResult,
    CriterionMetrics,
    CriterionMetricsUnion,
    JudgeMetrics,
    KappaResult,
    MetricsResult,
    NAMode,
    NAStats,
    NominalCriterionMetrics,
    OptionMetrics,
    OrdinalCriterionMetrics,
)
from .distribution import systematic_bias

# Try to import Fleiss' kappa from statsmodels (hard dep; guard kept for safety).
# Import to a temp name and bind to a fresh variable so the guard's `= None` fallback
# doesn't conflict with the imported symbol's declared type.
_fleiss_kappa: Any = None
HAS_STATSMODELS = False
try:
    from statsmodels.stats.inter_rater import fleiss_kappa as _sm_fleiss_kappa

    _fleiss_kappa = _sm_fleiss_kappa
    HAS_STATSMODELS = True
except ImportError:
    pass

# Krippendorff's alpha is the general, recommended inter-judge agreement statistic
# (handles unequal/missing raters and is level-aware). Hard dep; guard kept for safety.
_krippendorff: Any = None
HAS_KRIPPENDORFF = False
try:
    import krippendorff as _kd

    _krippendorff = _kd
    HAS_KRIPPENDORFF = True
except ImportError:
    pass

if TYPE_CHECKING:
    from ..dataset import RubricDataset
    from ..eval import EvalResult


def _interpret_kappa(kappa: float) -> str:
    """Return human-readable interpretation of kappa value.

    Based on Landis & Koch (1977) guidelines.
    """
    if kappa < 0:
        return "poor"
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


def _interpret_correlation(r: float) -> str:
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


# =============================================================================
# Multi-choice Metric Functions
# =============================================================================


def _compute_per_option_metrics(
    pred_indices: list[int],
    true_indices: list[int],
    criterion: Criterion,
) -> list[OptionMetrics]:
    """Compute precision/recall/F1 for each option in a multi-choice criterion.

    Args:
        pred_indices: Predicted option indices.
        true_indices: Ground truth option indices.
        criterion: The multi-choice criterion.

    Returns:
        List of OptionMetrics, one per option.
    """
    n_options = len(criterion.options)
    option_metrics = []

    for opt_idx in range(n_options):
        label = criterion.options[opt_idx].label

        # Binary classification: is this option selected or not?
        pred_binary = [1 if p == opt_idx else 0 for p in pred_indices]
        true_binary = [1 if t == opt_idx else 0 for t in true_indices]

        support_true = sum(true_binary)
        support_pred = sum(pred_binary)

        if not pred_binary or not true_binary:
            option_metrics.append(
                OptionMetrics(
                    label=label,
                    index=opt_idx,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    support_true=support_true,
                    support_pred=support_pred,
                )
            )
            continue

        # Compute metrics
        opt_precision = precision_score(true_binary, pred_binary, zero_division=0)
        opt_recall = recall_score(true_binary, pred_binary, zero_division=0)
        opt_f1 = f1_score(true_binary, pred_binary, zero_division=0)

        option_metrics.append(
            OptionMetrics(
                label=label,
                index=opt_idx,
                precision=float(opt_precision),
                recall=float(opt_recall),
                f1=float(opt_f1),
                support_true=support_true,
                support_pred=support_pred,
            )
        )

    return option_metrics


def _compute_confusion_matrix(
    pred_indices: list[int],
    true_indices: list[int],
    n_options: int,
) -> list[list[int]]:
    """Compute confusion matrix for multi-choice predictions.

    Args:
        pred_indices: Predicted option indices.
        true_indices: Ground truth option indices.
        n_options: Number of options (determines matrix size).

    Returns:
        N×N confusion matrix as nested lists (row=true, col=pred).
    """
    if not pred_indices or not true_indices:
        return [[0] * n_options for _ in range(n_options)]

    # sklearn's confusion_matrix may not include all labels if not present
    cm = confusion_matrix(
        true_indices,
        pred_indices,
        labels=list(range(n_options)),
    )
    return cm.tolist()


def _compute_adjacent_accuracy(
    pred_indices: list[int],
    true_indices: list[int],
) -> float:
    """Compute adjacent accuracy (prediction within ±1 of true).

    Only meaningful for ordinal scales.

    Args:
        pred_indices: Predicted option indices.
        true_indices: Ground truth option indices.

    Returns:
        Proportion of predictions within ±1 of ground truth.
    """
    if not pred_indices:
        return 0.0

    adjacent_correct = sum(1 for p, t in zip(pred_indices, true_indices) if abs(p - t) <= 1)
    return adjacent_correct / len(pred_indices)


def _compute_fleiss_kappa(
    ratings_matrix: list[list[int]] | None,
) -> float | None:
    """Compute Fleiss' kappa for multi-rater agreement.

    Args:
        ratings_matrix: Matrix where each row is a subject and each column
            contains the count of raters who assigned each category.
            Shape: (n_subjects, n_categories)

    Returns:
        Fleiss' kappa value, or None if statsmodels not available or
        insufficient data.
    """
    if not HAS_STATSMODELS or _fleiss_kappa is None:
        return None

    if not ratings_matrix or len(ratings_matrix) < 2:
        return None

    try:
        # statsmodels expects numpy array
        matrix = np.array(ratings_matrix)
        result = float(_fleiss_kappa(matrix))
    except Exception:
        return None

    # Varying rater counts per subject (from error/CA exclusions) can yield NaN.
    if math.isnan(result):
        return None
    return result


def _compute_krippendorff_alpha(
    reliability_data: list[list[float]] | None,
    level: Literal["nominal", "ordinal"],
) -> float | None:
    """Compute Krippendorff's alpha — the general inter-judge agreement statistic.

    Unlike Fleiss' kappa, alpha natively handles missing/unequal raters and is
    level-aware (``"nominal"`` vs ``"ordinal"``), so it is the recommended statistic
    for all criterion types.

    Args:
        reliability_data: 2D matrix shaped (raters x units). Cells are numeric codes,
            with ``np.nan`` marking a missing rating (errored/excluded/absent judge).
        level: ``"nominal"`` or ``"ordinal"``.

    Returns:
        Krippendorff's alpha, or None if krippendorff is unavailable, there are < 2
        units, or the value is undefined (NaN) / computation fails.
    """
    if not HAS_KRIPPENDORFF or _krippendorff is None:
        return None

    if not reliability_data or len(reliability_data) < 1:
        return None

    # Need at least 2 units (columns) for agreement to be defined.
    n_units = len(reliability_data[0]) if reliability_data else 0
    if n_units < 2:
        return None

    try:
        matrix = np.array(reliability_data, dtype=float)
        result = float(_krippendorff.alpha(reliability_data=matrix, level_of_measurement=level))
    except Exception:
        return None

    if math.isnan(result):
        return None
    return result


def _build_fleiss_row(
    cr: object,
    criterion: Criterion,
    c_type: str,
    cannot_assess: CannotAssessMode,
    n_judges: int,
) -> list[int] | None:
    """Build one complete-case Fleiss' kappa subject row for a criterion.

    Counts genuine (error-free) ensemble votes per category for a single item. statsmodels'
    ``fleiss_kappa`` requires a uniform number of raters per subject, so a row is included
    ONLY if its counted votes sum to exactly ``n_judges`` (i.e. every judge cast a genuine,
    counted vote). Items with any errored / excluded / CANNOT_ASSESS-under-``exclude`` vote
    are dropped from the Fleiss matrix (they remain in Krippendorff's alpha as missing cells).

    Args:
        cr: An ``EnsembleCriterionReport`` (single-judge ``CriterionReport``s lack votes).
        criterion: The criterion (supplies option count for multi-choice).
        c_type: One of "binary", "ordinal", "nominal".
        cannot_assess: How to map CANNOT_ASSESS for binary criteria.
        n_judges: Number of judges in the ensemble (required complete-case rater count).

    Returns:
        A list of per-category counts summing to ``n_judges``, or None otherwise.
    """
    if c_type == "binary":
        # MET=0, UNMET=1, CANNOT_ASSESS=2 (CA column only when as_category).
        n_cats = 3 if cannot_assess == "as_category" else 2
        votes = getattr(cr, "votes", None)
        if votes is None:  # single-judge CriterionReport: no ensemble votes
            return None
        counts = [0] * n_cats
        for v in votes:
            if v.error is not None:
                continue
            verdict = v.verdict
            if verdict == CriterionVerdict.CANNOT_ASSESS:
                if cannot_assess == "exclude":
                    continue
                if cannot_assess == "as_unmet":
                    counts[1] += 1
                    continue
                # as_category
                counts[2] += 1
                continue
            if verdict == CriterionVerdict.MET:
                counts[0] += 1
            else:
                counts[1] += 1
    else:
        # Multi-choice: one column per option; genuine NA is an ordinary column.
        n_cats = len(criterion.options or [])
        mc_votes = getattr(cr, "multi_choice_votes", None)
        if mc_votes is None:
            return None
        counts = [0] * n_cats
        for v in mc_votes:
            if v.error is not None:
                continue
            if 0 <= v.selected_index < n_cats:
                counts[v.selected_index] += 1

    # Complete-case: only include subjects rated by every judge (uniform rater count).
    if n_judges < 2 or sum(counts) != n_judges:
        return None
    return counts


def _build_alpha_cell(
    vote: object,
    c_type: str,
    cannot_assess: CannotAssessMode,
) -> float:
    """Map a single judge vote to its Krippendorff reliability-matrix cell value.

    Errored votes, and CANNOT_ASSESS under ``exclude``, become ``np.nan`` (missing).

    Args:
        vote: A ``JudgeVote`` (binary) or ``MultiChoiceJudgeVote`` (multi-choice).
        c_type: One of "binary", "ordinal", "nominal".
        cannot_assess: How to map CANNOT_ASSESS for binary criteria.

    Returns:
        The numeric code for the cell, or ``np.nan`` when the rating is missing.
    """
    if getattr(vote, "error", None) is not None:
        return float("nan")

    if c_type == "binary":
        # MET=0, UNMET=1, CANNOT_ASSESS=2 (only under as_category).
        verdict = vote.verdict
        if verdict == CriterionVerdict.CANNOT_ASSESS:
            if cannot_assess == "exclude":
                return float("nan")
            if cannot_assess == "as_unmet":
                return 1.0
            return 2.0  # as_category
        return 0.0 if verdict == CriterionVerdict.MET else 1.0

    # Multi-choice (ordinal/nominal): code = selected option index (genuine NA included).
    return float(vote.selected_index)


def _compute_ordinal_criterion_metrics(
    pred_indices: list[int],
    true_indices: list[int],
    criterion: Criterion,
    index: int,
    fleiss_matrix: list[list[int]] | None = None,
    krippendorff_alpha: float | None = None,
) -> OrdinalCriterionMetrics:
    """Compute metrics for an ordinal multi-choice criterion.

    Args:
        pred_indices: Predicted option indices.
        true_indices: Ground truth option indices.
        criterion: The ordinal criterion.
        index: Index of this criterion in the rubric.
        fleiss_matrix: Optional ratings matrix for Fleiss' kappa (ensemble).
        krippendorff_alpha: Optional precomputed Krippendorff's alpha (ensemble).

    Returns:
        OrdinalCriterionMetrics with comprehensive ordinal metrics.
    """
    name = criterion.name or f"Criterion {index + 1}"
    n_options = len(criterion.options)
    n_samples = len(pred_indices)
    option_labels = [opt.label for opt in criterion.options]

    # Handle empty data
    if n_samples == 0:
        return OrdinalCriterionMetrics(
            name=name,
            index=index,
            n_samples=0,
            n_options=n_options,
            exact_accuracy=0.0,
            adjacent_accuracy=0.0,
            weighted_kappa=0.0,
            kappa_interpretation="undefined",
            krippendorff_alpha=krippendorff_alpha,
            fleiss_kappa=None,
            spearman=CorrelationResult(
                coefficient=0.0,
                p_value=1.0,
                interpretation="insufficient data",
                n_samples=0,
                method="spearman",
            ),
            kendall=CorrelationResult(
                coefficient=0.0,
                p_value=1.0,
                interpretation="insufficient data",
                n_samples=0,
                method="kendall",
            ),
            rmse=0.0,
            mae=0.0,
            per_option=[],
            confusion_matrix=[[0] * n_options for _ in range(n_options)],
            option_labels=option_labels,
        )

    # Exact accuracy
    exact_accuracy = accuracy_score(true_indices, pred_indices)

    # Adjacent accuracy (within ±1)
    adjacent_accuracy = _compute_adjacent_accuracy(pred_indices, true_indices)

    # Weighted kappa (quadratic weights for ordinal)
    try:
        weighted_kappa = cohen_kappa_score(true_indices, pred_indices, weights="quadratic")
    except Exception:
        weighted_kappa = 0.0

    # Fleiss' kappa (for ensemble with 3+ judges)
    fleiss_kappa = None
    if fleiss_matrix is not None:
        fleiss_kappa = _compute_fleiss_kappa(fleiss_matrix)

    # Convert indices to option values for correlation/RMSE
    pred_values = [get_option_value(criterion, i) for i in pred_indices]
    true_values = [get_option_value(criterion, i) for i in true_indices]

    # Correlations
    spearman = _compute_correlation(pred_values, true_values, "spearman")
    kendall = _compute_correlation(pred_values, true_values, "kendall")

    # RMSE and MAE on option values
    rmse = float(np.sqrt(mean_squared_error(true_values, pred_values)))
    mae = float(mean_absolute_error(true_values, pred_values))

    # Per-option metrics
    per_option = _compute_per_option_metrics(pred_indices, true_indices, criterion)

    # Confusion matrix
    conf_matrix = _compute_confusion_matrix(pred_indices, true_indices, n_options)

    return OrdinalCriterionMetrics(
        name=name,
        index=index,
        n_samples=n_samples,
        n_options=n_options,
        exact_accuracy=float(exact_accuracy),
        adjacent_accuracy=float(adjacent_accuracy),
        weighted_kappa=float(weighted_kappa),
        kappa_interpretation=_interpret_kappa(weighted_kappa),
        krippendorff_alpha=krippendorff_alpha,
        fleiss_kappa=fleiss_kappa,
        spearman=spearman,
        kendall=kendall,
        rmse=rmse,
        mae=mae,
        per_option=per_option,
        confusion_matrix=conf_matrix,
        option_labels=option_labels,
    )


def _compute_nominal_criterion_metrics(
    pred_indices: list[int],
    true_indices: list[int],
    criterion: Criterion,
    index: int,
    fleiss_matrix: list[list[int]] | None = None,
    krippendorff_alpha: float | None = None,
) -> NominalCriterionMetrics:
    """Compute metrics for a nominal multi-choice criterion.

    Args:
        pred_indices: Predicted option indices.
        true_indices: Ground truth option indices.
        criterion: The nominal criterion.
        index: Index of this criterion in the rubric.
        fleiss_matrix: Optional ratings matrix for Fleiss' kappa (ensemble).
        krippendorff_alpha: Optional precomputed Krippendorff's alpha (ensemble).

    Returns:
        NominalCriterionMetrics with comprehensive nominal metrics.
    """
    name = criterion.name or f"Criterion {index + 1}"
    n_options = len(criterion.options)
    n_samples = len(pred_indices)
    option_labels = [opt.label for opt in criterion.options]

    # Handle empty data
    if n_samples == 0:
        return NominalCriterionMetrics(
            name=name,
            index=index,
            n_samples=0,
            n_options=n_options,
            exact_accuracy=0.0,
            kappa=0.0,
            kappa_interpretation="undefined",
            krippendorff_alpha=krippendorff_alpha,
            fleiss_kappa=None,
            per_option=[],
            confusion_matrix=[[0] * n_options for _ in range(n_options)],
            option_labels=option_labels,
        )

    # Exact accuracy
    exact_accuracy = accuracy_score(true_indices, pred_indices)

    # Unweighted kappa (nominal scale - no ordering)
    try:
        kappa = cohen_kappa_score(true_indices, pred_indices)
    except Exception:
        kappa = 0.0

    # Fleiss' kappa (for ensemble with 3+ judges)
    fleiss_kappa = None
    if fleiss_matrix is not None:
        fleiss_kappa = _compute_fleiss_kappa(fleiss_matrix)

    # Per-option metrics
    per_option = _compute_per_option_metrics(pred_indices, true_indices, criterion)

    # Confusion matrix
    conf_matrix = _compute_confusion_matrix(pred_indices, true_indices, n_options)

    return NominalCriterionMetrics(
        name=name,
        index=index,
        n_samples=n_samples,
        n_options=n_options,
        exact_accuracy=float(exact_accuracy),
        kappa=float(kappa),
        kappa_interpretation=_interpret_kappa(kappa),
        krippendorff_alpha=krippendorff_alpha,
        fleiss_kappa=fleiss_kappa,
        per_option=per_option,
        confusion_matrix=conf_matrix,
        option_labels=option_labels,
    )


def _compute_correlation(x: list[float], y: list[float], method: str) -> CorrelationResult:
    """Compute correlation with interpretation."""
    if len(x) < 3:
        return CorrelationResult(
            coefficient=0.0,
            p_value=1.0,
            interpretation="insufficient data",
            n_samples=len(x),
            method=method,
        )

    x_arr = np.array(x)
    y_arr = np.array(y)

    if method == "spearman":
        coef, p_val = stats.spearmanr(x_arr, y_arr)
    elif method == "kendall":
        coef, p_val = stats.kendalltau(x_arr, y_arr)
    else:  # pearson
        coef, p_val = stats.pearsonr(x_arr, y_arr)

    # Handle NaN from constant arrays
    if np.isnan(coef):
        coef = 0.0
        p_val = 1.0

    return CorrelationResult(
        coefficient=float(coef),
        p_value=float(p_val),
        interpretation=_interpret_correlation(float(coef)),
        n_samples=len(x),
        method=method,
    )


def _compute_bootstrap_ci(
    y_true: Sequence[int] | Sequence[str],
    y_pred: Sequence[int] | Sequence[str],
    true_scores: list[float],
    pred_scores: list[float],
    n_bootstrap: int,
    confidence_level: float,
    seed: int | None,
) -> BootstrapResults:
    """Compute bootstrap confidence intervals for key metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    if n == 0:
        return BootstrapResults(
            accuracy_ci=(0.0, 0.0),
            kappa_ci=(0.0, 0.0),
            rmse_ci=(0.0, 0.0),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    true_scores_arr = np.array(true_scores)
    pred_scores_arr = np.array(pred_scores)

    acc_samples = []
    kappa_samples = []
    rmse_samples = []

    for _ in range(n_bootstrap):
        # Sample indices with replacement
        idx = rng.choice(n, size=n, replace=True)

        # Criterion-level metrics
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]

        if len(np.unique(yt)) > 1 and len(np.unique(yp)) > 1:
            acc_samples.append(accuracy_score(yt, yp))
            try:
                kappa_samples.append(cohen_kappa_score(yt, yp))
            except Exception:
                pass
        else:
            acc_samples.append(accuracy_score(yt, yp))

        # Score-level metrics
        score_idx = rng.choice(len(true_scores), size=len(true_scores), replace=True)
        ts = true_scores_arr[score_idx]
        ps = pred_scores_arr[score_idx]
        rmse_samples.append(np.sqrt(mean_squared_error(ts, ps)))

    alpha = 1 - confidence_level
    lower_q = alpha / 2 * 100
    upper_q = (1 - alpha / 2) * 100

    def get_ci(samples: list[float]) -> tuple[float, float]:
        if not samples:
            return (0.0, 0.0)
        return (
            float(np.percentile(samples, lower_q)),
            float(np.percentile(samples, upper_q)),
        )

    return BootstrapResults(
        accuracy_ci=get_ci(acc_samples),
        kappa_ci=get_ci(kappa_samples) if kappa_samples else (0.0, 0.0),
        rmse_ci=get_ci(rmse_samples),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def _compute_judge_metrics(
    judge_id: str,
    judge_scores: list[float],
    true_scores: list[float],
    judge_verdicts: list[list[CriterionVerdict]],
    judge_errors: list[list[str | None]],
    true_verdicts: list[list[CriterionVerdict | int]],
    criterion_types: list[str],
    cannot_assess: CannotAssessMode,
) -> JudgeMetrics:
    """Compute metrics for a single judge (binary criteria only).

    ``judge_verdicts``, ``judge_errors`` and ``true_verdicts`` are all items x criteria
    and aligned 1:1. Only binary criteria with a genuine (error-free) judge vote and a
    ``CriterionVerdict`` ground truth are included.
    """
    # Flatten over binary criteria, skipping errored votes, then build label
    # (accuracy/kappa) and MET-vs-rest (precision/recall/f1) representations.
    pred_flat: list[CriterionVerdict] = []
    true_flat: list[CriterionVerdict] = []
    for item_idx, pred_v in enumerate(judge_verdicts):
        if item_idx >= len(true_verdicts):
            break
        true_v = true_verdicts[item_idx]
        err_v = judge_errors[item_idx] if item_idx < len(judge_errors) else [None] * len(pred_v)
        for c in range(len(pred_v)):
            if c >= len(criterion_types) or criterion_types[c] != "binary":
                continue
            if c < len(err_v) and err_v[c] is not None:
                continue
            if c >= len(true_v):
                continue
            true_val = true_v[c]
            if not isinstance(true_val, CriterionVerdict):
                continue
            pred_flat.append(pred_v[c])
            true_flat.append(true_val)

    label_pred, label_true, met_pred, met_true = prepare_binary_metric_inputs(
        pred_flat, true_flat, cannot_assess
    )

    # Criterion-level metrics
    if label_pred:
        criterion_accuracy = accuracy_score(label_true, label_pred)
        criterion_precision = precision_score(met_true, met_pred, zero_division=0)
        criterion_recall = recall_score(met_true, met_pred, zero_division=0)
        criterion_f1 = f1_score(met_true, met_pred, zero_division=0)
        try:
            kappa = cohen_kappa_score(label_true, label_pred)
        except Exception:
            kappa = 0.0
    else:
        criterion_accuracy = 0.0
        criterion_precision = 0.0
        criterion_recall = 0.0
        criterion_f1 = 0.0
        kappa = 0.0

    # Score-level metrics
    score_rmse = float(np.sqrt(mean_squared_error(true_scores, judge_scores)))
    score_mae = float(mean_absolute_error(true_scores, judge_scores))

    score_spearman = _compute_correlation(judge_scores, true_scores, "spearman")
    score_kendall = _compute_correlation(judge_scores, true_scores, "kendall")
    score_pearson = _compute_correlation(judge_scores, true_scores, "pearson")

    # Bias
    bias = systematic_bias(judge_scores, true_scores)

    return JudgeMetrics(
        judge_id=judge_id,
        criterion_accuracy=criterion_accuracy,
        criterion_precision=criterion_precision,
        criterion_recall=criterion_recall,
        criterion_f1=criterion_f1,
        mean_kappa=kappa,
        score_rmse=score_rmse,
        score_mae=score_mae,
        score_spearman=score_spearman,
        score_kendall=score_kendall,
        score_pearson=score_pearson,
        bias=bias,
    )


def compute_metrics(
    eval_result: EvalResult,
    dataset: RubricDataset,
    *,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    per_judge: bool = False,
    cannot_assess: CannotAssessMode = "exclude",
    na_mode: NAMode = "exclude",
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> MetricsResult:
    """Compute comprehensive evaluation metrics.

    This is the main entry point for computing metrics from an evaluation run.
    It compares predicted verdicts and scores against ground truth from the dataset.
    Supports binary, ordinal, and nominal (multi-choice) criteria.

    Args:
        eval_result: The evaluation result from EvalRunner.
        dataset: The dataset with ground truth labels.
        bootstrap: If True, compute bootstrap confidence intervals (expensive).
        n_bootstrap: Number of bootstrap samples if bootstrap=True.
        per_judge: If True and ensemble, compute per-judge metrics.
        cannot_assess: How to handle CANNOT_ASSESS verdicts (binary criteria):
            - "exclude": Skip pairs where either is CANNOT_ASSESS (default)
            - "as_unmet": Treat CANNOT_ASSESS as UNMET
            - "as_category": Keep CANNOT_ASSESS as a distinct third class. Accuracy and
              Cohen's kappa are then computed over three classes (a CANNOT_ASSESS
              prediction matching a CANNOT_ASSESS ground truth counts as correct);
              precision/recall/f1 remain MET-vs-rest.
        na_mode: How to handle NA options (multi-choice criteria). Mirrors
            ``cannot_assess`` for binary — NA on multi-choice is the structural
            analog of CANNOT_ASSESS on binary:

            - "exclude": Skip pairs where either is NA (default).
            - "as_unmet": Remap NA to the score-minimizing non-NA option,
              weight-sign aware (lowest ``value`` for non-negative weight,
              highest ``value`` for negative weight). Shares
              ``Criterion.worst_scored_option()`` with the grader's
              ``unknown``-error worst-case path so the layers cannot drift.
            - "as_category": Keep NA as a distinct categorical column.
              **Refused for ordinal criteria with an NA option** (raises
              ``ValueError``): NA has no ordinal position, so quadratic
              weighted Cohen's kappa would assign NA a geometrically
              meaningless distance.
        confidence_level: Confidence level for bootstrap CIs (default 0.95).
        seed: Random seed for bootstrap reproducibility.

    Returns:
        MetricsResult with comprehensive metrics and optional per-judge breakdown.

    Raises:
        ValueError: If no common items between eval_result and dataset.

    Example:
        >>> result = await evaluate(dataset, grader)
        >>> metrics = result.compute_metrics(dataset)
        >>> print(metrics.summary())
        >>> df = metrics.to_dataframe()
    """
    result_warnings: list[str] = []

    # Build map of item_idx -> ItemResult
    eval_map = {ir.item_idx: ir for ir in eval_result.item_results}

    # Check for missing/extra items
    dataset_indices = set(range(len(dataset)))
    eval_indices = set(eval_map.keys())

    missing = dataset_indices - eval_indices
    if missing:
        result_warnings.append(f"{len(missing)} items from dataset not found in eval_result")

    extra = eval_indices - dataset_indices
    if extra:
        result_warnings.append(f"{len(extra)} items in eval_result not in dataset")

    # Use intersection
    common_indices = sorted(dataset_indices & eval_indices)

    if not common_indices:
        raise ValueError("No common items between eval_result and dataset")

    # Validate rubric homogeneity for metrics computation
    # If using per-item rubrics, all must have the same structure
    if dataset.rubric is not None:
        reference_rubric = dataset.rubric
    else:
        # Get rubric from first item
        reference_rubric = dataset.get_item_rubric(common_indices[0])

    reference_n_criteria = len(reference_rubric.rubric)

    for idx in common_indices:
        item_rubric = dataset.get_item_rubric(idx)
        if len(item_rubric.rubric) != reference_n_criteria:
            raise ValueError(
                f"Cannot compute metrics: items have different rubric structures. "
                f"Item {idx} has {len(item_rubric.rubric)} criteria but "
                f"expected {reference_n_criteria}. "
                f"Metrics require homogeneous rubric structures across all items."
            )

    # Use the reference rubric for classification
    criteria = list(reference_rubric.rubric)
    criterion_types = classify_criteria(criteria)
    n_criteria = len(criteria)

    # Count criteria by type
    n_binary = sum(1 for ct in criterion_types if ct == "binary")
    n_ordinal = sum(1 for ct in criterion_types if ct == "ordinal")
    n_nominal = sum(1 for ct in criterion_types if ct == "nominal")

    # Per-criterion data storage
    # For binary: list[CriterionVerdict]
    # For multi-choice: list[int] (option indices)
    per_criterion_pred: list[list[CriterionVerdict | int]] = [[] for _ in range(n_criteria)]
    per_criterion_true: list[list[CriterionVerdict | int]] = [[] for _ in range(n_criteria)]

    # Overall scores
    all_pred_scores: list[float] = []
    all_true_scores: list[float] = []

    # For ensemble: per-judge data (binary only for now)
    judge_scores: dict[str, list[float]] = {}
    judge_verdicts: dict[str, list[list[CriterionVerdict]]] = {}
    judge_errors: dict[str, list[list[str | None]]] = {}
    is_ensemble = False

    # Per-item ground-truth verdicts (all criteria) aligned 1:1 with each item that
    # contributes ensemble per-judge data; used for the per-judge metrics fix.
    per_item_true: list[list[CriterionVerdict | int]] = []

    # Fleiss' kappa ratings rows, per criterion (only ensemble reports produce rows).
    fleiss_rows: dict[int, list[list[int]]] = {c: [] for c in range(n_criteria)}

    # Krippendorff's alpha: per criterion, one dict per ensemble item mapping
    # judge_id -> numeric cell value (np.nan = missing). Rows (judges) and columns
    # (items) are assembled after the loop using the final judge id set.
    alpha_cells: dict[int, list[dict[str, float]]] = {c: [] for c in range(n_criteria)}

    items_with_ground_truth = 0

    # NA tracking for multi-choice
    total_na_true = 0
    total_na_pred = 0
    total_na_fp = 0
    total_na_fn = 0

    for idx in common_indices:
        item = dataset.items[idx]
        item_result = eval_map[idx]
        report = item_result.report

        if item.ground_truth is None:
            result_warnings.append(f"Item {idx} has no ground truth, skipping")
            continue

        if item_result.error is not None:
            continue

        items_with_ground_truth += 1

        # Extract predictions using type-aware extraction
        pred_all = extract_all_verdicts_from_report(report, criteria)

        # Resolve ground truth (string labels → indices for multi-choice)
        try:
            true_all = resolve_ground_truth(list(item.ground_truth), criteria)
        except ValueError as e:
            result_warnings.append(f"Item {idx}: {e}")
            continue

        # Store per-criterion data
        for c_idx in range(n_criteria):
            pred_val = pred_all[c_idx]
            true_val = true_all[c_idx]

            # Handle None predictions (failed extraction)
            if pred_val is None:
                if criterion_types[c_idx] == "binary":
                    pred_val = CriterionVerdict.UNMET
                else:
                    pred_val = 0  # Default to first option

            per_criterion_pred[c_idx].append(pred_val)
            per_criterion_true[c_idx].append(true_val)

        # Compute scores
        pred_score = report.score if not report.error else 0.0
        # For true score, need to pass the original ground truth format
        # compute_weighted_score expects CriterionVerdict for binary, str for multi-choice
        true_score_verdicts = []
        for c_idx in range(n_criteria):
            if criterion_types[c_idx] == "binary":
                true_score_verdicts.append(true_all[c_idx])
            else:
                # For multi-choice, pass the option label (string)
                criterion = criteria[c_idx]
                opt_idx = true_all[c_idx]
                if isinstance(opt_idx, int) and 0 <= opt_idx < len(criterion.options):
                    true_score_verdicts.append(criterion.options[opt_idx].label)
                else:
                    # Default to first option if index is invalid
                    true_score_verdicts.append(criterion.options[0].label)

        true_score = dataset.compute_weighted_score(true_score_verdicts)

        all_pred_scores.append(pred_score)
        all_true_scores.append(true_score)

        # Check if ensemble and collect per-judge data
        if hasattr(report, "judge_scores") and report.judge_scores:
            is_ensemble = True
            for jid, score in report.judge_scores.items():
                if jid not in judge_scores:
                    judge_scores[jid] = []
                    judge_verdicts[jid] = []
                    judge_errors[jid] = []
                judge_scores[jid].append(score)

            # Align ground truth (all criteria) once per ensemble item.
            per_item_true.append(list(true_all))

            # Extract per-judge verdicts + errors from EnsembleCriterionReport.votes.
            if hasattr(report, "report") and report.report:
                for jid in judge_scores.keys():
                    judge_v: list[CriterionVerdict] = []
                    judge_e: list[str | None] = []
                    for cr in report.report:
                        if hasattr(cr, "votes"):
                            for vote in cr.votes:
                                if vote.judge_id == jid:
                                    judge_v.append(vote.verdict)
                                    judge_e.append(vote.error)
                                    break
                            else:
                                judge_v.append(CriterionVerdict.UNMET)
                                judge_e.append(None)
                        else:
                            judge_v.append(CriterionVerdict.UNMET)
                            judge_e.append(None)
                    if jid in judge_verdicts:
                        judge_verdicts[jid].append(judge_v)
                        judge_errors[jid].append(judge_e)

            # Inter-judge agreement collection (binary + multi-choice) from ensemble votes.
            if hasattr(report, "report") and report.report:
                n_judges = len(report.judge_scores)
                for c_idx in range(n_criteria):
                    cr = report.report[c_idx]
                    c_type = criterion_types[c_idx]
                    # Fleiss: complete-case ratings row (uniform rater count).
                    row = _build_fleiss_row(
                        cr,
                        criteria[c_idx],
                        c_type,
                        cannot_assess,
                        n_judges,
                    )
                    if row is not None:
                        fleiss_rows[c_idx].append(row)
                    # Krippendorff alpha: per-judge cells (missing handled natively).
                    votes = (
                        cr.votes if c_type == "binary" else getattr(cr, "multi_choice_votes", [])
                    )
                    cell_map: dict[str, float] = {
                        v.judge_id: _build_alpha_cell(v, c_type, cannot_assess)
                        for v in (votes or [])
                    }
                    alpha_cells[c_idx].append(cell_map)

    n_items = items_with_ground_truth

    if n_items == 0:
        raise ValueError("No valid items with ground truth found")

    # Compute per-criterion metrics by type
    per_criterion: list[CriterionMetricsUnion] = []
    criterion_kappas: list[float] = []

    # Inter-judge agreement (Krippendorff's alpha + Fleiss' kappa) is only meaningful
    # with an ensemble of >=2 judges (>=2 items is enforced downstream).
    eligible = is_ensemble and len(judge_scores) >= 2

    # Precompute Krippendorff's alpha per criterion from the collected reliability cells.
    # Rows = judges (fixed judge-id order), columns = items; np.nan marks missing ratings.
    # Alpha uses ALL items (missing handled natively) — no complete-case dropping.
    judge_ids = list(judge_scores.keys())
    krippendorff_alphas: dict[int, float | None] = dict.fromkeys(range(n_criteria))
    if eligible:
        for c_idx in range(n_criteria):
            level: Literal["nominal", "ordinal"] = (
                "ordinal" if criterion_types[c_idx] == "ordinal" else "nominal"
            )
            cell_maps = alpha_cells.get(c_idx, [])
            reliability_data = [
                [cm.get(jid, float("nan")) for cm in cell_maps] for jid in judge_ids
            ]
            krippendorff_alphas[c_idx] = _compute_krippendorff_alpha(reliability_data, level)

    # For binary-only aggregate metrics:
    # label_*_flat feed accuracy/kappa (may be 3-class under "as_category");
    # met_*_flat feed precision/recall/f1 (MET one-vs-rest).
    label_pred_flat: list[str] = []
    label_true_flat: list[str] = []
    met_pred_flat: list[int] = []
    met_true_flat: list[int] = []

    for c_idx in range(n_criteria):
        criterion = criteria[c_idx]
        c_type = criterion_types[c_idx]
        pred_data = per_criterion_pred[c_idx]
        true_data = per_criterion_true[c_idx]

        if c_type == "binary":
            # Binary criterion metrics
            pred_verdicts = [v for v in pred_data if isinstance(v, CriterionVerdict)]
            true_verdicts = [v for v in true_data if isinstance(v, CriterionVerdict)]

            # Handle CANNOT_ASSESS centrally and build label + MET-vs-rest reps.
            label_pred, label_true, met_pred, met_true = prepare_binary_metric_inputs(
                pred_verdicts, true_verdicts, cannot_assess
            )

            # Add to aggregate
            label_pred_flat.extend(label_pred)
            label_true_flat.extend(label_true)
            met_pred_flat.extend(met_pred)
            met_true_flat.extend(met_true)

            name = criterion.name or f"Criterion {c_idx + 1}"

            fleiss_kappa = _compute_fleiss_kappa(fleiss_rows.get(c_idx)) if eligible else None
            krippendorff_alpha = krippendorff_alphas.get(c_idx) if eligible else None

            if not label_pred:
                per_criterion.append(
                    CriterionMetrics(
                        name=name,
                        index=c_idx,
                        n_samples=0,
                        accuracy=0.0,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        kappa=0.0,
                        kappa_interpretation="undefined",
                        krippendorff_alpha=krippendorff_alpha,
                        fleiss_kappa=fleiss_kappa,
                        support_true=0,
                        support_pred=0,
                    )
                )
                continue

            c_acc = accuracy_score(label_true, label_pred)
            c_prec = precision_score(met_true, met_pred, zero_division=0)
            c_rec = recall_score(met_true, met_pred, zero_division=0)
            c_f1 = f1_score(met_true, met_pred, zero_division=0)

            try:
                c_kappa = cohen_kappa_score(label_true, label_pred)
            except Exception:
                c_kappa = 0.0

            criterion_kappas.append(c_kappa)

            per_criterion.append(
                CriterionMetrics(
                    name=name,
                    index=c_idx,
                    n_samples=len(label_pred),
                    accuracy=float(c_acc),
                    precision=float(c_prec),
                    recall=float(c_rec),
                    f1=float(c_f1),
                    kappa=float(c_kappa),
                    kappa_interpretation=_interpret_kappa(c_kappa),
                    krippendorff_alpha=krippendorff_alpha,
                    fleiss_kappa=fleiss_kappa,
                    support_true=sum(met_true),
                    support_pred=sum(met_pred),
                )
            )

        elif c_type == "ordinal":
            # Ordinal multi-choice criterion metrics
            pred_indices = [v for v in pred_data if isinstance(v, int)]
            true_indices = [v for v in true_data if isinstance(v, int)]

            # Filter NA options. na_agree is unused here (the NAStats block below
            # computes kappa on the {NA, not-NA} dichotomy from per-criterion data).
            pred_filtered, true_filtered, _na_agree, na_fp, na_fn = filter_na_multi_choice(
                pred_indices, true_indices, criterion, mode=na_mode
            )

            # Track NA stats (FP/FN feed the diagnostic counts on NAStats)
            total_na_fp += na_fp
            total_na_fn += na_fn

            metrics = _compute_ordinal_criterion_metrics(
                pred_filtered,
                true_filtered,
                criterion,
                c_idx,
                fleiss_matrix=(fleiss_rows.get(c_idx) if eligible else None),
                krippendorff_alpha=(krippendorff_alphas.get(c_idx) if eligible else None),
            )
            per_criterion.append(metrics)

            # Use weighted kappa for ordinal in mean calculation
            criterion_kappas.append(metrics.weighted_kappa)

        else:  # nominal
            # Nominal multi-choice criterion metrics
            pred_indices = [v for v in pred_data if isinstance(v, int)]
            true_indices = [v for v in true_data if isinstance(v, int)]

            # Filter NA options. na_agree is unused here (the NAStats block below
            # computes kappa on the {NA, not-NA} dichotomy from per-criterion data).
            pred_filtered, true_filtered, _na_agree, na_fp, na_fn = filter_na_multi_choice(
                pred_indices, true_indices, criterion, mode=na_mode
            )

            # Track NA stats (FP/FN feed the diagnostic counts on NAStats)
            total_na_fp += na_fp
            total_na_fn += na_fn

            metrics = _compute_nominal_criterion_metrics(
                pred_filtered,
                true_filtered,
                criterion,
                c_idx,
                fleiss_matrix=(fleiss_rows.get(c_idx) if eligible else None),
                krippendorff_alpha=(krippendorff_alphas.get(c_idx) if eligible else None),
            )
            per_criterion.append(metrics)

            # Use unweighted kappa for nominal
            criterion_kappas.append(metrics.kappa)

    # Aggregate metrics
    mean_kappa = sum(criterion_kappas) / len(criterion_kappas) if criterion_kappas else 0.0

    # Binary-only aggregate metrics (precision/recall/f1 only make sense for binary)
    if label_pred_flat:
        criterion_accuracy = accuracy_score(label_true_flat, label_pred_flat)
        criterion_precision = precision_score(met_true_flat, met_pred_flat, zero_division=0)
        criterion_recall = recall_score(met_true_flat, met_pred_flat, zero_division=0)
        criterion_f1 = f1_score(met_true_flat, met_pred_flat, zero_division=0)
    else:
        # No binary criteria - compute accuracy across all multi-choice
        # For multi-choice, accuracy is exact match
        all_correct = 0
        all_total = 0
        for c_idx in range(n_criteria):
            c_type = criterion_types[c_idx]
            if c_type != "binary":
                pred_data = per_criterion_pred[c_idx]
                true_data = per_criterion_true[c_idx]
                for p, t in zip(pred_data, true_data):
                    if isinstance(p, int) and isinstance(t, int):
                        all_total += 1
                        if p == t:
                            all_correct += 1

        criterion_accuracy = all_correct / all_total if all_total > 0 else 0.0
        # Precision/recall/f1 not meaningful for pure multi-choice rubrics
        criterion_precision = 0.0
        criterion_recall = 0.0
        criterion_f1 = 0.0

    # Score-level metrics
    score_rmse = float(np.sqrt(mean_squared_error(all_true_scores, all_pred_scores)))
    score_mae = float(mean_absolute_error(all_true_scores, all_pred_scores))

    score_spearman = _compute_correlation(all_pred_scores, all_true_scores, "spearman")
    score_kendall = _compute_correlation(all_pred_scores, all_true_scores, "kendall")
    score_pearson = _compute_correlation(all_pred_scores, all_true_scores, "pearson")

    # Bias analysis
    bias = systematic_bias(all_pred_scores, all_true_scores)

    # Bootstrap CIs (optional) - uses binary metrics for backwards compat
    bootstrap_results = None
    if bootstrap and label_pred_flat:
        bootstrap_results = _compute_bootstrap_ci(
            label_true_flat,
            label_pred_flat,
            all_true_scores,
            all_pred_scores,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )

    # Per-judge metrics (optional, for ensemble) - binary only for now
    per_judge_metrics = None
    if per_judge and is_ensemble and judge_scores:
        per_judge_metrics = {}
        for jid in judge_scores.keys():
            jv = judge_verdicts.get(jid, [])
            if not jv:
                continue

            per_judge_metrics[jid] = _compute_judge_metrics(
                judge_id=jid,
                judge_scores=judge_scores[jid],
                true_scores=all_true_scores,
                judge_verdicts=jv,
                judge_errors=judge_errors.get(jid, []),
                true_verdicts=per_item_true,
                criterion_types=list(criterion_types),
                cannot_assess=cannot_assess,
            )

    # NA stats (for multi-choice criteria)
    # Cohen's kappa on the {NA, not-NA} dichotomy across all multi-choice
    # criteria that define an NA option, paired pred-vs-truth. Reuses the
    # same chance-corrected statistic as the rest of the framework's
    # prediction-vs-ground-truth agreement metrics (binary `kappa`, ordinal
    # `weighted_kappa`, nominal `kappa`). Returns None when undefined.
    na_stats = None
    if n_ordinal > 0 or n_nominal > 0:
        na_pred_bool: list[bool] = []
        na_true_bool: list[bool] = []
        for c_idx in range(n_criteria):
            if criterion_types[c_idx] == "binary":
                continue
            criterion = criteria[c_idx]
            na_indices = {i for i, opt in enumerate(criterion.options) if opt.na}
            if not na_indices:
                continue
            for p, t in zip(per_criterion_pred[c_idx], per_criterion_true[c_idx]):
                if isinstance(p, int) and isinstance(t, int):
                    p_is_na = p in na_indices
                    t_is_na = t in na_indices
                    na_pred_bool.append(p_is_na)
                    na_true_bool.append(t_is_na)
                    if p_is_na:
                        total_na_pred += 1
                    if t_is_na:
                        total_na_true += 1

        na_kappa: float | None = None
        if na_pred_bool:
            try:
                k = float(cohen_kappa_score(na_true_bool, na_pred_bool))
                na_kappa = None if math.isnan(k) else k
            except Exception:
                na_kappa = None
        na_kappa_interpretation = (
            KappaResult.interpret_kappa(na_kappa) if na_kappa is not None else None
        )

        na_stats = NAStats(
            na_count_true=total_na_true,
            na_count_pred=total_na_pred,
            na_kappa=na_kappa,
            na_kappa_interpretation=na_kappa_interpretation,
            na_false_positive=total_na_fp,
            na_false_negative=total_na_fn,
        )

    return MetricsResult(
        criterion_accuracy=float(criterion_accuracy),
        criterion_precision=float(criterion_precision),
        criterion_recall=float(criterion_recall),
        criterion_f1=float(criterion_f1),
        mean_kappa=float(mean_kappa),
        per_criterion=per_criterion,
        score_rmse=score_rmse,
        score_mae=score_mae,
        score_spearman=score_spearman,
        score_kendall=score_kendall,
        score_pearson=score_pearson,
        bias=bias,
        bootstrap=bootstrap_results,
        per_judge=per_judge_metrics,
        n_items=n_items,
        n_criteria=n_criteria,
        n_binary_criteria=n_binary,
        n_ordinal_criteria=n_ordinal,
        n_nominal_criteria=n_nominal,
        na_stats=na_stats,
        warnings=result_warnings,
    )
