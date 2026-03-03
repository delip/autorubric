"""Core compute_metrics implementation.

This module provides the main compute_metrics function that computes
comprehensive evaluation metrics from an EvalResult and RubricDataset.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

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
    extract_verdicts_from_report,
    filter_na_multi_choice,
    get_option_value,
    resolve_ground_truth,
)
from ._types import (
    BiasResult,
    BootstrapResults,
    ConfidenceInterval,
    CorrelationResult,
    CriterionMetrics,
    CriterionMetricsUnion,
    JudgeMetrics,
    MetricsResult,
    NAStats,
    NominalCriterionMetrics,
    OptionMetrics,
    OrdinalCriterionMetrics,
)
from .distribution import systematic_bias

# Try to import Fleiss' kappa from statsmodels (optional dependency)
try:
    from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss_kappa

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    _fleiss_kappa = None

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

    adjacent_correct = sum(
        1 for p, t in zip(pred_indices, true_indices) if abs(p - t) <= 1
    )
    return adjacent_correct / len(pred_indices)


def _compute_fleiss_kappa(
    ratings_matrix: list[list[int]],
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
        return float(_fleiss_kappa(matrix))
    except Exception:
        return None


def _compute_ordinal_criterion_metrics(
    pred_indices: list[int],
    true_indices: list[int],
    criterion: Criterion,
    index: int,
    fleiss_matrix: list[list[int]] | None = None,
) -> OrdinalCriterionMetrics:
    """Compute metrics for an ordinal multi-choice criterion.

    Args:
        pred_indices: Predicted option indices.
        true_indices: Ground truth option indices.
        criterion: The ordinal criterion.
        index: Index of this criterion in the rubric.
        fleiss_matrix: Optional ratings matrix for Fleiss' kappa (ensemble).

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
        weighted_kappa = cohen_kappa_score(
            true_indices, pred_indices, weights="quadratic"
        )
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
) -> NominalCriterionMetrics:
    """Compute metrics for a nominal multi-choice criterion.

    Args:
        pred_indices: Predicted option indices.
        true_indices: Ground truth option indices.
        criterion: The nominal criterion.
        index: Index of this criterion in the rubric.
        fleiss_matrix: Optional ratings matrix for Fleiss' kappa (ensemble).

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
        fleiss_kappa=fleiss_kappa,
        per_option=per_option,
        confusion_matrix=conf_matrix,
        option_labels=option_labels,
    )


def _verdict_to_binary(verdict: CriterionVerdict) -> int:
    """Convert a single verdict to binary (MET=1, else=0)."""
    return 1 if verdict == CriterionVerdict.MET else 0


def _compute_correlation(
    x: list[float], y: list[float], method: str
) -> CorrelationResult:
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
    y_true: list[int],
    y_pred: list[int],
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
    true_verdicts: list[list[CriterionVerdict]],
    cannot_assess: Literal["exclude", "as_unmet"],
) -> JudgeMetrics:
    """Compute metrics for a single judge."""
    # Flatten verdicts for criterion-level metrics
    pred_flat = []
    true_flat = []

    for pred_v, true_v in zip(judge_verdicts, true_verdicts):
        for p, t in zip(pred_v, true_v):
            if cannot_assess == "exclude":
                if p == CriterionVerdict.CANNOT_ASSESS or t == CriterionVerdict.CANNOT_ASSESS:
                    continue
            pred_flat.append(_verdict_to_binary(p))
            true_flat.append(_verdict_to_binary(t))

    # Criterion-level metrics
    if pred_flat:
        criterion_accuracy = accuracy_score(true_flat, pred_flat)
        criterion_precision = precision_score(true_flat, pred_flat, zero_division=0)
        criterion_recall = recall_score(true_flat, pred_flat, zero_division=0)
        criterion_f1 = f1_score(true_flat, pred_flat, zero_division=0)
        try:
            kappa = cohen_kappa_score(true_flat, pred_flat)
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
    eval_result: "EvalResult",
    dataset: "RubricDataset",
    *,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    per_judge: bool = False,
    cannot_assess: Literal["exclude", "as_unmet"] = "exclude",
    na_mode: Literal["exclude", "as_worst"] = "exclude",
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
        na_mode: How to handle NA options (multi-choice criteria):
            - "exclude": Skip pairs where either is NA (default)
            - "as_worst": Keep NA in metrics (no special treatment)
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
        result_warnings.append(
            f"{len(missing)} items from dataset not found in eval_result"
        )

    extra = eval_indices - dataset_indices
    if extra:
        result_warnings.append(
            f"{len(extra)} items in eval_result not in dataset"
        )

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
    is_ensemble = False

    items_with_ground_truth = 0

    # NA tracking for multi-choice
    total_na_true = 0
    total_na_pred = 0
    total_na_agreement = 0
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
                judge_scores[jid].append(score)

            # Extract per-judge verdicts from EnsembleCriterionReport.votes (binary only)
            if hasattr(report, "report") and report.report:
                for jid in judge_scores.keys():
                    judge_v = []
                    for cr in report.report:
                        if hasattr(cr, "votes"):
                            for vote in cr.votes:
                                if vote.judge_id == jid:
                                    judge_v.append(vote.verdict)
                                    break
                            else:
                                judge_v.append(CriterionVerdict.UNMET)
                        else:
                            judge_v.append(CriterionVerdict.UNMET)
                    if jid in judge_verdicts:
                        judge_verdicts[jid].append(judge_v)

    n_items = items_with_ground_truth

    if n_items == 0:
        raise ValueError("No valid items with ground truth found")

    # Compute per-criterion metrics by type
    per_criterion: list[CriterionMetricsUnion] = []
    criterion_kappas: list[float] = []

    # For binary-only aggregate metrics
    binary_pred_flat: list[int] = []
    binary_true_flat: list[int] = []

    for c_idx in range(n_criteria):
        criterion = criteria[c_idx]
        c_type = criterion_types[c_idx]
        pred_data = per_criterion_pred[c_idx]
        true_data = per_criterion_true[c_idx]

        if c_type == "binary":
            # Binary criterion metrics
            pred_verdicts = [v for v in pred_data if isinstance(v, CriterionVerdict)]
            true_verdicts = [v for v in true_data if isinstance(v, CriterionVerdict)]

            # Filter CANNOT_ASSESS
            pred_filtered = []
            true_filtered = []
            for p, t in zip(pred_verdicts, true_verdicts):
                if cannot_assess == "exclude":
                    if p == CriterionVerdict.CANNOT_ASSESS or t == CriterionVerdict.CANNOT_ASSESS:
                        continue
                pred_filtered.append(_verdict_to_binary(p))
                true_filtered.append(_verdict_to_binary(t))

            # Add to aggregate
            binary_pred_flat.extend(pred_filtered)
            binary_true_flat.extend(true_filtered)

            name = criterion.name or f"Criterion {c_idx + 1}"

            if not pred_filtered:
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
                        support_true=0,
                        support_pred=0,
                    )
                )
                continue

            c_acc = accuracy_score(true_filtered, pred_filtered)
            c_prec = precision_score(true_filtered, pred_filtered, zero_division=0)
            c_rec = recall_score(true_filtered, pred_filtered, zero_division=0)
            c_f1 = f1_score(true_filtered, pred_filtered, zero_division=0)

            try:
                c_kappa = cohen_kappa_score(true_filtered, pred_filtered)
            except Exception:
                c_kappa = 0.0

            criterion_kappas.append(c_kappa)

            per_criterion.append(
                CriterionMetrics(
                    name=name,
                    index=c_idx,
                    n_samples=len(pred_filtered),
                    accuracy=float(c_acc),
                    precision=float(c_prec),
                    recall=float(c_rec),
                    f1=float(c_f1),
                    kappa=float(c_kappa),
                    kappa_interpretation=_interpret_kappa(c_kappa),
                    support_true=sum(true_filtered),
                    support_pred=sum(pred_filtered),
                )
            )

        elif c_type == "ordinal":
            # Ordinal multi-choice criterion metrics
            pred_indices = [v for v in pred_data if isinstance(v, int)]
            true_indices = [v for v in true_data if isinstance(v, int)]

            # Filter NA options
            pred_filtered, true_filtered, na_agree, na_fp, na_fn = filter_na_multi_choice(
                pred_indices, true_indices, criterion, mode=na_mode
            )

            # Track NA stats
            total_na_agreement += na_agree
            total_na_fp += na_fp
            total_na_fn += na_fn

            metrics = _compute_ordinal_criterion_metrics(
                pred_filtered, true_filtered, criterion, c_idx
            )
            per_criterion.append(metrics)

            # Use weighted kappa for ordinal in mean calculation
            criterion_kappas.append(metrics.weighted_kappa)

        else:  # nominal
            # Nominal multi-choice criterion metrics
            pred_indices = [v for v in pred_data if isinstance(v, int)]
            true_indices = [v for v in true_data if isinstance(v, int)]

            # Filter NA options
            pred_filtered, true_filtered, na_agree, na_fp, na_fn = filter_na_multi_choice(
                pred_indices, true_indices, criterion, mode=na_mode
            )

            # Track NA stats
            total_na_agreement += na_agree
            total_na_fp += na_fp
            total_na_fn += na_fn

            metrics = _compute_nominal_criterion_metrics(
                pred_filtered, true_filtered, criterion, c_idx
            )
            per_criterion.append(metrics)

            # Use unweighted kappa for nominal
            criterion_kappas.append(metrics.kappa)

    # Aggregate metrics
    mean_kappa = (
        sum(criterion_kappas) / len(criterion_kappas) if criterion_kappas else 0.0
    )

    # Binary-only aggregate metrics (precision/recall/f1 only make sense for binary)
    if binary_pred_flat:
        criterion_accuracy = accuracy_score(binary_true_flat, binary_pred_flat)
        criterion_precision = precision_score(binary_true_flat, binary_pred_flat, zero_division=0)
        criterion_recall = recall_score(binary_true_flat, binary_pred_flat, zero_division=0)
        criterion_f1 = f1_score(binary_true_flat, binary_pred_flat, zero_division=0)
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
    if bootstrap and binary_pred_flat:
        bootstrap_results = _compute_bootstrap_ci(
            binary_true_flat,
            binary_pred_flat,
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

            # Extract binary verdicts for this judge
            binary_true_verdicts = []
            for true_item in per_criterion_true:
                binary_true_verdicts.append(
                    [v for v in true_item if isinstance(v, CriterionVerdict)]
                )

            per_judge_metrics[jid] = _compute_judge_metrics(
                judge_id=jid,
                judge_scores=judge_scores[jid],
                true_scores=all_true_scores,
                judge_verdicts=jv,
                true_verdicts=binary_true_verdicts[0] if binary_true_verdicts else [],
                cannot_assess=cannot_assess,
            )

    # NA stats (for multi-choice criteria)
    na_stats = None
    if n_ordinal > 0 or n_nominal > 0:
        # Calculate total NA counts
        for c_idx in range(n_criteria):
            if criterion_types[c_idx] != "binary":
                criterion = criteria[c_idx]
                na_indices = {i for i, opt in enumerate(criterion.options) if opt.na}
                if na_indices:
                    pred_data = per_criterion_pred[c_idx]
                    true_data = per_criterion_true[c_idx]
                    for p in pred_data:
                        if isinstance(p, int) and p in na_indices:
                            total_na_pred += 1
                    for t in true_data:
                        if isinstance(t, int) and t in na_indices:
                            total_na_true += 1

        total_na = total_na_true + total_na_pred
        na_stats = NAStats(
            na_count_true=total_na_true,
            na_count_pred=total_na_pred,
            na_agreement=total_na_agreement / max(1, total_na) if total_na > 0 else 0.0,
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
