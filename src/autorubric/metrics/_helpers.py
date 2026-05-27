"""Helper functions for extracting data from EvaluationReport.

These functions bridge the gap between autorubric's report types
and the metric computation functions.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from ..types import (
    Criterion,
    CriterionVerdict,
    EnsembleEvaluationReport,
    EvaluationReport,
)
from ._types import CannotAssessMode, CriterionType

if TYPE_CHECKING:
    pass


def _extract_raw_verdicts(cr) -> tuple:
    """Extract raw verdict and multi-choice verdict from a criterion report.

    Handles both CriterionReport (verdict, multi_choice_verdict) and
    EnsembleCriterionReport (final_verdict, final_multi_choice_verdict).

    Returns:
        (binary_verdict, multi_choice_verdict) — either may be None.
    """
    if hasattr(cr, "final_verdict"):
        return cr.final_verdict, getattr(cr, "final_multi_choice_verdict", None)
    if hasattr(cr, "verdict"):
        return cr.verdict, getattr(cr, "multi_choice_verdict", None)
    return None, None


def extract_verdicts_from_report(
    report: EvaluationReport | EnsembleEvaluationReport,
    num_criteria: int,
) -> list[CriterionVerdict]:
    """Extract verdicts from an EvaluationReport.

    Args:
        report: The evaluation report.
        num_criteria: Expected number of criteria.

    Returns:
        List of CriterionVerdict values.
    """
    if report.report is None:
        return [CriterionVerdict.UNMET] * num_criteria

    verdicts = []
    for cr in report.report:
        if isinstance(cr, dict):
            verdicts.append(cr.get("verdict", CriterionVerdict.UNMET))
        else:
            binary_v, _ = _extract_raw_verdicts(cr)
            verdicts.append(binary_v if binary_v is not None else CriterionVerdict.UNMET)

    return verdicts


def filter_cannot_assess(
    pred_verdicts: list[CriterionVerdict],
    true_verdicts: list[CriterionVerdict],
    mode: CannotAssessMode = "exclude",
) -> tuple[list[CriterionVerdict], list[CriterionVerdict]]:
    """Filter or transform CANNOT_ASSESS verdicts based on mode.

    Args:
        pred_verdicts: Predicted verdicts.
        true_verdicts: Ground truth verdicts.
        mode: How to handle CANNOT_ASSESS:
            - "exclude": Remove pairs where either is CANNOT_ASSESS
            - "as_unmet": Convert CANNOT_ASSESS to UNMET
            - "as_category": Keep CANNOT_ASSESS as-is (3-class)

    Returns:
        Tuple of (filtered_pred, filtered_true).
    """
    CA = CriterionVerdict.CANNOT_ASSESS

    if mode == "exclude":
        filtered_pred = []
        filtered_true = []
        for p, t in zip(pred_verdicts, true_verdicts):
            if p != CA and t != CA:
                filtered_pred.append(p)
                filtered_true.append(t)
        return filtered_pred, filtered_true

    elif mode == "as_unmet":
        return (
            [CriterionVerdict.UNMET if v == CA else v for v in pred_verdicts],
            [CriterionVerdict.UNMET if v == CA else v for v in true_verdicts],
        )

    else:  # as_category
        return list(pred_verdicts), list(true_verdicts)


def verdict_to_binary(verdicts: Sequence[CriterionVerdict]) -> list[int]:
    """Convert verdicts to binary (MET=1, UNMET/CANNOT_ASSESS=0).

    Args:
        verdicts: List of CriterionVerdict values.

    Returns:
        List of 0/1 values.
    """
    return [1 if v == CriterionVerdict.MET else 0 for v in verdicts]


def prepare_binary_metric_inputs(
    pred_verdicts: list[CriterionVerdict],
    true_verdicts: list[CriterionVerdict],
    mode: CannotAssessMode = "exclude",
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Build label and MET-vs-rest representations for binary-criterion metrics.

    Applies ``filter_cannot_assess(mode)`` first, then returns two parallel
    representations of the surviving verdict pairs:

    - ``label_pred`` / ``label_true``: verdict ``.value`` strings, suitable for
      ``accuracy_score`` and ``cohen_kappa_score``. Under ``"as_category"`` these
      may include CANNOT_ASSESS as a distinct third class (so a CANNOT_ASSESS
      prediction matching a CANNOT_ASSESS ground truth counts as correct). Under
      ``"exclude"`` / ``"as_unmet"`` no CANNOT_ASSESS survives, so these are
      two-class and numerically identical to the prior MET/non-MET encoding.
    - ``met_pred`` / ``met_true``: MET one-vs-rest 0/1 ints (1 iff MET), suitable
      for ``precision_score`` / ``recall_score`` / ``f1_score`` and for the
      ``support_true`` / ``support_pred`` counts.

    Args:
        pred_verdicts: Predicted verdicts.
        true_verdicts: Ground truth verdicts.
        mode: CANNOT_ASSESS handling mode (see ``filter_cannot_assess``).

    Returns:
        Tuple of (label_pred, label_true, met_pred, met_true).
    """
    filtered_pred, filtered_true = filter_cannot_assess(pred_verdicts, true_verdicts, mode)
    return (
        verdict_to_string(filtered_pred),
        verdict_to_string(filtered_true),
        verdict_to_binary(filtered_pred),
        verdict_to_binary(filtered_true),
    )


def verdict_to_string(verdicts: Sequence[CriterionVerdict]) -> list[str]:
    """Convert verdicts to string values.

    Args:
        verdicts: List of CriterionVerdict values.

    Returns:
        List of string values.
    """
    return [v.value for v in verdicts]


def classify_criterion(criterion: Criterion) -> CriterionType:
    """Classify a criterion as binary, ordinal, or nominal.

    Args:
        criterion: The criterion to classify.

    Returns:
        CriterionType: "binary", "ordinal", or "nominal".
    """
    if criterion.is_binary:
        return "binary"
    elif criterion.scale_type == "ordinal":
        return "ordinal"
    else:
        return "nominal"


def classify_criteria(criteria: list[Criterion]) -> list[CriterionType]:
    """Classify all criteria in a rubric.

    Args:
        criteria: List of criteria from a rubric.

    Returns:
        List of CriterionType values corresponding to each criterion.
    """
    return [classify_criterion(c) for c in criteria]


def resolve_ground_truth(
    ground_truth: list[CriterionVerdict | str],
    criteria: list[Criterion],
) -> list[CriterionVerdict | int]:
    """Resolve ground truth to standardized format.

    For binary criteria, keeps CriterionVerdict unchanged.
    For multi-choice criteria, resolves string labels to option indices
    using Criterion.find_option_by_label().

    Args:
        ground_truth: Mixed list of CriterionVerdict (binary) or str labels (multi-choice).
        criteria: List of criteria from the rubric, matching ground_truth order.

    Returns:
        List where each element is:
        - CriterionVerdict for binary criteria
        - int (option index) for multi-choice criteria

    Raises:
        ValueError: If ground_truth length doesn't match criteria length.
        ValueError: If a multi-choice label doesn't match any option.
    """
    if len(ground_truth) != len(criteria):
        raise ValueError(
            f"Ground truth length ({len(ground_truth)}) doesn't match "
            f"criteria count ({len(criteria)})"
        )

    resolved = []
    for gt, criterion in zip(ground_truth, criteria):
        if criterion.is_binary:
            # Binary criterion - expect CriterionVerdict
            if isinstance(gt, CriterionVerdict):
                resolved.append(gt)
            elif isinstance(gt, str):
                # Try to parse as CriterionVerdict
                try:
                    resolved.append(CriterionVerdict(gt))
                except ValueError:
                    raise ValueError(
                        f"Invalid binary verdict '{gt}' for criterion '{criterion.name}'. "
                        f"Expected MET, UNMET, or CANNOT_ASSESS."
                    )
            else:
                raise ValueError(
                    f"Expected CriterionVerdict or str for binary criterion '{criterion.name}', "
                    f"got {type(gt).__name__}"
                )
        else:
            # Multi-choice criterion - expect string label
            if isinstance(gt, str):
                # Resolve label to index
                index = criterion.find_option_by_label(gt)
                resolved.append(index)
            elif isinstance(gt, int):
                # Already an index
                if gt < 0 or gt >= len(criterion.options):
                    raise ValueError(
                        f"Option index {gt} out of range [0, {len(criterion.options)}) "
                        f"for criterion '{criterion.name}'"
                    )
                resolved.append(gt)
            else:
                raise ValueError(
                    f"Expected str label or int index for multi-choice criterion "
                    f"'{criterion.name}', got {type(gt).__name__}"
                )

    return resolved


def extract_all_verdicts_from_report(
    report: EvaluationReport | EnsembleEvaluationReport,
    criteria: list[Criterion],
) -> list[CriterionVerdict | int | None]:
    """Extract verdicts for all criteria, handling both binary and multi-choice.

    Args:
        report: The evaluation report (single or ensemble).
        criteria: List of criteria from the rubric.

    Returns:
        List where each element is:
        - CriterionVerdict for binary criteria
        - int (selected_index) for multi-choice criteria
        - None if extraction failed
    """
    if report.report is None:
        # No report available - return defaults based on criterion type
        result = []
        for criterion in criteria:
            if criterion.is_binary:
                result.append(CriterionVerdict.UNMET)
            else:
                result.append(0)  # Default to first option
        return result

    verdicts: list[CriterionVerdict | int | None] = []

    for cr in report.report:
        if isinstance(cr, dict):
            if "verdict" in cr:
                verdicts.append(cr.get("verdict", CriterionVerdict.UNMET))
            elif "multi_choice_verdict" in cr:
                mc = cr["multi_choice_verdict"]
                if mc is not None and "selected_index" in mc:
                    verdicts.append(mc["selected_index"])
                else:
                    verdicts.append(None)
            else:
                verdicts.append(None)
        else:
            binary_v, mc_v = _extract_raw_verdicts(cr)
            if mc_v is not None:
                verdicts.append(mc_v.selected_index)
            elif binary_v is not None:
                verdicts.append(binary_v)
            else:
                verdicts.append(None)

    return verdicts


def filter_na_multi_choice(
    pred_indices: list[int],
    true_indices: list[int],
    criterion: Criterion,
    mode: Literal["exclude", "as_worst"] = "exclude",
) -> tuple[list[int], list[int], int, int, int]:
    """Filter or transform NA options in multi-choice data.

    NA options are treated similarly to CANNOT_ASSESS for binary criteria.

    Args:
        pred_indices: Predicted option indices.
        true_indices: Ground truth option indices.
        criterion: The criterion (used to check which options are NA).
        mode: How to handle NA:
            - "exclude": Remove pairs where either is NA
            - "as_worst": Keep NA but don't count as special

    Returns:
        Tuple of (filtered_pred, filtered_true, na_agreement, na_fp, na_fn):
        - filtered_pred: Filtered/transformed prediction indices
        - filtered_true: Filtered/transformed ground truth indices
        - na_agreement: Count where both agreed on NA
        - na_fp: Count where pred was NA but true was not
        - na_fn: Count where true was NA but pred was not
    """
    # Identify which options are NA
    na_indices = {i for i, opt in enumerate(criterion.options) if opt.na}

    if not na_indices:
        # No NA options - return as-is
        return list(pred_indices), list(true_indices), 0, 0, 0

    filtered_pred = []
    filtered_true = []
    na_agreement = 0
    na_fp = 0  # False positive: pred NA, true not NA
    na_fn = 0  # False negative: true NA, pred not NA

    for p, t in zip(pred_indices, true_indices):
        pred_is_na = p in na_indices
        true_is_na = t in na_indices

        if pred_is_na and true_is_na:
            na_agreement += 1
            if mode == "exclude":
                continue  # Skip this pair
        elif pred_is_na and not true_is_na:
            na_fp += 1
            if mode == "exclude":
                continue
        elif true_is_na and not pred_is_na:
            na_fn += 1
            if mode == "exclude":
                continue

        # Include this pair
        filtered_pred.append(p)
        filtered_true.append(t)

    return filtered_pred, filtered_true, na_agreement, na_fp, na_fn


def get_option_value(criterion: Criterion, index: int) -> float:
    """Get the score value for an option by index.

    Args:
        criterion: The multi-choice criterion.
        index: Zero-based option index.

    Returns:
        The option's value (0.0-1.0).

    Raises:
        ValueError: If criterion is binary or index is out of range.
    """
    if criterion.is_binary:
        raise ValueError("Cannot get option value for binary criterion")
    if index < 0 or index >= len(criterion.options):
        raise ValueError(f"Option index {index} out of range [0, {len(criterion.options)})")
    return criterion.options[index].value


def is_na_option(criterion: Criterion, index: int) -> bool:
    """Check if an option is marked as NA.

    Args:
        criterion: The multi-choice criterion.
        index: Zero-based option index.

    Returns:
        True if the option is NA, False otherwise.

    Raises:
        ValueError: If criterion is binary or index is out of range.
    """
    if criterion.is_binary:
        raise ValueError("Cannot check NA for binary criterion")
    if index < 0 or index >= len(criterion.options):
        raise ValueError(f"Option index {index} out of range [0, {len(criterion.options)})")
    return criterion.options[index].na
