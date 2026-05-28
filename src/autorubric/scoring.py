"""Single source of truth for weighted criterion scoring.

Used by the grader (live scores), ``Rubric.compute_score`` (ground-truth/expected
scores), and ``RubricDataset.compute_weighted_score``, so all paths agree across
``CannotAssessStrategy`` x {binary, multi-choice} x {+/- weight}.
"""

from __future__ import annotations

from autorubric.types import CannotAssessConfig, CannotAssessStrategy, CriterionReport


def _abstain_contribution(report: CriterionReport, config: CannotAssessConfig) -> float | None:
    """Weighted score contribution for an NA / CANNOT_ASSESS criterion under
    ``config.strategy``, or ``None`` if the criterion must be excluded from scoring
    entirely (SKIP). Weight-sign aware.

    FAIL uses the score-minimizing realizable outcome: for multi-choice this is
    ``Criterion.worst_scored_option()`` (the same canonical worst-case helper used by
    the grader's unknown-error path and metrics ``na_mode="as_unmet"``); for binary it
    is UNMET for positive weight (0) and MET for negative weight (the full weight).
    """
    w = report.weight
    strategy = config.strategy
    if strategy == CannotAssessStrategy.SKIP:
        return None
    if strategy == CannotAssessStrategy.ZERO:
        return 0.0
    if strategy == CannotAssessStrategy.PARTIAL:
        return config.partial_credit * w if w > 0 else 0.0
    # FAIL
    if report.is_multi_choice:
        _, opt = report.worst_scored_option()
        return opt.value * w
    return w if w < 0 else 0.0


def score_reports(
    reports: list[CriterionReport],
    config: CannotAssessConfig,
    normalize: bool = True,
) -> float:
    """Compute a weighted score from criterion reports, applying ``config`` uniformly
    to binary CANNOT_ASSESS and multi-choice NA.

    SKIP excludes abstained criteria from BOTH numerator and denominator; ZERO/PARTIAL/
    FAIL keep them in the denominator with a strategy-defined (weight-sign-aware)
    contribution. Returns the raw weighted sum when ``normalize`` is False, else a value
    clamped to [0, 1] (with the negative-weight-only fallback ``1 + sum/neg_weight``).
    """
    weighted_sum = 0.0
    total_positive_weight = 0.0
    total_negative_weight = 0.0
    for r in reports:
        w = r.weight
        if r.is_na:
            contribution = _abstain_contribution(r, config)
            if contribution is None:  # SKIP: excluded from numerator and denominator
                continue
            weighted_sum += contribution
        else:
            weighted_sum += r.score_value * w
        if w > 0:
            total_positive_weight += w
        else:
            total_negative_weight += abs(w)
    if not normalize:
        return weighted_sum
    if total_positive_weight > 0:
        return max(0.0, min(1.0, weighted_sum / total_positive_weight))
    if total_negative_weight > 0:
        return max(0.0, min(1.0, 1.0 + weighted_sum / total_negative_weight))
    return 0.0
