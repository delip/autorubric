"""Single source of truth for weighted criterion scoring.

Used by the grader (live scores), ``Rubric.compute_score`` (ground-truth/expected
scores), and ``RubricDataset.compute_weighted_score``, so all paths agree across
``CannotAssessStrategy`` x {binary, multi-choice} x {+/- weight}.

A score is undefined, ``None``, never a fabricated 0.0, when there is nothing to score:
``score_reports`` returns ``None`` when no criterion is left to score, and a report whose
every criterion's judgment failed (``_every_judgment_failed``) has no score at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from autorubric.types import (
    CannotAssessConfig,
    CannotAssessStrategy,
    CriterionReport,
    EnsembleCriterionReport,
)


def _every_judgment_failed(
    reports: Sequence[CriterionReport | EnsembleCriterionReport],
) -> bool:
    """Whether every judgment of ``reports`` failed: there is at least one, and each stands
    in for failed judge calls (``is_error``; for an ``EnsembleCriterionReport``, every vote
    aggregated on the criterion failed).

    The verdict standing in for a failed call keeps a criterion routed (an abstention, or
    the worst case for an unknown failure), but it is no judgment, so reports whose every
    judgment failed have no score.
    """
    return bool(reports) and all(report.is_error for report in reports)


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
) -> float | None:
    """Compute a weighted score from criterion reports, applying ``config`` uniformly
    to binary CANNOT_ASSESS and multi-choice NA.

    SKIP excludes abstained criteria from BOTH numerator and denominator; ZERO/PARTIAL/
    FAIL keep them in the denominator with a strategy-defined (weight-sign-aware)
    contribution. Returns the raw weighted sum when ``normalize`` is False, else a value
    clamped to [0, 1] (with the negative-weight-only fallback ``1 + sum/neg_weight``).

    Returns ``None``, normalized or raw, when no criterion is left to score: there are no
    reports, or SKIP excluded every one (every criterion abstained). Such a score is
    undefined, never a fabricated 0.0. Zero-weight criteria are scored, though they move
    nothing: when only they are left, the score is 0.0.
    """
    weighted_sum = 0.0
    total_positive_weight = 0.0
    total_negative_weight = 0.0
    n_scored = 0
    for r in reports:
        w = r.weight
        if r.is_na:
            contribution = _abstain_contribution(r, config)
            if contribution is None:  # SKIP: excluded from numerator and denominator
                continue
            weighted_sum += contribution
        else:
            weighted_sum += r.score_value * w
        n_scored += 1
        if w > 0:
            total_positive_weight += w
        else:
            total_negative_weight += abs(w)
    if n_scored == 0:
        return None
    if not normalize:
        return weighted_sum
    if total_positive_weight > 0:
        return max(0.0, min(1.0, weighted_sum / total_positive_weight))
    if total_negative_weight > 0:
        return max(0.0, min(1.0, 1.0 + weighted_sum / total_negative_weight))
    return 0.0  # only zero-weight criteria were scored
