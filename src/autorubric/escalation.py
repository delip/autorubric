"""Offline tools for confidence cascades: replay, diagnose and calibrate one.

A confidence cascade (``CriterionGrader(escalation=EscalationConfig(...))``) has a decision
model judge every criterion and escalates the criteria it is unsure of to LLM judges. How a
cascade would do at a threshold can be measured without running it there:

- ``replay_escalation`` builds, without a single call, the ``EvalResult`` a live cascade
  would have produced, from a run of the decision model alone and a run of the LLM
  judge(s) alone over the same items.
- ``escalation_stats`` summarizes a live or replayed cascade run as an ``EscalationPoint``:
  how much it escalated, how accurate the decision model was on the criteria it kept and
  on those it deferred for its confidence, how accurate the fallback was on the deferred
  ones, and the run's metric, cost and compute time.
- ``calibrate_escalation`` replays and measures a cascade at every threshold of a sweep,
  optionally with thresholds fitted per criterion, and returns the points as an
  ``EscalationCurve``, whose ``best`` picks one by an escalation budget and a tolerance.

A replay reuses the live cascade's own code: its escalation rule (``_escalates``, with the
threshold lookup of ``EscalationConfig.threshold_for``), its threshold validation and
unknown-name warning, its report assembly (``_ensemble_evaluation_report``) and the
scoring core (``score_reports``), so it cannot drift from a live run. A calibration
measures replays with ``escalation_stats``, scored with the scoring settings it is given.
When each run was graded by a ``CriterionGrader`` configured as the cascade for everything
that stage uses (``replay_escalation``'s exactness conditions: for the LLM run, the
cascade's escalation judges, ``seed``, ``shuffle_options``, system prompts and few-shot
settings, among others) and the scoring settings are the cascade's, a point equals the
diagnostics of the live cascade at its thresholds but for two estimates and the
configuration: its ``cost_usd`` and ``compute_seconds`` are pro-rated from per-item
totals, where a live run's are exact sums, and its ``threshold``, ``per_criterion`` and
``calibration_fingerprint`` record what the diagnostics of a live run leave ``None``.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from autorubric.dataset import DataItem, RubricDataset
from autorubric.decision import _is_positive_int, _is_real_number
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.graders.base import _caller_stacklevel
from autorubric.graders.criterion_grader import (
    _check_thresholds,
    _ensemble_evaluation_report,
    _escalates,
    _threshold_for,
    _warn_unknown_threshold_names,
)
from autorubric.metrics import MetricsResult, compute_metrics
from autorubric.metrics._helpers import resolve_ground_truth
from autorubric.scoring import score_reports
from autorubric.types import (
    CannotAssessConfig,
    Criterion,
    CriterionReport,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
    MultiChoiceVerdict,
)
from autorubric.utils import aggregate_completion_cost

if TYPE_CHECKING:
    import pandas as pd


class EscalationPoint(BaseModel):
    """One operating point of a confidence cascade: its thresholds, and what they achieve.

    ``escalation_stats`` measures one on a cascade run, live or replayed, and
    ``calibrate_escalation`` one per threshold of a sweep. Given to ``replay_escalation``, a
    point replays a cascade at its ``threshold`` and ``per_criterion`` thresholds.

    **Pairs and accuracy.** A pair is one criterion of one successfully graded item. A pair
    is *labelled* when its item has ground truth for the criterion that does not abstain:
    a binary label other than ``CANNOT_ASSESS``, or a multi-choice label that is not an NA
    option. An accuracy is the share of a set of labelled pairs, binary and multi-choice
    pooled, on which a prediction equals the label: the verdict of a binary criterion, the
    selected option of a multi-choice one. As ``compute_metrics`` measures one criterion
    under its default handling (``cannot_assess="exclude"``, ``na_mode="exclude"``:
    ``CriterionMetrics.accuracy``, ``exact_accuracy``), an abstention (``CANNOT_ASSESS``, an
    NA option) is not a prediction, so a pair on which the predictor abstained counts in
    neither the numerator nor the denominator. (The run-level ``criterion_accuracy`` of
    ``compute_metrics`` pools otherwise. On its per-criterion path, for items that share
    one rubric, it reads the binary pairs alone when there are any, else the multi-choice
    pairs with an NA prediction or label counted as one more option. On its pooled path,
    for items with different rubrics, it reads the pairs of every scale, leaves out a
    pair whose prediction abstains, and counts one whose label abstains as a miss.)

    - The decision model is measured as ``compute_metrics(..., per_judge=True)`` measures a
      judge: on its own votes, superseded ones included, leaving out a failed judgment (a
      vote with an ``error``), which is no prediction of the decision model's. A
      decision-model failure or abstention escalates at any threshold, so it counts in
      ``escalation_rate`` but in no accuracy: it leaves the decision model no prediction
      to measure, or to compare the fallback's with.
    - The fallback's predictions are the escalated pairs' final verdicts, which the
      escalation judges' votes alone decide, read as ``compute_metrics`` reads a run's: a
      final verdict that abstains is no prediction; one that the judges' failures decided
      (the worst case of an ``unknown`` failure) is one, as it is in the run's metrics.
    - The *deferred* pairs are the escalated labelled pairs the decision model answered:
      it escalated them for a confidence below the threshold. ``dm_accuracy_escalated``
      and ``fallback_accuracy_escalated`` are both measured on the deferred pairs whose
      final verdict does not abstain, where both judges made a prediction, so they compare
      the two judges on the same pairs.

    Attributes:
        threshold: The confidence threshold below which a criterion is escalated. ``None``
            when the point was measured on a run (``escalation_stats``), because an
            ``EvalResult`` does not record its grader's configuration.
        per_criterion: Thresholds for particular criteria, by criterion name (as
            ``EscalationConfig.per_criterion``); ``None`` when there are none or the point
            was measured on a run.
        escalation_rate: The share of the run's pairs, labelled or not, that were escalated.
            ``None`` when the run has no pairs.
        dm_accuracy_kept: The decision model's accuracy on the pairs it kept. A cascade can
            pay only if its confidence separates these from the pairs it defers.
        dm_accuracy_escalated: The decision model's accuracy on the deferred pairs whose
            final verdict does not abstain.
        fallback_accuracy_escalated: The accuracy of the final verdicts, which the fallback
            LLM judges decide, on the same pairs. On them, escalating rather than keeping
            the decision model's verdicts gains right verdicts exactly when this beats
            ``dm_accuracy_escalated``: the condition for a cascade to pay. The deferred
            pairs whose final verdict abstains are not among them; they leave a metric
            that leaves abstentions out (such as ``criterion_accuracy``), which rises when
            the decision model was wrong on them.
        metric: The chosen metric of the whole run, from ``compute_metrics(result,
            dataset)``.
        cost_usd: The run's cost: the sum of its items' ``completion_cost``, ``None`` when
            no item has one. Exact for a live run; for a replay, a sum of estimates (see
            ``replay_escalation``).
        compute_seconds: The sum of the items' grading times (``duration_seconds``), not
            the wall-clock time of the run. Exact for a live run; for a replay, a sum of
            estimates.
        calibration_fingerprint: The fingerprint of the items the point was calibrated
            on (``calibrate_escalation``), when it was; ``None`` otherwise.
            ``replay_escalation`` warns when it equals the fingerprint of the items being
            replayed, which it can compute only from runs that record the items'
            submissions (not from runs loaded with ``EvalResult.from_experiment``).

    Accuracies are ``None`` when their set of labelled pairs is empty.
    """

    model_config = ConfigDict(frozen=True)

    threshold: float | None = None
    per_criterion: dict[str, float] | None = None
    escalation_rate: float | None
    dm_accuracy_kept: float | None
    dm_accuracy_escalated: float | None
    fallback_accuracy_escalated: float | None
    metric: float | None
    cost_usd: float | None
    compute_seconds: float
    calibration_fingerprint: str | None = None


class EscalationCurve(BaseModel):
    """The points of a threshold sweep (``calibrate_escalation``), and the rule that picks one.

    Each point is a cascade at one global threshold (plus the fitted per-criterion
    thresholds, when the calibration fitted any), measured on the calibration items, and
    carries their ``calibration_fingerprint``. ``best`` picks a point by one rule;
    ``replay_escalation(dm_result, llm_result, point)``, given the scoring settings the
    calibration was given, rebuilds the point's run on the runs it was calibrated on, or
    replays its thresholds on held-out runs.

    **The selection rule** (``best``). Among the points within the escalation budget
    (``escalation_rate <= max_escalation_rate``; every point when there is no budget), find
    the best ``metric``; of the points whose metric is within ``tolerance`` of it
    (inclusively, a gap equal to ``tolerance`` up to floating-point rounding included), pick
    the one that escalates least. A higher metric is better. Ties are broken in order by
    the higher metric, then the lower ``threshold`` (equal here, but on new items a lower
    threshold never escalates more), then the earlier point. A point without a ``metric``
    (``None``, or NaN) is passed over; one without an ``escalation_rate`` (a run with no
    pairs) is outside any budget and ranks after every point that has one; one without a
    ``threshold`` ranks after those with one.
    ``tolerance > 0`` trades metric for less escalation, e.g. "about the same accuracy for
    fewer LLM calls": ``best(tolerance=0.005)`` gives up at most half a point of accuracy.

    **Same-data caveat.** Every point was measured on the pairs the sweep chose from, so the
    metric of the point ``best`` picks is optimistic: it is the best of many measurements
    on those pairs. Measure the chosen point on held-out items graded by the same judges
    (``replay_escalation`` on their runs, or a live cascade). Replaying it on its own
    calibration items warns (``calibration_fingerprint``) when either run records the items'
    submissions; runs loaded with ``EvalResult.from_experiment`` record none, so there the
    check cannot recognize them.

    Attributes:
        points: The operating points, in the order of the sweep.
    """

    model_config = ConfigDict(frozen=True)

    points: list[EscalationPoint]

    def best(
        self, max_escalation_rate: float | None = None, tolerance: float = 0.0
    ) -> EscalationPoint:
        """The point the selection rule picks (see the class docstring).

        Args:
            max_escalation_rate: The escalation budget, a share of pairs in [0, 1]
                (inclusive), or ``None`` for no budget.
            tolerance: How far below the best metric within the budget a point's metric
                may be, in the metric's units (a number >= 0).

        Returns:
            The least escalating point within ``tolerance`` of the best metric within the
            budget.

        Raises:
            ValueError: If ``max_escalation_rate`` or ``tolerance`` is out of range; if the
                curve has no points, none within the budget, or none within the budget
                with a metric.
        """
        if max_escalation_rate is not None and not (
            _is_real_number(max_escalation_rate) and 0 <= max_escalation_rate <= 1
        ):
            raise ValueError(
                "max_escalation_rate must be a number in [0, 1] or None; "
                f"got {max_escalation_rate!r}"
            )
        if not (_is_real_number(tolerance) and tolerance >= 0):
            raise ValueError(f"tolerance must be a number >= 0; got {tolerance!r}")
        candidates = [(p.escalation_rate, p.metric, p.threshold) for p in self.points]
        return self.points[_selected(candidates, max_escalation_rate, tolerance)]

    def to_dataframe(self) -> pd.DataFrame:
        """The points as a pandas ``DataFrame``: a row per point, a column per field of
        ``EscalationPoint`` (``per_criterion`` holds each point's dict, or ``None``).

        pandas is imported here, as ``MetricsResult.to_dataframe`` imports it.
        """
        import pandas as pd

        return pd.DataFrame(
            [point.model_dump() for point in self.points],
            columns=list(EscalationPoint.model_fields),
        )


def _selected(
    candidates: Sequence[tuple[float | None, float | None, float | None]],
    max_escalation_rate: float | None = None,
    tolerance: float = 0.0,
) -> int:
    """The position of the candidate the selection rule of ``EscalationCurve.best`` picks.

    The rule's one implementation: ``EscalationCurve.best`` applies it to a curve's points,
    and ``calibrate_escalation``, when it fits thresholds per criterion, to the global
    sweep (for the fit's start) and to each criterion's thresholds.

    Args:
        candidates: ``(escalation_rate, metric, threshold)`` per candidate, in order; a
            ``None`` or NaN metric is undefined.
        max_escalation_rate: The escalation budget, or ``None``.
        tolerance: How far below the best metric within the budget a metric may be.

    Raises:
        ValueError: If no candidate is within the budget, or none within it has a metric.
    """
    if not candidates:
        raise ValueError("the curve has no points to choose from")
    if max_escalation_rate is not None:
        rates = [rate for rate, _, _ in candidates if rate is not None]
        if not rates:
            raise ValueError(
                "no point of the curve has an escalation rate (its runs had no pairs), so none "
                f"is within max_escalation_rate={max_escalation_rate!r}"
            )
        if min(rates) > max_escalation_rate:
            raise ValueError(
                f"no point of the curve is within max_escalation_rate={max_escalation_rate!r}: "
                f"the lowest is {min(rates)!r}"
            )
    # (escalation rate, metric, threshold, position) of the candidates in the running.
    scored: list[tuple[float, float, float, int]] = []
    for position, (rate, metric, threshold) in enumerate(candidates):
        if metric is None or math.isnan(metric):
            continue
        if max_escalation_rate is not None and (rate is None or rate > max_escalation_rate):
            continue
        scored.append(
            (
                math.inf if rate is None else rate,
                metric,
                math.inf if threshold is None else threshold,
                position,
            )
        )
    if not scored:
        within = "" if max_escalation_rate is None else f" within {max_escalation_rate=}"
        raise ValueError(f"no point of the curve{within} has a metric (each one's is None or NaN)")
    best = max(metric for _, metric, _, _ in scored)
    # Within the tolerance, inclusively: a gap equal to it up to rounding (0.4 - 0.35 is
    # 0.05000000000000004 in floating point) is within it.
    close = [
        entry
        for entry in scored
        if entry[1] >= best - tolerance or math.isclose(best - entry[1], tolerance)
    ]
    # The least escalating; then the higher metric, the lower threshold, the earlier one.
    return min(close, key=lambda entry: (entry[0], -entry[1], entry[2], entry[3]))[3]


# =============================================================================
# Replay
# =============================================================================


def replay_escalation(
    dm_result: EvalResult,
    llm_result: EvalResult,
    escalation: EscalationPoint | float,
    *,
    per_criterion: Mapping[str, float] | None = None,
    cannot_assess_config: CannotAssessConfig | None = None,
    normalize: bool = True,
) -> EvalResult:
    """Build the ``EvalResult`` a confidence cascade would have produced, with no calls.

    ``dm_result`` is a run of the decision model alone and ``llm_result`` a run of the LLM
    judge(s) alone, over the same items; items are paired by ``item_idx``. For each
    criterion of each item, the decision model's recorded vote either stands or is
    superseded, by the rule of a live cascade (``CriterionGrader`` with ``escalation=``): the
    criterion is escalated when the vote failed (an ``error``), abstained (``CANNOT_ASSESS``
    or an NA option), or has a ``confidence`` below the criterion's threshold (its
    ``per_criterion`` threshold by name, else ``threshold``).

    - A kept criterion gets the decision-model run's report of it: its one vote, and the
      final verdict aggregated from it.
    - An escalated criterion gets the LLM run's report of it, with the decision model's vote
      in front, ``superseded=True``, and ``escalated=True``. A live cascade aggregates the
      votes that are not superseded, which are exactly the LLM run's, so the final verdict,
      ``final_reason``, ``error`` and ``agreement`` are the ones the LLM run's aggregation
      recorded; an LLM panel's votes stay aggregated with the LLM run's ``aggregation``,
      ``ordinal_aggregation`` and ``nominal_aggregation`` and its judges' weights.
    - Replayed votes keep their ``judge_id``s. ``judge_scores`` holds the decision model's
      own score over every criterion (its superseded votes included) and ``None`` for each
      LLM judge, by role. Scores are computed with ``score_reports`` under
      ``cannot_assess_config`` and ``normalize``, as the cascade's grader would.

    **Exactness.** The replay equals a live cascade exactly when each run was graded by a
    ``CriterionGrader`` configured as the cascade for everything that stage uses, over the
    same items and rubrics (guidelines included), and this call's ``cannot_assess_config``
    and ``normalize`` are the cascade's:

    - the decision-model run: one judge, the cascade's decision model (the same config,
      ``judge_id`` and weight);
    - the LLM run: the cascade's escalation judges (``judges=escalation.judges``, the same
      configs, ``judge_id``s and weights; a bare ``LLMConfig`` given to
      ``EscalationConfig`` becomes ``JudgeSpec(config, "escalation")``) with every setting
      they use as the cascade's: its ``seed`` (``CriterionGrader.seed``, generated when not
      given), ``shuffle_options``, ``system_prompt``, ``multi_choice_system_prompt``,
      ``training_data`` and ``few_shot_config``, and the default response formats (a
      cascade cannot have others);
    - both: the cascade's ``auto_na_option`` and aggregation settings (``aggregation``,
      ``ordinal_aggregation``, ``nominal_aggregation``).

    For ``CriterionGrader(judge_model_config=dm, escalation=escalation, seed=S)`` with
    otherwise default settings, the two runs are graded by
    ``CriterionGrader(judge_model_config=dm)`` and
    ``CriterionGrader(judges=escalation.judges, seed=S)``. Only the criteria are checked
    (as graded, NA options that ``auto_na_option`` adds included): an ``EvalResult``
    records no grader configuration, so another setting goes unnoticed. With only another
    ``seed`` or ``judge_id``, the LLM prompts differ in the order of a shuffled
    multi-choice criterion's options and in the few-shot examples, and the replay equals
    the cascade in distribution only; with another prompt setting (``shuffle_options``, a
    system prompt, the few-shot settings) the prompts themselves differ, and the replay is
    not the cascade's run.

    **Estimates.** Per-criterion LLM cost and time are not recorded, only per-item totals.
    An item's ``completion_cost`` is therefore its decision-model cost plus its LLM cost
    pro-rated by the share of its criteria that were escalated (``None`` when that is not
    positive, as in a live report), and its ``duration_seconds`` its decision-model
    duration plus its LLM duration pro-rated the same way. The run's total duration is the
    decision-model run's total plus the LLM run's, pro-rated by the share of the LLM run's
    pairs that were escalated; an item whose decision-model grading failed escalates none
    of its pairs, as a live cascade sends it to no LLM judge. These are estimates, which
    ``escalation_stats`` sums into ``cost_usd`` and ``compute_seconds``. The result says so
    in its ``experiment_name``, its descriptive field (``"replay of decision-model run
    '<name>' and LLM run '<name>' (estimated cost and time)"``), since a replay is not an
    experiment on disk (``experiment_dir`` is ``None``). It carries no ``token_usage``,
    which cannot be split by criterion.

    **Length penalty.** It is not re-applied: compare replayed scores with runs graded
    without one. Criterion-level metrics are unaffected.

    An item whose grading failed in ``dm_result`` stays failed, as it would in a live
    cascade, whose decision model grades first.

    Args:
        dm_result: A run of one decision-model judge alone (e.g.
            ``CriterionGrader(judge_model_config=DecisionModelConfig(...))``), live or
            loaded with ``EvalResult.from_experiment``.
        llm_result: A run of the LLM judge(s) alone over the same items and criteria, with
            a vote of every judge on every criterion of every item: whichever criteria a
            threshold escalates, the run costs a full LLM grading of its items.
        escalation: The threshold, a number in [0, 1], or an ``EscalationPoint`` (e.g. one
            of ``calibrate_escalation``'s curve) whose ``threshold`` and ``per_criterion``
            are used.
        per_criterion: Thresholds for particular criteria, by criterion name, each in
            [0, 1]; only with a number as ``escalation``.
        cannot_assess_config: How abstentions score (``score_reports``); default
            ``CannotAssessConfig()``, as ``CriterionGrader``'s.
        normalize: Whether scores are normalized to [0, 1], as ``CriterionGrader``'s.

    Returns:
        An ordinary ``EvalResult`` (``compute_metrics``, ``escalation_stats`` and
        ``ItemResult.to_dict`` work on it), with one item result per paired item, in
        ``item_idx`` order.

    Raises:
        ValueError: If the threshold or a ``per_criterion`` threshold is not a number in
            [0, 1]; if ``per_criterion`` is given with an ``EscalationPoint``, or the point
            has no threshold; if the two results do not cover the same items (by
            ``item_idx``, a result holding an item more than once, or an item whose
            recorded submissions differ) graded against the same criteria (as graded, NA
            options included); if ``dm_result`` does not come from one decision-model judge
            (every vote carries ``probabilities`` and a ``confidence``, unless its judgment
            failed); if ``llm_result`` has an item that failed, a criterion missing one of
            its judges' votes, a decision model's vote, or judges that differ between
            items; or if a ``judge_id`` names both the decision model and an LLM judge.

    Warns:
        UserWarning: If a ``per_criterion`` name matches no criterion of the replayed
            items (it sets no threshold), as a live cascade warns. If ``escalation`` is an
            ``EscalationPoint`` whose ``calibration_fingerprint`` is the fingerprint of the
            items being replayed: a threshold chosen and evaluated on the same pairs looks
            better than it will on new items. (An item loaded with
            ``EvalResult.from_experiment`` records no submission, so the check cannot
            recognize it.)
    """
    # The thresholds, checked as EscalationConfig checks them.
    if isinstance(escalation, EscalationPoint):
        if per_criterion is not None:
            raise ValueError(
                "per_criterion goes with a float threshold: an EscalationPoint carries its own "
                "(EscalationPoint.per_criterion)"
            )
        if escalation.threshold is None:
            raise ValueError(
                "This EscalationPoint has no threshold: it was measured on a run "
                "(escalation_stats), which does not record its grader's configuration; pass "
                "a threshold, or a point that has one"
            )
        threshold, per_criterion = escalation.threshold, escalation.per_criterion
        threshold_name = "EscalationPoint.threshold"
        per_criterion_name = "EscalationPoint.per_criterion"
        calibrated_on = escalation.calibration_fingerprint
    else:
        threshold = escalation
        threshold_name, per_criterion_name = "escalation", "per_criterion"
        calibrated_on = None
    per_criterion = _check_thresholds(
        threshold,
        per_criterion,
        threshold_name=threshold_name,
        per_criterion_name=per_criterion_name,
    )

    pairs = _paired_items(dm_result, llm_result)
    _warn_unknown_threshold_names(
        per_criterion,
        (_llm_criteria(llm_item) for _, llm_item in pairs),
        source=per_criterion_name,
        where="the replayed items",
    )
    if calibrated_on is not None and calibrated_on == _replayed_fingerprint(pairs):
        warnings.warn(
            "This EscalationPoint was calibrated on the items it replays (its "
            "calibration_fingerprint matches theirs): a threshold chosen and evaluated on the "
            "same pairs looks better than it will on new items; replay held-out items",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
    config = cannot_assess_config if cannot_assess_config is not None else CannotAssessConfig()

    def score(reports: list[CriterionReport], normalized: bool) -> float:
        return score_reports(reports, config, normalized)

    item_results: list[ItemResult] = []
    # The escalated pairs, and the pairs the LLM run judged.
    n_escalated = n_llm_pairs = 0
    for dm_item, llm_item in pairs:
        item_result, escalated, judged = _replayed_item(
            dm_item, llm_item, threshold, per_criterion, score, normalize
        )
        item_results.append(item_result)
        n_escalated += escalated
        n_llm_pairs += judged
    share = n_escalated / n_llm_pairs if n_llm_pairs else 0.0

    # Totals as EvalRunner computes them.
    errors = [(r.item_idx, r.error) for r in item_results if r.error]
    reports = [r.report for r in item_results if r.error is None]
    total_duration = (
        dm_result.timing_stats.total_duration_seconds
        + llm_result.timing_stats.total_duration_seconds * share
    )
    replayed_at = datetime.now()
    return EvalResult(
        item_results=item_results,
        total_items=len(item_results),
        successful_items=len(item_results) - len(errors),
        failed_items=len(errors),
        total_token_usage=None,
        total_completion_cost=aggregate_completion_cost([r.completion_cost for r in reports]),
        timing_stats=EvalTimingStats.from_durations(
            [r.duration_seconds for r in item_results], total_duration
        ),
        started_at=replayed_at,
        completed_at=replayed_at,
        errors=errors,
        experiment_name=(
            f"replay of {_run_name(dm_result, 'decision-model')} and "
            f"{_run_name(llm_result, 'LLM')} (estimated cost and time)"
        ),
        experiment_dir=None,
    )


def _run_name(result: EvalResult, kind: str) -> str:
    """How a replay's name refers to one of the runs it was built from."""
    if result.experiment_name is None:
        return f"an unnamed {kind} run"
    return f"{kind} run {result.experiment_name!r}"


def _paired_items(
    dm_result: EvalResult, llm_result: EvalResult
) -> list[tuple[ItemResult, ItemResult]]:
    """The two results' item results, paired by ``item_idx`` in index order, after checking
    everything a replay needs of them (see ``replay_escalation``'s Raises)."""
    dm_items = _items_by_index(dm_result, "dm_result")
    llm_items = _items_by_index(llm_result, "llm_result")
    if dm_items.keys() != llm_items.keys():
        raise ValueError(
            "dm_result and llm_result must cover the same items; "
            f"only in dm_result: {sorted(dm_items.keys() - llm_items.keys())}, "
            f"only in llm_result: {sorted(llm_items.keys() - dm_items.keys())}"
        )
    dm_judge_id: str | None = None
    llm_judge_ids: list[str] | None = None
    pairs: list[tuple[ItemResult, ItemResult]] = []
    for idx in sorted(dm_items):
        dm_item, llm_item = dm_items[idx], llm_items[idx]
        judge_ids = _checked_llm_judge_ids(llm_item)
        if llm_judge_ids is None:
            llm_judge_ids = judge_ids
        elif judge_ids != llm_judge_ids:
            raise ValueError(
                "llm_result must come from one set of LLM judges, as a cascade's escalation "
                f"judges are: item {idx} was graded by {judge_ids}, earlier items by "
                f"{llm_judge_ids}"
            )
        submissions = (dm_item.item.submission, llm_item.item.submission)
        # A result loaded with EvalResult.from_experiment records no submission ("").
        if all(submissions) and submissions[0] != submissions[1]:
            raise ValueError(
                f"item {idx} is not the same item in dm_result and llm_result: its "
                "submissions differ"
            )
        if dm_item.error is None:
            judge_id = _checked_dm_judge_id(dm_item)
            if dm_judge_id is None:
                dm_judge_id = judge_id
            elif judge_id != dm_judge_id:
                raise ValueError(
                    "dm_result must come from one decision-model judge: item "
                    f"{idx} was judged by {judge_id!r}, earlier items by {dm_judge_id!r}"
                )
            if judge_id in judge_ids:
                raise ValueError(
                    "judge_ids must be unique across the decision model and the LLM judges, "
                    f"as in a live cascade: {judge_id!r} names both (name the LLM run's judges "
                    "as the cascade's escalation judges are named, e.g. "
                    'JudgeSpec(config, "escalation"))'
                )
            dm_criteria = [cr.criterion for cr in _ensemble_report(dm_item, "dm_result")[1]]
            if dm_criteria != _llm_criteria(llm_item):
                raise ValueError(
                    f"item {idx} was graded against different criteria in dm_result and "
                    "llm_result (criteria as graded, NA options guaranteed by "
                    "auto_na_option included); a replay needs the same rubric graded the "
                    "same way"
                )
        pairs.append((dm_item, llm_item))
    return pairs


def _items_by_index(result: EvalResult, name: str) -> dict[int, ItemResult]:
    by_index: dict[int, ItemResult] = {}
    for item_result in result.item_results:
        if item_result.item_idx in by_index:
            raise ValueError(f"{name} holds item {item_result.item_idx} more than once")
        by_index[item_result.item_idx] = item_result
    return by_index


def _ensemble_report(
    item_result: ItemResult, source: str
) -> tuple[EnsembleEvaluationReport, list[EnsembleCriterionReport]]:
    """A graded item's report and its criterion reports (in rubric order), which hold the
    judges' votes; ``source`` names the result the item is from, for the error."""
    report = item_result.report
    if not isinstance(report, EnsembleEvaluationReport) or report.report is None:
        raise ValueError(
            f"{source} item {item_result.item_idx} has no judges' votes: its report is not an "
            "EnsembleEvaluationReport with criterion reports"
        )
    return report, report.report


def _llm_criteria(llm_item: ItemResult) -> list[Criterion]:
    """The criteria an ``llm_result`` item was graded against (NA options included)."""
    return [cr.criterion for cr in _ensemble_report(llm_item, "llm_result")[1]]


def _votes(report: EnsembleCriterionReport) -> Sequence[JudgeVote | MultiChoiceJudgeVote]:
    """A criterion report's votes: ``multi_choice_votes`` or binary ``votes``."""
    return report.multi_choice_votes if report.criterion.is_multi_choice else report.votes


def _is_decision_model_vote(vote: JudgeVote | MultiChoiceJudgeVote) -> bool:
    """Whether a vote is a decision model's: it carries ``probabilities`` and a
    ``confidence`` (an LLM vote carries neither), unless its judgment failed."""
    return vote.error is not None or (
        vote.probabilities is not None and vote.confidence is not None
    )


def _checked_llm_judge_ids(item_result: ItemResult) -> list[str]:
    """An ``llm_result`` item's judge ids, checked: graded, by LLM judges alone, with every
    judge's vote on every criterion."""
    idx = item_result.item_idx
    if item_result.error is not None:
        raise ValueError(
            f"llm_result has no votes on item {idx}: grading it failed ({item_result.error}); "
            "a replay needs the LLM judges' votes on every criterion of every item"
        )
    report, criterion_reports = _ensemble_report(item_result, "llm_result")
    judge_ids = list(report.judge_scores)
    for c, cr in enumerate(criterion_reports):
        votes = _votes(cr)
        if cr.escalated or any(
            v.superseded or v.confidence is not None or v.probabilities is not None for v in votes
        ):
            raise ValueError(
                "llm_result must come from LLM judges alone, as a cascade's escalation judges "
                f"are: item {idx}, criterion {c} holds a decision model's vote or is a "
                "cascade's"
            )
        voters = [v.judge_id for v in votes]
        if not voters or voters != judge_ids:
            raise ValueError(
                "llm_result must hold every judge's vote on every criterion: item "
                f"{idx}, criterion {c} has votes from {voters}; the item's judges are "
                f"{judge_ids}"
            )
    return judge_ids


def _checked_dm_judge_id(item_result: ItemResult) -> str:
    """A graded ``dm_result`` item's judge id, checked: one judge, a decision model, with
    one vote on every criterion."""
    idx = item_result.item_idx
    report, criterion_reports = _ensemble_report(item_result, "dm_result")
    judge_ids = list(report.judge_scores)
    if len(judge_ids) != 1:
        raise ValueError(
            f"dm_result must come from one decision-model judge: item {idx} was graded by "
            f"{judge_ids}"
        )
    (judge_id,) = judge_ids
    for c, cr in enumerate(criterion_reports):
        votes = _votes(cr)
        voters = [v.judge_id for v in votes]
        if cr.escalated or voters != [judge_id] or votes[0].superseded:
            raise ValueError(
                f"dm_result must come from one decision-model judge: item {idx}, criterion "
                f"{c} holds votes from {voters}"
            )
        if not _is_decision_model_vote(votes[0]):
            raise ValueError(
                f"dm_result must come from one decision-model judge: the vote of {judge_id!r} "
                f"on item {idx}, criterion {c} carries no confidence, as an LLM judge's does not"
            )
    return judge_id


def _recorded_item(dm_item: ItemResult, llm_item: ItemResult) -> DataItem:
    """The pair's item, taken from a result that recorded its submission if either did (a
    result loaded with ``EvalResult.from_experiment`` records none)."""
    if dm_item.item.submission or not llm_item.item.submission:
        return dm_item.item
    return llm_item.item


def _replayed_item(
    dm_item: ItemResult,
    llm_item: ItemResult,
    threshold: float,
    per_criterion: Mapping[str, float] | None,
    score: Callable[[list[CriterionReport], bool], float],
    normalize: bool,
) -> tuple[ItemResult, int, int]:
    """One item as a live cascade would have graded it, with the number of its criteria
    that were escalated and the number the LLM run judged (every criterion of the item).

    A live cascade sends an item whose decision-model grading failed to no LLM judge, so
    none of its criteria are escalated, though the LLM run judged them all.
    """
    item = _recorded_item(dm_item, llm_item)
    if dm_item.error is not None:
        failed = ItemResult(
            item_idx=dm_item.item_idx,
            item=item,
            report=dm_item.report,
            duration_seconds=dm_item.duration_seconds,
            error=dm_item.error,
        )
        return failed, 0, len(_llm_criteria(llm_item))

    dm_report, dm_criteria = _ensemble_report(dm_item, "dm_result")
    llm_report, llm_criteria = _ensemble_report(llm_item, "llm_result")
    reports: list[EnsembleCriterionReport] = []
    judged: list[CriterionReport] = []
    n_escalated = 0
    for dm_cr, llm_cr in zip(dm_criteria, llm_criteria, strict=True):
        (vote,) = _votes(dm_cr)
        dm_judged = _judged_report(dm_cr.criterion, vote)
        judged.append(dm_judged)
        if _escalates(dm_judged, _threshold_for(dm_cr.criterion, threshold, per_criterion)):
            n_escalated += 1
            reports.append(_escalated_report(llm_cr, vote))
        else:
            reports.append(dm_cr)
    share = n_escalated / len(reports) if reports else 0.0

    # The decision model's score over its own verdicts; each LLM judge's is None by role.
    (dm_judge_id,) = dm_report.judge_scores
    judge_scores: dict[str, float | None] = {dm_judge_id: score(judged, normalize)}
    judge_scores.update(dict.fromkeys(llm_report.judge_scores))
    llm_cost = llm_report.completion_cost
    cost = sum(
        part
        for part in (dm_report.completion_cost, None if llm_cost is None else llm_cost * share)
        if part is not None
    )
    report = _ensemble_evaluation_report(
        reports,
        judge_scores,
        score,
        normalize=normalize,
        token_usage=None,
        completion_cost=cost if cost > 0 else None,
    )
    replayed = ItemResult(
        item_idx=dm_item.item_idx,
        item=item,
        report=report,
        duration_seconds=dm_item.duration_seconds + llm_item.duration_seconds * share,
    )
    return replayed, n_escalated, len(reports)


def _judged_report(criterion: Criterion, vote: JudgeVote | MultiChoiceJudgeVote) -> CriterionReport:
    """The report a judge made on ``criterion``, recovered from the vote it cast there (the
    inverse of how ``CriterionGrader.aggregate`` records a judge's report as its vote)."""
    outcome: dict[str, Any]
    if isinstance(vote, MultiChoiceJudgeVote):
        outcome = {
            "verdict": None,
            "multi_choice_verdict": MultiChoiceVerdict(
                selected_index=vote.selected_index,
                selected_label=vote.selected_label,
                value=vote.value,
                na=vote.na,
            ),
            "shuffle_order": vote.shuffle_order,
        }
    else:
        outcome = {"verdict": vote.verdict}
    return CriterionReport(
        **dict(criterion),
        **outcome,
        reason=vote.reason,
        reasoning=vote.reasoning,
        error=vote.error,
        probabilities=vote.probabilities,
        confidence=vote.confidence,
    )


def _escalated_report(
    llm_report: EnsembleCriterionReport, dm_vote: JudgeVote | MultiChoiceJudgeVote
) -> EnsembleCriterionReport:
    """The report a live cascade builds for an escalated criterion: the LLM run's report of
    it, whose votes are the ones the cascade aggregates, with the decision model's vote in
    front, ``superseded=True``, and ``escalated=True``. ``agreement`` counts only the votes
    that are not superseded, so the LLM run's value stands."""
    votes = [dm_vote.model_copy(update={"superseded": True}), *_votes(llm_report)]
    field = "multi_choice_votes" if llm_report.criterion.is_multi_choice else "votes"
    return llm_report.model_copy(update={field: votes, "escalated": True})


def _calibration_fingerprint(items: Iterable[tuple[DataItem, Iterable[Criterion]]]) -> str:
    """SHA-256 over the contents of items, in order: each item's submission, prompt and
    reference submission, and the requirements of the criteria it is graded against.

    It identifies a set of calibration items where ``autorubric.eval._compute_dataset_hash``
    cannot: that hash covers only a dataset's name, prompt and sizes, which the two halves
    of ``RubricDataset.split_train_test`` share. The prompt and reference are the item's
    own (``DataItem.prompt``, ``DataItem.reference_submission``); a dataset-level one is
    shared by every item of a split and is not recorded on an ``EvalResult``. Computed from
    a dataset's items and ``get_item_rubric`` criteria or from a result's items and graded
    criteria alike, since grading keeps every requirement's text (and only adds NA
    options).
    """
    content = [
        [
            item.submission,
            item.prompt,
            item.reference_submission,
            [criterion.requirement for criterion in criteria],
        ]
        for item, criteria in items
    ]
    return hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()


def _replayed_fingerprint(pairs: Iterable[tuple[ItemResult, ItemResult]]) -> str:
    """The calibration fingerprint of the items a replay pairs."""
    return _calibration_fingerprint(
        (_recorded_item(dm_item, llm_item), _llm_criteria(llm_item)) for dm_item, llm_item in pairs
    )


# =============================================================================
# Diagnostics
# =============================================================================


def escalation_stats(
    result: EvalResult,
    dataset: RubricDataset,
    *,
    metric: str | Callable[[MetricsResult], float | None] = "criterion_accuracy",
) -> EscalationPoint:
    """Diagnostics of a live or replayed cascade run, as an ``EscalationPoint``.

    Reads each criterion report's ``escalated`` and ``superseded`` flags: a kept criterion's
    one vote is the decision model's, and an escalated criterion's first vote is the
    decision model's, superseded, while its final verdict is the fallback's. It pools
    (item, criterion) pairs, so it works on datasets whose items have different rubrics,
    where ``compute_metrics`` has no per-criterion table. The pairs and accuracies are
    defined in ``EscalationPoint``. Like ``compute_metrics``, it covers the items both in
    ``result`` and in ``dataset``; items whose grading failed have no pairs, but their
    grading time counts in ``compute_seconds``.

    The point's ``threshold`` and ``per_criterion`` are ``None`` (an ``EvalResult`` does
    not record its grader's configuration), as is its ``calibration_fingerprint``.
    ``cost_usd`` and ``compute_seconds`` are exact sums for a live run and sums of
    estimates for a replay (``replay_escalation``).

    Args:
        result: A cascade run: ``CriterionGrader(escalation=...)`` through ``evaluate``, or
            ``replay_escalation``. A run of its decision model alone reads as a cascade that
            escalated nothing.
        dataset: The dataset with the ground truth.
        metric: The metric of the whole run: the name of a ``MetricsResult`` attribute
            (default ``"criterion_accuracy"``), or a function of the ``MetricsResult``,
            e.g. one that reads ``pooled_by_scale`` on a dataset with per-item rubrics. Any
            real number (a NumPy scalar included; not a bool) is recorded as a float; a
            NaN is read as undefined: the point's ``metric`` is then ``None``.

    Returns:
        The run's ``EscalationPoint``.

    Raises:
        ValueError: If ``result`` is not a cascade run: an item with no judges' votes, a
            kept criterion with other than one vote, an escalated one whose first vote is
            not the only superseded one, or a first vote that is not a decision model's
            (one with ``probabilities`` and a ``confidence``, unless its judgment failed).
            If an item's report and its rubric in ``dataset`` differ in their number of
            criteria, or if ``metric`` names no attribute of ``MetricsResult`` or gives
            something other than a number or ``None``. ``compute_metrics``' own errors
            propagate (e.g. no item with ground truth).
    """
    items = _covered_items(result, dataset)
    n_pairs = n_escalated = 0
    dm_kept: list[bool] = []
    dm_escalated: list[bool] = []
    fallback: list[bool] = []
    for report, dm_vote, label in _cascade_pairs(items, dataset):
        n_pairs += 1
        n_escalated += report.escalated
        # A decision-model failure or abstention escalates at any threshold and is no
        # prediction to compare the fallback's with.
        if label is None or (prediction := _vote_prediction(dm_vote)) is None:
            continue
        if not report.escalated:
            dm_kept.append(prediction == label)
        elif (final := _final_prediction(report)) is not None:
            # A deferred pair the fallback answered: both judges are measured on it.
            dm_escalated.append(prediction == label)
            fallback.append(final == label)

    return EscalationPoint(
        escalation_rate=n_escalated / n_pairs if n_pairs else None,
        dm_accuracy_kept=_accuracy(dm_kept),
        dm_accuracy_escalated=_accuracy(dm_escalated),
        fallback_accuracy_escalated=_accuracy(fallback),
        metric=_metric_value(compute_metrics(result, dataset), metric),
        cost_usd=aggregate_completion_cost(
            [r.report.completion_cost for r in items if r.error is None]
        ),
        compute_seconds=float(sum(r.duration_seconds for r in items)),
    )


def _covered_items(result: EvalResult, dataset: RubricDataset) -> list[ItemResult]:
    """The item results of ``result`` whose items are in ``dataset`` (by ``item_idx``), in
    index order: the items ``compute_metrics(result, dataset)`` covers."""
    by_index = {r.item_idx: r for r in result.item_results}
    return [by_index[idx] for idx in sorted(by_index.keys() & set(range(len(dataset))))]


def _cascade_pairs(
    items: Iterable[ItemResult], dataset: RubricDataset
) -> Iterator[
    tuple[EnsembleCriterionReport, JudgeVote | MultiChoiceJudgeVote, CriterionVerdict | int | None]
]:
    """A cascade run's pairs: each criterion report of each graded item of ``items``, in
    order, with the decision model's vote on it (``_decision_model_vote``, which checks the
    cascade's shape) and its label in ``dataset`` (``None`` when it has none, as
    ``_labels``)."""
    for item_result in items:
        if item_result.error is not None:
            continue
        idx = item_result.item_idx
        _, reports = _ensemble_report(item_result, "result")
        criteria = dataset.get_item_rubric(idx).rubric
        if len(reports) != len(criteria):
            raise ValueError(
                f"result item {idx} has {len(reports)} criteria; its rubric in the dataset has "
                f"{len(criteria)}"
            )
        for c, (report, label) in enumerate(zip(reports, _labels(dataset[idx], criteria))):
            yield report, _decision_model_vote(report, idx, c), label


_NOT_A_CASCADE = "escalation_stats reads a cascade run (or one of its decision model alone)"


def _decision_model_vote(
    report: EnsembleCriterionReport, idx: int, c: int
) -> JudgeVote | MultiChoiceJudgeVote:
    """The decision model's vote in a cascade's criterion report: the only vote of a kept
    criterion, the first (and only superseded) vote of an escalated one."""
    votes = _votes(report)
    if report.escalated:
        if not (votes and votes[0].superseded and not any(v.superseded for v in votes[1:])):
            problem = "is escalated, but its first vote is not the only superseded one"
            raise ValueError(f"{_NOT_A_CASCADE}: item {idx}, criterion {c} {problem}")
    elif len(votes) != 1 or votes[0].superseded:
        problem = f"is not escalated, but has {len(votes)} votes, not the decision model's alone"
        raise ValueError(f"{_NOT_A_CASCADE}: item {idx}, criterion {c} {problem}")
    vote = votes[0]
    if not _is_decision_model_vote(vote):
        raise ValueError(
            f"{_NOT_A_CASCADE}: item {idx}, criterion {c}: the vote of {vote.judge_id!r} "
            "carries no confidence, as a decision model's does"
        )
    return vote


def _labels(item: DataItem, criteria: list[Criterion]) -> list[CriterionVerdict | int | None]:
    """An item's labels for its criteria: a binary verdict or an option index, ``None``
    where there is none or it abstains.

    An item whose ground truth does not resolve against its rubric has no labels, as
    ``compute_metrics`` skips its pairs.
    """
    if item.ground_truth is None:
        return [None] * len(criteria)
    try:
        labels = resolve_ground_truth(list(item.ground_truth), criteria)
    except ValueError:
        return [None] * len(criteria)
    return [
        None if _abstains(label, criterion) else label
        for label, criterion in zip(labels, criteria, strict=True)
    ]


def _abstains(label: CriterionVerdict | int, criterion: Criterion) -> bool:
    """Whether a resolved label abstains: ``CANNOT_ASSESS``, or an NA option."""
    if criterion.options is None:
        return label == CriterionVerdict.CANNOT_ASSESS
    return isinstance(label, int) and criterion.options[label].na


def _vote_prediction(vote: JudgeVote | MultiChoiceJudgeVote) -> CriterionVerdict | int | None:
    """A judge's prediction from its vote; ``None`` when it failed or abstained."""
    if vote.error is not None:
        return None
    if isinstance(vote, MultiChoiceJudgeVote):
        return None if vote.na else vote.selected_index
    return None if vote.verdict == CriterionVerdict.CANNOT_ASSESS else vote.verdict


def _final_prediction(report: EnsembleCriterionReport) -> CriterionVerdict | int | None:
    """A criterion's final verdict as a prediction; ``None`` when it abstains."""
    if report.criterion.is_multi_choice:
        final = report.final_multi_choice_verdict
        return None if final is None or final.na else final.selected_index
    verdict = report.final_verdict
    return None if verdict == CriterionVerdict.CANNOT_ASSESS else verdict


def _accuracy(outcomes: list[bool]) -> float | None:
    return sum(outcomes) / len(outcomes) if outcomes else None


def _metric_value(
    metrics: MetricsResult, metric: str | Callable[[MetricsResult], float | None]
) -> float | None:
    """The value of ``metric`` on ``metrics`` as a float, or ``None`` when undefined (a NaN,
    as a function computing it with NumPy may give, included).

    A number is any real number but a bool: a Python ``int`` or ``float``, or a NumPy
    scalar such as ``float32`` or ``int64``, which NumPy registers as ``numbers.Real``.
    """
    if isinstance(metric, str):
        try:
            value = getattr(metrics, metric)
        except AttributeError:
            raise ValueError(f"metric {metric!r} is not an attribute of MetricsResult") from None
    else:
        value = metric(metrics)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"metric {metric!r} must give a number or None, not {type(value).__name__}; pass "
            "a function that reads a number from the MetricsResult"
        )
    value = float(value)
    return None if math.isnan(value) else value


# =============================================================================
# Calibration
# =============================================================================

_DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(i / 50 for i in range(51))
"""The thresholds ``calibrate_escalation`` sweeps by default: 0.00, 0.02, ..., 1.00."""


def calibrate_escalation(
    dataset: RubricDataset,
    dm_result: EvalResult,
    llm_result: EvalResult,
    *,
    thresholds: Sequence[float] | None = None,
    metric: str | Callable[[MetricsResult], float | None] = "criterion_accuracy",
    per_criterion: bool = False,
    min_pairs_per_criterion: int = 30,
    cannot_assess_config: CannotAssessConfig | None = None,
    normalize: bool = True,
) -> EscalationCurve:
    """Measure a confidence cascade at every threshold of a sweep, with no calls.

    For each threshold of ``thresholds`` the cascade is replayed from the two runs
    (``replay_escalation``, scoring as the cascade's grader scores, by
    ``cannot_assess_config`` and ``normalize``) and the replay measured against ``dataset``
    (``escalation_stats``, whose ``metric`` is read from ``compute_metrics(replay,
    dataset)``). Each point of the curve carries its thresholds, its ``escalation_rate``,
    the decision model's accuracy on the pairs it kept and on those it deferred, the
    fallback's accuracy on the same deferred pairs (see ``EscalationPoint``), ``metric``,
    the replay's estimated
    ``cost_usd`` and ``compute_seconds`` (see ``replay_escalation``), and the
    ``calibration_fingerprint`` of the calibration items: the items of ``dataset`` the two
    runs graded (by ``item_idx``). ``EscalationCurve.best`` picks a point, and
    ``replay_escalation`` replays one.

    **Cost.** The sweep makes no calls, but its inputs are two full gradings of the
    calibration items: the decision model's, and the LLM judges' votes on every criterion
    of every item (``llm_result``), whichever criteria a threshold would escalate. That LLM
    run is the price of calibration; calibrate on a sample of the items the cascade is
    meant to grade.

    **Same-data caveat.** Every point is measured on the pairs the thresholds are chosen
    from, so the metric of the point ``best`` picks is optimistic: it is the best of many
    measurements on the same pairs. Measure the chosen point on held-out items graded by the
    same judges (e.g. the other part of ``RubricDataset.split_train_test``), with
    ``replay_escalation`` or a live cascade. Replaying it on its own calibration items warns,
    by its ``calibration_fingerprint``, when either run records the items' submissions: runs
    loaded with ``EvalResult.from_experiment`` record none, so replaying the point on them
    cannot be recognized and does not warn.

    **Exactness.** Each replay is the run a live cascade at the point's thresholds would
    produce when the two runs meet ``replay_escalation``'s exactness conditions (each
    graded by a ``CriterionGrader`` configured as the cascade for everything that stage
    uses: for the LLM run, ``judges=escalation.judges`` with the cascade's ``seed``,
    ``shuffle_options``, system prompts, few-shot settings, ``auto_na_option`` and
    aggregation settings; e.g. ``CriterionGrader(judges=escalation.judges, seed=S)`` for a
    cascade built with ``seed=S`` and otherwise default settings) and when
    ``cannot_assess_config`` and ``normalize`` are the cascade's grader's: score-level
    metrics (``score_rmse``, ``score_pearson``, ...) read the scores they set. With only
    another seed or ``judge_id`` the LLM prompts differ in the order of a shuffled
    multi-choice criterion's options and in the few-shot examples, and the curve holds for
    the cascade in distribution only; with another prompt setting it does not describe the
    cascade.

    **Per-criterion thresholds** (``per_criterion=True``). A threshold of its own
    (``EscalationConfig.per_criterion``) applies to every criterion of a name, so it is
    fitted on that name's pairs, across the calibration items.

    - *Which criteria.* Each named criterion with at least ``min_pairs_per_criterion``
      labelled pairs (a pair's item graded, its label not an abstention; see
      ``EscalationPoint``). The others use the swept global threshold: unnamed criteria,
      which cannot have a threshold of their own, and named criteria with fewer labelled
      pairs, which the call warns about.
    - *The same selection rule, on the run's metric.* The fit starts from the global
      threshold that ``EscalationCurve.best``'s rule, at its defaults (no budget, no
      tolerance), picks on the sweep alone. Each fitted criterion's threshold in turn, in
      order of first appearance, is then the threshold of the sweep that the same rule
      picks on the whole run's ``metric``, with the global threshold and the thresholds
      already fitted held where they are. A criterion's threshold decides whether its own
      pairs escalate and nothing else, so each fit is made on that criterion's pairs, by
      what they do to ``metric``, as ``metric`` counts them. ``criterion_accuracy`` (whose
      pooling on each of ``compute_metrics``' paths ``EscalationPoint`` describes), for
      one, leaves out an escalated pair whose final verdict abstains, but for an NA
      verdict it counts as an option (in a rubric without binary criteria, on the
      per-criterion path); a criterion it does not read (a multi-choice criterion in a
      rubric with binary criteria, on the per-criterion path) escalates as little as the
      sweep lets it, while on the pooled path it reads every criterion. No fit lowers the
      metric, so the point at the start's
      global threshold measures at least as high with the fits as without them, and so
      does the point ``best`` picks at its defaults. If ``metric`` is undefined at every
      threshold of the sweep, there is no start, and the call fits nothing and warns.
    - *The points.* Each point is one global threshold of the sweep plus the fitted
      thresholds, the same at every point: its ``threshold`` and ``per_criterion``, exactly
      what ``replay_escalation(dm_result, llm_result, point)`` replays and what a live
      cascade is configured with (``EscalationConfig(judges, point.threshold,
      per_criterion=point.per_criterion)``), measured on the whole run like any point.
      ``best`` therefore picks among global thresholds with the fits in place: its budget
      and tolerance trade on the criteria that follow the global threshold.

    Fitting replays the sweep once more for each fitted criterion, and once with the fits
    in place.
    Fitting on a few dozen pairs is where the same-data caveat bites hardest;
    ``min_pairs_per_criterion`` bounds it.

    Args:
        dataset: The calibration items with their ground truth: the dataset both runs
            graded.
        dm_result: A run of the cascade's decision-model judge alone over the calibration
            items (as for ``replay_escalation``).
        llm_result: A run of the cascade's escalation judge(s) alone over the same items,
            with a vote on every criterion (as for ``replay_escalation``).
        thresholds: The global thresholds to sweep, each a number in [0, 1], in the order
            the curve keeps; default 0.00, 0.02, ..., 1.00.
        metric: The metric of a whole replay, higher being better: the name of a
            ``MetricsResult`` attribute (default ``"criterion_accuracy"``) or a function of
            the ``MetricsResult`` (negate a metric where lower is better); a NaN is read as
            undefined (``None``). On a dataset whose items have different rubrics,
            ``compute_metrics`` takes its pooled path: ``MetricsResult.per_criterion`` is
            empty and the run-level fields are pooled over rubric points
            (``criterion_accuracy`` is the exact accuracy over the points of every scale,
            a point whose prediction abstains left out and one whose label abstains
            counted as a miss), so a metric of one scale type needs a function that reads
            ``pooled_by_scale``.
        per_criterion: Whether named criteria get thresholds of their own, fitted as above.
        min_pairs_per_criterion: The labelled pairs a criterion needs for a threshold of
            its own, a positive int.
        cannot_assess_config: How abstentions score in the replays (``score_reports``), as
            in ``replay_escalation``: the cascade's grader's; default
            ``CannotAssessConfig()``.
        normalize: Whether the replays' scores are normalized to [0, 1], as the cascade's
            grader's.

    Returns:
        The curve: one point per threshold of ``thresholds``, in their order.

    Raises:
        ValueError: If ``thresholds`` is empty or holds anything but a number in [0, 1]; if
            ``per_criterion`` is not a bool (thresholds by name are ``replay_escalation``'s)
            or ``min_pairs_per_criterion`` not a positive int. The errors of
            ``replay_escalation`` (the two runs are not a decision model's and its
            escalation judges' over the same items and criteria) and of
            ``escalation_stats`` (``metric`` gives no number; ``compute_metrics``' own)
            propagate.

    Warns:
        UserWarning: With ``per_criterion=True``, naming the named criteria that use the
            global threshold for want of labelled pairs, or because ``metric`` is undefined
            at every threshold of the sweep.
    """
    sweep = _DEFAULT_THRESHOLDS if thresholds is None else tuple(thresholds)
    if not sweep:
        raise ValueError("thresholds must hold at least one threshold")
    for i, threshold in enumerate(sweep):
        _check_thresholds(
            threshold, None, threshold_name=f"thresholds[{i}]", per_criterion_name="per_criterion"
        )
    if not isinstance(per_criterion, bool):
        raise ValueError(
            "per_criterion must be True or False (whether to fit thresholds per criterion); "
            f"got {per_criterion!r} (thresholds by criterion name go to replay_escalation)"
        )
    if not _is_positive_int(min_pairs_per_criterion):
        raise ValueError(
            f"min_pairs_per_criterion must be a positive int; got {min_pairs_per_criterion!r}"
        )

    fingerprint = _calibration_fingerprint(
        (dataset[r.item_idx], dataset.get_item_rubric(r.item_idx).rubric)
        for r in _covered_items(dm_result, dataset)
    )

    def measured(threshold: float, fitted: Mapping[str, float]) -> EscalationPoint:
        """The point of the replay at ``threshold`` with the ``fitted`` thresholds."""
        replay = replay_escalation(
            dm_result,
            llm_result,
            threshold,
            per_criterion=fitted or None,
            cannot_assess_config=cannot_assess_config,
            normalize=normalize,
        )
        configuration = {
            "threshold": float(threshold),
            "per_criterion": dict(fitted) or None,
            "calibration_fingerprint": fingerprint,
        }
        return escalation_stats(replay, dataset, metric=metric).model_copy(update=configuration)

    points = [measured(threshold, {}) for threshold in sweep]
    if per_criterion:
        fitted = _fitted_thresholds(
            dataset, dm_result, sweep, points, measured, min_pairs_per_criterion
        )
        if fitted:
            points = [measured(threshold, fitted) for threshold in sweep]
    return EscalationCurve(points=points)


def _fitted_thresholds(
    dataset: RubricDataset,
    dm_result: EvalResult,
    sweep: Sequence[float],
    global_points: Sequence[EscalationPoint],
    measured: Callable[[float, Mapping[str, float]], EscalationPoint],
    min_pairs: int,
) -> dict[str, float]:
    """Each named criterion's own threshold, fitted by the selection rule on the run's
    metric (see ``calibrate_escalation``), for the criteria with at least ``min_pairs``
    labelled pairs; warns about the named criteria that get none.

    Args:
        dataset: The calibration dataset.
        dm_result: The decision-model run, whose pairs are the calibration pairs (a replay
            has the same ones at every threshold).
        sweep: The thresholds of the sweep.
        global_points: The sweep's points without fitted thresholds, in its order.
        measured: The point of the replay at a global threshold with thresholds by name.
        min_pairs: The labelled pairs a criterion needs for a threshold of its own.

    Returns:
        The fitted thresholds by name, in order of the names' first appearance.
    """
    # Every named criterion, in order of first appearance, with its labelled pairs.
    labelled: Counter[str] = Counter()
    for report, _, label in _cascade_pairs(_covered_items(dm_result, dataset), dataset):
        if (name := report.criterion.name) is not None:
            labelled[name] += label is not None
    names = [name for name, n in labelled.items() if n >= min_pairs]
    thin = [name for name, n in labelled.items() if n < min_pairs]
    if thin:
        _warn_global_threshold(
            thin,
            labelled,
            f"a criterion's own threshold needs at least min_pairs_per_criterion={min_pairs} "
            "labelled pairs",
        )
    if not names:
        return {}
    if all(point.metric is None for point in global_points):
        _warn_global_threshold(
            names,
            labelled,
            "the metric is undefined (None) at every threshold of the sweep, so no threshold "
            "can be chosen by it",
        )
        return {}

    start = sweep[_selected([(p.escalation_rate, p.metric, p.threshold) for p in global_points])]
    fitted: dict[str, float] = {}
    for name in names:
        # The current thresholds are among the candidates (``name`` at ``start``), so the
        # metric never falls.
        candidates: list[tuple[float | None, float | None, float | None]] = []
        for threshold in sweep:
            point = measured(start, {**fitted, name: threshold})
            candidates.append((point.escalation_rate, point.metric, threshold))
        fitted[name] = float(sweep[_selected(candidates)])
    return fitted


def _warn_global_threshold(names: list[str], labelled: Mapping[str, int], reason: str) -> None:
    """Warn that the named criteria ``names`` use the global threshold, and why."""
    listed = ", ".join(
        f"{name!r} ({labelled[name]} labelled pair{'' if labelled[name] == 1 else 's'})"
        for name in names
    )
    warnings.warn(
        f"calibrate_escalation fits no threshold for {listed}: {reason}; "
        f"{'it uses' if len(names) == 1 else 'they use'} the swept global threshold",
        UserWarning,
        stacklevel=_caller_stacklevel(),
    )
