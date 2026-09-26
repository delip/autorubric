"""A failed criterion judgment is neither a prediction nor an abstention in the metrics.

When a judge call for a criterion fails, the grader stands in a verdict so that scoring can
go on (``_failed_judgment_result``): an abstention for an infrastructure or parse failure
(``CANNOT_ASSESS``, or the NA option of a multi-choice criterion; no option at all,
``selected_index=None``, for a forced-choice criterion without one) and the weight-sign
worst case for an unknown failure. The criterion report carries the failure in ``error``
(``is_error``; an ``EnsembleCriterionReport`` only when every vote aggregated on it failed).
An item every criterion of which failed is an errored item
(``tests/metrics/test_all_errored_items.py``). On any other item a failed criterion, a
failed cell, is no judgment, so ``compute_metrics`` reads it as neither a prediction nor an
abstention, under every ``cannot_assess`` and ``na_mode``, for binary and multi-choice
criteria alike:

- it is left out of the per-criterion metrics, the aggregate scalars, ``NAStats`` and
  ``CannotAssessStats``, and the bootstrap's verdict axis (items are resampled as before,
  with one index shared by every criterion, and each criterion's resampled failed cells
  are dropped);
- it never gives a criterion the auto-injected NA option, and neither does a judge's
  failed vote: only judgments some metric reads decide a criterion's options;
- the pooled per-item-rubric path makes it no point and no abstention;
- coverage (under the ``exclude`` modes) counts it as an errored pair: a criterion's
  ``n_errored`` is the errored items plus its failed cells, and the aggregate ``n_errored``
  is the sum over the criteria, a pair count like the aggregate ``n_total``;
- a warning says how many were left out.

The aggregate therefore agrees with the per-judge metrics, which leave out a failed vote:
for a one-judge ensemble a failed vote and a failed criterion are the same cell. The last
section runs a real grader whose LLM calls fail for some criteria (mocked, nothing reaches
the network).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest

from autorubric import LLMConfig, evaluate
from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMClient
from autorubric.metrics import (
    CannotAssessMode,
    CriterionMetrics,
    MetricsResult,
    NAMode,
    NominalCriterionMetrics,
    OrdinalCriterionMetrics,
    compute_metrics,
)
from autorubric.rubric import Rubric
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    Criterion,
    CriterionJudgment,
    CriterionOption,
    CriterionReport,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
    MultiChoiceJudgment,
    MultiChoiceVerdict,
    TokenUsage,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CA = CriterionVerdict.CANNOT_ASSESS

INFRA = "infrastructure: 503 overloaded"
PARSE = "parse: unparseable judge output"
UNKNOWN = "unknown: judge crashed"

CANNOT_ASSESS_MODES: list[CannotAssessMode] = ["exclude", "as_unmet", "as_category"]
NA_MODES: list[NAMode] = ["exclude", "as_unmet", "as_category"]

FAILED_CELLS = (
    "criterion judgment(s) on graded items failed and were left out of the criterion-level metrics."
)
ERRORED_ITEMS = "item(s) with ground truth were excluded from metrics because grading errored"

BINARY = [
    Criterion(name="c0", requirement="First requirement", weight=1.0),
    Criterion(name="c1", requirement="Second requirement", weight=1.0),
]
CITES = Criterion(name="cites", requirement="Cites a source", weight=1.0)
QUALITY = Criterion(
    name="quality",
    requirement="How good is the answer?",
    weight=1.0,
    scale_type="ordinal",
    options=[
        CriterionOption(label="Poor", value=0.0),
        CriterionOption(label="Fair", value=0.5),
        CriterionOption(label="Good", value=1.0),
    ],
)
TONE = Criterion(
    name="tone",
    requirement="Which tone does the answer take?",
    weight=1.0,
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Casual", value=0.5),
        CriterionOption(label="Slang", value=0.0),
    ],
)

Cell = tuple[int, int]
"""An (item, criterion) pair."""
Prediction = CriterionVerdict | int | None
"""A binary verdict, or a multi-choice option index (``None``: no option selected)."""


# =============================================================================
# Building runs
# =============================================================================


def _options(criterion: Criterion) -> list[CriterionOption]:
    assert criterion.options is not None
    return criterion.options


def _labels(criterion: Criterion) -> list[str]:
    return [option.label for option in _options(criterion)]


def _dataset(
    criteria: list[Criterion], ground_truth: Sequence[Sequence[CriterionVerdict | str]]
) -> RubricDataset:
    dataset = RubricDataset(prompt="p", rubric=Rubric(criteria), name="failed-criteria")
    for idx, truth in enumerate(ground_truth):
        dataset.add_item(submission=f"s{idx}", description=f"item {idx}", ground_truth=[*truth])
    return dataset


def _eval_result(items: list[ItemResult]) -> EvalResult:
    now = datetime.now()
    failed = sum(1 for item in items if item.error is not None)
    return EvalResult(
        item_results=items,
        total_items=len(items),
        successful_items=len(items) - failed,
        failed_items=failed,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=EvalTimingStats(
            total_duration_seconds=1.0,
            mean_item_duration_seconds=0.1,
            min_item_duration_seconds=0.1,
            max_item_duration_seconds=0.1,
            p50_item_duration_seconds=0.1,
            p95_item_duration_seconds=0.1,
            items_per_second=1.0,
        ),
        started_at=now,
        completed_at=now,
    )


def _reason(error: str | None) -> str:
    return f"Judge call failed: {error}" if error else "judged"


def _binary_vote(verdict: CriterionVerdict, error: str | None, judge_id: str) -> JudgeVote:
    return JudgeVote(judge_id=judge_id, verdict=verdict, reason=_reason(error), error=error)


def _mc_vote(
    criterion: Criterion, index: int | None, error: str | None, judge_id: str
) -> MultiChoiceJudgeVote:
    if index is None:  # a forced-choice abstention: no option to select
        return MultiChoiceJudgeVote(
            judge_id=judge_id,
            selected_index=None,
            selected_label=None,
            value=0.0,
            reason=_reason(error),
            na=True,
            error=error,
        )
    # An index past the author's options is the NA option the grader appends.
    option = _options(criterion.with_guaranteed_na_option())[index]
    return MultiChoiceJudgeVote(
        judge_id=judge_id,
        selected_index=index,
        selected_label=option.label,
        value=option.value,
        reason=_reason(error),
        na=option.na,
        error=error,
    )


def _criterion_error(votes: Sequence[JudgeVote | MultiChoiceJudgeVote]) -> str | None:
    """A criterion's error, as the grader sets it: only when every vote on it failed."""
    errors = [vote.error for vote in votes]
    return " | ".join(e for e in errors if e) if all(errors) else None


def _ensemble_criterion_report(
    criterion: Criterion, judged: Sequence[tuple[str, Prediction, str | None]]
) -> EnsembleCriterionReport:
    """A criterion's report from its judges' ``(judge_id, prediction, error)``: the final
    verdict is the first genuine vote's, or the stand-in when every vote failed."""
    if criterion.options is None:
        votes: list[JudgeVote] = []
        for judge_id, prediction, error in judged:
            assert isinstance(prediction, CriterionVerdict)
            votes.append(_binary_vote(prediction, error, judge_id))
        final = next((v for v in votes if v.error is None), votes[0])
        return EnsembleCriterionReport(
            criterion=criterion,
            final_verdict=final.verdict,
            final_reason="aggregated",
            votes=votes,
            error=_criterion_error(votes),
        )
    mc_votes: list[MultiChoiceJudgeVote] = []
    for judge_id, prediction, error in judged:
        assert prediction is None or isinstance(prediction, int)
        mc_votes.append(_mc_vote(criterion, prediction, error, judge_id))
    ref = next((v for v in mc_votes if v.error is None), mc_votes[0])
    return EnsembleCriterionReport(
        criterion=criterion,
        final_verdict=None,
        final_reason="aggregated",
        final_multi_choice_verdict=AggregatedMultiChoiceVerdict(
            selected_index=ref.selected_index,
            selected_label=ref.selected_label,
            value=ref.value,
            na=ref.na,
            aggregated_value=ref.value,
        ),
        multi_choice_votes=mc_votes,
        error=_criterion_error(mc_votes),
    )


def _criterion_fields(criterion: Criterion) -> dict[str, Any]:
    return {name: getattr(criterion, name) for name in Criterion.model_fields}


def _single_criterion_report(
    criterion: Criterion, prediction: Prediction, error: str | None
) -> CriterionReport:
    """A single judge's report, as the grader builds it (on the effective criterion)."""
    if criterion.options is None:
        assert isinstance(prediction, CriterionVerdict)
        return CriterionReport(
            **_criterion_fields(criterion), verdict=prediction, reason=_reason(error), error=error
        )
    effective = criterion.with_guaranteed_na_option()
    if prediction is None:
        verdict = MultiChoiceVerdict(selected_index=None, selected_label=None, value=0.0, na=True)
    else:
        assert isinstance(prediction, int)
        option = _options(effective)[prediction]
        verdict = MultiChoiceVerdict(
            selected_index=prediction, selected_label=option.label, value=option.value, na=option.na
        )
    return CriterionReport(
        **_criterion_fields(effective),
        multi_choice_verdict=verdict,
        reason=_reason(error),
        error=error,
    )


def _item(
    dataset: RubricDataset,
    idx: int,
    cells: Sequence[tuple[Criterion, Sequence[tuple[str, Prediction, str | None]]]],
    kind: str = "ensemble",
) -> ItemResult:
    """An item graded by its judges: per criterion, each judge's (id, prediction, error).
    ``kind="single"`` builds a single judge's ``EvaluationReport`` (one judge per cell)."""
    report: EvaluationReport | EnsembleEvaluationReport
    if kind == "single":
        crs = []
        for criterion, judged in cells:
            ((_, prediction, error),) = judged
            crs.append(_single_criterion_report(criterion, prediction, error))
        report = EvaluationReport(score=0.5, raw_score=0.5, report=crs)
    else:
        ecrs = [_ensemble_criterion_report(criterion, judged) for criterion, judged in cells]
        judge_ids = sorted({judge_id for _, judged in cells for judge_id, _, _ in judged})
        judge_scores: dict[str, float | None] = {judge_id: 0.5 for judge_id in judge_ids}
        report = EnsembleEvaluationReport(
            score=0.5, raw_score=0.5, report=ecrs, judge_scores=judge_scores
        )
    return ItemResult(item_idx=idx, item=dataset.items[idx], report=report, duration_seconds=0.1)


def _run(
    dataset: RubricDataset,
    criteria: list[Criterion],
    predictions: Sequence[Sequence[Prediction]],
    *,
    failed: dict[Cell, tuple[Prediction, str]] | None = None,
    kind: str = "ensemble",
) -> EvalResult:
    """Judge "a" predicts ``predictions``, but a ``failed`` cell carries the stand-in and
    error of a failed call instead."""
    failed = failed or {}
    items = []
    for idx, row in enumerate(predictions):
        cells = []
        for c, criterion in enumerate(criteria):
            prediction, error = failed.get((idx, c), (row[c], None))
            cells.append((criterion, [("a", prediction, error)]))
        items.append(_item(dataset, idx, cells, kind))
    return _eval_result(items)


def _grading_raised(dataset: RubricDataset, idx: int) -> ItemResult:
    report = EnsembleEvaluationReport(score=None, raw_score=None, error="boom")
    return ItemResult(
        item_idx=idx, item=dataset.items[idx], report=report, duration_seconds=0.1, error="boom"
    )


def _binary(metrics: MetricsResult, c: int) -> CriterionMetrics:
    cm = metrics.per_criterion[c]
    assert isinstance(cm, CriterionMetrics)
    return cm


def _multi_choice(
    metrics: MetricsResult, c: int
) -> OrdinalCriterionMetrics | NominalCriterionMetrics:
    cm = metrics.per_criterion[c]
    assert isinstance(cm, (OrdinalCriterionMetrics, NominalCriterionMetrics))
    return cm


# =============================================================================
# The aggregate agrees with the judge
# =============================================================================

REPRO_TRUTH = [[MET, MET], [UNMET, UNMET], [UNMET, MET]]


def test_the_aggregate_agrees_with_a_one_judge_ensemble_on_a_failed_criterion():
    """Judge "a" is right on every criterion it judged; its call for c1 on item 2 failed
    with an unknown error, whose stand-in (UNMET, the worst case of a positive weight) is
    wrong. The judge's metrics leave the failed vote out, and so does the aggregate: both
    read five judged cells, all right (the aggregate used to read six and report 5/6)."""
    dataset = _dataset(BINARY, REPRO_TRUTH)
    result = _run(dataset, BINARY, REPRO_TRUTH, failed={(2, 1): (UNMET, UNKNOWN)})

    metrics = compute_metrics(result, dataset, per_judge=True)

    assert metrics.per_judge is not None
    judge = metrics.per_judge["a"]
    assert metrics.criterion_accuracy == judge.criterion_accuracy == 1.0
    assert metrics.criterion_precision == judge.criterion_precision == 1.0
    assert metrics.criterion_recall == judge.criterion_recall == 1.0
    assert metrics.criterion_f1 == judge.criterion_f1 == 1.0
    assert metrics.mean_kappa == judge.mean_kappa == 1.0
    assert metrics.criterion_phi == judge.phi == 1.0
    assert metrics.n_samples == 5
    c1 = _binary(metrics, 1)
    assert c1.n_samples == 2
    assert c1.confusion_matrix is not None and c1.confusion_matrix.matrix == [[1, 0], [0, 1]]


# =============================================================================
# Binary criteria, every cannot_assess mode
# =============================================================================

BINARY_TRUTH = [[MET, MET], [UNMET, UNMET], [MET, UNMET], [MET, MET]]
FAILED_BINARY_CELL = (3, 0)  # its truth is MET


@pytest.mark.parametrize("mode", CANNOT_ASSESS_MODES)
@pytest.mark.parametrize(
    ("stand_in", "error", "weight"),
    [
        (CA, INFRA, 1.0),
        (CA, PARSE, 1.0),
        (UNMET, UNKNOWN, 1.0),  # the worst case of a positive weight
        (MET, UNKNOWN, -1.0),  # the worst case of a negative weight
    ],
    ids=["infrastructure", "parse", "unknown", "unknown-negative-weight"],
)
def test_a_failed_binary_criterion_is_no_prediction_in_any_mode(mode, stand_in, error, weight):
    criteria = [BINARY[0].model_copy(update={"weight": weight}), BINARY[1]]
    dataset = _dataset(criteria, BINARY_TRUTH)
    result = _run(dataset, criteria, BINARY_TRUTH, failed={FAILED_BINARY_CELL: (stand_in, error)})

    metrics = compute_metrics(result, dataset, cannot_assess=mode)

    c0 = _binary(metrics, 0)
    assert (c0.n_samples, c0.accuracy) == (3, 1.0)
    # Items 0 and 2 are true MET predicted MET, item 1 true UNMET predicted UNMET.
    assert c0.confusion_matrix is not None and c0.confusion_matrix.matrix == [[2, 0], [0, 1]]
    assert _binary(metrics, 1).n_samples == 4
    assert (metrics.n_samples, metrics.criterion_accuracy) == (7, 1.0)
    # Its stand-in abstention is no judge abstention.
    ca = metrics.cannot_assess_stats
    assert ca is not None
    assert (ca.ca_count_pred, ca.ca_count_true, ca.ca_false_positive, ca.ca_false_negative) == (
        0,
        0,
        0,
        0,
    )


@pytest.mark.parametrize(
    ("mode", "c1_samples", "c1_accuracy"),
    [("exclude", 3, 1.0), ("as_unmet", 4, 1.0), ("as_category", 4, 0.75)],
)
def test_a_genuine_cannot_assess_still_counts_and_a_failed_one_does_not(
    mode, c1_samples, c1_accuracy
):
    """The judge genuinely answered CANNOT_ASSESS on item 1's c1 (true UNMET), which each
    mode handles as an abstention; its call for c0 on item 3 (true CANNOT_ASSESS) failed,
    and that pair counts on neither side, whatever the mode."""
    truth = [[MET, MET], [UNMET, UNMET], [MET, UNMET], [CA, MET]]
    predictions = [[MET, MET], [UNMET, CA], [MET, UNMET], [CA, MET]]
    dataset = _dataset(BINARY, truth)
    result = _run(dataset, BINARY, predictions, failed={(3, 0): (CA, INFRA)})

    metrics = compute_metrics(result, dataset, cannot_assess=mode)

    assert _binary(metrics, 0).n_samples == 3
    c1 = _binary(metrics, 1)
    assert (c1.n_samples, c1.accuracy) == (c1_samples, c1_accuracy)
    ca = metrics.cannot_assess_stats
    assert ca is not None
    assert (ca.ca_count_pred, ca.ca_count_true, ca.ca_false_positive, ca.ca_false_negative) == (
        1,
        0,
        1,
        0,
    )


def test_a_criterion_failed_on_every_graded_item_has_no_criterion_metrics():
    """Nothing judged c1, so its metrics are undefined (None), never a stand-in's."""
    dataset = _dataset(BINARY, BINARY_TRUTH)
    failed: dict[Cell, tuple[Prediction, str]] = {
        (idx, 1): (UNMET, UNKNOWN) for idx in range(len(BINARY_TRUTH))
    }
    result = _run(dataset, BINARY, BINARY_TRUTH, failed=failed)

    metrics = compute_metrics(result, dataset)

    c1 = _binary(metrics, 1)
    assert (c1.n_samples, c1.accuracy, c1.kappa) == (0, None, None)
    assert metrics.n_items == 4
    assert metrics.criterion_accuracy == metrics.macro_accuracy == 1.0
    assert c1.coverage_stats is not None
    assert (c1.coverage_stats.n_covered, c1.coverage_stats.n_errored) == (0, 4)
    assert c1.coverage_stats.error_rate == 1.0
    assert f"4 {FAILED_CELLS}" in metrics.warnings


# =============================================================================
# Multi-choice criteria, every na_mode
# =============================================================================

MC_TRUTH = {
    "quality": ["Poor", "Fair", "Good", "Good"],
    "tone": ["Formal", "Casual", "Slang", "Formal"],
}
CITES_TRUTH = [MET, UNMET, MET, UNMET]
FAILED_ITEM = 3


def _stand_in(criterion: Criterion, stand_in: str) -> tuple[int | None, str]:
    """What the grader stands in for a failed multi-choice call."""
    if stand_in == "auto-na":  # infrastructure/parse with the default auto_na_option=True
        return len(_options(criterion)), PARSE
    if stand_in == "worst":  # unknown: the worst scored option
        return criterion.worst_scored_option()[0], UNKNOWN
    return None, INFRA  # infrastructure/parse, forced choice: no option to abstain with


def _mc_rows(
    criterion: Criterion,
) -> tuple[list[list[CriterionVerdict | str]], list[list[Prediction]]]:
    """The ground truth of a [criterion, CITES] rubric and a judge right on every cell."""
    labels = MC_TRUTH[criterion.name or ""]
    truth: list[list[CriterionVerdict | str]] = [
        [label, cites] for label, cites in zip(labels, CITES_TRUTH, strict=True)
    ]
    index = {label: i for i, label in enumerate(_labels(criterion))}
    predictions: list[list[Prediction]] = [
        [index[label], cites] for label, cites in zip(labels, CITES_TRUTH, strict=True)
    ]
    return truth, predictions


@pytest.mark.parametrize("kind", ["ensemble", "single"])
@pytest.mark.parametrize("na_mode", NA_MODES)
@pytest.mark.parametrize("stand_in", ["auto-na", "worst", "no-option"])
@pytest.mark.parametrize("criterion", [QUALITY, TONE], ids=["ordinal", "nominal"])
def test_a_failed_multi_choice_criterion_is_no_prediction_in_any_mode(
    criterion, stand_in, na_mode, kind
):
    """Whatever the stand-in, the failed cell is left out, and the criterion gains no NA
    option: its only NA-like cell failed. So ``na_mode="as_category"`` is not refused for
    the ordinal criterion, which has no NA option to place."""
    criteria = [criterion, CITES]
    truth, predictions = _mc_rows(criterion)
    dataset = _dataset(criteria, truth)
    result = _run(
        dataset,
        criteria,
        predictions,
        failed={(FAILED_ITEM, 0): _stand_in(criterion, stand_in)},
        kind=kind,
    )

    metrics = compute_metrics(result, dataset, na_mode=na_mode)

    cm = _multi_choice(metrics, 0)
    assert (cm.n_samples, cm.exact_accuracy, cm.n_options) == (3, 1.0, 3)
    assert cm.confusion_matrix.labels == _labels(criterion)
    assert cm.confusion_matrix.matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert len(cm.per_option) == 3
    assert metrics.n_samples == 7
    na = metrics.na_stats
    assert na is not None
    assert (na.na_count_pred, na.na_count_true, na.na_false_positive, na.na_false_negative) == (
        0,
        0,
        0,
        0,
    )


def test_a_genuine_na_answer_still_gives_the_criterion_its_na_option():
    """The judge genuinely picked the auto-injected NA option on item 2; its call on item 3
    failed with the same stand-in. The genuine answer counts as an NA prediction (and needs
    the NA option); the failed one counts nowhere."""
    criteria = [QUALITY, CITES]
    truth, predictions = _mc_rows(QUALITY)
    na_index = len(_options(QUALITY))
    predictions[2][0] = na_index
    dataset = _dataset(criteria, truth)
    result = _run(dataset, criteria, predictions, failed={(FAILED_ITEM, 0): (na_index, PARSE)})

    excluded = compute_metrics(result, dataset)
    as_unmet = compute_metrics(result, dataset, na_mode="as_unmet")

    na_label = _labels(QUALITY.with_guaranteed_na_option())[-1]
    for metrics in (excluded, as_unmet):
        cm = _multi_choice(metrics, 0)
        assert cm.confusion_matrix.labels == [*_labels(QUALITY), na_label]
        na = metrics.na_stats
        assert na is not None
        assert (na.na_count_pred, na.na_false_positive) == (1, 1)
    assert _multi_choice(excluded, 0).n_samples == 2
    # The genuine NA reads as the worst option (Poor) against a true Good.
    assert _multi_choice(as_unmet, 0).n_samples == 3
    assert _multi_choice(as_unmet, 0).exact_accuracy == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="as_category"):
        compute_metrics(result, dataset, na_mode="as_category")


def test_a_judges_failed_vote_gives_the_criterion_no_na_option():
    """A two-judge panel on a forced-choice criterion: judge "b"'s call on item 1 failed
    with no option to abstain with, while judge "a" judged the item, so the criterion did
    not fail. The failed vote is out of "b"'s metrics, and it adds no NA option to the
    criterion either."""
    criteria = [QUALITY, CITES]
    truth, predictions = _mc_rows(QUALITY)
    dataset = _dataset(criteria, truth)
    items = []
    for idx, row in enumerate(predictions):
        b_quality: tuple[str, Prediction, str | None] = (
            ("b", None, INFRA) if idx == 1 else ("b", row[0], None)
        )
        cells = [
            (QUALITY, [("a", row[0], None), b_quality]),
            (CITES, [("a", row[1], None), ("b", row[1], None)]),
        ]
        items.append(_item(dataset, idx, cells))
    result = _eval_result(items)

    for na_mode in NA_MODES:
        metrics = compute_metrics(result, dataset, na_mode=na_mode, per_judge=True)
        cm = _multi_choice(metrics, 0)
        assert (cm.n_samples, cm.exact_accuracy) == (4, 1.0)
        assert cm.confusion_matrix.labels == _labels(QUALITY)
        assert metrics.na_stats is not None and metrics.na_stats.na_count_pred == 0
        assert metrics.per_judge is not None
        assert metrics.per_judge["b"].criterion_accuracy == 1.0


# =============================================================================
# Coverage: a failed cell is an errored pair
# =============================================================================


def test_coverage_counts_a_failed_cell_as_an_errored_pair():
    """Five items with ground truth: item 4's grading raised, the judge's call for c0 on
    item 3 (true CANNOT_ASSESS) failed, and it genuinely answered CANNOT_ASSESS on item 2's
    c1. The failed pair is an errored pair, on neither side an abstention."""
    truth = [[MET, MET], [UNMET, UNMET], [MET, UNMET], [CA, MET], [MET, MET]]
    predictions = [[MET, MET], [UNMET, UNMET], [MET, CA], [CA, MET]]
    dataset = _dataset(BINARY, truth)
    graded = _run(dataset, BINARY, predictions, failed={(3, 0): (CA, INFRA)})
    result = _eval_result([*graded.item_results, _grading_raised(dataset, 4)])

    metrics = compute_metrics(result, dataset)

    c0 = _binary(metrics, 0).coverage_stats
    c1 = _binary(metrics, 1).coverage_stats
    total = metrics.coverage_stats
    assert c0 is not None and c1 is not None and total is not None
    assert (c0.n_total, c0.n_covered, c0.n_errored) == (5, 3, 2)
    assert (c0.coverage, c0.error_rate, c0.union_exclusion_rate) == pytest.approx((0.6, 0.4, 0.4))
    assert (c0.judge_abstain_rate, c0.gt_abstain_rate) == (0.0, 0.0)
    assert (c1.n_total, c1.n_covered, c1.n_errored) == (5, 3, 1)
    assert (c1.coverage, c1.error_rate, c1.judge_abstain_rate) == pytest.approx((0.6, 0.2, 0.2))
    assert c1.gt_abstain_rate == 0.0
    # The aggregate pools the pairs: n_errored sums the criteria's, as n_total does.
    assert (total.n_total, total.n_covered, total.n_errored) == (10, 6, 3)
    assert (total.coverage, total.error_rate, total.union_exclusion_rate) == pytest.approx(
        (0.6, 0.3, 0.4)
    )
    assert (total.judge_abstain_rate, total.gt_abstain_rate) == pytest.approx((0.1, 0.0))


def test_coverage_of_a_run_whose_only_errors_are_errored_items():
    """The aggregate n_errored counts pairs, n_criteria per errored item; the error rate is
    the fraction of items that errored, as before."""
    truth = [[MET, MET], [UNMET, UNMET], [MET, UNMET], [MET, MET]]
    dataset = _dataset(BINARY, truth)
    graded = _run(dataset, BINARY, truth[:3])
    result = _eval_result([*graded.item_results, _grading_raised(dataset, 3)])

    metrics = compute_metrics(result, dataset)

    for cm in metrics.per_criterion:
        assert cm.coverage_stats is not None
        assert (cm.coverage_stats.n_total, cm.coverage_stats.n_errored) == (4, 1)
        assert cm.coverage_stats.error_rate == 0.25
    total = metrics.coverage_stats
    assert total is not None
    assert (total.n_total, total.n_covered, total.n_errored) == (8, 6, 2)
    assert total.error_rate == 0.25


# =============================================================================
# Bootstrap
# =============================================================================

BOOTSTRAP_TRUTH = [
    [MET, MET],
    [UNMET, MET],
    [MET, UNMET],
    [UNMET, UNMET],
    [MET, MET],
    [UNMET, MET],
    [MET, UNMET],
    [UNMET, UNMET],
]


def test_the_bootstrap_leaves_out_failed_cells():
    """Items are resampled as before; each criterion's resampled failed cells are then
    dropped. Every judged cell is right, so every replicate's accuracy and kappa is 1.0,
    whatever the stand-ins, which never reach the bootstrap."""
    dataset = _dataset(BINARY, BOOTSTRAP_TRUTH)
    bootstraps = []
    for stand_in, error in [(UNMET, UNKNOWN), (CA, INFRA)]:
        result = _run(
            dataset,
            BINARY,
            BOOTSTRAP_TRUTH,
            failed={(0, 0): (stand_in, error), (4, 1): (stand_in, error)},
        )
        metrics = compute_metrics(result, dataset, bootstrap=True, n_bootstrap=200, seed=7)
        assert metrics.bootstrap is not None
        assert metrics.bootstrap.accuracy_ci == (1.0, 1.0)
        assert metrics.bootstrap.kappa_ci == (1.0, 1.0)
        bootstraps.append(metrics.bootstrap)
    assert bootstraps[0] == bootstraps[1]


# =============================================================================
# Pooled per-item rubrics
# =============================================================================


def test_the_pooled_path_makes_a_failed_cell_no_point_and_no_abstention():
    """Items with rubrics of their own: the call for item 1's binary criterion failed with
    an unknown error (a wrong UNMET), and the one for item 2's ordinal criterion with a
    parse error (the NA option)."""
    dataset = RubricDataset(prompt="p", rubric=None, name="per-item")
    truth: list[list[CriterionVerdict | str]] = [[MET, "Good"], [MET, "Fair"], [MET, "Poor"]]
    rubrics = []
    for idx, row in enumerate(truth):
        criteria = [
            CITES.model_copy(update={"requirement": f"Cites source {idx}"}),
            QUALITY.model_copy(update={"requirement": f"How good is answer {idx}?"}),
        ]
        rubrics.append(criteria)
        dataset.add_item(
            submission=f"s{idx}",
            description=f"item {idx}",
            ground_truth=row,
            rubric=Rubric(criteria),
        )
    # The judge is right on every cell but the failed ones.
    predictions: list[list[Prediction]] = [
        [MET, _labels(QUALITY).index(str(row[1]))] for row in truth
    ]
    failed: dict[Cell, tuple[Prediction, str]] = {(1, 0): (UNMET, UNKNOWN), (2, 1): (3, PARSE)}
    items = []
    for idx, criteria in enumerate(rubrics):
        cells = []
        for c, criterion in enumerate(criteria):
            prediction, error = failed.get((idx, c), (predictions[idx][c], None))
            cells.append((criterion, [("a", prediction, error)]))
        items.append(_item(dataset, idx, cells))

    metrics = compute_metrics(_eval_result(items), dataset)

    assert metrics.pooled_by_scale is not None
    by_scale = {entry.scale_type: entry for entry in metrics.pooled_by_scale}
    assert (by_scale["binary"].n_points, by_scale["binary"].n_abstain) == (2, 0)
    assert (by_scale["ordinal"].n_points, by_scale["ordinal"].n_abstain) == (2, 0)
    assert by_scale["binary"].exact_accuracy == by_scale["ordinal"].exact_accuracy == 1.0
    assert metrics.criterion_accuracy == 1.0
    assert f"2 {FAILED_CELLS}" in metrics.warnings


# =============================================================================
# Warnings and legacy reports
# =============================================================================


def test_the_warning_counts_the_failed_cells_of_graded_items():
    """Two failed cells on graded items; item 3's every criterion failed, which makes it an
    errored item (counted by the errored-item warning, not this one)."""
    dataset = _dataset(BINARY, BINARY_TRUTH)
    failed: dict[Cell, tuple[Prediction, str]] = {
        (1, 0): (UNMET, UNKNOWN),
        (2, 1): (CA, INFRA),
        (3, 0): (CA, INFRA),
        (3, 1): (CA, INFRA),
    }
    metrics = compute_metrics(_run(dataset, BINARY, BINARY_TRUTH, failed=failed), dataset)

    assert f"2 {FAILED_CELLS}" in metrics.warnings
    assert any(warning.startswith(f"1 {ERRORED_ITEMS}") for warning in metrics.warnings)
    assert metrics.n_items == 3


def test_no_failed_cell_no_warning():
    dataset = _dataset(BINARY, BINARY_TRUTH)
    metrics = compute_metrics(_run(dataset, BINARY, BINARY_TRUTH), dataset)
    assert not any(FAILED_CELLS in warning for warning in metrics.warnings)


def test_a_legacy_dict_shaped_criterion_report_fails_through_its_error_key():
    """A criterion report left as a dict (a hand-built or legacy report) failed when its
    ``error`` is set: on item 2, c1 failed; item 3 failed everywhere, an errored item."""
    dataset = _dataset(BINARY, BINARY_TRUTH)
    items = []
    for idx, truth in enumerate(BINARY_TRUTH):
        crs: list[dict[str, Any]] = [{"verdict": verdict, "error": None} for verdict in truth]
        if idx == 2:
            crs[1] = {"verdict": MET, "error": UNKNOWN}  # wrong: its truth is UNMET
        if idx == 3:
            crs = [{"verdict": CA, "error": INFRA} for _ in truth]
        report = EvaluationReport.model_construct(score=0.5, raw_score=0.5, report=crs)
        items.append(
            ItemResult(item_idx=idx, item=dataset.items[idx], report=report, duration_seconds=0.1)
        )

    metrics = compute_metrics(_eval_result(items), dataset)

    assert metrics.n_items == 3
    assert (metrics.n_samples, metrics.criterion_accuracy) == (5, 1.0)
    assert f"1 {FAILED_CELLS}" in metrics.warnings


# =============================================================================
# Through the grader: an LLM judge whose calls fail for some criteria
# =============================================================================

COUNT_LABELS = ["Neither", "One", "Both"]
GRADER_RUBRIC = Rubric(
    [
        Criterion(name="apples", weight=1.0, requirement="Mentions apples"),
        Criterion(name="pears", weight=1.0, requirement="Mentions pears"),
        Criterion(
            name="count",
            weight=1.0,
            requirement="How many of the two fruits does it name?",
            scale_type="ordinal",
            options=[
                CriterionOption(label=label, value=i / 2) for i, label in enumerate(COUNT_LABELS)
            ],
        ),
    ]
)
# (submission, ground truth). Item 4's call for pears crashes (an unknown error: the worst
# case, a wrong UNMET), item 5's for apples finds the provider down (an infrastructure
# error: CANNOT_ASSESS) and item 6's for count gets an unparseable answer (a parse error:
# the auto-injected NA option).
GRADER_ITEMS: list[tuple[str, list[CriterionVerdict | str]]] = [
    ("apples pears", [MET, MET, "Both"]),
    ("apples", [MET, UNMET, "One"]),
    ("pears", [UNMET, MET, "One"]),
    ("nothing", [UNMET, UNMET, "Neither"]),
    ("apples pears CRASH-PEARS", [MET, MET, "Both"]),
    ("apples DOWN-APPLES", [MET, UNMET, "One"]),
    ("pears GARBLED-COUNT", [UNMET, MET, "One"]),
]


def _grader_dataset() -> RubricDataset:
    dataset = RubricDataset(prompt="Q", rubric=GRADER_RUBRIC, name="failed-calls")
    for submission, truth in GRADER_ITEMS:
        dataset.add_item(submission=submission, description=submission, ground_truth=truth)
    return dataset


def _option_number(user_prompt: str, label: str) -> int:
    """The number the prompt shows ``label`` under (the options may be shuffled)."""
    options = user_prompt.split("<options>")[1].split("</options>")[0].strip().splitlines()
    for line in options:
        number, _, text = line.partition(". ")
        if text == label:
            return int(number)
    raise AssertionError(f"{label!r} is not among the options")


async def _generate(
    self: LLMClient,
    system_prompt: str,
    user_prompt: str,
    response_format: Any = None,
    **kwargs: Any,
) -> GenerateResult:
    """Judge by keyword, failing as ``GRADER_ITEMS`` says."""
    submission = user_prompt.split("<submission>")[-1].split("</submission>")[0]
    named = [fruit for fruit in ("apples", "pears") if fruit in submission]
    parsed: Any
    if "<options>" in user_prompt:
        if "GARBLED-COUNT" in submission:
            raise ValueError("unparseable judge output")
        number = _option_number(user_prompt, COUNT_LABELS[len(named)])
        parsed = MultiChoiceJudgment(selected_option=number, explanation="counted")
    else:
        fruit = "apples" if "Mentions apples" in user_prompt else "pears"
        if f"CRASH-{fruit.upper()}" in submission:
            raise RuntimeError("judge crashed")
        if f"DOWN-{fruit.upper()}" in submission:
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://llm.test"))
        verdict = MET if fruit in named else UNMET
        parsed = CriterionJudgment(criterion_status=verdict, explanation="looked")
    usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return GenerateResult(content="{}", usage=usage, parsed=parsed)


@pytest.mark.asyncio
async def test_a_grader_whose_calls_fail_for_some_criteria(tmp_path: Path):
    grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4o-mini"))
    dataset = _grader_dataset()
    with patch.object(LLMClient, "generate", new=_generate):
        result = await evaluate(dataset, grader, show_progress=False, experiments_dir=tmp_path)

    # Every item was graded; on items 4 to 6 one criterion's call failed.
    assert result.failed_items == 0
    errors = {
        (ir.item_idx, c): str(cr.error).split(":")[0]
        for ir in result.item_results
        for c, cr in enumerate(ir.report.report or [])
        if cr.is_error
    }
    assert errors == {(4, 1): "unknown", (5, 0): "infrastructure", (6, 2): "parse"}

    metrics = compute_metrics(result, dataset, per_judge=True)

    assert metrics.n_items == len(GRADER_ITEMS)
    assert metrics.n_samples == 3 * len(GRADER_ITEMS) - 3
    assert metrics.criterion_accuracy == metrics.macro_accuracy == 1.0
    count = _multi_choice(metrics, 2)
    assert (count.n_samples, count.exact_accuracy) == (6, 1.0)
    # The parse failure's NA stand-in gives the criterion no NA option.
    assert count.confusion_matrix.labels == COUNT_LABELS
    assert metrics.na_stats is not None and metrics.na_stats.na_count_pred == 0
    assert f"3 {FAILED_CELLS}" in metrics.warnings
    assert metrics.per_judge is not None
    (judge,) = metrics.per_judge.values()
    assert judge.criterion_accuracy == metrics.criterion_accuracy
    assert judge.mean_kappa == metrics.mean_kappa == 1.0
    coverage = metrics.coverage_stats
    assert coverage is not None
    assert (coverage.n_total, coverage.n_errored, coverage.judge_abstain_rate) == (21, 3, 0.0)
    for cm in metrics.per_criterion:
        assert cm.coverage_stats is not None and cm.coverage_stats.n_errored == 1

    as_unmet = compute_metrics(result, dataset, cannot_assess="as_unmet", na_mode="as_unmet")
    assert (as_unmet.n_samples, as_unmet.criterion_accuracy) == (18, 1.0)
    assert as_unmet.cannot_assess_stats is not None
    assert as_unmet.cannot_assess_stats.ca_count_pred == 0
