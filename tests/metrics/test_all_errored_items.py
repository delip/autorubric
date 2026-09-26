"""An item nothing judged is an errored item; an item with nothing left to score has no score.

When every judge's judgment of every criterion of an item failed (each criterion report
carries an ``error``), nothing was judged. The grader's report then has no score (``score``,
``raw_score`` and ``llm_raw_score`` are ``None``) and carries an ``error``, as any failed
grade does; ``EvalRunner`` records that error on the item; and ``compute_metrics`` treats
the item as it treats one whose grading raised: it is left out of every metric, counted in
``CoverageStats.n_errored`` and warned about (#18). ``compute_metrics`` also recognizes such
an item by its criterion errors alone, so a run saved before these reports carried an
``error`` (with a fabricated score of 0.0) gets the same metrics. The same holds per judge:
a judge whose every vote on an item failed has no score for it.

An item with nothing left to score (every criterion abstained under ``SKIP`` and nothing
failed) has no score either, but it is no errored item: its verdicts count, and only the
score-level metrics leave it out, as they leave out an item whose ground truth leaves
nothing to score.

The last section runs real graders whose LLM calls fail (mocked, nothing reaches the
network) and pins the resulting reports and metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest

from autorubric import LLMConfig, evaluate
from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMClient
from autorubric.metrics import compute_metrics
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    CriterionJudgment,
    CriterionReport,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EvaluationReport,
    JudgeVote,
    TokenUsage,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CA = CriterionVerdict.CANNOT_ASSESS

CRITERIA = [
    Criterion(name="c0", requirement="r0", weight=1.0),
    Criterion(name="c1", requirement="r1", weight=1.0),
]
GROUND_TRUTH = [[MET, MET], [MET, UNMET], [UNMET, UNMET], [MET, MET]]
# Judge "a" matches the ground truth wherever it judges.
SCORES = [1.0, 0.5, 0.0, 1.0]
FAILED_ITEM = 3  # ground truth [MET, MET], a true score of 1.0
INFRA = "infrastructure: 503 overloaded"
UNKNOWN = "unknown: boom"
# How a report of an item nothing judged was saved: by this release (no score, an error) or
# by an earlier one (a fabricated score, no error).
STYLES = ["current", "saved-before"]


def _dataset(ground_truth: list[list[CriterionVerdict | str]] = GROUND_TRUTH) -> RubricDataset:
    dataset = RubricDataset(prompt="p", rubric=Rubric(CRITERIA), name="all-errored")
    for idx, gt in enumerate(ground_truth):
        dataset.add_item(submission=f"s{idx}", description=f"i{idx}", ground_truth=gt)
    return dataset


def _eval_result(items: list[ItemResult]) -> EvalResult:
    failed = sum(1 for item in items if item.error is not None)
    return EvalResult(
        item_results=items,
        total_items=len(items),
        successful_items=len(items) - failed,
        failed_items=failed,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=None,
        started_at=None,
        completed_at=None,
    )


def _vote(judge_id: str, verdict: CriterionVerdict, error: str | None = None) -> JudgeVote:
    reason = f"Judge call failed: {error}" if error else f"{judge_id} ok"
    return JudgeVote(judge_id=judge_id, verdict=verdict, reason=reason, error=error)


def _ensemble_item(idx: int, votes_per_criterion: list[list[JudgeVote]]) -> ItemResult:
    reports = []
    for criterion, votes in zip(CRITERIA, votes_per_criterion, strict=True):
        errors = [v.error for v in votes]
        reports.append(
            EnsembleCriterionReport(
                criterion=criterion,
                final_verdict=votes[0].verdict,
                final_reason=None,
                votes=votes,
                error=" | ".join(e for e in errors if e) if all(errors) else None,
            )
        )
    judge_ids = [v.judge_id for v in votes_per_criterion[0]]
    report = EnsembleEvaluationReport(
        score=SCORES[idx],
        raw_score=SCORES[idx] * 2,
        report=reports,
        judge_scores={jid: SCORES[idx] for jid in judge_ids},
    )
    return ItemResult(item_idx=idx, item=None, report=report, duration_seconds=0.1)


def _judged(idx: int, judge_ids: tuple[str, ...] = ("a",)) -> ItemResult:
    return _ensemble_item(
        idx, [[_vote(jid, GROUND_TRUTH[idx][c]) for jid in judge_ids] for c in range(2)]
    )


def _every_judgment_failed(idx: int, error: str = INFRA, style: str = "current") -> ItemResult:
    """What the grader builds when every call (or the one request) of the item failed."""
    verdict = CA if error.startswith("infrastructure") else UNMET
    item = _ensemble_item(idx, [[_vote("a", verdict, error)] for _ in range(2)])
    if style == "current":
        message = f"Every criterion's judgment failed: {error}"
        item.report = item.report.model_copy(
            update={
                "score": None,
                "raw_score": None,
                "llm_raw_score": None,
                "judge_scores": {"a": None},
                "error": message,
            }
        )
        item.error = message
    else:
        # Before #18: SKIP left nothing to score (infrastructure) or the worst case scored 0
        # (unknown), and neither the report nor the item carried an error.
        item.report = item.report.model_copy(
            update={"score": 0.0, "raw_score": 0.0, "judge_scores": {"a": 0.0}}
        )
    return item


def _grading_raised(idx: int) -> ItemResult:
    """The canonical errored item: its grading raised."""
    report = EnsembleEvaluationReport(score=None, raw_score=None, error="boom")
    return ItemResult(item_idx=idx, item=None, report=report, duration_seconds=0.1, error="boom")


def _errored_item_warnings(metrics) -> list[str]:
    return [w for w in metrics.warnings if "excluded from metrics because grading errored" in w]


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("error", [INFRA, UNKNOWN])
def test_an_item_with_every_judgment_failed_is_an_errored_item(error, style):
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    failed = compute_metrics(
        _eval_result([*judged, _every_judgment_failed(FAILED_ITEM, error, style)]), _dataset()
    )
    raised = compute_metrics(_eval_result([*judged, _grading_raised(FAILED_ITEM)]), _dataset())

    for metrics in (failed, raised):
        assert metrics.n_items == FAILED_ITEM
        # Its pair of each criterion is an errored pair: one item of the four errored.
        assert metrics.coverage_stats is not None
        assert metrics.coverage_stats.n_errored == len(CRITERIA)
        assert metrics.coverage_stats.error_rate == pytest.approx(1 / 4)
        assert _errored_item_warnings(metrics)
        # Only the judged items are paired, and judge "a" matched them all.
        assert metrics.score_rmse == pytest.approx(0.0)
        assert metrics.score_pearson.n_samples == FAILED_ITEM
    assert failed.criterion_accuracy == raised.criterion_accuracy == pytest.approx(1.0)


@pytest.mark.parametrize("style", STYLES)
def test_the_verdicts_of_an_item_with_every_judgment_failed_do_not_count(style):
    """Its stand-in abstentions are no predictions: even ``cannot_assess="as_unmet"``,
    which counts abstentions as UNMET, leaves them out."""
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    result = _eval_result([*judged, _every_judgment_failed(FAILED_ITEM, INFRA, style)])

    as_unmet = compute_metrics(result, _dataset(), cannot_assess="as_unmet")

    assert as_unmet.n_samples == 2 * FAILED_ITEM
    assert as_unmet.criterion_accuracy == pytest.approx(1.0)
    assert as_unmet.cannot_assess_stats is not None
    assert as_unmet.cannot_assess_stats.ca_count_pred == 0


def test_a_single_judge_report_with_every_criterion_errored_is_an_errored_item():
    """Whatever the report type: here a single-judge ``EvaluationReport`` saved with a
    fabricated 0.0 and no error."""
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    single = EvaluationReport(
        score=0.0,
        raw_score=0.0,
        report=[
            CriterionReport(
                **{name: getattr(c, name) for name in Criterion.model_fields},
                verdict=CA,
                reason=f"Judge call failed: {INFRA}",
                error=INFRA,
            )
            for c in CRITERIA
        ],
    )
    failed_item = ItemResult(item_idx=FAILED_ITEM, item=None, report=single, duration_seconds=0.1)
    failed = compute_metrics(_eval_result([*judged, failed_item]), _dataset())

    assert failed.n_items == FAILED_ITEM
    # Its pair of each criterion is an errored pair.
    assert failed.coverage_stats is not None
    assert failed.coverage_stats.n_errored == len(CRITERIA)
    assert failed.score_rmse == pytest.approx(0.0)


def test_an_item_with_some_criteria_errored_is_still_scored():
    """The item counts; only its failed criterion is left out, an errored pair (see
    ``test_failed_criteria.py``), not an errored item."""
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    partial = _ensemble_item(
        FAILED_ITEM, [[_vote("a", MET)], [_vote("a", CA, INFRA)]]
    )  # c0 judged, c1 abstained on a failure
    metrics = compute_metrics(_eval_result([*judged, partial]), _dataset())

    assert metrics.n_items == 4
    assert metrics.coverage_stats is not None and metrics.coverage_stats.n_errored == 1
    assert not _errored_item_warnings(metrics)


@pytest.mark.parametrize("style", STYLES)
def test_no_metrics_when_every_items_every_judgment_failed(style):
    """Nothing was judged, so there is nothing to measure, as when every grading raised."""
    items = [_every_judgment_failed(i, INFRA, style) for i in range(len(GROUND_TRUTH))]
    with pytest.raises(ValueError, match="No valid items with ground truth found"):
        compute_metrics(_eval_result(items), _dataset())


@pytest.mark.parametrize("style", STYLES)
def test_a_judge_that_failed_a_whole_item_has_no_score_for_it(style):
    """The item stays (judge "b" judged it), but judge "a" judged none of it: its score pairs
    leave the item out, whatever its report's ``judge_scores`` entry says."""
    items = [_judged(i, ("a", "b")) for i in range(FAILED_ITEM)]
    items.append(
        _ensemble_item(
            FAILED_ITEM,
            [[_vote("a", CA, INFRA), _vote("b", GROUND_TRUTH[FAILED_ITEM][c])] for c in range(2)],
        )
    )
    entry = None if style == "current" else 0.0
    items[-1].report = items[-1].report.model_copy(
        update={"judge_scores": {"a": entry, "b": SCORES[FAILED_ITEM]}}
    )
    metrics = compute_metrics(_eval_result(items), _dataset(), per_judge=True)

    assert metrics.n_items == 4
    assert metrics.coverage_stats is not None and metrics.coverage_stats.n_errored == 0
    assert metrics.per_judge is not None
    judge_a, judge_b = metrics.per_judge["a"], metrics.per_judge["b"]
    assert judge_a.score_rmse == pytest.approx(0.0)
    assert judge_a.score_pearson is not None and judge_a.score_pearson.n_samples == 3
    assert judge_b.score_rmse == pytest.approx(0.0)
    assert judge_b.score_pearson is not None and judge_b.score_pearson.n_samples == 4


def test_a_panel_judge_down_for_the_whole_run_is_not_an_escalation_judge():
    """A judge with no score on any item is a cascade's escalation judge only when the
    cascade's structure says so; a panel judge whose every call failed is not one."""
    items = []
    for idx in range(len(GROUND_TRUTH)):
        item = _ensemble_item(
            idx, [[_vote("a", GROUND_TRUTH[idx][c]), _vote("b", CA, INFRA)] for c in range(2)]
        )
        item.report = item.report.model_copy(update={"judge_scores": {"a": SCORES[idx], "b": None}})
        items.append(item)

    metrics = compute_metrics(_eval_result(items), _dataset(), per_judge=True)

    assert metrics.per_judge is not None
    judge_b = metrics.per_judge["b"]
    assert (judge_b.coverage, judge_b.n_pairs) == ("full", None)
    assert judge_b.score_rmse is None
    assert metrics.per_judge["a"].score_rmse == pytest.approx(0.0)


def test_per_item_rubrics_leave_out_an_item_with_every_judgment_failed():
    """The pooled per-item-rubric path leaves it out too, like an item that raised."""
    dataset = RubricDataset(prompt="p", rubric=None, name="per-item")
    for idx, gt in enumerate(GROUND_TRUTH):
        criteria = [
            c.model_copy(update={"requirement": f"{c.requirement}-{idx}"}) for c in CRITERIA
        ]
        dataset.add_item(
            submission=f"s{idx}", description=f"i{idx}", ground_truth=gt, rubric=Rubric(criteria)
        )
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    failed = compute_metrics(
        _eval_result([*judged, _every_judgment_failed(FAILED_ITEM, UNKNOWN, "saved-before")]),
        dataset,
    )
    raised = compute_metrics(_eval_result([*judged, _grading_raised(FAILED_ITEM)]), dataset)

    assert failed.n_items == raised.n_items == FAILED_ITEM
    (failed_binary,) = failed.pooled_by_scale
    (raised_binary,) = raised.pooled_by_scale
    assert failed_binary.n_points == raised_binary.n_points
    assert failed_binary.exact_accuracy == pytest.approx(1.0)
    # Left out, as on the per-criterion path: counted and warned about.
    assert _errored_item_warnings(failed) and _errored_item_warnings(raised)

    # With every item errored there is nothing to measure, as on the per-criterion path.
    every_item = [_every_judgment_failed(i, UNKNOWN, "saved-before") for i in range(4)]
    with pytest.raises(ValueError, match="No valid items with ground truth found"):
        compute_metrics(_eval_result(every_item), dataset)


# =============================================================================
# Nothing left to score: no score, but no errored item
# =============================================================================


def test_an_item_with_nothing_left_to_score_counts_only_its_verdicts():
    """Every judge genuinely abstained on every criterion: the report has no score and no
    error. The item counts, its abstentions are predictions like any other, and only the
    score pairs leave it out."""
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    abstained = _ensemble_item(FAILED_ITEM, [[_vote("a", CA)] for _ in range(2)])
    abstained.report = abstained.report.model_copy(
        update={"score": None, "raw_score": None, "judge_scores": {"a": None}}
    )
    result = _eval_result([*judged, abstained])

    metrics = compute_metrics(result, _dataset())
    as_unmet = compute_metrics(result, _dataset(), cannot_assess="as_unmet")

    assert metrics.n_items == 4
    assert metrics.coverage_stats is not None and metrics.coverage_stats.n_errored == 0
    assert not _errored_item_warnings(metrics)
    assert metrics.score_pearson.n_samples == FAILED_ITEM
    assert metrics.score_rmse == pytest.approx(0.0)
    assert as_unmet.n_samples == 2 * 4
    assert as_unmet.cannot_assess_stats is not None
    assert as_unmet.cannot_assess_stats.ca_count_pred == 2


def test_an_item_with_nothing_left_to_score_counts_for_each_judge():
    """Its verdicts count for each judge as they do for the aggregate: under ``as_unmet`` the
    judges' abstentions read as UNMET against a true MET, for the judges too."""
    judged = [_judged(i, ("a", "b")) for i in range(FAILED_ITEM)]
    abstained = _ensemble_item(FAILED_ITEM, [[_vote("a", CA), _vote("b", CA)] for _ in range(2)])
    abstained.report = abstained.report.model_copy(
        update={"score": None, "raw_score": None, "judge_scores": {"a": None, "b": None}}
    )

    metrics = compute_metrics(
        _eval_result([*judged, abstained]), _dataset(), per_judge=True, cannot_assess="as_unmet"
    )

    assert metrics.criterion_accuracy == pytest.approx(6 / 8)
    assert metrics.per_judge is not None
    for judge in metrics.per_judge.values():
        assert judge.criterion_accuracy == pytest.approx(6 / 8)
        assert judge.score_pearson is not None and judge.score_pearson.n_samples == FAILED_ITEM


def test_no_score_pairs_at_all_is_refused_with_its_cause():
    """Every graded item has nothing left to score: the score-level metrics are undefined,
    and ``MetricsResult`` holds none, so ``compute_metrics`` refuses and says why."""
    items = []
    for idx in range(len(GROUND_TRUTH)):
        item = _ensemble_item(idx, [[_vote("a", CA)] for _ in range(2)])
        item.report = item.report.model_copy(
            update={"score": None, "raw_score": None, "judge_scores": {"a": None}}
        )
        items.append(item)
    with pytest.raises(ValueError, match="nothing left to score"):
        compute_metrics(_eval_result(items), _dataset(), cannot_assess="as_unmet")


def test_an_item_whose_ground_truth_leaves_nothing_to_score_leaves_the_score_pairs():
    """Ground truth that abstains on every criterion has no true score under SKIP: the item
    has nothing to pair its score with, for the aggregate or any judge."""
    ground_truth = [*GROUND_TRUTH[:FAILED_ITEM], [CA, CA]]
    items = [_judged(i, ("a", "b")) for i in range(FAILED_ITEM)]
    items.append(
        _ensemble_item(FAILED_ITEM, [[_vote("a", MET), _vote("b", MET)] for _ in range(2)])
    )

    metrics = compute_metrics(_eval_result(items), _dataset(ground_truth), per_judge=True)

    assert metrics.n_items == 4
    assert metrics.score_pearson.n_samples == FAILED_ITEM
    assert metrics.per_judge is not None
    for judge in metrics.per_judge.values():
        assert judge.score_pearson is not None and judge.score_pearson.n_samples == FAILED_ITEM


# =============================================================================
# Through the grader: LLM judges whose calls fail
# =============================================================================

LLM_RUBRIC = Rubric(
    [
        Criterion(weight=1.0, requirement="Mentions apples"),
        Criterion(weight=1.0, requirement="Mentions pears"),
    ]
)
# (submission, ground truth): true scores 1.0, 0.5, 0.5, 0.0 and 0.5.
LLM_ITEMS = [
    ("apples pears", [MET, MET]),
    ("apples", [MET, UNMET]),
    ("pears", [UNMET, MET]),
    ("nothing", [UNMET, UNMET]),
    ("FAIL apples", [MET, UNMET]),
]


def _llm_dataset() -> RubricDataset:
    dataset = RubricDataset(prompt="Q", rubric=LLM_RUBRIC, name="failed-calls")
    for submission, gt in LLM_ITEMS:
        dataset.add_item(submission=submission, description=submission, ground_truth=gt)
    return dataset


def _fake_generate(failing_models: set[str], fail_all: bool = False):
    """An ``LLMClient.generate`` that judges by keyword and fails for chosen models/items."""

    async def generate(self, system_prompt, user_prompt, response_format=None, **kwargs):
        submission = user_prompt.split("<submission>")[-1].split("</submission>")[0]
        if fail_all or self.config.model in failing_models or "FAIL" in submission:
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://llm.test"))
        fruit = "apples" if "Mentions apples" in user_prompt else "pears"
        status = "MET" if fruit in submission else "UNMET"
        return GenerateResult(
            content="",
            parsed=CriterionJudgment(criterion_status=status, explanation="x"),
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=None,
        )

    return generate


async def _evaluate_llm(
    grader: CriterionGrader, generate: Any, experiments_dir: Path, **kwargs: Any
) -> EvalResult:
    with patch.object(LLMClient, "generate", new=generate):
        return await evaluate(
            _llm_dataset(),
            grader,
            show_progress=False,
            experiments_dir=experiments_dir,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_llm_grader_item_whose_calls_all_failed_is_an_errored_item(tmp_path):
    grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4o-mini"))
    result = await _evaluate_llm(grader, _fake_generate(set()), tmp_path)

    failed = result.item_results[4]
    assert failed.report.score is None and failed.report.raw_score is None
    assert failed.report.error is not None
    assert failed.report.error.startswith("Every criterion's judgment failed: infrastructure: ")
    assert failed.error == failed.report.error
    assert all(cr.error.startswith("infrastructure: ") for cr in failed.report.report)
    assert (result.successful_items, result.failed_items) == (4, 1)

    metrics = compute_metrics(result, _llm_dataset())
    assert metrics.n_items == 4
    # Its pair of each criterion is an errored pair.
    assert metrics.coverage_stats is not None
    assert metrics.coverage_stats.n_errored == len(LLM_RUBRIC.rubric)
    assert _errored_item_warnings(metrics)
    assert metrics.score_rmse == pytest.approx(0.0)
    assert metrics.criterion_accuracy == pytest.approx(1.0)

    as_unmet = compute_metrics(result, _llm_dataset(), cannot_assess="as_unmet")
    assert (as_unmet.n_samples, as_unmet.criterion_accuracy) == (8, pytest.approx(1.0))
    assert as_unmet.cannot_assess_stats is not None
    assert as_unmet.cannot_assess_stats.ca_count_pred == 0


@pytest.mark.asyncio
async def test_llm_grader_whose_every_call_failed_has_nothing_to_measure(tmp_path):
    grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4o-mini"))
    result = await _evaluate_llm(grader, _fake_generate(set(), fail_all=True), tmp_path)

    assert result.failed_items == len(LLM_ITEMS)
    with pytest.raises(ValueError, match="No valid items with ground truth found"):
        compute_metrics(result, _llm_dataset())


@pytest.mark.asyncio
async def test_fail_fast_stops_at_an_item_whose_every_judgment_failed(tmp_path):
    grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4o-mini"))
    with pytest.raises(RuntimeError, match="Evaluation failed at item 4"):
        await _evaluate_llm(grader, _fake_generate(set()), tmp_path, fail_fast=True)


@pytest.mark.asyncio
async def test_llm_judge_whose_every_call_failed_has_no_score_metrics(tmp_path):
    """An LLM ensemble where one provider is down: that judge judged nothing, so it has no
    score on any item and no score-level metrics; it is still a panel judge, not an
    escalation judge. The judge that answered decides every item."""
    grader = CriterionGrader(
        judges=[
            JudgeSpec(LLMConfig(model="openai/gpt-4o-mini"), "A"),
            JudgeSpec(LLMConfig(model="anthropic/claude-x"), "B"),
        ]
    )
    result = await _evaluate_llm(grader, _fake_generate({"anthropic/claude-x"}), tmp_path)

    assert result.failed_items == 1  # only the item whose every call failed
    graded = [ir.report for ir in result.item_results if ir.error is None]
    assert all(
        isinstance(report, EnsembleEvaluationReport) and report.judge_scores["B"] is None
        for report in graded
    )
    metrics = compute_metrics(result, _llm_dataset(), per_judge=True)

    assert metrics.per_judge is not None
    judge_a, judge_b = metrics.per_judge["A"], metrics.per_judge["B"]
    assert (judge_b.coverage, judge_b.score_rmse, judge_b.bias) == ("full", None, None)
    assert judge_a.score_rmse == pytest.approx(0.0)
    lines = [
        f"{jid}: RMSE={'n/a' if jm.score_rmse is None else f'{jm.score_rmse:.4f}'}"
        for jid, jm in metrics.per_judge.items()
    ]
    assert lines == ["A: RMSE=0.0000", "B: RMSE=n/a"]
