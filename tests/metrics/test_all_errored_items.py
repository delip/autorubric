"""An item on which every judgment failed is scored like any other item.

When every judge's judgment of every criterion of an item failed (each criterion report
carries an ``error``), the grader still builds a report: infrastructure and parse failures
abstain and, under the default ``SKIP`` strategy, leave nothing to score, so the report's
``score`` is the scoring core's empty-denominator ``0.0``; unknown failures take the
conservative worst case. Only an item whose grading *raised* (``ItemResult.error``) is an
errored item. ``compute_metrics`` has always scored the other kind like any item, for either
judge kind (a decision model's failed request builds the same report as an LLM grader whose
every call failed), and it keeps doing so: its verdicts and score enter the metrics, and it
is not counted in ``CoverageStats.n_errored``. The same holds per judge: a judge whose every
vote on an item failed keeps its report's ``judge_scores`` entry for that item.

The last section runs real graders whose LLM calls fail (mocked, nothing reaches the
network) and pins the resulting metrics.
"""

from __future__ import annotations

import math
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


def _dataset() -> RubricDataset:
    dataset = RubricDataset(prompt="p", rubric=Rubric(CRITERIA), name="all-errored")
    for idx, gt in enumerate(GROUND_TRUTH):
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


def _every_judgment_failed(idx: int, error: str = INFRA) -> ItemResult:
    """What the grader builds when every call (or the one request) of the item failed."""
    verdict = CA if error.startswith("infrastructure") else UNMET
    item = _ensemble_item(idx, [[_vote("a", verdict, error)] for _ in range(2)])
    # SKIP leaves nothing to score (infrastructure) or the worst case scores 0 (unknown).
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


@pytest.mark.parametrize("error", [INFRA, UNKNOWN])
def test_an_item_with_every_judgment_failed_is_scored_not_errored(error):
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    failed = compute_metrics(
        _eval_result([*judged, _every_judgment_failed(FAILED_ITEM, error)]), _dataset()
    )
    raised = compute_metrics(_eval_result([*judged, _grading_raised(FAILED_ITEM)]), _dataset())

    assert failed.n_items == FAILED_ITEM + 1
    assert failed.coverage_stats is not None and failed.coverage_stats.n_errored == 0
    assert not _errored_item_warnings(failed)
    # Its 0.0 is paired with its true score of 1.0, as for any scored item.
    assert failed.score_rmse == pytest.approx(math.sqrt(1.0 / 4))
    assert failed.bias is not None and failed.bias.mean_bias == pytest.approx(-1.0 / 4)
    # Only an item whose grading raised is an errored item.
    assert raised.n_items == FAILED_ITEM
    assert raised.coverage_stats is not None and raised.coverage_stats.n_errored == 1
    assert _errored_item_warnings(raised)


def test_the_verdicts_of_an_item_with_every_judgment_failed_are_counted():
    """Its error-abstentions are handled like any abstention: excluded under the default
    ``cannot_assess="exclude"``, counted as UNMET (and as abstention predictions) under
    ``"as_unmet"``, exactly as on an item with only some criteria errored."""
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    result = _eval_result([*judged, _every_judgment_failed(FAILED_ITEM, INFRA)])

    excluded = compute_metrics(result, _dataset())
    as_unmet = compute_metrics(result, _dataset(), cannot_assess="as_unmet")

    assert excluded.criterion_accuracy == pytest.approx(1.0)
    assert as_unmet.n_samples == 2 * (FAILED_ITEM + 1)
    assert as_unmet.criterion_accuracy == pytest.approx(6 / 8)  # its two METs read as UNMET
    assert as_unmet.cannot_assess_stats is not None
    assert as_unmet.cannot_assess_stats.ca_count_pred == 2


def test_a_single_judge_report_with_every_criterion_errored_is_scored():
    """Whatever the report type: here a single-judge ``EvaluationReport``."""
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

    assert failed.n_items == FAILED_ITEM + 1
    assert failed.coverage_stats is not None and failed.coverage_stats.n_errored == 0
    assert failed.score_rmse == pytest.approx(math.sqrt(1.0 / 4))


def test_an_item_with_some_criteria_errored_is_still_scored():
    judged = [_judged(i) for i in range(FAILED_ITEM)]
    partial = _ensemble_item(
        FAILED_ITEM, [[_vote("a", MET)], [_vote("a", CA, INFRA)]]
    )  # c0 judged, c1 abstained on a failure
    metrics = compute_metrics(_eval_result([*judged, partial]), _dataset())

    assert metrics.n_items == 4
    assert metrics.coverage_stats is not None and metrics.coverage_stats.n_errored == 0


def test_every_item_with_every_judgment_failed_still_computes_metrics():
    """No item was judged, yet every item was scored: the metrics are computed, not refused."""
    items = [_every_judgment_failed(i, INFRA) for i in range(len(GROUND_TRUTH))]
    metrics = compute_metrics(_eval_result(items), _dataset())

    assert metrics.n_items == len(GROUND_TRUTH)
    assert metrics.n_samples == 0  # every pair is an abstention, excluded
    assert metrics.score_rmse == pytest.approx(math.sqrt((1 + 0.25 + 0 + 1) / 4))


def test_a_judge_that_failed_a_whole_item_keeps_its_score_for_it():
    """The item stays (judge "b" judged it), and judge "a" keeps its report's
    ``judge_scores`` entry for it (its abstentions' 0.0) in its score pairs."""
    items = [_judged(i, ("a", "b")) for i in range(FAILED_ITEM)]
    items.append(
        _ensemble_item(
            FAILED_ITEM,
            [[_vote("a", CA, INFRA), _vote("b", GROUND_TRUTH[FAILED_ITEM][c])] for c in range(2)],
        )
    )
    # The report's per-judge score for "a" is its abstentions' 0.0; ground truth is 1.0.
    items[-1].report = items[-1].report.model_copy(
        update={"judge_scores": {"a": 0.0, "b": SCORES[FAILED_ITEM]}}
    )
    metrics = compute_metrics(_eval_result(items), _dataset(), per_judge=True)

    assert metrics.n_items == 4
    assert metrics.coverage_stats is not None and metrics.coverage_stats.n_errored == 0
    assert metrics.per_judge is not None
    judge_a, judge_b = metrics.per_judge["a"], metrics.per_judge["b"]
    assert judge_a.score_rmse == pytest.approx(math.sqrt(1.0 / 4))
    assert judge_a.score_pearson is not None and judge_a.score_pearson.n_samples == 4
    assert judge_b.score_rmse == pytest.approx(0.0)
    assert judge_b.score_pearson is not None and judge_b.score_pearson.n_samples == 4


def test_per_item_rubrics_score_an_item_with_every_judgment_failed():
    """The pooled per-item-rubric path scores it too, and skips only an item that raised."""
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
        _eval_result([*judged, _every_judgment_failed(FAILED_ITEM, UNKNOWN)]), dataset
    )
    raised = compute_metrics(_eval_result([*judged, _grading_raised(FAILED_ITEM)]), dataset)

    assert failed.n_items == FAILED_ITEM + 1
    assert raised.n_items == FAILED_ITEM
    # Its worst-case UNMETs (ground truth MET) enter the pooled binary accuracy.
    (failed_binary,) = failed.pooled_by_scale
    (raised_binary,) = raised.pooled_by_scale
    assert failed_binary.n_points == raised_binary.n_points + 2
    assert failed_binary.exact_accuracy == pytest.approx(6 / 8)


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


async def _evaluate_llm(grader: CriterionGrader, generate: Any) -> EvalResult:
    with patch.object(LLMClient, "generate", new=generate):
        return await evaluate(_llm_dataset(), grader, show_progress=False)


@pytest.mark.asyncio
async def test_llm_grader_item_whose_calls_all_failed_is_scored():
    grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4o-mini"))
    result = await _evaluate_llm(grader, _fake_generate(set()))

    failed = result.item_results[4]
    assert failed.error is None and failed.report.score == 0.0
    assert all(cr.error.startswith("infrastructure: ") for cr in failed.report.report)

    metrics = compute_metrics(result, _llm_dataset())
    assert metrics.n_items == 5
    assert metrics.coverage_stats is not None
    assert (metrics.coverage_stats.n_total, metrics.coverage_stats.n_errored) == (10, 0)
    assert not _errored_item_warnings(metrics)
    assert metrics.score_rmse == pytest.approx(math.sqrt(0.5**2 / 5))
    assert metrics.criterion_accuracy == pytest.approx(1.0)

    as_unmet = compute_metrics(result, _llm_dataset(), cannot_assess="as_unmet")
    assert (as_unmet.n_samples, as_unmet.criterion_accuracy) == (10, pytest.approx(0.9))
    assert as_unmet.cannot_assess_stats is not None
    assert as_unmet.cannot_assess_stats.ca_count_pred == 2


@pytest.mark.asyncio
async def test_llm_grader_whose_every_call_failed_still_gets_metrics():
    grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4o-mini"))
    result = await _evaluate_llm(grader, _fake_generate(set(), fail_all=True))

    metrics = compute_metrics(result, _llm_dataset())

    assert metrics.n_items == 5
    assert metrics.n_samples == 0
    true_scores = [1.0, 0.5, 0.5, 0.0, 0.5]
    assert metrics.score_rmse == pytest.approx(
        math.sqrt(sum(t**2 for t in true_scores) / len(true_scores))
    )


@pytest.mark.asyncio
async def test_llm_judge_whose_every_call_failed_keeps_float_score_metrics():
    """An LLM ensemble where one provider is down: that judge's per-judge score metrics are
    computed from its report's ``judge_scores`` entries, as floats, so the documented
    ``f"{judge_id}: RMSE={jm.score_rmse:.4f}"`` keeps working."""
    grader = CriterionGrader(
        judges=[
            JudgeSpec(LLMConfig(model="openai/gpt-4o-mini"), "A"),
            JudgeSpec(LLMConfig(model="anthropic/claude-x"), "B"),
        ]
    )
    result = await _evaluate_llm(grader, _fake_generate({"anthropic/claude-x"}))

    metrics = compute_metrics(result, _llm_dataset(), per_judge=True)

    assert metrics.per_judge is not None
    judge_b = metrics.per_judge["B"]
    # B's entry is 0.0 on every item; the true scores are 1.0, 0.5, 0.5, 0.0 and 0.5.
    assert judge_b.score_rmse == pytest.approx(math.sqrt((1 + 0.25 + 0.25 + 0 + 0.25) / 5))
    assert judge_b.bias is not None and judge_b.bias.mean_bias == pytest.approx(-0.5)
    lines = [f"{jid}: RMSE={jm.score_rmse:.4f}" for jid, jm in metrics.per_judge.items()]
    assert "B: RMSE=0.5916" in lines
