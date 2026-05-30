"""Tests that EvalResult.compute_metrics accepts the full CANNOT_ASSESS mode set.

The ``EvalResult.compute_metrics`` wrapper's ``cannot_assess`` parameter is widened from
the old two-value literal to the full mode set, so ``"as_category"`` is now accepted and
threaded through to the metrics layer (and recorded on the result).
"""

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    CriterionReport,
    CriterionVerdict,
    EvaluationReport,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET


def _two_criteria():
    return [
        Criterion(name="c0", requirement="r0", weight=1.0),
        Criterion(name="c1", requirement="r1", weight=1.0),
    ]


def _wrap_eval(item_results):
    """Build a minimal EvalResult wrapping the given item results."""
    return EvalResult(
        item_results=item_results,
        total_items=len(item_results),
        successful_items=len(item_results),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=None,
        started_at=None,
        completed_at=None,
    )


def _dataset(criteria, ground_truths):
    rubric = Rubric(criteria)
    dataset = RubricDataset(prompt="Test prompt", rubric=rubric, name="modes-test")
    for idx, gt in enumerate(ground_truths):
        dataset.add_item(submission=f"Response {idx}", description=f"Item {idx}", ground_truth=gt)
    return dataset


def _eval(criteria, predictions, scores):
    item_results = []
    for idx, preds in enumerate(predictions):
        reports = [
            CriterionReport(
                requirement=criteria[c].requirement or criteria[c].name or "req",
                weight=criteria[c].weight,
                verdict=v,
                reason="Test",
            )
            for c, v in enumerate(preds)
        ]
        item_results.append(
            ItemResult(
                item_idx=idx,
                item=None,
                report=EvaluationReport(score=scores[idx], raw_score=scores[idx], report=reports),
                duration_seconds=0.5,
            )
        )
    return _wrap_eval(item_results)


class TestEvalComputeMetricsModes:
    def test_eval_wrapper_accepts_as_category(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [UNMET, UNMET]])
        eval_result = _eval(criteria, [[MET, MET], [UNMET, UNMET]], [1.0, 0.0])
        metrics = eval_result.compute_metrics(dataset, cannot_assess="as_category")
        assert metrics.cannot_assess_mode == "as_category"

    def test_eval_wrapper_records_default_modes(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [UNMET, UNMET]])
        eval_result = _eval(criteria, [[MET, MET], [UNMET, UNMET]], [1.0, 0.0])
        metrics = eval_result.compute_metrics(dataset)
        assert metrics.cannot_assess_mode == "exclude"
        assert metrics.na_mode == "exclude"
