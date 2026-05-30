"""Tests for the diagnostic warnings appended by ``compute_metrics``.

Covers the score-collapse warning (the ground-truth score array has <=2 distinct values,
so score-level correlations are uninformative) and the degeneracy warning (a criterion had
samples but its agreement coefficient collapsed to None).
"""

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import compute_metrics
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    CriterionReport,
    CriterionVerdict,
    EvaluationReport,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET


def _one_criterion():
    return [Criterion(name="c0", requirement="r0", weight=1.0)]


def _three_criteria():
    return [
        Criterion(name="c0", requirement="r0", weight=1.0),
        Criterion(name="c1", requirement="r1", weight=2.0),
        Criterion(name="c2", requirement="r2", weight=3.0),
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
    dataset = RubricDataset(prompt="Test prompt", rubric=rubric, name="warn-test")
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


class TestScoreCollapseWarning:
    def test_warns_when_scores_collapse(self):
        # Both ground-truth verdict patterns are identical → ground-truth score takes a single
        # distinct value → score range collapsed → warning.
        criteria = _one_criterion()
        dataset = _dataset(criteria, [[MET], [MET]])
        eval_result = _eval(criteria, [[MET], [UNMET]], [0.5, 0.6])
        metrics = compute_metrics(eval_result, dataset)
        assert any("distinct" in w.lower() and "score" in w.lower() for w in metrics.warnings)

    def test_no_collapse_warning_with_spread_ground_truth_scores(self):
        # Three criteria with distinct weights → distinct ground-truth verdict mixes give >2
        # distinct ground-truth scores, so no collapse warning.
        criteria = _three_criteria()
        gts = [
            [MET, MET, MET],
            [MET, MET, UNMET],
            [MET, UNMET, UNMET],
            [UNMET, UNMET, UNMET],
        ]
        preds = gts
        dataset = _dataset(criteria, gts)
        eval_result = _eval(criteria, preds, [1.0, 0.5, 0.3, 0.0])
        metrics = compute_metrics(eval_result, dataset)
        assert not any("distinct" in w.lower() for w in metrics.warnings)


class TestDegeneracyWarning:
    def test_warns_on_degenerate_criterion(self):
        # All MET → single class → kappa None with data → degenerate → warning.
        criteria = _one_criterion()
        dataset = _dataset(criteria, [[MET], [MET]])
        eval_result = _eval(criteria, [[MET], [MET]], [1.0, 0.9])
        metrics = compute_metrics(eval_result, dataset)
        assert metrics.per_criterion[0].is_degenerate is True
        assert any("degenerat" in w.lower() for w in metrics.warnings)

    def test_no_degeneracy_warning_on_healthy_criterion(self):
        criteria = _one_criterion()
        dataset = _dataset(criteria, [[MET], [UNMET], [MET], [UNMET]])
        eval_result = _eval(criteria, [[MET], [UNMET], [UNMET], [UNMET]], [1.0, 0.0, 0.5, 0.2])
        metrics = compute_metrics(eval_result, dataset)
        assert metrics.per_criterion[0].is_degenerate is False
        assert not any("degenerat" in w.lower() for w in metrics.warnings)
