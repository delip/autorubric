"""Tests for the compute-layer coverage/error wiring in ``compute_metrics``.

Builds per-criterion and aggregate ``CoverageStats`` from the live counts under the
``exclude`` handling mode (raw pre-exclusion denominator, union-exclusion), counts items
lost to a grading error (with a warning), and sets the top-level ``n_samples``. Under
non-``exclude`` modes coverage is trivially 1.0 and is left ``None``.
"""

import pytest

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
CA = CriterionVerdict.CANNOT_ASSESS


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
    dataset = RubricDataset(prompt="Test prompt", rubric=rubric, name="cov-test")
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


class TestNSamples:
    def test_top_level_n_samples_sums_per_criterion(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [UNMET, UNMET]])
        eval_result = _eval(criteria, [[MET, MET], [UNMET, UNMET]], [1.0, 0.0])
        metrics = compute_metrics(eval_result, dataset)
        assert metrics.n_samples == sum(cm.n_samples for cm in metrics.per_criterion)
        assert metrics.n_samples == 4


class TestPerCriterionCoverage:
    def test_coverage_present_and_correct_under_exclude(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [MET, MET]])
        # c0 has one CANNOT_ASSESS prediction → excluded → covered 1 of 2; c1 fully covered.
        eval_result = _eval(criteria, [[CA, MET], [MET, MET]], [1.0, 0.9])
        metrics = compute_metrics(eval_result, dataset)
        cm0 = metrics.per_criterion[0]
        cm1 = metrics.per_criterion[1]
        assert cm0.coverage_stats is not None
        assert cm0.coverage_stats.n_total == 2
        assert cm0.coverage_stats.n_covered == 1
        assert cm0.coverage_stats.coverage == pytest.approx(0.5)
        assert cm0.coverage_stats.union_exclusion_rate == pytest.approx(0.5)
        # The abstaining side was the judge/prediction here.
        assert cm0.coverage_stats.judge_abstain_rate == pytest.approx(0.5)
        assert cm1.coverage_stats is not None
        assert cm1.coverage_stats.coverage == pytest.approx(1.0)

    def test_aggregate_coverage_rollup(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [MET, MET]])
        eval_result = _eval(criteria, [[CA, MET], [MET, MET]], [1.0, 0.9])
        metrics = compute_metrics(eval_result, dataset)
        assert metrics.coverage_stats is not None
        # 2 criteria x 2 items = 4 raw pairs; 3 covered (one CA-excluded).
        assert metrics.coverage_stats.n_total == 4
        assert metrics.coverage_stats.n_covered == 3
        assert metrics.coverage_stats.coverage == pytest.approx(0.75)


class TestCoverageNoneUnderNonExclude:
    def test_coverage_none_under_as_category(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [UNMET, UNMET]])
        eval_result = _eval(criteria, [[MET, MET], [UNMET, UNMET]], [1.0, 0.0])
        metrics = compute_metrics(eval_result, dataset, cannot_assess="as_category")
        assert metrics.coverage_stats is None
        for cm in metrics.per_criterion:
            assert cm.coverage_stats is None


class TestErroredItemCounting:
    def test_errored_item_counted_and_warned(self):
        criteria = _two_criteria()
        # 3 GT-bearing items, one of which is item-level errored.
        dataset = _dataset(criteria, [[MET, MET], [UNMET, UNMET], [MET, UNMET]])
        eval_result = _eval(
            criteria,
            [[MET, MET], [UNMET, UNMET], [MET, UNMET]],
            [1.0, 0.0, 0.5],
        )
        # Force an item-LEVEL error on item index 1 (skipped, counted as errored).
        eval_result.item_results[1].error = "boom"
        metrics = compute_metrics(eval_result, dataset)
        assert metrics.coverage_stats is not None
        assert metrics.coverage_stats.n_errored == 1
        assert metrics.coverage_stats.error_rate is not None
        assert metrics.coverage_stats.error_rate > 0
        assert any("error" in w.lower() and "exclud" in w.lower() for w in metrics.warnings)
