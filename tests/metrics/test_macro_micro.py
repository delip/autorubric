"""Tests for the aggregate macro/micro scalars in ``compute_metrics``.

Covers ``macro_accuracy`` (unweighted mean of per-criterion accuracies), ``micro_kappa``
(pooled Cohen's kappa across criteria), and ``mean_krippendorff_alpha`` (macro mean of
per-criterion alpha). All honour "undefined → None, never a fabricated 0.0", and the
macro/micro split is distinct when criteria have unequal support.
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
    dataset = RubricDataset(prompt="Test prompt", rubric=rubric, name="mm-test")
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


class TestMacroMicroAccuracy:
    def test_macro_equals_mean_of_per_criterion(self):
        criteria = _two_criteria()
        # c0: 1 wrong of 2 → 0.5 ; c1: 2 of 2 correct → 1.0 ; macro = 0.75.
        dataset = _dataset(criteria, [[MET, MET], [MET, UNMET]])
        eval_result = _eval(criteria, [[UNMET, MET], [MET, UNMET]], [0.5, 0.6])
        metrics = compute_metrics(eval_result, dataset)
        per_acc = [cm.accuracy for cm in metrics.per_criterion]
        assert metrics.macro_accuracy == pytest.approx(sum(per_acc) / len(per_acc))
        assert metrics.macro_accuracy == pytest.approx(0.75)

    def test_macro_distinct_from_micro_on_unequal_support(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [MET, UNMET]])
        # c0 item2 prediction CANNOT_ASSESS → excluded under default mode (support 1).
        eval_result = _eval(criteria, [[MET, MET], [CA, MET]], [0.5, 0.6])
        metrics = compute_metrics(eval_result, dataset)
        assert metrics.macro_accuracy is not None
        assert metrics.criterion_accuracy is not None
        # c0 covered=1 (acc 1.0), c1 covered=2 (acc 0.5): macro=0.75, micro pooled=2/3.
        assert metrics.macro_accuracy != pytest.approx(metrics.criterion_accuracy)

    def test_macro_present_on_healthy_run(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, UNMET]])
        metrics = compute_metrics(_eval(criteria, [[MET, UNMET]], [0.5]), dataset)
        assert metrics.macro_accuracy is not None


class TestMicroKappa:
    def test_micro_kappa_present_for_binary(self):
        criteria = _two_criteria()
        dataset = _dataset(
            criteria,
            [[MET, MET], [UNMET, UNMET], [MET, UNMET], [UNMET, MET]],
        )
        eval_result = _eval(
            criteria,
            [[MET, MET], [UNMET, UNMET], [MET, MET], [UNMET, MET]],
            [1.0, 0.0, 0.5, 0.4],
        )
        assert compute_metrics(eval_result, dataset).micro_kappa is not None

    def test_micro_kappa_none_on_single_class(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [MET, MET]])
        eval_result = _eval(criteria, [[MET, MET], [MET, MET]], [1.0, 1.0])
        # Pooled flats are a single class → pooled kappa undefined → None.
        assert compute_metrics(eval_result, dataset).micro_kappa is None


class TestMeanKrippendorffAlpha:
    def test_none_without_ensemble(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, UNMET], [UNMET, MET]])
        eval_result = _eval(criteria, [[MET, UNMET], [UNMET, MET]], [0.5, 0.4])
        # Single-judge (no ensemble) → no inter-judge alpha → mean is None.
        assert compute_metrics(eval_result, dataset).mean_krippendorff_alpha is None
