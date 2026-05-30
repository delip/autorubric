"""Tests for the Matthews correlation coefficient (phi) in the metrics compute layer.

Covers the ``_mcc_or_none`` helper (constant/single-class → None, never sklearn's
misleading 0.0), the per-criterion binary phi / FPR / FNR / 2x2 confusion matrix wired
into ``compute_metrics``, and the aggregate (micro) ``criterion_phi``. The framework's
"undefined → None, never a fabricated 0.0" invariant is checked on the degenerate paths.
"""

import pytest

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import ConfusionMatrix, compute_metrics
from autorubric.metrics._compute import _kappa_or_none, _mcc_or_none
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    CriterionOption,
    CriterionReport,
    CriterionVerdict,
    EvaluationReport,
    MultiChoiceVerdict,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET


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


def _binary_dataset(criteria, ground_truths):
    rubric = Rubric(criteria)
    dataset = RubricDataset(prompt="Test prompt", rubric=rubric, name="phi-test")
    for idx, gt in enumerate(ground_truths):
        dataset.add_item(submission=f"Response {idx}", description=f"Item {idx}", ground_truth=gt)
    return dataset


def _binary_eval(criteria, predictions, scores):
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


def _one_criterion():
    return [Criterion(name="c0", requirement="r0", weight=1.0)]


class TestMccOrNone:
    def test_constant_single_class_is_none_not_zero(self):
        # sklearn's matthews_corrcoef returns a misleading 0.0 on constant input; the helper
        # must return None (genuinely undefined), never that 0.0. Constant on EITHER side is
        # degenerate (no variation → no correlation to estimate).
        assert _mcc_or_none([1, 1, 1], [1, 1, 1]) is None
        assert _mcc_or_none([0, 0, 0], [0, 0, 0]) is None
        assert _mcc_or_none([1, 1, 1], [1, 0, 1]) is None  # constant true side
        assert _mcc_or_none([1, 0, 1], [1, 1, 1]) is None  # constant pred side

    def test_matches_sklearn_on_two_class(self):
        from sklearn.metrics import matthews_corrcoef

        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1]
        assert _mcc_or_none(y_true, y_pred) == pytest.approx(matthews_corrcoef(y_true, y_pred))

    def test_phi_equals_kappa_on_matched_marginals(self):
        # Balanced 2x2 ([[30,10],[10,30]]) → equal row/col marginals → phi == kappa.
        y_true = [1] * 30 + [1] * 10 + [0] * 10 + [0] * 30
        y_pred = [1] * 30 + [0] * 10 + [1] * 10 + [0] * 30
        phi = _mcc_or_none(y_true, y_pred)
        kappa = _kappa_or_none([str(v) for v in y_true], [str(v) for v in y_pred])
        assert phi is not None and kappa is not None
        assert phi == pytest.approx(kappa)
        assert phi == pytest.approx(0.5)

    def test_paper_example_phi_exceeds_kappa(self):
        # 2x2 = [[40,10],[20,30]] (rows true MET/UNMET, cols pred): kappa≈0.400, phi≈0.408.
        y_true = [1] * 40 + [1] * 10 + [0] * 20 + [0] * 30
        y_pred = [1] * 40 + [0] * 10 + [1] * 20 + [0] * 30
        phi = _mcc_or_none(y_true, y_pred)
        kappa = _kappa_or_none([str(v) for v in y_true], [str(v) for v in y_pred])
        assert kappa == pytest.approx(0.400, abs=1e-3)
        assert phi == pytest.approx(0.408, abs=1e-3)
        assert phi > kappa


class TestPerCriterionBinaryConfusionAndRates:
    # tp=2, fn=1, fp=1, tn=2 on a single criterion.
    GTS = [[MET], [MET], [MET], [UNMET], [UNMET], [UNMET]]
    PREDS = [[MET], [MET], [UNMET], [MET], [UNMET], [UNMET]]
    SCORES = [1.0, 0.9, 0.4, 0.6, 0.1, 0.0]

    def _metrics(self):
        criteria = _one_criterion()
        dataset = _binary_dataset(criteria, self.GTS)
        eval_result = _binary_eval(criteria, self.PREDS, self.SCORES)
        return compute_metrics(eval_result, dataset)

    def test_binary_2x2_counts_and_supports(self):
        cmet = self._metrics().per_criterion[0]
        cm = cmet.confusion_matrix
        assert isinstance(cm, ConfusionMatrix)
        assert cm.labels == ["MET", "UNMET"]
        assert (cm.tp, cm.fn, cm.fp, cm.tn) == (2, 1, 1, 2)
        assert cm.tp + cm.fn == cmet.support_true
        assert cm.tp + cm.fp == cmet.support_pred

    def test_fpr_fnr_values(self):
        cmet = self._metrics().per_criterion[0]
        # fpr = fp/(fp+tn) = 1/3 ; fnr = fn/(fn+tp) = 1/3
        assert cmet.fpr == pytest.approx(1 / 3)
        assert cmet.fnr == pytest.approx(1 / 3)

    def test_per_criterion_phi_present(self):
        assert self._metrics().per_criterion[0].phi is not None


class TestDegenerateAndNoNegatives:
    def test_fpr_none_when_no_true_negatives(self):
        # All ground truth MET → no actual negatives → FPR undefined (None); FNR defined.
        criteria = _one_criterion()
        dataset = _binary_dataset(criteria, [[MET], [MET], [MET]])
        eval_result = _binary_eval(criteria, [[MET], [UNMET], [MET]], [1.0, 0.5, 0.9])
        cmet = compute_metrics(eval_result, dataset).per_criterion[0]
        assert cmet.fpr is None
        assert cmet.fnr is not None

    def test_per_criterion_phi_none_on_single_class(self):
        criteria = _one_criterion()
        dataset = _binary_dataset(criteria, [[MET], [MET], [MET]])
        eval_result = _binary_eval(criteria, [[MET], [MET], [MET]], [1.0, 1.0, 1.0])
        assert compute_metrics(eval_result, dataset).per_criterion[0].phi is None


class TestAggregateCriterionPhi:
    def test_micro_phi_present_and_matches_single_criterion(self):
        criteria = _one_criterion()
        gts = [[MET], [MET], [MET], [UNMET], [UNMET], [UNMET]]
        preds = [[MET], [MET], [UNMET], [MET], [UNMET], [UNMET]]
        dataset = _binary_dataset(criteria, gts)
        eval_result = _binary_eval(criteria, preds, [1.0, 0.9, 0.4, 0.6, 0.1, 0.0])
        metrics = compute_metrics(eval_result, dataset)
        assert metrics.criterion_phi is not None
        # Single binary criterion → aggregate (micro) phi == that criterion's phi.
        assert metrics.criterion_phi == pytest.approx(metrics.per_criterion[0].phi)

    def test_micro_phi_none_for_multi_choice_only(self):
        criterion = Criterion(
            name="q",
            requirement="quality",
            options=[
                CriterionOption(label="poor", value=0.0),
                CriterionOption(label="good", value=1.0),
            ],
        )
        rubric = Rubric([criterion])
        dataset = RubricDataset(prompt="p", rubric=rubric, name="mc")
        dataset.add_item(submission="r1", description="i1", ground_truth=["good"])
        dataset.add_item(submission="r2", description="i2", ground_truth=["poor"])
        item_results = []
        for idx, (lbl, lidx, val) in enumerate([("good", 1, 1.0), ("poor", 0, 0.0)]):
            cr = CriterionReport(
                requirement="quality",
                weight=1.0,
                verdict=UNMET,
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=lidx, selected_label=lbl, value=val
                ),
                reason="Test",
            )
            item_results.append(
                ItemResult(
                    item_idx=idx,
                    item=None,
                    report=EvaluationReport(score=float(idx), raw_score=float(idx), report=[cr]),
                    duration_seconds=0.5,
                )
            )
        metrics = compute_metrics(_wrap_eval(item_results), dataset)
        assert metrics.criterion_phi is None
