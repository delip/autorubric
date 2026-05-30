"""Tests for the per-judge confusion matrix and phi in ``compute_metrics``.

Each judge gets a 3x3 confusion matrix (binary MET/UNMET with the abstain CANNOT_ASSESS
class last), pooled across all criteria from the RAW pre-filter judge codes, plus a
per-judge Matthews correlation coefficient (phi) on the MET-vs-rest dichotomy. The
single-judge "ensemble" must reproduce the aggregate field-for-field (including phi).
"""

import pytest

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import ConfusionMatrix, compute_metrics
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CA = CriterionVerdict.CANNOT_ASSESS


def _two_criteria():
    return [
        Criterion(name="c0", requirement="r0", weight=1.0),
        Criterion(name="c1", requirement="r1", weight=1.0),
    ]


def _dataset(criteria, ground_truths):
    rubric = Rubric(criteria)
    dataset = RubricDataset(prompt="Test prompt", rubric=rubric, name="pj-test")
    for idx, gt in enumerate(ground_truths):
        dataset.add_item(submission=f"Response {idx}", description=f"Item {idx}", ground_truth=gt)
    return dataset


def _ensemble_item(criteria, verdicts_per_judge, judge_scores, score):
    """verdicts_per_judge: dict judge_id -> per-criterion verdict list."""
    criterion_reports = []
    for c_idx, criterion in enumerate(criteria):
        votes = [
            JudgeVote(judge_id=jid, verdict=verds[c_idx], reason="Test", weight=1.0)
            for jid, verds in verdicts_per_judge.items()
        ]
        # Final verdict is irrelevant to the per-judge matrix; use the first judge's.
        final = next(iter(verdicts_per_judge.values()))[c_idx]
        criterion_reports.append(
            EnsembleCriterionReport(
                criterion=criterion,
                final_verdict=final,
                final_reason="agg",
                votes=votes,
            )
        )
    return EnsembleEvaluationReport(
        report=criterion_reports,
        score=score,
        raw_score=score,
        judge_scores=judge_scores,
    )


def _ensemble_eval(items):
    item_results = [
        ItemResult(item_idx=i, item=None, report=r, duration_seconds=0.5)
        for i, r in enumerate(items)
    ]
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


class TestPerJudgeConfusionMatrix:
    def _two_judge_metrics(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [UNMET, UNMET]])
        items = [
            _ensemble_item(
                criteria,
                {"judge_a": [MET, MET], "judge_b": [MET, CA]},
                {"judge_a": 1.0, "judge_b": 0.9},
                1.0,
            ),
            _ensemble_item(
                criteria,
                {"judge_a": [UNMET, UNMET], "judge_b": [UNMET, UNMET]},
                {"judge_a": 0.0, "judge_b": 0.1},
                0.0,
            ),
        ]
        return compute_metrics(_ensemble_eval(items), dataset, per_judge=True)

    def test_per_judge_cm_is_3x3_with_abstain_class(self):
        metrics = self._two_judge_metrics()
        assert metrics.per_judge is not None
        for jm in metrics.per_judge.values():
            cm = jm.confusion_matrix
            assert isinstance(cm, ConfusionMatrix)
            assert cm.n_classes == 3
            assert cm.labels == ["MET", "UNMET", "CANNOT_ASSESS"]
            assert len(cm.matrix) == 3 and all(len(r) == 3 for r in cm.matrix)

    def test_per_judge_cm_pools_across_criteria(self):
        metrics = self._two_judge_metrics()
        assert metrics.per_judge is not None
        jb = metrics.per_judge["judge_b"]
        assert jb.confusion_matrix is not None
        # judge_b: (item1, c1) truth MET, pred CANNOT_ASSESS → pooled cell [row MET=0][col CA=2].
        assert jb.confusion_matrix.matrix[0][2] == 1
        # judge_a never abstained → its abstain column is all zeros.
        ja = metrics.per_judge["judge_a"]
        assert ja.confusion_matrix is not None
        assert all(row[2] == 0 for row in ja.confusion_matrix.matrix)

    def test_per_judge_phi_present(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [UNMET, UNMET]])
        items = [
            _ensemble_item(
                criteria,
                {"judge_a": [MET, MET], "judge_b": [MET, UNMET]},
                {"judge_a": 1.0, "judge_b": 0.9},
                1.0,
            ),
            _ensemble_item(
                criteria,
                {"judge_a": [UNMET, UNMET], "judge_b": [UNMET, MET]},
                {"judge_a": 0.0, "judge_b": 0.1},
                0.0,
            ),
        ]
        metrics = compute_metrics(_ensemble_eval(items), dataset, per_judge=True)
        assert metrics.per_judge is not None
        for jm in metrics.per_judge.values():
            assert jm.phi is not None


class TestSingleJudgeEqualsAggregate:
    def test_one_judge_phi_equals_aggregate(self):
        criteria = _two_criteria()
        dataset = _dataset(criteria, [[MET, MET], [UNMET, UNMET], [MET, UNMET]])
        verds = [[MET, MET], [UNMET, MET], [MET, UNMET]]
        items = [
            _ensemble_item(criteria, {"only": v}, {"only": float(i)}, float(i))
            for i, v in enumerate(verds)
        ]
        metrics = compute_metrics(_ensemble_eval(items), dataset, per_judge=True)
        assert metrics.per_judge is not None
        jm = metrics.per_judge["only"]
        # 1-judge == aggregate for phi and the pooled criterion accuracy.
        assert jm.phi == pytest.approx(metrics.criterion_phi)
        assert jm.criterion_accuracy == pytest.approx(metrics.criterion_accuracy)
