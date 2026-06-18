"""Tests for pooled rubric-point metrics on per-item, heterogeneous-rubric datasets.

When a dataset has no global rubric and item rubrics genuinely differ (e.g. HealthBench),
``compute_metrics`` cannot build a shared per-criterion table, so it pools every rubric-point
decision per scale type (``MetricsResult.pooled_by_scale``) instead of raising.
"""

from datetime import datetime

import pytest

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.rubric import Rubric
from autorubric.types import (
    CriterionReport,
    CriterionVerdict,
    EvaluationReport,
    MultiChoiceVerdict,
)


def _eval_result(reports: list[EvaluationReport], dataset: RubricDataset) -> EvalResult:
    item_results = [
        ItemResult(item_idx=i, item=dataset.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    return EvalResult(
        item_results=item_results,
        total_items=len(reports),
        successful_items=len(reports),
        failed_items=0,
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
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


def _binary_report(verdicts: list[CriterionVerdict], rubric: Rubric) -> EvaluationReport:
    return EvaluationReport(
        score=0.0,
        raw_score=0.0,
        report=[
            CriterionReport(
                weight=rubric.rubric[i].weight,
                requirement=rubric.rubric[i].requirement,
                name=rubric.rubric[i].name,
                verdict=v,
                reason="t",
            )
            for i, v in enumerate(verdicts)
        ],
    )


class TestHeterogeneousBinaryPooling:
    def test_pools_and_does_not_raise(self):
        # Two items with DIFFERENT binary criteria (heterogeneous) — previously a ValueError.
        r0 = Rubric.from_dict(
            [
                {"name": "a", "weight": 10.0, "requirement": "Requirement A"},
                {"name": "b", "weight": 5.0, "requirement": "Requirement B"},
            ]
        )
        r1 = Rubric.from_dict(
            [
                {"name": "c", "weight": 8.0, "requirement": "Requirement C"},
                {"name": "d", "weight": 5.0, "requirement": "Requirement D"},
            ]
        )
        ds = RubricDataset(prompt="p", rubric=None, name="hetero")
        ds.add_item(
            submission="s0",
            description="i0",
            rubric=r0,
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
        )
        ds.add_item(
            submission="s1",
            description="i1",
            rubric=r1,
            ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET],
        )
        # Predictions: item0 [MET, MET] (1 wrong), item1 [UNMET, MET] (both right) -> 3/4.
        ev = _eval_result(
            [
                _binary_report([CriterionVerdict.MET, CriterionVerdict.MET], r0),
                _binary_report([CriterionVerdict.UNMET, CriterionVerdict.MET], r1),
            ],
            ds,
        )
        m = ev.compute_metrics(ds)  # auto -> pooled

        assert m.pooled_by_scale is not None
        assert m.per_criterion == []
        assert m.criterion_accuracy == pytest.approx(0.75)
        assert m.n_items == 2
        assert m.n_criteria == 4  # pooled rubric points
        binary = next(e for e in m.pooled_by_scale if e.scale_type == "binary")
        assert binary.n_points == 4
        assert binary.exact_accuracy == pytest.approx(0.75)
        assert binary.confusion_matrix is not None
        assert any("pooled rubric-point" in w for w in m.warnings)

    def test_abstention_excluded_from_pool(self):
        r0 = Rubric.from_dict([{"name": "a", "weight": 10.0, "requirement": "Requirement A"}])
        r1 = Rubric.from_dict([{"name": "b", "weight": 10.0, "requirement": "Requirement B"}])
        ds = RubricDataset(prompt="p", rubric=None, name="hetero")
        ds.add_item(
            submission="s0", description="i0", rubric=r0, ground_truth=[CriterionVerdict.MET]
        )
        ds.add_item(
            submission="s1", description="i1", rubric=r1, ground_truth=[CriterionVerdict.MET]
        )
        ev = _eval_result(
            [
                _binary_report([CriterionVerdict.MET], r0),
                _binary_report([CriterionVerdict.CANNOT_ASSESS], r1),
            ],
            ds,
        )
        m = ev.compute_metrics(ds)
        assert m.pooled_by_scale is not None
        binary = next(e for e in m.pooled_by_scale if e.scale_type == "binary")
        assert binary.n_points == 1  # the CANNOT_ASSESS point is excluded
        assert binary.n_abstain == 1
        assert binary.exact_accuracy == pytest.approx(1.0)


class TestHeterogeneousMixedScales:
    def test_all_scale_types_pooled(self):
        # Item 0: binary + ordinal; Item 1: nominal + binary. Different rubrics -> heterogeneous.
        r0 = Rubric.from_dict(
            [
                {"name": "py", "weight": 10.0, "requirement": "Mentions Python"},
                {
                    "name": "help",
                    "weight": 8.0,
                    "requirement": "How helpful?",
                    "scale_type": "ordinal",
                    "options": [
                        {"label": "Low", "value": 0.0},
                        {"label": "Mid", "value": 0.5},
                        {"label": "High", "value": 1.0},
                    ],
                },
            ]
        )
        r1 = Rubric.from_dict(
            [
                {
                    "name": "tone",
                    "weight": 5.0,
                    "requirement": "Tone?",
                    "scale_type": "nominal",
                    "options": [
                        {"label": "Formal", "value": 1.0},
                        {"label": "Casual", "value": 1.0},
                        {"label": "Rude", "value": 0.0},
                    ],
                },
                {"name": "code", "weight": 10.0, "requirement": "Has code"},
            ]
        )
        ds = RubricDataset(prompt="p", rubric=None, name="mixed")
        ds.add_item(
            submission="s0",
            description="i0",
            rubric=r0,
            ground_truth=[CriterionVerdict.MET, "High"],
        )
        ds.add_item(
            submission="s1",
            description="i1",
            rubric=r1,
            ground_truth=["Casual", CriterionVerdict.MET],
        )

        rep0 = EvaluationReport(
            score=0.0,
            raw_score=0.0,
            report=[
                CriterionReport(
                    weight=10.0,
                    requirement="Mentions Python",
                    name="py",
                    verdict=CriterionVerdict.MET,
                    reason="t",
                ),
                CriterionReport(
                    weight=8.0,
                    requirement="How helpful?",
                    name="help",
                    verdict=CriterionVerdict.MET,
                    reason="t",
                    multi_choice_verdict=MultiChoiceVerdict(
                        selected_index=2, selected_label="High", value=1.0
                    ),
                ),
            ],
        )
        rep1 = EvaluationReport(
            score=0.0,
            raw_score=0.0,
            report=[
                CriterionReport(
                    weight=5.0,
                    requirement="Tone?",
                    name="tone",
                    verdict=CriterionVerdict.MET,
                    reason="t",
                    multi_choice_verdict=MultiChoiceVerdict(
                        selected_index=1, selected_label="Casual", value=1.0
                    ),
                ),
                CriterionReport(
                    weight=10.0,
                    requirement="Has code",
                    name="code",
                    verdict=CriterionVerdict.UNMET,
                    reason="t",
                ),
            ],
        )
        m = _eval_result([rep0, rep1], ds).compute_metrics(ds)

        assert m.pooled_by_scale is not None
        scales = {e.scale_type: e for e in m.pooled_by_scale}
        assert set(scales) == {"binary", "ordinal", "nominal"}
        # Binary: py MET==MET (hit), code UNMET vs MET (miss) -> 1/2.
        assert scales["binary"].n_points == 2
        assert scales["binary"].exact_accuracy == pytest.approx(0.5)
        assert scales["binary"].confusion_matrix is not None
        # Ordinal/nominal: exact match, and option-set-dependent categorical metrics are None.
        assert scales["ordinal"].n_points == 1
        assert scales["ordinal"].exact_accuracy == pytest.approx(1.0)
        assert scales["ordinal"].kappa is None
        assert scales["ordinal"].confusion_matrix is None
        assert scales["nominal"].n_points == 1
        assert scales["nominal"].exact_accuracy == pytest.approx(1.0)
        assert scales["nominal"].confusion_matrix is None


class TestHomogeneousPerItemAndOverride:
    def test_homogeneous_per_item_uses_per_criterion_and_does_not_crash(self):
        # Same rubric attached per-item (no global rubric). Not heterogeneous -> per-criterion
        # path; the score step must not crash on the missing global rubric.
        rub = Rubric.from_dict(
            [
                {"name": "a", "weight": 10.0, "requirement": "Requirement A"},
                {"name": "b", "weight": 5.0, "requirement": "Requirement B"},
            ]
        )
        ds = RubricDataset(prompt="p", rubric=None, name="homo")
        for i in range(3):
            ds.add_item(
                submission=f"s{i}",
                description=f"i{i}",
                rubric=rub,
                ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
            )
        ev = _eval_result(
            [_binary_report([CriterionVerdict.MET, CriterionVerdict.UNMET], rub) for _ in range(3)],
            ds,
        )
        m = ev.compute_metrics(ds)  # auto -> per-criterion (homogeneous)
        assert m.pooled_by_scale is None
        assert len(m.per_criterion) == 2  # a real per-criterion table
        assert m.criterion_accuracy == pytest.approx(1.0)

    def test_force_pooled_with_global_rubric(self):
        rub = Rubric.from_dict([{"name": "a", "weight": 10.0, "requirement": "Requirement A"}])
        ds = RubricDataset(prompt="p", rubric=rub, name="global")
        ds.add_item(submission="s0", description="i0", ground_truth=[CriterionVerdict.MET])
        ds.add_item(submission="s1", description="i1", ground_truth=[CriterionVerdict.UNMET])
        ev = _eval_result(
            [
                _binary_report([CriterionVerdict.MET], rub),
                _binary_report([CriterionVerdict.UNMET], rub),
            ],
            ds,
        )
        forced = ev.compute_metrics(ds, per_item_metrics="pooled")
        assert forced.pooled_by_scale is not None
        assert forced.per_criterion == []
        # "auto" on a global-rubric dataset stays on the per-criterion path.
        auto = ev.compute_metrics(ds, per_item_metrics="auto")
        assert auto.pooled_by_scale is None
        assert len(auto.per_criterion) == 1
