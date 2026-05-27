"""Tests for inter-judge agreement (Krippendorff's alpha + Fleiss' kappa) wiring in
compute_metrics, MultiChoiceJudgeVote.error, and the per-judge binary metrics alignment
fix (Issue #2).

Krippendorff's alpha is the general, recommended statistic: it handles unequal/missing
raters and is level-aware (nominal vs ordinal). Fleiss' kappa is the classic fixed-rater
nominal measure, computed complete-case (only items where every judge cast a genuine vote).
"""

from datetime import datetime

import pytest

from autorubric.dataset import DataItem, RubricDataset
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.metrics import compute_metrics
from autorubric.rubric import Rubric
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    Criterion,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
)

# =============================================================================
# Builders
# =============================================================================


def _timing() -> EvalTimingStats:
    return EvalTimingStats(
        total_duration_seconds=1.0,
        mean_item_duration_seconds=0.5,
        min_item_duration_seconds=0.4,
        max_item_duration_seconds=0.6,
        p50_item_duration_seconds=0.5,
        p95_item_duration_seconds=0.55,
        items_per_second=2.0,
    )


def _eval_result(item_results: list[ItemResult]) -> EvalResult:
    return EvalResult(
        item_results=item_results,
        total_items=len(item_results),
        successful_items=len(item_results),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=_timing(),
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


def _binary_ensemble_report(
    criterion: Criterion,
    judge_verdicts: list[tuple[str, CriterionVerdict]],
    *,
    errors: dict[str, str] | None = None,
) -> EnsembleEvaluationReport:
    """One-criterion binary ensemble report from per-judge verdicts."""
    errors = errors or {}
    votes = [
        JudgeVote(
            judge_id=jid,
            verdict=verdict,
            reason="r",
            error=errors.get(jid),
        )
        for jid, verdict in judge_verdicts
    ]
    # Majority final verdict (ties -> MET, irrelevant to Fleiss).
    n_met = sum(1 for _, v in judge_verdicts if v == CriterionVerdict.MET)
    final = CriterionVerdict.MET if n_met * 2 >= len(judge_verdicts) else CriterionVerdict.UNMET
    ecr = EnsembleCriterionReport(
        criterion=criterion,
        final_verdict=final,
        final_reason="agg",
        votes=votes,
    )
    judge_scores = {jid: (1.0 if v == CriterionVerdict.MET else 0.0) for jid, v in judge_verdicts}
    return EnsembleEvaluationReport(
        score=1.0 if final == CriterionVerdict.MET else 0.0,
        raw_score=0.0,
        report=[ecr],
        judge_scores=judge_scores,
    )


def _multi_choice_ensemble_report(
    criterion: Criterion,
    judge_indices: list[tuple[str, int]],
    *,
    errors: dict[str, str] | None = None,
) -> EnsembleEvaluationReport:
    """One-criterion multi-choice ensemble report from per-judge selected indices."""
    errors = errors or {}
    mc_votes = []
    for jid, idx in judge_indices:
        opt = criterion.options[idx]
        mc_votes.append(
            MultiChoiceJudgeVote(
                judge_id=jid,
                selected_index=idx,
                selected_label=opt.label,
                value=opt.value,
                reason="r",
                na=opt.na,
                error=errors.get(jid),
            )
        )
    # Final = first vote's index (irrelevant to Fleiss).
    first_idx = judge_indices[0][1]
    opt0 = criterion.options[first_idx]
    final = AggregatedMultiChoiceVerdict(
        selected_index=first_idx,
        selected_label=opt0.label,
        value=opt0.value,
        na=opt0.na,
        aggregated_value=opt0.value,
    )
    judge_scores = {jid: criterion.options[idx].value for jid, idx in judge_indices}
    return EnsembleEvaluationReport(
        score=opt0.value,
        raw_score=0.0,
        report=[ecr_mc(criterion, final, mc_votes)],
        judge_scores=judge_scores,
    )


def ecr_mc(criterion, final, mc_votes) -> EnsembleCriterionReport:
    return EnsembleCriterionReport(
        criterion=criterion,
        final_verdict=None,
        final_reason="agg",
        votes=[],
        final_multi_choice_verdict=final,
        multi_choice_votes=mc_votes,
    )


def _binary_dataset(name: str = "ds") -> RubricDataset:
    rubric = Rubric([Criterion(name="acc", weight=10.0, requirement="Be accurate")])
    ds = RubricDataset(prompt="p", rubric=rubric, name=name)
    return ds


def _ordinal_criterion() -> Criterion:
    return Criterion(
        name="sat",
        weight=10.0,
        requirement="satisfaction",
        scale_type="ordinal",
        options=[
            {"label": "Low", "value": 0.0},
            {"label": "Mid", "value": 0.5},
            {"label": "High", "value": 1.0},
        ],
    )


def _nominal_criterion_with_na() -> Criterion:
    return Criterion(
        name="len",
        weight=5.0,
        requirement="length",
        scale_type="nominal",
        options=[
            {"label": "Too brief", "value": 0.0},
            {"label": "Just right", "value": 1.0},
            {"label": "N/A", "value": 0.0, "na": True},
        ],
    )


# =============================================================================
# Binary Fleiss
# =============================================================================


def test_binary_fleiss_perfect_agreement():
    ds = _binary_dataset()
    crit = ds.rubric.rubric[0]
    # 2 items, 3 judges; item1 all MET, item2 all UNMET -> perfect inter-judge agreement.
    reports = [
        _binary_ensemble_report(
            crit,
            [("ja", CriterionVerdict.MET)] * 1
            + [("jb", CriterionVerdict.MET), ("jc", CriterionVerdict.MET)],
        ),
        _binary_ensemble_report(
            crit,
            [
                ("ja", CriterionVerdict.UNMET),
                ("jb", CriterionVerdict.UNMET),
                ("jc", CriterionVerdict.UNMET),
            ],
        ),
    ]
    ds.add_item(submission="s1", description="d", ground_truth=[CriterionVerdict.MET])
    ds.add_item(submission="s2", description="d", ground_truth=[CriterionVerdict.UNMET])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    assert metrics.per_criterion[0].fleiss_kappa == pytest.approx(1.0)


def test_binary_fleiss_independent_of_ground_truth():
    """Fleiss reflects inter-judge agreement, not ground truth."""
    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")

    def build(ds_gt):
        ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
        reports = [
            _binary_ensemble_report(
                crit,
                [
                    ("ja", CriterionVerdict.MET),
                    ("jb", CriterionVerdict.MET),
                    ("jc", CriterionVerdict.UNMET),
                ],
            ),
            _binary_ensemble_report(
                crit,
                [
                    ("ja", CriterionVerdict.UNMET),
                    ("jb", CriterionVerdict.UNMET),
                    ("jc", CriterionVerdict.MET),
                ],
            ),
        ]
        for i, gt in enumerate(ds_gt):
            ds.add_item(submission=f"s{i}", description="d", ground_truth=[gt])
        item_results = [
            ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
            for i, r in enumerate(reports)
        ]
        return compute_metrics(_eval_result(item_results), ds)

    m1 = build([CriterionVerdict.MET, CriterionVerdict.UNMET])
    m2 = build([CriterionVerdict.UNMET, CriterionVerdict.MET])
    assert m1.per_criterion[0].fleiss_kappa is not None
    assert m1.per_criterion[0].fleiss_kappa == pytest.approx(m2.per_criterion[0].fleiss_kappa)


def test_binary_fleiss_as_category_three_categories():
    """as_category -> 3-column matrix including CANNOT_ASSESS."""
    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [
        _binary_ensemble_report(
            crit,
            [
                ("ja", CriterionVerdict.CANNOT_ASSESS),
                ("jb", CriterionVerdict.CANNOT_ASSESS),
                ("jc", CriterionVerdict.CANNOT_ASSESS),
            ],
        ),
        _binary_ensemble_report(
            crit,
            [
                ("ja", CriterionVerdict.MET),
                ("jb", CriterionVerdict.MET),
                ("jc", CriterionVerdict.MET),
            ],
        ),
    ]
    ds.add_item(submission="s1", description="d", ground_truth=[CriterionVerdict.MET])
    ds.add_item(submission="s2", description="d", ground_truth=[CriterionVerdict.MET])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds, cannot_assess="as_category")
    # Perfect agreement within each item across 3 categories -> kappa 1.0.
    assert metrics.per_criterion[0].fleiss_kappa == pytest.approx(1.0)


def test_binary_fleiss_complete_case_drops_errored_items():
    """Fleiss is complete-case: an item with an errored judge is dropped entirely.

    With an extra judge that errors on both (only) items, every Fleiss row falls below the
    required uniform rater count, so the complete-case matrix is empty -> Fleiss is None.
    Krippendorff's alpha, in contrast, stays finite (errors become missing cells).
    """
    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")

    def build(with_error_judge):
        ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
        items = [
            [("ja", CriterionVerdict.MET), ("jb", CriterionVerdict.MET)],
            [("ja", CriterionVerdict.UNMET), ("jb", CriterionVerdict.UNMET)],
        ]
        errors = {}
        if with_error_judge:
            # jc errors on both items; complete-case drops both rows from Fleiss.
            items = [v + [("jc", CriterionVerdict.UNMET)] for v in items]
            errors = {"jc": "infrastructure: down"}
        reports = [_binary_ensemble_report(crit, v, errors=errors) for v in items]
        for i in range(2):
            ds.add_item(submission=f"s{i}", description="d", ground_truth=[CriterionVerdict.MET])
        item_results = [
            ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
            for i, r in enumerate(reports)
        ]
        return compute_metrics(_eval_result(item_results), ds).per_criterion[0]

    without = build(False)
    with_err = build(True)
    # Complete-case Fleiss is meaningful for the clean (uniform-rater) ensemble.
    assert without.fleiss_kappa is not None
    # Every item lost a rater -> no complete-case rows -> Fleiss None.
    assert with_err.fleiss_kappa is None
    # Krippendorff's alpha remains finite despite the missing (errored) cells.
    assert with_err.krippendorff_alpha is not None
    assert with_err.krippendorff_alpha == pytest.approx(1.0)


# =============================================================================
# Gating
# =============================================================================


def test_fleiss_none_single_judge():
    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [
        _binary_ensemble_report(crit, [("ja", CriterionVerdict.MET)]),
        _binary_ensemble_report(crit, [("ja", CriterionVerdict.UNMET)]),
    ]
    ds.add_item(submission="s1", description="d", ground_truth=[CriterionVerdict.MET])
    ds.add_item(submission="s2", description="d", ground_truth=[CriterionVerdict.UNMET])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    assert metrics.per_criterion[0].fleiss_kappa is None


def test_fleiss_none_single_non_ensemble_report():
    from autorubric.types import CriterionReport, EvaluationReport

    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = []
    for v in (CriterionVerdict.MET, CriterionVerdict.UNMET):
        reports.append(
            EvaluationReport(
                score=1.0 if v == CriterionVerdict.MET else 0.0,
                raw_score=0.0,
                report=[
                    CriterionReport(
                        weight=10.0, requirement="Be accurate", name="acc", verdict=v, reason="r"
                    )
                ],
            )
        )
    ds.add_item(submission="s1", description="d", ground_truth=[CriterionVerdict.MET])
    ds.add_item(submission="s2", description="d", ground_truth=[CriterionVerdict.UNMET])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    assert metrics.per_criterion[0].fleiss_kappa is None


def test_fleiss_none_single_item():
    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [
        _binary_ensemble_report(
            crit,
            [
                ("ja", CriterionVerdict.MET),
                ("jb", CriterionVerdict.MET),
                ("jc", CriterionVerdict.MET),
            ],
        ),
    ]
    ds.add_item(submission="s1", description="d", ground_truth=[CriterionVerdict.MET])
    item_results = [
        ItemResult(item_idx=0, item=ds.items[0], report=reports[0], duration_seconds=0.1)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    assert metrics.per_criterion[0].fleiss_kappa is None


# =============================================================================
# Multi-choice Fleiss (ordinal + nominal)
# =============================================================================


def test_ordinal_fleiss_populated():
    crit = _ordinal_criterion()
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [
        _multi_choice_ensemble_report(crit, [("ja", 2), ("jb", 2), ("jc", 2)]),
        _multi_choice_ensemble_report(crit, [("ja", 0), ("jb", 0), ("jc", 0)]),
    ]
    ds.add_item(submission="s1", description="d", ground_truth=["High"])
    ds.add_item(submission="s2", description="d", ground_truth=["Low"])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    assert metrics.per_criterion[0].fleiss_kappa == pytest.approx(1.0)


def test_nominal_fleiss_counts_genuine_na_as_category():
    crit = _nominal_criterion_with_na()
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    # Item1: all judges pick NA (index 2). Item2: all pick "Just right" (index 1).
    reports = [
        _multi_choice_ensemble_report(crit, [("ja", 2), ("jb", 2), ("jc", 2)]),
        _multi_choice_ensemble_report(crit, [("ja", 1), ("jb", 1), ("jc", 1)]),
    ]
    ds.add_item(submission="s1", description="d", ground_truth=["N/A"])
    ds.add_item(submission="s2", description="d", ground_truth=["Just right"])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    # Genuine NA counted as its column -> perfect agreement -> kappa 1.0.
    assert metrics.per_criterion[0].fleiss_kappa == pytest.approx(1.0)


def test_multi_choice_fleiss_complete_case_drops_errored_items():
    """Multi-choice Fleiss is complete-case; alpha tolerates the errored (missing) cell."""
    crit = _ordinal_criterion()

    def build(with_error_judge):
        ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
        items = [
            [("ja", 2), ("jb", 2)],
            [("ja", 0), ("jb", 0)],
        ]
        errors = {}
        if with_error_judge:
            items = [v + [("jc", 1)] for v in items]
            errors = {"jc": "parse: bad json"}
        reports = [_multi_choice_ensemble_report(crit, v, errors=errors) for v in items]
        ds.add_item(submission="s1", description="d", ground_truth=["High"])
        ds.add_item(submission="s2", description="d", ground_truth=["Low"])
        item_results = [
            ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
            for i, r in enumerate(reports)
        ]
        return compute_metrics(_eval_result(item_results), ds).per_criterion[0]

    without = build(False)
    with_err = build(True)
    assert without.fleiss_kappa is not None
    # Errored jc on both items -> no complete-case rows -> Fleiss None.
    assert with_err.fleiss_kappa is None
    # Alpha stays finite (perfect agreement among the genuine raters).
    assert with_err.krippendorff_alpha is not None
    assert with_err.krippendorff_alpha == pytest.approx(1.0)


# =============================================================================
# MultiChoiceJudgeVote.error round-trip
# =============================================================================


def test_multi_choice_vote_error_round_trips():
    crit = _ordinal_criterion()
    report = _multi_choice_ensemble_report(
        crit, [("ja", 2), ("jb", 1)], errors={"jb": "infrastructure: down"}
    )
    item = DataItem(submission="s", description="d")
    item_result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=0.1)

    import json

    payload = json.loads(json.dumps(item_result.to_dict()))
    restored = ItemResult.from_dict(payload, item)

    assert restored.report.report is not None
    votes = restored.report.report[0].multi_choice_votes
    by_id = {v.judge_id: v for v in votes}
    assert by_id["ja"].error is None
    assert by_id["jb"].error == "infrastructure: down"


# =============================================================================
# Per-judge binary metrics alignment fix
# =============================================================================


def test_per_judge_criterion_accuracy_hand_computed():
    """Multi-criteria ensemble, per_judge=True: criterion_accuracy is correct.

    Pre-fix, truth was mis-aligned (criterion-0 only, char-iterated) -> garbage.
    """
    c1 = Criterion(name="c1", weight=10.0, requirement="r1")
    c2 = Criterion(name="c2", weight=10.0, requirement="r2")
    ds = RubricDataset(prompt="p", rubric=Rubric([c1, c2]), name="x")

    # Ground truth: item1 = [MET, UNMET], item2 = [UNMET, MET]
    ds.add_item(
        submission="s1",
        description="d",
        ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
    )
    ds.add_item(
        submission="s2",
        description="d",
        ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET],
    )

    # Judge "ja" verdicts:
    # item1: [MET, MET]  -> c1 correct, c2 wrong
    # item2: [UNMET, MET] -> c1 correct, c2 correct
    # 3/4 correct = 0.75
    def ecr(c, ja_v, jb_v):
        votes = [
            JudgeVote(judge_id="ja", verdict=ja_v, reason="r"),
            JudgeVote(judge_id="jb", verdict=jb_v, reason="r"),
        ]
        final = ja_v
        return EnsembleCriterionReport(
            criterion=c, final_verdict=final, final_reason="a", votes=votes
        )

    def report(ja_v1, ja_v2, jb_v1, jb_v2):
        return EnsembleEvaluationReport(
            score=0.5,
            raw_score=0.0,
            report=[ecr(c1, ja_v1, jb_v1), ecr(c2, ja_v2, jb_v2)],
            judge_scores={"ja": 0.5, "jb": 0.5},
        )

    reports = [
        report(
            CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.UNMET
        ),
        report(
            CriterionVerdict.UNMET,
            CriterionVerdict.MET,
            CriterionVerdict.UNMET,
            CriterionVerdict.MET,
        ),
    ]
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds, per_judge=True)
    assert metrics.per_judge is not None
    assert metrics.per_judge["ja"].criterion_accuracy == pytest.approx(0.75)


def test_per_judge_excludes_errored_votes():
    c1 = Criterion(name="c1", weight=10.0, requirement="r1")
    ds = RubricDataset(prompt="p", rubric=Rubric([c1]), name="x")
    ds.add_item(submission="s1", description="d", ground_truth=[CriterionVerdict.MET])
    ds.add_item(submission="s2", description="d", ground_truth=[CriterionVerdict.MET])

    def report(ja_v, ja_err, jb_v):
        votes = [
            JudgeVote(judge_id="ja", verdict=ja_v, reason="r", error=ja_err),
            JudgeVote(judge_id="jb", verdict=jb_v, reason="r"),
        ]
        return EnsembleEvaluationReport(
            score=1.0,
            raw_score=0.0,
            report=[
                EnsembleCriterionReport(
                    criterion=c1, final_verdict=jb_v, final_reason="a", votes=votes
                )
            ],
            judge_scores={"ja": 0.0, "jb": 1.0},
        )

    # ja: item1 errored UNMET (excluded), item2 genuine MET (correct).
    reports = [
        report(CriterionVerdict.UNMET, "infrastructure: x", CriterionVerdict.MET),
        report(CriterionVerdict.MET, None, CriterionVerdict.MET),
    ]
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds, per_judge=True)
    assert metrics.per_judge is not None
    # Only 1 genuine ja vote (item2 MET vs GT MET) -> accuracy 1.0.
    assert metrics.per_judge["ja"].criterion_accuracy == pytest.approx(1.0)


# =============================================================================
# Krippendorff's alpha (the general, recommended inter-judge agreement statistic)
# =============================================================================


def _binary_alpha_dataset(judge_verdicts_per_item, ground_truth, *, errors_per_item=None):
    """Build a binary ensemble dataset/result from per-item judge verdict lists."""
    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    errors_per_item = errors_per_item or [None] * len(judge_verdicts_per_item)
    reports = [
        _binary_ensemble_report(crit, jv, errors=err)
        for jv, err in zip(judge_verdicts_per_item, errors_per_item)
    ]
    for i, gt in enumerate(ground_truth):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[gt])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    return compute_metrics(_eval_result(item_results), ds).per_criterion[0]


def _multi_alpha_dataset(crit, judge_indices_per_item, ground_truth):
    """Build a multi-choice ensemble dataset/result from per-item judge index lists."""
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [_multi_choice_ensemble_report(crit, ji) for ji in judge_indices_per_item]
    for i, gt in enumerate(ground_truth):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[gt])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    return compute_metrics(_eval_result(item_results), ds).per_criterion[0]


def test_binary_krippendorff_alpha_perfect_agreement():
    cm = _binary_alpha_dataset(
        [
            [
                ("ja", CriterionVerdict.MET),
                ("jb", CriterionVerdict.MET),
                ("jc", CriterionVerdict.MET),
            ],
            [
                ("ja", CriterionVerdict.UNMET),
                ("jb", CriterionVerdict.UNMET),
                ("jc", CriterionVerdict.UNMET),
            ],
        ],
        [CriterionVerdict.MET, CriterionVerdict.UNMET],
    )
    assert cm.krippendorff_alpha == pytest.approx(1.0)


def test_ordinal_krippendorff_alpha_perfect_agreement():
    crit = _ordinal_criterion()
    cm = _multi_alpha_dataset(
        crit,
        [[("ja", 2), ("jb", 2), ("jc", 2)], [("ja", 0), ("jb", 0), ("jc", 0)]],
        ["High", "Low"],
    )
    assert cm.krippendorff_alpha == pytest.approx(1.0)


def test_nominal_krippendorff_alpha_populated():
    crit = _nominal_criterion_with_na()
    cm = _multi_alpha_dataset(
        crit,
        [[("ja", 1), ("jb", 1), ("jc", 1)], [("ja", 0), ("jb", 0), ("jc", 0)]],
        ["Just right", "Too brief"],
    )
    assert cm.krippendorff_alpha == pytest.approx(1.0)


def test_krippendorff_alpha_handles_unequal_raters():
    """The key test distinguishing alpha from Fleiss.

    One judge errors/abstains on one item -> alpha is finite (NOT None), while Fleiss
    drops that item complete-case. Here the errored item still leaves a second complete
    item, so Fleiss remains computable; the point is that alpha tolerates the missing cell.
    """
    cm = _binary_alpha_dataset(
        [
            # item1: jc errors -> missing cell for alpha, item dropped from Fleiss.
            [
                ("ja", CriterionVerdict.MET),
                ("jb", CriterionVerdict.MET),
                ("jc", CriterionVerdict.UNMET),
            ],
            # item2: all three genuine and agree.
            [
                ("ja", CriterionVerdict.UNMET),
                ("jb", CriterionVerdict.UNMET),
                ("jc", CriterionVerdict.UNMET),
            ],
        ],
        [CriterionVerdict.MET, CriterionVerdict.UNMET],
        errors_per_item=[{"jc": "infrastructure: down"}, None],
    )
    # Alpha is finite despite the missing (errored) cell on item1.
    assert cm.krippendorff_alpha is not None
    # Fleiss only sees item2 (the complete-case item) -> a single subject -> None.
    assert cm.fleiss_kappa is None


def test_krippendorff_alpha_is_ordinal_aware():
    """Ordinal alpha penalizes a near-miss (adjacent) less than a far-miss.

    Built so the only difference between the two ensembles is whether the lone dissenting
    judge picks an adjacent option (near) or the opposite extreme (far). Ordinal alpha is
    higher for the near-miss; this would be identical under a nominal (Fleiss-like) treatment.
    """
    crit = _ordinal_criterion()  # 3 ordered options: Low(0), Mid(1), High(2)
    # 3 items, 3 judges. Items 1 & 2 unanimous; item 3 has one dissenter.
    near = _multi_alpha_dataset(
        crit,
        [
            [("ja", 0), ("jb", 0), ("jc", 0)],
            [("ja", 2), ("jb", 2), ("jc", 2)],
            [("ja", 1), ("jb", 1), ("jc", 2)],  # dissent: adjacent (Mid vs High)
        ],
        ["Low", "High", "Mid"],
    )
    far = _multi_alpha_dataset(
        crit,
        [
            [("ja", 0), ("jb", 0), ("jc", 0)],
            [("ja", 2), ("jb", 2), ("jc", 2)],
            [("ja", 0), ("jb", 0), ("jc", 2)],  # dissent: far (Low vs High)
        ],
        ["Low", "High", "Low"],
    )
    assert near.krippendorff_alpha is not None
    assert far.krippendorff_alpha is not None
    # Ordinal-aware: the near-miss disagreement yields higher agreement than the far-miss.
    assert near.krippendorff_alpha > far.krippendorff_alpha


def test_krippendorff_alpha_none_single_judge():
    cm = _binary_alpha_dataset(
        [[("ja", CriterionVerdict.MET)], [("ja", CriterionVerdict.UNMET)]],
        [CriterionVerdict.MET, CriterionVerdict.UNMET],
    )
    assert cm.krippendorff_alpha is None


def test_krippendorff_alpha_none_single_item():
    cm = _binary_alpha_dataset(
        [
            [
                ("ja", CriterionVerdict.MET),
                ("jb", CriterionVerdict.MET),
                ("jc", CriterionVerdict.MET),
            ],
        ],
        [CriterionVerdict.MET],
    )
    assert cm.krippendorff_alpha is None


def test_krippendorff_alpha_none_single_non_ensemble_report():
    from autorubric.types import CriterionReport, EvaluationReport

    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = []
    for v in (CriterionVerdict.MET, CriterionVerdict.UNMET):
        reports.append(
            EvaluationReport(
                score=1.0 if v == CriterionVerdict.MET else 0.0,
                raw_score=0.0,
                report=[
                    CriterionReport(
                        weight=10.0, requirement="Be accurate", name="acc", verdict=v, reason="r"
                    )
                ],
            )
        )
    ds.add_item(submission="s1", description="d", ground_truth=[CriterionVerdict.MET])
    ds.add_item(submission="s2", description="d", ground_truth=[CriterionVerdict.UNMET])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    assert metrics.per_criterion[0].krippendorff_alpha is None


def test_krippendorff_alpha_error_vote_becomes_missing_cell():
    """An errored vote becomes a missing cell and does not corrupt alpha.

    Compare a clean 2-judge ensemble against the same data plus a third judge that errors
    on every item: the errored cells are missing, so alpha is unchanged.
    """
    clean = _binary_alpha_dataset(
        [
            [("ja", CriterionVerdict.MET), ("jb", CriterionVerdict.MET)],
            [("ja", CriterionVerdict.UNMET), ("jb", CriterionVerdict.UNMET)],
        ],
        [CriterionVerdict.MET, CriterionVerdict.UNMET],
    )
    with_err = _binary_alpha_dataset(
        [
            [
                ("ja", CriterionVerdict.MET),
                ("jb", CriterionVerdict.MET),
                ("jc", CriterionVerdict.UNMET),
            ],
            [
                ("ja", CriterionVerdict.UNMET),
                ("jb", CriterionVerdict.UNMET),
                ("jc", CriterionVerdict.MET),
            ],
        ],
        [CriterionVerdict.MET, CriterionVerdict.UNMET],
        errors_per_item=[{"jc": "parse: bad"}, {"jc": "parse: bad"}],
    )
    assert clean.krippendorff_alpha is not None
    assert with_err.krippendorff_alpha == pytest.approx(clean.krippendorff_alpha)


# =============================================================================
# Rendering: summary() and to_dataframe() surface agreement stats (T1-D)
# =============================================================================


def _binary_alpha_metrics(judge_verdicts_per_item, ground_truth):
    """Like ``_binary_alpha_dataset`` but returns the full MetricsResult."""
    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [_binary_ensemble_report(crit, jv) for jv in judge_verdicts_per_item]
    for i, gt in enumerate(ground_truth):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[gt])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    return compute_metrics(_eval_result(item_results), ds)


def _multi_alpha_metrics(crit, judge_indices_per_item, ground_truth):
    """Like ``_multi_alpha_dataset`` but returns the full MetricsResult."""
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [_multi_choice_ensemble_report(crit, ji) for ji in judge_indices_per_item]
    for i, gt in enumerate(ground_truth):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[gt])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    return compute_metrics(_eval_result(item_results), ds)


_PERFECT_BINARY = (
    [
        [
            ("ja", CriterionVerdict.MET),
            ("jb", CriterionVerdict.MET),
            ("jc", CriterionVerdict.MET),
        ],
        [
            ("ja", CriterionVerdict.UNMET),
            ("jb", CriterionVerdict.UNMET),
            ("jc", CriterionVerdict.UNMET),
        ],
    ],
    [CriterionVerdict.MET, CriterionVerdict.UNMET],
)


def test_summary_surfaces_alpha_and_fleiss_binary():
    metrics = _binary_alpha_metrics(*_PERFECT_BINARY)
    # Guard: the ensemble path actually populated the agreement stats.
    assert metrics.per_criterion[0].krippendorff_alpha is not None
    assert metrics.per_criterion[0].fleiss_kappa is not None
    summary = metrics.summary()
    assert "Kripp-α" in summary  # "Kripp-α"
    assert "Fleiss" in summary


def test_to_dataframe_has_agreement_columns_binary():
    pytest.importorskip("pandas")
    metrics = _binary_alpha_metrics(*_PERFECT_BINARY)
    df = metrics.to_dataframe()
    assert "krippendorff_alpha" in df.columns
    assert "fleiss_kappa" in df.columns
    crit_row = df[df["level"] == "criterion"].iloc[0]
    assert crit_row["krippendorff_alpha"] == pytest.approx(1.0)
    assert crit_row["fleiss_kappa"] == pytest.approx(1.0)


def test_summary_omits_agreement_columns_for_single_judge():
    metrics = _binary_alpha_metrics(
        [[("ja", CriterionVerdict.MET)], [("ja", CriterionVerdict.UNMET)]],
        [CriterionVerdict.MET, CriterionVerdict.UNMET],
    )
    assert metrics.per_criterion[0].krippendorff_alpha is None
    summary = metrics.summary()
    assert "Kripp-α" not in summary
    assert "Fleiss" not in summary


def test_to_dataframe_agreement_none_for_single_judge():
    import pandas as pd

    metrics = _binary_alpha_metrics(
        [[("ja", CriterionVerdict.MET)], [("ja", CriterionVerdict.UNMET)]],
        [CriterionVerdict.MET, CriterionVerdict.UNMET],
    )
    df = metrics.to_dataframe()
    # Columns are always present (uniform schema); values are NaN when not applicable.
    assert "krippendorff_alpha" in df.columns
    assert "fleiss_kappa" in df.columns
    crit_row = df[df["level"] == "criterion"].iloc[0]
    assert pd.isna(crit_row["krippendorff_alpha"])
    assert pd.isna(crit_row["fleiss_kappa"])


def test_dataframe_and_summary_surface_agreement_ordinal_and_nominal():
    pytest.importorskip("pandas")
    ordinal = _multi_alpha_metrics(
        _ordinal_criterion(),
        [[("ja", 2), ("jb", 2), ("jc", 2)], [("ja", 0), ("jb", 0), ("jc", 0)]],
        ["High", "Low"],
    )
    nominal = _multi_alpha_metrics(
        _nominal_criterion_with_na(),
        [[("ja", 1), ("jb", 1), ("jc", 1)], [("ja", 0), ("jb", 0), ("jc", 0)]],
        ["Just right", "Too brief"],
    )
    for metrics in (ordinal, nominal):
        assert metrics.per_criterion[0].krippendorff_alpha is not None
        df = metrics.to_dataframe()
        crit_row = df[df["level"] == "criterion"].iloc[0]
        assert crit_row["krippendorff_alpha"] == pytest.approx(1.0)
        assert "Kripp-α" in metrics.summary()
