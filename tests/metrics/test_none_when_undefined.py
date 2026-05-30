"""Tests for the "metric is None when genuinely undefined" principle (Phase 3).

A metric is ``None`` when computation failed / degenerate single-class / no samples,
NEVER a fake ``0.0``. Counts (``n_samples``, ``support_*``, confusion-matrix cells) STAY
``0`` (zero is the true count). This file covers Issue #1 (kappa→None), Issue #3
(empty-data criteria→None), and the bootstrap-CI undefined→None changes.
"""

import warnings
from datetime import datetime

import pytest

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.metrics import compute_metrics
from autorubric.metrics._compute import (
    _compute_adjacent_accuracy,
    _compute_bootstrap_ci,
    _compute_correlation,
    _compute_nominal_criterion_metrics,
    _compute_ordinal_criterion_metrics,
    _compute_per_option_metrics,
    _kappa_or_none,
)
from autorubric.rubric import Rubric
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    Criterion,
    CriterionReport,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EvaluationReport,
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


def _nominal_criterion() -> Criterion:
    return Criterion(
        name="cat",
        weight=4.0,
        requirement="category",
        scale_type="nominal",
        options=[
            {"label": "Alpha", "value": 1.0},
            {"label": "Beta", "value": 0.0},
            {"label": "Gamma", "value": 0.5},
        ],
    )


def _ecr_mc(criterion, final, mc_votes) -> EnsembleCriterionReport:
    return EnsembleCriterionReport(
        criterion=criterion,
        final_verdict=None,
        final_reason="agg",
        votes=[],
        final_multi_choice_verdict=final,
        multi_choice_votes=mc_votes,
    )


def _multi_criterion_ensemble_report(
    criteria: list[Criterion],
    per_criterion_judge_picks: list[list[tuple[str, object]]],
    judge_ids: list[str],
) -> EnsembleEvaluationReport:
    """Build a multi-criterion ensemble report for one item (mirror of the harness in
    test_interjudge_agreement.py)."""
    ecrs = []
    judge_score_acc: dict[str, float] = {jid: 0.0 for jid in judge_ids}
    for crit, picks in zip(criteria, per_criterion_judge_picks):
        if crit.is_binary:
            votes = [
                JudgeVote(judge_id=jid, verdict=pick, reason="r")  # type: ignore[arg-type]
                for jid, pick in picks
            ]
            n_met = sum(1 for _, v in picks if v == CriterionVerdict.MET)
            final = CriterionVerdict.MET if n_met * 2 >= len(picks) else CriterionVerdict.UNMET
            ecrs.append(
                EnsembleCriterionReport(
                    criterion=crit, final_verdict=final, final_reason="agg", votes=votes
                )
            )
            for jid, v in picks:
                judge_score_acc[jid] += crit.weight if v == CriterionVerdict.MET else 0.0
        else:
            mc_votes = []
            for jid, idx in picks:
                opt = crit.options[idx]  # type: ignore[index]
                mc_votes.append(
                    MultiChoiceJudgeVote(
                        judge_id=jid,
                        selected_index=idx,  # type: ignore[arg-type]
                        selected_label=opt.label,
                        value=opt.value,
                        reason="r",
                        na=opt.na,
                    )
                )
                judge_score_acc[jid] += crit.weight * opt.value
            first_idx = picks[0][1]
            opt0 = crit.options[first_idx]  # type: ignore[index]
            final = AggregatedMultiChoiceVerdict(
                selected_index=first_idx,  # type: ignore[arg-type]
                selected_label=opt0.label,
                value=opt0.value,
                na=opt0.na,
                aggregated_value=opt0.value,
            )
            ecrs.append(_ecr_mc(crit, final, mc_votes))
    return EnsembleEvaluationReport(
        score=0.5,
        raw_score=0.0,
        report=ecrs,
        judge_scores={jid: judge_score_acc[jid] for jid in judge_ids},
    )


def _binary_report(verdicts: list[CriterionVerdict], criteria: list[Criterion]) -> EvaluationReport:
    return EvaluationReport(
        score=0.0,
        raw_score=0.0,
        report=[
            CriterionReport(
                weight=criteria[i].weight,
                requirement=criteria[i].requirement,
                verdict=v,
                reason="r",
            )
            for i, v in enumerate(verdicts)
        ],
    )


# =============================================================================
# _kappa_or_none
# =============================================================================


@pytest.mark.parametrize(
    ("y1", "y2", "weights", "expected"),
    [
        # Single-class data makes cohen_kappa_score return NaN with NO exception; the
        # helper must catch the NaN and return None (the latent bug the old `except: 0.0`
        # missed).
        ([0, 0, 0], [0, 0, 0], None, None),
        # Quadratic-weighted degenerate single-class → None.
        ([0, 0], [0, 0], "quadratic", None),
        # Normal varied input → a real float (perfect agreement → 1.0).
        ([0, 1, 0, 1], [0, 1, 0, 1], None, pytest.approx(1.0)),
    ],
)
def test_kappa_or_none(y1, y2, weights, expected):
    kwargs = {} if weights is None else {"weights": weights}
    k = _kappa_or_none(y1, y2, **kwargs)
    if expected is None:
        assert k is None
    else:
        assert k is not None
        assert k == expected


# =============================================================================
# Issue #1 — kappa → None for degenerate single-class criteria
# =============================================================================


def test_ordinal_degenerate_single_class_weighted_kappa_none():
    crit = _ordinal_criterion()
    m = _compute_ordinal_criterion_metrics([0, 0], [0, 0], crit, 0)
    assert m.weighted_kappa is None
    assert m.kappa_interpretation == "undefined"
    # Non-kappa metrics that are still defined remain floats.
    assert m.exact_accuracy == pytest.approx(1.0)
    assert m.n_samples == 2


def test_nominal_degenerate_single_class_kappa_none():
    crit = _nominal_criterion()
    m = _compute_nominal_criterion_metrics([0, 0], [0, 0], crit, 0)
    assert m.kappa is None
    assert m.kappa_interpretation == "undefined"
    assert m.exact_accuracy == pytest.approx(1.0)
    assert m.n_samples == 2


def test_binary_degenerate_single_class_kappa_none_via_compute_metrics():
    """A binary criterion where pred and true are all MET → kappa undefined → None."""
    crit = Criterion(name="acc", weight=10.0, requirement="accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    for i in range(3):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[CriterionVerdict.MET])
    reports = [_binary_report([CriterionVerdict.MET], [crit]) for _ in range(3)]
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    cm = metrics.per_criterion[0]
    assert cm.kappa is None
    assert cm.kappa_interpretation == "undefined"
    assert cm.n_samples == 3  # count stays


# =============================================================================
# Issue #3 — empty-data criteria → None (counts stay 0)
# =============================================================================


def test_ordinal_empty_data_metrics_none_counts_zero():
    crit = _ordinal_criterion()
    m = _compute_ordinal_criterion_metrics([], [], crit, 0)
    assert m.exact_accuracy is None
    assert m.adjacent_accuracy is None
    assert m.weighted_kappa is None
    assert m.rmse is None
    assert m.mae is None
    assert m.kappa_interpretation == "undefined"
    assert m.n_samples == 0
    assert m.per_option == []
    # Confusion matrix cells stay 0 (real counts).
    assert m.confusion_matrix == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_nominal_empty_data_metrics_none_counts_zero():
    crit = _nominal_criterion()
    m = _compute_nominal_criterion_metrics([], [], crit, 0)
    assert m.exact_accuracy is None
    assert m.kappa is None
    assert m.kappa_interpretation == "undefined"
    assert m.n_samples == 0
    assert m.per_option == []
    assert m.confusion_matrix == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_per_option_metrics_empty_precision_recall_f1_none_support_counts_real():
    crit = _ordinal_criterion()
    opts = _compute_per_option_metrics([], [], crit)
    assert len(opts) == 3
    for om in opts:
        assert om.precision is None
        assert om.recall is None
        assert om.f1 is None
        # Support counts are real (zero) counts, not None.
        assert om.support_true == 0
        assert om.support_pred == 0


def test_adjacent_accuracy_empty_returns_none():
    assert _compute_adjacent_accuracy([], []) is None
    # Non-empty path is still a float.
    assert _compute_adjacent_accuracy([0, 1], [0, 1]) == pytest.approx(1.0)


# =============================================================================
# mean_kappa excludes None (degenerate criterion not dragged toward 0)
# =============================================================================


def test_mean_kappa_excludes_none_degenerate_criterion():
    """2-criterion binary rubric where one criterion is degenerate (single-class, kappa
    None) and one is good. mean_kappa must equal the good kappa, NOT (good + 0.0) / 2."""
    from sklearn.metrics import cohen_kappa_score

    good = Criterion(name="good", weight=10.0, requirement="good")
    degen = Criterion(name="degen", weight=10.0, requirement="degen")
    ds = RubricDataset(prompt="p", rubric=Rubric([good, degen]), name="x")
    # good: ground truth varies MET/UNMET; degen: always MET (single-class).
    gts = [
        [CriterionVerdict.MET, CriterionVerdict.MET],
        [CriterionVerdict.UNMET, CriterionVerdict.MET],
        [CriterionVerdict.MET, CriterionVerdict.MET],
        [CriterionVerdict.UNMET, CriterionVerdict.MET],
    ]
    # predictions: good predicted perfectly; degen always MET (single-class).
    preds = [
        [CriterionVerdict.MET, CriterionVerdict.MET],
        [CriterionVerdict.UNMET, CriterionVerdict.MET],
        [CriterionVerdict.MET, CriterionVerdict.MET],
        [CriterionVerdict.UNMET, CriterionVerdict.MET],
    ]
    for i, gt in enumerate(gts):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=gt)
    reports = [_binary_report(p, [good, degen]) for p in preds]
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)

    good_kappa = float(
        cohen_kappa_score(
            ["MET", "UNMET", "MET", "UNMET"],
            ["MET", "UNMET", "MET", "UNMET"],
        )
    )
    # degen criterion contributes kappa None → excluded.
    degen_cm = metrics.per_criterion[1]
    assert degen_cm.kappa is None
    assert metrics.mean_kappa is not None
    assert metrics.mean_kappa == pytest.approx(good_kappa)


# =============================================================================
# Single-judge parity WITH a degenerate criterion
# =============================================================================


def test_single_judge_parity_with_degenerate_criterion():
    """KEY INVARIANT (degenerate variant): a 1-judge "ensemble" over a mixed rubric where
    ONE criterion is single-class (kappa None) still yields per_judge[only] equal to the
    aggregate field-for-field (both None in that slot, _mean_or_none excludes identically)."""
    binc = Criterion(name="acc", weight=10.0, requirement="accurate")
    ordc = _ordinal_criterion()
    nomc = _nominal_criterion()
    ds = RubricDataset(prompt="p", rubric=Rubric([binc, ordc, nomc]), name="x")

    # ordinal criterion is DEGENERATE: ground truth + pred both always "High" (index 2).
    gts = [
        [CriterionVerdict.MET, "High", "Alpha"],
        [CriterionVerdict.UNMET, "High", "Beta"],
        [CriterionVerdict.MET, "High", "Gamma"],
    ]
    for i, gt in enumerate(gts):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=gt)

    ja_binary = [CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.MET]
    ja_ordinal = [2, 2, 2]  # single-class → weighted kappa None
    ja_nominal = [0, 1, 0]  # vs true [0, 1, 2]
    reports = []
    for i in range(3):
        reports.append(
            _multi_criterion_ensemble_report(
                [binc, ordc, nomc],
                [
                    [("ja", ja_binary[i])],
                    [("ja", ja_ordinal[i])],
                    [("ja", ja_nominal[i])],
                ],
                ["ja"],
            )
        )
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds, per_judge=True)
    assert metrics.per_judge is not None
    only = metrics.per_judge["ja"]

    # The ordinal criterion is degenerate → its weighted kappa is None.
    ord_cm = metrics.per_criterion[1]
    assert ord_cm.weighted_kappa is None

    # Field-for-field equality with the aggregate (the single judge IS the aggregate),
    # including mean_kappa where the degenerate criterion is excluded identically.
    assert only.criterion_accuracy == pytest.approx(metrics.criterion_accuracy)
    assert metrics.mean_kappa is not None
    assert only.mean_kappa is not None
    assert only.mean_kappa == pytest.approx(metrics.mean_kappa)
    assert metrics.criterion_precision is not None
    assert only.criterion_precision == pytest.approx(metrics.criterion_precision)
    assert only.criterion_recall == pytest.approx(metrics.criterion_recall)
    assert only.criterion_f1 == pytest.approx(metrics.criterion_f1)


# =============================================================================
# Round-trip with None per-criterion fields
# =============================================================================


def test_to_file_roundtrip_with_none_fields(tmp_path):
    """to_file / model_dump_json must not raise when per-criterion fields are None."""
    crit = Criterion(name="acc", weight=10.0, requirement="accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    for i in range(3):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[CriterionVerdict.MET])
    reports = [_binary_report([CriterionVerdict.MET], [crit]) for _ in range(3)]
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    assert metrics.per_criterion[0].kappa is None  # degenerate

    out = tmp_path / "metrics.json"
    metrics.to_file(out)
    assert out.exists()
    # summary() must not raise either.
    assert isinstance(metrics.summary(), str)


# =============================================================================
# Bootstrap CI undefined → None
# =============================================================================


def test_bootstrap_ci_empty_returns_none():
    res = _compute_bootstrap_ci(
        [],  # per_criterion_pred
        [],  # per_criterion_true
        [],  # criterion_types
        [],  # effective_criteria
        "exclude",  # cannot_assess
        "exclude",  # na_mode
        [],  # true_scores
        [],  # pred_scores
        n_bootstrap=10,
        confidence_level=0.95,
        seed=0,
    )
    assert res.accuracy_ci is None
    assert res.kappa_ci is None
    assert res.rmse_ci is None


def test_bootstrap_ci_empty_verdicts_nonempty_scores_n1_rmse():
    """Two independent axes: an empty verdict axis leaves accuracy/kappa None, while a single
    scored item still yields a degenerate-but-real ``(v, v)`` rmse_ci (RMSE is defined n>=1)."""
    res = _compute_bootstrap_ci(
        [],  # per_criterion_pred (empty verdict axis)
        [],
        [],
        [],
        "exclude",
        "exclude",
        [0.5],  # true_scores (single scored item)
        [0.3],  # pred_scores
        n_bootstrap=50,
        confidence_level=0.95,
        seed=0,
    )
    assert res.accuracy_ci is None
    assert res.kappa_ci is None
    assert res.rmse_ci is not None
    assert res.rmse_ci[0] == res.rmse_ci[1]  # n=1 → every resample identical


def test_bootstrap_ci_single_class_kappa_none_summary_ok():
    """When kappa is never defined across bootstrap resamples (single-class), kappa_ci is
    None and summary() does not raise."""
    crit = Criterion(name="acc", weight=10.0, requirement="accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    for i in range(4):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[CriterionVerdict.MET])
    reports = [_binary_report([CriterionVerdict.MET], [crit]) for _ in range(4)]
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(
        _eval_result(item_results), ds, bootstrap=True, n_bootstrap=20, seed=0
    )
    assert metrics.bootstrap is not None
    # Single-class → kappa never defined across resamples → None.
    assert metrics.bootstrap.kappa_ci is None
    # summary() must None-guard the CI lines.
    assert isinstance(metrics.summary(), str)


# =============================================================================
# Issue #2 — correlation undefined → None (constant array / <3 samples)
# =============================================================================


def test_correlation_constant_array_coefficient_none():
    """A constant input array makes spearmanr/kendalltau/pearsonr return NaN (zero
    variance → correlation genuinely undefined). The coefficient must be None, not a fake
    0.0; p_value None; interpretation a non-crashing 'undefined' string."""
    res = _compute_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], "spearman")
    assert res.coefficient is None
    assert res.p_value is None
    assert res.interpretation == "undefined"
    assert res.n_samples == 3
    assert res.method == "spearman"


def test_correlation_constant_array_suppresses_scipy_warning():
    """A constant input array is a handled case (coefficient → None), so scipy's
    ConstantInputWarning / NearConstantInputWarning must NOT leak to the caller — it is
    suppressed locally around the scipy call. Behavior (None coefficient) is unchanged."""
    for method in ("spearman", "kendall", "pearson"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _compute_correlation([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0], method)
        assert res.coefficient is None, method
        assert res.interpretation == "undefined", method
        leaked = [w for w in caught if "constant" in str(w.message).lower()]
        assert not leaked, (method, [str(w.message) for w in leaked])


def test_correlation_fewer_than_three_samples_coefficient_none():
    """With <3 samples a correlation coefficient is genuinely undefined → None (not 0.0).
    The interpretation stays the honest 'insufficient data' string."""
    res = _compute_correlation([1.0, 2.0], [1.0, 2.0], "pearson")
    assert res.coefficient is None
    assert res.p_value is None
    assert res.interpretation == "insufficient data"
    assert res.n_samples == 2
    assert res.method == "pearson"


def test_correlation_varied_input_returns_float():
    """A normal varied input still yields a real float coefficient (no regression)."""
    res = _compute_correlation([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], "spearman")
    assert res.coefficient is not None
    assert res.coefficient == pytest.approx(1.0)
    assert res.p_value is not None
    assert res.interpretation != "undefined"


def test_compute_metrics_constant_predicted_scores_score_spearman_none():
    """When every predicted score is identical (constant array), the score-level Spearman
    coefficient is undefined → None, and summary() renders it as 'n/a' without raising."""
    # A negative-weight criterion: ground truth varies (MET/UNMET) so true scores vary,
    # but predictions are all MET so predicted scores are constant → zero variance on the
    # predicted axis → Spearman/Pearson undefined.
    crit = Criterion(name="acc", weight=10.0, requirement="accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    gts = [
        [CriterionVerdict.MET],
        [CriterionVerdict.UNMET],
        [CriterionVerdict.MET],
    ]
    for i, gt in enumerate(gts):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=gt)
    # All predictions MET → all predicted scores identical (1.0) → constant array.
    reports = [_binary_report([CriterionVerdict.MET], [crit]) for _ in range(3)]
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    metrics = compute_metrics(_eval_result(item_results), ds)
    assert metrics.score_spearman.coefficient is None
    assert metrics.score_spearman.p_value is None
    assert metrics.score_spearman.interpretation == "undefined"
    # summary() must None-guard the score-level correlation lines and render n/a.
    out = metrics.summary()
    assert isinstance(out, str)
    assert "n/a" in out


def test_ordinal_empty_data_spearman_kendall_coefficient_none():
    """The ordinal n_samples==0 empty-data sentinels carry coefficient None (Issue #2
    consistency now that CorrelationResult.coefficient is Optional)."""
    crit = _ordinal_criterion()
    m = _compute_ordinal_criterion_metrics([], [], crit, 0)
    assert m.spearman.coefficient is None
    assert m.spearman.p_value is None
    assert m.kendall.coefficient is None
    assert m.kendall.p_value is None
    # The empty-data interpretation stays the honest "insufficient data" string.
    assert m.spearman.interpretation == "insufficient data"
    assert m.kendall.interpretation == "insufficient data"
