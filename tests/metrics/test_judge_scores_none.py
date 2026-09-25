"""Per-judge score metrics when ``EnsembleEvaluationReport.judge_scores`` holds ``None``.

A ``None`` judge score means the judge's whole-rubric score is undefined for its role (a
judge consulted only on some criteria is ``None`` on every item). A judge with no defined
score on any item has undefined score-level metrics: every score field of its
``JudgeMetrics`` (``score_rmse``, ``score_mae``, ``score_spearman``, ``score_kendall``,
``score_pearson``, ``bias``) is ``None``, while its criterion-level metrics are still
computed from its votes. A ``None`` score on only some items (e.g. hand-built reports) is
excluded from that judge's score pairs, as the aggregate excludes score-less items. Judges
whose scores are all defined are unaffected. A judge *absent* from some items'
``judge_scores`` (the judge set changed between items) has no entry rather than a ``None``
entry; per-judge metrics reject that case with an error naming the judge, and the other
metrics stay available.
"""

from __future__ import annotations

import pytest

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import compute_metrics
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

CRITERIA = [
    Criterion(name="c0", requirement="r0", weight=1.0),
    Criterion(name="c1", requirement="r1", weight=1.0),
]
GROUND_TRUTH = [[MET, MET], [UNMET, UNMET], [MET, UNMET], [UNMET, MET]]
JUDGE_A = [[MET, MET], [UNMET, UNMET], [MET, MET], [UNMET, MET]]
JUDGE_B = [[MET, UNMET], [UNMET, UNMET], [MET, UNMET], [MET, MET]]
SCORES_A = [1.0, 0.0, 1.0, 0.5]


def _dataset() -> RubricDataset:
    dataset = RubricDataset(prompt="p", rubric=Rubric(CRITERIA), name="judge-scores-none")
    for idx, gt in enumerate(GROUND_TRUTH):
        dataset.add_item(submission=f"s{idx}", description=f"i{idx}", ground_truth=gt)
    return dataset


def _result(scores_b: list[float | None]) -> EvalResult:
    items = []
    for i in range(len(GROUND_TRUTH)):
        reports = [
            EnsembleCriterionReport(
                criterion=c,
                final_verdict=JUDGE_A[i][c_idx],
                final_reason="agg",
                votes=[
                    JudgeVote(judge_id="a", verdict=JUDGE_A[i][c_idx], reason="a"),
                    JudgeVote(judge_id="b", verdict=JUDGE_B[i][c_idx], reason=None),
                ],
            )
            for c_idx, c in enumerate(CRITERIA)
        ]
        report = EnsembleEvaluationReport(
            score=SCORES_A[i],
            raw_score=SCORES_A[i] * 2,
            report=reports,
            judge_scores={"a": SCORES_A[i], "b": scores_b[i]},
        )
        items.append(ItemResult(item_idx=i, item=None, report=report, duration_seconds=0.1))
    return EvalResult(
        item_results=items,
        total_items=len(items),
        successful_items=len(items),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=None,
        started_at=None,
        completed_at=None,
    )


def test_judge_without_any_score_has_undefined_score_metrics():
    metrics = compute_metrics(_result([None] * 4), _dataset(), per_judge=True)
    assert metrics.per_judge is not None
    jb = metrics.per_judge["b"]
    assert jb.score_rmse is None
    assert jb.score_mae is None
    assert jb.score_spearman is None
    assert jb.score_kendall is None
    assert jb.score_pearson is None
    assert jb.bias is None
    # Criterion-level metrics come from the votes and stay defined.
    assert jb.criterion_accuracy == pytest.approx(6 / 8)


def test_judge_with_scores_keeps_defined_score_metrics():
    metrics = compute_metrics(_result([None] * 4), _dataset(), per_judge=True)
    assert metrics.per_judge is not None
    ja = metrics.per_judge["a"]
    assert ja.score_rmse is not None
    assert ja.score_mae is not None
    assert ja.score_spearman is not None and ja.score_spearman.n_samples == 4
    assert ja.score_kendall is not None and ja.score_kendall.n_samples == 4
    assert ja.score_pearson is not None and ja.score_pearson.n_samples == 4
    assert ja.bias is not None and ja.bias.n_samples == 4


def test_judge_with_all_scores_is_unaffected_by_another_judges_none():
    with_none = compute_metrics(_result([None] * 4), _dataset(), per_judge=True)
    with_scores = compute_metrics(_result([0.5, 0.0, 0.5, 1.0]), _dataset(), per_judge=True)
    assert with_none.per_judge is not None and with_scores.per_judge is not None
    assert with_none.per_judge["a"] == with_scores.per_judge["a"]


def test_partially_undefined_scores_use_the_defined_items_only():
    partial = compute_metrics(_result([0.5, None, 0.5, 1.0]), _dataset(), per_judge=True)
    assert partial.per_judge is not None
    jb = partial.per_judge["b"]
    # Defined pairs: items 0, 2, 3 with true scores 1.0, 0.5, 0.5 -> errors -0.5, 0.0, 0.5.
    assert jb.score_mae == pytest.approx((0.5 + 0.0 + 0.5) / 3)
    assert jb.score_rmse == pytest.approx(((0.25 + 0.0 + 0.25) / 3) ** 0.5)
    assert jb.bias is not None and jb.bias.n_samples == 3
    assert jb.score_spearman is not None and jb.score_spearman.n_samples == 3
    assert jb.score_kendall is not None and jb.score_kendall.n_samples == 3
    assert jb.score_pearson is not None and jb.score_pearson.n_samples == 3


def test_summary_renders_undefined_score_metrics_as_na():
    metrics = compute_metrics(_result([None] * 4), _dataset(), per_judge=True)
    text = metrics.summary(verbose=True)
    b_line = next(line for line in text.splitlines() if line.strip().startswith("RMSE=n/a"))
    assert b_line.strip() == "RMSE=n/a, Spearman=n/a, MAE=n/a"


def test_dataframe_judge_row_has_none_for_undefined_score_metrics():
    pytest.importorskip("pandas")
    metrics = compute_metrics(_result([None] * 4), _dataset(), per_judge=True)
    assert metrics.per_judge is not None
    df = metrics.to_dataframe()
    rows = df[(df["level"] == "judge") & (df["name"] == "b")]
    assert len(rows) == 1
    row = rows.iloc[0]
    for column in ("rmse", "mae", "spearman", "kendall", "pearson", "bias"):
        assert row[column] is None or row[column] != row[column], column  # None or NaN
    a_row = df[(df["level"] == "judge") & (df["name"] == "a")].iloc[0]
    assert a_row["rmse"] == pytest.approx(metrics.per_judge["a"].score_rmse)


def _mixed_judge_sets_result() -> EvalResult:
    """Judges ``a`` and ``b`` grade items 0-1, judges ``a`` and ``c`` items 2-3, as when a run
    is resumed after swapping a judge. ``b`` and ``c`` are absent from the other items'
    ``judge_scores`` (no entry at all), which is not the same as a ``None`` entry."""
    items = []
    for i in range(len(GROUND_TRUTH)):
        second = "b" if i < 2 else "c"
        reports = [
            EnsembleCriterionReport(
                criterion=c,
                final_verdict=JUDGE_A[i][c_idx],
                final_reason="agg",
                votes=[
                    JudgeVote(judge_id="a", verdict=JUDGE_A[i][c_idx], reason="a"),
                    JudgeVote(judge_id=second, verdict=JUDGE_B[i][c_idx], reason="b"),
                ],
            )
            for c_idx, c in enumerate(CRITERIA)
        ]
        report = EnsembleEvaluationReport(
            score=SCORES_A[i],
            raw_score=SCORES_A[i] * 2,
            report=reports,
            judge_scores={"a": SCORES_A[i], second: 0.5},
        )
        items.append(ItemResult(item_idx=i, item=None, report=report, duration_seconds=0.1))
    return EvalResult(
        item_results=items,
        total_items=len(items),
        successful_items=len(items),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=None,
        started_at=None,
        completed_at=None,
    )


def test_judge_absent_from_some_items_is_rejected_by_name():
    """Per-judge metrics do not support a judge set that changed between items: the error
    names the first such judge and how many scored items it is on."""
    with pytest.raises(
        ValueError,
        match=r"^per_judge=True needs every scored item graded by the same judges, but "
        r"judge 'b' is in the judge_scores of 2 of 4 scored items",
    ):
        compute_metrics(_mixed_judge_sets_result(), _dataset(), per_judge=True)


def test_judge_absent_from_some_items_leaves_the_other_metrics_available():
    metrics = compute_metrics(_mixed_judge_sets_result(), _dataset())
    assert metrics.per_judge is None
    # The aggregate compares the final verdicts (judge a's) with the ground truth.
    assert metrics.criterion_accuracy == pytest.approx(7 / 8)
