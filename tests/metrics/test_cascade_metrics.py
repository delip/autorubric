"""Cascade-aware metrics: missing votes, escalation-judge coverage, Fleiss and alpha.

A confidence cascade grades each item with one decision model and escalates some criteria
to LLM judges. Its reports have this shape (hand-built here, as the grader emits them):

- a criterion that was not escalated carries only the decision model's vote, and its final
  verdict is that vote;
- an escalated criterion (``escalated=True``) carries the decision model's vote with
  ``superseded=True``, followed by the escalation judges' votes, which alone produce the
  final verdict;
- ``judge_scores`` holds the decision model's own whole-rubric score and ``None`` for each
  escalation judge (undefined by role).

An escalation judge therefore has no vote on the criteria that were not escalated. Those
cells are missing: they are neither a verdict (binary ``UNMET``) nor an abstention (a
multi-choice NA), so they are left out of the judge's criterion-level metrics and never
trigger NA-option reconstruction. The decision model's superseded votes are its
predictions. Fleiss' kappa skips superseded votes (no cascade row is complete, so it is
``None``), while Krippendorff's alpha keeps them.
"""

from __future__ import annotations

import pytest

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import JudgeMetrics, MetricsResult, compute_metrics
from autorubric.rubric import Rubric
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    Criterion,
    CriterionOption,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET

DM = "jev"
ESC = "escalation"


def _eval_result(reports: list[EnsembleEvaluationReport]) -> EvalResult:
    items = [
        ItemResult(item_idx=i, item=None, report=report, duration_seconds=0.1)
        for i, report in enumerate(reports)
    ]
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


def _dataset(criteria: list[Criterion], ground_truth: list[list]) -> RubricDataset:
    dataset = RubricDataset(prompt="p", rubric=Rubric(criteria), name="cascade")
    for i, gt in enumerate(ground_truth):
        dataset.add_item(submission=f"s{i}", description=f"d{i}", ground_truth=gt)
    return dataset


# =============================================================================
# Binary cascade fixture
# =============================================================================

BIN_CRITERIA = [
    Criterion(name="accurate", requirement="Is accurate", weight=1.0),
    Criterion(name="complete", requirement="Is complete", weight=1.0),
]
BIN_GT = [[MET, UNMET], [UNMET, MET], [MET, MET], [UNMET, MET]]
# The decision model's vote on every cell (superseded where escalated).
BIN_DM = [[MET, MET], [MET, UNMET], [UNMET, MET], [UNMET, UNMET]]
# Escalated cells -> the escalation judge's vote there.
BIN_ESC = {(0, 0): MET, (1, 0): UNMET, (2, 0): UNMET, (3, 1): MET}


def _binary_cascade_reports(
    esc_errors: dict[tuple[int, int], str] | None = None,
) -> list[EnsembleEvaluationReport]:
    esc_errors = esc_errors or {}
    reports = []
    for i, dm_row in enumerate(BIN_DM):
        crs = []
        for c, criterion in enumerate(BIN_CRITERIA):
            escalated = (i, c) in BIN_ESC
            dm_vote = JudgeVote(
                judge_id=DM,
                verdict=dm_row[c],
                reason=None,
                probabilities={"MET": 0.6, "UNMET": 0.4},
                confidence=0.2 if escalated else 0.9,
                superseded=escalated,
            )
            if escalated:
                esc_vote = JudgeVote(
                    judge_id=ESC,
                    verdict=BIN_ESC[(i, c)],
                    reason="llm",
                    error=esc_errors.get((i, c)),
                )
                votes = [dm_vote, esc_vote]
                final = BIN_ESC[(i, c)]
            else:
                votes = [dm_vote]
                final = dm_row[c]
            crs.append(
                EnsembleCriterionReport(
                    criterion=criterion,
                    final_verdict=final,
                    final_reason="agg",
                    votes=votes,
                    escalated=escalated,
                )
            )
        score = sum(1.0 for cr in crs if cr.final_verdict == MET) / len(crs)
        dm_score = sum(1.0 for v in dm_row if v == MET) / len(dm_row)
        reports.append(
            EnsembleEvaluationReport(
                score=score,
                raw_score=score * len(crs),
                report=crs,
                judge_scores={DM: dm_score, ESC: None},
            )
        )
    return reports


def _binary_metrics(**kwargs) -> MetricsResult:
    return compute_metrics(
        _eval_result(_binary_cascade_reports()),
        _dataset(BIN_CRITERIA, BIN_GT),
        per_judge=True,
        **kwargs,
    )


# =============================================================================
# Multi-choice cascade fixture (no NA option anywhere)
# =============================================================================

QUALITY = Criterion(
    name="quality",
    requirement="Overall quality",
    weight=1.0,
    scale_type="ordinal",
    options=[
        CriterionOption(label="Low", value=0.0),
        CriterionOption(label="Mid", value=0.5),
        CriterionOption(label="High", value=1.0),
    ],
)
REGISTER = Criterion(
    name="register",
    requirement="Which register?",
    weight=1.0,
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Casual", value=0.5),
        CriterionOption(label="Slang", value=0.0),
    ],
)
MC_CRITERIA = [QUALITY, REGISTER]
MC_GT = [["High", "Formal"], ["Mid", "Casual"], ["Low", "Slang"], ["Mid", "Formal"]]
MC_DM = [[2, 0], [0, 1], [1, 0], [1, 2]]
MC_ESC = {(1, 0): 1, (2, 0): 0, (3, 1): 1}


def _mc_vote(judge_id: str, criterion: Criterion, idx: int, **kwargs) -> MultiChoiceJudgeVote:
    option = criterion.options[idx]
    return MultiChoiceJudgeVote(
        judge_id=judge_id,
        selected_index=idx,
        selected_label=option.label,
        value=option.value,
        **kwargs,
    )


def _mc_final(criterion: Criterion, idx: int) -> AggregatedMultiChoiceVerdict:
    option = criterion.options[idx]
    return AggregatedMultiChoiceVerdict(
        selected_index=idx,
        selected_label=option.label,
        value=option.value,
        aggregated_value=option.value,
    )


def _mc_cascade_reports() -> list[EnsembleEvaluationReport]:
    reports = []
    for i, dm_row in enumerate(MC_DM):
        crs = []
        for c, criterion in enumerate(MC_CRITERIA):
            escalated = (i, c) in MC_ESC
            dm_vote = _mc_vote(
                DM, criterion, dm_row[c], reason=None, confidence=0.1, superseded=escalated
            )
            if escalated:
                final_idx = MC_ESC[(i, c)]
                votes = [dm_vote, _mc_vote(ESC, criterion, final_idx, reason="llm")]
            else:
                final_idx = dm_row[c]
                votes = [dm_vote]
            crs.append(
                EnsembleCriterionReport(
                    criterion=criterion,
                    final_verdict=None,
                    final_reason="agg",
                    final_multi_choice_verdict=_mc_final(criterion, final_idx),
                    multi_choice_votes=votes,
                    escalated=escalated,
                )
            )
        score = sum(cr.score_value for cr in crs) / len(crs)
        dm_score = sum(MC_CRITERIA[c].options[dm_row[c]].value for c in range(2)) / 2
        reports.append(
            EnsembleEvaluationReport(
                score=score,
                raw_score=score * 2,
                report=crs,
                judge_scores={DM: dm_score, ESC: None},
            )
        )
    return reports


def _mc_metrics(**kwargs) -> MetricsResult:
    return compute_metrics(
        _eval_result(_mc_cascade_reports()),
        _dataset(MC_CRITERIA, MC_GT),
        per_judge=True,
        **kwargs,
    )


# =============================================================================
# Missing votes are missing
# =============================================================================


def test_missing_binary_votes_are_skipped_not_unmet():
    """The escalation judge is measured on its four escalated cells only (3 of 4 correct).

    Filling its four missing cells with UNMET would add 2 correct and 2 wrong cells (5/8).
    """
    esc = _binary_metrics().per_judge[ESC]
    assert esc.criterion_accuracy == pytest.approx(3 / 4)
    assert esc.confusion_matrix is not None
    assert sum(sum(row) for row in esc.confusion_matrix.matrix) == 4
    # Genuine MET votes: (0, accurate) and (3, complete), both true MET.
    assert esc.confusion_matrix.matrix[0][0] == 2
    assert esc.criterion_recall == pytest.approx(2 / 3)


def test_missing_multi_choice_votes_are_not_na_predictions():
    """Missing multi-choice cells neither count as NA predictions nor add an NA option.

    The escalation judge has 3 genuine votes (2 exact). Counting its 5 missing cells as NA
    predictions would give 2/8, and would reconstruct an NA option on both criteria.
    """
    metrics = _mc_metrics()
    esc = metrics.per_judge[ESC]
    assert esc.criterion_accuracy == pytest.approx(2 / 3)
    for cm, criterion in zip(metrics.per_criterion, MC_CRITERIA, strict=True):
        assert cm.confusion_matrix.labels == [o.label for o in criterion.options]
    assert metrics.na_stats is not None
    assert metrics.na_stats.na_count_pred == 0


@pytest.mark.parametrize("na_mode", ["as_unmet", "as_category"])
def test_missing_multi_choice_votes_are_ignored_under_every_na_mode(na_mode):
    """``as_unmet`` would remap a fabricated NA to the worst option and count it;
    ``as_category`` would refuse the ordinal criterion once a fabricated NA option exists."""
    esc = _mc_metrics(na_mode=na_mode).per_judge[ESC]
    assert esc.criterion_accuracy == pytest.approx(2 / 3)
    assert esc.n_pairs == 3


def test_decision_model_is_measured_on_every_criterion_with_its_superseded_votes():
    metrics = _binary_metrics()
    dm = metrics.per_judge[DM]
    assert dm.coverage == "full"
    assert dm.n_pairs is None
    # 3 of 8 of its verdicts (superseded included) match the ground truth.
    assert dm.criterion_accuracy == pytest.approx(3 / 8)
    assert dm.confusion_matrix is not None
    assert sum(sum(row) for row in dm.confusion_matrix.matrix) == 8
    # Its whole-rubric score is defined, so its score-level metrics are too.
    assert dm.score_rmse is not None
    assert dm.bias is not None

    mc_dm = _mc_metrics().per_judge[DM]
    assert mc_dm.criterion_accuracy == pytest.approx(4 / 8)


def test_aggregate_metrics_use_the_final_verdicts():
    metrics = _binary_metrics()
    assert metrics.criterion_accuracy == pytest.approx(5 / 8)


# =============================================================================
# Escalation-judge coverage
# =============================================================================


def test_escalation_judge_coverage_and_pairs():
    esc = _binary_metrics().per_judge[ESC]
    assert esc.coverage == "escalated"
    assert esc.n_pairs == 4
    for field in (
        "score_rmse",
        "score_mae",
        "score_spearman",
        "score_kendall",
        "score_pearson",
        "bias",
    ):
        assert getattr(esc, field) is None, field


def test_n_pairs_leaves_out_errored_escalation_votes():
    reports = _binary_cascade_reports(esc_errors={(2, 0): "infrastructure: timeout"})
    metrics = compute_metrics(_eval_result(reports), _dataset(BIN_CRITERIA, BIN_GT), per_judge=True)
    esc = metrics.per_judge[ESC]
    assert esc.n_pairs == 3
    assert esc.criterion_accuracy == pytest.approx(3 / 3)


def test_n_pairs_counts_abstentions_before_the_handling_mode():
    """``n_pairs`` is the escalated subset's size; CANNOT_ASSESS handling applies after."""
    reports = _binary_cascade_reports()
    first = reports[0].report[0]
    abstaining = first.model_copy(
        update={
            "votes": [
                first.votes[0],
                JudgeVote(judge_id=ESC, verdict=CriterionVerdict.CANNOT_ASSESS, reason="?"),
            ]
        }
    )
    reports[0] = reports[0].model_copy(update={"report": [abstaining, reports[0].report[1]]})
    dataset = _dataset(BIN_CRITERIA, BIN_GT)

    excluded = compute_metrics(_eval_result(reports), dataset, per_judge=True).per_judge[ESC]
    assert excluded.n_pairs == 4
    assert excluded.criterion_accuracy == pytest.approx(2 / 3)

    as_unmet = compute_metrics(
        _eval_result(reports), dataset, per_judge=True, cannot_assess="as_unmet"
    ).per_judge[ESC]
    assert as_unmet.n_pairs == 4
    assert as_unmet.criterion_accuracy == pytest.approx(2 / 4)


def test_escalation_judge_consulted_on_no_criterion_has_zero_pairs():
    reports = []
    for dm_row in BIN_DM:
        crs = [
            EnsembleCriterionReport(
                criterion=criterion,
                final_verdict=dm_row[c],
                final_reason=None,
                votes=[JudgeVote(judge_id=DM, verdict=dm_row[c], reason=None)],
            )
            for c, criterion in enumerate(BIN_CRITERIA)
        ]
        score = sum(1.0 for v in dm_row if v == MET) / 2
        reports.append(
            EnsembleEvaluationReport(
                score=score, raw_score=score * 2, report=crs, judge_scores={DM: score, ESC: None}
            )
        )
    metrics = compute_metrics(_eval_result(reports), _dataset(BIN_CRITERIA, BIN_GT), per_judge=True)
    esc = metrics.per_judge[ESC]
    assert esc.coverage == "escalated"
    assert esc.n_pairs == 0
    assert esc.criterion_accuracy is None
    assert esc.mean_kappa is None
    assert esc.confusion_matrix is None
    assert "(escalated subset, 0 pairs)" in metrics.summary()


def test_every_criterion_escalated_is_still_an_escalation_subset():
    """Coverage follows the judge's role (no whole-rubric score on any item), not how many
    criteria happened to escalate: a run where every criterion escalated (e.g. after the
    decision model failed on every item) is still reported as the escalated subset."""
    reports = []
    for i in range(len(BIN_GT)):
        crs = []
        for c, criterion in enumerate(BIN_CRITERIA):
            esc_verdict = BIN_GT[i][c]
            crs.append(
                EnsembleCriterionReport(
                    criterion=criterion,
                    final_verdict=esc_verdict,
                    final_reason="agg",
                    votes=[
                        JudgeVote(
                            judge_id=DM,
                            verdict=CriterionVerdict.CANNOT_ASSESS,
                            reason=None,
                            error="infrastructure: down",
                            superseded=True,
                        ),
                        JudgeVote(judge_id=ESC, verdict=esc_verdict, reason="llm"),
                    ],
                    escalated=True,
                )
            )
        score = sum(1.0 for cr in crs if cr.final_verdict == MET) / 2
        reports.append(
            EnsembleEvaluationReport(
                score=score, raw_score=score * 2, report=crs, judge_scores={DM: 0.0, ESC: None}
            )
        )
    metrics = compute_metrics(_eval_result(reports), _dataset(BIN_CRITERIA, BIN_GT), per_judge=True)
    esc = metrics.per_judge[ESC]
    assert esc.coverage == "escalated"
    assert esc.n_pairs == 8
    assert esc.criterion_accuracy == pytest.approx(1.0)
    # The decision model's votes all errored: none enters its criterion-level metrics.
    assert metrics.per_judge[DM].coverage == "full"
    assert metrics.per_judge[DM].criterion_accuracy is None


def test_summary_labels_the_escalated_subset():
    """The escalation judge's cells: ``accurate`` pred (MET, UNMET, UNMET) vs true (MET,
    UNMET, MET) gives kappa 0.4 (``complete`` has one cell, kappa undefined); pooled, 2 true
    positives, 1 false negative and 1 true negative give phi = 2/sqrt(12)."""
    metrics = _binary_metrics()
    lines = metrics.summary().splitlines()
    assert "  escalation: Acc=75.0%, Mean Kappa=0.400, Phi=0.577 (escalated subset, 4 pairs)" in (
        lines
    )
    # A full-coverage judge's line is unchanged.
    assert "  jev: Acc=37.5%, Mean Kappa=-0.250, Phi=-0.258" in lines
    verbose = metrics.summary(verbose=True).splitlines()
    esc_at = verbose.index(
        "  escalation: Acc=75.0%, Mean Kappa=0.400, Phi=0.577 (escalated subset, 4 pairs)"
    )
    assert verbose[esc_at + 1] == "      RMSE=n/a, Spearman=n/a, MAE=n/a"


# =============================================================================
# Inter-judge agreement
# =============================================================================


def test_fleiss_is_none_for_a_cascade():
    """Fleiss skips superseded votes, so no cascade row has every judge's vote."""
    for cm in _binary_metrics().per_criterion:
        assert cm.fleiss_kappa is None
    for cm in _mc_metrics().per_criterion:
        assert cm.fleiss_kappa is None


def test_krippendorff_alpha_keeps_superseded_votes():
    """On ``accurate`` the pairable units are the escalated items 0-2, where the decision
    model's superseded vote and the escalation vote are (MET, MET), (MET, UNMET) and
    (UNMET, UNMET); item 3 has one rating and is not pairable.

    Coincidences (nominal): o(MET,MET)=2, o(MET,UNMET)=o(UNMET,MET)=1, o(UNMET,UNMET)=2,
    so n_MET = n_UNMET = 3 and n = 6. D_o = 2/6, D_e = (3*3 + 3*3)/(6*5) = 18/30, and
    alpha = 1 - D_o/D_e = 1 - (1/3)/(3/5) = 4/9. Without the superseded votes no unit would
    have two ratings and alpha would be undefined.
    """
    pytest.importorskip("krippendorff")
    accurate = _binary_metrics().per_criterion[0]
    assert accurate.krippendorff_alpha == pytest.approx(4 / 9)


def test_mixed_ensemble_without_escalation_is_an_ordinary_panel():
    """A decision model and an LLM judging every criterion form a full panel: both judges
    have full coverage and Fleiss is defined."""
    reports = []
    for i, dm_row in enumerate(BIN_DM):
        crs = []
        for c, criterion in enumerate(BIN_CRITERIA):
            llm_verdict = BIN_GT[i][c]
            votes = [
                JudgeVote(judge_id=DM, verdict=dm_row[c], reason=None, confidence=0.5),
                JudgeVote(judge_id="llm", verdict=llm_verdict, reason="llm"),
            ]
            crs.append(
                EnsembleCriterionReport(
                    criterion=criterion, final_verdict=llm_verdict, final_reason="agg", votes=votes
                )
            )
        dm_score = sum(1.0 for v in dm_row if v == MET) / 2
        llm_score = sum(1.0 for v in BIN_GT[i] if v == MET) / 2
        reports.append(
            EnsembleEvaluationReport(
                score=llm_score,
                raw_score=llm_score * 2,
                report=crs,
                judge_scores={DM: dm_score, "llm": llm_score},
            )
        )
    metrics = compute_metrics(_eval_result(reports), _dataset(BIN_CRITERIA, BIN_GT), per_judge=True)
    for judge in (DM, "llm"):
        assert metrics.per_judge[judge].coverage == "full"
        assert metrics.per_judge[judge].n_pairs is None
    assert metrics.per_judge["llm"].criterion_accuracy == pytest.approx(1.0)
    pytest.importorskip("statsmodels")
    assert any(cm.fleiss_kappa is not None for cm in metrics.per_criterion)


# =============================================================================
# Mixed judge sets (a run resumed after the judges changed)
# =============================================================================


def test_mixed_judge_sets_do_not_fabricate_an_na_option():
    """Judges ``a``/``b`` grade items 0-1 and ``a``/``c`` items 2-3. ``b`` has no vote on
    items 2-3: a missing cell, not an abstention, so no NA option is reconstructed."""
    reports = []
    for i, (a_idx, other_idx) in enumerate([(0, 0), (1, 2), (2, 2), (2, 1)]):
        other = "b" if i < 2 else "c"
        cr = EnsembleCriterionReport(
            criterion=QUALITY,
            final_verdict=None,
            final_reason="agg",
            final_multi_choice_verdict=_mc_final(QUALITY, a_idx),
            multi_choice_votes=[
                _mc_vote("a", QUALITY, a_idx, reason="a"),
                _mc_vote(other, QUALITY, other_idx, reason=other),
            ],
        )
        value = QUALITY.options[a_idx].value
        reports.append(
            EnsembleEvaluationReport(
                score=value,
                raw_score=value,
                report=[cr],
                judge_scores={"a": value, other: QUALITY.options[other_idx].value},
            )
        )
    metrics = compute_metrics(
        _eval_result(reports), _dataset([QUALITY], [["Low"], ["Mid"], ["High"], ["Mid"]])
    )
    quality = metrics.per_criterion[0]
    assert quality.confusion_matrix.labels == ["Low", "Mid", "High"]
    assert [o.label for o in quality.per_option] == ["Low", "Mid", "High"]


# =============================================================================
# Serialization and tabular export
# =============================================================================


def test_full_coverage_judge_dump_omits_the_cascade_fields():
    dm_dump = _binary_metrics().per_judge[DM].model_dump()
    assert "coverage" not in dm_dump
    assert "n_pairs" not in dm_dump


def test_escalation_judge_dump_carries_the_cascade_fields_and_round_trips():
    metrics = _binary_metrics()
    dump = metrics.model_dump(mode="json")
    assert dump["per_judge"][ESC]["coverage"] == "escalated"
    assert dump["per_judge"][ESC]["n_pairs"] == 4
    assert MetricsResult.model_validate(dump) == metrics
    assert MetricsResult.model_validate_json(metrics.model_dump_json()) == metrics
    assert JudgeMetrics.model_validate(dump["per_judge"][DM]) == metrics.per_judge[DM]


def test_dataframe_labels_escalation_judges():
    pytest.importorskip("pandas")
    df = _binary_metrics().to_dataframe()
    assert list(df.columns[-2:]) == ["judge_coverage", "n_pairs"]
    judges = df[df["level"] == "judge"].set_index("name")
    assert judges.loc[ESC, "judge_coverage"] == "escalated"
    assert judges.loc[ESC, "n_pairs"] == 4
    assert judges.loc[DM, "judge_coverage"] == "full"
    assert (
        judges.loc[DM, "n_pairs"] is None or judges.loc[DM, "n_pairs"] != judges.loc[DM, "n_pairs"]
    )
    others = df[df["level"] != "judge"]
    assert others["judge_coverage"].isna().all()
    assert others["n_pairs"].isna().all()


def test_dataframe_of_a_full_panel_has_no_cascade_columns():
    pytest.importorskip("pandas")
    reports = _binary_cascade_reports()
    # Same items judged by the decision model alone: an ordinary (one-judge) panel.
    full = [
        report.model_copy(
            update={
                "judge_scores": {DM: report.judge_scores[DM]},
                "report": [
                    cr.model_copy(
                        update={
                            "votes": [cr.votes[0].model_copy(update={"superseded": False})],
                            "escalated": False,
                        }
                    )
                    for cr in report.report
                ],
            }
        )
        for report in reports
    ]
    df = compute_metrics(
        _eval_result(full), _dataset(BIN_CRITERIA, BIN_GT), per_judge=True
    ).to_dataframe()
    assert "judge_coverage" not in df.columns
    assert "n_pairs" not in df.columns


# =============================================================================
# Per-item heterogeneous rubrics (pooled path)
# =============================================================================


def test_pooled_path_reads_only_final_verdicts():
    """The pooled path compares final verdicts, so superseded votes and missing cells
    cannot change it: a cascade and a one-judge run with the same final verdicts agree."""
    per_item_criteria = [BIN_CRITERIA, [BIN_CRITERIA[0]], BIN_CRITERIA, [BIN_CRITERIA[1]]]
    cascade = _binary_cascade_reports()
    dataset = RubricDataset(prompt="p", rubric=None, name="per-item")
    cascade_reports = []
    plain_reports = []
    for i, criteria in enumerate(per_item_criteria):
        keep = [c for c, criterion in enumerate(BIN_CRITERIA) if criterion in criteria]
        dataset.add_item(
            submission=f"s{i}",
            description=f"d{i}",
            ground_truth=[BIN_GT[i][c] for c in keep],
            rubric=Rubric(criteria),
        )
        crs = [cascade[i].report[c] for c in keep]
        cascade_reports.append(cascade[i].model_copy(update={"report": crs}))
        plain_reports.append(
            cascade[i].model_copy(
                update={
                    "judge_scores": {"solo": cascade[i].score},
                    "report": [
                        EnsembleCriterionReport(
                            criterion=cr.criterion,
                            final_verdict=cr.final_verdict,
                            final_reason="agg",
                            votes=[JudgeVote(judge_id="solo", verdict=cr.final_verdict, reason="")],
                        )
                        for cr in crs
                    ],
                }
            )
        )
    pooled_cascade = compute_metrics(_eval_result(cascade_reports), dataset, per_judge=True)
    pooled_plain = compute_metrics(_eval_result(plain_reports), dataset, per_judge=True)
    assert pooled_cascade.pooled_by_scale
    assert pooled_cascade == pooled_plain
