"""Tests for the metrics-consistency rendering contract on ``MetricsResult``.

Covers the rendering surface that ``summary()`` and ``to_dataframe()`` expose:

- a keyword-only ``verbose`` flag on ``summary()`` that swaps the per-judge default line
  from RMSE/Spearman to accuracy/mean_kappa(+phi) and only then reveals numeric columns +
  confusion matrices;
- a "Handling modes:" line naming the CANNOT_ASSESS and NA estimands;
- micro/macro level labels on the criterion-level block, plus the new macro accuracy,
  micro kappa, phi, and mean Krippendorff alpha lines;
- the score-level header declaring the continuous per-item weighted score;
- type-aware inter-judge rendering that drops the bare Fleiss column for binary/nominal
  (Krippendorff alpha is primary) while keeping it for ordinal with a different-geometry
  note, plus the phi-cluster conflation note;
- distinct rendering of a degenerate criterion vs a no-data criterion;
- the binary FP/FN counts and FPR/FNR, plus a per-criterion coverage continuation line;
- ``to_dataframe()`` aggregate-row key renames (micro/macro), the dropped bare Fleiss
  column for binary/nominal, and the new aggregate/per-criterion/per-judge columns; and
- a round-trip of the handling modes through ``to_file``.
"""

import json
from datetime import datetime

import pytest

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import compute_metrics
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

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CA = CriterionVerdict.CANNOT_ASSESS


# =============================================================================
# Builders (non-ensemble binary)
# =============================================================================


def _wrap_eval(item_results):
    return EvalResult(
        item_results=item_results,
        total_items=len(item_results),
        successful_items=len(item_results),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=None,
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


def _dataset(criteria, ground_truths):
    rubric = Rubric(criteria)
    dataset = RubricDataset(prompt="Test prompt", rubric=rubric, name="render-test")
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


def _binary_metrics():
    """A varied binary-only result so accuracy/kappa/phi/FP-FN are all defined."""
    criteria = [Criterion(name="c0", requirement="r0", weight=1.0)]
    dataset = _dataset(criteria, [[MET], [MET], [UNMET], [UNMET]])
    eval_result = _eval(criteria, [[MET], [UNMET], [MET], [UNMET]], [1.0, 0.0, 0.5, 0.5])
    return compute_metrics(eval_result, dataset)


# =============================================================================
# Ensemble builders (so inter-judge alpha/Fleiss are populated)
# =============================================================================


def _binary_ensemble_report(criterion, judge_verdicts):
    votes = [
        JudgeVote(judge_id=jid, verdict=verdict, reason="r") for jid, verdict in judge_verdicts
    ]
    n_met = sum(1 for _, v in judge_verdicts if v == MET)
    final = MET if n_met * 2 >= len(judge_verdicts) else UNMET
    ecr = EnsembleCriterionReport(
        criterion=criterion, final_verdict=final, final_reason="agg", votes=votes
    )
    judge_scores = {jid: (1.0 if v == MET else 0.0) for jid, v in judge_verdicts}
    return EnsembleEvaluationReport(
        score=1.0 if final == MET else 0.0,
        raw_score=0.0,
        report=[ecr],
        judge_scores=judge_scores,
    )


def _ecr_mc(criterion, final, mc_votes):
    return EnsembleCriterionReport(
        criterion=criterion,
        final_verdict=None,
        final_reason="agg",
        votes=[],
        final_multi_choice_verdict=final,
        multi_choice_votes=mc_votes,
    )


def _multi_choice_ensemble_report(criterion, judge_indices):
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
            )
        )
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
        report=[_ecr_mc(criterion, final, mc_votes)],
        judge_scores=judge_scores,
    )


def _ordinal_criterion():
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


def _nominal_criterion():
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


def _binary_ensemble_metrics():
    crit = Criterion(name="acc", weight=10.0, requirement="Be accurate")
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [
        _binary_ensemble_report(crit, [("ja", MET), ("jb", MET), ("jc", MET)]),
        _binary_ensemble_report(crit, [("ja", UNMET), ("jb", UNMET), ("jc", UNMET)]),
    ]
    for i, gt in enumerate([MET, UNMET]):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[gt])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    return compute_metrics(_wrap_eval(item_results), ds, per_judge=True)


def _ordinal_ensemble_metrics():
    crit = _ordinal_criterion()
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [
        _multi_choice_ensemble_report(crit, [("ja", 2), ("jb", 2), ("jc", 2)]),
        _multi_choice_ensemble_report(crit, [("ja", 0), ("jb", 0), ("jc", 0)]),
    ]
    for i, gt in enumerate(["High", "Low"]):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[gt])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    return compute_metrics(_wrap_eval(item_results), ds)


def _nominal_ensemble_metrics():
    crit = _nominal_criterion()
    ds = RubricDataset(prompt="p", rubric=Rubric([crit]), name="x")
    reports = [
        _multi_choice_ensemble_report(crit, [("ja", 0), ("jb", 0), ("jc", 0)]),
        _multi_choice_ensemble_report(crit, [("ja", 1), ("jb", 1), ("jc", 1)]),
    ]
    for i, gt in enumerate(["Alpha", "Beta"]):
        ds.add_item(submission=f"s{i}", description="d", ground_truth=[gt])
    item_results = [
        ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
        for i, r in enumerate(reports)
    ]
    return compute_metrics(_wrap_eval(item_results), ds)


# =============================================================================
# summary(): handling modes line
# =============================================================================


class TestHandlingModesLine:
    def test_summary_shows_handling_modes(self):
        m = _binary_metrics()
        text = m.summary()
        assert "Handling modes:" in text
        assert "CANNOT_ASSESS=exclude" in text
        assert "NA=exclude" in text

    def test_summary_reflects_non_default_modes(self):
        criteria = [Criterion(name="c0", requirement="r0", weight=1.0)]
        dataset = _dataset(criteria, [[MET], [UNMET], [CA]])
        eval_result = _eval(criteria, [[MET], [UNMET], [CA]], [1.0, 0.0, 0.5])
        m = compute_metrics(eval_result, dataset, cannot_assess="as_category")
        text = m.summary()
        assert "CANNOT_ASSESS=as_category" in text


# =============================================================================
# summary(): micro/macro labels + new aggregate lines
# =============================================================================


class TestMicroMacroLabels:
    def test_criterion_level_labels(self):
        text = _binary_metrics().summary()
        assert "Accuracy (micro)" in text
        assert "Mean Kappa (macro)" in text

    def test_new_aggregate_lines_present(self):
        text = _binary_metrics().summary()
        assert "Accuracy (macro)" in text
        assert "Kappa (micro)" in text
        # Aggregate Matthews phi line.
        assert "Phi (micro)" in text

    def test_score_block_declares_continuous_scale(self):
        text = _binary_metrics().summary()
        assert "continuous per-item weighted score" in text


# =============================================================================
# summary(): type-aware inter-judge rendering (Fleiss drop) + phi cluster note
# =============================================================================


class TestTypeAwareAgreementRendering:
    def test_binary_drops_bare_fleiss_keeps_alpha(self):
        m = _binary_ensemble_metrics()
        assert m.per_criterion[0].krippendorff_alpha is not None
        text = m.summary()
        assert "Kripp-α" in text
        # Binary group does not render a bare Fleiss data COLUMN (header). The
        # single-source conflation note still names Fleiss to explain its omission, so
        # the contract is "no Fleiss column header", not "the word Fleiss is absent".
        for line in text.splitlines():
            if "Kripp-α" in line:
                assert "Fleiss" not in line

    def test_nominal_drops_bare_fleiss_keeps_alpha(self):
        m = _nominal_ensemble_metrics()
        assert m.per_criterion[0].krippendorff_alpha is not None
        text = m.summary()
        assert "Kripp-α" in text
        for line in text.splitlines():
            if "Kripp-α" in line:
                assert "Fleiss" not in line

    def test_ordinal_keeps_fleiss_with_geometry_note(self):
        m = _ordinal_ensemble_metrics()
        assert m.per_criterion[0].fleiss_kappa is not None
        text = m.summary()
        assert "Kripp-α" in text
        assert "Fleiss" in text
        # The note distinguishes ordinal (distance-aware) alpha from nominal Fleiss.
        assert "geometry" in text.lower()

    def test_binary_alpha_conflation_note_present(self):
        text = _binary_ensemble_metrics().summary()
        # Single-source note: alpha and Fleiss coincide up to a finite-sample correction.
        assert "finite-sample" in text.lower()

    def test_phi_cluster_conflation_note_present(self):
        text = _binary_ensemble_metrics().summary()
        # phi coincides with Pearson/Spearman/Kendall/MCC on binary data.
        low = text.lower()
        assert "phi" in low or "φ" in text
        assert "pearson" in low and "kendall" in low


# =============================================================================
# summary(): degenerate vs no-data rendering
# =============================================================================


class TestDegenerateRendering:
    def test_degenerate_renders_distinct_from_no_data(self):
        criteria = [Criterion(name="c0", requirement="r0", weight=1.0)]
        # Constant ground truth and prediction → samples present, kappa undefined → degenerate.
        dataset = _dataset(criteria, [[MET], [MET], [MET]])
        eval_result = _eval(criteria, [[MET], [MET], [MET]], [1.0, 1.0, 1.0])
        m = compute_metrics(eval_result, dataset)
        cm = m.per_criterion[0]
        assert cm.is_degenerate is True
        text = m.summary()
        # A degenerate criterion is marked distinctly, not just rendered as 'n/a'.
        assert "degenerate" in text.lower()


# =============================================================================
# summary(): binary FP/FN/FPR/FNR + coverage continuation line
# =============================================================================


class TestBinaryConfusionAndCoverageRendering:
    def test_summary_shows_fp_fn_and_rates(self):
        text = _binary_metrics().summary()
        low = text.lower()
        # FP/FN counts and the FPR/FNR rates surface in the binary rendering.
        assert "fp" in low and "fn" in low
        assert "fpr" in low and "fnr" in low

    def test_summary_shows_coverage_when_excluded(self):
        criteria = [Criterion(name="c0", requirement="r0", weight=1.0)]
        # One CANNOT_ASSESS prediction → coverage < 1 under exclude.
        dataset = _dataset(criteria, [[MET], [UNMET], [MET]])
        eval_result = _eval(criteria, [[MET], [UNMET], [CA]], [1.0, 0.0, 0.5])
        m = compute_metrics(eval_result, dataset)
        assert m.per_criterion[0].coverage_stats is not None
        text = m.summary()
        assert "coverage" in text.lower()


# =============================================================================
# summary(): per-judge swap + verbose
# =============================================================================


class TestPerJudgeSwapAndVerbose:
    def test_default_per_judge_line_swaps_to_accuracy_and_kappa(self):
        m = _binary_ensemble_metrics()
        assert m.per_judge is not None
        text = m.summary()
        # The default per-judge line shows accuracy + mean_kappa, not RMSE + Spearman.
        low = text.lower()
        assert "per-judge metrics" in low
        # The per-judge line no longer leads with RMSE/Spearman by default.
        # Find the per-judge section text.
        section = text.split("Per-Judge Metrics:", 1)[1]
        sec_low = section.lower()
        assert "acc" in sec_low
        assert "kappa" in sec_low
        # RMSE/Spearman are demoted out of the default per-judge line.
        # (They reappear only under verbose=True; see below.)
        assert "spearman" not in section.split("\n")[1].lower()

    def test_verbose_reveals_numeric_columns_and_confusion_matrix(self):
        m = _binary_ensemble_metrics()
        default = m.summary()
        verbose = m.summary(verbose=True)
        # verbose is a superset that adds detail.
        assert len(verbose) > len(default)
        # The per-judge confusion matrix appears only under verbose.
        assert "confusion" in verbose.lower()
        assert "confusion" not in default.lower()

    def test_summary_verbose_is_keyword_only(self):
        m = _binary_metrics()
        with pytest.raises(TypeError):
            m.summary(True)  # type: ignore[misc]


# =============================================================================
# to_dataframe(): key renames + dropped/added columns
# =============================================================================


class TestDataFrameRenames:
    def test_aggregate_row_uses_micro_macro_keys(self):
        pytest.importorskip("pandas")
        df = _binary_metrics().to_dataframe()
        agg = df[df["level"] == "aggregate"].iloc[0]
        # New micro/macro aggregate keys present.
        for key in (
            "accuracy_micro",
            "mean_kappa_macro",
            "precision_micro",
            "recall_micro",
            "f1_micro",
            "accuracy_macro",
            "kappa_micro",
            "phi_micro",
            "mean_krippendorff_alpha",
        ):
            assert key in df.columns, key
        assert agg["accuracy_micro"] == pytest.approx(_binary_metrics().criterion_accuracy)

    def test_aggregate_old_bare_keys_absent(self):
        pytest.importorskip("pandas")
        df = _binary_metrics().to_dataframe()
        # The bare aggregate accuracy/kappa/precision/recall/f1 keys were renamed away.
        for old in ("accuracy", "precision", "recall", "f1"):
            assert old not in df.columns, old
        # The bare aggregate "kappa" column was renamed too.
        assert "kappa" not in df.columns

    def test_aggregate_modes_and_n_samples_columns(self):
        pytest.importorskip("pandas")
        m = _binary_metrics()
        df = m.to_dataframe()
        agg = df[df["level"] == "aggregate"].iloc[0]
        assert agg["cannot_assess_mode"] == "exclude"
        assert agg["na_mode"] == "exclude"
        assert agg["n_samples"] == m.n_samples

    def test_aggregate_coverage_columns(self):
        pytest.importorskip("pandas")
        criteria = [Criterion(name="c0", requirement="r0", weight=1.0)]
        dataset = _dataset(criteria, [[MET], [UNMET], [MET]])
        eval_result = _eval(criteria, [[MET], [UNMET], [CA]], [1.0, 0.0, 0.5])
        m = compute_metrics(eval_result, dataset)
        df = m.to_dataframe()
        for col in (
            "coverage",
            "judge_abstain_rate",
            "gt_abstain_rate",
            "union_exclusion_rate",
            "n_errored",
            "error_rate",
        ):
            assert col in df.columns, col

    def test_per_criterion_new_columns(self):
        pytest.importorskip("pandas")
        df = _binary_metrics().to_dataframe()
        for col in ("phi", "fpr", "fnr", "is_degenerate"):
            assert col in df.columns, col
        crit = df[df["level"] == "criterion"].iloc[0]
        # The binary criterion carries a real FPR/FNR (0.5 each in this fixture).
        assert crit["fpr"] == pytest.approx(0.5)
        assert crit["fnr"] == pytest.approx(0.5)

    def test_binary_criterion_row_drops_bare_fleiss(self):
        pytest.importorskip("pandas")
        import pandas as pd

        m = _binary_ensemble_metrics()
        df = m.to_dataframe()
        crit = df[df["level"] == "criterion"].iloc[0]
        # Binary criterion keeps the alpha value but emits no bare Fleiss value.
        assert crit["krippendorff_alpha"] == pytest.approx(1.0)
        assert pd.isna(crit["fleiss_kappa"])

    def test_ordinal_criterion_row_keeps_fleiss(self):
        pytest.importorskip("pandas")
        m = _ordinal_ensemble_metrics()
        df = m.to_dataframe()
        crit = df[df["level"] == "criterion"].iloc[0]
        assert crit["fleiss_kappa"] == pytest.approx(1.0)

    def test_per_judge_row_has_phi(self):
        pytest.importorskip("pandas")
        m = _binary_ensemble_metrics()
        df = m.to_dataframe()
        judge = df[df["level"] == "judge"].iloc[0]
        assert "phi" in df.columns
        assert judge["phi"] is not None


# =============================================================================
# to_file(): handling-mode round-trip
# =============================================================================


class TestToFileModeRoundTrip:
    def test_modes_round_trip_through_to_file(self, tmp_path):
        criteria = [Criterion(name="c0", requirement="r0", weight=1.0)]
        dataset = _dataset(criteria, [[MET], [UNMET], [CA]])
        eval_result = _eval(criteria, [[MET], [UNMET], [CA]], [1.0, 0.0, 0.5])
        m = compute_metrics(eval_result, dataset, cannot_assess="as_category")
        out = tmp_path / "metrics.json"
        m.to_file(out)
        payload = json.loads(out.read_text())
        assert payload["cannot_assess_mode"] == "as_category"
        assert payload["na_mode"] == "exclude"


# =============================================================================
# 1-judge == aggregate for the rendered new fields
# =============================================================================


class TestSingleJudgeEqualsAggregateRendered:
    def test_one_judge_phi_and_macro_accuracy_equal_aggregate(self):
        criteria = [
            Criterion(name="c0", requirement="r0", weight=1.0),
            Criterion(name="c1", requirement="r1", weight=1.0),
        ]
        ds = RubricDataset(prompt="p", rubric=Rubric(criteria), name="x")
        gts = [[MET, MET], [UNMET, UNMET], [MET, UNMET]]
        for i, gt in enumerate(gts):
            ds.add_item(submission=f"s{i}", description="d", ground_truth=gt)
        verds = [[MET, MET], [UNMET, MET], [MET, UNMET]]
        reports = []
        for i, v in enumerate(verds):
            ecrs = [
                EnsembleCriterionReport(
                    criterion=criteria[c],
                    final_verdict=v[c],
                    final_reason="a",
                    votes=[JudgeVote(judge_id="only", verdict=v[c], reason="r")],
                )
                for c in range(2)
            ]
            reports.append(
                EnsembleEvaluationReport(
                    score=float(i), raw_score=float(i), report=ecrs, judge_scores={"only": float(i)}
                )
            )
        item_results = [
            ItemResult(item_idx=i, item=ds.items[i], report=r, duration_seconds=0.1)
            for i, r in enumerate(reports)
        ]
        m = compute_metrics(_wrap_eval(item_results), ds, per_judge=True)
        assert m.per_judge is not None
        only = m.per_judge["only"]
        assert only.phi == pytest.approx(m.criterion_phi)
        assert only.criterion_accuracy == pytest.approx(m.criterion_accuracy)
