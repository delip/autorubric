"""Held-out validation diagnostics: per-criterion kappa, coverage, CA-rate, and a
2x2 confusion matrix; abstention-mode threading; serialization round-trip; and the
HTML report's handling-mode label plus Raw % agreement / Kappa / Coverage / CA-rate /
Precision columns and the per-criterion 2x2 — with NO statistical conflation note.

The judge is mocked by patching ``rubric.grade`` (and, for the runner path, the
``CriterionGrader`` constructor) with ``AsyncMock``s returning hand-built
``EnsembleEvaluationReport`` objects — the same approach the held-out tests in
``test_improve.py`` use, so no LLM is ever called.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric, TokenUsage
from autorubric.dataset import DataItem, RubricDataset
from autorubric.llm import LLMConfig
from autorubric.meta._display import render_improvement_report_html
from autorubric.meta._improve import (
    CriterionErrorReport,
    HeldOutValidationResult,
    ImprovementConfig,
    ImprovementRunner,
    IterationResult,
    validate_held_out,
)
from autorubric.metrics import ConfusionMatrix
from autorubric.types import (
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_criterion_rubric() -> Rubric:
    return Rubric(
        [
            Criterion(name="intro", weight=1.0, requirement="Has introduction"),
            Criterion(name="conclusion", weight=1.0, requirement="Has conclusion"),
        ]
    )


def _dataset(ground_truths: list[list[CriterionVerdict]]) -> RubricDataset:
    rubric = _two_criterion_rubric()
    items = [
        DataItem(submission=f"submission {i}", ground_truth=gt, description=f"item {i}")
        for i, gt in enumerate(ground_truths)
    ]
    return RubricDataset(rubric=rubric, items=items, prompt="Write an essay")


def _report(verdicts: list[CriterionVerdict]) -> EnsembleEvaluationReport:
    """An ensemble report whose per-criterion final_verdict is the given verdict."""
    names = ["intro", "conclusion"]
    crs = [
        EnsembleCriterionReport(
            criterion=Criterion(name=names[i], weight=1.0, requirement=f"req {i}"),
            final_verdict=v,
            final_reason="reason",
            votes=[JudgeVote(judge_id="judge_0", verdict=v, reason="reason")],
            agreement=1.0,
        )
        for i, v in enumerate(verdicts)
    ]
    return EnsembleEvaluationReport(
        score=0.5,
        raw_score=0.5,
        report=crs,
        mean_agreement=1.0,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        completion_cost=0.01,
    )


def _uniform_reports(
    per_item: list[CriterionVerdict],
    n_items: int,
) -> list[EnsembleEvaluationReport]:
    """One report per item, each emitting the same per-criterion verdict vector."""
    return [_report(per_item) for _ in range(n_items)]


def _mock_grader() -> MagicMock:
    """A stand-in grader; ``rubric.grade`` is patched, so it is never invoked."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Kappa + confusion matrix
# ---------------------------------------------------------------------------


class TestHeldOutKappaAndConfusionMatrix:
    @pytest.mark.asyncio
    async def test_constant_arrays_yield_none_kappa(self) -> None:
        """Both judge and GT all-MET for a criterion -> Cohen's kappa is undefined
        and reported as None, never a fake 0.0."""
        rubric = _two_criterion_rubric()
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.MET, CriterionVerdict.MET],
            ]
        )
        reports = _uniform_reports([CriterionVerdict.MET, CriterionVerdict.MET], 2)
        with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
            result = await validate_held_out(
                rubric, ds, _mock_grader(), task_prompt="Write an essay"
            )
        for cr in result.per_criterion:
            assert cr.kappa is None

    @pytest.mark.asyncio
    async def test_confusion_matrix_cells_present(self) -> None:
        """The 2x2 MET/UNMET ConfusionMatrix carries the live tp/fp/tn/fn cells."""
        rubric = _two_criterion_rubric()
        # intro GT [MET, UNMET]; judge MET both -> item0 tp, item1 fp.
        # conclusion GT [MET, UNMET]; judge UNMET both -> item0 fn, item1 tn.
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
            ]
        )
        reports = _uniform_reports([CriterionVerdict.MET, CriterionVerdict.UNMET], 2)
        with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
            result = await validate_held_out(
                rubric, ds, _mock_grader(), task_prompt="Write an essay"
            )

        intro = result.per_criterion[0]
        assert isinstance(intro.confusion_matrix, ConfusionMatrix)
        cm = intro.confusion_matrix
        assert cm.labels == ["MET", "UNMET"]
        assert (cm.tp, cm.fp, cm.tn, cm.fn) == (1, 1, 0, 0)
        assert cm.total == intro.n_samples == 2

        conclusion = result.per_criterion[1]
        cm2 = conclusion.confusion_matrix
        assert cm2 is not None
        assert (cm2.tp, cm2.fp, cm2.tn, cm2.fn) == (0, 0, 1, 1)
        # two distinct classes on each side -> kappa is defined
        assert conclusion.kappa is not None

    @pytest.mark.asyncio
    async def test_confusion_matrix_none_when_no_samples(self) -> None:
        """A criterion the judge always abstains on (exclude mode) has no usable
        samples -> confusion_matrix None, kappa None, n_samples 0."""
        rubric = _two_criterion_rubric()
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
            ]
        )
        reports = _uniform_reports([CriterionVerdict.CANNOT_ASSESS, CriterionVerdict.MET], 2)
        with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
            result = await validate_held_out(
                rubric, ds, _mock_grader(), task_prompt="Write an essay"
            )
        intro = result.per_criterion[0]
        assert intro.n_samples == 0
        assert intro.confusion_matrix is None
        assert intro.kappa is None


# ---------------------------------------------------------------------------
# Coverage + CA-rate (raw denominator) and roll-ups
# ---------------------------------------------------------------------------


class TestHeldOutCoverageAndCaRate:
    @pytest.mark.asyncio
    async def test_coverage_and_ca_rate_raw_denominator(self) -> None:
        """coverage = covered / n_gt_paired and ca_rate = abstentions / n_gt_paired
        over the RAW (pre-exclusion) per-criterion denominator. The judge abstains
        on intro for exactly one of four items."""
        rubric = _two_criterion_rubric()
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.MET],
            ]
        )
        # item 0 -> intro CANNOT_ASSESS; all other cells MET.
        reports = [
            _report([CriterionVerdict.CANNOT_ASSESS, CriterionVerdict.MET]),
            _report([CriterionVerdict.MET, CriterionVerdict.MET]),
            _report([CriterionVerdict.MET, CriterionVerdict.MET]),
            _report([CriterionVerdict.MET, CriterionVerdict.MET]),
        ]
        with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
            result = await validate_held_out(
                rubric, ds, _mock_grader(), task_prompt="Write an essay"
            )

        intro = result.per_criterion[0]
        assert intro.n_samples == 3
        assert intro.coverage == pytest.approx(3 / 4)
        assert intro.ca_rate == pytest.approx(1 / 4)

        conclusion = result.per_criterion[1]
        assert conclusion.coverage == pytest.approx(1.0)
        assert conclusion.ca_rate == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_mean_coverage_and_mean_ca_rate_rollup(self) -> None:
        """Result-level mean_coverage / mean_ca_rate roll up the per-criterion
        values via a None-skipping mean."""
        rubric = _two_criterion_rubric()
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
            ]
        )
        reports = _uniform_reports([CriterionVerdict.MET, CriterionVerdict.MET], 2)
        with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
            result = await validate_held_out(
                rubric, ds, _mock_grader(), task_prompt="Write an essay"
            )
        # No abstentions -> every coverage 1.0, every ca_rate 0.0.
        assert result.mean_coverage == pytest.approx(1.0)
        assert result.mean_ca_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Abstention-mode threading
# ---------------------------------------------------------------------------


class TestHeldOutCannotAssessModeThreading:
    @pytest.mark.asyncio
    async def test_default_mode_is_exclude(self) -> None:
        """The default estimand excludes abstentions (prior silent-exclude behavior)
        and the chosen mode is recorded on the result."""
        rubric = _two_criterion_rubric()
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
            ]
        )
        reports = _uniform_reports([CriterionVerdict.CANNOT_ASSESS, CriterionVerdict.MET], 2)
        with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
            result = await validate_held_out(
                rubric, ds, _mock_grader(), task_prompt="Write an essay"
            )
        assert result.cannot_assess == "exclude"
        assert result.per_criterion[0].n_samples == 0

    @pytest.mark.asyncio
    async def test_as_unmet_mode_keeps_pairs(self) -> None:
        """Under as_unmet, abstentions fold into UNMET rather than being dropped, so
        the criterion retains its paired samples."""
        rubric = _two_criterion_rubric()
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
            ]
        )
        reports = _uniform_reports([CriterionVerdict.CANNOT_ASSESS, CriterionVerdict.MET], 2)
        with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
            result = await validate_held_out(
                rubric,
                ds,
                _mock_grader(),
                task_prompt="Write an essay",
                cannot_assess="as_unmet",
            )
        assert result.cannot_assess == "as_unmet"
        assert result.per_criterion[0].n_samples == 2

    @pytest.mark.asyncio
    async def test_config_threads_cannot_assess_into_held_out(self) -> None:
        """ImprovementConfig.cannot_assess flows through the runner into
        validate_held_out, surfacing on the recorded diagnostics."""
        rubric = _two_criterion_rubric()
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
            ]
        )
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),
            strategy="held_out",
            validation_data=ds,
            max_iterations=1,
            # Converge immediately at iteration 0 (no revision LLM call needed).
            held_out_min_accuracy=0.0,
            save_artifacts=False,
            show_progress=False,
            display=None,
            cannot_assess="as_unmet",
        )
        reports = _uniform_reports([CriterionVerdict.CANNOT_ASSESS, CriterionVerdict.MET], 2)
        with (
            patch("autorubric.meta._improve.CriterionGrader", return_value=_mock_grader()),
            patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports),
        ):
            runner = ImprovementRunner(rubric, "Write an essay", config=config)
            res = await runner.run()
        diag = res.iterations[0].held_out_diagnostics
        assert diag is not None
        assert diag.cannot_assess == "as_unmet"
        assert diag.per_criterion[0].n_samples == 2


# ---------------------------------------------------------------------------
# Serialization round-trip (iter-NN.json)
# ---------------------------------------------------------------------------


class TestHeldOutSerializationRoundTrip:
    @pytest.mark.asyncio
    async def test_iter_json_round_trips_confusion_matrix(self, tmp_path: Path) -> None:
        """iter-NN.json serializes the new per-criterion diagnostics, including a
        JSON-safe confusion_matrix and the handling mode, and round-trips back."""
        rubric = _two_criterion_rubric()
        ds = _dataset(
            [
                [CriterionVerdict.MET, CriterionVerdict.MET],
                [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
            ]
        )
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),
            strategy="held_out",
            validation_data=ds,
            max_iterations=1,
            # Converge immediately at iteration 0 (no revision LLM call needed).
            held_out_min_accuracy=0.0,
            save_artifacts=True,
            artifacts_dir=str(tmp_path / "artifacts"),
            show_progress=False,
            display=None,
        )
        reports = _uniform_reports([CriterionVerdict.MET, CriterionVerdict.UNMET], 2)
        with (
            patch("autorubric.meta._improve.CriterionGrader", return_value=_mock_grader()),
            patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports),
        ):
            runner = ImprovementRunner(rubric, "Write an essay", config=config)
            await runner.run()

        iter_file = tmp_path / "artifacts" / "iter-00.json"
        assert iter_file.exists()
        data = json.loads(iter_file.read_text())
        ho = data["held_out_diagnostics"]
        assert ho["cannot_assess"] == "exclude"
        assert "mean_coverage" in ho
        assert "mean_ca_rate" in ho
        first = ho["per_criterion"][0]
        assert "kappa" in first
        assert "coverage" in first
        assert "ca_rate" in first
        cm = first["confusion_matrix"]
        assert cm is not None
        assert cm["labels"] == ["MET", "UNMET"]
        restored = ConfusionMatrix.model_validate(cm)
        assert restored.total == first["n_samples"]


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


class TestHeldOutHtmlReport:
    def _iteration(self) -> IterationResult:
        cm = ConfusionMatrix(matrix=[[1, 0], [1, 0]], labels=["MET", "UNMET"])
        cr = CriterionErrorReport(
            criterion_index=0,
            criterion_name="intro",
            n_samples=2,
            accuracy=0.5,
            false_positive_rate=1.0,
            false_negative_rate=0.0,
            disagreement_exemplars=[],
            agreement_exemplars=[],
            kappa=None,
            coverage=1.0,
            ca_rate=0.0,
            confusion_matrix=cm,
        )
        ho = HeldOutValidationResult(
            mean_accuracy=0.5,
            per_criterion=[cr],
            total_cost=None,
            item_reports=[],
            cannot_assess="exclude",
            mean_coverage=1.0,
            mean_ca_rate=0.0,
        )
        return IterationResult(
            iteration=0,
            rubric=_two_criterion_rubric(),
            quality_score=0.5,
            agreement=None,
            per_criterion_agreement=None,
            issues=[],
            issues_fixed=[],
            issues_introduced=[],
            accepted=True,
            rejection_reason=None,
            quality_report=None,
            token_usage=None,
            completion_cost=None,
            held_out_diagnostics=ho,
        )

    def test_html_has_new_columns_and_mode_label(self) -> None:
        html = render_improvement_report_html(
            [self._iteration()],
            convergence_reason="max_iterations",
            total_cost=0.0,
            original_rubric=_two_criterion_rubric(),
            final_rubric=_two_criterion_rubric(),
        )
        assert "Raw % Agreement" in html
        assert "Kappa" in html
        assert "Coverage" in html
        assert "CA-rate" in html
        assert "Precision" in html
        # Neutral handling-mode label
        assert "CANNOT_ASSESS" in html
        assert "exclude" in html
        # Per-criterion 2x2 cells surfaced
        assert "TP" in html and "FP" in html and "TN" in html and "FN" in html

    def test_html_has_no_conflation_note(self) -> None:
        """The held-out HTML carries a neutral handling-mode label but NEVER the
        statistical conflation/cluster note (single-source discipline: those live
        only in MetricsResult.summary())."""
        html = render_improvement_report_html(
            [self._iteration()],
            convergence_reason="max_iterations",
            total_cost=0.0,
            original_rubric=_two_criterion_rubric(),
            final_rubric=_two_criterion_rubric(),
        )
        lowered = html.lower()
        assert "conflation" not in lowered
        assert "krippendorff" not in lowered
        assert "fleiss" not in lowered
        assert "pearson" not in lowered
        assert "positive-rate drift" not in lowered
