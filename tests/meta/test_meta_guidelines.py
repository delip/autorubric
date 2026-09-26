"""Rubric guidelines in meta-rubric evaluation and the improvement loop.

- The meta-judge sees the guidelines as part of the rubric under review.
- The revision LLM sees them as fixed context; it revises criteria only.
- Every revised rubric carries the input rubric's guidelines unchanged, so later iterations
  grade with them, and the loop's rubric artifacts record them in the dict form.

A rubric without guidelines produces exactly the prompts and artifacts it did before
guidelines existed (``test_meta_prompt_goldens.py`` pins the prompts against the
unmodified library).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import autorubric.meta._evaluate as evaluate_module
from autorubric import Criterion, CriterionVerdict, LLMConfig, Rubric
from autorubric.llm import GenerateResult
from autorubric.meta import (
    CriterionErrorReport,
    HeldOutValidationResult,
    ImprovementConfig,
    ImprovementRunner,
    IssueDetail,
    IterationResult,
    evaluate_rubric_in_context,
    evaluate_rubric_standalone,
    revise_rubric,
    revise_rubric_held_out,
)
from autorubric.meta._display import render_improvement_report_html
from autorubric.prompts import RUBRIC_REVISION_GUIDELINES_BLOCK
from autorubric.types import (
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
    TokenUsage,
)

GUIDELINES = (
    "Writers are English-language learners in grades 8-12; judge against that level.\n"
    "'Cited' means any attribution, e.g. {Author, year} <or> a URL & title."
)
TASK = "Write a persuasive essay on school uniforms."


def _criteria() -> list[Criterion]:
    return [
        Criterion(name="thesis", weight=3.0, requirement="States a clear, arguable thesis"),
        Criterion(name="evidence", weight=2.0, requirement="Supports claims with cited evidence"),
    ]


def _rubric(guidelines: str | None) -> Rubric:
    return Rubric(_criteria(), guidelines=guidelines)


def _revision_json(suffix: str) -> str:
    return json.dumps(
        [
            {"name": "thesis", "weight": 3, "requirement": f"States a clear thesis {suffix}"},
            {"name": "evidence", "weight": 2, "requirement": f"Cites evidence {suffix}"},
        ]
    )


def _config(**kwargs: Any) -> ImprovementConfig:
    defaults: dict[str, Any] = {
        "eval_llm": LLMConfig(model="test-eval"),
        "revision_llm": LLMConfig(model="test-revision"),
        "save_artifacts": False,
        "show_progress": False,
    }
    defaults.update(kwargs)
    return ImprovementConfig(**defaults)


def _revision_client(*contents: str) -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(
        side_effect=[GenerateResult(content=c, cost=0.001) for c in contents]
    )
    return client


ISSUES = [
    IssueDetail(
        criterion_name="clear_requirements",
        requirement="Each criterion has a clear requirement",
        weight=10.0,
        is_antipattern=False,
        feedback="Criterion 2 conflicts with the guidelines' definition of 'cited'.",
    )
]


# ---------------------------------------------------------------------------
# Meta-rubric evaluation
# ---------------------------------------------------------------------------


async def _meta_submission(run: str, rubric: Rubric) -> str:
    """The submission the meta-judge grades, captured from the meta rubric's grade call."""
    report = EnsembleEvaluationReport(score=1.0, raw_score=1.0, report=[])
    meta_rubric = MagicMock()
    meta_rubric.grade = AsyncMock(return_value=report)
    getter = "get_standalone_meta_rubric" if run == "standalone" else "get_in_context_meta_rubric"
    with patch.object(evaluate_module, getter, return_value=meta_rubric):
        if run == "standalone":
            await evaluate_rubric_standalone(rubric, LLMConfig(model="meta"))
        else:
            await evaluate_rubric_in_context(rubric, TASK, LLMConfig(model="meta"))
    return meta_rubric.grade.await_args.kwargs["to_grade"]


def _split_submission(submission: str) -> tuple[str, Any]:
    """Split the meta-judge submission into its instruction text and its JSON payload."""
    start = submission.index("{")
    return submission[:start], json.loads(submission[start:])


class TestMetaJudgeSeesGuidelines:
    @pytest.mark.asyncio
    async def test_standalone_rubric_data_leads_with_guidelines(self) -> None:
        instruction, payload = _split_submission(
            await _meta_submission("standalone", _rubric(GUIDELINES))
        )
        assert list(payload) == ["guidelines", "criteria"]
        assert payload["guidelines"] == GUIDELINES
        assert [c["index"] for c in payload["criteria"]] == [1, 2]
        assert instruction == (
            evaluate_module._CRITERIA_REF_INSTRUCTION + evaluate_module._GUIDELINES_REF_INSTRUCTION
        )
        assert "The criterion text governs" in instruction

    @pytest.mark.asyncio
    async def test_in_context_rubric_data_leads_with_guidelines(self) -> None:
        instruction, payload = _split_submission(
            await _meta_submission("in_context", _rubric(GUIDELINES))
        )
        assert list(payload) == ["task_prompt", "rubric"]
        assert payload["task_prompt"] == TASK
        assert list(payload["rubric"]) == ["guidelines", "criteria"]
        assert payload["rubric"]["guidelines"] == GUIDELINES
        assert instruction.endswith(evaluate_module._GUIDELINES_REF_INSTRUCTION)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("run", ["standalone", "in_context"])
    async def test_without_guidelines_nothing_changes(self, run: str) -> None:
        submission = await _meta_submission(run, _rubric(None))
        instruction, payload = _split_submission(submission)
        assert instruction == evaluate_module._CRITERIA_REF_INSTRUCTION
        rubric_data = payload if run == "standalone" else payload["rubric"]
        assert list(rubric_data) == ["criteria"]
        assert "guidelines" not in submission

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("guidelines", "expected"),
        [
            (GUIDELINES, "Evaluating rubric with 2 criteria and guidelines"),
            (None, "Evaluating rubric with 2 criteria"),
        ],
    )
    async def test_display_summary_names_guidelines(
        self, guidelines: str | None, expected: str
    ) -> None:
        report = EnsembleEvaluationReport(score=1.0, raw_score=1.0, report=[])
        meta_rubric = MagicMock()
        meta_rubric.grade = AsyncMock(return_value=report)
        display = MagicMock()
        with (
            patch.object(evaluate_module, "get_standalone_meta_rubric", return_value=meta_rubric),
            patch.object(evaluate_module, "get_in_context_meta_rubric", return_value=meta_rubric),
            patch.object(evaluate_module, "display_meta_rubric_result", display),
        ):
            await evaluate_rubric_standalone(
                _rubric(guidelines), LLMConfig(model="meta"), display="stdout"
            )
            await evaluate_rubric_in_context(
                _rubric(guidelines), TASK, LLMConfig(model="meta"), display="stdout"
            )
        summaries = [c.kwargs["rubric_summary"] for c in display.call_args_list]
        assert summaries == [expected, f"{expected}\nTask: {TASK}"]


# ---------------------------------------------------------------------------
# Revision
# ---------------------------------------------------------------------------


async def _revise(run: str, rubric: Rubric, **kwargs: Any) -> tuple[Rubric, str, str]:
    """Run one revision; return the revised rubric and the (system, user) prompts sent."""
    capture: dict[str, str] = {}
    client = _revision_client(kwargs.pop("response", _revision_json("v2")))
    with patch("autorubric.meta._improve.LLMClient", return_value=client):
        if run == "revise_rubric":
            revised, _ = await revise_rubric(
                rubric,
                TASK,
                ISSUES,
                "## Validation\nrho=0.4",
                "none\n",
                _config(),
                _capture=capture,
                **kwargs,
            )
        else:
            revised, _ = await revise_rubric_held_out(
                rubric,
                TASK,
                "Criterion 1: FN rate 40%",
                "none\n",
                _config(),
                _capture=capture,
                **kwargs,
            )
    system_prompt, user_prompt = client.generate.await_args.args[:2]
    assert capture["user_prompt"] == user_prompt
    return revised, system_prompt, user_prompt


class TestRevisionCarriesGuidelines:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("run", ["revise_rubric", "revise_rubric_held_out"])
    async def test_revised_rubric_keeps_guidelines(self, run: str) -> None:
        revised, _, _ = await _revise(run, _rubric(GUIDELINES))
        assert revised.guidelines == GUIDELINES
        assert [c.requirement for c in revised.rubric] == [
            "States a clear thesis v2",
            "Cites evidence v2",
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("run", ["revise_rubric", "revise_rubric_held_out"])
    async def test_revised_rubric_without_guidelines_has_none(self, run: str) -> None:
        revised, _, _ = await _revise(run, _rubric(None))
        assert revised.guidelines is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("run", ["revise_rubric", "revise_rubric_held_out"])
    async def test_revision_llm_sees_guidelines_as_a_leading_block(self, run: str) -> None:
        """The prompt is the one without guidelines, preceded by the guidelines block."""
        _, system_with, user_with = await _revise(run, _rubric(GUIDELINES))
        _, system_without, user_without = await _revise(run, _rubric(None))
        block = RUBRIC_REVISION_GUIDELINES_BLOCK.format(guidelines=GUIDELINES)
        assert user_with == block + user_without
        assert system_with == system_without
        assert block.startswith(f"<guidelines>\n{GUIDELINES}\n\n")
        assert block.endswith("</guidelines>\n\n")
        assert "<guidelines>" not in user_without

    @pytest.mark.asyncio
    async def test_custom_user_template_also_gets_the_block(self) -> None:
        template = "Revise:\n{original_criteria}\n{issues_text}{validation_text}{history_text}"
        _, _, user_with = await _revise(
            "revise_rubric", _rubric(GUIDELINES), user_prompt_template=template
        )
        _, _, user_without = await _revise(
            "revise_rubric", _rubric(None), user_prompt_template=template
        )
        assert user_with == (
            RUBRIC_REVISION_GUIDELINES_BLOCK.format(guidelines=GUIDELINES) + user_without
        )

    @pytest.mark.asyncio
    async def test_held_out_structural_violation_returns_the_input_rubric(self) -> None:
        rubric = _rubric(GUIDELINES)
        one_criterion = json.dumps([{"weight": 1, "requirement": "Only one"}])
        revised, _, _ = await _revise("revise_rubric_held_out", rubric, response=one_criterion)
        assert revised is rubric
        assert revised.guidelines == GUIDELINES


# ---------------------------------------------------------------------------
# Improvement loop
# ---------------------------------------------------------------------------


def _quality_report(verdict: CriterionVerdict, score: float) -> EnsembleEvaluationReport:
    criterion = Criterion(name="clear_requirements", weight=10.0, requirement="Clear")
    return EnsembleEvaluationReport(
        score=score,
        raw_score=score,
        report=[
            EnsembleCriterionReport(
                criterion=criterion,
                final_verdict=verdict,
                final_reason="Criterion 2 is vague.",
                votes=[JudgeVote(judge_id="j", verdict=verdict, reason="Criterion 2 is vague.")],
                agreement=1.0,
            )
        ],
        mean_agreement=1.0,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        completion_cost=0.001,
    )


async def _run_meta_loop(rubric: Rubric, artifacts_dir: Path) -> tuple[Any, AsyncMock]:
    """Three meta-rubric iterations (two revisions), every meta evaluation mocked."""
    evaluate = AsyncMock(
        side_effect=[
            _quality_report(CriterionVerdict.UNMET, 0.4),
            _quality_report(CriterionVerdict.UNMET, 0.6),
            _quality_report(CriterionVerdict.MET, 1.0),
        ]
    )
    client = _revision_client(_revision_json("v1"), _revision_json("v2"))
    config = _config(
        save_artifacts=True,
        artifacts_dir=str(artifacts_dir),
        max_iterations=3,
        score_plateau_threshold=0.0,
    )
    with (
        patch("autorubric.meta._improve.evaluate_rubric_in_context", evaluate),
        patch("autorubric.meta._improve.LLMClient", return_value=client),
    ):
        result = await ImprovementRunner(rubric, TASK, config=config).run()
    return result, evaluate


class TestImprovementLoopCarriesGuidelines:
    @pytest.mark.asyncio
    async def test_every_iteration_grades_and_revises_with_the_guidelines(
        self, tmp_path: Path
    ) -> None:
        result, evaluate = await _run_meta_loop(_rubric(GUIDELINES), tmp_path)
        assert len(result.iterations) == 3
        assert [it.rubric.guidelines for it in result.iterations] == [GUIDELINES] * 3
        assert result.iterations[2].rubric.rubric[0].requirement == "States a clear thesis v2"
        # Each meta evaluation received the rubric of its iteration, guidelines included.
        evaluated = [call.args[0] for call in evaluate.await_args_list]
        assert [r.guidelines for r in evaluated] == [GUIDELINES] * 3
        for rubric in (result.original_rubric, result.final_rubric, result.best_rubric):
            assert rubric.guidelines == GUIDELINES

    @pytest.mark.asyncio
    async def test_rubric_artifacts_use_the_dict_form(self, tmp_path: Path) -> None:
        result, _ = await _run_meta_loop(_rubric(GUIDELINES), tmp_path)
        for i, it in enumerate(result.iterations):
            data = json.loads((tmp_path / f"rubric-iter-{i:02d}.json").read_text())
            assert list(data) == ["guidelines", "criteria"]
            assert data["guidelines"] == GUIDELINES
            assert [c["requirement"] for c in data["criteria"]] == [
                c.requirement for c in it.rubric.rubric
            ]
            # The artifact loads back as the iteration's rubric.
            loaded = Rubric.from_dict(data)
            assert loaded.guidelines == GUIDELINES
        summary = json.loads((tmp_path / "summary.json").read_text())
        for key in ("original_rubric", "final_rubric"):
            assert summary[key]["guidelines"] == GUIDELINES
            assert Rubric.from_dict(summary[key]).guidelines == GUIDELINES
        iter_0 = json.loads((tmp_path / "iter-00.json").read_text())
        assert iter_0["revision"]["user_prompt"].startswith(f"<guidelines>\n{GUIDELINES}")
        assert isinstance(iter_0["rubric_criteria"], list)
        html = (tmp_path / "improvement_report.html").read_text()
        assert "<h4>Guidelines</h4>" in html
        assert "{Author, year} &lt;or&gt; a URL &amp; title" in html

    @pytest.mark.asyncio
    async def test_without_guidelines_artifacts_keep_the_list_form(self, tmp_path: Path) -> None:
        await _run_meta_loop(_rubric(None), tmp_path)
        expected_first = (
            "[\n"
            "  {\n"
            '    "weight": 3.0,\n'
            '    "requirement": "States a clear, arguable thesis",\n'
            '    "name": "thesis"\n'
            "  },\n"
            "  {\n"
            '    "weight": 2.0,\n'
            '    "requirement": "Supports claims with cited evidence",\n'
            '    "name": "evidence"\n'
            "  }\n"
            "]"
        )
        assert (tmp_path / "rubric-iter-00.json").read_text(encoding="utf-8") == expected_first
        for i in range(3):
            data = json.loads((tmp_path / f"rubric-iter-{i:02d}.json").read_text())
            assert isinstance(data, list)
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert json.dumps(summary["original_rubric"], indent=2) == expected_first
        assert isinstance(summary["final_rubric"], list)
        assert "Guidelines" not in (tmp_path / "improvement_report.html").read_text()

    @pytest.mark.asyncio
    async def test_held_out_loop_carries_guidelines(self, tmp_path: Path) -> None:
        result, validate, client = await _run_held_out_loop(_rubric(GUIDELINES), tmp_path)

        assert [it.rubric.guidelines for it in result.iterations] == [GUIDELINES] * 2
        validated = [call.args[0] for call in validate.await_args_list]
        assert [r.guidelines for r in validated] == [GUIDELINES] * 2
        user_prompt = client.generate.await_args.args[1]
        assert user_prompt.startswith(
            RUBRIC_REVISION_GUIDELINES_BLOCK.format(guidelines=GUIDELINES)
        )
        for i in range(2):
            data = json.loads((tmp_path / f"rubric-iter-{i:02d}.json").read_text())
            assert data["guidelines"] == GUIDELINES
        # The held-out loop writes its own summary.json; both rubrics use the dict form.
        summary = json.loads((tmp_path / "summary.json").read_text())
        for key in ("original_rubric", "final_rubric"):
            assert list(summary[key]) == ["guidelines", "criteria"], key
            assert summary[key]["guidelines"] == GUIDELINES, key
            assert Rubric.from_dict(summary[key]).guidelines == GUIDELINES, key
        assert Rubric.from_dict(summary["original_rubric"]).rubric == result.original_rubric.rubric

    @pytest.mark.asyncio
    async def test_held_out_loop_without_guidelines_keeps_the_list_form(
        self, tmp_path: Path
    ) -> None:
        result, _, _ = await _run_held_out_loop(_rubric(None), tmp_path)

        assert [it.rubric.guidelines for it in result.iterations] == [None] * 2
        for i in range(2):
            data = json.loads((tmp_path / f"rubric-iter-{i:02d}.json").read_text())
            assert isinstance(data, list)
        summary = json.loads((tmp_path / "summary.json").read_text())
        for key in ("original_rubric", "final_rubric"):
            assert isinstance(summary[key], list), key
        assert summary["original_rubric"] == [
            {"weight": 3.0, "requirement": "States a clear, arguable thesis", "name": "thesis"},
            {
                "weight": 2.0,
                "requirement": "Supports claims with cited evidence",
                "name": "evidence",
            },
        ]


async def _run_held_out_loop(
    rubric: Rubric, artifacts_dir: Path
) -> tuple[Any, AsyncMock, MagicMock]:
    """Two held-out iterations (one revision), every held-out validation mocked."""
    from autorubric import DataItem, RubricDataset

    dataset = RubricDataset(
        prompt=TASK,
        rubric=_rubric(None),
        items=[
            DataItem(
                submission="essay",
                description="d",
                ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
            )
        ],
    )

    def held_out(accuracy: float) -> HeldOutValidationResult:
        return HeldOutValidationResult(
            mean_accuracy=accuracy,
            per_criterion=[
                CriterionErrorReport(
                    criterion_index=i,
                    criterion_name=name,
                    n_samples=1,
                    accuracy=accuracy,
                    false_positive_rate=0.0,
                    false_negative_rate=1.0 - accuracy,
                    disagreement_exemplars=[],
                    agreement_exemplars=[],
                )
                for i, name in enumerate(["thesis", "evidence"])
            ],
            total_cost=0.0,
            item_reports=[],
        )

    validate = AsyncMock(side_effect=[held_out(0.5), held_out(1.0)])
    client = _revision_client(_revision_json("v1"))
    config = _config(
        strategy="held_out",
        validation_data=dataset,
        held_out_min_accuracy=0.9,
        save_artifacts=True,
        artifacts_dir=str(artifacts_dir),
        max_iterations=3,
    )
    with (
        patch("autorubric.meta._improve.validate_held_out", validate),
        patch("autorubric.meta._improve.LLMClient", return_value=client),
    ):
        result = await ImprovementRunner(rubric, TASK, config=config).run()
    return result, validate, client


class TestImprovementReportHtml:
    def _render(self, final_rubric: Rubric) -> str:
        iteration = IterationResult(
            iteration=0,
            rubric=final_rubric,
            quality_score=1.0,
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
        )
        return render_improvement_report_html(
            [iteration], "no_issues", 0.0, final_rubric, final_rubric
        )

    def test_final_rubric_panel_shows_escaped_guidelines(self) -> None:
        html = self._render(_rubric(GUIDELINES))
        final_panel = html[html.index("<h2>Final Rubric</h2>") :]
        assert "<h4>Guidelines</h4>" in final_panel
        assert "{Author, year} &lt;or&gt; a URL &amp; title" in final_panel
        assert final_panel.index("<h4>Guidelines</h4>") < final_panel.index("<table>")

    def test_without_guidelines_report_is_unchanged(self) -> None:
        with_none = self._render(_rubric(None))
        assert "Guidelines" not in with_none
