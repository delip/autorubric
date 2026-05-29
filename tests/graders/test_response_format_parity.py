"""Parity tests for binary vs. multi-choice custom response formats (T7-B).

Binary criteria support a custom ``binary_response_format`` and inject an
``[Affects: #i, #j]`` tag when the parsed judgment carries non-empty
``affected_criteria``. These tests pin the symmetric multi-choice capability
(``multi_choice_response_format``, default ``MultiChoiceJudgment``) plus the
shared injection helper, and verify meta-eval wiring of ``MultiChoiceMetaJudgment``.

Grading returns an ensemble report, so the per-criterion reason is read via
``result.report[0].final_reason``. A single ordinal rubric with
``shuffle_options=False`` keeps option mapping deterministic.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import Field

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    Rubric,
    TokenUsage,
)
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import MultiChoiceJudgment


class _MCAffected(MultiChoiceJudgment):
    """Custom multi-choice format carrying affected_criteria (test fixture)."""

    affected_criteria: list[int] = Field(default_factory=list)


def _ordinal_rubric() -> Rubric:
    """A single ordinal multi-choice criterion (deterministic option values)."""
    return Rubric(
        [
            Criterion(
                name="quality",
                weight=1.0,
                requirement="Rate the quality",
                scale_type="ordinal",
                options=[
                    CriterionOption(label="poor", value=0.0),
                    CriterionOption(label="good", value=1.0),
                ],
            )
        ]
    )


def _gen_result(parsed: object) -> GenerateResult:
    return GenerateResult(
        content="",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.01,
        parsed=parsed,
    )


@pytest.mark.asyncio
async def test_multi_choice_custom_response_format_is_passed_to_client():
    """A custom multi_choice_response_format is forwarded and its [Affects:] injected."""
    parsed = _MCAffected(selected_option=1, explanation="x", affected_criteria=[1, 3])
    generate_mock = AsyncMock(return_value=_gen_result(parsed))

    grader = CriterionGrader(
        llm_config=LLMConfig(model="test-model"),
        multi_choice_response_format=_MCAffected,
        shuffle_options=False,
    )

    with patch.object(list(grader._clients.values())[0], "generate", generate_mock):
        result = await _ordinal_rubric().grade(to_grade="test submission", grader=grader)

    assert generate_mock.call_args.kwargs["response_format"] is _MCAffected
    assert result.report is not None
    assert "[Affects: #1, #3]" in result.report[0].final_reason


@pytest.mark.asyncio
async def test_multi_choice_default_no_affects_tag():
    """Default multi-choice uses MultiChoiceJudgment and injects no [Affects:] tag."""
    parsed = MultiChoiceJudgment(selected_option=1, explanation="plain")
    generate_mock = AsyncMock(return_value=_gen_result(parsed))

    grader = CriterionGrader(
        llm_config=LLMConfig(model="test-model"),
        shuffle_options=False,
    )

    with patch.object(list(grader._clients.values())[0], "generate", generate_mock):
        result = await _ordinal_rubric().grade(to_grade="test submission", grader=grader)

    assert generate_mock.call_args.kwargs["response_format"] is MultiChoiceJudgment
    assert result.report is not None
    assert "[Affects:" not in result.report[0].final_reason
    assert "plain" in result.report[0].final_reason


@pytest.mark.asyncio
async def test_binary_affects_still_injected_after_refactor():
    """Binary [Affects:] injection survives the shared-helper refactor."""
    from autorubric.meta._evaluate import MetaCriterionJudgment

    parsed = MetaCriterionJudgment(
        criterion_status=CriterionVerdict.UNMET,
        explanation="unclear",
        affected_criteria=[1, 3],
    )
    generate_mock = AsyncMock(return_value=_gen_result(parsed))

    grader = CriterionGrader(
        llm_config=LLMConfig(model="test-model"),
        binary_response_format=MetaCriterionJudgment,
    )

    with patch.object(list(grader._clients.values())[0], "generate", generate_mock):
        rubric = Rubric([Criterion(name="clarity", weight=1.0, requirement="Must be clear")])
        result = await rubric.grade(to_grade="test submission", grader=grader)

    assert result.report is not None
    assert "[Affects: #1, #3]" in result.report[0].final_reason


def test_multi_choice_meta_judgment_type():
    """MultiChoiceMetaJudgment subclasses MultiChoiceJudgment with empty default affected."""
    from autorubric.meta._evaluate import MultiChoiceMetaJudgment

    assert issubclass(MultiChoiceMetaJudgment, MultiChoiceJudgment)
    instance = MultiChoiceMetaJudgment(selected_option=2, explanation="why")
    assert instance.affected_criteria == []
    assert instance.selected_option == 2
    assert instance.explanation == "why"


@pytest.mark.asyncio
async def test_meta_eval_standalone_wires_multi_choice_format():
    """evaluate_rubric_standalone constructs the grader with both response formats."""
    from autorubric.meta._evaluate import (
        MetaCriterionJudgment,
        MultiChoiceMetaJudgment,
        evaluate_rubric_standalone,
    )

    captured = {}

    def _fake_grader(**kwargs):
        captured.update(kwargs)
        grader = AsyncMock()
        return grader

    async def _fake_grade(to_grade, grader):
        return object()

    rubric = Rubric([Criterion(name="clarity", weight=1.0, requirement="Must be clear")])

    with (
        patch("autorubric.meta._evaluate.CriterionGrader", side_effect=_fake_grader),
        patch.object(Rubric, "grade", new=AsyncMock(side_effect=_fake_grade)),
    ):
        await evaluate_rubric_standalone(rubric, LLMConfig(model="test-model"), display=None)

    assert captured["binary_response_format"] is MetaCriterionJudgment
    assert captured["multi_choice_response_format"] is MultiChoiceMetaJudgment


@pytest.mark.asyncio
async def test_meta_eval_in_context_wires_multi_choice_format():
    """evaluate_rubric_in_context constructs the grader with both response formats."""
    from autorubric.meta._evaluate import (
        MetaCriterionJudgment,
        MultiChoiceMetaJudgment,
        evaluate_rubric_in_context,
    )

    captured = {}

    def _fake_grader(**kwargs):
        captured.update(kwargs)
        grader = AsyncMock()
        return grader

    async def _fake_grade(to_grade, grader):
        return object()

    rubric = Rubric([Criterion(name="clarity", weight=1.0, requirement="Must be clear")])

    with (
        patch("autorubric.meta._evaluate.CriterionGrader", side_effect=_fake_grader),
        patch.object(Rubric, "grade", new=AsyncMock(side_effect=_fake_grade)),
    ):
        await evaluate_rubric_in_context(
            rubric, "task prompt", LLMConfig(model="test-model"), display=None
        )

    assert captured["binary_response_format"] is MetaCriterionJudgment
    assert captured["multi_choice_response_format"] is MultiChoiceMetaJudgment
