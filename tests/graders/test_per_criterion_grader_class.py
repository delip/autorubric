"""Tests for CriterionGrader using the new LLMConfig-based API."""

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric, TokenUsage
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment, EvaluationReport


def _wrap_in_generate_result(
    judgment: CriterionJudgment, return_result: bool
) -> GenerateResult | CriterionJudgment:
    """Helper to wrap CriterionJudgment in GenerateResult when return_result is True."""
    if return_result:
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            cost=0.001,
            parsed=judgment,
        )
    return judgment


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create a mock LLMConfig for testing."""
    return LLMConfig(model="test-model")


@pytest.mark.asyncio
async def test_per_criterion_grader_class_integration(
    sample_rubric, sample_output, mock_llm_config, per_criterion_mock_client
):
    """Test that CriterionGrader works with mocked LLMClient."""
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=per_criterion_mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)

        report: EvaluationReport = await sample_rubric.grade(sample_output, grader=grader)

        assert report.score == pytest.approx(1.0)
        assert report.report is not None
        assert len(report.report) == len(sample_rubric.rubric)
        assert [criterion.final_verdict for criterion in report.report] == [
            CriterionVerdict.MET,
            CriterionVerdict.MET,
            CriterionVerdict.MET,
            CriterionVerdict.UNMET,
        ]


@pytest.mark.asyncio
async def test_per_criterion_grader_handles_invalid_json(sample_rubric, mock_llm_config):
    """Parse failures use conservative defaults based on criterion type."""
    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=Exception("Parse error"))

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)

        report = await grader.grade(
            to_grade="Example submission",
            rubric=sample_rubric.rubric,
        )

        # Score is 0.0 because:
        # - Positive criteria (weights 2.0, 1.0, 1.0) default to UNMET = 0 points
        # - Negative criterion (weight -0.5) defaults to MET = -0.5 points (error assumed present)
        # weighted_sum = -0.5, total_positive = 4.0, score = max(0, -0.5/4.0) = 0.0
        assert report.score == 0.0
        assert report.report is not None

        # Verify conservative defaults: positive->UNMET, negative->MET
        verdicts = [r.final_verdict for r in report.report]
        weights = [r.criterion.weight for r in report.report]
        for verdict, weight in zip(verdicts, weights):
            if weight < 0:
                assert verdict == CriterionVerdict.MET, "Negative criteria should default to MET on parse failure"
            else:
                assert verdict == CriterionVerdict.UNMET, "Positive criteria should default to UNMET on parse failure"

        for criterion_report in report.report:
            assert "Error parsing judge response" in criterion_report.final_reason


@pytest.mark.asyncio
async def test_per_criterion_grader_with_negative_criterion_unmet(sample_rubric, mock_llm_config):
    """Test that negative criteria can be UNMET (error not present = good)."""
    criterion_pattern = re.compile(r"<criterion_type>(.*?)</criterion_type>", re.DOTALL)

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        match = criterion_pattern.search(user_prompt)
        criterion_type = match.group(1).strip().lower() if match else "positive"

        if criterion_type == "negative":
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="Error not present",
            )
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Requirement met",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)

        report = await sample_rubric.grade("Test", grader=grader)

        assert report.score == pytest.approx(1.0)
        assert report.report is not None
        verdicts = [criterion.final_verdict for criterion in report.report]
        assert verdicts == [CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.UNMET]


@pytest.mark.asyncio
async def test_all_negative_criteria_all_unmet_returns_perfect_score(mock_llm_config):
    """All-negative rubric with no errors present should return 1.0."""
    rubric = Rubric([
        Criterion(weight=-1.0, requirement="Contains factual errors"),
        Criterion(weight=-1.0, requirement="Contains profanity"),
        Criterion(weight=-1.0, requirement="Contains harmful content"),
    ])

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.UNMET,
            explanation="Error not present",
        )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await rubric.grade("Clean, accurate text", grader=grader)

        assert result.score == pytest.approx(1.0)
        assert result.raw_score == pytest.approx(0.0)
        assert all(r.final_verdict == CriterionVerdict.UNMET for r in result.report)


@pytest.mark.asyncio
async def test_all_negative_criteria_all_met_returns_zero_score(mock_llm_config):
    """All-negative rubric with all errors present should return 0.0."""
    rubric = Rubric([
        Criterion(weight=-1.0, requirement="Contains factual errors"),
        Criterion(weight=-1.0, requirement="Contains profanity"),
        Criterion(weight=-1.0, requirement="Contains harmful content"),
    ])

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="Error is present",
        )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await rubric.grade("Bad text with errors", grader=grader)

        assert result.score == pytest.approx(0.0)
        assert result.raw_score == pytest.approx(-3.0)
        assert all(r.final_verdict == CriterionVerdict.MET for r in result.report)


@pytest.mark.asyncio
async def test_all_negative_criteria_partial_errors_returns_partial_score(mock_llm_config):
    """All-negative rubric with some errors should return partial score."""
    rubric = Rubric([
        Criterion(weight=-1.0, requirement="Contains factual errors"),
        Criterion(weight=-1.0, requirement="Contains profanity"),
        Criterion(weight=-1.0, requirement="Contains harmful content"),
    ])

    call_count = 0

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        nonlocal call_count
        call_count += 1
        # First criterion has error, others don't
        if call_count == 1:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Error is present",
            )
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="Error not present",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await rubric.grade("Text with one error", grader=grader)

        # 1 error out of 3: score = 1.0 + (-1.0 / 3.0) = 2/3 ~ 0.667
        assert result.score == pytest.approx(2.0 / 3.0)
        assert result.raw_score == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_all_negative_criteria_with_different_weights(mock_llm_config):
    """All-negative rubric with varying weights should weight errors appropriately."""
    rubric = Rubric([
        Criterion(weight=-2.0, requirement="Contains major factual errors"),
        Criterion(weight=-1.0, requirement="Contains minor typos"),
    ])

    call_count = 0

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        nonlocal call_count
        call_count += 1
        # Only the minor error (second criterion) is present
        if call_count == 2:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Minor error present",
            )
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="Error not present",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await rubric.grade("Text with minor error", grader=grader)

        # total_negative_weight = 3.0, weighted_score_sum = -1.0
        # score = 1.0 + (-1.0 / 3.0) = 2/3 ~ 0.667
        assert result.score == pytest.approx(2.0 / 3.0)
        assert result.raw_score == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_parse_failure_no_bias_with_negative_heavy_rubric(mock_llm_config):
    """Parse failures should not artificially inflate scores for negative-heavy rubrics.

    Previously, parse failures defaulted all criteria to UNMET, which meant:
    - Negative criteria were treated as "error not present" (good outcome)
    - This artificially inflated scores when the rubric had many negative criteria

    With the fix, negative criteria default to MET (error assumed present),
    ensuring parse failures result in worst-case scores.
    """
    # Rubric with mostly negative criteria (error detection focused)
    rubric = Rubric([
        Criterion(weight=1.0, requirement="Is helpful"),
        Criterion(weight=-1.0, requirement="Contains factual errors"),
        Criterion(weight=-1.0, requirement="Contains harmful content"),
        Criterion(weight=-1.0, requirement="Contains profanity"),
    ])

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=Exception("Cannot evaluate"))

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await rubric.grade("Test input", grader=grader)

        # With conservative defaults:
        # - Positive (weight=1.0): UNMET = 0 points
        # - Negative (weight=-1.0): MET = -1 point each (3 total = -3)
        # weighted_sum = 0 + (-1) + (-1) + (-1) = -3
        # total_positive = 1.0
        # score = max(0, -3/1) = 0.0
        assert result.score == 0.0

        # Verify verdicts
        verdicts = {r.criterion.requirement: r.final_verdict for r in result.report}
        assert verdicts["Is helpful"] == CriterionVerdict.UNMET
        assert verdicts["Contains factual errors"] == CriterionVerdict.MET
        assert verdicts["Contains harmful content"] == CriterionVerdict.MET
        assert verdicts["Contains profanity"] == CriterionVerdict.MET


@pytest.mark.asyncio
async def test_parse_failure_all_negative_rubric_returns_zero(mock_llm_config):
    """All-negative rubric with parse failures should return 0.0 (worst case)."""
    rubric = Rubric([
        Criterion(weight=-1.0, requirement="Contains errors"),
        Criterion(weight=-1.0, requirement="Contains harmful content"),
    ])

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=Exception("Invalid response"))

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await rubric.grade("Test", grader=grader)

        # All negative criteria default to MET (errors assumed present)
        # This gives the worst possible score for an all-negative rubric
        assert result.score == 0.0
        assert all(r.final_verdict == CriterionVerdict.MET for r in result.report)


@pytest.mark.asyncio
async def test_criterion_name_propagates_to_report(mock_llm_config):
    """Test that Criterion.name is propagated to CriterionReport."""
    rubric = Rubric([
        Criterion(name="accuracy", weight=2.0, requirement="Is factually accurate"),
        Criterion(name="clarity", weight=1.0, requirement="Is clearly written"),
        Criterion(weight=1.0, requirement="No name criterion"),  # name=None
    ])

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="Requirement met",
        )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await rubric.grade("Test submission", grader=grader)

        assert result.report is not None
        assert len(result.report) == 3

        # Verify names are propagated
        assert result.report[0].criterion.name == "accuracy"
        assert result.report[1].criterion.name == "clarity"
        assert result.report[2].criterion.name is None  # No name was set
