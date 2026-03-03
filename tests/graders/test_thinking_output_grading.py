"""Integration tests for thinking/output token support in grading using new LLMConfig-based API."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import Criterion, CriterionVerdict, LengthPenalty, Rubric, TokenUsage
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create a mock LLMConfig for testing."""
    return LLMConfig(model="test-model")


@pytest.fixture
def simple_criteria():
    """Simple criteria for testing."""
    return [
        Criterion(weight=10.0, requirement="Output is correct"),
        Criterion(weight=5.0, requirement="Output is well-explained"),
    ]


def create_all_met_mock_client() -> MagicMock:
    """Generate function that marks all criteria as MET."""

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="Requirement satisfied.",
        )

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

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)
    return mock_client


class TestInputFormats:
    """Test different input formats."""

    @pytest.mark.asyncio
    async def test_dict_input(self, simple_criteria, mock_llm_config):
        """Test grading with dict input."""
        rubric = Rubric(simple_criteria)
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(llm_config=mock_llm_config)

            result = await rubric.grade(
                {"thinking": "reasoning...", "output": "answer"}, grader=grader
            )

            assert result.score == 1.0
            assert result.raw_score == 15.0

    @pytest.mark.asyncio
    async def test_string_with_markers(self, simple_criteria, mock_llm_config):
        """Test grading with string containing markers."""
        rubric = Rubric(simple_criteria)
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(llm_config=mock_llm_config)

            result = await rubric.grade(
                "<thinking>reasoning</thinking><output>answer</output>", grader=grader
            )

            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_plain_string_backwards_compatible(self, simple_criteria, mock_llm_config):
        """Test plain string still works (backwards compatible)."""
        rubric = Rubric(simple_criteria)
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(llm_config=mock_llm_config)

            result = await rubric.grade("plain response", grader=grader)

            assert result.score == 1.0


class TestLengthPenaltyWithPenaltyType:
    """Test length penalty with different penalty types."""

    @pytest.mark.asyncio
    async def test_output_only_penalty(self, simple_criteria, mock_llm_config):
        """Test OUTPUT_ONLY penalty ignores thinking length."""
        rubric = Rubric(simple_criteria)
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
            llm_config=mock_llm_config,
                length_penalty=LengthPenalty(
                    free_budget=5, max_cap=10, penalty_at_cap=0.5, penalty_type="OUTPUT_ONLY"
                ),
            )

            # Long thinking, short output
            result = await rubric.grade(
                {"thinking": " ".join(["word"] * 100), "output": "short"}, grader=grader
            )

            assert result.score == 1.0  # No penalty despite long thinking

    @pytest.mark.asyncio
    async def test_thinking_only_penalty(self, simple_criteria, mock_llm_config):
        """Test THINKING_ONLY penalty ignores output length."""
        rubric = Rubric(simple_criteria)
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
            llm_config=mock_llm_config,
                length_penalty=LengthPenalty(
                    free_budget=5, max_cap=10, penalty_at_cap=0.5, penalty_type="THINKING_ONLY"
                ),
            )

            # Short thinking, long output
            result = await rubric.grade(
                {"thinking": "brief", "output": " ".join(["word"] * 100)}, grader=grader
            )

            assert result.score == 1.0  # No penalty despite long output

    @pytest.mark.asyncio
    async def test_all_penalty_type_default(self, simple_criteria, mock_llm_config):
        """Test that ALL penalty type counts both sections."""
        rubric = Rubric(simple_criteria)
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
            llm_config=mock_llm_config,
                length_penalty=LengthPenalty(
                    free_budget=5, max_cap=10, penalty_at_cap=0.5  # penalty_type="ALL" by default
                ),
            )

            # Long response
            result = await rubric.grade(" ".join(["word"] * 20), grader=grader)

            assert result.score < 1.0  # Should be penalized

    @pytest.mark.asyncio
    async def test_raw_scores_with_penalty(self, simple_criteria, mock_llm_config):
        """Test raw scores (normalize=False) with length penalty."""
        rubric = Rubric(simple_criteria)
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
            llm_config=mock_llm_config,
                normalize=False,
                length_penalty=LengthPenalty(
                    free_budget=5, max_cap=10, penalty_at_cap=50.0, penalty_type="OUTPUT_ONLY"
                ),
            )

            result = await rubric.grade(
                {"thinking": "brief", "output": " ".join(["word"] * 20)}, grader=grader
            )

            assert result.score < 0  # Heavily penalized in raw score mode
            assert result.raw_score == 15.0  # Base score before penalty
