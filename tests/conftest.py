"""Common fixtures for autorubric tests."""

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric, TokenUsage
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment

CriterionList = list[Criterion]


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create a mock LLMConfig for testing."""
    return LLMConfig(model="test-model")


@pytest.fixture
def sample_output() -> str:
    return "Paris is the capital of France. It is a beautiful city with rich history."


@pytest.fixture
def sample_criteria() -> CriterionList:
    return [
        Criterion(
            weight=2.0,
            requirement="Output mentions Paris",
        ),
        Criterion(
            weight=1.0,
            requirement="Output mentions France",
        ),
        Criterion(
            weight=1.0,
            requirement="Output is written in complete sentences",
        ),
        Criterion(
            weight=-0.5,
            requirement="Output contains profanity or offensive language",
        ),
    ]


@pytest.fixture
def sample_rubric(sample_criteria: CriterionList) -> Rubric:
    return Rubric(sample_criteria)


def _extract_field(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        raise ValueError("Expected field not found in prompt")
    return match.group(1).strip()


def create_per_criterion_mock_client(sample_criteria: CriterionList | None = None) -> MagicMock:
    """Create a mock LLMClient for CriterionGrader tests.

    This mock returns GenerateResult objects with CriterionJudgment in the parsed field.
    """
    criterion_pattern = re.compile(r"<criterion>(.*?)</criterion>", re.DOTALL)
    type_pattern = re.compile(r"<criterion_type>(.*?)</criterion_type>", re.DOTALL)
    positive_requirements_met = {
        "Output mentions Paris",
        "Output mentions France",
        "Output is written in complete sentences",
    }
    negative_errors_present = {
        "Output contains profanity or offensive language": False,  # Error NOT present
    }

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        criterion_text = _extract_field(criterion_pattern, user_prompt)
        criterion_type = _extract_field(type_pattern, user_prompt).lower()

        if criterion_type == "negative":
            # For negative criteria: criterion_status=MET means error IS present (bad)
            # criterion_status=UNMET means error is NOT present (good)
            error_present = negative_errors_present.get(criterion_text, False)
            explanation = (
                "Error detected in the output."
                if error_present
                else "Error not present in the output."
            )
            judgment = CriterionJudgment(
                criterion_status=(
                    CriterionVerdict.MET if error_present else CriterionVerdict.UNMET
                ),
                explanation=explanation,
            )
        else:
            criteria_met = criterion_text in positive_requirements_met
            explanation = (
                "Requirement satisfied by the submission."
                if criteria_met
                else "Requirement not satisfied by the submission."
            )
            judgment = CriterionJudgment(
                criterion_status=(
                    CriterionVerdict.MET if criteria_met else CriterionVerdict.UNMET
                ),
                explanation=explanation,
            )

        # Return GenerateResult when return_result=True (as the grader now uses)
        if return_result:
            return GenerateResult(
                content="{}",  # JSON content (not used when parsed is set)
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


@pytest.fixture
def per_criterion_mock_client(sample_criteria: CriterionList) -> MagicMock:
    """Mock LLMClient for CriterionGrader tests."""
    return create_per_criterion_mock_client(sample_criteria)
