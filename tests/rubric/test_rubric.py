"""Tests for Rubric class."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric, TokenUsage
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment

MOCK_DATASET = [
    {
        "input": "What is 2+2?",
        "output": "The answer is 4.",
        "expected": "4",
    },
    {
        "input": "What is the capital of France?",
        "output": "Paris is the capital of France.",
        "expected": "Paris",
    },
    {
        "input": "List three primary colors",
        "output": "Red, blue, and yellow are the three primary colors.",
        "expected": "red, blue, yellow",
    },
]


def create_mock_client() -> MagicMock:
    """Create a mock LLMClient that returns MET judgments."""

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=None,
            parsed=CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Mock test explanation",
            ),
        )

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)
    return mock_client


@pytest.mark.asyncio
async def test_rubric():
    """Test rubric grading with mock LLM client."""
    formatting_criteria = [
        Criterion(
            weight=1.0,
            requirement="Output uses proper capitalization and punctuation",
        ),
        Criterion(
            weight=1.0,
            requirement="Output is concise and avoids unnecessary verbosity",
        ),
    ]

    formatting_rubric = Rubric(formatting_criteria)

    mock_config = LLMConfig(model="test-model")
    mock_client = create_mock_client()

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(judge_model_config=mock_config)

        for idx, dataset_item in enumerate(MOCK_DATASET):
            correctness_criteria = [
                Criterion(
                    weight=2.0,
                    requirement=f"Output correctly answers the question: '{dataset_item['input']}'",
                ),
                Criterion(
                    weight=1.0,
                    requirement=(
                        f"Output includes the expected information: '{dataset_item['expected']}'"
                    ),
                ),
            ]

            correctness_rubric = Rubric(correctness_criteria)

            correctness_report = await correctness_rubric.grade(
                dataset_item["output"],
                grader=grader,
            )

            formatting_report = await formatting_rubric.grade(
                dataset_item["output"],
                grader=grader,
            )

            assert correctness_report is not None, f"Item {idx + 1}: Correctness report is None"
            assert formatting_report is not None, f"Item {idx + 1}: Formatting report is None"

            # Every criterion is met, so both normalized scores are 1.0.
            assert correctness_report.error is None and formatting_report.error is None
            assert correctness_report.score == 1.0, (
                f"Item {idx + 1}: Correctness score {correctness_report.score}"
            )
            assert formatting_report.score == 1.0, (
                f"Item {idx + 1}: Formatting score {formatting_report.score}"
            )

            assert correctness_report.report is not None, (
                f"Item {idx + 1}: Correctness report is None"
            )
            assert formatting_report.report is not None, (
                f"Item {idx + 1}: Formatting report is None"
            )

            assert len(correctness_report.report) == len(correctness_criteria), (
                f"Item {idx + 1}: Wrong number of correctness criteria"
            )

            assert len(formatting_report.report) == len(formatting_criteria), (
                f"Item {idx + 1}: Wrong number of formatting criteria"
            )

            for criterion in correctness_report.report:
                assert criterion.final_verdict in [CriterionVerdict.MET, CriterionVerdict.UNMET], (
                    f"Item {idx + 1}: Invalid correctness verdict {criterion.final_verdict}"
                )
                assert criterion.final_reason, (
                    f"Item {idx + 1}: Missing reason for correctness criterion"
                )

            for criterion in formatting_report.report:
                assert criterion.final_verdict in [CriterionVerdict.MET, CriterionVerdict.UNMET], (
                    f"Item {idx + 1}: Invalid formatting verdict {criterion.final_verdict}"
                )
                assert criterion.final_reason, (
                    f"Item {idx + 1}: Missing reason for formatting criterion"
                )
