"""Tests for length penalty functionality using the new LLMConfig-based API."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import Criterion, CriterionVerdict, LengthPenalty, Rubric, TokenUsage
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment
from autorubric.utils import compute_length_penalty, word_count


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create a mock LLMConfig for testing."""
    return LLMConfig(model="test-model")


@pytest.fixture
def simple_rubric() -> Rubric:
    return Rubric(
        [
            Criterion(weight=10.0, requirement="Contains greeting"),
            Criterion(weight=5.0, requirement="Contains farewell"),
            Criterion(weight=-3.0, requirement="Contains profanity"),
        ]
    )


def create_all_met_mock_client() -> MagicMock:
    """Create a mock LLMClient that marks all criteria appropriately."""

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        if "negative" in user_prompt.lower():
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="No profanity found",
            )
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Requirement satisfied",
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


class TestLengthPenaltyComputation:
    @pytest.mark.parametrize(
        "n_words,exponent,expected,bounded",
        [
            # under-budget region: 50 words < free_budget=100 -> no penalty
            (50, 2.0, 0.0, False),
            # at/over-cap region: 250 words > max_cap=200 -> max penalty
            (250, 2.0, 50.0, False),
            # midpoint between budget and cap (exponent=1.0): linear -> 25.0,
            # and strictly between 0.0 and the cap penalty
            (150, 1.0, 25.0, True),
        ],
    )
    def test_compute_length_penalty_regions(self, n_words, exponent, expected, bounded):
        config = LengthPenalty(free_budget=100, max_cap=200, penalty_at_cap=50.0, exponent=exponent)
        text = " ".join(["word"] * n_words)
        penalty = compute_length_penalty(text, config)
        if bounded:
            assert 0.0 < penalty < 50.0
        assert penalty == pytest.approx(expected)

    def test_custom_count_fn(self):
        config = LengthPenalty(
            free_budget=10,
            max_cap=20,
            penalty_at_cap=10.0,
            count_fn=lambda text: len(text),
        )
        text = "a" * 25
        assert compute_length_penalty(text, config) == 10.0


class TestWordCount:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("hello world", 2),  # basic two words
            ("", 0),  # empty-string boundary: ''.split() == [] -> 0
            ("hello   world", 2),  # collapsing repeated whitespace
        ],
    )
    def test_word_count(self, text, expected):
        assert word_count(text) == expected


@pytest.mark.asyncio
class TestLengthPenaltyWithNormalize:
    async def test_normalized_score_with_length_penalty(self, simple_rubric, mock_llm_config):
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=True,
                length_penalty=LengthPenalty(
                    free_budget=10,
                    max_cap=20,
                    penalty_at_cap=0.5,
                ),
            )

            short_text = "Hello goodbye"
            result = await simple_rubric.grade(short_text, grader=grader)

            assert result.raw_score == 15.0
            assert result.score == pytest.approx(1.0)

    async def test_normalized_score_with_length_penalty_applied(
        self, simple_rubric, mock_llm_config
    ):
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=True,
                length_penalty=LengthPenalty(
                    free_budget=5,
                    max_cap=10,
                    penalty_at_cap=0.5,
                ),
            )

            long_text = " ".join(["word"] * 15)
            result = await simple_rubric.grade(long_text, grader=grader)

            assert result.raw_score == 15.0
            assert result.score == pytest.approx(0.5)


@pytest.mark.asyncio
class TestLengthPenaltyWithoutNormalize:
    async def test_raw_score_no_length_penalty(self, simple_rubric, mock_llm_config):
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=False,
            )

            result = await simple_rubric.grade("Hello goodbye", grader=grader)

            assert result.raw_score == 15.0
            assert result.score == 15.0

    async def test_raw_score_with_length_penalty_under_budget(self, simple_rubric, mock_llm_config):
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=False,
                length_penalty=LengthPenalty(
                    free_budget=100,
                    max_cap=200,
                    penalty_at_cap=50.0,
                ),
            )

            short_text = "Hello goodbye"
            result = await simple_rubric.grade(short_text, grader=grader)

            assert result.raw_score == 15.0
            assert result.score == 15.0

    async def test_raw_score_with_length_penalty_at_cap(self, simple_rubric, mock_llm_config):
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=False,
                length_penalty=LengthPenalty(
                    free_budget=5,
                    max_cap=10,
                    penalty_at_cap=50.0,
                ),
            )

            long_text = " ".join(["word"] * 15)
            result = await simple_rubric.grade(long_text, grader=grader)

            assert result.raw_score == 15.0
            assert result.score == pytest.approx(-35.0)

    async def test_raw_score_with_partial_length_penalty(self, simple_rubric, mock_llm_config):
        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=False,
                length_penalty=LengthPenalty(
                    free_budget=5,
                    max_cap=15,
                    penalty_at_cap=10.0,
                    exponent=1.0,
                ),
            )

            text = " ".join(["word"] * 10)
            result = await simple_rubric.grade(text, grader=grader)

            assert result.raw_score == 15.0
            expected_penalty = 5.0
            assert result.score == pytest.approx(15.0 - expected_penalty)

    async def test_raw_score_can_go_negative(self, mock_llm_config):
        rubric = Rubric(
            [
                Criterion(weight=5.0, requirement="Contains greeting"),
            ]
        )

        mock_client = create_all_met_mock_client()

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=False,
                length_penalty=LengthPenalty(
                    free_budget=5,
                    max_cap=10,
                    penalty_at_cap=100.0,
                ),
            )

            long_text = " ".join(["word"] * 20)
            result = await rubric.grade(long_text, grader=grader)

            assert result.raw_score == 5.0
            assert result.score == pytest.approx(-95.0)


@pytest.mark.asyncio
class TestLengthPenaltyWithNegativeCriteria:
    async def test_negative_criteria_met_reduces_raw_score(self, mock_llm_config):
        rubric = Rubric(
            [
                Criterion(weight=10.0, requirement="Contains greeting"),
                Criterion(weight=-5.0, requirement="Contains spam"),
            ]
        )

        async def mock_generate(
            system_prompt: str,
            user_prompt: str,
            response_format: type | None = None,
            return_result: bool = False,
            **kwargs: Any,
        ) -> GenerateResult | CriterionJudgment:
            if "negative" in user_prompt.lower():
                judgment = CriterionJudgment(
                    criterion_status=CriterionVerdict.MET,
                    explanation="Spam detected",
                )
            else:
                judgment = CriterionJudgment(
                    criterion_status=CriterionVerdict.MET,
                    explanation="Greeting found",
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

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=False,
            )

            result = await rubric.grade("Hello spam", grader=grader)

            assert result.raw_score == 5.0
            assert result.score == 5.0

    async def test_negative_criteria_with_length_penalty(self, mock_llm_config):
        rubric = Rubric(
            [
                Criterion(weight=10.0, requirement="Contains greeting"),
                Criterion(weight=-5.0, requirement="Contains spam"),
            ]
        )

        async def mock_generate(
            system_prompt: str,
            user_prompt: str,
            response_format: type | None = None,
            return_result: bool = False,
            **kwargs: Any,
        ) -> GenerateResult | CriterionJudgment:
            if "negative" in user_prompt.lower():
                judgment = CriterionJudgment(
                    criterion_status=CriterionVerdict.MET,
                    explanation="Spam detected",
                )
            else:
                judgment = CriterionJudgment(
                    criterion_status=CriterionVerdict.MET,
                    explanation="Greeting found",
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

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=mock_client,
        ):
            grader = CriterionGrader(
                llm_config=mock_llm_config,
                normalize=False,
                length_penalty=LengthPenalty(
                    free_budget=5,
                    max_cap=10,
                    penalty_at_cap=10.0,
                ),
            )

            long_text = " ".join(["word"] * 15)
            result = await rubric.grade(long_text, grader=grader)

            assert result.raw_score == 5.0
            assert result.score == pytest.approx(-5.0)
