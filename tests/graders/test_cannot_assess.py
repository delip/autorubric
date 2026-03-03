"""Tests for CANNOT_ASSESS verdict handling in graders."""

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    Criterion,
    CriterionVerdict,
    Rubric,
    TokenUsage,
)
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment


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


@pytest.fixture
def sample_rubric() -> Rubric:
    """Create a sample rubric for testing CANNOT_ASSESS."""
    return Rubric([
        Criterion(name="fact1", weight=10.0, requirement="States the correct fact 1"),
        Criterion(name="fact2", weight=5.0, requirement="States the correct fact 2"),
        Criterion(name="error", weight=-3.0, requirement="Contains factual errors"),
    ])


# =============================================================================
# Basic CANNOT_ASSESS verdict tests
# =============================================================================


@pytest.mark.asyncio
async def test_cannot_assess_count_in_report(mock_llm_config, sample_rubric):
    """Test that cannot_assess_count is correctly reported."""
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
        # First criterion: CANNOT_ASSESS
        if call_count == 1:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.CANNOT_ASSESS,
                explanation="Submission doesn't address this topic",
            )
        # Second criterion: MET
        elif call_count == 2:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Requirement met",
            )
        # Third criterion: UNMET
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="No errors present",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await sample_rubric.grade("Test submission", grader=grader)

        assert result.cannot_assess_count == 1
        assert result.report is not None
        assert result.report[0].final_verdict == CriterionVerdict.CANNOT_ASSESS


# =============================================================================
# SKIP strategy tests (default)
# =============================================================================


@pytest.mark.asyncio
async def test_skip_strategy_excludes_cannot_assess_from_scoring(
    mock_llm_config, sample_rubric
):
    """Test that SKIP strategy excludes CANNOT_ASSESS criteria from scoring."""
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
        # First criterion (weight=10): CANNOT_ASSESS
        if call_count == 1:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.CANNOT_ASSESS,
                explanation="Cannot assess",
            )
        # Second criterion (weight=5): MET
        elif call_count == 2:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Met",
            )
        # Third criterion (weight=-3): UNMET (no error)
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="No error",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        # Default strategy is SKIP
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await sample_rubric.grade("Test", grader=grader)

        # With SKIP: only fact2 (weight=5) is considered
        # Score = 5 / 5 = 1.0 (since fact1 is excluded)
        assert result.score == pytest.approx(1.0)
        assert result.cannot_assess_count == 1


@pytest.mark.asyncio
async def test_skip_strategy_all_cannot_assess_returns_zero(mock_llm_config):
    """Test SKIP strategy when all criteria are CANNOT_ASSESS."""
    rubric = Rubric([
        Criterion(weight=10.0, requirement="R1"),
        Criterion(weight=5.0, requirement="R2"),
    ])

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.CANNOT_ASSESS,
            explanation="Cannot assess",
        )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        result = await rubric.grade("Test", grader=grader)

        # All excluded -> 0 score
        assert result.score == 0.0
        assert result.cannot_assess_count == 2


# =============================================================================
# FAIL strategy tests
# =============================================================================


@pytest.mark.asyncio
async def test_fail_strategy_treats_cannot_assess_as_worst_case(
    mock_llm_config, sample_rubric
):
    """Test that FAIL strategy treats CANNOT_ASSESS as worst case."""
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
        # First criterion (positive, weight=10): CANNOT_ASSESS -> treated as UNMET
        if call_count == 1:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.CANNOT_ASSESS,
                explanation="Cannot assess",
            )
        # Second criterion (positive, weight=5): MET
        elif call_count == 2:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Met",
            )
        # Third criterion (negative, weight=-3): UNMET (no error)
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="No error",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config,
            cannot_assess_config=CannotAssessConfig(
                strategy=CannotAssessStrategy.FAIL
            ),
        )
        result = await sample_rubric.grade("Test", grader=grader)

        # With FAIL: CANNOT_ASSESS (weight=10) is treated as UNMET = 0 points
        # fact2 (weight=5): MET = 5 points
        # error (weight=-3): UNMET = 0 points (no penalty)
        # Total positive weight = 10 + 5 = 15
        # Score = 5 / 15 = 0.333...
        assert result.score == pytest.approx(5.0 / 15.0)


@pytest.mark.asyncio
async def test_fail_strategy_negative_criterion_cannot_assess(mock_llm_config):
    """Test FAIL strategy treats negative CANNOT_ASSESS as MET (error assumed)."""
    rubric = Rubric([
        Criterion(weight=10.0, requirement="Positive criterion"),
        Criterion(weight=-5.0, requirement="Contains errors"),
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
        if call_count == 1:
            # Positive criterion: MET
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Met",
            )
        else:
            # Negative criterion: CANNOT_ASSESS -> treated as MET (error assumed)
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.CANNOT_ASSESS,
                explanation="Cannot assess",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config,
            cannot_assess_config=CannotAssessConfig(
                strategy=CannotAssessStrategy.FAIL
            ),
        )
        result = await rubric.grade("Test", grader=grader)

        # Positive (weight=10): MET = 10 points
        # Negative (weight=-5): CANNOT_ASSESS -> MET = -5 points
        # weighted_sum = 10 - 5 = 5
        # total_positive = 10
        # Score = 5 / 10 = 0.5
        assert result.score == pytest.approx(0.5)


# =============================================================================
# ZERO strategy tests
# =============================================================================


@pytest.mark.asyncio
async def test_zero_strategy_treats_cannot_assess_as_unmet(
    mock_llm_config, sample_rubric
):
    """Test that ZERO strategy treats CANNOT_ASSESS as UNMET (0 contribution)."""
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
        if call_count == 1:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.CANNOT_ASSESS,
                explanation="Cannot assess",
            )
        elif call_count == 2:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Met",
            )
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="No error",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config,
            cannot_assess_config=CannotAssessConfig(
                strategy=CannotAssessStrategy.ZERO
            ),
        )
        result = await sample_rubric.grade("Test", grader=grader)

        # CANNOT_ASSESS (weight=10): treated as UNMET = 0
        # MET (weight=5): 5 points
        # UNMET (weight=-3): 0 points
        # Total positive = 15, Score = 5/15 = 0.333...
        assert result.score == pytest.approx(5.0 / 15.0)


# =============================================================================
# PARTIAL strategy tests
# =============================================================================


@pytest.mark.asyncio
async def test_partial_strategy_gives_partial_credit(mock_llm_config, sample_rubric):
    """Test that PARTIAL strategy gives partial credit for CANNOT_ASSESS."""
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
        if call_count == 1:
            # Positive criterion (weight=10): CANNOT_ASSESS
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.CANNOT_ASSESS,
                explanation="Cannot assess",
            )
        elif call_count == 2:
            # Positive criterion (weight=5): MET
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Met",
            )
        else:
            # Negative criterion (weight=-3): UNMET
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="No error",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config,
            cannot_assess_config=CannotAssessConfig(
                strategy=CannotAssessStrategy.PARTIAL,
                partial_credit=0.5,  # 50% credit
            ),
        )
        result = await sample_rubric.grade("Test", grader=grader)

        # CANNOT_ASSESS (weight=10): 0.5 * 10 = 5 points
        # MET (weight=5): 5 points
        # UNMET (weight=-3): 0 points
        # Total = 10, total_positive = 15
        # Score = 10/15 = 0.667...
        assert result.score == pytest.approx(10.0 / 15.0)


@pytest.mark.asyncio
async def test_partial_strategy_custom_credit(mock_llm_config):
    """Test PARTIAL strategy with custom partial_credit value."""
    rubric = Rubric([
        Criterion(weight=10.0, requirement="R1"),
    ])

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.CANNOT_ASSESS,
            explanation="Cannot assess",
        )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config,
            cannot_assess_config=CannotAssessConfig(
                strategy=CannotAssessStrategy.PARTIAL,
                partial_credit=0.3,  # 30% credit
            ),
        )
        result = await rubric.grade("Test", grader=grader)

        # CANNOT_ASSESS: 0.3 * 10 = 3 points
        # Score = 3/10 = 0.3
        assert result.score == pytest.approx(0.3)


# =============================================================================
# Mixed verdict tests
# =============================================================================


@pytest.mark.asyncio
async def test_mixed_verdicts_all_strategies(mock_llm_config):
    """Test grading with all three verdict types (MET, UNMET, CANNOT_ASSESS)."""
    rubric = Rubric([
        Criterion(weight=10.0, requirement="R1"),
        Criterion(weight=10.0, requirement="R2"),
        Criterion(weight=10.0, requirement="R3"),
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
        if call_count == 1:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Met",
            )
        elif call_count == 2:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="Not met",
            )
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.CANNOT_ASSESS,
                explanation="Cannot assess",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        # SKIP strategy
        grader_skip = CriterionGrader(
            llm_config=mock_llm_config,
            cannot_assess_config=CannotAssessConfig(strategy=CannotAssessStrategy.SKIP),
        )
        # Reset call count
        call_count = 0
        result_skip = await rubric.grade("Test", grader=grader_skip)
        # SKIP: only MET (10) and UNMET (0) counted, total_positive = 20
        # Score = 10/20 = 0.5
        assert result_skip.score == pytest.approx(0.5)

        # FAIL strategy
        call_count = 0
        grader_fail = CriterionGrader(
            llm_config=mock_llm_config,
            cannot_assess_config=CannotAssessConfig(strategy=CannotAssessStrategy.FAIL),
        )
        result_fail = await rubric.grade("Test", grader=grader_fail)
        # FAIL: CANNOT_ASSESS -> UNMET, so 10 + 0 + 0 = 10, total = 30
        # Score = 10/30 = 0.333...
        assert result_fail.score == pytest.approx(10.0 / 30.0)

        # PARTIAL strategy (0.5)
        call_count = 0
        grader_partial = CriterionGrader(
            llm_config=mock_llm_config,
            cannot_assess_config=CannotAssessConfig(
                strategy=CannotAssessStrategy.PARTIAL, partial_credit=0.5
            ),
        )
        result_partial = await rubric.grade("Test", grader=grader_partial)
        # PARTIAL: 10 (MET) + 0 (UNMET) + 5 (0.5*10) = 15, total = 30
        # Score = 15/30 = 0.5
        assert result_partial.score == pytest.approx(0.5)


# =============================================================================
# Raw score (normalize=False) tests
# =============================================================================


@pytest.mark.asyncio
async def test_raw_score_with_cannot_assess(mock_llm_config, sample_rubric):
    """Test raw scores with CANNOT_ASSESS verdicts."""
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
        if call_count == 1:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.CANNOT_ASSESS,
                explanation="Cannot assess",
            )
        elif call_count == 2:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.MET,
                explanation="Met",
            )
        else:
            judgment = CriterionJudgment(
                criterion_status=CriterionVerdict.UNMET,
                explanation="No error",
            )
        return _wrap_in_generate_result(judgment, return_result)

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=mock_client,
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config,
            normalize=False,
            cannot_assess_config=CannotAssessConfig(
                strategy=CannotAssessStrategy.PARTIAL,
                partial_credit=0.5,
            ),
        )
        result = await sample_rubric.grade("Test", grader=grader)

        # Raw score: 0.5*10 + 5 + 0 = 10
        assert result.score == pytest.approx(10.0)
        assert result.raw_score == pytest.approx(10.0)
