"""Tests for preserving the extended-thinking ``reasoning`` trace.

The judge LLM can emit a verbose extended-thinking *deliberation trace* (the
provider's ``reasoning_content`` channel), which ``LLMClient.generate`` injects into
``judgment.reasoning`` when thinking is enabled. ``reason``/``explanation`` is the
concise conclusion distilled from that trace. Historically the grader copied only
``explanation`` into ``CriterionReport.reason`` and dropped ``reasoning`` entirely.

These tests pin the data flow ``judgment.reasoning -> CriterionReport.reasoning ->
JudgeVote/MultiChoiceJudgeVote.reasoning`` end-to-end through ``Rubric.grade``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    Rubric,
    TokenUsage,
)
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment, EnsembleEvaluationReport, MultiChoiceJudgment

TRACE = "Step 1: read the submission. Step 2: weighed the evidence. Concluded MET."


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    return LLMConfig(model="test-model")


def _binary_result_with_reasoning() -> GenerateResult:
    return GenerateResult(
        content="{}",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        parsed=CriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="Requirement met.",
            reasoning=TRACE,
        ),
    )


def _mc_result_with_reasoning(selected_option: int) -> GenerateResult:
    return GenerateResult(
        content="{}",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        parsed=MultiChoiceJudgment(
            selected_option=selected_option,
            explanation="High quality.",
            reasoning=TRACE,
        ),
    )


@pytest.mark.asyncio
async def test_binary_reasoning_reaches_vote(mock_llm_config):
    """A binary judgment's reasoning trace flows onto the ensemble JudgeVote."""
    client = MagicMock()
    client.generate = AsyncMock(return_value=_binary_result_with_reasoning())
    rubric = Rubric([Criterion(weight=10.0, requirement="Is accurate")])

    with patch("autorubric.graders.criterion_grader.LLMClient", return_value=client):
        report = await rubric.grade(
            "submission", grader=CriterionGrader(llm_config=mock_llm_config)
        )

    assert isinstance(report, EnsembleEvaluationReport)
    vote = report.report[0].votes[0]
    assert vote.reasoning == TRACE
    # The brief explanation is still preserved separately on reason.
    assert vote.reason == "Requirement met."


@pytest.mark.asyncio
async def test_multi_choice_reasoning_reaches_vote(mock_llm_config):
    """A multi-choice judgment's reasoning trace flows onto the MultiChoiceJudgeVote."""
    client = MagicMock()
    client.generate = AsyncMock(return_value=_mc_result_with_reasoning(selected_option=2))
    options = [CriterionOption(label="Low", value=0.0), CriterionOption(label="High", value=1.0)]
    rubric = Rubric([Criterion(weight=10.0, requirement="Quality?", options=options)])

    with patch("autorubric.graders.criterion_grader.LLMClient", return_value=client):
        report = await rubric.grade(
            "submission", grader=CriterionGrader(llm_config=mock_llm_config)
        )

    assert isinstance(report, EnsembleEvaluationReport)
    mc_vote = report.report[0].multi_choice_votes[0]
    assert mc_vote.reasoning == TRACE
    assert mc_vote.reason == "High quality."
