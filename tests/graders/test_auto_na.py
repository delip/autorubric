"""Tests for the auto-injected multi-choice abstain channel (T2-A).

A multi-choice criterion that lacks an NA option has, historically, no way to
express "cannot assess" — the judge is forced to pick a scored option. T2-A gives
multi-choice criteria a first-class abstain channel analogous to binary
CANNOT_ASSESS by auto-injecting a canonical NA option (``CriterionGrader``'s
``auto_na_option=True`` default), with an opt-out for forced-choice classification.

These tests exercise the behavior end-to-end through ``Rubric.grade``.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
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
from autorubric.types import CriterionJudgment, MultiChoiceJudgment


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    return LLMConfig(model="test-model")


def _ok_mc_result(selected_option: int, explanation: str = "ok") -> GenerateResult:
    return GenerateResult(
        content="{}",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        parsed=MultiChoiceJudgment(selected_option=selected_option, explanation=explanation),
    )


def _ok_binary_result(verdict: CriterionVerdict, explanation: str = "ok") -> GenerateResult:
    return GenerateResult(
        content="{}",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        parsed=CriterionJudgment(criterion_status=verdict, explanation=explanation),
    )


def _client_raising(exc: BaseException) -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(side_effect=exc)
    return client


def _mc_client(selected_option: int) -> MagicMock:
    """A client that always returns the given multi-choice selection."""
    client = MagicMock()
    client.generate = AsyncMock(return_value=_ok_mc_result(selected_option))
    return client


def _routing_client(mc_selected_option: int, binary_verdict: CriterionVerdict) -> MagicMock:
    """A client that routes by prompt: multi-choice prompts have an <options> section."""

    async def gen(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult:
        if "<options>" in user_prompt:
            return _ok_mc_result(mc_selected_option)
        return _ok_binary_result(binary_verdict)

    client = MagicMock()
    client.generate = AsyncMock(side_effect=gen)
    return client


def _routing_client_mc_raises(binary_verdict: CriterionVerdict, exc: BaseException) -> MagicMock:
    """Routes by prompt: raises ``exc`` on multi-choice prompts, returns ``binary_verdict``
    on binary prompts."""

    async def gen(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult:
        if "<options>" in user_prompt:
            raise exc
        return _ok_binary_result(binary_verdict)

    client = MagicMock()
    client.generate = AsyncMock(side_effect=gen)
    return client


def _na_free_ordinal(weight: float = 5.0) -> Criterion:
    return Criterion(
        name="quality",
        requirement="How good is it?",
        weight=weight,
        scale_type="ordinal",
        options=[
            CriterionOption(label="Bad", value=0.0),
            CriterionOption(label="Ok", value=0.5),
            CriterionOption(label="Great", value=1.0),
        ],
    )


def _ordinal_with_author_na() -> Criterion:
    return Criterion(
        name="citations",
        requirement="Are there citations?",
        weight=8.0,
        scale_type="ordinal",
        options=[
            CriterionOption(label="None", value=0.0),
            CriterionOption(label="Some", value=0.5),
            CriterionOption(label="All", value=1.0),
            CriterionOption(label="N/A", value=0.0, na=True),
        ],
    )


@pytest.mark.asyncio
async def test_judge_can_select_injected_na_option(mock_llm_config):
    """Default: a NA-free multi-choice criterion gains an abstain option the judge can pick.

    The criterion declares 3 options; with auto-injection the judge may select option 4
    (the injected NA) and abstain — previously option 4 was out of range (an error).
    """
    rubric = Rubric([_na_free_ordinal()])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_mc_client(selected_option=4),  # the injected NA option
    ):
        grader = CriterionGrader(llm_config=mock_llm_config, shuffle_options=False)
        report = await rubric.grade("submission", grader=grader)

    cr = report.report[0]
    # The effective criterion carries the injected NA at the last index.
    assert len(cr.criterion.options) == 4
    assert cr.criterion.options[-1].na is True
    # The judge abstained — a genuine NA verdict (not an error).
    assert cr.final_multi_choice_verdict is not None
    assert cr.final_multi_choice_verdict.na is True
    assert cr.final_multi_choice_verdict.selected_index == 3
    assert not cr.is_error
    # Excluded from scoring under the default SKIP strategy.
    assert report.score == 0.0


@pytest.mark.asyncio
async def test_auto_na_off_does_not_inject(mock_llm_config):
    """auto_na_option=False keeps the author's option set (forced choice)."""
    rubric = Rubric([_na_free_ordinal()])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_mc_client(selected_option=1),  # "Bad"
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config, shuffle_options=False, auto_na_option=False
        )
        report = await rubric.grade("submission", grader=grader)

    cr = report.report[0]
    assert len(cr.criterion.options) == 3  # no injection
    assert all(not o.na for o in cr.criterion.options)
    assert cr.final_multi_choice_verdict is not None
    assert cr.final_multi_choice_verdict.na is False
    assert cr.final_multi_choice_verdict.selected_index == 0


@pytest.mark.asyncio
async def test_author_na_not_stripped_when_auto_off(mock_llm_config):
    """auto_na_option=False must NOT remove an author-supplied NA option."""
    rubric = Rubric([_ordinal_with_author_na()])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_mc_client(selected_option=1),
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config, shuffle_options=False, auto_na_option=False
        )
        report = await rubric.grade("submission", grader=grader)

    cr = report.report[0]
    assert len(cr.criterion.options) == 4
    assert sum(1 for o in cr.criterion.options if o.na) == 1  # author NA preserved


@pytest.mark.asyncio
async def test_author_na_not_duplicated_when_auto_on(mock_llm_config):
    """Default auto-injection is idempotent — an author NA option is not duplicated."""
    rubric = Rubric([_ordinal_with_author_na()])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_mc_client(selected_option=1),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config, shuffle_options=False)
        report = await rubric.grade("submission", grader=grader)

    cr = report.report[0]
    assert len(cr.criterion.options) == 4  # no extra NA injected
    assert sum(1 for o in cr.criterion.options if o.na) == 1


@pytest.mark.asyncio
async def test_infra_failure_points_at_genuine_injected_na(mock_llm_config):
    """T2-B default-case fix: an infra error abstains via the genuine (injected) NA option.

    Previously, with no NA option, the grader set na=True against a real scored option
    (index 0) — an internally contradictory verdict. With auto-injection the abstain
    verdict points at an option that is genuinely na=True.
    """
    rubric = Rubric([_na_free_ordinal()])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(litellm.Timeout("timed out", model="m", llm_provider="p")),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config, shuffle_options=False)
        report = await rubric.grade("submission", grader=grader)

    cr = report.report[0]
    assert cr.final_multi_choice_verdict is not None
    mcv = cr.final_multi_choice_verdict
    assert mcv.na is True
    # Points at the genuinely-NA option (index 3), not a scored option.
    assert mcv.selected_index == 3
    assert cr.criterion.options[mcv.selected_index].na is True
    assert mcv.value == 0.0
    assert cr.is_error
    assert cr.error is not None and cr.error.startswith("infrastructure:")
    assert report.score == 0.0


@pytest.mark.asyncio
async def test_injected_na_excluded_under_skip_like_binary(mock_llm_config):
    """Parity: an abstained injected-NA multi-choice criterion drops out under SKIP,
    exactly like a binary CANNOT_ASSESS, leaving the binary criterion to drive the score.
    """
    rubric = Rubric(
        [
            Criterion(name="accurate", requirement="Is it accurate?", weight=5.0),
            _na_free_ordinal(weight=8.0),
        ]
    )
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        # Multi-choice → injected NA (option 4); binary → MET.
        return_value=_routing_client(mc_selected_option=4, binary_verdict=CriterionVerdict.MET),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config, shuffle_options=False)
        report = await rubric.grade("submission", grader=grader)

    mc_cr = report.report[1]
    assert mc_cr.is_na is True
    # A clean abstain (judge selected the injected NA), NOT an error-induced NA.
    assert not mc_cr.is_error
    # Only the binary MET (weight 5) counts: 5 / 5 = 1.0.
    assert report.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_forced_choice_infra_error_is_genuine_abstain(mock_llm_config):
    """T2-B residual fix: forced-choice (auto_na_option=False) criterion with no NA option.

    On an infrastructure/parse error there is no NA option to abstain into. The verdict
    must be a GENUINE abstain — na=True with selected_index/selected_label = None — never
    na=True pointing at a real scored option (the old contradiction).
    """
    rubric = Rubric([_na_free_ordinal()])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(litellm.Timeout("timed out", model="m", llm_provider="p")),
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config, shuffle_options=False, auto_na_option=False
        )
        report = await rubric.grade("submission", grader=grader)

    cr = report.report[0]
    # No NA option was injected (forced choice).
    assert len(cr.criterion.options) == 3
    assert all(not o.na for o in cr.criterion.options)
    mcv = cr.final_multi_choice_verdict
    assert mcv is not None
    assert mcv.na is True
    # Genuine abstain: no option is selected.
    assert mcv.selected_index is None
    assert mcv.selected_label is None
    assert mcv.value == 0.0
    assert cr.is_error
    assert cr.error is not None and cr.error.startswith("infrastructure:")
    assert cr.is_na is True
    # Single abstained criterion → excluded under SKIP → empty denominator → 0.0.
    assert report.score == 0.0


@pytest.mark.asyncio
async def test_forced_choice_infra_error_excluded_under_skip(mock_llm_config):
    """The genuine abstain stays na=True, so under the default SKIP strategy it drops out
    of the denominator and never penalizes the submission — exactly like binary CANNOT_ASSESS.
    """
    rubric = Rubric(
        [
            Criterion(name="accurate", requirement="Is it accurate?", weight=5.0),
            _na_free_ordinal(weight=8.0),
        ]
    )
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_routing_client_mc_raises(
            binary_verdict=CriterionVerdict.MET,
            exc=litellm.Timeout("timed out", model="m", llm_provider="p"),
        ),
    ):
        grader = CriterionGrader(
            llm_config=mock_llm_config, shuffle_options=False, auto_na_option=False
        )
        report = await rubric.grade("submission", grader=grader)

    mc_cr = report.report[1]
    assert mc_cr.is_na is True
    assert mc_cr.is_error
    assert mc_cr.final_multi_choice_verdict is not None
    assert mc_cr.final_multi_choice_verdict.selected_index is None
    # Only the binary MET (weight 5) counts: 5 / 5 = 1.0; the errored multi-choice is excluded.
    assert report.score == pytest.approx(1.0)
