"""Tests for grading-error classification and error routing.

Covers:
- ``classify_grading_error`` taxonomy (infrastructure / parse / unknown).
- Binary criterion error routing in ``CriterionGrader`` (infra/parse -> CANNOT_ASSESS,
  unknown -> conservative worst-case) and the resulting score behavior (errored
  criteria dropped from the denominator under the default SKIP strategy).
- Ensemble aggregation: an ensemble error is set only when every contributing judge
  vote errored; a mixed ensemble keeps a real verdict from the successful judge.
- Multi-choice infra failure -> NA verdict (excluded from scoring).
- Serialization round-trip of the ``error`` field on ensemble reports and judge votes.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from pydantic import BaseModel, ValidationError

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    ErrorCategory,
    Rubric,
    TokenUsage,
    classify_grading_error,
)
from autorubric.dataset import DataItem
from autorubric.eval import ItemResult
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment, MultiChoiceJudgment


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    return LLMConfig(model="test-model")


def _ok_binary_result(verdict: CriterionVerdict, explanation: str = "ok") -> GenerateResult:
    """A successful binary GenerateResult with a parsed CriterionJudgment."""
    return GenerateResult(
        content="{}",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        parsed=CriterionJudgment(criterion_status=verdict, explanation=explanation),
    )


def _ok_mc_result(selected_option: int, explanation: str = "ok") -> GenerateResult:
    """A successful multi-choice GenerateResult with a parsed MultiChoiceJudgment."""
    return GenerateResult(
        content="{}",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        parsed=MultiChoiceJudgment(selected_option=selected_option, explanation=explanation),
    )


def _client_raising(exc: BaseException) -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(side_effect=exc)
    return client


# =============================================================================
# classify_grading_error unit tests
# =============================================================================


class TestClassifyGradingError:
    """Unit tests for the error taxonomy."""

    def test_litellm_timeout_is_infrastructure(self):
        exc = litellm.Timeout("timed out", model="m", llm_provider="p")
        assert classify_grading_error(exc) == "infrastructure"

    def test_litellm_rate_limit_is_infrastructure(self):
        exc = litellm.RateLimitError("rate limited", model="m", llm_provider="p")
        assert classify_grading_error(exc) == "infrastructure"

    def test_json_decode_error_is_parse(self):
        try:
            json.loads("{")
        except ValueError as e:  # json.JSONDecodeError subclasses ValueError
            assert classify_grading_error(e) == "parse"
        else:
            pytest.fail("json.loads('{') should have raised")

    def test_pydantic_validation_error_is_parse(self):
        class _M(BaseModel):
            x: int

        try:
            _M.model_validate({"x": "not-an-int"})
        except ValidationError as e:
            assert classify_grading_error(e) == "parse"
        else:
            pytest.fail("expected a ValidationError")

    def test_value_error_is_parse(self):
        assert classify_grading_error(ValueError("bad value")) == "parse"

    def test_runtime_error_is_unknown(self):
        assert classify_grading_error(RuntimeError("boom")) == "unknown"

    def test_return_type_is_error_category_literal(self):
        # Sanity: the documented literal values are exactly what we return.
        result: ErrorCategory = classify_grading_error(RuntimeError("boom"))
        assert result in ("infrastructure", "parse", "unknown")


# =============================================================================
# Binary criterion error routing
# =============================================================================


@pytest.mark.asyncio
async def test_binary_infrastructure_failure_cannot_assess_and_no_penalty(mock_llm_config):
    """Infra failure -> CANNOT_ASSESS, flagged as error, and excluded from scoring.

    The errored criterion must drop out of the denominator under the default SKIP
    strategy: the score should equal a baseline where that criterion is simply absent.
    """
    # Two positive criteria. The second one's judge call fails with an infra error.
    rubric = Rubric([
        Criterion(weight=2.0, requirement="Criterion A"),
        Criterion(weight=1.0, requirement="Criterion B"),
    ])

    async def mock_generate(system_prompt, user_prompt, **kwargs) -> Any:
        if "Criterion B" in user_prompt:
            raise litellm.Timeout("timed out", model="m", llm_provider="p")
        return _ok_binary_result(CriterionVerdict.MET)

    client = MagicMock()
    client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    a_report, b_report = report.report[0], report.report[1]

    # Criterion A succeeded.
    assert a_report.final_verdict == CriterionVerdict.MET
    assert not a_report.is_error
    assert a_report.error is None

    # Criterion B failed with infrastructure error -> CANNOT_ASSESS, flagged.
    assert b_report.final_verdict == CriterionVerdict.CANNOT_ASSESS
    assert b_report.is_error
    assert b_report.error is not None
    assert b_report.error.startswith("infrastructure:")

    # Score is NOT penalized: B is excluded from the denominator (SKIP).
    # Only A remains: MET with weight 2.0 -> 2.0 / 2.0 = 1.0.
    assert report.score == pytest.approx(1.0)

    # Baseline: a rubric where the errored criterion is simply absent.
    baseline_rubric = Rubric([Criterion(weight=2.0, requirement="Criterion A")])

    async def baseline_generate(system_prompt, user_prompt, **kwargs) -> Any:
        return _ok_binary_result(CriterionVerdict.MET)

    baseline_client = MagicMock()
    baseline_client.generate = AsyncMock(side_effect=baseline_generate)
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=baseline_client,
    ):
        baseline_grader = CriterionGrader(llm_config=mock_llm_config)
        baseline_report = await baseline_rubric.grade("submission", grader=baseline_grader)

    assert report.score == pytest.approx(baseline_report.score)


@pytest.mark.asyncio
async def test_binary_parse_failure_cannot_assess(mock_llm_config):
    """Parse failure (pydantic.ValidationError) -> CANNOT_ASSESS with parse error."""

    class _Schema(BaseModel):
        x: int

    try:
        _Schema.model_validate({"x": "nope"})
    except ValidationError as e:
        parse_exc: ValidationError = e

    rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(parse_exc),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]
    assert cr.final_verdict == CriterionVerdict.CANNOT_ASSESS
    assert cr.is_error
    assert cr.error is not None
    assert cr.error.startswith("parse:")


@pytest.mark.asyncio
async def test_binary_unknown_failure_keeps_worst_case(mock_llm_config):
    """Unknown failure (RuntimeError) -> conservative worst-case verdict, flagged.

    Positive-weight criterion -> UNMET; negative-weight criterion -> MET.
    """
    rubric = Rubric([
        Criterion(weight=2.0, requirement="Positive criterion"),
        Criterion(weight=-1.0, requirement="Negative criterion"),
    ])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(RuntimeError("boom")),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    pos, neg = report.report[0], report.report[1]

    assert pos.final_verdict == CriterionVerdict.UNMET
    assert pos.is_error
    assert pos.error is not None
    assert pos.error.startswith("unknown:")

    assert neg.final_verdict == CriterionVerdict.MET
    assert neg.is_error
    assert neg.error is not None
    assert neg.error.startswith("unknown:")


# =============================================================================
# Ensemble aggregation with errors
# =============================================================================


def _two_judge_grader() -> CriterionGrader:
    return CriterionGrader(
        judges=[
            JudgeSpec(LLMConfig(model="judge-a-model"), "judge_a"),
            JudgeSpec(LLMConfig(model="judge-b-model"), "judge_b"),
        ],
        aggregation="majority",
    )


@pytest.mark.asyncio
async def test_ensemble_mixed_one_failure_keeps_successful_verdict():
    """One judge fails (infra), the other returns MET -> final verdict from success.

    The ensemble report's error must be None (is_error False) because at least one
    judge produced a genuine judgment.
    """
    rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])

    # Build distinct clients keyed by the judge's model so we can route per-judge.
    client_a = _client_raising(litellm.Timeout("timed out", model="m", llm_provider="p"))
    client_b = MagicMock()
    client_b.generate = AsyncMock(return_value=_ok_binary_result(CriterionVerdict.MET))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = _two_judge_grader()
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]

    # Successful judge drives the final verdict.
    assert cr.final_verdict == CriterionVerdict.MET
    # Mixed ensemble: not flagged as an error.
    assert cr.error is None
    assert not cr.is_error

    # One of the two votes carries an infra error, the other is clean.
    vote_errors = [v.error for v in cr.votes]
    assert any(e is not None and e.startswith("infrastructure:") for e in vote_errors)
    assert any(e is None for e in vote_errors)


@pytest.mark.asyncio
async def test_ensemble_all_judges_fail_cannot_assess_and_flagged():
    """Every judge fails (infra) -> CANNOT_ASSESS and ensemble flagged as error."""
    rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])

    client_a = _client_raising(litellm.Timeout("a down", model="m", llm_provider="p"))
    client_b = _client_raising(litellm.RateLimitError("b down", model="m", llm_provider="p"))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = _two_judge_grader()
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]

    assert cr.final_verdict == CriterionVerdict.CANNOT_ASSESS
    assert cr.is_error
    assert cr.error is not None
    # Both votes errored; the combined message references both.
    assert all(v.error is not None for v in cr.votes)


# =============================================================================
# Multi-choice error routing
# =============================================================================


@pytest.mark.asyncio
async def test_multi_choice_infrastructure_failure_is_na(mock_llm_config):
    """Multi-choice infra failure -> NA verdict (excluded from scoring), flagged."""
    rubric = Rubric([
        Criterion(
            name="quality",
            requirement="How good is it?",
            weight=5.0,
            scale_type="ordinal",
            options=[
                CriterionOption(label="Bad", value=0.0),
                CriterionOption(label="Ok", value=0.5),
                CriterionOption(label="Great", value=1.0),
            ],
        ),
    ])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(litellm.Timeout("timed out", model="m", llm_provider="p")),
    ):
        # Disable shuffling for deterministic behavior.
        grader = CriterionGrader(llm_config=mock_llm_config, shuffle_options=False)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]
    assert cr.final_multi_choice_verdict is not None
    assert cr.final_multi_choice_verdict.na is True
    assert cr.is_error
    assert cr.error is not None
    assert cr.error.startswith("infrastructure:")

    # NA criterion excluded from scoring under SKIP: only criterion present -> 0.0.
    assert report.score == 0.0


# =============================================================================
# Serialization round-trip
# =============================================================================


@pytest.mark.asyncio
async def test_error_survives_serialization_round_trip(mock_llm_config):
    """``error`` round-trips on the ensemble report and on each JudgeVote.

    Build a real errored ensemble report by grading with a failing ensemble, then
    serialize via ItemResult.to_dict and deserialize via ItemResult.from_dict.
    """
    rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])

    client_a = _client_raising(litellm.Timeout("a down", model="m", llm_provider="p"))
    client_b = _client_raising(litellm.RateLimitError("b down", model="m", llm_provider="p"))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = _two_judge_grader()
        report = await rubric.grade("submission", grader=grader)

    item = DataItem(submission="submission", description="test item")
    item_result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=0.1)

    # Round-trip through the public serialization helpers.
    payload = json.loads(json.dumps(item_result.to_dict()))
    restored = ItemResult.from_dict(payload, item)

    assert restored.report.report is not None
    restored_cr = restored.report.report[0]

    # Ensemble-level error preserved.
    assert restored_cr.error is not None
    assert restored_cr.is_error

    # Per-judge vote errors preserved (both judges failed with infra errors).
    assert len(restored_cr.votes) == 2
    assert all(v.error is not None for v in restored_cr.votes)
    assert all(v.error.startswith("infrastructure:") for v in restored_cr.votes)
