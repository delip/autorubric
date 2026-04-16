"""Tests for behavioral signal integration in the improvement loop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric.llm import LLMConfig
from autorubric.meta._improve import (
    ImprovementConfig,
    IssueDetail,
    IterationResult,
    _serialize_iteration,
    behavioral_plateau_converged,
    format_issues_for_prompt,
    revise_rubric,
)
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    TokenUsage,
)


def _make_criterion(name, weight, requirement):
    return Criterion(name=name, weight=weight, requirement=requirement)


def _make_iteration_result(
    iteration=0,
    quality_score=0.7,
    agreement=0.8,
    issues=None,
    accepted=True,
    evidence=None,
):
    return IterationResult(
        iteration=iteration,
        rubric=Rubric([_make_criterion("c1", 10, "test requirement")]),
        quality_score=quality_score,
        agreement=agreement,
        per_criterion_agreement=None,
        issues=issues or [],
        issues_fixed=[],
        issues_introduced=[],
        accepted=accepted,
        rejection_reason=None,
        quality_report=None,
        token_usage=TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        ),
        completion_cost=0.01,
        evidence=evidence,
    )


# ============================================================================
# IssueDetail.signal_source
# ============================================================================


class TestIssueDetailSignalSource:

    def test_default_is_text(self):
        issue = IssueDetail(
            criterion_name="test",
            requirement="test req",
            weight=8,
            is_antipattern=False,
            feedback="test feedback",
        )
        assert issue.signal_source == "text"

    def test_custom_signal_source(self):
        issue = IssueDetail(
            criterion_name="test",
            requirement="test req",
            weight=-8,
            is_antipattern=True,
            feedback="test feedback",
            signal_source="variance, discrimination",
        )
        assert issue.signal_source == "variance, discrimination"


# ============================================================================
# IterationResult.evidence
# ============================================================================


class TestIterationResultEvidence:

    def test_default_none(self):
        result = _make_iteration_result()
        assert result.evidence is None

    def test_evidence_stored(self):
        ev = {"variance": {"c1": 0.1, "c2": 0.02}}
        result = _make_iteration_result(evidence=ev)
        assert result.evidence == ev
        assert result.evidence["variance"]["c1"] == 0.1


# ============================================================================
# format_issues_for_prompt with signal source
# ============================================================================


class TestFormatIssuesWithSource:

    def test_text_source_no_tag(self):
        issue = IssueDetail(
            criterion_name="clarity",
            requirement="be clear",
            weight=8,
            is_antipattern=False,
            feedback="Not clear enough",
        )
        output = format_issues_for_prompt([issue])
        assert "[source:" not in output
        assert "[QUALITY GAP]" in output

    def test_behavioral_source_has_tag(self):
        issue = IssueDetail(
            criterion_name="clarity",
            requirement="be clear",
            weight=8,
            is_antipattern=False,
            feedback="High variance",
            signal_source="variance",
        )
        output = format_issues_for_prompt([issue])
        assert "[source: variance]" in output

    def test_antipattern_with_source(self):
        issue = IssueDetail(
            criterion_name="hackable",
            requirement="is hackable",
            weight=-8,
            is_antipattern=True,
            feedback="Low discrimination",
            signal_source="discrimination",
        )
        output = format_issues_for_prompt([issue])
        assert "[ANTI-PATTERN DETECTED]" in output
        assert "[source: discrimination]" in output


# ============================================================================
# behavioral_plateau_converged
# ============================================================================


class TestBehavioralPlateauConverged:

    def test_insufficient_history(self):
        current = _make_iteration_result(iteration=0, quality_score=0.8)
        assert behavioral_plateau_converged(current, [current]) is None

    def test_quality_plateau_no_evidence(self):
        history = [
            _make_iteration_result(iteration=i, quality_score=0.85)
            for i in range(4)
        ]
        result = behavioral_plateau_converged(
            history[-1], history, patience=2
        )
        assert result == "behavioral_plateau"

    def test_quality_improving_returns_none(self):
        history = [
            _make_iteration_result(iteration=0, quality_score=0.70),
            _make_iteration_result(iteration=1, quality_score=0.75),
            _make_iteration_result(iteration=2, quality_score=0.82),
            _make_iteration_result(iteration=3, quality_score=0.90),
        ]
        result = behavioral_plateau_converged(
            history[-1], history, patience=2
        )
        assert result is None

    def test_evidence_plateau_converges(self):
        ev = {"variance": {"c1": 0.05}}
        history = [
            _make_iteration_result(iteration=i, quality_score=0.85, evidence=ev)
            for i in range(4)
        ]
        result = behavioral_plateau_converged(
            history[-1], history, patience=2
        )
        assert result == "behavioral_plateau"

    def test_evidence_changing_prevents_convergence(self):
        history = [
            _make_iteration_result(
                iteration=0, quality_score=0.85,
                evidence={"variance": {"c1": 0.20}}
            ),
            _make_iteration_result(
                iteration=1, quality_score=0.85,
                evidence={"variance": {"c1": 0.15}}
            ),
            _make_iteration_result(
                iteration=2, quality_score=0.85,
                evidence={"variance": {"c1": 0.05}}
            ),
            _make_iteration_result(
                iteration=3, quality_score=0.85,
                evidence={"variance": {"c1": 0.01}}
            ),
        ]
        result = behavioral_plateau_converged(
            history[-1], history, patience=2, variance_threshold=0.01
        )
        assert result is None


# ============================================================================
# ImprovementConfig new fields
# ============================================================================


class TestImprovementConfigNewFields:

    def test_evidence_fn_default_none(self):
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )
        assert config.evidence_fn is None

    def test_behavioral_signal_frequency_default(self):
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )
        assert config.behavioral_signal_frequency == "first_and_last"


# ============================================================================
# _serialize_iteration with evidence
# ============================================================================


class TestSerializeIterationEvidence:

    def test_no_evidence_key_absent(self):
        result = _make_iteration_result()
        serialized = _serialize_iteration(result)
        assert "evidence" not in serialized

    def test_evidence_key_present(self):
        ev = {"variance": {"c1": 0.1}}
        result = _make_iteration_result(evidence=ev)
        serialized = _serialize_iteration(result)
        assert serialized["evidence"] == ev


# ============================================================================
# revise_rubric with evidence
# ============================================================================


class TestReviseRubricEvidence:

    @pytest.mark.asyncio
    async def test_evidence_none_no_section(self):
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )
        rubric = Rubric([_make_criterion("c1", 10, "test")])
        captured = {}

        mock_result = MagicMock()
        mock_result.content = '[{"weight": 10, "name": "c1", "requirement": "test"}]'
        mock_result.cost = 0.01

        with patch("autorubric.meta._improve.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            _, _ = await revise_rubric(
                rubric, "task", [], "", "", config,
                evidence=None, _capture=captured,
            )

        assert "Behavioral Signals" not in captured.get("user_prompt", "")

    @pytest.mark.asyncio
    async def test_evidence_in_prompt(self):
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )
        rubric = Rubric([_make_criterion("c1", 10, "test")])
        captured = {}

        mock_result = MagicMock()
        mock_result.content = '[{"weight": 10, "name": "c1", "requirement": "test"}]'
        mock_result.cost = 0.01

        with patch("autorubric.meta._improve.LLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            _, _ = await revise_rubric(
                rubric, "task", [], "", "", config,
                evidence={"variance": {"c1": 0.15}}, _capture=captured,
            )

        assert "Behavioral Signals" in captured["user_prompt"]
        assert "0.15" in captured["user_prompt"]
