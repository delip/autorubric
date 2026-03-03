"""Tests for structured output types used in LLM responses."""

import pytest
from pydantic import ValidationError

from autorubric.types import (
    CriterionJudgment,
    CriterionVerdict,
)


class TestCriterionJudgment:
    """Tests for CriterionJudgment structured output type."""

    def test_valid_met_judgment(self):
        """CriterionJudgment accepts valid MET status."""
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="The requirement is satisfied.",
        )
        assert judgment.criterion_status == CriterionVerdict.MET
        assert judgment.explanation == "The requirement is satisfied."
        assert judgment.reasoning is None

    def test_valid_unmet_judgment(self):
        """CriterionJudgment accepts valid UNMET status."""
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.UNMET,
            explanation="The requirement is not satisfied.",
        )
        assert judgment.criterion_status == CriterionVerdict.UNMET
        assert judgment.explanation == "The requirement is not satisfied."

    def test_with_reasoning(self):
        """CriterionJudgment can include optional reasoning."""
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="Requirement met.",
            reasoning="I thought about this carefully and determined...",
        )
        assert judgment.reasoning == "I thought about this carefully and determined..."

    def test_invalid_status_rejected(self):
        """CriterionJudgment rejects invalid criterion_status values."""
        with pytest.raises(ValidationError) as exc_info:
            CriterionJudgment(
                criterion_status="PARTIAL",  # Invalid
                explanation="Some explanation",
            )
        assert "criterion_status" in str(exc_info.value)

    def test_missing_required_fields(self):
        """CriterionJudgment requires criterion_status and explanation."""
        with pytest.raises(ValidationError):
            CriterionJudgment(criterion_status=CriterionVerdict.MET)

        with pytest.raises(ValidationError):
            CriterionJudgment(explanation="Some explanation")

    def test_is_frozen(self):
        """CriterionJudgment instances are immutable."""
        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="Test",
        )
        with pytest.raises(ValidationError):
            judgment.criterion_status = CriterionVerdict.UNMET
