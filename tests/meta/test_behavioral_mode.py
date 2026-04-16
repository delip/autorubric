"""Tests for behavioral meta-evaluation (evidence parameter and evidence_cited)."""

from unittest.mock import patch

import pytest

from autorubric.llm import LLMConfig
from autorubric.meta._evaluate import (
    MetaCriterionJudgment,
    _format_evidence_section,
    evaluate_rubric_in_context,
    evaluate_rubric_standalone,
)
from autorubric.rubric import Rubric
from autorubric.types import (
    CriterionVerdict,
    EnsembleEvaluationReport,
    TokenUsage,
)


def _make_rubric():
    return Rubric.from_dict([
        {"weight": 10, "name": "clarity", "requirement": "Uses clear language"},
        {"weight": -5, "name": "errors", "requirement": "Contains factual errors"},
    ])


def _make_mock_report():
    return EnsembleEvaluationReport(
        score=0.8,
        raw_score=0.8,
        llm_raw_score=0.8,
        report=None,
        judge_scores={},
        mean_agreement=1.0,
        cannot_assess_count=0,
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
        completion_cost=0.01,
        error=None,
    )


# ============================================================================
# MetaCriterionJudgment.evidence_cited
# ============================================================================


class TestEvidenceCitedField:
    """Verify evidence_cited field on MetaCriterionJudgment."""

    def test_default_empty_list(self):
        j = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="test",
        )
        assert j.evidence_cited == []

    def test_set_values(self):
        j = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="high variance",
            evidence_cited=["variance", "discrimination"],
        )
        assert j.evidence_cited == ["variance", "discrimination"]

    def test_in_json_schema(self):
        schema = MetaCriterionJudgment.model_json_schema()
        assert "evidence_cited" in schema["properties"]

    def test_round_trip(self):
        j = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.UNMET,
            explanation="low agreement",
            evidence_cited=["agreement"],
            affected_criteria=[1, 3],
        )
        data = j.model_dump()
        restored = MetaCriterionJudgment.model_validate(data)
        assert restored.evidence_cited == ["agreement"]
        assert restored.affected_criteria == [1, 3]


# ============================================================================
# Evidence section formatting
# ============================================================================


class TestFormatEvidenceSection:
    """Verify the evidence section format."""

    def test_contains_header(self):
        section = _format_evidence_section({"variance": {"c1": 0.1}})
        assert "## Supplementary Behavioral Signals" in section

    def test_contains_json(self):
        evidence = {"variance": {"c1": 0.15, "c2": 0.02}}
        section = _format_evidence_section(evidence)
        assert '"variance"' in section
        assert "0.15" in section

    def test_contains_instruction(self):
        section = _format_evidence_section({"agreement": {"c1": 0.9}})
        assert "evidence_cited" in section


# ============================================================================
# evaluate_rubric_standalone with evidence
# ============================================================================


class TestStandaloneEvidence:
    """Verify evidence handling in evaluate_rubric_standalone."""

    @pytest.mark.asyncio
    async def test_no_evidence_no_section(self):
        rubric = _make_rubric()
        captured_submission = {}

        async def capture_grade(to_grade, grader, **kwargs):
            captured_submission["text"] = to_grade
            return _make_mock_report()

        with patch.object(Rubric, "grade", side_effect=capture_grade):
            await evaluate_rubric_standalone(
                rubric,
                LLMConfig(model="test"),
                evidence=None,
            )

        assert "Supplementary" not in captured_submission["text"]

    @pytest.mark.asyncio
    async def test_evidence_included_in_submission(self):
        rubric = _make_rubric()
        captured_submission = {}

        async def capture_grade(to_grade, grader, **kwargs):
            captured_submission["text"] = to_grade
            return _make_mock_report()

        evidence = {"variance": {"clarity": 0.2, "errors": 0.01}}

        with patch.object(Rubric, "grade", side_effect=capture_grade):
            await evaluate_rubric_standalone(
                rubric,
                LLMConfig(model="test"),
                evidence=evidence,
            )

        text = captured_submission["text"]
        assert "Supplementary Behavioral Signals" in text
        assert '"variance"' in text
        assert "0.2" in text


# ============================================================================
# evaluate_rubric_in_context with evidence
# ============================================================================


class TestInContextEvidence:
    """Verify evidence handling in evaluate_rubric_in_context."""

    @pytest.mark.asyncio
    async def test_no_evidence_no_section(self):
        rubric = _make_rubric()
        captured_submission = {}

        async def capture_grade(to_grade, grader, **kwargs):
            captured_submission["text"] = to_grade
            return _make_mock_report()

        with patch.object(Rubric, "grade", side_effect=capture_grade):
            await evaluate_rubric_in_context(
                rubric,
                "Write a summary",
                LLMConfig(model="test"),
                evidence=None,
            )

        assert "Supplementary" not in captured_submission["text"]

    @pytest.mark.asyncio
    async def test_evidence_included_in_submission(self):
        rubric = _make_rubric()
        captured_submission = {}

        async def capture_grade(to_grade, grader, **kwargs):
            captured_submission["text"] = to_grade
            return _make_mock_report()

        evidence = {"agreement": {"clarity": 0.95}}

        with patch.object(Rubric, "grade", side_effect=capture_grade):
            await evaluate_rubric_in_context(
                rubric,
                "Write a summary",
                LLMConfig(model="test"),
                evidence=evidence,
            )

        text = captured_submission["text"]
        assert "Supplementary Behavioral Signals" in text
        assert '"agreement"' in text
        assert "0.95" in text

    @pytest.mark.asyncio
    async def test_submission_still_has_task_prompt(self):
        rubric = _make_rubric()
        captured_submission = {}

        async def capture_grade(to_grade, grader, **kwargs):
            captured_submission["text"] = to_grade
            return _make_mock_report()

        with patch.object(Rubric, "grade", side_effect=capture_grade):
            await evaluate_rubric_in_context(
                rubric,
                "Write a summary of the article",
                LLMConfig(model="test"),
                evidence={"variance": {"c1": 0.1}},
            )

        text = captured_submission["text"]
        assert "Write a summary of the article" in text
        assert "Supplementary" in text
