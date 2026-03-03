"""Shared abstractions for grader implementations."""

from abc import ABC, abstractmethod
from typing import Any

from autorubric.types import (
    Criterion,
    EnsembleEvaluationReport,
    EvaluationReport,
    LengthPenalty,
    ToGradeInput,
)
from autorubric.utils import compute_length_penalty, normalize_to_grade_input


class Grader(ABC):
    """Base class for LLM-backed grading implementations.

    All graders require an LLMConfig for the LLM client. Subclasses must
    implement judge() and aggregate() methods.

    Args:
        length_penalty: Optional configuration for penalizing overly long outputs.
            When provided, a penalty based on the token/word count is subtracted
            from the final score.
        normalize: If True (default), scores are normalized to 0-1. If False, raw
            weighted sums are returned, which is useful for RL training scenarios.
    """

    def __init__(
        self,
        *,
        length_penalty: LengthPenalty | None = None,
        normalize: bool = True,
    ):
        self.length_penalty: LengthPenalty | None = length_penalty
        self.normalize: bool = normalize

    @abstractmethod
    async def judge(
        self,
        to_grade: str,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> Any:
        """Collect raw judge results for the provided submission.

        Args:
            to_grade: The text to evaluate.
            rubric: List of criteria to evaluate against.
            query: Optional input/query that prompted the response.
            reference_submission: Optional exemplar response for grading context.

        Returns:
            Raw judge results (format depends on implementation).
        """
        pass

    @abstractmethod
    async def aggregate(
        self, judge_results: Any, *, normalize: bool = True
    ) -> EvaluationReport:
        """Transform judge results into an EvaluationReport.

        Args:
            judge_results: Raw results from judge().
            normalize: If True, normalize score to 0-1. If False, return raw weighted sum.

        Returns:
            EvaluationReport with score and optional per-criterion breakdown.
        """
        pass

    async def grade(
        self,
        to_grade: ToGradeInput,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> EvaluationReport:
        """Grade the submission against the rubric.

        This is the main entry point for the grader.

        Args:
            to_grade: The text to evaluate. Can be either:
                - A string (optionally with <thinking>/<output> markers)
                - A dict with 'thinking' and 'output' keys
            rubric: List of criteria to evaluate against.
            query: Optional input/query that prompted the response.
            reference_submission: Optional exemplar response for grading context.

        Returns:
            EvaluationReport with score and optional per-criterion breakdown.
            If normalize=True (default), score is 0-1. If normalize=False, score is raw
            weighted sum. If length_penalty was configured, the penalty is subtracted from
            the score. The raw_score field contains the unnormalized weighted sum before
            length penalty.
        """
        # Convert to_grade to string for judge() call (maintains compatibility)
        if isinstance(to_grade, str):
            to_grade_str = to_grade
        else:
            # Dict format - reconstruct string with markers for judge()
            thinking = to_grade.get("thinking", "")
            output = to_grade.get("output", "")
            parts = []
            if thinking:
                parts.append(f"<thinking>{thinking}</thinking>")
            if output:
                parts.append(f"<output>{output}</output>")
            to_grade_str = "\n".join(parts) if parts else ""

        # Call judge with string format (maintains compatibility)
        judge_results = await self.judge(to_grade_str, rubric, query, reference_submission)
        report = await self.aggregate(judge_results, normalize=self.normalize)

        if self.length_penalty is not None:
            # Normalize to_grade to dict format for penalty calculation
            to_grade_normalized = normalize_to_grade_input(to_grade)

            # Compute penalty
            penalty = compute_length_penalty(to_grade_normalized, self.length_penalty)

            # Apply penalty (penalty is always non-negative, so we subtract)
            adjusted_score = report.score - penalty
            if self.normalize:
                adjusted_score = max(0.0, adjusted_score)

            # Return the same report type with adjusted score
            if isinstance(report, EnsembleEvaluationReport):
                return EnsembleEvaluationReport(
                    score=adjusted_score,
                    raw_score=report.raw_score,
                    llm_raw_score=report.llm_raw_score,
                    report=report.report,
                    judge_scores=report.judge_scores,
                    mean_agreement=report.mean_agreement,
                    cannot_assess_count=report.cannot_assess_count,
                    token_usage=report.token_usage,
                    completion_cost=report.completion_cost,
                    error=report.error,
                )
            else:
                return EvaluationReport(
                    score=adjusted_score,
                    raw_score=report.raw_score,
                    llm_raw_score=report.llm_raw_score,
                    report=report.report,
                    cannot_assess_count=report.cannot_assess_count,
                    error=report.error,
                    token_usage=report.token_usage,
                    completion_cost=report.completion_cost,
                )

        return report
