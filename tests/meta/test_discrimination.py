"""Tests for evaluate_rubric_discrimination."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric, TokenUsage
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.meta import DiscriminationReport, evaluate_rubric_discrimination
from autorubric.meta._discrimination import _level_descriptor, _spearman
from autorubric.types import CriterionJudgment


def _make_rubric() -> Rubric:
    return Rubric([
        Criterion(name="clarity", weight=2.0, requirement="Output is clear"),
        Criterion(name="depth", weight=2.0, requirement="Output is substantive"),
    ])


def _make_gen_client(submissions: list[str]) -> MagicMock:
    """Mock gen_client that returns submissions in order, one per call."""
    calls = iter(submissions)

    async def gen(*, system_prompt: str, user_prompt: str, return_result: bool = False, **kw):
        text = next(calls)
        return GenerateResult(
            content=text,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=0.0001,
        )

    client = MagicMock()
    client.generate = AsyncMock(side_effect=gen)
    return client


def _make_grader_client_factory(score_per_submission: dict[str, float]):
    """Factory producing a grader LLMClient mock whose verdicts realise the
    requested per-submission rubric score (0.0..1.0).

    Score = fraction of criteria the grader marks MET. With a 2-criterion
    rubric (both weight 2.0), 0.0 = both UNMET, 0.5 = one MET, 1.0 = both MET.
    """

    def _factory(_cfg) -> MagicMock:
        # Track which submission's text we're seeing per call so we can pick verdicts.
        async def grader_generate(
            *, system_prompt: str, user_prompt: str, response_format=None,
            return_result: bool = False, **kw,
        ):
            target_score = next(
                (v for k, v in score_per_submission.items() if k in user_prompt),
                0.0,
            )
            # First call for a submission: criterion 1 (clarity).
            # Second: criterion 2 (depth). Split the score evenly.
            # We track per-submission criterion calls via the user_prompt's
            # criterion text — clarity first then depth in our rubric.
            if "Output is clear" in user_prompt:
                # criterion 1: MET if target_score >= 0.5
                verdict = (
                    CriterionVerdict.MET
                    if target_score >= 0.5
                    else CriterionVerdict.UNMET
                )
            else:
                # criterion 2: MET if target_score >= 1.0
                verdict = (
                    CriterionVerdict.MET
                    if target_score >= 1.0
                    else CriterionVerdict.UNMET
                )
            judgment = CriterionJudgment(
                criterion_status=verdict,
                explanation=f"target={target_score}",
            )
            return GenerateResult(
                content="{}",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                cost=0.0002,
                parsed=judgment,
            )

        client = MagicMock()
        client.generate = AsyncMock(side_effect=grader_generate)
        return client

    return _factory


# ============================================================================
# Helper functions
# ============================================================================


class TestLevelDescriptor:
    def test_endpoints(self):
        assert _level_descriptor(1, 5) == "poor"
        assert _level_descriptor(5, 5) == "excellent"

    def test_middle(self):
        # n_levels=5 maps levels 1..5 → bands 0..4 directly
        assert _level_descriptor(3, 5) == "adequate"

    def test_single_level_falls_back(self):
        assert _level_descriptor(1, 1) == "average quality"

    def test_two_levels_uses_extremes(self):
        assert _level_descriptor(1, 2) == "poor"
        assert _level_descriptor(2, 2) == "excellent"


class TestSpearman:
    def test_perfect_monotonic(self):
        assert _spearman([1, 2, 3, 4, 5], [0.1, 0.3, 0.5, 0.7, 0.9]) == pytest.approx(1.0)

    def test_perfect_inverse(self):
        assert _spearman([1, 2, 3, 4, 5], [0.9, 0.7, 0.5, 0.3, 0.1]) == pytest.approx(-1.0)

    def test_constant_scores_returns_zero(self):
        assert _spearman([1, 2, 3, 4, 5], [0.5, 0.5, 0.5, 0.5, 0.5]) == 0.0

    def test_empty(self):
        assert _spearman([], []) == 0.0


# ============================================================================
# evaluate_rubric_discrimination end-to-end
# ============================================================================


class TestEvaluateDiscrimination:
    @pytest.fixture
    def gen_llm(self) -> LLMConfig:
        return LLMConfig(model="gen-model")

    @pytest.fixture
    def grader_llm(self) -> LLMConfig:
        return LLMConfig(model="grader-model")

    @pytest.mark.asyncio
    async def test_n_levels_too_small_raises(self, gen_llm, grader_llm):
        rubric = _make_rubric()
        with pytest.raises(ValueError, match="n_levels must be at least 2"):
            await evaluate_rubric_discrimination(
                rubric, "task", gen_llm=gen_llm, grader_llm=grader_llm, n_levels=1
            )

    @pytest.mark.asyncio
    async def test_perfect_monotonic_discrimination(self, gen_llm, grader_llm):
        """When generated submissions and grader scores rise together, monotonicity → +1."""
        rubric = _make_rubric()
        submissions = [f"level_{i}" for i in range(1, 6)]
        scores = {
            "level_1": 0.0,
            "level_2": 0.0,
            "level_3": 0.5,
            "level_4": 0.5,
            "level_5": 1.0,
        }

        with (
            patch(
                "autorubric.meta._discrimination.LLMClient",
                return_value=_make_gen_client(submissions),
            ),
            patch(
                "autorubric.graders.criterion_grader.LLMClient",
                side_effect=_make_grader_client_factory(scores),
            ),
        ):
            report = await evaluate_rubric_discrimination(
                rubric, "task prompt", gen_llm=gen_llm, grader_llm=grader_llm, n_levels=5
            )

        assert isinstance(report, DiscriminationReport)
        assert report.n_levels == 5
        assert len(report.per_level_scores) == 5
        assert len(report.submissions) == 5
        # Score range and std should be positive (rubric does discriminate).
        assert report.score_range > 0
        assert report.score_std > 0
        # Monotonicity should be high (≥ 0.9 for our weakly increasing pattern).
        assert report.monotonicity >= 0.9
        # Cost is tracked across both gen and grader.
        assert report.total_cost is not None
        assert report.total_cost > 0

    @pytest.mark.asyncio
    async def test_flat_scores_yield_zero_spread(self, gen_llm, grader_llm):
        """When the grader gives every submission the same score, score_range = 0."""
        rubric = _make_rubric()
        submissions = [f"level_{i}" for i in range(1, 4)]
        # All 0.0 → grader marks both criteria UNMET for every submission.
        scores = {f"level_{i}": 0.0 for i in range(1, 4)}

        with (
            patch(
                "autorubric.meta._discrimination.LLMClient",
                return_value=_make_gen_client(submissions),
            ),
            patch(
                "autorubric.graders.criterion_grader.LLMClient",
                side_effect=_make_grader_client_factory(scores),
            ),
        ):
            report = await evaluate_rubric_discrimination(
                rubric, "task", gen_llm=gen_llm, grader_llm=grader_llm, n_levels=3
            )

        assert report.score_range == pytest.approx(0.0)
        assert report.score_std == pytest.approx(0.0)
        # Spearman is undefined / falls back to 0 when scores are constant.
        assert report.monotonicity == 0.0

    @pytest.mark.asyncio
    async def test_separate_gen_and_grader_llms_are_used(self, gen_llm, grader_llm):
        """gen_llm and grader_llm should produce separate clients."""
        rubric = _make_rubric()
        submissions = ["a", "b"]
        scores = {"a": 0.0, "b": 1.0}

        gen_client = _make_gen_client(submissions)
        grader_factory = _make_grader_client_factory(scores)

        with (
            patch("autorubric.meta._discrimination.LLMClient", return_value=gen_client),
            patch(
                "autorubric.graders.criterion_grader.LLMClient",
                side_effect=grader_factory,
            ),
        ):
            await evaluate_rubric_discrimination(
                rubric, "task", gen_llm=gen_llm, grader_llm=grader_llm, n_levels=2
            )

        # Generator was invoked exactly n_levels times.
        assert gen_client.generate.await_count == 2
