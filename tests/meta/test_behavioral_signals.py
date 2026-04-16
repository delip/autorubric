"""Tests for behavioral signal computation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from autorubric.llm import LLMConfig
from autorubric.meta._signals import compute_reward_variance
from autorubric.rubric import Rubric
from autorubric.types import Criterion, CriterionReport, CriterionVerdict, EvaluationReport

# ============================================================================
# Helpers
# ============================================================================


def _make_criterion(name: str, weight: float = 1.0, requirement: str | None = None) -> Criterion:
    return Criterion(
        name=name,
        weight=weight,
        requirement=requirement or f"Requirement for {name}",
    )


def _make_evaluation_report(
    criteria: list[Criterion],
    verdicts: list[CriterionVerdict],
    score: float = 0.5,
) -> EvaluationReport:
    report = [
        CriterionReport(
            name=c.name,
            weight=c.weight,
            requirement=c.requirement,
            verdict=v,
            reason="test reason",
        )
        for c, v in zip(criteria, verdicts)
    ]
    return EvaluationReport(score=score, report=report)


# ============================================================================
# TestComputeRewardVariance
# ============================================================================


class TestComputeRewardVariance:
    """Tests for compute_reward_variance."""

    @pytest.mark.asyncio
    async def test_empty_items_returns_zeros(self):
        criteria = [_make_criterion("clarity"), _make_criterion("accuracy")]
        rubric = Rubric(criteria)
        llm_config = LLMConfig(model="gpt-4o", temperature=0.0)

        result = await compute_reward_variance(
            rubric, [], llm_config=llm_config
        )

        assert result == {"clarity": 0.0, "accuracy": 0.0}

    @pytest.mark.asyncio
    @patch("autorubric.rubric.Rubric.grade", new_callable=AsyncMock)
    async def test_consistent_verdicts_zero_variance(self, mock_grade):
        criteria = [_make_criterion("clarity"), _make_criterion("accuracy")]
        rubric = Rubric(criteria)
        llm_config = LLMConfig(model="gpt-4o", temperature=0.0)

        mock_grade.return_value = _make_evaluation_report(
            criteria,
            [CriterionVerdict.MET, CriterionVerdict.MET],
        )

        result = await compute_reward_variance(
            rubric,
            ["item1", "item2"],
            llm_config=llm_config,
            n_samples=3,
            seed=42,
        )

        assert result["clarity"] == pytest.approx(0.0)
        assert result["accuracy"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    @patch("autorubric.rubric.Rubric.grade", new_callable=AsyncMock)
    async def test_inconsistent_verdicts_high_variance(self, mock_grade):
        criteria = [_make_criterion("clarity")]
        rubric = Rubric(criteria)
        llm_config = LLMConfig(model="gpt-4o", temperature=0.0)

        # Alternate MET/UNMET across samples to maximise variance
        met_report = _make_evaluation_report(criteria, [CriterionVerdict.MET])
        unmet_report = _make_evaluation_report(criteria, [CriterionVerdict.UNMET])

        # 3 samples x 1 item = 3 calls; alternate verdicts
        mock_grade.side_effect = [met_report, unmet_report, met_report]

        result = await compute_reward_variance(
            rubric,
            ["item1"],
            llm_config=llm_config,
            n_samples=3,
            seed=0,
        )

        # Values: [1.0, 0.0, 1.0] -> variance = 1/3
        assert result["clarity"] == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    @patch("autorubric.rubric.Rubric.grade", new_callable=AsyncMock)
    async def test_single_sample_zero_variance(self, mock_grade):
        criteria = [_make_criterion("clarity")]
        rubric = Rubric(criteria)
        llm_config = LLMConfig(model="gpt-4o", temperature=0.0)

        mock_grade.return_value = _make_evaluation_report(
            criteria, [CriterionVerdict.MET]
        )

        result = await compute_reward_variance(
            rubric,
            ["item1"],
            llm_config=llm_config,
            n_samples=1,
            seed=0,
        )

        # Only 1 value -> cannot compute variance -> 0.0
        assert result["clarity"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    @patch("autorubric.rubric.Rubric.grade", new_callable=AsyncMock)
    async def test_cannot_assess_excluded(self, mock_grade):
        criteria = [_make_criterion("clarity")]
        rubric = Rubric(criteria)
        llm_config = LLMConfig(model="gpt-4o", temperature=0.0)

        met_report = _make_evaluation_report(criteria, [CriterionVerdict.MET])
        ca_report = _make_evaluation_report(criteria, [CriterionVerdict.CANNOT_ASSESS])

        # 3 samples x 1 item; middle sample is CANNOT_ASSESS
        mock_grade.side_effect = [met_report, ca_report, met_report]

        result = await compute_reward_variance(
            rubric,
            ["item1"],
            llm_config=llm_config,
            n_samples=3,
            seed=0,
        )

        # Only [1.0, 1.0] collected (CANNOT_ASSESS excluded) -> variance 0.0
        assert result["clarity"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    @patch("autorubric.graders.CriterionGrader.__init__", return_value=None)
    @patch("autorubric.rubric.Rubric.grade", new_callable=AsyncMock)
    async def test_seed_propagation(self, mock_grade, mock_grader_init):
        criteria = [_make_criterion("clarity")]
        rubric = Rubric(criteria)
        llm_config = LLMConfig(model="gpt-4o", temperature=0.0)

        mock_grade.return_value = _make_evaluation_report(
            criteria, [CriterionVerdict.MET]
        )

        await compute_reward_variance(
            rubric,
            ["item1"],
            llm_config=llm_config,
            n_samples=3,
            seed=10,
        )

        # Verify CriterionGrader was called 3 times with seeds 10, 11, 12
        assert mock_grader_init.call_count == 3
        seeds_used = [
            call.kwargs["llm_config"].seed
            for call in mock_grader_init.call_args_list
        ]
        assert seeds_used == [10, 11, 12]

        # Verify temperature was forced >= 0.5
        temps_used = [
            call.kwargs["llm_config"].temperature
            for call in mock_grader_init.call_args_list
        ]
        assert all(t >= 0.5 for t in temps_used)
