"""Roles a decision model cannot fill: meta-rubric evaluation and rubric improvement.

Meta-rubric evaluation and the improvement loop work from generated text: the meta-judge's
explanations (and its ``affected_criteria``), the evaluation judges' explanations of their
verdicts, and the revised rubric the revision model writes. A decision model returns
probabilities, not text, so every entry point that takes a judge configuration for these
roles rejects a decision model, or a ``JudgeSpec`` wrapping one, with ``ValueError`` before
it builds a client or makes any call, as these APIs validate their other arguments.

Nothing here reaches the network, and nothing may be called at all: every test checks that
the rejection comes before any grading, evaluation or generation.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from autorubric import (
    Criterion,
    CriterionVerdict,
    DataItem,
    DecisionModelConfig,
    LLMConfig,
    Rubric,
    RubricDataset,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.meta import (
    ImprovementConfig,
    ImprovementRunner,
    evaluate_rubric_in_context,
    evaluate_rubric_standalone,
    improve_rubric,
    revise_rubric,
    revise_rubric_held_out,
    validate_agreement,
    validate_ground_truth,
    validate_held_out,
)
from autorubric.meta._improve import _get_eval_llm_config

API_KEY = "ts-test-key-never-persisted-e07b"
LLM = LLMConfig(model="test-model")
RUBRIC = Rubric([Criterion(name="light", weight=5.0, requirement="Mentions light")])
NOT_AN_LLM = "must be an LLM, not a decision model"


def dm() -> DecisionModelConfig:
    return DecisionModelConfig(model="jev-latest", api_key=API_KEY)


def validation_data() -> RubricDataset:
    return RubricDataset(
        prompt="Explain photosynthesis.",
        rubric=RUBRIC,
        items=[
            DataItem("Light drives it.", "a", ground_truth=[CriterionVerdict.MET]),
            DataItem("Soil feeds it.", "b", ground_truth=[CriterionVerdict.UNMET]),
        ],
    )


def mixed_judges() -> list[JudgeSpec]:
    return [JudgeSpec(LLM, "llm"), JudgeSpec(dm(), "jev")]


# =============================================================================
# Meta-rubric evaluation
# =============================================================================


class TestMetaRubricEvaluation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("evaluate", ["standalone", "in_context"])
    async def test_a_decision_model_meta_judge_is_rejected_before_grading(self, evaluate):
        call = {
            "standalone": lambda: evaluate_rubric_standalone(RUBRIC, dm()),
            "in_context": lambda: evaluate_rubric_in_context(RUBRIC, "Explain it.", dm()),
        }[evaluate]
        with patch("autorubric.meta._evaluate.CriterionGrader") as grader_cls:
            with pytest.raises(ValueError, match=f"^llm_config {NOT_AN_LLM}"):
                await call()
        grader_cls.assert_not_called()


# =============================================================================
# ImprovementConfig and improve_rubric
# =============================================================================


class TestImprovementConfig:
    def test_a_decision_model_revision_llm_is_rejected(self):
        with pytest.raises(ValueError, match=f"^ImprovementConfig.revision_llm {NOT_AN_LLM}"):
            ImprovementConfig(eval_llm=LLM, revision_llm=dm())

    def test_a_decision_model_eval_judge_is_named(self):
        with pytest.raises(
            ValueError, match=f"^ImprovementConfig.eval_llm judge 'jev' {NOT_AN_LLM}"
        ):
            ImprovementConfig(eval_llm=mixed_judges(), revision_llm=LLM)

    def test_replace_revalidates(self):
        config = ImprovementConfig(eval_llm=LLM, revision_llm=LLM)
        with pytest.raises(ValueError, match="revision_llm"):
            dataclasses.replace(config, revision_llm=dm())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"eval_llm": dm(), "revision_llm": LLM},
            {"eval_llm": [JudgeSpec(dm(), "jev")], "revision_llm": LLM},
            {"eval_llm": LLM, "revision_llm": dm()},
            {"config": ImprovementConfig(eval_llm=LLM, revision_llm=LLM), "revision_llm": dm()},
        ],
        ids=["eval_llm", "eval_llm-judges", "revision_llm", "config-override"],
    )
    async def test_improve_rubric_rejects_before_any_call(self, kwargs):
        evaluate = AsyncMock()
        with patch("autorubric.meta._improve.evaluate_rubric_in_context", evaluate):
            with pytest.raises(ValueError, match=NOT_AN_LLM):
                await improve_rubric(RUBRIC, "Explain it.", show_progress=False, **kwargs)
        evaluate.assert_not_awaited()


# =============================================================================
# ImprovementRunner: the config is re-checked when the run starts
# =============================================================================


class TestImprovementRunner:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", ["meta_rubric", "held_out"])
    @pytest.mark.parametrize("field_name", ["eval_llm", "revision_llm"])
    async def test_a_decision_model_assigned_after_construction_is_rejected_at_run(
        self, strategy, field_name
    ):
        config = ImprovementConfig(
            eval_llm=LLM,
            revision_llm=LLM,
            strategy=strategy,
            validation_data=validation_data(),
            save_artifacts=False,
            show_progress=False,
        )
        # A dataclass field can be reassigned after __post_init__ validated it.
        setattr(config, field_name, mixed_judges() if field_name == "eval_llm" else dm())
        runner = ImprovementRunner(RUBRIC, "Explain it.", config=config)

        evaluate, grade = AsyncMock(), AsyncMock()
        with (
            patch("autorubric.meta._improve.evaluate_rubric_in_context", evaluate),
            patch.object(Rubric, "grade", grade),
            patch("autorubric.meta._improve.CriterionGrader") as grader_cls,
        ):
            with pytest.raises(ValueError, match=f"ImprovementConfig.{field_name}"):
                await runner.run()
        evaluate.assert_not_awaited()
        grade.assert_not_awaited()
        grader_cls.assert_not_called()

    @pytest.mark.parametrize(
        "eval_llm, named",
        [
            (dm(), "ImprovementConfig.eval_llm"),
            (
                [JudgeSpec(dm(), "jev"), JudgeSpec(LLM, "llm")],
                "ImprovementConfig.eval_llm judge 'jev'",
            ),
            (
                [JudgeSpec(LLM, "llm"), JudgeSpec(dm(), "jev")],
                "ImprovementConfig.eval_llm judge 'jev'",
            ),
        ],
        ids=["config", "first-judge", "second-judge"],
    )
    def test_get_eval_llm_config_rejects_a_decision_model(self, eval_llm, named):
        """The guard for an ``eval_llm`` reassigned after the config validated it."""
        with pytest.raises(ValueError, match=f"^{named} {NOT_AN_LLM}"):
            _get_eval_llm_config(eval_llm)


# =============================================================================
# Building blocks: validation and revision
# =============================================================================


class TestBuildingBlocks:
    @pytest.mark.asyncio
    async def test_validate_agreement_rejects_a_decision_model_judge(self):
        with (
            patch.object(Rubric, "grade", AsyncMock()) as grade,
            patch("autorubric.meta._improve.CriterionGrader") as grader_cls,
        ):
            with pytest.raises(ValueError, match=f"^judge 'jev' {NOT_AN_LLM}"):
                await validate_agreement(RUBRIC, ["Light drives it."], mixed_judges())
        grader_cls.assert_not_called()
        grade.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("validate", ["ground_truth", "held_out"])
    async def test_validation_rejects_a_grader_with_a_decision_model_judge(
        self, fake_sdk, validate
    ):
        grader = CriterionGrader(judges=mixed_judges())
        data = validation_data()
        call: dict[str, Any] = {
            "ground_truth": lambda: validate_ground_truth(RUBRIC, data, [1.0, 0.0], grader),
            "held_out": lambda: validate_held_out(RUBRIC, data, grader),
        }
        with patch.object(Rubric, "grade", AsyncMock()) as grade:
            with pytest.raises(ValueError, match=f"^the grader's judge 'jev' {NOT_AN_LLM}"):
                await call[validate]()
        grade.assert_not_awaited()
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("revise", ["meta_rubric", "held_out"])
    async def test_revision_rejects_a_decision_model_revision_llm(self, revise):
        config = ImprovementConfig(eval_llm=LLM, revision_llm=LLM)
        config.revision_llm = dm()  # reassigned after construction
        call = {
            "meta_rubric": lambda: revise_rubric(RUBRIC, "Explain it.", [], "", "", config),
            "held_out": lambda: revise_rubric_held_out(RUBRIC, "Explain it.", "", "", config),
        }[revise]
        with patch("autorubric.meta._improve.LLMClient") as client_cls:
            with pytest.raises(ValueError, match=f"^ImprovementConfig.revision_llm {NOT_AN_LLM}"):
                await call()
        client_cls.assert_not_called()
