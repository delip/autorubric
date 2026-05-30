"""Tests for per-judge / per-criterion decorrelation of few-shot example selection.

Option *shuffling* already de-correlates per ``(seed, item, criterion, judge)`` via
``_derive_shuffle_rng``. Few-shot example *selection*, however, historically used a flat
``random.Random(config.seed)`` with no judge/criterion/item axis, so every ensemble judge
saw the *same* few-shot examples in the *same* order (undercutting the ensemble-independence
assumption behind the inter-judge agreement metrics) and ordering correlated across criteria.

Selection keys the RNG on ``(few_shot_seed, criterion_idx, judge_id)`` (reusing the
same ``_derive_shuffle_rng`` helper, with a constant ``FEW_SHOT_DOMAIN`` in the item-key
slot — few-shot examples are a fixed property of criterion+judge, not item-specific). The
example dicts are now keyed by ``(criterion_idx, judge_id)`` tuples. Selection stays fully
reproducible (deterministic in the master seed).
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    FewShotConfig,
    LLMConfig,
    Rubric,
    TokenUsage,
)
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.types import CriterionJudgment

# Tag emitted for each few-shot example submission (prompts.py binary :235 / multi-choice :519).
_EXAMPLE_PATTERN = re.compile(r"<example_submission>(.*?)</example_submission>", re.DOTALL)


def _two_judges() -> list[JudgeSpec]:
    return [
        JudgeSpec(LLMConfig(model="judge-a-model"), "judge_a"),
        JudgeSpec(LLMConfig(model="judge-b-model"), "judge_b"),
    ]


def _binary_dataset() -> RubricDataset:
    """12 binary training items, all MET, single criterion."""
    rubric = Rubric([Criterion(weight=1.0, requirement="Crit 0")])
    ds = RubricDataset(name="fs-binary", prompt="prompt", rubric=rubric)
    for i in range(12):
        ds.add_item(
            submission=f"binary-submission-{i}",
            description=f"item {i}",
            ground_truth=[CriterionVerdict.MET],
        )
    return ds


def _two_criterion_binary_dataset() -> RubricDataset:
    """12 binary training items, two criteria, identical GT (both MET) on both."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="Crit 0"),
            Criterion(weight=1.0, requirement="Crit 1"),
        ]
    )
    ds = RubricDataset(name="fs-binary-2crit", prompt="prompt", rubric=rubric)
    for i in range(12):
        ds.add_item(
            submission=f"binary2-submission-{i}",
            description=f"item {i}",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
        )
    return ds


def _multi_choice_dataset() -> RubricDataset:
    """12 multi-choice training items, all the same option label, single ordinal criterion."""
    rubric = Rubric(
        [
            Criterion(
                name="quality",
                requirement="How good is it?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="Bad", value=0.0),
                    CriterionOption(label="Ok", value=0.33),
                    CriterionOption(label="Good", value=0.66),
                    CriterionOption(label="Great", value=1.0),
                ],
            )
        ]
    )
    ds = RubricDataset(name="fs-mc", prompt="prompt", rubric=rubric)
    for i in range(12):
        ds.add_item(
            submission=f"mc-submission-{i}",
            description=f"item {i}",
            ground_truth=["Good"],
        )
    return ds


# =============================================================================
# 1-4: in-memory decorrelation / determinism of the precomputed example pools
# =============================================================================


def test_binary_examples_decorrelated_across_judges():
    """Two judges must select different 3-of-12 binary example subsets."""
    ds = _binary_dataset()
    g = CriterionGrader(
        judges=_two_judges(),
        training_data=ds,
        few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=False),
        seed=42,
    )
    subs_a = {e.submission for e in g._criterion_examples[(0, "judge_a")]}
    subs_b = {e.submission for e in g._criterion_examples[(0, "judge_b")]}
    assert len(subs_a) == 3
    assert len(subs_b) == 3
    assert subs_a != subs_b


def test_multi_choice_examples_decorrelated_across_judges():
    """Two judges must select different 3-of-12 multi-choice example subsets."""
    ds = _multi_choice_dataset()
    g = CriterionGrader(
        judges=_two_judges(),
        training_data=ds,
        few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=False),
        seed=42,
    )
    subs_a = {s for s, _i, _r in g._multi_choice_examples[(0, "judge_a")]}
    subs_b = {s for s, _i, _r in g._multi_choice_examples[(0, "judge_b")]}
    assert len(subs_a) == 3
    assert len(subs_b) == 3
    assert subs_a != subs_b


def test_determinism_same_seed():
    """Same dataset + same seed => identical per-judge example pools."""
    ds = _binary_dataset()
    g1 = CriterionGrader(
        judges=_two_judges(),
        training_data=ds,
        few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=False),
        seed=7,
    )
    g2 = CriterionGrader(
        judges=_two_judges(),
        training_data=ds,
        few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=False),
        seed=7,
    )
    for jid in ("judge_a", "judge_b"):
        assert g1._criterion_examples[(0, jid)] == g2._criterion_examples[(0, jid)]


def test_determinism_different_seed():
    """Different seeds => different example pools for the same judge."""
    ds = _binary_dataset()
    g1 = CriterionGrader(
        judges=_two_judges(),
        training_data=ds,
        few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=False),
        seed=1,
    )
    g2 = CriterionGrader(
        judges=_two_judges(),
        training_data=ds,
        few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=False),
        seed=2,
    )
    assert g1._criterion_examples[(0, "judge_a")] != g2._criterion_examples[(0, "judge_a")]


def test_cross_criterion_decorrelation_same_judge():
    """For one judge, two criteria with identical data must get different example subsets."""
    ds = _two_criterion_binary_dataset()
    g = CriterionGrader(
        judges=_two_judges(),
        training_data=ds,
        few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=False),
        seed=42,
    )
    subs_c0 = {e.submission for e in g._criterion_examples[(0, "judge_a")]}
    subs_c1 = {e.submission for e in g._criterion_examples[(1, "judge_a")]}
    assert len(subs_c0) == 3
    assert len(subs_c1) == 3
    assert subs_c0 != subs_c1


# =============================================================================
# 5: end-to-end pin — each judge's prompt carries a distinct example set
# =============================================================================


@pytest.mark.asyncio
async def test_few_shot_examples_differ_per_judge_in_prompt():
    """THE PIN: each judge's actual user_prompt must embed a distinct set of example
    submissions when grading the same item."""
    ds = _binary_dataset()
    rubric = Rubric([Criterion(weight=1.0, requirement="Crit 0")])

    captured: dict[str, list[str]] = {}

    def make_client(config: LLMConfig) -> MagicMock:
        model = config.model

        async def mock_generate(
            system_prompt,
            user_prompt,
            response_format=None,
            return_result=False,
            **kwargs,
        ):
            captured.setdefault(model, []).append(user_prompt)
            return GenerateResult(
                content="{}",
                thinking=None,
                raw_response=None,
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                cost=0.0,
                parsed=CriterionJudgment(criterion_status=CriterionVerdict.MET, explanation="ok"),
            )

        client = MagicMock()
        client.generate = AsyncMock(side_effect=mock_generate)
        return client

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=make_client,
    ):
        grader = CriterionGrader(
            judges=_two_judges(),
            training_data=ds,
            few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=False),
            seed=42,
        )
        await rubric.grade("X", grader=grader)

    prompt_a = captured["judge-a-model"][0]
    prompt_b = captured["judge-b-model"][0]
    set_a = set(_EXAMPLE_PATTERN.findall(prompt_a))
    set_b = set(_EXAMPLE_PATTERN.findall(prompt_b))
    assert set_a
    assert set_b
    assert set_a != set_b
