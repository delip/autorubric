"""A binary criterion's few-shot labels may be given as a ``CriterionVerdict`` or its value.

``DataItem.ground_truth`` accepts a binary label as a ``CriterionVerdict`` or as its value
(``"MET"``), as ``resolve_ground_truth`` reads it. ``CriterionGrader`` draws few-shot
examples from ``training_data`` items either way: a label given as its value makes
exactly the prompts the ``CriterionVerdict`` makes, whether the judge is called per
criterion or per item, with or without balanced selection. A label that names no verdict
(``"yes"``) cannot label an example, so its item is left out of the examples, as if it were
not in the training data, and the grader warns once, naming every such label.

The prompts are the ones the grader sends: ``LLMClient.generate`` is replaced by an
``AsyncMock`` that records every user prompt.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from autorubric import (
    Criterion,
    CriterionVerdict,
    FewShotConfig,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMClient
from autorubric.types import CriterionJudgment, LLMCalls, RubricJudgment

MET, UNMET, CANNOT_ASSESS = (
    CriterionVerdict.MET,
    CriterionVerdict.UNMET,
    CriterionVerdict.CANNOT_ASSESS,
)

SOURCES = Criterion(name="sources", weight=1.0, requirement="Cites a peer-reviewed source")
RUBRIC = Rubric([SOURCES])

# (submission, label). Two label classes, so balance_verdicts=True takes the balanced path.
TRAINING = [
    ("Answer A: Smith (2020) in Nature shows the claim holds.", MET),
    ("Answer B: my buddy says so lol.", UNMET),
    ("Answer C: The WHO fact sheet states it.", UNMET),
    ("Answer D: Jones (2019) in Science backs it.", MET),
    ("Answer E: I cannot open the attachment.", CANNOT_ASSESS),
]

MODES: list[LLMCalls] = ["per_criterion", "per_item"]


def _training_data(rows: list[tuple[str, Any]]) -> RubricDataset:
    dataset = RubricDataset(prompt="Is the claim true?", rubric=RUBRIC)
    for submission, label in rows:
        dataset.add_item(submission=submission, description=submission, ground_truth=[label])
    return dataset


def _as_values(rows: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    return [(s, label.value if isinstance(label, CriterionVerdict) else label) for s, label in rows]


def _recording_generate(prompts: list[str]) -> AsyncMock:
    """An ``LLMClient.generate`` that records each user prompt and answers validly."""

    async def generate(*, user_prompt: str, response_format: type, **_: Any) -> GenerateResult:
        prompts.append(user_prompt)
        if issubclass(response_format, RubricJudgment):
            parsed: Any = RubricJudgment.model_validate(
                {
                    "judgments": [
                        {"criterion_id": "c0", "criterion_status": "MET", "explanation": "ok"}
                    ]
                }
            )
        else:
            parsed = CriterionJudgment(criterion_status=MET, explanation="ok")
        return GenerateResult(
            content="{}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=None,
            parsed=parsed,
        )

    return AsyncMock(side_effect=generate)


async def _user_prompts(
    rows: list[tuple[str, Any]], llm_calls: LLMCalls, balance_verdicts: bool
) -> list[str]:
    """The user prompts the grader sends to grade one item with these training rows."""
    prompts: list[str] = []
    with patch.object(LLMClient, "generate", new=_recording_generate(prompts)):
        grader = CriterionGrader(
            judge_model_config=LLMConfig(model="test-model"),
            training_data=_training_data(rows),
            few_shot_config=FewShotConfig(n_examples=4, balance_verdicts=balance_verdicts),
            llm_calls=llm_calls,
            seed=0,
        )
        report = await RUBRIC.grade(
            "Answer F: it is true.", grader=grader, query="Is the claim true?"
        )
    assert report.error is None
    assert prompts
    return prompts


@pytest.mark.asyncio
@pytest.mark.parametrize("balance_verdicts", [True, False])
@pytest.mark.parametrize("llm_calls", MODES)
async def test_labels_given_as_values_make_the_same_prompts(llm_calls, balance_verdicts):
    """``"MET"`` labels the example ``CriterionVerdict.MET`` labels, in every mode."""
    as_verdicts = await _user_prompts(TRAINING, llm_calls, balance_verdicts)
    as_values = await _user_prompts(_as_values(TRAINING), llm_calls, balance_verdicts)

    assert as_values == as_verdicts
    assert all("<example_submission>" in prompt for prompt in as_values)


@pytest.mark.asyncio
@pytest.mark.parametrize("balance_verdicts", [True, False])
@pytest.mark.parametrize("llm_calls", MODES)
async def test_labels_may_mix_verdicts_and_values(llm_calls, balance_verdicts):
    """Some items labelled by verdict and others by value make one consistent draw."""
    mixed = [(s, label.value if i % 2 else label) for i, (s, label) in enumerate(TRAINING)]

    assert await _user_prompts(mixed, llm_calls, balance_verdicts) == await _user_prompts(
        TRAINING, llm_calls, balance_verdicts
    )


@pytest.mark.parametrize("balance_verdicts", [True, False])
def test_examples_drawn_per_criterion_carry_verdicts(balance_verdicts):
    """Every example drawn per criterion holds a ``CriterionVerdict``, never the raw value."""
    grader = CriterionGrader(
        judge_model_config=LLMConfig(model="test-model"),
        training_data=_training_data(_as_values(TRAINING)),
        few_shot_config=FewShotConfig(n_examples=4, balance_verdicts=balance_verdicts),
        seed=0,
    )

    (examples,) = grader._criterion_examples.values()
    assert examples
    assert all(type(example.verdict) is CriterionVerdict for example in examples)


@pytest.mark.asyncio
@pytest.mark.parametrize("unusable", ["yes", "met", "PASS"])
@pytest.mark.parametrize("balance_verdicts", [True, False])
@pytest.mark.parametrize("llm_calls", MODES)
async def test_a_label_naming_no_verdict_leaves_its_item_out(llm_calls, balance_verdicts, unusable):
    """An item whose binary label names no verdict is drawn as if it were not there."""
    stray = ("Answer G: a stray item.", unusable)
    with_stray = [*_as_values(TRAINING[:2]), stray, *_as_values(TRAINING[2:])]

    with pytest.warns(UserWarning, match="names no verdict"):
        drawn = await _user_prompts(with_stray, llm_calls, balance_verdicts)
    assert drawn == await _user_prompts(_as_values(TRAINING), llm_calls, balance_verdicts)


def _build(rows: list[tuple[str, Any]], llm_calls: LLMCalls, *, judges: int = 1) -> None:
    config = FewShotConfig(n_examples=4)
    if judges == 1:
        CriterionGrader(
            judge_model_config=LLMConfig(model="test-model"),
            training_data=_training_data(rows),
            few_shot_config=config,
            llm_calls=llm_calls,
            seed=0,
        )
    else:
        CriterionGrader(
            judges=[JudgeSpec(LLMConfig(model="test-model"), f"judge-{i}") for i in range(judges)],
            training_data=_training_data(rows),
            few_shot_config=config,
            llm_calls=llm_calls,
            seed=0,
        )


@pytest.mark.parametrize("llm_calls", MODES)
def test_unusable_labels_warn_once_naming_them(llm_calls):
    """One warning names the criterion and every label that names no verdict."""
    rows = [*_as_values(TRAINING), ("Answer G.", "yes"), ("Answer H.", "met"), ("Answer I.", "yes")]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(rows, llm_calls)

    (warning,) = [w for w in caught if "names no verdict" in str(w.message)]
    assert warning.category is UserWarning
    message = str(warning.message)
    assert "3 training items" in message
    assert "'sources'" in message
    assert "'met'" in message and "'yes'" in message
    assert warning.filename == __file__


@pytest.mark.parametrize("llm_calls", MODES)
def test_a_panel_warns_once(llm_calls):
    """However many judges draw examples, the grader warns once."""
    rows = [*_as_values(TRAINING), ("Answer G.", "yes")]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(rows, llm_calls, judges=3)

    assert len([w for w in caught if "names no verdict" in str(w.message)]) == 1


@pytest.mark.parametrize("llm_calls", MODES)
def test_usable_labels_do_not_warn(llm_calls):
    """Labels given as verdicts or their values raise no warning."""
    mixed = [(s, label.value if i % 2 else label) for i, (s, label) in enumerate(TRAINING)]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build(mixed, llm_calls)
