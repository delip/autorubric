"""Few-shot examples show their training item's ground-truth reasons.

``CriterionGrader`` draws each criterion's few-shot examples from ``training_data`` items.
With ``FewShotConfig.include_reason=True`` an example shows its item's written reason for
that criterion (``DataItem.ground_truth_reasons``), when the item has one. Reasons never
change which items are drawn or their order, and without a reason to show a prompt is
exactly what it would be without reasons.

The prompts are the ones the grader sends: ``LLMClient.generate`` is replaced by an
``AsyncMock`` that records every user prompt.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    FewShotConfig,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
)
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMClient
from autorubric.types import CriterionJudgment, MultiChoiceJudgment

MET, UNMET = CriterionVerdict.MET, CriterionVerdict.UNMET

SOURCES = Criterion(name="sources", weight=1.0, requirement="Cites a peer-reviewed source")
TONE = Criterion(
    name="tone",
    weight=1.0,
    requirement="Which tone best describes the answer?",
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Casual", value=0.0),
    ],
)

A = "Answer A: Smith (2020) in Nature shows the claim holds."
B = "Answer B: my buddy says so lol."
C = "Answer C: The WHO fact sheet states it."
D = "Answer D: hey there, Jones (2019) in Science backs it."

# (submission, ground truth, reasons). Each criterion has two label classes among these
# items, so balance_verdicts=True takes the balanced selection path for both.
TRAINING = [
    (A, [MET, "Formal"], ["Nature is peer-reviewed.", "No slang or contractions."]),
    (B, [UNMET, "Casual"], ["A friend is not a source.", None]),
    (C, [UNMET, "Formal"], None),
    (D, [MET, "Casual"], [None, "Opens with 'hey there'."]),
]

_EXAMPLE = re.compile(r"<example_(\d+)>\n(.*?)\n</example_\1>", re.DOTALL)
_EXAMPLE_SUBMISSION = re.compile(r"<example_submission>(.*?)</example_submission>", re.DOTALL)
_REASON = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)
_REASON_LINE = re.compile(r"<reason>.*?</reason>\n", re.DOTALL)


def _training_data(*, with_reasons: bool = True) -> RubricDataset:
    dataset = RubricDataset(prompt="Is the claim true?", rubric=Rubric([SOURCES, TONE]))
    for submission, ground_truth, reasons in TRAINING:
        dataset.add_item(
            submission=submission,
            description=submission,
            ground_truth=ground_truth,
            ground_truth_reasons=reasons if with_reasons else None,
        )
    return dataset


def _recording_generate(prompts: list[str]) -> AsyncMock:
    """An ``LLMClient.generate`` that records each user prompt and answers validly."""

    async def generate(*, user_prompt: str, response_format: type, **_: Any) -> GenerateResult:
        prompts.append(user_prompt)
        if issubclass(response_format, MultiChoiceJudgment):
            parsed: Any = MultiChoiceJudgment(selected_option=1, explanation="ok")
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
    training_data: RubricDataset, few_shot_config: FewShotConfig
) -> tuple[str, str]:
    """The user prompts of the binary and the multi-choice criterion for one graded item."""
    prompts: list[str] = []
    with patch.object(LLMClient, "generate", new=_recording_generate(prompts)):
        grader = CriterionGrader(
            judge_model_config=LLMConfig(model="test-model"),
            training_data=training_data,
            few_shot_config=few_shot_config,
            seed=0,
        )
        await Rubric([SOURCES, TONE]).grade(
            "Answer E: it is true.", grader=grader, query="Is the claim true?"
        )
    (binary,) = [p for p in prompts if "<criterion>" in p]
    (multi_choice,) = [p for p in prompts if "<question>" in p]
    return binary, multi_choice


def _examples(user_prompt: str) -> dict[str, str | None]:
    """Each few-shot example's submission, in prompt order, with the reason it shows."""
    examples: dict[str, str | None] = {}
    for _number, body in _EXAMPLE.findall(user_prompt):
        submission = _EXAMPLE_SUBMISSION.search(body)
        assert submission is not None
        reason = _REASON.search(body)
        examples[submission.group(1)] = reason.group(1) if reason else None
    return examples


@pytest.mark.asyncio
@pytest.mark.parametrize("balance_verdicts", [True, False])
async def test_binary_examples_show_their_items_reasons(balance_verdicts):
    """Each binary example shows its item's reason for the criterion, or none without one."""
    config = FewShotConfig(n_examples=4, balance_verdicts=balance_verdicts, include_reason=True)
    binary, _ = await _user_prompts(_training_data(), config)

    assert _examples(binary) == {
        A: "Nature is peer-reviewed.",
        B: "A friend is not a source.",
        C: None,
        D: None,
    }
    assert "<verdict>MET</verdict>\n<reason>Nature is peer-reviewed.</reason>" in binary


@pytest.mark.asyncio
@pytest.mark.parametrize("balance_verdicts", [True, False])
async def test_multi_choice_examples_show_their_items_reasons(balance_verdicts):
    """Each multi-choice example shows its item's reason for the criterion, or none."""
    config = FewShotConfig(n_examples=4, balance_verdicts=balance_verdicts, include_reason=True)
    _, multi_choice = await _user_prompts(_training_data(), config)

    assert _examples(multi_choice) == {
        A: "No slang or contractions.",
        B: None,
        C: None,
        D: "Opens with 'hey there'.",
    }
    assert (
        "<selected_label>Casual</selected_label>\n<reason>Opens with 'hey there'.</reason>"
        in multi_choice
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("with_reasons", "include_reason"),
    [
        (True, False),  # reasons in the data, not asked for
        (False, True),  # asked for, but the data has none
    ],
)
async def test_prompts_are_unchanged_without_reasons_to_show(with_reasons, include_reason):
    """No reason to show leaves every prompt byte-identical to one without reasons."""
    config = FewShotConfig(n_examples=2, include_reason=include_reason)
    baseline = await _user_prompts(
        _training_data(with_reasons=False), FewShotConfig(n_examples=2, include_reason=False)
    )

    prompts = await _user_prompts(_training_data(with_reasons=with_reasons), config)

    assert all(_examples(p) for p in baseline)
    assert prompts == baseline


@pytest.mark.asyncio
@pytest.mark.parametrize("balance_verdicts", [True, False])
async def test_reasons_do_not_change_which_examples_are_drawn(balance_verdicts):
    """With the same seed, the same examples come in the same order; reasons only ride along."""
    config = FewShotConfig(n_examples=2, balance_verdicts=balance_verdicts, include_reason=True)
    without = await _user_prompts(_training_data(with_reasons=False), config)

    with_reasons = await _user_prompts(_training_data(), config)

    for prompt, baseline in zip(with_reasons, without, strict=True):
        assert list(_examples(prompt)) == list(_examples(baseline))
        assert _REASON_LINE.sub("", prompt) == baseline
