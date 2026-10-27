"""Few-shot examples are shown only for a graded criterion they describe.

A training item's ground-truth label for criterion ``i`` answers the question its rubric's
criterion ``i`` poses. ``CriterionGrader`` draws its few-shot examples from the training
data's rubric and shows them for the criterion at the same index of the rubric being graded.
That is right only when the graded criterion poses the same question as the training one
(``_posed_question``): the same kind and requirement; for a binary criterion the same
criterion type (a negative weight or not); for a multi-choice criterion the same options,
labels and NA flags, in the same order, among the options the judge is shown (with
``auto_na_option``, the NA option it adds included). A graded criterion that poses another
question gets no examples, and a training item whose own rubric poses another question at an
index lends no label for it. Weight magnitude, name, option values, ``scale_type`` and
``aggregation`` are not shown to the judge and do not change what a label means, so they do
not matter.

The prompts are the ones the grader sends: ``LLMClient.generate`` is replaced by an
``AsyncMock`` that records every user prompt.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from test_per_item_calls import RubricLLM, make_grader

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
from autorubric.dataset import DataItem
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMClient
from autorubric.types import CANONICAL_NA_OPTION, CriterionJudgment, MultiChoiceJudgment

# Grading a rubric that differs from the training rubric warns which criteria get no
# examples; TestAWarningForCriteriaShownNoExamples checks that warning, the rest of this
# module what the prompts show.
pytestmark = pytest.mark.filterwarnings("ignore:Few-shot examples are not shown:UserWarning")

MET, UNMET = CriterionVerdict.MET, CriterionVerdict.UNMET
QUERY = "Is the claim true?"
GRADED = "Answer E: it is true."

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
TRAIN_RUBRIC = [SOURCES, TONE]

A = "Answer A: Smith (2020) in Nature shows the claim holds."
B = "Answer B: my buddy says so lol."
C = "Answer C: The WHO fact sheet states it."
D = "Answer D: hey there, Jones (2019) in Science backs it."
TRAINING = [
    (A, [MET, "Formal"]),
    (B, [UNMET, "Casual"]),
    (C, [UNMET, "Formal"]),
    (D, [MET, "Casual"]),
]
LABELS = dict(TRAINING)

_EXAMPLE = re.compile(r"<example_(\d+)>\n(.*?)\n</example_\1>", re.DOTALL)
_SUBMISSION = re.compile(r"<example_submission>(.*?)</example_submission>", re.DOTALL)
_VERDICT = re.compile(r"<verdict>(.*?)</verdict>")
_SELECTED_LABEL = re.compile(r"<selected_label>(.*?)</selected_label>")


def training_data(items: list[DataItem] | None = None) -> RubricDataset:
    # A list of its own: a test may change the training rubric's criteria in place.
    dataset = RubricDataset(prompt=QUERY, rubric=Rubric(list(TRAIN_RUBRIC)))
    for submission, ground_truth in TRAINING:
        dataset.add_item(submission=submission, description=submission, ground_truth=ground_truth)
    for item in items or []:
        dataset.items.append(item)
    return dataset


def _recording_generate(prompts: list[str]) -> AsyncMock:
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


async def prompts_for(
    rubric: list[Criterion],
    data: RubricDataset | None = None,
    *,
    after_build: Callable[[RubricDataset], None] | None = None,
    **grader_kwargs: Any,
) -> list[str]:
    """The user prompts, in rubric order, of grading one item on ``rubric`` per criterion.

    ``after_build`` is called with the training data once the grader is built.
    """
    prompts: list[str] = []
    data = data if data is not None else training_data()
    with patch.object(LLMClient, "generate", new=_recording_generate(prompts)):
        grader = CriterionGrader(
            judge_model_config=LLMConfig(model="test-model"),
            training_data=data,
            few_shot_config=FewShotConfig(n_examples=10),
            seed=0,
            **grader_kwargs,
        )
        if after_build is not None:
            after_build(data)
        await Rubric(rubric).grade(GRADED, grader=grader, query=QUERY)
    by_requirement = {c.requirement: i for i, c in enumerate(rubric)}

    def position(prompt: str) -> int:
        for requirement, idx in by_requirement.items():
            if f"\n{requirement}\n" in prompt:
                return idx
        raise AssertionError("prompt names no criterion of the rubric")

    return sorted(prompts, key=position)


def examples(user_prompt: str) -> dict[str, str]:
    """Each example's submission, in prompt order, with the verdict or label it shows."""
    shown: dict[str, str] = {}
    for _number, body in _EXAMPLE.findall(user_prompt):
        submission = _SUBMISSION.search(body)
        assert submission is not None
        label = _VERDICT.search(body) or _SELECTED_LABEL.search(body)
        assert label is not None
        shown[submission.group(1)] = label.group(1)
    return shown


class TestTheTrainingRubric:
    @pytest.mark.asyncio
    async def test_every_criterion_gets_its_examples(self):
        binary, multi_choice = await prompts_for(TRAIN_RUBRIC)
        assert examples(binary) == {s: labels[0].value for s, labels in LABELS.items()}
        assert examples(multi_choice) == {s: labels[1] for s, labels in LABELS.items()}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "graded",
        [
            # Not shown to the judge, and no change to what a label means.
            [SOURCES.model_copy(update={"weight": 7.0, "name": "refs"}), TONE],
            [
                SOURCES,
                TONE.model_copy(
                    update={
                        "weight": -3.0,
                        "name": "register",
                        "scale_type": "ordinal",
                        "aggregation": "mode",
                        "options": [
                            CriterionOption(label="Formal", value=0.25),
                            CriterionOption(label="Casual", value=0.75),
                        ],
                    }
                ),
            ],
            # The NA option auto_na_option adds, given explicitly.
            [
                SOURCES,
                TONE.model_copy(update={"options": [*(TONE.options or []), CANONICAL_NA_OPTION]}),
            ],
        ],
        ids=["binary-weight-and-name", "multi-choice-unshown-fields", "explicit-na-option"],
    )
    async def test_fields_the_judge_is_not_shown_do_not_matter(self, graded):
        baseline = await prompts_for(TRAIN_RUBRIC)
        prompts = await prompts_for(graded)
        assert [examples(p) for p in prompts] == [examples(p) for p in baseline]
        assert all(examples(p) for p in prompts)

    @pytest.mark.asyncio
    async def test_a_criterion_beyond_the_training_rubric_gets_none(self):
        extra = Criterion(name="length", weight=1.0, requirement="Is short")
        binary, multi_choice, extra_prompt = await prompts_for([*TRAIN_RUBRIC, extra])
        assert examples(binary) and examples(multi_choice)
        assert "<examples>" not in extra_prompt


class TestAnotherBinaryQuestion:
    @pytest.mark.asyncio
    async def test_another_requirement_gets_no_examples(self):
        other = Criterion(name="sources", weight=1.0, requirement="Is written in English")
        binary, multi_choice = await prompts_for([other, TONE])
        assert "<examples>" not in binary
        assert examples(multi_choice)

    @pytest.mark.asyncio
    async def test_another_criterion_type_gets_no_examples(self):
        # A negative weight asks whether the submission makes an error, which changes what
        # MET means (NEGATIVE_MET_DEFINITION), so the positive criterion's verdicts do not
        # carry over.
        negative = SOURCES.model_copy(update={"weight": -1.0})
        binary, _ = await prompts_for([negative, TONE])
        assert "<criterion_type>\nnegative\n</criterion_type>" in binary
        assert "<examples>" not in binary

    @pytest.mark.asyncio
    async def test_a_multi_choice_criterion_at_a_binary_index_gets_none(self):
        multi_at_0 = Criterion(
            name="sources",
            weight=1.0,
            requirement="How many sources?",
            options=[
                CriterionOption(label="None", value=0.0),
                CriterionOption(label="Some", value=1.0),
            ],
        )
        first, _ = await prompts_for([multi_at_0, TONE])
        assert "<examples>" not in first


class TestAnotherMultiChoiceQuestion:
    @pytest.mark.asyncio
    async def test_reordered_options_get_no_examples(self):
        # The training label "Formal" is option 0 of TONE but "Casual" here: shown as it was,
        # every example would carry the other label.
        reordered = TONE.model_copy(update={"options": list(reversed(TONE.options or []))})
        _, multi_choice = await prompts_for([SOURCES, reordered])
        assert "<examples>" not in multi_choice

    @staticmethod
    def three_tone_data() -> RubricDataset:
        """Training data whose tone criterion has a third option, "Playful"."""
        playful = CriterionOption(label="Playful", value=0.5)
        rubric = [SOURCES, TONE.model_copy(update={"options": [*(TONE.options or []), playful]})]
        dataset = RubricDataset(prompt=QUERY, rubric=Rubric(rubric))
        for submission, ground_truth in [*TRAINING, ("Answer F: lol jk", [UNMET, "Playful"])]:
            dataset.add_item(
                submission=submission, description=submission, ground_truth=ground_truth
            )
        return dataset

    @pytest.mark.asyncio
    async def test_fewer_options_get_no_examples(self):
        # Without auto_na_option, the example labelled "Playful" (option 2) names no option
        # of the graded criterion.
        _, multi_choice = await prompts_for(
            [SOURCES, TONE], self.three_tone_data(), auto_na_option=False
        )
        assert "<examples>" not in multi_choice

    @pytest.mark.asyncio
    async def test_relabelled_options_get_no_examples(self):
        relabelled = TONE.model_copy(
            update={
                "options": [
                    CriterionOption(label="Stiff", value=1.0),
                    CriterionOption(label="Relaxed", value=0.0),
                ]
            }
        )
        _, multi_choice = await prompts_for([SOURCES, relabelled])
        assert "<examples>" not in multi_choice

    @pytest.mark.asyncio
    async def test_another_na_flag_gets_no_examples(self):
        playful_is_na = TONE.model_copy(
            update={
                "options": [
                    *(TONE.options or []),
                    CriterionOption(label="Playful", value=0.0, na=True),
                ]
            }
        )
        _, multi_choice = await prompts_for(
            [SOURCES, playful_is_na], self.three_tone_data(), auto_na_option=False
        )
        assert "<examples>" not in multi_choice

    @pytest.mark.asyncio
    async def test_another_requirement_gets_no_examples(self):
        other = TONE.model_copy(update={"requirement": "Which register is used?"})
        _, multi_choice = await prompts_for([SOURCES, other])
        assert "<examples>" not in multi_choice

    @pytest.mark.asyncio
    async def test_a_binary_criterion_at_a_multi_choice_index_gets_none(self):
        binary_at_1 = Criterion(name="tone", weight=1.0, requirement="Is formal")
        _, second = await prompts_for([SOURCES, binary_at_1])
        assert "<examples>" not in second


class TestTrainingItemsWithTheirOwnRubric:
    """A training item's labels describe its own rubric (``DataItem.rubric``), if it has one."""

    @staticmethod
    def own_rubric_item(rubric: list[Criterion], ground_truth: list[Any]) -> DataItem:
        return DataItem(
            submission="Answer F: trust me.",
            description="own rubric",
            ground_truth=ground_truth,
            rubric=Rubric(rubric),
        )

    @pytest.mark.asyncio
    async def test_the_same_questions_lend_their_labels(self):
        same = [SOURCES.model_copy(update={"weight": 3.0}), TONE]
        item = self.own_rubric_item(same, [MET, "Casual"])
        binary, multi_choice = await prompts_for(TRAIN_RUBRIC, training_data([item]))
        assert examples(binary)["Answer F: trust me."] == "MET"
        assert examples(multi_choice)["Answer F: trust me."] == "Casual"

    @pytest.mark.asyncio
    async def test_another_question_lends_no_label_for_it(self):
        other = Criterion(name="english", weight=1.0, requirement="Is written in English")
        item = self.own_rubric_item([other, TONE], [MET, "Casual"])
        binary, multi_choice = await prompts_for(TRAIN_RUBRIC, training_data([item]))
        assert "Answer F: trust me." not in examples(binary)
        assert examples(multi_choice)["Answer F: trust me."] == "Casual"

    @pytest.mark.asyncio
    async def test_a_shorter_own_rubric_lends_labels_for_the_criteria_it_has(self):
        item = self.own_rubric_item([SOURCES], [UNMET])
        binary, multi_choice = await prompts_for(TRAIN_RUBRIC, training_data([item]))
        assert examples(binary)["Answer F: trust me."] == "UNMET"
        assert "Answer F: trust me." not in examples(multi_choice)

    @pytest.mark.asyncio
    async def test_another_option_order_lends_no_label(self):
        reordered = TONE.model_copy(update={"options": list(reversed(TONE.options or []))})
        # "Formal" names the same option either way, but the item was labelled on another
        # question: its options in another order.
        item = self.own_rubric_item([SOURCES, reordered], [MET, "Formal"])
        _, multi_choice = await prompts_for(TRAIN_RUBRIC, training_data([item]))
        assert "Answer F: trust me." not in examples(multi_choice)


# =============================================================================
# One call per item
# =============================================================================

_JUDGMENT = re.compile(r'<judgment id="(?P<id>c\d+)">\n(?P<body>.*?)\n</judgment>', re.S)


async def per_item_prompt(
    rubric: list[Criterion],
    data: RubricDataset | None = None,
    *,
    after_build: Callable[[RubricDataset], None] | None = None,
) -> str:
    """The user prompt of grading one item on ``rubric`` in one call (see ``prompts_for``)."""
    fake = RubricLLM()
    data = data if data is not None else training_data()
    grader = make_grader(
        {"default": fake},
        judge_model_config=LLMConfig(model="test-model"),
        llm_calls="per_item",
        training_data=data,
        few_shot_config=FewShotConfig(n_examples=10),
        shuffle_options=False,
    )
    if after_build is not None:
        after_build(data)
    await Rubric(rubric).grade(GRADED, grader=grader, query=QUERY)
    (call,) = fake.calls
    return call.user_prompt


def per_item_judgments(user_prompt: str) -> dict[str, dict[str, str]]:
    """Each example's submission with its judgments' content by criterion id."""
    if "<examples>" not in user_prompt:
        return {}
    block = user_prompt.split("<examples>\n", 1)[1].split("\n</examples>", 1)[0]
    shown: dict[str, dict[str, str]] = {}
    for _number, body in _EXAMPLE.findall(block):
        submission = _SUBMISSION.search(body)
        assert submission is not None
        shown[submission.group(1)] = {j["id"]: j["body"] for j in _JUDGMENT.finditer(body)}
    return shown


class TestOneCallPerItem:
    @pytest.mark.asyncio
    async def test_the_training_rubric_gets_every_judgment(self):
        shown = per_item_judgments(await per_item_prompt(TRAIN_RUBRIC))
        assert set(shown) == set(LABELS)
        assert all(list(judgments) == ["c0", "c1"] for judgments in shown.values())

    @pytest.mark.asyncio
    async def test_reordered_options_get_no_judgment(self):
        reordered = TONE.model_copy(update={"options": list(reversed(TONE.options or []))})
        shown = per_item_judgments(await per_item_prompt([SOURCES, reordered]))
        assert set(shown) == set(LABELS)
        assert all(list(judgments) == ["c0"] for judgments in shown.values())

    @pytest.mark.asyncio
    async def test_another_binary_requirement_gets_no_judgment(self):
        other = Criterion(name="sources", weight=1.0, requirement="Is written in English")
        shown = per_item_judgments(await per_item_prompt([other, TONE]))
        assert all(list(judgments) == ["c1"] for judgments in shown.values())

    @pytest.mark.asyncio
    async def test_another_criterion_type_gets_no_judgment(self):
        negative = SOURCES.model_copy(update={"weight": -1.0})
        shown = per_item_judgments(await per_item_prompt([negative, TONE]))
        assert all(list(judgments) == ["c1"] for judgments in shown.values())

    @staticmethod
    def own_rubric_item(rubric: list[Criterion], ground_truth: list[Any]) -> DataItem:
        return DataItem(
            submission="Answer F: trust me.",
            description="own rubric",
            ground_truth=ground_truth,
            rubric=Rubric(rubric),
        )

    @pytest.mark.asyncio
    async def test_a_training_item_with_another_question_lends_the_other_labels(self):
        # As per criterion: the item has no label for the criterion its own rubric poses
        # another question at, so its example has no judgment there.
        other = Criterion(name="english", weight=1.0, requirement="Is written in English")
        item = self.own_rubric_item([other, TONE], [MET, "Casual"])
        shown = per_item_judgments(await per_item_prompt(TRAIN_RUBRIC, training_data([item])))
        assert list(shown["Answer F: trust me."]) == ["c1"]
        assert "Casual" in shown["Answer F: trust me."]["c1"]

    @pytest.mark.asyncio
    async def test_a_shorter_own_rubric_lends_labels_for_the_criteria_it_has(self):
        item = self.own_rubric_item([SOURCES], [UNMET])
        shown = per_item_judgments(await per_item_prompt(TRAIN_RUBRIC, training_data([item])))
        assert shown["Answer F: trust me."] == {"c0": "<verdict>UNMET</verdict>"}

    @pytest.mark.asyncio
    async def test_a_training_item_with_no_label_for_the_training_rubric_is_no_example(self):
        other = Criterion(name="english", weight=1.0, requirement="Is written in English")
        item = self.own_rubric_item([other], [MET])
        shown = per_item_judgments(await per_item_prompt(TRAIN_RUBRIC, training_data([item])))
        assert set(shown) == set(LABELS)

    @pytest.mark.parametrize("seed", range(20))
    def test_a_missing_label_covers_nothing(self, seed):
        # With balance_verdicts, an item's labels count toward coverage; a criterion it has
        # no label for is no (criterion index, label) pair, so the item adds less than an
        # item labelled on both and is never picked over one.
        item = self.own_rubric_item([ENGLISH, TONE], [MET, "Casual"])
        data = RubricDataset(prompt=QUERY, rubric=Rubric(list(TRAIN_RUBRIC)))
        data.add_item(submission=A, description=A, ground_truth=[MET, "Formal"])
        data.add_item(submission=B, description=B, ground_truth=[UNMET, "Casual"])
        data.items.append(item)
        grader = make_grader(
            {"default": RubricLLM()},
            judge_model_config=LLMConfig(model="test-model"),
            llm_calls="per_item",
            training_data=data,
            few_shot_config=FewShotConfig(n_examples=2, balance_verdicts=True, seed=seed),
        )
        assert {s for s, _, _ in grader._item_examples["default"]} == {A, B}

    @pytest.mark.asyncio
    async def test_an_unusable_label_still_makes_an_item_no_example(self):
        item = self.own_rubric_item([SOURCES, TONE], [MET, "Sarcastic"])
        shown = per_item_judgments(await per_item_prompt(TRAIN_RUBRIC, training_data([item])))
        assert "Answer F: trust me." not in shown


# =============================================================================
# The training rubric changed after the grader was built
# =============================================================================

ENGLISH = Criterion(name="english", weight=1.0, requirement="Is written in English")
SHORT = Criterion(name="length", weight=1.0, requirement="Is short")


def replace_rubric(data: RubricDataset) -> None:
    data.rubric = Rubric([ENGLISH, TONE])


def replace_first_criterion(data: RubricDataset) -> None:
    assert data.rubric is not None
    data.rubric.rubric[0] = ENGLISH


def append_criterion(data: RubricDataset) -> None:
    assert data.rubric is not None
    data.rubric.rubric.append(SHORT)


class TestTheTrainingRubricAsTheGraderWasBuilt:
    """The examples are drawn at construction, so they describe the training rubric as it was
    then: a later change to it changes no prompt."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("change", [replace_rubric, replace_first_criterion])
    async def test_the_new_question_gets_no_examples(self, change):
        english, multi_choice = await prompts_for([ENGLISH, TONE], after_build=change)
        assert "<examples>" not in english
        assert examples(multi_choice) == {s: labels[1] for s, labels in LABELS.items()}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("change", [replace_rubric, replace_first_criterion, append_criterion])
    async def test_the_old_questions_keep_their_examples(self, change):
        assert await prompts_for(TRAIN_RUBRIC, after_build=change) == await prompts_for(
            TRAIN_RUBRIC
        )

    @pytest.mark.asyncio
    async def test_an_appended_criterion_gets_no_examples(self):
        *_, short = await prompts_for([*TRAIN_RUBRIC, SHORT], after_build=append_criterion)
        assert "<examples>" not in short

    @pytest.mark.asyncio
    @pytest.mark.parametrize("change", [replace_rubric, replace_first_criterion])
    async def test_one_call_per_item_shows_no_judgment_for_the_new_question(self, change):
        shown = per_item_judgments(await per_item_prompt([ENGLISH, TONE], after_build=change))
        assert set(shown) == set(LABELS)
        assert all(list(judgments) == ["c1"] for judgments in shown.values())

    @pytest.mark.asyncio
    async def test_one_call_per_item_shows_no_judgment_for_an_appended_criterion(self):
        shown = per_item_judgments(
            await per_item_prompt([*TRAIN_RUBRIC, SHORT], after_build=append_criterion)
        )
        assert set(shown) == set(LABELS)
        assert all(list(judgments) == ["c0", "c1"] for judgments in shown.values())


# =============================================================================
# A warning names the criteria shown no examples
# =============================================================================

NOT_SHOWN = "Few-shot examples are not shown for"


def warning_grader(llm_calls: str = "per_criterion", data: RubricDataset | None = None) -> Any:
    return make_grader(
        {"default": RubricLLM()},
        judge_model_config=LLMConfig(model="test-model"),
        llm_calls=llm_calls,
        training_data=data if data is not None else training_data(),
        few_shot_config=FewShotConfig(n_examples=10),
    )


async def grade_on(grader: CriterionGrader, rubric: list[Criterion]) -> None:
    await Rubric(rubric).grade(GRADED, grader=grader, query=QUERY)


def not_shown_messages(record: Any) -> list[str]:
    return [str(w.message) for w in record if str(w.message).startswith(NOT_SHOWN)]


REORDERED_TONE = TONE.model_copy(update={"options": list(reversed(TONE.options or []))})


class TestAWarningForCriteriaShownNoExamples:
    """A graded criterion the examples do not describe gets none, silently but for this
    ``UserWarning``: once per grader per (criterion index, graded question), in both
    ``llm_calls`` modes, naming the criterion and why."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    @pytest.mark.parametrize(
        ("graded", "message"),
        [
            (
                [ENGLISH, TONE],
                "criterion c0 ('english'): it differs from the training rubric's "
                "criterion at its index in its requirement,",
            ),
            (
                [SOURCES.model_copy(update={"weight": -1.0}), TONE],
                "criterion c0 ('sources'): it differs from the training rubric's "
                "criterion at its index in its criterion type,",
            ),
            (
                [SOURCES, REORDERED_TONE],
                "criterion c1 ('tone'): it differs from the training rubric's criterion "
                "at its index in its options,",
            ),
            (
                [SOURCES, Criterion(name="tone", weight=1.0, requirement="Is formal")],
                "criterion c1 ('tone'): it is binary and the training rubric's criterion at "
                "its index is multi-choice,",
            ),
            (
                [SOURCES, TONE, SHORT],
                "criterion c2 ('length'): the training rubric has no criterion at its index,",
            ),
            (
                [ENGLISH.model_copy(update={"weight": -1.0}), TONE],
                "in its requirement and criterion type,",
            ),
        ],
        ids=["requirement", "criterion-type", "options", "kind", "beyond", "two-differences"],
    )
    async def test_the_criterion_and_the_reason_are_named(self, llm_calls, graded, message):
        grader = warning_grader(llm_calls)
        with pytest.warns(UserWarning) as record:
            await grade_on(grader, graded)
        (warned,) = not_shown_messages(record)
        assert message in warned
        assert warned.endswith("so the training labels do not answer it")
        # Attributed to the code that asked for grading.
        assert next(w for w in record if str(w.message) == warned).filename == __file__

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    async def test_once_per_grader_not_once_per_item(self, llm_calls):
        grader = warning_grader(llm_calls)
        with pytest.warns(UserWarning) as record:
            for _ in range(3):
                await grade_on(grader, [ENGLISH, REORDERED_TONE])
        assert len(not_shown_messages(record)) == 2

    @pytest.mark.asyncio
    async def test_another_question_at_the_same_index_warns_again(self):
        grader = warning_grader()
        with pytest.warns(UserWarning) as record:
            await grade_on(grader, [ENGLISH, TONE])
            await grade_on(grader, [SHORT, TONE])
            await grade_on(grader, [ENGLISH, TONE])
        messages = not_shown_messages(record)
        assert len(messages) == 2
        assert "'english'" in messages[0] and "'length'" in messages[1]

    @pytest.mark.asyncio
    async def test_each_grader_warns_for_itself(self):
        with pytest.warns(UserWarning) as record:
            await grade_on(warning_grader(), [ENGLISH, TONE])
            await grade_on(warning_grader(), [ENGLISH, TONE])
        assert len(not_shown_messages(record)) == 2

    @pytest.mark.asyncio
    async def test_under_warnings_as_errors_it_raises_on_every_call(self):
        grader = warning_grader()
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with pytest.raises(UserWarning, match=NOT_SHOWN):
                    await grade_on(grader, [ENGLISH, TONE])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    @pytest.mark.parametrize(
        "graded",
        [
            TRAIN_RUBRIC,
            # Fields the judge is not shown, and the NA option auto_na_option adds.
            [
                SOURCES.model_copy(update={"weight": 7.0, "name": "refs"}),
                TONE.model_copy(update={"options": [*(TONE.options or []), CANONICAL_NA_OPTION]}),
            ],
            [SOURCES],  # a shorter rubric
        ],
        ids=["training-rubric", "unshown-fields", "shorter"],
    )
    async def test_criteria_the_examples_describe_warn_nothing(self, llm_calls, graded):
        grader = warning_grader(llm_calls)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await grade_on(grader, graded)

    @pytest.mark.asyncio
    async def test_a_grader_without_training_data_warns_nothing(self):
        grader = make_grader(
            {"default": RubricLLM()}, judge_model_config=LLMConfig(model="test-model")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await grade_on(grader, [ENGLISH, REORDERED_TONE, SHORT])

    @pytest.mark.asyncio
    async def test_a_training_item_with_its_own_rubric_warns_nothing(self):
        # Its labels for another question are left out of the examples (training data, not
        # the rubric being graded), and the graded criteria get the other items' examples.
        item = TestTrainingItemsWithTheirOwnRubric.own_rubric_item([ENGLISH], [MET])
        grader = warning_grader(data=training_data([item]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await grade_on(grader, TRAIN_RUBRIC)

    @pytest.mark.asyncio
    async def test_a_grader_restored_without_the_record_warns(self):
        """A grader pickled before the warning existed (or a subclass that skips
        ``__init__``) starts from the class default: it has warned about nothing."""
        grader = warning_grader()
        restored = CriterionGrader.__new__(CriterionGrader)
        restored.__dict__.update(
            {k: v for k, v in vars(grader).items() if k != "_examples_not_shown_warned"}
        )
        with pytest.warns(UserWarning, match=NOT_SHOWN):
            await grade_on(restored, [ENGLISH, TONE])
        assert CriterionGrader._examples_not_shown_warned == frozenset()
