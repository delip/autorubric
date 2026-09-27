"""Item-level few-shot examples under ``CriterionGrader(llm_calls="per_item")``.

When an LLM judge grades an item's whole rubric in one call, its few-shot examples are whole
training items, each shown once per call with its ground truth for every criterion, rather
than examples drawn per criterion (which would put N x k example submissions in one call).

Selection (``_select_item_examples``) happens once per LLM judge at construction:

- An item is eligible when it has ground truth and every label is usable for its criterion's
  kind in the training rubric (a criterion the training rubric lacks counts as binary): a
  ``CriterionVerdict``, or its value, for a binary criterion; for a multi-choice one, an
  option label that resolves or an ``int`` (not ``bool``) option index in range, among the
  options the judge is shown (with ``auto_na_option``, the NA option it adds too).
- Each judge's examples are kept as the training items were at construction.
- The eligible items, in dataset order, are shuffled by
  ``_derive_shuffle_rng(few_shot_seed, "few_shot", -1, judge_id)``.
- ``balance_verdicts=False`` takes the first ``n_examples`` of that order.
- ``balance_verdicts=True`` picks by greedy label coverage over (criterion index, label)
  pairs, a multi-choice label being the option index it resolves to however it is named,
  earliest in the seeded order on a tie, never an item whose submission was already picked,
  then fills the remaining slots in the seeded order.

Rendering (``_judge_rubric_in_one_call``) puts one ``<examples>`` block between
``</criteria>`` and the input, a ``<judgment id="cK">`` per criterion in the tags of a
per-criterion example, a multi-choice option numbered as this call presents the options.
The system prompt says how to read the examples exactly when there are any. The examples are
rendered with the prompts, before the call, so an error rendering them raises.

Nothing here reaches the network: LLM judges are the recording ``RubricLLM`` fakes of the
per-item call tests, and a cascade's decision model answers through the decision-model
tests' fake SDK client.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
from typing import Any, cast

import pytest
import typesafe_sdk

# The decision-model tests' guards, registered for this module by importing them: no
# ``httpx2`` transport may send, the ``TYPESAFE_*`` variables are cleared, and the rate-limit
# pool starts fresh for each test.
from decision.conftest import (  # noqa: F401
    FakeSDK,
    FakeSDKClient,
    _fresh_rate_limit_pool,
    _isolate_typesafe_env,
    _no_network,
    make_response,
)
from test_per_item_calls import RubricLLM, make_grader, one_judge, presented_labels, rubric_blocks

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DecisionModelConfig,
    EscalationConfig,
    FewShotConfig,
    LLMConfig,
    Rubric,
    RubricDataset,
    fill_ground_truth,
)
from autorubric.dataset import DataItem
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.prompts import (
    FEW_SHOT_SYSTEM_PROMPT_ADDITION,
    GRADER_SYSTEM_PROMPT_DEFAULT,
    MULTI_CHOICE_FEW_SHOT_ADDITION,
    MULTI_CHOICE_SYSTEM_PROMPT,
    RUBRIC_JUDGMENT_EXAMPLES,
    build_rubric_system_prompt,
)
from autorubric.types import CANONICAL_NA_OPTION, RubricJudgment

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CANNOT_ASSESS = CriterionVerdict.CANNOT_ASSESS

SEED = 11
LLM = LLMConfig(model="test-model")
OTHER_LLM = LLMConfig(model="other-model")
API_KEY = "ts-test-key-never-persisted-5a1d"
NA_LABEL = CANONICAL_NA_OPTION.label
QUERY = "How do plants make food?"
REFERENCE = "Photosynthesis turns light, water and carbon dioxide into glucose and oxygen."
SUBMISSION = "Plants use light to turn water and carbon dioxide into sugar."
OTHER_SUBMISSION = "Plants eat soil."
GUIDELINES = "Judge at a middle-school level."

# The examples sentence of the system prompt, written out literally.
EXAMPLES_SENTENCE = (
    "The <examples> show earlier submissions with the correct judgment of each criterion, "
    "each under the criterion's id; use them as the guides describe."
)

LIGHT = Criterion(name="light", weight=5.0, requirement="Mentions light")
CLARITY = Criterion(
    name="clarity",
    weight=2.0,
    requirement="How clear is the explanation?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Unclear", value=0.0),
        CriterionOption(label="Mostly clear", value=0.6),
        CriterionOption(label="Very clear", value=1.0),
    ],
)
JARGON = Criterion(name="jargon", weight=-2.0, requirement="Uses unexplained jargon")
TONE = Criterion(
    name="tone",
    weight=1.0,
    requirement="What tone does the answer take?",
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Casual", value=0.5),
        CriterionOption(label="Playful", value=0.5),
        CriterionOption(label="Neutral", value=1.0),
    ],
)
# c0 light, c1 clarity, c2 jargon, c3 tone.
TRAIN_RUBRIC = [LIGHT, CLARITY, JARGON, TONE]
# Criteria the training rubric lacks, of each kind.
WATER = Criterion(name="water", weight=1.0, requirement="Mentions water")
LENGTH = Criterion(
    name="length",
    weight=1.0,
    requirement="How long is the answer?",
    options=[CriterionOption(label="Short", value=1.0), CriterionOption(label="Long", value=0.0)],
)

# Four labelled training items: (submission, ground truth, reasons). The first has an empty
# reason (never shown) and a None one; the second has no reasons at all.
TRAIN_ROWS: list[tuple[str, list[Any], list[str | None] | None]] = [
    (
        "Plants use sunlight to make sugar.",
        [MET, "Very clear", UNMET, "Formal"],
        ["Names sunlight.", None, "", "Plain register."],
    ),
    ("Plants eat dirt, lol.", [UNMET, "Unclear", UNMET, "Playful"], None),
    (
        "Chlorophyll drives photolysis!",
        [MET, "Mostly clear", MET, "Neutral"],
        [None, "Some terms.", "Photolysis is unexplained.", None],
    ),
    (
        "Light and water, basically.",
        [MET, "Mostly clear", UNMET, "Casual"],
        ["Light.", "Clear enough.", "No jargon.", "Informal."],
    ),
]
TRAINING = {submission: (labels, reasons) for submission, labels, reasons in TRAIN_ROWS}

EXAMPLE = re.compile(
    r"<example_(?P<n>\d+)>\n<example_submission>(?P<submission>.*?)</example_submission>\n"
    r"(?P<body>.*?)\n</example_(?P=n)>",
    re.S,
)
JUDGMENT = re.compile(r'<judgment id="(?P<id>c\d+)">\n(?P<body>.*?)\n</judgment>', re.S)


# =============================================================================
# Helpers
# =============================================================================


def training_data(rows: Sequence[tuple[str, list[Any], list[str | None] | None]] = TRAIN_ROWS):
    dataset = RubricDataset(prompt=QUERY, rubric=Rubric(TRAIN_RUBRIC))
    for submission, labels, reasons in rows:
        dataset.add_item(
            submission, submission, ground_truth=list(labels), ground_truth_reasons=reasons
        )
    return dataset


def few_shot(n: int = 4, *, balance: bool = True, include_reason: bool = False, **kwargs: Any):
    return FewShotConfig(
        n_examples=n, balance_verdicts=balance, include_reason=include_reason, **kwargs
    )


def seeded_order(items: Sequence[Any], seed: int, judge_id: str) -> list[Any]:
    """``items`` shuffled as a judge's item-level examples are: by the RNG keyed on
    ``"{seed}:few_shot:-1:{judge_id}"`` (the whole rubric's ``-1`` in the criterion slot),
    written out here independently of the library."""
    key = f"{seed}:few_shot:-1:{judge_id}"
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**31))
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def set_label(item: DataItem, criterion_idx: int, label: object) -> None:
    """Set one ground-truth label as is, bypassing ``DataItem``'s validation."""
    labels = cast(list[Any], item.ground_truth)
    labels[criterion_idx] = label


def selected_option(judgment: str) -> int:
    """The option number of a multi-choice judgment's content."""
    match = re.search(r"<selected_option>(\d+)</selected_option>", judgment)
    assert match is not None
    return int(match[1])


def names(items: Sequence[DataItem]) -> list[str]:
    return [item.description for item in items]


def picked(grader: CriterionGrader, judge_id: str) -> list[DataItem]:
    """The training items a judge's examples show, in order: the judge's selection
    (``_select_item_examples``), which is what the grader keeps as its examples."""
    items = grader._select_item_examples(judge_id)
    assert [s for s, _, _ in grader._item_examples[judge_id]] == [i.submission for i in items]
    return items


def per_item_grader(
    fake: RubricLLM, data: RubricDataset | None = None, config: FewShotConfig | None = None, **kw
) -> CriterionGrader:
    """A single-judge ``per_item`` grader with training data, its LLM judge ``fake``."""
    return make_grader(
        {"default": fake},
        judge_model_config=LLM,
        llm_calls="per_item",
        training_data=data if data is not None else training_data(),
        few_shot_config=config if config is not None else few_shot(),
        **kw,
    )


def split_examples(user_prompt: str) -> tuple[str, str]:
    """The prompt's ``<criteria>`` part and its ``<examples>`` block."""
    criteria, rest = user_prompt.split("</criteria>\n\n", 1)
    assert rest.startswith("<examples>\n")
    return criteria, rest.split("\n</examples>\n\n", 1)[0] + "\n</examples>"


def shown_examples(user_prompt: str) -> list[tuple[str, dict[str, str]]]:
    """Each example of the prompt: its submission and each judgment's content by id."""
    _, block = split_examples(user_prompt)
    examples = []
    for number, match in enumerate(EXAMPLE.finditer(block), 1):
        assert int(match["n"]) == number
        judgments = {j["id"]: j["body"] for j in JUDGMENT.finditer(match["body"])}
        examples.append((match["submission"], judgments))
    return examples


def expected_judgment(label: Any, reason: str | None, presented: list[str] | None) -> str:
    """A judgment's content, written out: the verdict, or the option's number in the order
    the call presents the options (``presented``) and its label; then the reason if shown."""
    if presented is None:
        lines = [f"<verdict>{label.value}</verdict>"]
    else:
        lines = [
            f"<selected_option>{presented.index(label) + 1}</selected_option>",
            f"<selected_label>{label}</selected_label>",
        ]
    if reason:
        lines.append(f"<reason>{reason}</reason>")
    return "\n".join(lines)


def assert_examples_match(
    user_prompt: str, grader: CriterionGrader, judge_id: str, include_reason: bool
) -> None:
    """The prompt shows the judge's selected items, in order, each judged on every criterion
    as expected_judgment writes it out, multi-choice options numbered as this very prompt
    presents them."""
    criteria, _ = split_examples(user_prompt)
    blocks = rubric_blocks(criteria)
    examples = shown_examples(user_prompt)
    assert [s for s, _ in examples] == [s for s, _, _ in grader._item_examples[judge_id]]
    for submission, judgments in examples:
        labels, reasons = TRAINING[submission]
        assert list(judgments) == [f"c{idx}" for idx in range(len(labels))]
        for idx, label in enumerate(labels):
            block = blocks[f"c{idx}"]
            presented = presented_labels(block) if "<options>" in block else None
            reason = reasons[idx] if include_reason and reasons else None
            assert judgments[f"c{idx}"] == expected_judgment(label, reason, presented)


# =============================================================================
# Selection
# =============================================================================


def big_pool(n_items: int = 24) -> RubricDataset:
    rng = random.Random(99)
    dataset = RubricDataset(prompt=QUERY, rubric=Rubric(TRAIN_RUBRIC))
    for k in range(n_items):
        labels = [
            rng.choice([MET, UNMET, CANNOT_ASSESS]),
            rng.choice(["Unclear", "Mostly clear", "Very clear"]),
            rng.choice([MET, UNMET]),
            rng.choice(["Formal", "Casual", "Playful", "Neutral"]),
        ]
        dataset.add_item(f"submission {k}", f"i{k}", ground_truth=labels)
    return dataset


PANEL = [JudgeSpec(LLM, "alpha"), JudgeSpec(OTHER_LLM, "beta")]


def panel_grader(data: RubricDataset, config: FewShotConfig, seed: int = SEED, **kw: Any):
    return make_grader(
        {"alpha": RubricLLM(), "beta": RubricLLM()},
        judges=PANEL,
        llm_calls="per_item",
        training_data=data,
        few_shot_config=config,
        seed=seed,
        **kw,
    )


class TestSelectionIsSeededPerJudge:
    @pytest.mark.parametrize("balance", [True, False])
    def test_deterministic_for_a_seed_and_judge(self, balance):
        data = big_pool()
        first = panel_grader(data, few_shot(4, balance=balance))
        second = panel_grader(data, few_shot(4, balance=balance))
        for judge_id in ("alpha", "beta"):
            assert len(first._item_examples[judge_id]) == 4
            assert first._item_examples[judge_id] == second._item_examples[judge_id]

    @pytest.mark.parametrize("balance", [True, False])
    def test_each_judge_draws_its_own_examples(self, balance):
        grader = panel_grader(big_pool(), few_shot(4, balance=balance))
        assert names(picked(grader, "alpha")) != names(picked(grader, "beta"))

    def test_another_seed_draws_other_examples(self):
        data = big_pool()
        one = panel_grader(data, few_shot(4, balance=False), seed=3)
        other = panel_grader(data, few_shot(4, balance=False), seed=4)
        assert names(picked(one, "alpha")) != names(picked(other, "alpha"))

    def test_without_balance_the_examples_are_the_first_of_the_seeded_order(self):
        data = big_pool()
        grader = panel_grader(data, few_shot(5, balance=False))
        for judge_id in ("alpha", "beta"):
            expected = seeded_order(data.items, SEED, judge_id)[:5]
            assert names(picked(grader, judge_id)) == names(expected)

    def test_the_few_shot_seed_takes_precedence_over_the_master_seed(self):
        data = big_pool()
        config = few_shot(5, balance=False, seed=77)
        one = panel_grader(data, config, seed=3)
        other = panel_grader(data, config, seed=1234)
        for judge_id in ("alpha", "beta"):
            expected = names(seeded_order(data.items, 77, judge_id)[:5])
            assert names(picked(one, judge_id)) == expected
            assert names(picked(other, judge_id)) == expected

    def test_an_unset_few_shot_seed_follows_the_master_seed(self):
        data = big_pool()
        grader = panel_grader(data, few_shot(5, balance=False), seed=42)
        assert names(picked(grader, "alpha")) == names(seeded_order(data.items, 42, "alpha")[:5])

    def test_every_llm_judge_gets_examples_and_the_per_criterion_ones_stay_empty(self):
        grader = panel_grader(big_pool(), few_shot(3))
        assert set(grader._item_examples) == {"alpha", "beta"}
        assert grader._criterion_examples == {}
        assert grader._multi_choice_examples == {}

    def test_a_per_criterion_grader_draws_no_item_examples(self):
        grader = make_grader(
            {"alpha": RubricLLM(), "beta": RubricLLM()},
            judges=PANEL,
            training_data=big_pool(),
            few_shot_config=few_shot(3),
        )
        assert grader._item_examples == {}
        assert grader._criterion_examples and grader._multi_choice_examples

    def test_a_decision_model_gets_no_examples(self):
        grader = make_grader(
            {"llm": RubricLLM()},
            judges=[JudgeSpec(DecisionModelConfig(model="jev-latest", api_key=API_KEY), "jev")]
            + [JudgeSpec(LLM, "llm")],
            llm_calls="per_item",
            training_data=big_pool(),
            few_shot_config=few_shot(3),
        )
        assert list(grader._item_examples) == ["llm"]
        assert len(grader._item_examples["llm"]) == 3


class TestEligibility:
    """Only an item with ground truth whose every label is usable for its criterion's kind
    can be a whole-rubric example."""

    @staticmethod
    def pool() -> tuple[RubricDataset, list[str]]:
        dataset = RubricDataset(prompt=QUERY, rubric=Rubric(TRAIN_RUBRIC))
        eligible = []

        def add(name: str, labels: list[Any] | None, ok: bool) -> DataItem:
            dataset.add_item(f"{name} text", name, ground_truth=labels)
            if ok:
                eligible.append(name)
            return dataset.items[-1]

        add("good", [MET, "Very clear", UNMET, "Formal"], True)
        # Option labels resolve case- and whitespace-insensitively.
        add("loose label", [UNMET, "  mostly CLEAR ", MET, "neutral"], True)
        add("no ground truth", None, False)
        add("unresolvable label", [MET, "Crystal clear", UNMET, "Formal"], False)
        add("non-verdict binary label", ["nope", "Unclear", UNMET, "Casual"], False)
        add("option label at a binary criterion", [MET, "Unclear", "Formal", "Casual"], False)
        # A binary label may be a verdict's value, as resolve_ground_truth reads it.
        add("verdict values", ["UNMET", "Unclear", "MET", "Casual"], True)
        add("lowercase verdict value", ["met", "Unclear", UNMET, "Casual"], False)
        # Clarity and tone have no NA option of their own; the grader adds one (the default
        # auto_na_option), and the label fill_ground_truth records when a judge chose it
        # resolves to it.
        add("NA label", [MET, NA_LABEL, UNMET, NA_LABEL], ok=True)
        # Option indices bypass DataItem's validation (it takes labels and verdicts only).
        set_label(add("index in range", [MET, "Unclear", UNMET, "Casual"], True), 1, 2)
        set_label(add("NA option index", [MET, "Unclear", UNMET, "Casual"], True), 3, 4)
        set_label(add("index out of range", [MET, "Unclear", UNMET, "Casual"], False), 3, 5)
        set_label(add("negative index", [MET, "Unclear", UNMET, "Casual"], False), 1, -1)
        set_label(add("bool index", [MET, "Unclear", UNMET, "Casual"], False), 1, True)
        add("also good", [CANNOT_ASSESS, "Unclear", MET, "Playful"], True)
        return dataset, eligible

    @pytest.mark.parametrize("balance", [True, False])
    def test_the_examples_are_the_eligible_items(self, balance):
        data, eligible = self.pool()
        grader = per_item_grader(RubricLLM(), data, few_shot(50, balance=balance))
        assert sorted(names(picked(grader, "default"))) == sorted(eligible)

    def test_the_seeded_order_is_over_the_eligible_items_only(self):
        data, eligible = self.pool()
        grader = per_item_grader(RubricLLM(), data, few_shot(50, balance=False))
        eligible_items = [item for item in data.items if item.description in eligible]
        assert names(picked(grader, "default")) == names(
            seeded_order(eligible_items, SEED, "default")
        )

    def test_without_auto_na_option_there_is_no_na_option_to_name(self):
        data, eligible = self.pool()
        grader = per_item_grader(
            RubricLLM(), data, few_shot(50, balance=False), auto_na_option=False
        )
        without_na = sorted(set(eligible) - {"NA label", "NA option index"})
        assert sorted(names(picked(grader, "default"))) == without_na

    @pytest.mark.asyncio
    async def test_an_option_index_is_rendered_as_its_option(self):
        data, _ = self.pool()
        fake = RubricLLM()
        grader = per_item_grader(fake, data, few_shot(50, balance=False), shuffle_options=False)
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        examples = dict(shown_examples(call.user_prompt))
        assert examples["index in range text"]["c1"] == (
            "<selected_option>3</selected_option>\n<selected_label>Very clear</selected_label>"
        )
        # The NA option the grader adds is shown last, after tone's four options.
        assert examples["NA option index text"]["c3"] == (
            f"<selected_option>5</selected_option>\n<selected_label>{NA_LABEL}</selected_label>"
        )

    @pytest.mark.asyncio
    async def test_labels_are_rendered_as_what_they_name(self):
        """A verdict's value as that verdict, the NA label as the NA option the grader adds."""
        data, _ = self.pool()
        fake = RubricLLM()
        grader = per_item_grader(fake, data, few_shot(50, balance=False), shuffle_options=False)
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        examples = dict(shown_examples(call.user_prompt))
        assert examples["verdict values text"]["c0"] == "<verdict>UNMET</verdict>"
        assert examples["verdict values text"]["c2"] == "<verdict>MET</verdict>"
        assert examples["NA label text"] == {
            "c0": "<verdict>MET</verdict>",
            "c1": (
                f"<selected_option>4</selected_option>\n<selected_label>{NA_LABEL}</selected_label>"
            ),
            "c2": "<verdict>UNMET</verdict>",
            "c3": (
                f"<selected_option>5</selected_option>\n<selected_label>{NA_LABEL}</selected_label>"
            ),
        }

    @pytest.mark.asyncio
    async def test_an_item_fill_ground_truth_labels_with_the_na_option_is_an_example(self):
        """The label a judge's choice of the NA option the grader adds leaves in the ground
        truth names that option, so the item stays a whole example."""
        unlabelled = RubricDataset(prompt=QUERY, rubric=Rubric(TRAIN_RUBRIC))
        for submission in ("Leaves catch light.", "Roots drink water."):
            unlabelled.add_item(submission, submission)
        labeller = one_judge(RubricLLM({"c1": {"label": NA_LABEL}}), shuffle_options=False)
        labelled = await fill_ground_truth(unlabelled, labeller, show_progress=False)
        assert [item.ground_truth for item in labelled] == [[MET, NA_LABEL, MET, "Formal"]] * 2

        fake = RubricLLM()
        grader = per_item_grader(fake, labelled, few_shot(2), shuffle_options=False)
        assert names(picked(grader, "default")) != []
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        examples = dict(shown_examples(call.user_prompt))
        assert sorted(examples) == ["Leaves catch light.", "Roots drink water."]
        for judgments in examples.values():
            assert judgments["c1"] == (
                f"<selected_option>4</selected_option>\n<selected_label>{NA_LABEL}</selected_label>"
            )

    @pytest.mark.asyncio
    async def test_no_eligible_item_means_no_examples(self):
        dataset = RubricDataset(prompt=QUERY, rubric=Rubric(TRAIN_RUBRIC))
        dataset.add_item("unlabelled", "unlabelled")
        dataset.add_item("bad", "bad", ground_truth=[MET, "Crystal clear", UNMET, "Formal"])
        fake = RubricLLM()
        grader = per_item_grader(fake, dataset)
        assert grader._item_examples == {"default": []}

        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        assert "<examples>" not in call.user_prompt
        assert "</criteria>\n\n<input>" in call.user_prompt
        assert EXAMPLES_SENTENCE not in call.system_prompt
        # The guides keep their few-shot additions: they come with the training data.
        assert call.system_prompt == build_rubric_system_prompt(
            GRADER_SYSTEM_PROMPT_DEFAULT + FEW_SHOT_SYSTEM_PROMPT_ADDITION,
            MULTI_CHOICE_SYSTEM_PROMPT + MULTI_CHOICE_FEW_SHOT_ADDITION,
            with_examples=False,
        )

    @staticmethod
    def longer_rubric_pool() -> RubricDataset:
        """Items with their own rubric, one criterion longer than the training rubric: the
        extra criterion is one the training rubric lacks, so it counts as binary."""
        dataset = RubricDataset(prompt=QUERY, rubric=Rubric(TRAIN_RUBRIC))
        dataset.add_item("global text", "global", ground_truth=[MET, "Very clear", UNMET, "Formal"])
        dataset.add_item(
            "extra verdict text",
            "extra verdict",
            ground_truth=[UNMET, "Unclear", MET, "Casual", UNMET],
            rubric=Rubric([*TRAIN_RUBRIC, WATER]),
        )
        dataset.add_item(
            "extra option label text",
            "extra option label",
            ground_truth=[MET, "Mostly clear", UNMET, "Neutral", "Short"],
            rubric=Rubric([*TRAIN_RUBRIC, LENGTH]),
        )
        return dataset

    @pytest.mark.parametrize("balance", [True, False])
    def test_a_label_the_training_rubric_has_no_criterion_for_counts_as_binary(self, balance):
        """A verdict there keeps the item eligible; an option label, no verdict, does not."""
        data = self.longer_rubric_pool()
        grader = per_item_grader(RubricLLM(), data, few_shot(50, balance=balance))
        assert sorted(names(picked(grader, "default"))) == ["extra verdict", "global"]

    @pytest.mark.asyncio
    async def test_a_label_the_training_rubric_has_no_criterion_for_gets_no_judgment(self):
        """Even when the graded rubric has a binary criterion at that index."""
        fake = RubricLLM()
        grader = per_item_grader(fake, self.longer_rubric_pool(), few_shot(50, balance=False))
        await Rubric([*TRAIN_RUBRIC, WATER]).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        assert '<rubric_criterion id="c4">' in call.user_prompt
        examples = dict(shown_examples(call.user_prompt))
        assert set(examples) == {"global text", "extra verdict text"}
        assert list(examples["extra verdict text"]) == ["c0", "c1", "c2", "c3"]
        assert examples["extra verdict text"]["c0"] == "<verdict>UNMET</verdict>"


# A crafted pool for greedy coverage over [binary, binary, multi-choice X/Y/Z]. Its eight
# (criterion index, label) pairs are (0, MET/UNMET/CANNOT_ASSESS), (1, MET/UNMET) and
# (2, X/Y/Z). Items b and d share a submission; d alone shows (0, CANNOT_ASSESS).
COVERAGE_RUBRIC = [
    Criterion(weight=1.0, requirement="Mentions light"),
    Criterion(weight=1.0, requirement="Mentions water"),
    Criterion(
        weight=1.0,
        requirement="Which pigment is named?",
        scale_type="nominal",
        options=[
            CriterionOption(label="X", value=1.0),
            CriterionOption(label="Y", value=0.5),
            CriterionOption(label="Z", value=0.0),
        ],
    ),
]
COVERAGE_ROWS = {
    "a": ("A", [MET, MET, "X"]),
    "b": ("B", [UNMET, UNMET, "Y"]),
    "c": ("C", [MET, UNMET, "Z"]),
    "d": ("B", [CANNOT_ASSESS, MET, "Z"]),
    "e": ("E", [MET, MET, "X"]),
    "f": ("F", [UNMET, MET, "X"]),
    "g": ("G", [MET, MET, "Y"]),
}
COVERAGE_SEED = 15
# The pool's seeded order for COVERAGE_SEED and the "default" judge.
COVERAGE_ORDER = ["g", "b", "a", "f", "c", "d", "e"]


def coverage_pool() -> RubricDataset:
    dataset = RubricDataset(prompt=QUERY, rubric=Rubric(COVERAGE_RUBRIC))
    for name, (submission, labels) in COVERAGE_ROWS.items():
        dataset.add_item(submission, name, ground_truth=labels)
    return dataset


class TestGreedyCoverage:
    """Hand-derived picks on the crafted pool, whose seeded order is g b a f c d e.

    1. Every item shows 3 new pairs: a tie, won by g, the earliest.
       Shown: (0,MET) (1,MET) (2,Y).
    2. Gains: b 2, a 1, f 2, c 2, d 2, e 1. A tie among b, f, c and d, won by b.
       Shown: + (0,UNMET) (1,UNMET).
    3. d shares b's submission and is skipped, though it would show 2 new pairs,
       (0,CANNOT_ASSESS) and (2,Z), more than any other. Gains: a 1 (2,X), f 1 (2,X),
       c 1 (2,Z), e 1 (2,X). A tie, won by a. Shown: + (2,X).
    4. Gains: f 0, c 1 (2,Z), e 0: c. Shown: + (2,Z).
    5. No item left adds a pair (only d could), so the remaining slots are filled in the
       seeded order, skipping picked items and d's submission: f, then e.
    """

    def grader(self, n: int, balance: bool = True, judge_id: str = "default"):
        return make_grader(
            {judge_id: RubricLLM()},
            judges=[JudgeSpec(LLM, judge_id)],
            llm_calls="per_item",
            training_data=coverage_pool(),
            few_shot_config=few_shot(n, balance=balance),
            seed=COVERAGE_SEED,
        )

    def test_the_seeded_order(self):
        assert names(seeded_order(coverage_pool().items, COVERAGE_SEED, "default")) == (
            COVERAGE_ORDER
        )
        # Without balance the examples are that order, a shared submission included.
        grader = self.grader(len(COVERAGE_ROWS), balance=False)
        assert names(picked(grader, "default")) == COVERAGE_ORDER

    @pytest.mark.parametrize(
        "n, expected",
        [
            (1, ["g"]),
            (2, ["g", "b"]),
            (3, ["g", "b", "a"]),
            (4, ["g", "b", "a", "c"]),
            (5, ["g", "b", "a", "c", "f"]),
            (6, ["g", "b", "a", "c", "f", "e"]),
        ],
    )
    def test_the_greedy_picks(self, n, expected):
        assert names(picked(self.grader(n), "default")) == expected

    @pytest.mark.parametrize("n", [7, 50])
    def test_more_slots_than_items_shows_each_submission_once(self, n):
        picks = names(picked(self.grader(n), "default"))
        assert picks == ["g", "b", "a", "c", "f", "e"]
        assert "d" not in picks

    @pytest.mark.parametrize("n", [7, 50])
    def test_without_balance_more_slots_than_items_takes_them_all(self, n):
        picks = names(picked(self.grader(n, balance=False), "default"))
        assert picks == COVERAGE_ORDER

    def test_n_examples_zero_selects_nothing(self):
        assert self.grader(0)._item_examples["default"] == []
        assert self.grader(0, balance=False)._item_examples["default"] == []


# A crafted pool over [binary, multi-choice Unclear/Mostly clear/Very clear] (LIGHT, CLARITY)
# in which four items name the option "Very clear" each in its own way: three spellings of
# its label, and its index. Every item has its own submission. Listed in dataset order.
SPELLING_ROWS: dict[str, tuple[str, list[Any]]] = {
    "un": ("U", [UNMET, "Unclear"]),
    "mc": ("M", [MET, "Mostly clear"]),
    "ca": ("C", [CANNOT_ASSESS, "Very clear"]),
    "v4": ("V4", [MET, 2]),
    "v3": ("V3", [MET, "  VERY CLEAR "]),
    "v2": ("V2", [MET, "very clear"]),
    "v1": ("V1", [MET, "Very clear"]),
}
SPELLING_SEED = 1607
# The pool's seeded order for SPELLING_SEED and the "default" judge.
SPELLING_ORDER = ["v1", "v2", "v3", "v4", "ca", "mc", "un"]


def spelling_pool() -> RubricDataset:
    dataset = RubricDataset(prompt=QUERY, rubric=Rubric([LIGHT, CLARITY]))
    for name, (submission, labels) in SPELLING_ROWS.items():
        # An option index bypasses DataItem's validation (it takes labels and verdicts only):
        # it replaces an unresolvable placeholder once the item is added.
        dataset.add_item(
            submission,
            name,
            ground_truth=[label if isinstance(label, str) else "placeholder" for label in labels],
        )
        for criterion_idx, label in enumerate(labels):
            if not isinstance(label, str):
                set_label(dataset.items[-1], criterion_idx, label)
    return dataset


class TestCoverageIdentifiesOptionsByIndex:
    """A multi-choice pair's label is the option it resolves to, however it is named.

    The pairs: v1, v2, v3 and v4 each show (0,MET) (1,2), option 2 being "Very clear"; ca
    shows (0,CANNOT_ASSESS) (1,2), mc (0,MET) (1,1) and un (0,UNMET) (1,0). In the seeded
    order v1 v2 v3 v4 ca mc un:

    1. Every item shows 2 new pairs: a tie, won by v1. Shown: (0,MET) (1,2).
    2. Gains: v2, v3 and v4 0, as they show v1's option; ca 1, mc 1, un 2: un.
       Shown: + (0,UNMET) (1,0).
    3. Gains: ca 1 (0,CANNOT_ASSESS), mc 1 (1,1): a tie, won by ca.
    4. mc 1: mc. Every pair is shown.
    5. The remaining slots are filled in the seeded order: v2, v3, v4.
    """

    @staticmethod
    def grader(n: int, fake: RubricLLM | None = None, **kw: Any) -> CriterionGrader:
        return make_grader(
            {"default": fake or RubricLLM()},
            judge_model_config=LLM,
            llm_calls="per_item",
            training_data=spelling_pool(),
            few_shot_config=few_shot(n),
            seed=SPELLING_SEED,
            **kw,
        )

    def test_the_seeded_order(self):
        assert names(seeded_order(spelling_pool().items, SPELLING_SEED, "default")) == (
            SPELLING_ORDER
        )

    @pytest.mark.parametrize(
        "n, expected",
        [
            (1, ["v1"]),
            (2, ["v1", "un"]),
            (3, ["v1", "un", "ca"]),
            (4, ["v1", "un", "ca", "mc"]),
            (5, ["v1", "un", "ca", "mc", "v2"]),
            (7, ["v1", "un", "ca", "mc", "v2", "v3", "v4"]),
        ],
    )
    def test_the_greedy_picks(self, n, expected):
        assert names(picked(self.grader(n), "default")) == expected

    @pytest.mark.asyncio
    async def test_each_naming_is_shown_as_the_option(self):
        fake = RubricLLM()
        grader = self.grader(7, fake, shuffle_options=False)
        await Rubric([LIGHT, CLARITY]).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        examples = dict(shown_examples(call.user_prompt))
        very_clear = (
            "<selected_option>3</selected_option>\n<selected_label>Very clear</selected_label>"
        )
        for submission in ("V1", "V2", "V3", "V4", "C"):
            assert examples[submission]["c1"] == very_clear
        assert examples["C"]["c0"] == "<verdict>CANNOT_ASSESS</verdict>"


# =============================================================================
# Rendering
# =============================================================================


class TestRendering:
    @pytest.mark.asyncio
    async def test_the_examples_block_written_out(self):
        """Shuffling off, so option numbers are the training rubric's; reasons shown."""
        fake = RubricLLM()
        grader = per_item_grader(
            fake, config=few_shot(2, balance=False, include_reason=True), shuffle_options=False
        )
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls

        written = {
            "Plants use sunlight to make sugar.": (
                '<judgment id="c0">\n<verdict>MET</verdict>\n<reason>Names sunlight.</reason>\n'
                "</judgment>\n"
                '<judgment id="c1">\n<selected_option>3</selected_option>\n'
                "<selected_label>Very clear</selected_label>\n</judgment>\n"
                '<judgment id="c2">\n<verdict>UNMET</verdict>\n</judgment>\n'
                '<judgment id="c3">\n<selected_option>1</selected_option>\n'
                "<selected_label>Formal</selected_label>\n<reason>Plain register.</reason>\n"
                "</judgment>"
            ),
            "Plants eat dirt, lol.": (
                '<judgment id="c0">\n<verdict>UNMET</verdict>\n</judgment>\n'
                '<judgment id="c1">\n<selected_option>1</selected_option>\n'
                "<selected_label>Unclear</selected_label>\n</judgment>\n"
                '<judgment id="c2">\n<verdict>UNMET</verdict>\n</judgment>\n'
                '<judgment id="c3">\n<selected_option>3</selected_option>\n'
                "<selected_label>Playful</selected_label>\n</judgment>"
            ),
            "Chlorophyll drives photolysis!": (
                '<judgment id="c0">\n<verdict>MET</verdict>\n</judgment>\n'
                '<judgment id="c1">\n<selected_option>2</selected_option>\n'
                "<selected_label>Mostly clear</selected_label>\n<reason>Some terms.</reason>\n"
                "</judgment>\n"
                '<judgment id="c2">\n<verdict>MET</verdict>\n'
                "<reason>Photolysis is unexplained.</reason>\n</judgment>\n"
                '<judgment id="c3">\n<selected_option>4</selected_option>\n'
                "<selected_label>Neutral</selected_label>\n</judgment>"
            ),
            "Light and water, basically.": (
                '<judgment id="c0">\n<verdict>MET</verdict>\n<reason>Light.</reason>\n'
                "</judgment>\n"
                '<judgment id="c1">\n<selected_option>2</selected_option>\n'
                "<selected_label>Mostly clear</selected_label>\n<reason>Clear enough.</reason>\n"
                "</judgment>\n"
                '<judgment id="c2">\n<verdict>UNMET</verdict>\n<reason>No jargon.</reason>\n'
                "</judgment>\n"
                '<judgment id="c3">\n<selected_option>2</selected_option>\n'
                "<selected_label>Casual</selected_label>\n<reason>Informal.</reason>\n"
                "</judgment>"
            ),
        }
        chosen = [s for s, _, _ in grader._item_examples["default"]]
        assert len(chosen) == 2
        block = "\n".join(
            [
                "<examples>",
                *(
                    f"<example_{n}>\n<example_submission>{submission}</example_submission>\n"
                    f"{written[submission]}\n</example_{n}>"
                    for n, submission in enumerate(chosen, 1)
                ),
                "</examples>",
            ]
        )
        assert f"</criteria>\n\n{block}\n\n<input>{QUERY}</input>\n\n<submission>\n" in (
            call.user_prompt
        )

    @pytest.mark.asyncio
    async def test_one_block_between_the_criteria_and_the_input(self):
        fake = RubricLLM()
        grader = per_item_grader(fake)
        await Rubric(TRAIN_RUBRIC, guidelines=GUIDELINES).grade(
            SUBMISSION, grader=grader, query=QUERY, reference_submission=REFERENCE
        )
        (call,) = fake.calls
        prompt = call.user_prompt
        assert call.response_format is RubricJudgment
        assert prompt.count("<examples>") == prompt.count("</examples>") == 1
        assert prompt.startswith("<guidelines>\n")
        assert prompt.index("</criteria>") < prompt.index("<examples>")
        assert "</criteria>\n\n<examples>\n<example_1>\n" in prompt
        assert (
            f"</example_4>\n</examples>\n\n<input>{QUERY}</input>\n\n"
            f"<reference_submission>\n{REFERENCE}\n</reference_submission>\n\n"
            f"<submission>\n{SUBMISSION}\n</submission>"
        ) in prompt
        assert prompt.endswith(f"<submission>\n{SUBMISSION}\n</submission>")

    @pytest.mark.parametrize(
        "shuffle_options, auto_na_option",
        [(True, True), (True, False), (False, True), (False, False)],
        ids=["shuffled", "shuffled_forced_choice", "unshuffled", "unshuffled_forced_choice"],
    )
    @pytest.mark.parametrize("include_reason", [False, True], ids=["no_reason", "reason"])
    @pytest.mark.asyncio
    async def test_each_criterion_is_judged_in_the_calls_numbering(
        self, shuffle_options, auto_na_option, include_reason
    ):
        fake = RubricLLM()
        grader = per_item_grader(
            fake,
            config=few_shot(4, include_reason=include_reason),
            shuffle_options=shuffle_options,
            auto_na_option=auto_na_option,
        )
        for submission in (SUBMISSION, OTHER_SUBMISSION):
            await Rubric(TRAIN_RUBRIC).grade(submission, grader=grader, query=QUERY)
        assert len(fake.calls) == 2
        for call in fake.calls:
            assert_examples_match(call.user_prompt, grader, "default", include_reason)
            if not include_reason:
                assert "<reason>" not in call.user_prompt

    @pytest.mark.asyncio
    async def test_the_numbering_follows_the_shuffle_not_the_training_rubric(self):
        """Guards the test above: with shuffling on, some example's option number differs
        from its option's position in the training rubric."""
        fake = RubricLLM()
        grader = per_item_grader(fake)
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        renumbered = False
        for submission, judgments in shown_examples(call.user_prompt):
            labels, _ = TRAINING[submission]
            for idx in (1, 3):
                original = [o.label for o in TRAIN_RUBRIC[idx].options or []].index(labels[idx])
                renumbered |= selected_option(judgments[f"c{idx}"]) != original + 1
        assert renumbered

    @pytest.mark.asyncio
    async def test_the_examples_are_the_same_for_every_item(self):
        fake = RubricLLM()
        grader = per_item_grader(fake, config=few_shot(3))
        for submission in (SUBMISSION, OTHER_SUBMISSION):
            await Rubric(TRAIN_RUBRIC).grade(submission, grader=grader, query=QUERY)
        first, second = ([s for s, _ in shown_examples(c.user_prompt)] for c in fake.calls)
        assert first == second == [s for s, _, _ in grader._item_examples["default"]]

    @pytest.mark.asyncio
    async def test_changing_the_training_items_afterwards_changes_no_prompt(self):
        """The examples are kept as the training items were when the grader was built, as
        per-criterion examples are: a later change to an item's submission, labels or
        reasons, even one that leaves a label unusable, shows in no prompt."""
        data = training_data()
        fakes = {"per_item": RubricLLM(), "per_criterion": RubricLLM()}
        config = few_shot(4, include_reason=True)
        per_item = per_item_grader(fakes["per_item"], data, config)
        per_criterion = make_grader(
            {"default": fakes["per_criterion"]},
            judge_model_config=LLM,
            training_data=data,
            few_shot_config=config,
        )
        for grader in (per_item, per_criterion):
            await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        for item in data.items:
            labels = cast(list[Any], item.ground_truth)
            item.submission = f"{item.submission} (edited)"
            set_label(item, 0, UNMET if labels[0] == MET else MET)
            set_label(item, 1, "Crystal clear")
            item.ground_truth_reasons = ["Edited."] * len(labels)
        for grader in (per_item, per_criterion):
            await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        for mode, fake in fakes.items():
            half = len(fake.calls) // 2
            before, after = fake.calls[:half], fake.calls[half:]
            assert [c.user_prompt for c in after] == [c.user_prompt for c in before], mode
        (call, _) = fakes["per_item"].calls
        assert "(edited)" not in call.user_prompt and "Edited." not in call.user_prompt

    @pytest.mark.parametrize("include_reason", [False, True])
    @pytest.mark.asyncio
    async def test_reasons_only_with_include_reason_and_only_when_given(self, include_reason):
        fake = RubricLLM()
        grader = per_item_grader(fake, config=few_shot(4, include_reason=include_reason))
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        shown = re.findall(r"<reason>(.*?)</reason>", call.user_prompt)
        every_reason = {
            reason
            for _, _, reasons in TRAIN_ROWS
            for reason in reasons or []
            if reason  # None and "" are never shown
        }
        if include_reason:
            assert sorted(shown) == sorted(every_reason)
        else:
            assert shown == []

    @pytest.mark.parametrize(
        "target",
        [
            "autorubric.graders.criterion_grader._format_rubric_examples",
            "autorubric.graders.criterion_grader.CriterionGrader._rubric_examples",
        ],
        ids=["format", "select"],
    )
    @pytest.mark.asyncio
    async def test_an_error_rendering_the_examples_raises_before_the_call(
        self, monkeypatch, target
    ):
        """The examples are rendered with the prompts, outside the call's failure handling:
        an error there raises, instead of failing every criterion as a failed call would."""

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("rendering failed")

        monkeypatch.setattr(target, fail)
        fake = RubricLLM()
        grader = per_item_grader(fake)
        with pytest.raises(ValueError, match="rendering failed"):
            await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        assert fake.calls == []


class TestGradedRubricDiffersFromTraining:
    """A judgment appears only for a criterion index both rubrics have, with a criterion of
    the same kind, and (multi-choice) whose option the graded criterion has."""

    @staticmethod
    async def prompt(rubric: list[Criterion]) -> tuple[str, str]:
        fake = RubricLLM()
        grader = per_item_grader(fake, shuffle_options=False, auto_na_option=False)
        await Rubric(rubric).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        return call.system_prompt, call.user_prompt

    @pytest.mark.asyncio
    async def test_a_criterion_the_training_rubric_lacks_gets_no_judgment(self):
        _, prompt = await self.prompt([*TRAIN_RUBRIC, WATER])
        assert '<rubric_criterion id="c4">' in prompt
        examples = shown_examples(prompt)
        assert len(examples) == 4
        for _, judgments in examples:
            assert list(judgments) == ["c0", "c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_a_shorter_graded_rubric_is_judged_on_its_criteria(self):
        _, prompt = await self.prompt([LIGHT, CLARITY])
        for _, judgments in shown_examples(prompt):
            assert list(judgments) == ["c0", "c1"]

    @pytest.mark.asyncio
    async def test_a_criterion_of_another_kind_gets_no_judgment(self):
        binary_at_1 = Criterion(name="clear", weight=1.0, requirement="Is clear")
        multi_choice_at_2 = Criterion(
            name="jargon level",
            weight=1.0,
            requirement="How much jargon?",
            options=[
                CriterionOption(label="None", value=1.0),
                CriterionOption(label="Some", value=0.5),
            ],
        )
        _, prompt = await self.prompt([LIGHT, binary_at_1, multi_choice_at_2, TONE])
        for submission, judgments in shown_examples(prompt):
            assert list(judgments) == ["c0", "c3"]
            labels, _ = TRAINING[submission]
            assert judgments["c0"] == f"<verdict>{labels[0].value}</verdict>"

    @pytest.mark.asyncio
    async def test_an_option_the_graded_criterion_lacks_gets_no_judgment(self):
        two_options = Criterion(
            name="clarity",
            weight=2.0,
            requirement="How clear is the explanation?",
            scale_type="ordinal",
            options=[
                CriterionOption(label="Unclear", value=0.0),
                CriterionOption(label="Mostly clear", value=1.0),
            ],
        )
        _, prompt = await self.prompt([LIGHT, two_options, JARGON, TONE])
        examples = shown_examples(prompt)
        # Only the judgment is left out, never the example: all four items are still shown,
        # the one labelled "Very clear" (an option the graded criterion lacks) among them.
        assert len(examples) == 4
        assert list(dict(examples)["Plants use sunlight to make sugar."]) == ["c0", "c2", "c3"]
        for submission, judgments in examples:
            labels, _ = TRAINING[submission]
            if labels[1] == "Very clear":
                assert list(judgments) == ["c0", "c2", "c3"]
            else:
                assert list(judgments) == ["c0", "c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_no_judgment_left_means_no_examples(self):
        multi_choice_at_0 = Criterion(
            name="light level",
            weight=1.0,
            requirement="How much light is mentioned?",
            options=[
                CriterionOption(label="None", value=0.0),
                CriterionOption(label="Some", value=1.0),
            ],
        )
        system_prompt, prompt = await self.prompt([multi_choice_at_0])
        assert "<examples>" not in prompt
        assert "<judgment" not in prompt
        assert "</criteria>\n\n<input>" in prompt
        assert EXAMPLES_SENTENCE not in system_prompt
        assert system_prompt == build_rubric_system_prompt(
            None, MULTI_CHOICE_SYSTEM_PROMPT + MULTI_CHOICE_FEW_SHOT_ADDITION
        )


class TestSystemPrompt:
    @pytest.mark.asyncio
    async def test_with_examples_it_says_how_to_read_them(self):
        fake = RubricLLM()
        grader = per_item_grader(fake)
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        assert RUBRIC_JUDGMENT_EXAMPLES == EXAMPLES_SENTENCE
        assert call.system_prompt.count(EXAMPLES_SENTENCE) == 1
        # Its own paragraph, right after the independence paragraph.
        assert (
            "<input> and <reference_submission>, apply to every criterion.\n\n"
            f"{EXAMPLES_SENTENCE}\n\nA guide's RESPONSE FORMAT"
        ) in call.system_prompt
        # The guides are the grader's system prompts with their few-shot additions.
        assert call.system_prompt == build_rubric_system_prompt(
            GRADER_SYSTEM_PROMPT_DEFAULT + FEW_SHOT_SYSTEM_PROMPT_ADDITION,
            MULTI_CHOICE_SYSTEM_PROMPT + MULTI_CHOICE_FEW_SHOT_ADDITION,
            with_examples=True,
        )
        assert (
            "<binary_criterion_guide>\n"
            + GRADER_SYSTEM_PROMPT_DEFAULT
            + FEW_SHOT_SYSTEM_PROMPT_ADDITION
            + "\n</binary_criterion_guide>"
        ) in call.system_prompt
        assert (
            "<multi_choice_criterion_guide>\n"
            + MULTI_CHOICE_SYSTEM_PROMPT
            + MULTI_CHOICE_FEW_SHOT_ADDITION
            + "\n</multi_choice_criterion_guide>"
        ) in call.system_prompt

    @pytest.mark.asyncio
    async def test_custom_guides_are_embedded_as_given(self):
        fake = RubricLLM()
        grader = per_item_grader(
            fake, system_prompt="Custom binary guide.", multi_choice_system_prompt="Custom MC."
        )
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        assert call.system_prompt == build_rubric_system_prompt(
            "Custom binary guide.", "Custom MC.", with_examples=True
        )
        assert "<examples>" in call.user_prompt

    @pytest.mark.asyncio
    async def test_without_training_data_there_is_no_examples_sentence(self):
        fake = RubricLLM()
        grader = make_grader({"default": fake}, judge_model_config=LLM, llm_calls="per_item")
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        assert "<examples>" not in call.user_prompt
        assert call.system_prompt == build_rubric_system_prompt(
            GRADER_SYSTEM_PROMPT_DEFAULT, MULTI_CHOICE_SYSTEM_PROMPT
        )


# =============================================================================
# Panels, cost and cascades
# =============================================================================


class TestPanel:
    @pytest.mark.asyncio
    async def test_each_judge_shows_its_own_examples_in_its_own_numbering(self):
        fakes = {"alpha": RubricLLM(), "beta": RubricLLM()}
        grader = make_grader(
            fakes,
            judges=PANEL,
            llm_calls="per_item",
            training_data=training_data(),
            few_shot_config=few_shot(2, include_reason=True),
        )
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        orders = {}
        for judge_id, fake in fakes.items():
            (call,) = fake.calls
            assert_examples_match(call.user_prompt, grader, judge_id, include_reason=True)
            orders[judge_id] = [s for s, _ in shown_examples(call.user_prompt)]
        assert orders["alpha"] != orders["beta"]


class TestCost:
    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.asyncio
    async def test_each_call_carries_k_example_submissions_once(self, k):
        fake = RubricLLM()
        grader = per_item_grader(fake, config=few_shot(k))
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        (call,) = fake.calls
        submissions = re.findall(
            r"<example_submission>(.*?)</example_submission>", call.user_prompt
        )
        assert len(submissions) == len(set(submissions)) == k

    @pytest.mark.asyncio
    async def test_per_criterion_carries_k_per_criterion(self):
        """The contrast: called per criterion, an item carries N x k example submissions."""
        k = 2
        fake = RubricLLM()
        grader = make_grader(
            {"default": fake},
            judge_model_config=LLM,
            training_data=training_data(),
            few_shot_config=few_shot(k),
        )
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        assert len(fake.calls) == len(TRAIN_RUBRIC)
        total = sum(c.user_prompt.count("<example_submission>") for c in fake.calls)
        assert total == len(TRAIN_RUBRIC) * k


# A decision model sure of c0, c2 and c3 and unsure of c1 (confidence 0.6 < 0.7), so only
# c1 escalates.
DM_ANSWERS = {
    "c0": {"type": "noul", "noul": 0.95},
    "c1": {
        "type": "choice",
        "choice": "Mostly clear",
        "confidence": 0.6,
        "probabilities": {"Unclear": 0.1, "Mostly clear": 0.7, "Very clear": 0.15, NA_LABEL: 0.05},
    },
    "c2": {"type": "noul", "noul": 0.05},
    "c3": {
        "type": "choice",
        "choice": "Formal",
        "confidence": 0.9,
        "probabilities": {
            "Formal": 0.92,
            "Casual": 0.02,
            "Playful": 0.02,
            "Neutral": 0.02,
            NA_LABEL: 0.02,
        },
    },
}


@pytest.fixture
def decision_model(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    sdk = FakeSDK(response=make_response(DM_ANSWERS))
    monkeypatch.setattr(
        typesafe_sdk, "AsyncTypeSafeClient", lambda **kwargs: FakeSDKClient(sdk, **kwargs)
    )
    return sdk


class TestCascade:
    @pytest.mark.parametrize("include_reason", [False, True])
    @pytest.mark.asyncio
    async def test_the_escalation_prompt_equals_a_plain_per_item_graders(
        self, decision_model, include_reason
    ):
        config = few_shot(3, include_reason=include_reason)
        escalation_fake, plain_fake = RubricLLM(), RubricLLM()
        escalation_judge = JudgeSpec(LLM, "esc")
        cascade = make_grader(
            {"esc": escalation_fake},
            judge_model_config=DecisionModelConfig(model="jev-latest", api_key=API_KEY),
            escalation=EscalationConfig(judges=[escalation_judge], threshold=0.7),
            llm_calls="per_item",
            training_data=training_data(),
            few_shot_config=config,
        )
        plain = make_grader(
            {"esc": plain_fake},
            judges=[escalation_judge],
            llm_calls="per_item",
            training_data=training_data(),
            few_shot_config=config,
        )
        assert list(cascade._item_examples) == ["esc"]
        assert names(picked(cascade, "esc")) == names(picked(plain, "esc"))

        report = await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=cascade, query=QUERY)
        await Rubric(TRAIN_RUBRIC).grade(SUBMISSION, grader=plain, query=QUERY)

        assert len(decision_model.calls) == 1
        assert report.report is not None
        assert [c.escalated for c in report.report] == [False, True, False, False]
        assert len(escalation_fake.calls) == len(plain_fake.calls) == 1
        assert escalation_fake.calls == plain_fake.calls
        (call,) = escalation_fake.calls
        assert_examples_match(call.user_prompt, cascade, "esc", include_reason)
        assert EXAMPLES_SENTENCE in call.system_prompt
