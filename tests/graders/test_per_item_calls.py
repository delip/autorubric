"""One LLM call per item: ``CriterionGrader(llm_calls="per_item")``.

Under ``llm_calls="per_item"`` each LLM judge grades an item's whole rubric in one call: the
system prompt embeds the grader's binary and multi-choice system prompts, for the kinds of
criteria the rubric has, between a preamble and a batched response format; the user prompt
lists every criterion under its id (``c0``, ``c1``, ...) exactly as a per-criterion prompt
poses it, options shuffled by the same key; and the answer (``RubricJudgment``) maps back to
one result per criterion. A missing or unusable answer fails only its criterion (``parse``),
a failed call fails every criterion, and the call's usage and cost ride on the first result.
The default, ``"per_criterion"``, is today's behaviour, unchanged.

Nothing here reaches the network. LLM judges are ``RubricLLM`` fakes patched in for
``LLMClient``: for ``RubricJudgment`` they build a reply from the ``id="cK"`` attributes of the
prompt and run it through the real validation (``RubricJudgment.model_validate``); a
``ParsedLLM`` fake answers every call with one given parsed reply (a custom response format,
say). The end-to-end tests keep the real ``LLMClient`` and patch ``litellm.acompletion``.
Decision-model requests go to the recording fake SDK client of the decision-model tests.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
import warnings
from pathlib import Path
from typing import Any, cast, get_args
from unittest.mock import patch

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
from litellm import ModelResponse
from pydantic import BaseModel
from test_error_routing import FAILURES

import autorubric
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
    TokenUsage,
    evaluate,
)
from autorubric.eval import _serialize_grader_config
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.prompts import GRADER_SYSTEM_PROMPT_DEFAULT, MULTI_CHOICE_SYSTEM_PROMPT
from autorubric.types import (
    CANONICAL_NA_OPTION,
    CriterionJudgment,
    MultiChoiceJudgment,
    RubricJudgment,
)

LLM_CLIENT = "autorubric.graders.criterion_grader.LLMClient"
API_KEY = "ts-test-key-never-persisted-5a1d"
SEED = 11
LLM = LLMConfig(model="test-model")
NA_LABEL = CANONICAL_NA_OPTION.label
QUERY = "How do plants make food?"
REFERENCE = "Photosynthesis turns light, water and carbon dioxide into glucose and oxygen."
SUBMISSION = "Plants use light to turn water and carbon dioxide into sugar."
OTHER_SUBMISSION = "Plants eat soil."
USAGE = TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140)
COST = 0.002

LIGHT = Criterion(name="light", weight=5.0, requirement="Mentions light")
JARGON = Criterion(name="jargon", weight=-2.0, requirement="Uses unexplained jargon")
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
# Unnamed, with an NA option of its own (so no NA option is added).
DEPTH = Criterion(
    weight=1.0,
    requirement="How deep is the answer?",
    options=[
        CriterionOption(label="Shallow", value=0.0),
        CriterionOption(label="Adequate", value=0.5),
        CriterionOption(label="Deep", value=1.0),
        CriterionOption(label="Not applicable", value=0.0, na=True),
    ],
)
BINARY = [LIGHT, JARGON]
MULTI_CHOICE = [CLARITY, TONE]
# c0 light, c1 clarity, c2 jargon, c3 tone.
MIXED = [LIGHT, CLARITY, JARGON, TONE]

GUIDELINES = (
    "Writers are English-language learners in grades 8-12; judge against that level.\n"
    "'Cited' means any attribution, e.g. {author, year}, not formal citation style."
)
# Written out literally, as in tests/test_prompt_goldens.py, so any change to the block fails.
GUIDELINES_PREFIX = (
    "<guidelines>\n"
    "Writers are English-language learners in grades 8-12; judge against that level.\n"
    "'Cited' means any attribution, e.g. {author, year}, not formal citation style.\n"
    "\n"
    "These guidelines apply to every criterion. The criterion text governs; the guidelines "
    "clarify how to apply it.\n"
    "</guidelines>\n"
    "\n"
)

DM_ANSWERS = {
    "c0": {"type": "noul", "noul": 0.9},
    "c1": {
        "type": "choice",
        "choice": "Mostly clear",
        "confidence": 0.6,
        "probabilities": {"Unclear": 0.1, "Mostly clear": 0.7, "Very clear": 0.15, NA_LABEL: 0.05},
    },
}


# =============================================================================
# A recording LLM judge that answers RubricJudgment per criterion id
# =============================================================================

DROP = "drop"
"""An answer spec: the reply has no judgment for the criterion."""

RUBRIC_CRITERION = re.compile(
    r'<rubric_criterion id="(?P<id>[^"]+)">\n(?P<body>.*?)\n</rubric_criterion>', re.S
)


def rubric_blocks(user_prompt: str) -> dict[str, str]:
    """Each ``<rubric_criterion>``'s content by its id, in the order the prompt lists them."""
    return {m["id"]: m["body"] for m in RUBRIC_CRITERION.finditer(user_prompt)}


def presented_labels(block: str) -> list[str]:
    """The option labels of a multi-choice block, in the order the judge is shown them."""
    options = block.split("<options>\n", 1)[1].split("\n</options>", 1)[0]
    return [
        re.sub(r" \(cannot assess / not applicable\)$", "", line.split(". ", 1)[1])
        for line in options.splitlines()
    ]


@dataclasses.dataclass
class Call:
    system_prompt: str
    user_prompt: str
    response_format: Any


class RubricLLM:
    """Stand-in for ``LLMClient`` that records every call.

    For ``RubricJudgment`` it answers every ``<rubric_criterion>`` of the prompt, in order: a
    binary criterion ``MET`` and a multi-choice one its first presented option, each with the
    explanation ``"judged cK"``. ``answers`` changes that per id: ``DROP`` leaves the id out,
    a list is the id's raw entries verbatim (duplicates, unusable entries), and a dict is
    merged over the default entry (``{"label": ...}`` picks that option by its label, in the
    order the prompt presents it). ``extra`` entries are appended; ``reverse`` lists the
    entries backwards; ``raw`` replaces the whole reply. The reply then goes through the real
    validation, ``RubricJudgment.model_validate``, with ``reasoning`` injected at the top level
    as ``LLMClient.generate`` does. Per-criterion calls are answered ``MET`` or the first
    option. ``error`` is raised by every call, or only by those about ``fail_on``.
    """

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        *,
        extra: list[Any] | None = None,
        reverse: bool = False,
        raw: Any = None,
        reasoning: str | None = None,
        error: BaseException | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.answers = answers or {}
        self.extra = extra or []
        self.reverse = reverse
        self.raw = raw
        self.reasoning = reasoning
        self.error = error
        self.fail_on = fail_on
        self.calls: list[Call] = []

    def reply(self, user_prompt: str) -> dict[str, Any]:
        entries: list[Any] = []
        for criterion_id, block in rubric_blocks(user_prompt).items():
            spec = self.answers.get(criterion_id)
            if spec == DROP:
                continue
            if isinstance(spec, list):
                entries.extend(spec)
                continue
            multi_choice = "<options>" in block
            entry: dict[str, Any] = {
                "criterion_id": criterion_id,
                "criterion_status": None if multi_choice else "MET",
                "selected_option": 1 if multi_choice else None,
                "explanation": f"judged {criterion_id}",
            }
            spec = dict(spec or {})
            if "label" in spec:
                spec["selected_option"] = presented_labels(block).index(spec.pop("label")) + 1
            entry.update(spec)
            entries.append(entry)
        if self.reverse:
            entries.reverse()
        return {"judgments": [*entries, *self.extra]}

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Any = None,
        **kwargs: Any,
    ) -> GenerateResult:
        self.calls.append(Call(system_prompt, user_prompt, response_format))
        if self.error is not None and (self.fail_on is None or self.fail_on in user_prompt):
            raise self.error
        parsed: Any
        if response_format is RubricJudgment:
            data = self.raw if self.raw is not None else self.reply(user_prompt)
            if self.reasoning is not None:
                # As LLMClient.generate injects the trace: into the reply's object.
                data = {**data, "reasoning": self.reasoning}
            parsed = RubricJudgment.model_validate(data)
        elif response_format is MultiChoiceJudgment:
            parsed = MultiChoiceJudgment(selected_option=1, explanation="first")
        else:
            parsed = CriterionJudgment(criterion_status=CriterionVerdict.MET, explanation="present")
        return GenerateResult(
            content="{}",
            thinking=self.reasoning,
            raw_response=None,
            usage=USAGE,
            cost=COST,
            parsed=parsed,
        )


class ParsedLLM:
    """Stand-in for ``LLMClient`` whose every call is answered with ``parsed`` as it is,
    for replies a ``RubricLLM`` does not build (custom response formats, a per-criterion
    judgment carrying a thinking trace)."""

    def __init__(self, parsed: Any) -> None:
        self.parsed = parsed
        self.calls: list[Call] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Any = None,
        **kwargs: Any,
    ) -> GenerateResult:
        self.calls.append(Call(system_prompt, user_prompt, response_format))
        return GenerateResult(content="{}", usage=USAGE, cost=COST, parsed=self.parsed)


def make_grader(fakes: dict[str, Any], **kwargs: Any) -> CriterionGrader:
    """A ``CriterionGrader`` (seed ``SEED``) whose LLM judges are ``fakes``, by judge id."""
    kwargs.setdefault("seed", SEED)

    def client(config: Any, cache_namespace: str | None = None) -> Any:
        return fakes[cache_namespace or "default"]

    with patch(LLM_CLIENT, side_effect=client):
        return CriterionGrader(**kwargs)


def one_judge(fake: RubricLLM, **kwargs: Any) -> CriterionGrader:
    return make_grader({"default": fake}, judge_model_config=LLM, llm_calls="per_item", **kwargs)


def dm() -> DecisionModelConfig:
    return DecisionModelConfig(model="jev-latest", api_key=API_KEY)


def build(**kwargs: Any) -> tuple[CriterionGrader, list[warnings.WarningMessage]]:
    """Build a grader, returning it with every ``UserWarning`` its construction issued."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grader = CriterionGrader(**kwargs)
    return grader, [w for w in caught if w.category is UserWarning]


def request_json(call: tuple[Any, Any, dict[str, Any]]) -> str:
    """One recorded decision-model request (state and questions) as JSON; key order counts."""
    state, questions, _ = call
    wire = {qid: question.model_dump(mode="json") for qid, question in questions.items()}
    return json.dumps([state, wire])


@pytest.fixture
def decision_model(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    sdk = FakeSDK(response=make_response(DM_ANSWERS))
    monkeypatch.setattr(
        typesafe_sdk, "AsyncTypeSafeClient", lambda **kwargs: FakeSDKClient(sdk, **kwargs)
    )
    return sdk


def effective(criterion: Criterion) -> Criterion:
    """The criterion as a default grader judges it (its NA option guaranteed)."""
    return criterion.with_guaranteed_na_option()


# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    def test_the_default_is_per_criterion(self):
        grader = CriterionGrader(judge_model_config=LLM)
        assert grader._llm_calls == "per_criterion"
        parameter = inspect.signature(CriterionGrader.__init__).parameters["llm_calls"]
        assert parameter.default == "per_criterion"
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_parameter_sits_between_escalation_and_the_deprecated_alias(self):
        names = list(inspect.signature(CriterionGrader.__init__).parameters)
        assert names[names.index("escalation") + 1] == "llm_calls"
        assert names[-1] == "llm_config"

    def test_llm_calls_is_exported_with_its_two_values(self):
        assert get_args(autorubric.LLMCalls) == ("per_criterion", "per_item")
        assert "LLMCalls" in autorubric.__all__

    def test_per_item_is_stored(self):
        grader = CriterionGrader(judge_model_config=LLM, llm_calls="per_item")
        assert grader._llm_calls == "per_item"

    @pytest.mark.parametrize("value", ["per_call", "PER_ITEM", None, ""])
    def test_an_invalid_value_raises(self, value):
        expected = f"llm_calls must be one of ('per_criterion', 'per_item'); got {value!r}"
        invalid: Any = value
        with pytest.raises(ValueError) as caught:
            CriterionGrader(judge_model_config=LLM, llm_calls=invalid)
        assert str(caught.value) == expected

    def test_an_invalid_value_is_checked_before_the_response_formats(self):
        invalid = cast(Any, "per_call")
        with pytest.raises(ValueError, match=r"^llm_calls must be one of"):
            CriterionGrader(
                judge_model_config=dm(),
                binary_response_format=CriterionJudgment,
                llm_calls=invalid,
            )

    @pytest.mark.parametrize(
        "formats, named",
        [
            ({"binary_response_format": CriterionJudgment}, "binary_response_format"),
            (
                {"multi_choice_response_format": MultiChoiceJudgment},
                "multi_choice_response_format",
            ),
            (
                {
                    "binary_response_format": CriterionJudgment,
                    "multi_choice_response_format": MultiChoiceJudgment,
                },
                "binary_response_format and multi_choice_response_format",
            ),
        ],
        ids=["binary", "multi_choice", "both"],
    )
    def test_custom_response_formats_cannot_be_combined_with_per_item(self, formats, named):
        with pytest.raises(ValueError) as caught:
            CriterionGrader(judge_model_config=LLM, llm_calls="per_item", **formats)
        assert str(caught.value) == (
            f"{named} cannot be combined with llm_calls='per_item': a response format "
            "describes one criterion's judgment, and a per_item call answers every criterion "
            "at once"
        )

    def test_custom_response_formats_still_work_per_criterion(self):
        grader = CriterionGrader(
            judge_model_config=LLM,
            llm_calls="per_criterion",
            binary_response_format=CriterionJudgment,
            multi_choice_response_format=MultiChoiceJudgment,
        )
        assert grader._llm_calls == "per_criterion"

    def test_with_a_decision_model_the_response_format_message_is_unchanged(self):
        with pytest.raises(ValueError) as without_setting:
            CriterionGrader(judge_model_config=dm(), binary_response_format=CriterionJudgment)
        with pytest.raises(ValueError) as with_setting:
            CriterionGrader(
                judge_model_config=dm(),
                binary_response_format=CriterionJudgment,
                llm_calls="per_item",
            )
        assert (
            str(with_setting.value)
            == str(without_setting.value)
            == (
                "binary_response_format cannot be combined with decision-model judges "
                "('default'): a response format describes a generated judgment, which only an "
                "LLM judge produces"
            )
        )

    @pytest.mark.parametrize(
        "panel",
        [
            {"judge_model_config": dm()},
            {"judges": [JudgeSpec(dm(), "jev-a"), JudgeSpec(dm(), "jev-b")]},
        ],
        ids=["decision_model", "decision_models"],
    )
    def test_an_all_decision_model_grader_warns(self, panel):
        grader, caught = build(**panel, llm_calls="per_item")
        assert [str(w.message) for w in caught] == [
            "llm_calls applies to LLM judges only; this grader has none (every judge is a "
            "decision model), so it has no effect"
        ]
        # Attributed to the line that built the grader (stacklevel), not the library.
        assert caught[0].filename == __file__
        assert grader._llm_calls == "per_item"

    @pytest.mark.parametrize(
        "panel",
        [
            {"judge_model_config": LLM},
            {"judges": [JudgeSpec(dm(), "jev"), JudgeSpec(LLM, "llm")]},
            {"judge_model_config": dm(), "escalation": EscalationConfig(judges=LLM, threshold=0.7)},
        ],
        ids=["llm", "mixed", "cascade"],
    )
    def test_a_grader_with_an_llm_judge_does_not_warn(self, panel):
        _, caught = build(**panel, llm_calls="per_item")
        assert caught == []

    def test_per_item_is_accepted_with_a_cascade(self):
        grader, caught = build(
            judge_model_config=dm(),
            escalation=EscalationConfig(judges=LLM, threshold=0.7),
            llm_calls="per_item",
        )
        assert caught == []
        assert grader._llm_calls == "per_item"

    def test_per_item_is_accepted_with_training_data(self):
        training = RubricDataset(prompt=QUERY, rubric=Rubric([LIGHT]))
        training.add_item("Light matters.", "a", ground_truth=[CriterionVerdict.MET])
        training.add_item("Water matters.", "b", ground_truth=[CriterionVerdict.UNMET])
        grader = CriterionGrader(
            judge_model_config=LLM,
            training_data=training,
            few_shot_config=FewShotConfig(n_examples=1),
            llm_calls="per_item",
        )
        assert grader._llm_calls == "per_item"


# =============================================================================
# Graders restored without the setting, and the explicit default
# =============================================================================


class TestPerCriterionUnchanged:
    @pytest.mark.asyncio
    async def test_a_grader_restored_without_llm_calls_grades_per_criterion(self, tmp_path):
        """A grader pickled before the setting existed (or a subclass that skips
        ``__init__``) falls back to the class default and calls per criterion."""
        fake, plain_fake = RubricLLM(), RubricLLM()
        grader = one_judge(fake)
        restored = CriterionGrader.__new__(CriterionGrader)
        restored.__dict__.update({k: v for k, v in vars(grader).items() if k != "_llm_calls"})
        plain = make_grader({"default": plain_fake}, judge_model_config=LLM)

        assert restored._llm_calls == "per_criterion"
        assert "llm_calls" not in _serialize_grader_config(restored)
        report = await Rubric(MIXED).grade(SUBMISSION, grader=restored, query=QUERY)
        expected = await Rubric(MIXED).grade(SUBMISSION, grader=plain, query=QUERY)

        assert report.model_dump() == expected.model_dump()
        assert [c.response_format for c in fake.calls] == [
            CriterionJudgment,
            MultiChoiceJudgment,
            CriterionJudgment,
            MultiChoiceJudgment,
        ]
        assert [(c.system_prompt, c.user_prompt) for c in fake.calls] == [
            (c.system_prompt, c.user_prompt) for c in plain_fake.calls
        ]

        dataset = RubricDataset(prompt=QUERY, rubric=Rubric(MIXED), name="one-item")
        dataset.add_item(SUBMISSION, "item")
        result = await evaluate(
            dataset,
            restored,
            show_progress=False,
            experiments_dir=tmp_path,
            experiment_name="restored",
        )
        assert result.successful_items == 1

    @pytest.mark.asyncio
    async def test_explicit_per_criterion_equals_the_default(self):
        """Prompts, reports and manifest of ``llm_calls="per_criterion"`` are a default
        grader's, for a panel, with guidelines, a query and a reference."""
        runs = []
        for setting in ({}, {"llm_calls": "per_criterion"}):
            fakes = {"a": RubricLLM(), "b": RubricLLM()}
            grader = make_grader(
                fakes, judges=[JudgeSpec(LLM, "a"), JudgeSpec(LLM, "b")], **setting
            )
            report = await Rubric(MIXED, guidelines=GUIDELINES).grade(
                SUBMISSION, grader=grader, query=QUERY, reference_submission=REFERENCE
            )
            prompts = {
                judge_id: [dataclasses.astuple(call) for call in fake.calls]
                for judge_id, fake in fakes.items()
            }
            runs.append((prompts, report.model_dump(), _serialize_grader_config(grader)))

        (default_prompts, default_report, default_config), (prompts, report, config) = runs
        assert prompts == default_prompts
        assert all(len(calls) == len(MIXED) for calls in prompts.values())
        assert report == default_report
        assert json.dumps(config) == json.dumps(default_config)
        assert "llm_calls" not in config


# =============================================================================
# Calls
# =============================================================================


def expected_system_prompt(binary_guide: str | None, multi_choice_guide: str | None) -> str:
    """The system prompt of a one-call judge, written out literally from its guides."""
    kinds = []
    guides = []
    rules = []
    if binary_guide is not None:
        kinds.append(
            "- A binary criterion holds a <criterion_type> and a <criterion>. Judge it as the "
            "binary criterion guide below describes."
        )
        guides.append(f"<binary_criterion_guide>\n{binary_guide}\n</binary_criterion_guide>")
        rules.append(
            '- For a binary criterion, "criterion_status" is "MET", "UNMET" or '
            '"CANNOT_ASSESS", and "selected_option" is null.'
        )
    if multi_choice_guide is not None:
        kinds.append(
            "- A multi-choice criterion holds a <question> and numbered <options>. Judge it as "
            "the multi-choice criterion guide below describes."
        )
        guides.append(
            f"<multi_choice_criterion_guide>\n{multi_choice_guide}\n</multi_choice_criterion_guide>"
        )
        rules.append(
            '- For a multi-choice criterion, "selected_option" is the number of the chosen '
            'option, as that criterion\'s <options> number it, and "criterion_status" is null.'
        )
    if binary_guide is not None and multi_choice_guide is not None:
        skeleton = (
            '{"judgments": [{"criterion_id": "c0", "criterion_status": "MET", '
            '"selected_option": null, "explanation": "..."}, {"criterion_id": "c1", '
            '"criterion_status": null, "selected_option": 2, "explanation": "..."}]}'
        )
    elif binary_guide is not None:
        skeleton = (
            '{"judgments": [{"criterion_id": "c0", "criterion_status": "MET", '
            '"selected_option": null, "explanation": "..."}]}'
        )
    else:
        skeleton = (
            '{"judgments": [{"criterion_id": "c0", "criterion_status": null, '
            '"selected_option": 2, "explanation": "..."}]}'
        )
    preamble = "\n\n".join(
        [
            "You are an expert evaluation judge. Your task is to judge every criterion in the "
            "<criteria> list against one submission, in a single response. Be precise, "
            "evidence-based, and consistent.",
            "Each criterion in <criteria> is a <rubric_criterion> with an id (c0, c1, ...).\n"
            + "\n".join(kinds),
            "Each guide is written for judging one criterion at a time. Apply it to each "
            "criterion of its kind separately, as if that criterion were the only one: judge "
            "every criterion independently, and never let your judgment of one criterion "
            "influence another. The <submission>, and any <guidelines>, <input> and "
            "<reference_submission>, apply to every criterion.",
            "A guide's RESPONSE FORMAT and EXAMPLES show the fields of one criterion's "
            "judgment. Your response holds one judgment per criterion, as the RESPONSE FORMAT "
            "at the end of these instructions says.",
        ]
    )
    response_format = (
        "RESPONSE FORMAT:\n"
        "Respond with valid JSON holding exactly one judgment per criterion, in the order the "
        f"criteria are listed:\n{skeleton}\n\n"
        '- "criterion_id" is the criterion\'s id.\n'
        + "\n".join(rules)
        + '\n- "explanation" is the 1-2 sentence explanation the criterion\'s guide asks for.\n\n'
        "Return only raw JSON starting with {, no back-ticks, no 'json' prefix."
    )
    return "\n\n".join([preamble, *guides, response_format])


class TestCalls:
    @pytest.mark.asyncio
    async def test_one_call_per_item_answered_as_a_rubric_judgment(self):
        fake = RubricLLM()
        grader = one_judge(fake)
        for submission in (SUBMISSION, OTHER_SUBMISSION):
            await Rubric(MIXED).grade(submission, grader=grader, query=QUERY)

        assert [c.response_format for c in fake.calls] == [RubricJudgment, RubricJudgment]
        assert [list(rubric_blocks(c.user_prompt)) for c in fake.calls] == [
            ["c0", "c1", "c2", "c3"]
        ] * 2
        assert SUBMISSION in fake.calls[0].user_prompt
        assert OTHER_SUBMISSION in fake.calls[1].user_prompt
        # One system prompt for every item of a rubric, so provider prompt caching applies.
        assert fake.calls[0].system_prompt == fake.calls[1].system_prompt

    @pytest.mark.asyncio
    async def test_each_llm_judge_of_a_panel_makes_one_call_per_item(self):
        fakes = {"a": RubricLLM(), "b": RubricLLM()}
        grader = make_grader(
            fakes, judges=[JudgeSpec(LLM, "a"), JudgeSpec(LLM, "b")], llm_calls="per_item"
        )
        for submission in (SUBMISSION, OTHER_SUBMISSION):
            report = await Rubric(MIXED).grade(submission, grader=grader, query=QUERY)
            assert report.error is None
        for fake in fakes.values():
            assert [c.response_format for c in fake.calls] == [RubricJudgment, RubricJudgment]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rubric, guides",
        [
            (BINARY, (GRADER_SYSTEM_PROMPT_DEFAULT, None)),
            (MULTI_CHOICE, (None, MULTI_CHOICE_SYSTEM_PROMPT)),
            (MIXED, (GRADER_SYSTEM_PROMPT_DEFAULT, MULTI_CHOICE_SYSTEM_PROMPT)),
        ],
        ids=["binary", "multi_choice", "mixed"],
    )
    async def test_the_system_prompt_holds_the_guides_of_the_kinds_present(self, rubric, guides):
        fake = RubricLLM()
        await Rubric(rubric).grade(SUBMISSION, grader=one_judge(fake))
        (call,) = fake.calls
        assert call.system_prompt == expected_system_prompt(*guides)
        assert ("<binary_criterion_guide>" in call.system_prompt) == (guides[0] is not None)
        assert ("<multi_choice_criterion_guide>" in call.system_prompt) == (guides[1] is not None)

    @pytest.mark.asyncio
    async def test_custom_system_prompts_are_embedded_verbatim(self):
        fake = RubricLLM()
        grader = one_judge(
            fake, system_prompt="Grade strictly.", multi_choice_system_prompt="Pick one option."
        )
        await Rubric(MIXED).grade(SUBMISSION, grader=grader)
        (call,) = fake.calls
        assert call.system_prompt == expected_system_prompt("Grade strictly.", "Pick one option.")

    @pytest.mark.asyncio
    async def test_the_user_prompt_lists_every_criterion_under_its_id(self):
        fake = RubricLLM()
        grader = one_judge(fake, shuffle_options=False)
        await Rubric([LIGHT, JARGON, CLARITY]).grade(
            SUBMISSION, grader=grader, query=QUERY, reference_submission=REFERENCE
        )
        (call,) = fake.calls
        assert call.user_prompt == (
            "<criteria>\n"
            '<rubric_criterion id="c0">\n'
            "<criterion_type>\npositive\n</criterion_type>\n\n"
            "<criterion>\nMentions light\n</criterion>\n"
            "</rubric_criterion>\n\n"
            '<rubric_criterion id="c1">\n'
            "<criterion_type>\nnegative\n</criterion_type>\n\n"
            "<criterion>\nUses unexplained jargon\n</criterion>\n"
            "</rubric_criterion>\n\n"
            '<rubric_criterion id="c2">\n'
            "<question>\nHow clear is the explanation?\n</question>\n\n"
            "<options>\n"
            "1. Unclear\n2. Mostly clear\n3. Very clear\n4. Cannot assess / not applicable\n"
            "</options>\n"
            "</rubric_criterion>\n"
            "</criteria>\n\n"
            f"<input>{QUERY}</input>\n\n"
            f"<reference_submission>\n{REFERENCE}\n</reference_submission>\n\n"
            f"<submission>\n{SUBMISSION}\n</submission>"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("auto_na_option", [True, False], ids=["auto_na", "forced_choice"])
    async def test_each_criterion_is_posed_as_a_per_criterion_call_poses_it(self, auto_na_option):
        """The k-th ``<rubric_criterion>`` holds exactly the criterion part of the
        per-criterion prompt of the same seed and judge: the same block, and the same option
        permutation (shuffle key), for every judge of a panel. Under forced choice
        (``auto_na_option=False``) a criterion without an NA option of its own is shown
        without one, in both modes."""
        rubric = [*MIXED, DEPTH]
        judges = [JudgeSpec(LLM, "a"), JudgeSpec(LLM, "b")]
        per_item = {"a": RubricLLM(), "b": RubricLLM()}
        per_criterion = {"a": RubricLLM(), "b": RubricLLM()}
        item_grader = make_grader(
            per_item, judges=judges, llm_calls="per_item", auto_na_option=auto_na_option
        )
        criterion_grader = make_grader(per_criterion, judges=judges, auto_na_option=auto_na_option)
        item_report = await Rubric(rubric).grade(SUBMISSION, grader=item_grader, query=QUERY)
        criterion_report = await Rubric(rubric).grade(
            SUBMISSION, grader=criterion_grader, query=QUERY
        )

        blocks_by_judge = {}
        for judge_id in ("a", "b"):
            # Each per-criterion prompt is its criterion part, then the input and submission.
            parts = {}
            for call in per_criterion[judge_id].calls:
                part, rest = call.user_prompt.split("\n\n<input>", 1)
                assert rest == f"{QUERY}</input>\n\n<submission>\n{SUBMISSION}\n</submission>"
                parts[next(c.requirement for c in rubric if c.requirement in part)] = part
            (call,) = per_item[judge_id].calls
            blocks = rubric_blocks(call.user_prompt)
            assert list(blocks) == [f"c{k}" for k in range(len(rubric))]
            for k, criterion in enumerate(rubric):
                assert blocks[f"c{k}"] == parts[criterion.requirement]
            # CLARITY has three options of its own; the NA option is added only by default.
            assert len(presented_labels(blocks["c1"])) == (4 if auto_na_option else 3)
            assert (NA_LABEL in presented_labels(blocks["c1"])) == auto_na_option
            blocks_by_judge[judge_id] = blocks
        # The judge is part of the shuffle key: the two judges see some options differently.
        assert blocks_by_judge["a"] != blocks_by_judge["b"]

        def shuffle_orders(report):
            return [[v.shuffle_order for v in cr.multi_choice_votes] for cr in report.report]

        assert shuffle_orders(item_report) == shuffle_orders(criterion_report)

    @pytest.mark.asyncio
    async def test_guidelines_prefix_the_prompt(self):
        with_guidelines, without = RubricLLM(), RubricLLM()
        await Rubric(MIXED, guidelines=GUIDELINES).grade(
            SUBMISSION, grader=one_judge(with_guidelines), query=QUERY
        )
        await Rubric(MIXED).grade(SUBMISSION, grader=one_judge(without), query=QUERY)
        (call,), (plain_call,) = with_guidelines.calls, without.calls
        assert call.user_prompt == GUIDELINES_PREFIX + plain_call.user_prompt
        assert plain_call.user_prompt.startswith("<criteria>\n")
        assert call.system_prompt == plain_call.system_prompt

    @pytest.mark.asyncio
    async def test_an_empty_rubric_makes_no_call(self):
        fake = RubricLLM()
        (judge_results,) = await one_judge(fake).judge(SUBMISSION, [], QUERY)
        assert judge_results.criterion_results == []
        assert fake.calls == []


# A lone surrogate (e.g. half of an emoji's JSON escape, decoded by ``json.loads``) cannot be
# encoded, so hashing it raises.
UNENCODABLE = "Plants use light \ud83d to make sugar."


class TestTheShuffleKey:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "llm_calls, rubric, shuffle_options",
        [
            ("per_criterion", MULTI_CHOICE, False),
            ("per_item", MULTI_CHOICE, False),
            ("per_item", BINARY, True),
            ("per_item", MIXED, False),
        ],
        ids=[
            "per_criterion_unshuffled",
            "per_item_unshuffled",
            "per_item_binary_only",
            "per_item_mixed_unshuffled",
        ],
    )
    async def test_the_submission_is_hashed_only_to_shuffle_options(
        self, llm_calls, rubric, shuffle_options
    ):
        """The submission's hash keys the option shuffle and nothing else, so it is computed
        only for a multi-choice criterion whose options are shuffled, in either mode: a
        submission that cannot be hashed is graded when there is none."""
        with pytest.raises(UnicodeEncodeError):
            UNENCODABLE.encode()
        fake = RubricLLM()
        grader = make_grader(
            {"default": fake},
            judge_model_config=LLM,
            llm_calls=llm_calls,
            shuffle_options=shuffle_options,
        )
        report = await Rubric(rubric).grade(UNENCODABLE, grader=grader)
        assert report.error is None and report.score is not None
        assert all(UNENCODABLE in call.user_prompt for call in fake.calls)


# =============================================================================
# Mapping answers back
# =============================================================================


async def judge_results(fake: RubricLLM, rubric: list[Criterion] = MIXED, **kwargs: Any) -> Any:
    """The one judge's ``CriterionResult``s for ``rubric``, from a one-call grader."""
    (results,) = await one_judge(fake, **kwargs).judge(SUBMISSION, rubric, QUERY)
    assert len(fake.calls) == 1
    return results.criterion_results


async def shuffle_orders_of(rubric: list[Criterion], **kwargs: Any) -> list[list[int] | None]:
    """The option permutation of each criterion when every judgment succeeds."""
    return [r.report.shuffle_order for r in await judge_results(RubricLLM(), rubric, **kwargs)]


class TestMapping:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("verdict", list(CriterionVerdict), ids=lambda v: v.value)
    async def test_every_binary_verdict_maps(self, verdict):
        fake = RubricLLM(
            {
                "c0": {"criterion_status": verdict.value, "explanation": "About light."},
                "c2": {"criterion_status": verdict.value, "explanation": "About jargon."},
            }
        )
        results = await judge_results(fake)
        for idx, reason in ((0, "About light."), (2, "About jargon.")):
            report = results[idx].report
            assert (report.verdict, report.reason, report.error) == (verdict, reason, None)
            assert report.requirement == MIXED[idx].requirement
            assert report.weight == MIXED[idx].weight

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reverse", [False, True], ids=["in_order", "out_of_order"])
    async def test_multi_choice_options_map_back_through_the_shuffle(self, reverse):
        fake = RubricLLM(
            {
                "c1": {"label": "Very clear", "explanation": "Clear."},
                "c3": {"label": NA_LABEL, "explanation": "No tone to speak of."},
            },
            reverse=reverse,
        )
        results = await judge_results(fake)
        blocks = rubric_blocks(fake.calls[0].user_prompt)

        clarity, tone = results[1].report, results[3].report
        assert clarity.multi_choice_verdict is not None and tone.multi_choice_verdict is not None
        assert (clarity.multi_choice_verdict.selected_index, clarity.reason) == (2, "Clear.")
        assert clarity.multi_choice_verdict.selected_label == "Very clear"
        assert clarity.multi_choice_verdict.value == 1.0
        assert not clarity.multi_choice_verdict.na
        # The auto-added NA option is the last of the effective criterion, and abstains.
        assert tone.multi_choice_verdict.selected_index == 4
        assert tone.multi_choice_verdict.selected_label == NA_LABEL
        assert tone.multi_choice_verdict.na
        for idx, report in ((1, clarity), (3, tone)):
            options = effective(MIXED[idx]).options
            assert options is not None and report.shuffle_order is not None
            # The recorded permutation is the order the judge was shown.
            assert [options[i].label for i in report.shuffle_order] == presented_labels(
                blocks[f"c{idx}"]
            )
        # The seed shuffles both, so mapping back is exercised.
        assert clarity.shuffle_order != sorted(clarity.shuffle_order)
        assert tone.shuffle_order != sorted(tone.shuffle_order)
        # The binary criteria map too, whatever the order of the entries.
        assert results[0].report.verdict == results[2].report.verdict == CriterionVerdict.MET

    @pytest.mark.asyncio
    async def test_without_shuffling_the_option_number_is_the_rubric_order(self):
        fake = RubricLLM({"c1": {"selected_option": 2}, "c3": {"selected_option": 4}})
        results = await judge_results(fake, shuffle_options=False)
        assert results[1].report.multi_choice_verdict.selected_label == "Mostly clear"
        assert results[3].report.multi_choice_verdict.selected_label == "Neutral"
        assert results[1].report.shuffle_order is None
        assert results[3].report.shuffle_order is None

    @pytest.mark.asyncio
    async def test_unknown_ids_and_the_other_kinds_field_are_ignored(self):
        fake = RubricLLM(
            {
                "c0": {"criterion_status": "UNMET", "selected_option": 3},
                "c1": {"criterion_status": "MET", "label": "Unclear"},
            },
            extra=[
                {"criterion_id": "c9", "criterion_status": "MET", "explanation": "stray"},
                {"criterion_id": "overall", "selected_option": 1, "explanation": "stray"},
                {"criterion_id": "c7", "criterion_status": "met", "explanation": "unusable"},
            ],
        )
        results = await judge_results(fake)
        assert [r.report.error for r in results] == [None] * 4
        assert results[0].report.verdict == CriterionVerdict.UNMET
        assert results[0].report.multi_choice_verdict is None
        assert results[1].report.verdict is None
        assert results[1].report.multi_choice_verdict.selected_label == "Unclear"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "criterion_id, spec, error",
        [
            ("c2", DROP, "the reply has no judgment with its id"),
            (
                "c0",
                [
                    {"criterion_id": "c0", "criterion_status": "MET", "explanation": "yes"},
                    {"criterion_id": "c0", "criterion_status": "UNMET", "explanation": "no"},
                ],
                "the reply has 2 judgments with its id",
            ),
            (
                "c2",
                [
                    {"criterion_id": "c2", "criterion_status": "UNMET", "explanation": "no"},
                    {"criterion_id": "c2", "criterion_status": "UNMET", "explanation": "no"},
                ],
                "the reply has 2 judgments with its id",
            ),
            (
                "c1",
                [
                    {"criterion_id": "c1", "selected_option": 1, "explanation": "one"},
                    {"criterion_id": "c1", "selected_option": "two", "explanation": "two"},
                    {"criterion_id": "c1", "selected_option": 3, "explanation": "three"},
                ],
                "the reply has 3 judgments with its id",
            ),
            (
                "c0",
                [{"criterion_id": "c0", "criterion_status": "met", "explanation": "yes"}],
                "criterion_status: Input should be 'MET', 'UNMET' or 'CANNOT_ASSESS'",
            ),
            (
                "c1",
                [{"criterion_id": "c1", "selected_option": "two", "explanation": "two"}],
                "selected_option: Input should be a valid integer, unable to parse string as an "
                "integer",
            ),
            (
                "c3",
                [{"criterion_id": "c3", "selected_option": 1}],
                "explanation: Field required",
            ),
            (
                "c0",
                [{"criterion_id": "c0", "criterion_status": "met"}],
                "criterion_status: Input should be 'MET', 'UNMET' or 'CANNOT_ASSESS'; "
                "explanation: Field required",
            ),
            (
                "c2",
                [{"criterion_id": 2, "criterion_status": "MET", "explanation": "id not a string"}],
                "the reply has no judgment with its id",
            ),
            (
                "c2",
                ["MET"],
                "the reply has no judgment with its id",
            ),
            (
                "c2",
                {"criterion_status": None, "selected_option": 1},
                "a binary criterion needs a criterion_status, got null",
            ),
            (
                "c1",
                {"criterion_status": "MET", "selected_option": None},
                "a multi-choice criterion needs a selected_option, got null",
            ),
        ],
        ids=[
            "missing",
            "duplicated",
            "duplicated_identically",
            "duplicated_with_an_unusable_entry",
            "unusable_verdict",
            "unusable_option",
            "unusable_without_explanation",
            "unusable_with_two_faults",
            "unreadable_id",
            "not_an_object",
            "binary_without_status",
            "multi_choice_without_option",
        ],
    )
    async def test_an_unusable_answer_fails_only_its_criterion(self, criterion_id, spec, error):
        idx = int(criterion_id[1:])
        criterion = MIXED[idx]
        name = f"criterion {criterion_id} ({criterion.name!r})"
        await self.assert_fails_alone(
            RubricLLM({criterion_id: spec}),
            idx,
            f"parse: no usable judgment for {name}: {error}",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shuffle_options", [True, False], ids=["shuffled", "unshuffled"])
    @pytest.mark.parametrize(
        "criterion_id, option, n_options",
        [("c1", 0, 4), ("c1", 5, 4), ("c3", 6, 5), ("c3", -1, 5)],
        ids=["clarity_0", "clarity_5", "tone_6", "tone_negative"],
    )
    async def test_an_out_of_range_option_fails_only_its_criterion(
        self, criterion_id, option, n_options, shuffle_options
    ):
        await self.assert_fails_alone(
            RubricLLM({criterion_id: {"selected_option": option}}),
            int(criterion_id[1:]),
            f"parse: Selected option {option} out of range [1, {n_options}]",
            shuffle_options=shuffle_options,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spec", [DROP, {"selected_option": None}], ids=["missing", "null"])
    async def test_an_unnamed_criterion_is_named_by_its_id_alone(self, spec):
        """Messages name a criterion as a decision model's do (``_describe``): its id, and
        its name only when it has one."""
        rubric = [*MIXED, DEPTH]
        error = (
            "the reply has no judgment with its id"
            if spec == DROP
            else "a multi-choice criterion needs a selected_option, got null"
        )
        await self.assert_fails_alone(
            RubricLLM({"c4": spec}),
            4,
            f"parse: no usable judgment for criterion c4: {error}",
            rubric=rubric,
        )

    @staticmethod
    async def assert_fails_alone(
        fake: RubricLLM,
        idx: int,
        error: str,
        *,
        rubric: list[Criterion] = MIXED,
        shuffle_options: bool = True,
    ) -> None:
        results = await judge_results(fake, rubric, shuffle_options=shuffle_options)
        orders = await shuffle_orders_of(rubric, shuffle_options=shuffle_options)
        for k, result in enumerate(results):
            report = result.report
            if k != idx:
                assert report.error is None, k
                continue
            assert report.error == error
            assert report.reason == f"Judge call failed (parse): {error.removeprefix('parse: ')}"
            if rubric[k].is_multi_choice:
                # Abstains on the NA option and records the permutation it was shown, if
                # its options were shuffled (as a per-criterion failure does).
                shown = effective(rubric[k])
                assert shown.options is not None and shown.na_option_index is not None
                assert report.multi_choice_verdict is not None and report.multi_choice_verdict.na
                assert (
                    report.multi_choice_verdict.selected_label
                    == shown.options[shown.na_option_index].label
                )
                assert report.shuffle_order == orders[k]
                assert (report.shuffle_order is not None) == shuffle_options
            else:
                assert report.verdict == CriterionVerdict.CANNOT_ASSESS


# =============================================================================
# The shared answer mapping reads a per-criterion judgment as it always has
# =============================================================================


class TraceUnavailableMultiChoice(BaseModel):
    """A custom multi-choice format whose explanation may be null and whose thinking trace
    cannot be read."""

    selected_option: int
    explanation: str | None = None

    @property
    def reasoning(self) -> str:
        raise RuntimeError("trace unavailable")


class BareMultiChoice(BaseModel):
    """A custom multi-choice format without an explanation."""

    selected_option: int


class TraceUnavailableBinary(BaseModel):
    """A custom binary format whose explanation may be null and whose thinking trace cannot
    be read."""

    criterion_status: CriterionVerdict
    explanation: str | None = None

    @property
    def reasoning(self) -> str:
        raise RuntimeError("trace unavailable")


class TestPerCriterionReadOrder:
    """Both modes map an answer with the same helpers, which read a per-criterion judgment
    in the order it has always been read: a multi-choice judgment's option and its range,
    then its explanation, then the thinking trace; a binary judgment's explanation, its
    verdict, the null-explanation check, then the trace. A custom response format with more
    than one fault therefore fails on the same one, with the same category."""

    @staticmethod
    async def judged(parsed: BaseModel, criterion: Criterion) -> Any:
        """The one report a per-criterion grader makes of ``parsed`` for ``criterion``."""
        format_keyword = (
            "multi_choice_response_format"
            if criterion.is_multi_choice
            else "binary_response_format"
        )
        grader = make_grader(
            {"default": ParsedLLM(parsed)},
            judge_model_config=LLM,
            **{format_keyword: type(parsed)},
        )
        (results,) = await grader.judge(SUBMISSION, [criterion], QUERY)
        (result,) = results.criterion_results
        assert result is not None
        return result.report

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "parsed",
        [
            TraceUnavailableMultiChoice(selected_option=9),
            BareMultiChoice(selected_option=9),
            TraceUnavailableMultiChoice(selected_option=9, explanation="Clear."),
        ],
        ids=["null_explanation", "no_explanation", "trace_unavailable"],
    )
    async def test_an_out_of_range_option_fails_before_anything_else_is_read(self, parsed):
        report = await self.judged(parsed, CLARITY)
        assert report.error == "parse: Selected option 9 out of range [1, 4]"
        assert report.multi_choice_verdict is not None and report.multi_choice_verdict.na

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "parsed, criterion",
        [
            (TraceUnavailableMultiChoice(selected_option=1), CLARITY),
            (TraceUnavailableBinary(criterion_status=CriterionVerdict.MET), LIGHT),
        ],
        ids=["multi_choice", "binary"],
    )
    async def test_a_null_explanation_fails_before_the_trace_is_read(self, parsed, criterion):
        report = await self.judged(parsed, criterion)
        assert report.error == "parse: LLM judgment has no explanation (explanation is null)"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "parsed, criterion",
        [
            (TraceUnavailableMultiChoice(selected_option=1, explanation="Clear."), CLARITY),
            (
                TraceUnavailableBinary(criterion_status=CriterionVerdict.MET, explanation="Yes."),
                LIGHT,
            ),
        ],
        ids=["multi_choice", "binary"],
    )
    async def test_the_trace_is_read_last(self, parsed, criterion):
        """Read it is, all the same: a trace that cannot be read fails the judgment as an
        unknown error, as it always has."""
        report = await self.judged(parsed, criterion)
        assert report.error == "unknown: trace unavailable"


# =============================================================================
# Whole-call failures
# =============================================================================


class TestWholeCallFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("shuffle_options", [True, False], ids=["shuffled", "unshuffled"])
    @pytest.mark.parametrize("category", list(FAILURES))
    async def test_a_failed_call_fails_every_criterion(self, category, shuffle_options):
        fake = RubricLLM(error=FAILURES[category]())
        results = await judge_results(fake, shuffle_options=shuffle_options)
        orders = await shuffle_orders_of(MIXED, shuffle_options=shuffle_options)

        assert all(r.report.error.startswith(f"{category}: ") for r in results)
        assert all(r.usage is None and r.cost is None for r in results)
        light, clarity, jargon, tone = (r.report for r in results)
        if category == "unknown":
            # The conservative worst case, by the sign of the weight.
            assert (light.verdict, jargon.verdict) == (CriterionVerdict.UNMET, CriterionVerdict.MET)
            for report, criterion in ((clarity, CLARITY), (tone, TONE)):
                worst_idx, _ = effective(criterion).worst_scored_option()
                assert report.multi_choice_verdict.selected_index == worst_idx
                assert not report.multi_choice_verdict.na
        else:
            assert light.verdict == jargon.verdict == CriterionVerdict.CANNOT_ASSESS
            for report in (clarity, tone):
                assert report.multi_choice_verdict.na
                assert report.multi_choice_verdict.selected_label == NA_LABEL
        # Multi-choice failures carry the permutation the call showed, if it shuffled the
        # options (as a failed per-criterion call does).
        assert [light.shuffle_order, jargon.shuffle_order] == [None, None]
        assert [clarity.shuffle_order, tone.shuffle_order] == [orders[1], orders[3]]
        assert (orders[1] is not None, orders[3] is not None) == (shuffle_options,) * 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("category", list(FAILURES))
    async def test_under_forced_choice_a_failure_abstains_without_an_option(self, category):
        """With ``auto_na_option=False`` a multi-choice criterion without an NA option of its
        own abstains by selecting no option, whether the whole call failed or its answer
        alone did; one with its own NA option (DEPTH) selects it."""
        rubric = [*MIXED, DEPTH]
        failed_call = await judge_results(
            RubricLLM(error=FAILURES[category]()), rubric, auto_na_option=False
        )
        failed_alone = await judge_results(
            RubricLLM({"c1": DROP, "c3": {"selected_option": 9}, "c4": DROP}),
            rubric,
            auto_na_option=False,
        )
        for results, abstains in ((failed_call, category != "unknown"), (failed_alone, True)):
            for criterion, result in zip(rubric, results, strict=True):
                if not criterion.is_multi_choice:
                    continue
                verdict = result.report.multi_choice_verdict
                assert verdict is not None and result.report.error is not None
                assert criterion.options is not None
                if not abstains:
                    worst_idx, _ = criterion.worst_scored_option()
                    assert (verdict.selected_index, verdict.na) == (worst_idx, False)
                elif criterion.na_option_index is None:
                    assert (verdict.selected_index, verdict.selected_label) == (None, None)
                    assert verdict.na
                else:
                    assert verdict.selected_index == criterion.na_option_index
                    assert verdict.na
                # The report keeps the criterion's own options: none was added.
                assert result.report.options == criterion.options

    @pytest.mark.asyncio
    @pytest.mark.parametrize("category", list(FAILURES))
    async def test_the_item_of_a_failed_call_has_no_score(self, category):
        fake = RubricLLM(error=FAILURES[category]())
        report = await Rubric(MIXED).grade(SUBMISSION, grader=one_judge(fake), query=QUERY)
        assert len(fake.calls) == 1  # never re-sent one criterion at a time
        assert report.report is not None
        assert all(cr.error and cr.error.startswith(f"{category}:") for cr in report.report)
        assert (report.score, report.raw_score, report.llm_raw_score) == (None, None, None)
        assert report.error is not None
        assert report.error.startswith("Every criterion's judgment failed")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw",
        [{"judgments": "none"}, {"verdicts": []}, ["c0", "c1"], "MET"],
        ids=["judgments_not_a_list", "judgments_missing", "array", "string"],
    )
    async def test_a_reply_that_is_no_rubric_judgment_fails_every_criterion_as_parse(self, raw):
        """The container itself fails validation, so the call fails as a whole: every
        criterion gets the call's one validation error, and no result carries usage (the
        reply is lost, as any reply that does not parse is)."""
        fake = RubricLLM(raw=raw)
        results = await judge_results(fake)
        errors = {r.report.error for r in results}
        (error,) = errors
        assert error.startswith("parse: 1 validation error for RubricJudgment")
        assert "no usable judgment" not in error
        assert all(r.usage is None and r.cost is None for r in results)

    @pytest.mark.asyncio
    async def test_in_a_panel_the_other_judge_still_grades_the_item(self):
        fakes = {"a": RubricLLM(error=FAILURES["infrastructure"]()), "b": RubricLLM()}
        grader = make_grader(
            fakes, judges=[JudgeSpec(LLM, "a"), JudgeSpec(LLM, "b")], llm_calls="per_item"
        )
        report = await Rubric(MIXED).grade(SUBMISSION, grader=grader, query=QUERY)
        assert report.error is None and report.score is not None
        assert report.judge_scores["a"] is None and report.judge_scores["b"] is not None


# =============================================================================
# Usage and thinking
# =============================================================================


class TestUsageAndThinking:
    @pytest.mark.asyncio
    async def test_usage_and_cost_ride_on_the_first_result(self):
        fake = RubricLLM()
        grader = one_judge(fake)
        (results,) = await grader.judge(SUBMISSION, MIXED, QUERY)
        criterion_results = [r for r in results.criterion_results if r is not None]
        assert len(criterion_results) == len(MIXED)
        assert [r.usage for r in criterion_results] == [USAGE, None, None, None]
        assert [r.cost for r in criterion_results] == [COST, None, None, None]
        assert (results.total_usage, results.total_cost) == (USAGE, COST)

        report = await Rubric(MIXED).grade(SUBMISSION, grader=grader, query=QUERY)
        assert (report.token_usage, report.completion_cost) == (USAGE, COST)

    @pytest.mark.asyncio
    async def test_usage_rides_on_the_first_result_even_when_it_failed(self):
        results = await judge_results(RubricLLM({"c0": DROP}))
        first, *rest = results
        assert first.report.error is not None and first.report.error.startswith("parse:")
        assert (first.usage, first.cost) == (USAGE, COST)
        assert all((r.usage, r.cost, r.report.error) == (None, None, None) for r in rest)

    @pytest.mark.asyncio
    async def test_a_panel_sums_one_call_per_judge(self):
        fakes = {"a": RubricLLM(), "b": RubricLLM()}
        grader = make_grader(
            fakes, judges=[JudgeSpec(LLM, "a"), JudgeSpec(LLM, "b")], llm_calls="per_item"
        )
        report = await Rubric(MIXED).grade(SUBMISSION, grader=grader, query=QUERY)
        assert report.token_usage == TokenUsage(200, 80, 280)
        assert report.completion_cost == pytest.approx(2 * COST)

    @pytest.mark.asyncio
    async def test_the_thinking_trace_is_on_every_report_and_vote(self):
        fake = RubricLLM(reasoning="I weighed each criterion on its own.")
        grader = one_judge(fake)
        results = await judge_results(fake)
        assert [r.report.reasoning for r in results] == ["I weighed each criterion on its own."] * 4

        report = await Rubric(MIXED).grade(SUBMISSION, grader=grader, query=QUERY)
        assert report.report is not None
        votes = [v for cr in report.report for v in (*cr.votes, *cr.multi_choice_votes)]
        assert len(votes) == 4
        assert {v.reasoning for v in votes} == {"I weighed each criterion on its own."}

    @pytest.mark.asyncio
    async def test_a_criterion_that_fails_alone_has_no_trace(self):
        """The trace is the deliberation behind a verdict, so it goes on each criterion
        judged from a usable judgment; a criterion that fails alone gets its failure report
        from ``_failed_judgment_result``, without the trace, as a per-criterion call whose
        reply carried a trace but could not be used does."""
        trace = "I weighed each criterion on its own."
        results = await judge_results(
            RubricLLM({"c0": DROP, "c1": {"selected_option": 9}}, reasoning=trace)
        )
        assert [r.report.error is None for r in results] == [False, False, True, True]
        assert [r.report.reasoning for r in results] == [None, None, trace, trace]

        # The same failure, in a call of its own with the same trace: the same report.
        per_criterion = ParsedLLM(
            MultiChoiceJudgment(selected_option=9, explanation="Clear.", reasoning=trace)
        )
        grader = make_grader({"default": per_criterion}, judge_model_config=LLM)
        (alone,) = await grader.judge(SUBMISSION, [CLARITY], QUERY)
        (in_one_call,) = await one_judge(
            RubricLLM({"c0": {"selected_option": 9}}, reasoning=trace)
        ).judge(SUBMISSION, [CLARITY], QUERY)
        (result,) = alone.criterion_results
        (batched,) = in_one_call.criterion_results
        assert result is not None and batched is not None
        assert result.report.error == "parse: Selected option 9 out of range [1, 4]"
        assert result.report.reasoning is None
        assert batched.report.model_dump() == result.report.model_dump()


# =============================================================================
# Panels with a decision model
# =============================================================================


class TestMixedPanel:
    @pytest.mark.asyncio
    async def test_the_decision_models_request_is_unchanged(self, decision_model):
        """Byte for byte, with every part of the item a request can carry: the rubric's
        guidelines, the input and the reference submission."""
        rubric = [LIGHT, CLARITY]
        per_item_llm, default_llm = RubricLLM(), RubricLLM()
        per_item = make_grader(
            {"llm": per_item_llm},
            judges=[JudgeSpec(dm(), "jev"), JudgeSpec(LLM, "llm")],
            llm_calls="per_item",
        )
        default = make_grader(
            {"llm": default_llm}, judges=[JudgeSpec(dm(), "jev"), JudgeSpec(LLM, "llm")]
        )
        graded = Rubric(rubric, guidelines=GUIDELINES)
        report = await graded.grade(
            SUBMISSION, grader=per_item, query=QUERY, reference_submission=REFERENCE
        )
        await graded.grade(SUBMISSION, grader=default, query=QUERY, reference_submission=REFERENCE)

        assert len(decision_model.calls) == 2
        assert request_json(decision_model.calls[0]) == request_json(decision_model.calls[1])
        state, _, _ = decision_model.calls[0]
        assert (state["guidelines"], state["input"], state["reference_submission"]) == (
            GUIDELINES,
            QUERY,
            REFERENCE,
        )
        assert [c.response_format for c in per_item_llm.calls] == [RubricJudgment]
        assert len(default_llm.calls) == len(rubric)

        assert report.error is None and report.report is not None
        assert [v.judge_id for v in report.report[0].votes] == ["jev", "llm"]
        assert [v.judge_id for v in report.report[1].multi_choice_votes] == ["jev", "llm"]


# =============================================================================
# Manifest
# =============================================================================


class TestManifest:
    def test_the_key_appears_only_under_per_item(self):
        per_item = one_judge(RubricLLM())
        default = make_grader({"default": RubricLLM()}, judge_model_config=LLM)
        config = _serialize_grader_config(per_item)
        default_config = _serialize_grader_config(default)

        assert "llm_calls" not in default_config
        assert config.pop("llm_calls") == "per_item"
        assert json.dumps(config) == json.dumps(default_config)

    @pytest.mark.asyncio
    async def test_an_evaluation_records_it_and_fails_an_item_whose_call_failed(self, tmp_path):
        fake = RubricLLM(error=FAILURES["infrastructure"](), fail_on=OTHER_SUBMISSION)
        dataset = RubricDataset(prompt=QUERY, rubric=Rubric(MIXED), name="per-item")
        dataset.add_item(SUBMISSION, "good")
        dataset.add_item(OTHER_SUBMISSION, "failed")
        result = await evaluate(
            dataset,
            one_judge(fake),
            show_progress=False,
            experiments_dir=tmp_path,
            experiment_name="per-item",
        )

        assert len(fake.calls) == 2
        assert (result.successful_items, result.failed_items) == (1, 1)
        errors = {r.item.submission: r.error for r in result.item_results}
        assert errors[SUBMISSION] is None
        failed_error = errors[OTHER_SUBMISSION]
        assert failed_error is not None
        assert failed_error.startswith("Every criterion's judgment failed")
        manifest = json.loads((tmp_path / "per-item" / "manifest.json").read_text())
        assert manifest["grader_config"]["llm_calls"] == "per_item"


# =============================================================================
# End to end: the real LLMClient, its transport and its response cache
# =============================================================================


def model_response(content: str | None, reasoning: str | None = None) -> ModelResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return ModelResponse(
        model="gpt-4.1-mini",
        choices=[{"message": message, "finish_reason": "stop", "index": 0}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def real_llm(tmp_path: Path | None = None) -> LLMConfig:
    if tmp_path is None:
        return LLMConfig(model="openai/gpt-4.1-mini")
    return LLMConfig(
        model="openai/gpt-4.1-mini", cache_enabled=True, cache_dir=str(tmp_path / "cache")
    )


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_a_rerun_reads_the_cached_answer_and_maps_it_identically(self, tmp_path):
        """The cached ``RubricJudgment`` pickles with its unusable entries, so a rerun makes
        no call and fails the same criteria with the same messages."""
        content = json.dumps(
            {
                "judgments": [
                    {
                        "criterion_id": "c0",
                        "criterion_status": "MET",
                        "selected_option": None,
                        "explanation": "Mentions light.",
                    },
                    {
                        "criterion_id": "c0",
                        "criterion_status": "met",
                        "selected_option": None,
                        "explanation": "Mentions light.",
                    },
                    {
                        "criterion_id": "c1",
                        "criterion_status": None,
                        "selected_option": "two",
                        "explanation": "Clear enough.",
                    },
                    {
                        "criterion_id": "c2",
                        "criterion_status": "UNMET",
                        "selected_option": None,
                        "explanation": "No jargon.",
                    },
                    {
                        "criterion_id": "c3",
                        "criterion_status": None,
                        "selected_option": 1,
                        "explanation": "A plain tone.",
                    },
                ]
            }
        )
        calls: list[dict[str, Any]] = []

        async def fake_acompletion(**params: Any) -> ModelResponse:
            calls.append(params)
            return model_response(content, reasoning="Each criterion, one at a time.")

        reports = []
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            for _ in range(2):
                calls.clear()
                grader = CriterionGrader(
                    judge_model_config=real_llm(tmp_path), llm_calls="per_item", seed=SEED
                )
                report = await Rubric(MIXED).grade(SUBMISSION, grader=grader, query=QUERY)
                reports.append((report, len(calls)))
                if len(reports) == 1:
                    (params,) = calls
                    response_format = params["response_format"]
                    assert response_format["json_schema"]["name"] == "RubricJudgment"

        (first, first_calls), (rerun, rerun_calls) = reports
        assert (first_calls, rerun_calls) == (1, 0)
        assert rerun.model_dump() == first.model_dump()

        assert first.report is not None
        errors = [cr.error for cr in first.report]
        assert errors[0] == (
            "parse: no usable judgment for criterion c0 ('light'): the reply has 2 judgments "
            "with its id"
        )
        assert errors[1] == (
            "parse: no usable judgment for criterion c1 ('clarity'): selected_option: Input "
            "should be a valid integer, unable to parse string as an integer"
        )
        assert errors[2:] == [None, None]
        assert first.report[2].final_verdict == CriterionVerdict.UNMET
        # The transport's thinking trace reaches the top level and every judged criterion.
        assert first.report[2].votes[0].reasoning == "Each criterion, one at a time."
        assert first.report[3].multi_choice_votes[0].reasoning == "Each criterion, one at a time."
        assert first.token_usage == TokenUsage(10, 5, 15)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content, error, reasoning",
        [
            ('{"judgments": [{"criterion_id": "c0", "crit', "parse: Unterminated string", None),
            ('{"judgments": "none"}', "parse: 1 validation error for RubricJudgment", None),
            ('{"verdicts": []}', "parse: 1 validation error for RubricJudgment", None),
            ("[1, 2]", "parse: 1 validation error for RubricJudgment", None),
            ('"MET"', "parse: 1 validation error for RubricJudgment", None),
            ("null", "parse: 1 validation error for RubricJudgment", None),
            (
                '{"judgments": [{"criterion_id": "c0", "crit',
                "parse: Unterminated string",
                "Each criterion, one at a time.",
            ),
            (
                '{"judgments": "none"}',
                "parse: 1 validation error for RubricJudgment",
                "Each criterion, one at a time.",
            ),
            (
                '{"verdicts": []}',
                "parse: 1 validation error for RubricJudgment",
                "Each criterion, one at a time.",
            ),
        ],
        ids=[
            "truncated",
            "judgments_not_a_list",
            "judgments_missing",
            "array",
            "string",
            "null",
            "truncated-trace",
            "judgments_not_a_list-trace",
            "judgments_missing-trace",
        ],
    )
    async def test_a_reply_that_does_not_parse_fails_every_criterion(
        self, tmp_path, content, error, reasoning
    ):
        """A reply that is not JSON, or whose ``judgments`` is missing or not a list, fails
        the whole call as ``parse`` (every criterion abstains), with an object reply's
        thinking trace as without one. The reply is lost with its usage and is never cached,
        so a rerun calls the model again."""
        calls: list[dict[str, Any]] = []

        async def fake_acompletion(**params: Any) -> ModelResponse:
            calls.append(params)
            return model_response(content, reasoning=reasoning)

        reports = []
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            for _ in range(2):
                grader = CriterionGrader(
                    judge_model_config=real_llm(tmp_path), llm_calls="per_item"
                )
                reports.append(await Rubric(MIXED).grade(SUBMISSION, grader=grader, query=QUERY))

        assert len(calls) == 2
        for report in reports:
            assert report.report is not None
            (only_error,) = {cr.error for cr in report.report}
            assert only_error is not None and only_error.startswith(error)
            assert report.report[0].final_verdict == CriterionVerdict.CANNOT_ASSESS
            assert report.score is None
            assert (report.token_usage, report.completion_cost) == (None, None)
            assert report.error is not None
            assert report.error.startswith("Every criterion's judgment failed")
