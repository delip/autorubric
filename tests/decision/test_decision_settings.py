"""Grader settings that decision-model judges cannot use.

``system_prompt``, ``multi_choice_system_prompt`` and ``shuffle_options`` shape the prompts an
LLM judge is sent. A decision model's request has no system prompt and keeps the rubric's
option order, so in a mixed ensemble these settings apply to the LLM judges, and a grader
whose every judge is a decision model ignores them. Passing one explicitly to such a grader
warns (``UserWarning``). ``shuffle_options`` defaults to ``True``, so only a value the caller
passed warns, never the default, and the effective value is exactly what it always was.

``binary_response_format`` / ``multi_choice_response_format`` define the fields of a
generated judgment, which only an LLM judge fills, so either one raises ``ValueError`` at
construction when any judge is a decision model.

Nothing here reaches the network: decision-model requests go to the recording fake SDK
client of ``conftest.py`` and LLM judges are recording fakes patched in for ``LLMClient``.
"""

from __future__ import annotations

import copy
import inspect
import json
import pickle
import sys
import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from typesafe_sdk import SystemOneResponse

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DecisionModelConfig,
    LLMConfig,
    Rubric,
    TokenUsage,
)
from autorubric.eval import _serialize_grader_config
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.types import CANONICAL_NA_OPTION, CriterionJudgment, MultiChoiceJudgment

API_KEY = "ts-test-key-never-persisted-5a1d"
LLM = LLMConfig(model="test-model")
NA_LABEL = CANONICAL_NA_OPTION.label
SUBMISSION = "Plants use light to turn water and carbon dioxide into sugar."
RUBRIC = [
    Criterion(name="light", weight=5.0, requirement="Mentions light"),
    Criterion(
        name="clarity",
        weight=2.0,
        requirement="How clear is the explanation?",
        scale_type="ordinal",
        options=[
            CriterionOption(label="Unclear", value=0.0),
            CriterionOption(label="Mostly clear", value=0.6),
            CriterionOption(label="Very clear", value=1.0),
        ],
    ),
]
ANSWERS = {
    "c0": {"type": "noul", "noul": 0.9},
    "c1": {
        "type": "choice",
        "choice": "Mostly clear",
        "confidence": 0.6,
        "probabilities": {"Unclear": 0.1, "Mostly clear": 0.7, "Very clear": 0.15, NA_LABEL: 0.05},
    },
}


def dm(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(**{"model": "jev-latest", "api_key": API_KEY, **overrides})


def response() -> SystemOneResponse:
    payload = {"model": "jev-test", "usage": {"input_tokens": 50}, "answers": ANSWERS}
    return SystemOneResponse.model_validate_json(json.dumps(payload))


def request_json(call: tuple[Any, Any, dict[str, Any]]) -> str:
    """One recorded decision-model request (state and questions) as JSON; key order counts."""
    state, questions, _ = call
    wire = {qid: question.model_dump(mode="json") for qid, question in questions.items()}
    return json.dumps([state, wire])


# Every panel shape, by kind of judge. A factory, so each grader gets fresh configs.
PANELS: dict[str, Callable[[], dict[str, Any]]] = {
    "decision_model": lambda: {"judge_model_config": dm()},
    "decision_models": lambda: {"judges": [JudgeSpec(dm(), "jev-a"), JudgeSpec(dm(), "jev-b")]},
    "mixed": lambda: {"judges": [JudgeSpec(dm(), "jev"), JudgeSpec(LLM, "llm")]},
    "llm": lambda: {"judge_model_config": LLM},
    "llms": lambda: {"judges": [JudgeSpec(LLM, "a"), JudgeSpec(LLM, "b")]},
}
ALL_DECISION_MODELS = {"decision_model", "decision_models"}
WITH_A_DECISION_MODEL = ["decision_model", "decision_models", "mixed"]

EXPLICIT_SETTINGS = [
    ("system_prompt", "Grade strictly."),
    ("multi_choice_system_prompt", "Pick exactly one option."),
    ("shuffle_options", True),
    ("shuffle_options", False),
]


def llm_only_warning(setting: str) -> str:
    return (
        f"{setting} applies to LLM judges only; this grader has none (every judge is a "
        "decision model), so it has no effect"
    )


def pickle_round_trip(value: Any) -> Any:
    return pickle.loads(pickle.dumps(value))


def build(**kwargs: Any) -> tuple[CriterionGrader, list[warnings.WarningMessage]]:
    """Build a grader, returning it with every ``UserWarning`` its construction issued."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grader = CriterionGrader(**kwargs)
    return grader, [w for w in caught if w.category is UserWarning]


class RecordingLLMClient:
    """Stand-in for ``LLMClient``: records prompts; MET, or the first option presented."""

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> GenerateResult:
        self.prompts.append((system_prompt, user_prompt))
        parsed: Any
        if "<options>" in user_prompt:
            parsed = MultiChoiceJudgment(selected_option=1, explanation="first")
        else:
            parsed = CriterionJudgment(criterion_status=CriterionVerdict.MET, explanation="present")
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=None,
            parsed=parsed,
        )


# =============================================================================
# LLM-only settings: the warning matrix
# =============================================================================


class TestLLMOnlySettingsWarning:
    @pytest.mark.parametrize("panel", list(PANELS))
    @pytest.mark.parametrize(
        "setting, value", EXPLICIT_SETTINGS, ids=[f"{s}={v!r}" for s, v in EXPLICIT_SETTINGS]
    )
    def test_explicit_setting_warns_only_when_every_judge_is_a_decision_model(
        self, panel, setting, value
    ):
        _, caught = build(**PANELS[panel](), **{setting: value})

        if panel in ALL_DECISION_MODELS:
            assert [str(w.message) for w in caught] == [llm_only_warning(setting)]
            # Attributed to the line that built the grader (stacklevel), not the library.
            assert caught[0].filename == __file__
        else:
            assert caught == []

    @pytest.mark.parametrize("panel", list(PANELS))
    def test_defaults_never_warn(self, panel):
        _, caught = build(**PANELS[panel]())
        assert caught == []

    def test_explicit_none_prompts_are_the_defaults_and_do_not_warn(self):
        _, caught = build(
            judge_model_config=dm(), system_prompt=None, multi_choice_system_prompt=None
        )
        assert caught == []

    def test_each_explicit_setting_warns_once(self):
        _, caught = build(
            judges=[JudgeSpec(dm(), "jev-a"), JudgeSpec(dm(), "jev-b")],
            system_prompt="Grade strictly.",
            multi_choice_system_prompt="Pick exactly one option.",
            shuffle_options=False,
        )
        assert [str(w.message) for w in caught] == [
            llm_only_warning("system_prompt"),
            llm_only_warning("multi_choice_system_prompt"),
            llm_only_warning("shuffle_options"),
        ]

    @pytest.mark.asyncio
    async def test_ignored_settings_change_nothing_a_decision_model_sends(self, fake_sdk):
        fake_sdk.response = response()
        plain, _ = build(judge_model_config=dm())
        tuned, caught = build(
            judge_model_config=dm(),
            system_prompt="Grade strictly.",
            multi_choice_system_prompt="Pick exactly one option.",
            shuffle_options=False,
        )
        assert len(caught) == 3

        plain_report = await Rubric(RUBRIC).grade(SUBMISSION, grader=plain)
        tuned_report = await Rubric(RUBRIC).grade(SUBMISSION, grader=tuned)

        assert len(fake_sdk.calls) == 2
        assert request_json(fake_sdk.calls[0]) == request_json(fake_sdk.calls[1])
        assert tuned_report.model_dump() == plain_report.model_dump()


# =============================================================================
# LLM-only settings keep applying to the LLM judges of a mixed ensemble
# =============================================================================


class TestLLMOnlySettingsInMixedEnsembles:
    @pytest.mark.asyncio
    async def test_system_prompts_reach_the_llm_judge_only(self, fake_sdk):
        fake_sdk.response = response()
        default_llm, custom_llm = RecordingLLMClient(), RecordingLLMClient()
        with patch("autorubric.graders.criterion_grader.LLMClient", return_value=default_llm):
            default, _ = build(**PANELS["mixed"](), seed=7)
        with patch("autorubric.graders.criterion_grader.LLMClient", return_value=custom_llm):
            custom, caught = build(
                **PANELS["mixed"](),
                seed=7,
                system_prompt="Grade strictly.",
                multi_choice_system_prompt="Pick exactly one option.",
            )
        assert caught == []

        await Rubric(RUBRIC).grade(SUBMISSION, grader=default)
        await Rubric(RUBRIC).grade(SUBMISSION, grader=custom)

        # The LLM judge gets the custom system prompt for each criterion kind...
        system_by_kind = {"<options>" in user: system for system, user in custom_llm.prompts}
        assert system_by_kind == {False: "Grade strictly.", True: "Pick exactly one option."}
        # ...with the same user prompts (same seed) as under the default system prompts.
        assert sorted(u for _, u in custom_llm.prompts) == sorted(u for _, u in default_llm.prompts)
        # The decision model's request is the same either way.
        assert len(fake_sdk.calls) == 2
        assert request_json(fake_sdk.calls[0]) == request_json(fake_sdk.calls[1])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shuffle", [None, True, False], ids=["default", "True", "False"])
    async def test_shuffle_options_applies_to_the_llm_judge_only(self, fake_sdk, shuffle):
        fake_sdk.response = response()
        kwargs = {} if shuffle is None else {"shuffle_options": shuffle}
        with patch(
            "autorubric.graders.criterion_grader.LLMClient", return_value=RecordingLLMClient()
        ):
            grader, caught = build(**PANELS["mixed"](), **kwargs)
        assert caught == []
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader)

        clarity = report.report[1]
        jev_vote, llm_vote = clarity.multi_choice_votes
        assert jev_vote.shuffle_order is None  # a decision model keeps the rubric's order
        if shuffle is False:
            assert llm_vote.shuffle_order is None
        else:
            assert sorted(llm_vote.shuffle_order) == [0, 1, 2, 3]
        # Its question offers the options in the rubric's order, NA last.
        (call,) = fake_sdk.calls
        assert list(call[1]["c1"].criteria) == ["Unclear", "Mostly clear", "Very clear", NA_LABEL]


# =============================================================================
# shuffle_options: the default is True, exactly as before
# =============================================================================


class TestShuffleOptionsDefault:
    @pytest.mark.parametrize("panel", list(PANELS))
    def test_default_is_the_value_true(self, panel):
        grader, _ = build(**PANELS[panel]())
        assert grader._shuffle_options is True

    @pytest.mark.parametrize("value", [True, False])
    def test_explicit_value_is_kept(self, value):
        grader, _ = build(judge_model_config=LLM, shuffle_options=value)
        assert grader._shuffle_options is value

    def test_manifest_records_true_by_default(self):
        grader, _ = build(judge_model_config=LLM)
        assert _serialize_grader_config(grader)["shuffle_options"] is True

    def test_signature_reads_as_a_true_default(self):
        param = inspect.signature(CriterionGrader).parameters["shuffle_options"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert str(param) == str(param.replace(default=True))
        assert bool(param.default) is True

    def test_the_default_object_is_the_bool_true(self):
        """Telling a passed ``shuffle_options`` from the default changes no default object:
        signature introspection, ``__kwdefaults__`` and the source (which the API reference
        renders) all show the plain ``True`` they always did."""
        param = inspect.signature(CriterionGrader).parameters["shuffle_options"]
        assert param.default is True
        assert CriterionGrader.__init__.__kwdefaults__["shuffle_options"] is True
        assert "shuffle_options: bool = True," in inspect.getsource(CriterionGrader)

    @pytest.mark.parametrize(
        "target", [CriterionGrader, CriterionGrader.__init__], ids=["class", "init"]
    )
    def test_the_full_argspec_lists_every_keyword_only_parameter_and_default(self, target):
        """``inspect.getfullargspec`` does not follow ``__wrapped__`` as ``inspect.signature``
        does, so the keyword-recording wrapper must carry ``__init__``'s signature itself:
        otherwise the argspec reads ``(self, *args, **kwargs)``, with no keyword-only
        parameters and no defaults, for every tool built on it."""
        spec = inspect.getfullargspec(target)
        assert spec.args == ["self"]
        assert spec.varargs is None and spec.varkw is None
        assert spec.kwonlyargs == list(inspect.signature(CriterionGrader).parameters)
        assert spec.kwonlydefaults == CriterionGrader.__init__.__kwdefaults__
        assert spec.kwonlydefaults is not None and spec.kwonlydefaults["shuffle_options"] is True

    def test_every_default_is_an_ordinary_serializable_value(self):
        """Tools that log or rebuild a configuration from the signature defaults work."""
        defaults = {
            name: param.default
            for name, param in inspect.signature(CriterionGrader).parameters.items()
        }
        assert json.loads(json.dumps(defaults))["shuffle_options"] is True
        assert json.loads(json.dumps(CriterionGrader.__init__.__kwdefaults__)) == defaults

    @pytest.mark.parametrize(
        "duplicate",
        [copy.copy, copy.deepcopy, pickle_round_trip, lambda v: json.loads(json.dumps(v))],
    )
    def test_a_copied_default_is_the_value_true(self, duplicate):
        """A default copied, pickled or serialized by a tool is the plain ``True``. Passing it
        back is passing ``shuffle_options`` explicitly: the value in effect is ``True``, and
        a grader with only decision-model judges warns, as for any explicit value."""
        default = inspect.signature(CriterionGrader).parameters["shuffle_options"].default
        assert duplicate(default) is True
        grader, caught = build(judge_model_config=dm(), shuffle_options=duplicate(default))
        assert [str(w.message) for w in caught] == [llm_only_warning("shuffle_options")]
        assert grader._shuffle_options is True

    def test_calls_the_grader_rejects_fail_exactly_as_before(self):
        with pytest.raises(TypeError) as positional:
            CriterionGrader(LLM)  # every parameter is keyword-only
        assert str(positional.value) == (
            "CriterionGrader.__init__() takes 1 positional argument but 2 were given"
        )
        with pytest.raises(TypeError) as unknown:
            CriterionGrader(judge_model_config=LLM, bogus=1)
        assert str(unknown.value) == (
            "CriterionGrader.__init__() got an unexpected keyword argument 'bogus'"
        )

    def test_passed_settings_are_tracked_per_construction(self):
        """What one construction was passed never leaks into the next."""
        build(judge_model_config=dm(), shuffle_options=False)
        _, caught = build(judge_model_config=dm())
        assert caught == []

    def test_a_subclass_passing_the_setting_up_passes_it(self):
        class Tuned(CriterionGrader):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(shuffle_options=False, **kwargs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grader = Tuned(judge_model_config=dm())
        assert [str(w.message) for w in caught] == [llm_only_warning("shuffle_options")]
        assert grader._shuffle_options is False


# =============================================================================
# Response formats need a generative judge
# =============================================================================


class TaggedJudgment(CriterionJudgment):
    """A custom binary response format with a field only a generative judge can fill."""

    tag: str = ""


class TaggedMultiChoiceJudgment(MultiChoiceJudgment):
    """A custom multi-choice response format with a field only a generative judge can fill."""

    tag: str = ""


RESPONSE_FORMATS: dict[str, type] = {
    "binary_response_format": TaggedJudgment,
    "multi_choice_response_format": TaggedMultiChoiceJudgment,
}
FORMAT_CHOICES = [
    ["binary_response_format"],
    ["multi_choice_response_format"],
    ["binary_response_format", "multi_choice_response_format"],
]


class TestResponseFormats:
    @pytest.mark.parametrize("panel", WITH_A_DECISION_MODEL)
    @pytest.mark.parametrize("names", FORMAT_CHOICES, ids=["binary", "multi_choice", "both"])
    def test_any_decision_model_judge_rejects_a_response_format(self, panel, names):
        with pytest.raises(ValueError) as excinfo:
            CriterionGrader(**PANELS[panel](), **{name: RESPONSE_FORMATS[name] for name in names})
        message = str(excinfo.value)
        assert message.startswith(" and ".join(names) + " cannot be combined with decision-model")
        assert "only an LLM judge" in message

    def test_the_error_names_the_decision_model_judges(self):
        with pytest.raises(ValueError, match=r"decision-model judges \('jev-a', 'jev-b'\)"):
            CriterionGrader(
                judges=[
                    JudgeSpec(dm(), "jev-a"),
                    JudgeSpec(LLM, "llm"),
                    JudgeSpec(dm(), "jev-b"),
                ],
                binary_response_format=TaggedJudgment,
            )

    @pytest.mark.parametrize("panel", ["llm", "llms"])
    def test_llm_judges_accept_response_formats(self, panel):
        grader = CriterionGrader(**PANELS[panel](), **RESPONSE_FORMATS)
        assert grader._binary_response_format is TaggedJudgment
        assert grader._multi_choice_response_format is TaggedMultiChoiceJudgment

    def test_rejected_before_any_client_is_built(self, monkeypatch):
        """A configuration error is reported as such, even where no client could be built."""
        monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
        with pytest.raises(ValueError, match="binary_response_format cannot be combined"):
            CriterionGrader(judge_model_config=dm(), binary_response_format=TaggedJudgment)
