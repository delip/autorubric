"""The confidence cascade: a decision model judges everything, LLM judges what it is unsure of.

``CriterionGrader(judge_model_config=DecisionModelConfig(...), escalation=EscalationConfig(...))``
grades an item with one decision-model request for the whole rubric. A criterion is
*escalated* when the decision model's vote errored, abstained (``CANNOT_ASSESS`` or an NA
option), or has a ``confidence`` below the criterion's threshold; only the escalated
criteria go to the escalation judges (LLMs), each through the per-criterion LLM path with
the criterion's original index. The decision model's vote on an escalated criterion stays
in the vote list with ``superseded=True`` and the final verdict aggregates the escalation
judges' votes only, never falling back to the decision model's.

The real ``DecisionModelClient`` runs on the recording fake SDK client of ``conftest.py``
(``fake_sdk``), so ``fake_sdk.calls`` counts every request that would have reached the
endpoint. LLM judges are scripted ``LLMClient`` stand-ins that answer by criterion and
record every prompt. Nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import warnings
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import litellm
import pytest
import typesafe_sdk
from typesafe_sdk import SystemOneResponse

import autorubric
import autorubric.graders
from autorubric import (
    CannotAssessConfig,
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DataItem,
    DecisionModelConfig,
    EnsembleEvaluationReport,
    EscalationConfig,
    EvaluationReport,
    FewShotConfig,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
    evaluate,
    fill_ground_truth,
)
from autorubric.eval import EvalRunner, ItemResult, _serialize_grader_config
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.types import (
    CANONICAL_NA_OPTION,
    CriterionJudgment,
    MultiChoiceJudgment,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CANNOT_ASSESS = CriterionVerdict.CANNOT_ASSESS
NA = CANONICAL_NA_OPTION.label

API_KEY = "ts-test-key-never-persisted-5a9d"
QUERY = "Explain photosynthesis."
SUBMISSION = "Plants use light to turn water and carbon dioxide into sugar."

LIGHT = Criterion(name="light", weight=5.0, requirement="Mentions light")
MYTH = Criterion(name="myth", weight=-3.0, requirement="Claims plants eat soil")
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
    requirement="Which tone does the answer take?",
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Casual", value=0.5),
        CriterionOption(label="Mixed", value=0.0),
    ],
)
SENTENCES = Criterion(weight=1.0, requirement="Uses complete sentences")  # unnamed
RUBRIC = [LIGHT, MYTH, CLARITY, TONE, SENTENCES]
REQUIREMENTS = [c.requirement for c in RUBRIC]

# The options a decision model is offered for each multi-choice criterion: the rubric's,
# in order, plus the NA option auto_na_option guarantees.
CLARITY_LABELS = ["Unclear", "Mostly clear", "Very clear", NA]
TONE_LABELS = ["Formal", "Casual", "Mixed", NA]

LLM = LLMConfig(model="gemini")


def dm(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(**{"model": "jev-latest", "api_key": API_KEY, **overrides})


# =============================================================================
# Decision-model answers (confidence in comments, from clamp((K p - 1) / (K - 1), 0, 1))
# =============================================================================


def noul(p: float) -> dict[str, Any]:
    """A Noul answer: P(MET) = p, confidence 2|p - 0.5|."""
    return {"type": "noul", "noul": p}


def pick(selected: str, labels: list[str], p: float) -> dict[str, Any]:
    """A Choice answer selecting ``selected`` with probability ``p``, the rest spread evenly."""
    rest = (1 - p) / (len(labels) - 1)
    probabilities = {label: p if label == selected else rest for label in labels}
    return {"type": "choice", "choice": selected, "confidence": 0.0, "probabilities": probabilities}


def verdict_choice(verdict: CriterionVerdict, p: float) -> dict[str, Any]:
    """A binary Choice answer (``binary_framing="choice"``)."""
    return pick(verdict.value, [v.value for v in CriterionVerdict], p)


SURE = {
    "c0": noul(0.95),  # MET, 0.9
    "c1": noul(0.05),  # UNMET, 0.9
    "c2": pick("Very clear", CLARITY_LABELS, 0.97),  # 0.96
    "c3": pick("Formal", TONE_LABELS, 0.97),  # 0.96
    "c4": noul(0.95),  # MET, 0.9
}


def answers(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``SURE`` with some answers replaced, keyed by question id (``c0=...``)."""
    return {**SURE, **overrides}


def response(
    answer_map: dict[str, Any], *, input_tokens: int = 1234, output_tokens: int = 7
) -> SystemOneResponse:
    payload = {
        "model": "jev-test",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "answers": answer_map,
    }
    return SystemOneResponse.model_validate_json(json.dumps(payload))


# =============================================================================
# Scripted LLM judges
# =============================================================================

DEFAULT_LLM_ANSWERS: dict[str, Any] = {
    LIGHT.requirement: "MET",
    MYTH.requirement: "UNMET",
    CLARITY.requirement: "Mostly clear",
    TONE.requirement: "Casual",
    SENTENCES.requirement: "MET",
}
LLM_COST = 0.001
LLM_USAGE = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
_SECTION = r"<{0}>\n(.*?)\n</{0}>"
_NA_SUFFIX = " (cannot assess / not applicable)"


@dataclasses.dataclass
class LLMCall:
    model: str
    requirement: str
    system_prompt: str
    user_prompt: str


class ScriptedLLM:
    """Stand-in for ``LLMClient``: answers from its script by the criterion's requirement.

    A script value is a verdict value (binary), an option label (multi-choice; answered as
    the label's position among the options as presented, so it holds whatever the shuffle),
    or an exception to raise. Every prompt is recorded in ``calls`` and in the shared log.
    """

    def __init__(self, model: str, script: dict[str, Any], log: list[LLMCall]) -> None:
        self.model = model
        self.script = script
        self.log = log
        self.calls: list[LLMCall] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult:
        multi_choice = "<options>" in user_prompt
        found = re.search(_SECTION.format("question" if multi_choice else "criterion"), user_prompt)
        assert found is not None
        requirement = found.group(1)
        call = LLMCall(self.model, requirement, system_prompt, user_prompt)
        self.calls.append(call)
        self.log.append(call)

        answer = self.script[requirement]
        if isinstance(answer, BaseException):
            raise answer
        explanation = f"says {answer}"
        parsed: Any
        if multi_choice:
            options = re.search(_SECTION.format("options"), user_prompt, re.S)
            assert options is not None
            presented = [
                line.removesuffix(_NA_SUFFIX)
                for line in re.findall(r"^\d+\. (.*)$", options.group(1), re.M)
            ]
            parsed = MultiChoiceJudgment(
                selected_option=presented.index(answer) + 1, explanation=explanation
            )
        else:
            parsed = CriterionJudgment(
                criterion_status=CriterionVerdict(answer), explanation=explanation
            )
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=LLM_USAGE,
            cost=LLM_COST,
            parsed=parsed,
        )


@pytest.fixture
def llm_log() -> list[LLMCall]:
    return []


@pytest.fixture
def make_grader(fake_sdk: Any, llm_log: list[LLMCall]) -> Iterator[Any]:
    """Build graders whose LLM judges are scripted (``scripts``: model -> requirement ->
    answer, over ``DEFAULT_LLM_ANSWERS``); close decision-model caches at teardown."""
    built: list[CriterionGrader] = []

    def build(*, scripts: dict[str, dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        scripts = scripts or {}

        def client(config: LLMConfig) -> ScriptedLLM:
            script = {**DEFAULT_LLM_ANSWERS, **scripts.get(config.model, {})}
            return ScriptedLLM(config.model, script, llm_log)

        with patch("autorubric.graders.criterion_grader.LLMClient", side_effect=client):
            grader = CriterionGrader(**kwargs)
        built.append(grader)
        return grader

    yield build
    for grader in built:
        for dm_client in grader._decision_clients.values():
            dm_client.close()


def cascade(
    threshold: float = 0.5,
    judges: Any = LLM,
    per_criterion: dict[str, float] | None = None,
) -> EscalationConfig:
    return EscalationConfig(judges=judges, threshold=threshold, per_criterion=per_criterion)


def panel(*weights: float) -> list[JudgeSpec]:
    """Two LLM escalation judges, ``a`` and ``b``."""
    w = weights or (1.0, 1.0)
    return [JudgeSpec(LLMConfig(model="a"), "a", w[0]), JudgeSpec(LLMConfig(model="b"), "b", w[1])]


async def grade(
    grader: CriterionGrader,
    rubric: list[Criterion] | Rubric = RUBRIC,
    *,
    reference_submission: str | None = None,
) -> Any:
    rubric = rubric if isinstance(rubric, Rubric) else Rubric(rubric)
    report = await rubric.grade(
        SUBMISSION, grader=grader, query=QUERY, reference_submission=reference_submission
    )
    assert isinstance(report, EnsembleEvaluationReport)
    return report


def votes_of(cr: Any) -> list[Any]:
    return list(cr.votes or cr.multi_choice_votes)


# =============================================================================
# EscalationConfig
# =============================================================================


class TestEscalationConfig:
    def test_a_bare_llm_config_becomes_one_judge_named_escalation(self):
        config = EscalationConfig(judges=LLM, threshold=0.72)
        assert config.judges == [JudgeSpec(LLM, "escalation", 1.0)]
        assert config.judges[0].llm_config is LLM
        assert config.threshold == 0.72
        assert config.per_criterion is None

    def test_judge_specs_are_kept_in_order_and_copied(self):
        judges = panel(2.0, 1.0)
        config = EscalationConfig(judges, 0.5)  # positional, like any dataclass
        judges.append(JudgeSpec(LLMConfig(model="c"), "c"))
        assert [j.judge_id for j in config.judges] == ["a", "b"]
        assert [j.weight for j in config.judges] == [2.0, 1.0]

    def test_a_tuple_of_judge_specs_is_accepted(self):
        assert EscalationConfig(tuple(panel()), 0.5).judges == panel()

    @pytest.mark.parametrize("threshold", [0, 0.0, 0.5, 1, 1.0])
    def test_threshold_bounds_are_inclusive(self, threshold):
        assert EscalationConfig(LLM, threshold).threshold == threshold

    @pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan, math.inf, True, "0.5", None])
    def test_threshold_must_be_a_number_in_the_unit_interval(self, threshold):
        with pytest.raises(ValueError, match=r"EscalationConfig\.threshold must be a number"):
            EscalationConfig(LLM, threshold)

    @pytest.mark.parametrize("override", [-0.5, 1.5, math.nan, False, "high", None])
    def test_per_criterion_thresholds_must_be_numbers_in_the_unit_interval(self, override):
        with pytest.raises(ValueError, match=r"EscalationConfig\.per_criterion\['light'\]"):
            EscalationConfig(LLM, 0.5, per_criterion={"light": override})

    def test_per_criterion_keys_are_criterion_names(self):
        with pytest.raises(ValueError, match="criterion names"):
            EscalationConfig(LLM, 0.5, per_criterion={3: 0.5})

    def test_per_criterion_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="must map criterion names to thresholds"):
            EscalationConfig(LLM, 0.5, per_criterion=[("light", 0.5)])

    def test_per_criterion_is_copied(self):
        overrides = {"light": 0.9}
        config = EscalationConfig(LLM, 0.5, per_criterion=overrides)
        overrides["light"] = 7.0
        assert config.per_criterion == {"light": 0.9}

    @pytest.mark.parametrize("judges", [[], ()], ids=["list", "tuple"])
    def test_judges_cannot_be_empty(self, judges):
        with pytest.raises(ValueError, match="at least one judge"):
            EscalationConfig(judges, 0.5)

    def test_a_bare_decision_model_cannot_be_an_escalation_judge(self):
        with pytest.raises(ValueError, match="must be LLM judges"):
            EscalationConfig(dm(), 0.5)

    def test_a_decision_model_judge_spec_cannot_be_an_escalation_judge(self):
        with pytest.raises(ValueError, match=r"must be LLM judges.*'jev2'"):
            EscalationConfig([JudgeSpec(LLM, "a"), JudgeSpec(dm(), "jev2")], 0.5)

    def test_list_entries_must_be_judge_specs(self):
        with pytest.raises(ValueError, match=r"judges\[0\] must be a JudgeSpec; got LLMConfig"):
            EscalationConfig([LLM], 0.5)

    @pytest.mark.parametrize(
        "judges", [JudgeSpec(LLM, "a"), "gemini", None], ids=["spec", "str", "none"]
    )
    def test_judges_must_be_an_llm_config_or_a_list_of_judge_specs(self, judges):
        with pytest.raises(ValueError, match="an LLMConfig or a list of JudgeSpec"):
            EscalationConfig(judges, 0.5)

    def test_it_is_frozen_and_replace_validates_again(self):
        config = EscalationConfig(LLM, 0.5, per_criterion={"light": 0.9})
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.threshold = 0.9
        replaced = dataclasses.replace(config, threshold=0.8)
        assert replaced.threshold == 0.8
        assert replaced.judges == config.judges
        assert replaced.per_criterion == {"light": 0.9}
        with pytest.raises(ValueError, match="threshold"):
            dataclasses.replace(config, threshold=2.0)

    def test_equal_configs_compare_equal(self):
        assert EscalationConfig(LLM, 0.5) == EscalationConfig([JudgeSpec(LLM, "escalation")], 0.5)
        assert EscalationConfig(LLM, 0.5) != EscalationConfig(LLM, 0.6)

    def test_threshold_for_looks_up_a_criterion_by_name(self):
        config = EscalationConfig(LLM, 0.5, per_criterion={"light": 0.95, "myth": 0.0})
        assert config.threshold_for(LIGHT) == 0.95
        assert config.threshold_for(MYTH) == 0.0
        assert config.threshold_for(CLARITY) == 0.5  # named, no override
        assert config.threshold_for(SENTENCES) == 0.5  # unnamed: always the global threshold
        assert EscalationConfig(LLM, 0.3).threshold_for(LIGHT) == 0.3

    def test_exported_from_autorubric_and_autorubric_graders(self):
        assert autorubric.EscalationConfig is autorubric.graders.EscalationConfig
        assert "EscalationConfig" in autorubric.__all__
        assert "EscalationConfig" in autorubric.graders.__all__


# =============================================================================
# CriterionGrader(escalation=...): construction
# =============================================================================


class TestCascadeConstruction:
    def test_a_decision_model_primary_with_an_llm_escalation_judge(self, make_grader):
        grader = make_grader(judge_model_config=dm(), escalation=cascade())
        assert [j.judge_id for j in grader._judges] == ["default"]
        assert grader._escalation_judges == [JudgeSpec(LLM, "escalation")]
        assert list(grader._decision_clients) == ["default"]
        assert list(grader._clients) == ["escalation"]
        assert grader.judge_ids == ["default", "escalation"]
        # A cascade is not an ensemble: its escalation judges judge only some criteria.
        assert not grader.is_ensemble

    def test_the_primary_may_be_one_judge_spec(self, make_grader):
        grader = make_grader(judges=[JudgeSpec(dm(), "jev")], escalation=cascade(judges=panel()))
        assert grader.judge_ids == ["jev", "a", "b"]
        assert list(grader._clients) == ["a", "b"]

    def test_without_escalation_there_are_no_escalation_judges(self, make_grader):
        grader = make_grader(judges=[JudgeSpec(LLM, "x"), JudgeSpec(dm(), "jev")])
        assert grader._escalation_judges == []
        assert grader.judge_ids == ["x", "jev"]

    @pytest.mark.parametrize(
        "primary",
        [
            {"judge_model_config": LLM},
            {"judges": [JudgeSpec(LLM, "llm")]},
            {"judges": [JudgeSpec(dm(), "jev"), JudgeSpec(LLMConfig(model="x"), "x")]},
            {"judges": [JudgeSpec(dm(), "jev"), JudgeSpec(dm(model="jev-2"), "jev2")]},
        ],
        ids=["llm-single", "llm-spec", "mixed-panel", "two-decision-models"],
    )
    def test_the_primary_must_be_exactly_one_decision_model(self, make_grader, primary):
        with pytest.raises(ValueError, match="exactly one primary judge, a decision model"):
            make_grader(escalation=cascade(), **primary)

    @pytest.mark.parametrize(
        ("primary_id", "escalation_judges", "repeated"),
        [
            ("escalation", LLM, "'escalation'"),
            ("jev", [JudgeSpec(LLM, "a"), JudgeSpec(LLMConfig(model="b"), "a")], "'a'"),
            ("a", panel(), "'a'"),
        ],
        ids=["primary-vs-bare", "within-escalation", "primary-vs-panel"],
    )
    def test_judge_ids_are_unique_across_primary_and_escalation_judges(
        self, make_grader, primary_id, escalation_judges, repeated
    ):
        with pytest.raises(ValueError, match=f"unique.*{repeated}"):
            make_grader(
                judges=[JudgeSpec(dm(), primary_id)],
                escalation=cascade(judges=escalation_judges),
            )

    def test_escalation_judges_changed_into_a_decision_model_in_place_are_refused(
        self, make_grader
    ):
        config = cascade()
        config.judges.append(JudgeSpec(dm(), "jev2"))  # the frozen config's list, in place
        with pytest.raises(ValueError, match=r"must be LLM judges.*'jev2'"):
            make_grader(judge_model_config=dm(), escalation=config)

    def test_escalation_judges_emptied_in_place_are_refused(self, make_grader):
        config = cascade()
        config.judges.clear()  # the frozen config's list, in place
        with pytest.raises(ValueError, match="at least one judge"):
            make_grader(judge_model_config=dm(), escalation=config)

    @pytest.mark.parametrize("override", [7.0, math.nan])
    def test_thresholds_changed_in_place_are_refused(self, make_grader, override):
        config = cascade(0.5, per_criterion={"light": 0.9})
        overrides = config.per_criterion
        assert isinstance(overrides, dict)
        overrides["light"] = override  # the frozen config's dict, in place
        with pytest.raises(ValueError, match=r"per_criterion\['light'\] must be a number"):
            make_grader(judge_model_config=dm(), escalation=config)

    @pytest.mark.asyncio
    async def test_the_grader_is_unaffected_by_later_changes_to_the_config(
        self, fake_sdk, make_grader, llm_log
    ):
        fake_sdk.response = response(answers(c0=noul(0.9)))  # MET, 0.8
        config = cascade(0.5, per_criterion={"light": 0.9})
        grader = make_grader(judge_model_config=dm(), escalation=config)
        manifest = _serialize_grader_config(grader)

        # The caller changes its config in place after building the grader.
        config.judges.append(JudgeSpec(LLMConfig(model="b"), "b"))
        overrides = config.per_criterion
        assert isinstance(overrides, dict)
        overrides["light"] = 0.0

        assert grader.judge_ids == ["default", "escalation"]
        assert _serialize_grader_config(grader) == manifest
        report = await grade(grader)
        assert escalated_flags(report) == [True, False, False, False, False]  # 0.8 < 0.9
        assert [v.judge_id for v in report.report[0].votes] == ["default", "escalation"]
        assert [call.model for call in llm_log] == ["gemini"]

    def test_few_shot_is_allowed_and_prepared_for_the_escalation_judges(self, make_grader):
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(judges=panel()),
            training_data=training_data(),
            few_shot_config=FewShotConfig(n_examples=2),
        )
        judge_ids = {judge_id for _, judge_id in grader._criterion_examples} | {
            judge_id for _, judge_id in grader._multi_choice_examples
        }
        assert judge_ids == {"a", "b"}  # never the decision model
        assert grader._criterion_examples[(0, "a")]
        assert grader._multi_choice_examples[(2, "b")]

    def test_llm_only_settings_do_not_warn_in_a_cascade(self, make_grader):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            make_grader(
                judge_model_config=dm(),
                escalation=cascade(),
                system_prompt="custom",
                multi_choice_system_prompt="custom mc",
                shuffle_options=True,
            )

    @pytest.mark.parametrize("name", ["binary_response_format", "multi_choice_response_format"])
    def test_response_formats_are_still_rejected(self, make_grader, name):
        response_format = {
            "binary_response_format": CriterionJudgment,
            "multi_choice_response_format": MultiChoiceJudgment,
        }[name]
        with pytest.raises(ValueError, match=f"{name} cannot be combined with decision-model"):
            make_grader(judge_model_config=dm(), escalation=cascade(), **{name: response_format})


def training_data() -> RubricDataset:
    data = RubricDataset(prompt=QUERY, rubric=Rubric(RUBRIC), name="train")
    data.add_item("Light drives it.", "1", ground_truth=[MET, UNMET, "Very clear", "Formal", MET])
    data.add_item("Plants eat soil.", "2", ground_truth=[UNMET, MET, "Unclear", "Casual", UNMET])
    data.add_item("Sun helps, ok.", "3", ground_truth=[MET, UNMET, "Mostly clear", "Mixed", MET])
    data.add_item("Leaves do it.", "4", ground_truth=[UNMET, UNMET, "Unclear", "Formal", MET])
    data.add_item("Photons power it.", "5", ground_truth=[MET, MET, "Very clear", "Casual", MET])
    return data


# =============================================================================
# The escalated set: error, abstention, or confidence below the criterion's threshold
# =============================================================================


def escalated_flags(report: Any) -> list[bool]:
    return [cr.escalated for cr in report.report]


class TestEscalatedSet:
    @pytest.mark.asyncio
    async def test_low_confidence_escalates_and_confident_votes_stand(
        self, fake_sdk, make_grader, llm_log
    ):
        fake_sdk.response = response(
            answers(
                c0=noul(0.6),  # MET, 0.2
                c2=pick("Very clear", CLARITY_LABELS, 0.4),  # 0.2
            )
        )
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5))
        report = await grade(grader)

        assert escalated_flags(report) == [True, False, True, False, False]
        assert len(fake_sdk.calls) == 1
        assert [call.requirement for call in llm_log] == [LIGHT.requirement, CLARITY.requirement]
        for cr in report.report:
            votes = votes_of(cr)
            if cr.escalated:
                assert [(v.judge_id, v.superseded) for v in votes] == [
                    ("default", True),
                    ("escalation", False),
                ]
                assert votes[0].confidence == pytest.approx(0.2)
            else:
                assert [(v.judge_id, v.superseded) for v in votes] == [("default", False)]
                assert votes[0].confidence == pytest.approx(0.9 if cr.votes else 0.96)

        light, myth, clarity, tone, sentences = report.report
        assert light.final_verdict == MET and light.votes[1].reason == "says MET"
        assert clarity.final_multi_choice_verdict.selected_label == "Mostly clear"
        # Not escalated: the decision model's vote is the verdict.
        assert myth.final_verdict == UNMET
        assert tone.final_multi_choice_verdict.selected_label == "Formal"
        assert sentences.final_verdict == MET

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("threshold", "escalated"), [(0.5, False), (0.51, True)])
    async def test_confidence_at_the_threshold_does_not_escalate(
        self, fake_sdk, make_grader, threshold, escalated
    ):
        fake_sdk.response = response(answers(c0=noul(0.75)))  # MET, exactly 0.5
        grader = make_grader(judge_model_config=dm(), escalation=cascade(threshold))
        report = await grade(grader)
        assert report.report[0].votes[0].confidence == 0.5
        assert escalated_flags(report) == [escalated, False, False, False, False]

    @pytest.mark.asyncio
    async def test_a_binary_cannot_assess_escalates_however_confident(
        self, fake_sdk, make_grader, llm_log
    ):
        fake_sdk.response = response(
            {
                "c0": verdict_choice(CANNOT_ASSESS, 0.95),  # 0.925
                "c1": verdict_choice(UNMET, 0.95),
                "c2": SURE["c2"],
                "c3": SURE["c3"],
                "c4": verdict_choice(MET, 0.95),
            }
        )
        grader = make_grader(
            judge_model_config=dm(binary_framing="choice"), escalation=cascade(0.0)
        )
        report = await grade(grader)

        light = report.report[0]
        assert light.votes[0].verdict == CANNOT_ASSESS
        assert light.votes[0].confidence == pytest.approx(0.925)
        assert escalated_flags(report) == [True, False, False, False, False]
        assert light.final_verdict == MET  # the LLM's verdict
        assert [call.requirement for call in llm_log] == [LIGHT.requirement]

    @pytest.mark.asyncio
    async def test_a_multi_choice_na_escalates_however_confident(self, fake_sdk, make_grader):
        fake_sdk.response = response(answers(c3=pick(NA, TONE_LABELS, 0.97)))
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.0))
        report = await grade(grader)

        tone = report.report[3]
        assert tone.multi_choice_votes[0].na and tone.multi_choice_votes[0].superseded
        assert escalated_flags(report) == [False, False, False, True, False]
        assert tone.final_multi_choice_verdict.selected_label == "Casual"
        assert not tone.final_multi_choice_verdict.na

    @pytest.mark.asyncio
    async def test_an_unusable_answer_escalates(self, fake_sdk, make_grader):
        missing_c3 = answers()
        del missing_c3["c3"]
        fake_sdk.response = response(missing_c3)
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.0))
        report = await grade(grader)

        tone = report.report[3]
        dm_vote, llm_vote = tone.multi_choice_votes
        assert dm_vote.superseded and dm_vote.error.startswith("parse: ")
        assert llm_vote.error is None
        assert escalated_flags(report) == [False, False, False, True, False]
        # The final verdict is the LLM's genuine one, so the criterion carries no error.
        assert tone.error is None
        assert tone.final_multi_choice_verdict.selected_label == "Casual"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused"), RuntimeError("x")],
        ids=["infrastructure", "unknown"],
    )
    async def test_a_failed_request_escalates_every_criterion(
        self, fake_sdk, make_grader, llm_log, error
    ):
        fake_sdk.error = error
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.0))
        report = await grade(grader)

        assert len(fake_sdk.calls) == 1
        assert escalated_flags(report) == [True] * 5
        assert sorted(call.requirement for call in llm_log) == sorted(REQUIREMENTS)
        for cr in report.report:
            assert votes_of(cr)[0].error is not None and votes_of(cr)[0].superseded
            assert cr.error is None  # the escalation judge answered
        assert [cr.final_verdict for cr in report.report if cr.votes] == [MET, UNMET, MET]

    @pytest.mark.asyncio
    async def test_a_criterion_no_question_can_express_escalates(
        self, fake_sdk, make_grader, llm_log
    ):
        duplicate = Criterion(
            name="dup",
            weight=1.0,
            requirement="Which tone?",
            scale_type="nominal",
            options=[
                CriterionOption(label="Formal", value=1.0),
                CriterionOption(label="Formal", value=0.0),
            ],
        )
        fake_sdk.response = response({"c0": noul(0.95)})
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.0),
            scripts={"gemini": {"Which tone?": "Formal"}},
        )
        report = await grade(grader, [LIGHT, duplicate])

        assert len(fake_sdk.calls) == 1 and list(fake_sdk.calls[0][1]) == ["c0"]
        assert escalated_flags(report) == [False, True]
        assert report.report[1].multi_choice_votes[0].error.startswith("parse: ")
        assert [call.requirement for call in llm_log] == ["Which tone?"]

    @pytest.mark.asyncio
    async def test_per_criterion_thresholds_are_looked_up_by_name(
        self, fake_sdk, make_grader, llm_log
    ):
        # Every binary vote has confidence 0.8.
        fake_sdk.response = response(answers(c0=noul(0.9), c1=noul(0.1), c4=noul(0.9)))
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.3, per_criterion={"light": 0.95, "tone": 0.99}),
        )
        report = await grade(grader)

        # light (0.8 < 0.95) and tone (0.96 < 0.99) escalate; myth and the unnamed
        # criterion use the global 0.3.
        assert escalated_flags(report) == [True, False, False, True, False]
        assert {call.requirement for call in llm_log} == {LIGHT.requirement, TONE.requirement}

    @pytest.mark.asyncio
    async def test_when_nothing_escalates_no_escalation_judge_is_called(
        self, fake_sdk, make_grader, llm_log
    ):
        fake_sdk.response = response(SURE)
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5, judges=panel()))
        report = await grade(grader)

        assert escalated_flags(report) == [False] * 5
        assert llm_log == []
        assert len(fake_sdk.calls) == 1
        assert all(len(votes_of(cr)) == 1 for cr in report.report)
        assert report.judge_scores["a"] is None and report.judge_scores["b"] is None
        assert report.token_usage == TokenUsage(1234, 7, 1241)
        assert report.completion_cost is None  # the decision model is unpriced


# =============================================================================
# Aggregation over the votes that are not superseded
# =============================================================================


class TestCascadeAggregation:
    @pytest.mark.asyncio
    async def test_binary_verdict_aggregates_only_the_escalation_votes(self, fake_sdk, make_grader):
        fake_sdk.response = response(answers(c0=noul(0.6)))  # MET, unsure
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, judges=panel()),
            scripts={"a": {LIGHT.requirement: "MET"}, "b": {LIGHT.requirement: "UNMET"}},
        )
        light = (await grade(grader)).report[0]

        assert [(v.judge_id, v.verdict, v.superseded) for v in light.votes] == [
            ("default", MET, True),
            ("a", MET, False),
            ("b", UNMET, False),
        ]
        # a 1-1 tie resolves to the worst case (UNMET for a positive weight); counting the
        # decision model's MET would have made it MET 2-1.
        assert light.final_verdict == UNMET
        assert light.agreement == 0.5  # agreement counts the aggregated votes only
        assert light.escalated and light.error is None

    @pytest.mark.asyncio
    async def test_weighted_aggregation_uses_the_escalation_judges_weights(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response(answers(c0=noul(0.4)))  # UNMET, unsure
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, judges=panel(2.0, 1.0)),
            aggregation="weighted",
            scripts={"a": {LIGHT.requirement: "MET"}, "b": {LIGHT.requirement: "UNMET"}},
        )
        light = (await grade(grader)).report[0]
        assert light.final_verdict == MET

    @pytest.mark.asyncio
    async def test_ordinal_verdict_aggregates_only_the_escalation_votes(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response(answers(c2=pick("Very clear", CLARITY_LABELS, 0.4)))
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, judges=panel()),
            ordinal_aggregation="mean",
            scripts={
                "a": {CLARITY.requirement: "Unclear"},
                "b": {CLARITY.requirement: "Mostly clear"},
            },
        )
        clarity = (await grade(grader)).report[2]

        # mean(0.0, 0.6) = 0.3 is equidistant from Unclear and Mostly clear, and ties go to
        # the worst option. With the decision model's Very clear the mean would be 0.53.
        final = clarity.final_multi_choice_verdict
        assert final.aggregated_value == pytest.approx(0.3)
        assert final.selected_label == "Unclear"
        assert [v.superseded for v in clarity.multi_choice_votes] == [True, False, False]
        assert clarity.agreement == 0.5

    @pytest.mark.asyncio
    async def test_nominal_verdict_aggregates_only_the_escalation_votes(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response(answers(c3=pick("Formal", TONE_LABELS, 0.4)))
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, judges=panel()),
            nominal_aggregation="mode",
            scripts={"a": {TONE.requirement: "Formal"}, "b": {TONE.requirement: "Casual"}},
        )
        tone = (await grade(grader)).report[3]

        # A 1-1 tie goes to the worst option (Casual, 0.5 < Formal, 1.0); counting the
        # decision model's Formal would have made it Formal 2-1.
        assert tone.final_multi_choice_verdict.selected_label == "Casual"

    @pytest.mark.asyncio
    async def test_the_criterion_abstains_when_every_escalation_judge_abstains(
        self, fake_sdk, make_grader
    ):
        # The decision model is unsure but has a verdict on both; the escalation judges
        # answer, and their answer is that they cannot assess: no call fails.
        fake_sdk.response = response(
            answers(c0=noul(0.6), c2=pick("Very clear", CLARITY_LABELS, 0.4))
        )
        abstaining = {LIGHT.requirement: "CANNOT_ASSESS", CLARITY.requirement: NA}
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, judges=panel()),
            scripts={"a": abstaining, "b": abstaining},
        )
        report = await grade(grader)
        light, _, clarity, _, _ = report.report

        assert [(v.verdict, v.superseded) for v in light.votes] == [
            (MET, True),
            (CANNOT_ASSESS, False),
            (CANNOT_ASSESS, False),
        ]
        assert light.final_verdict == CANNOT_ASSESS
        assert light.final_reason == "All judges could not assess"
        assert [(v.selected_label, v.superseded) for v in clarity.multi_choice_votes] == [
            ("Very clear", True),
            (NA, False),
            (NA, False),
        ]
        final = clarity.final_multi_choice_verdict
        assert final.na and final.selected_label == NA
        # A genuine abstention is a judgment, not a failure: no error on the criterion.
        for cr in (light, clarity):
            assert cr.escalated
            assert cr.error is None and not cr.is_error
            assert all(v.error is None for v in votes_of(cr))
        assert report.cannot_assess_count == 2

    @pytest.mark.asyncio
    async def test_one_abstaining_escalation_judge_leaves_the_other_to_decide(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response(
            answers(c0=noul(0.4), c2=pick("Very clear", CLARITY_LABELS, 0.4))  # UNMET, unsure
        )
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, judges=panel()),
            scripts={
                "a": {LIGHT.requirement: "CANNOT_ASSESS", CLARITY.requirement: NA},
                "b": {LIGHT.requirement: "MET", CLARITY.requirement: "Unclear"},
            },
        )
        light, _, clarity, _, _ = (await grade(grader)).report

        # b's vote alone decides. Standing in the decision model's vote for a's abstention
        # would make light a 1-1 tie (UNMET, the worst case) and clarity the mean of
        # Unclear and Very clear (Mostly clear).
        assert light.final_verdict == MET
        assert clarity.final_multi_choice_verdict.selected_label == "Unclear"
        for cr in (light, clarity):
            assert cr.escalated and cr.error is None

    @pytest.mark.asyncio
    async def test_final_reason_joins_only_the_escalation_judges_explanations(
        self, fake_sdk, make_grader
    ):
        fake_sdk.error = typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused")
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5, judges=panel()))
        light = (await grade(grader)).report[0]

        # The decision model's failure reason stays on its superseded vote.
        assert light.votes[0].reason.startswith("Judge call failed (infrastructure)")
        assert light.final_reason == "a: says MET | b: says MET"

    @pytest.mark.asyncio
    async def test_a_kept_decision_model_vote_has_no_reason(self, fake_sdk, make_grader):
        fake_sdk.response = response(SURE)
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5))
        light = (await grade(grader)).report[0]
        assert light.final_verdict == MET and light.final_reason is None
        assert light.agreement == 1.0 and not light.escalated

    @pytest.mark.asyncio
    async def test_usage_and_cost_sum_the_decision_model_and_the_llm_calls(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response(answers(c0=noul(0.6), c1=noul(0.4)))
        grader = make_grader(
            judge_model_config=dm(input_cost_per_token=1e-6),
            escalation=cascade(0.5, judges=panel()),
        )
        report = await grade(grader)

        n_llm_calls = 2 * 2  # two escalated criteria, two escalation judges
        assert report.completion_cost == pytest.approx(1234 * 1e-6 + n_llm_calls * LLM_COST)
        assert report.token_usage == TokenUsage(1234, 7, 1241) + TokenUsage(
            10 * n_llm_calls, 5 * n_llm_calls, 15 * n_llm_calls
        )

    @pytest.mark.asyncio
    async def test_an_unpriced_decision_model_adds_nothing_to_the_cost(self, fake_sdk, make_grader):
        fake_sdk.response = response(answers(c0=noul(0.6)))
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5))
        report = await grade(grader)
        assert report.completion_cost == pytest.approx(LLM_COST)

    @pytest.mark.asyncio
    async def test_the_score_is_computed_from_the_final_verdicts(self, fake_sdk, make_grader):
        # The decision model says MET on light (unsure); the LLM says UNMET.
        fake_sdk.response = response(answers(c0=noul(0.6)))
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5),
            scripts={"gemini": {LIGHT.requirement: "UNMET"}},
        )
        report = await grade(grader)
        # light UNMET (0 of 5), myth UNMET (no penalty), clarity Very clear (2 of 2),
        # tone Formal (1 of 1), sentences MET (1 of 1): 4 of 9.
        assert report.score == pytest.approx(4 / 9)
        assert report.judge_scores["default"] == pytest.approx(9 / 9)


# =============================================================================
# Escalation-judge failures: routed as always, never a fallback to the decision model
# =============================================================================


def infrastructure() -> Exception:
    return litellm.Timeout("timed out", model="m", llm_provider="p")


class TestEscalationJudgeFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("category", "make_error"),
        [("infrastructure", infrastructure), ("parse", lambda: ValueError("bad json"))],
    )
    async def test_the_criterion_abstains_when_every_escalation_vote_abstained(
        self, fake_sdk, make_grader, category, make_error
    ):
        # The decision model is unsure but has a verdict on both; it must not be used.
        fake_sdk.response = response(
            answers(c0=noul(0.6), c2=pick("Very clear", CLARITY_LABELS, 0.4))
        )
        failing = {LIGHT.requirement: make_error(), CLARITY.requirement: make_error()}
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, judges=panel()),
            scripts={"a": failing, "b": failing},
        )
        report = await grade(grader)
        light, _, clarity, _, _ = report.report

        assert light.final_verdict == CANNOT_ASSESS
        assert clarity.final_multi_choice_verdict.na
        for cr in (light, clarity):
            votes = votes_of(cr)
            assert votes[0].superseded and votes[0].error is None
            assert [v.error.split(":")[0] for v in votes[1:]] == [category, category]
            # The error combines the aggregated (escalation) votes' errors.
            assert cr.error == " | ".join(v.error for v in votes[1:])
            assert cr.escalated
        assert report.cannot_assess_count == 2

    @pytest.mark.asyncio
    async def test_an_unknown_failure_is_the_worst_case_aggregated_like_any_vote(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response(
            answers(
                c0=noul(0.6),  # MET, unsure
                c1=noul(0.4),  # UNMET, unsure (negative weight)
                c2=pick("Very clear", CLARITY_LABELS, 0.4),
            )
        )
        boom = {
            LIGHT.requirement: RuntimeError("boom"),
            MYTH.requirement: RuntimeError("boom"),
            CLARITY.requirement: RuntimeError("boom"),
        }
        grader = make_grader(
            judge_model_config=dm(), escalation=cascade(0.5), scripts={"gemini": boom}
        )
        light, myth, clarity, _, _ = (await grade(grader)).report

        assert light.final_verdict == UNMET  # worst case for a positive weight
        assert myth.final_verdict == MET  # worst case for a negative weight
        assert clarity.final_multi_choice_verdict.selected_label == "Unclear"
        for cr in (light, myth, clarity):
            assert cr.error == "unknown: boom"

    @pytest.mark.asyncio
    async def test_one_failed_escalation_vote_leaves_the_other_to_decide(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response(answers(c0=noul(0.4)))  # UNMET, unsure
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, judges=panel()),
            scripts={"a": {LIGHT.requirement: infrastructure()}, "b": {LIGHT.requirement: "MET"}},
        )
        light = (await grade(grader)).report[0]
        assert light.final_verdict == MET
        assert light.error is None

    @pytest.mark.asyncio
    async def test_a_failed_request_and_failed_escalations_leave_the_criterion_abstained(
        self, fake_sdk, make_grader
    ):
        fake_sdk.error = typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused")
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5),
            scripts={"gemini": {LIGHT.requirement: infrastructure()}},
        )
        light = (await grade(grader)).report[0]
        assert light.final_verdict == CANNOT_ASSESS
        # Only the aggregated vote's error: the decision model's is on its superseded vote.
        assert light.error == light.votes[1].error
        assert light.error.startswith("infrastructure: ")


# =============================================================================
# judge_scores: the decision model's own score; None by role for escalation judges
# =============================================================================


class TestCascadeJudgeScores:
    @pytest.mark.asyncio
    async def test_the_decision_model_scores_its_own_verdicts_escalation_judges_none(
        self, fake_sdk, make_grader
    ):
        dm_answers = answers(c0=noul(0.6), c2=pick("Unclear", CLARITY_LABELS, 0.4))
        fake_sdk.response = response(dm_answers)
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5, judges=panel()))
        report = await grade(grader)

        standalone = await grade(make_grader(judge_model_config=dm()))
        assert list(report.judge_scores) == ["default", "a", "b"]
        # Its own verdicts on every criterion, the superseded ones included.
        assert report.judge_scores["default"] == standalone.judge_scores["default"]
        assert report.judge_scores["default"] == standalone.score
        assert report.judge_scores["a"] is None and report.judge_scores["b"] is None

    @pytest.mark.asyncio
    async def test_escalation_judges_are_none_even_when_every_criterion_escalated(
        self, fake_sdk, make_grader
    ):
        fake_sdk.error = typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused")
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5))
        report = await grade(grader)
        assert escalated_flags(report) == [True] * 5
        assert report.judge_scores["escalation"] is None
        assert report.judge_scores["default"] is not None


# =============================================================================
# criterion_idx stability: an escalation judge's prompt equals a non-cascade grader's
# =============================================================================


class TestCriterionIndexStability:
    @pytest.mark.asyncio
    async def test_escalation_prompts_equal_a_non_cascade_grader_with_same_seed_and_judge_id(
        self, fake_sdk, make_grader
    ):
        seed = 7
        few_shot = {
            "training_data": training_data(),
            "few_shot_config": FewShotConfig(n_examples=2),
        }
        # Escalate myth, clarity and tone (indices 1, 2, 3); keep light and sentences.
        fake_sdk.response = response(
            answers(
                c1=noul(0.4),
                c2=pick("Very clear", CLARITY_LABELS, 0.4),
                c3=pick("Formal", TONE_LABELS, 0.4),
            )
        )
        cascade_grader = make_grader(
            judge_model_config=dm(), escalation=cascade(0.5), seed=seed, **few_shot
        )
        reference = make_grader(judges=[JudgeSpec(LLM, "escalation")], seed=seed, **few_shot)
        exemplar = "Light drives the conversion of water and carbon dioxide into glucose."
        cascaded = await grade(cascade_grader, reference_submission=exemplar)
        referenced = await grade(reference, reference_submission=exemplar)

        cascade_calls = {c.requirement: c for c in cascade_grader._clients["escalation"].calls}
        reference_calls = {c.requirement: c for c in reference._clients["escalation"].calls}
        assert set(cascade_calls) == {MYTH.requirement, CLARITY.requirement, TONE.requirement}
        for requirement, call in cascade_calls.items():
            expected = reference_calls[requirement]
            assert call.system_prompt == expected.system_prompt
            assert call.user_prompt == expected.user_prompt
            # Every part of the prompt is in play: the query, the reference submission and
            # few-shot examples.
            assert f"<input>{QUERY}</input>" in call.user_prompt
            assert f"<reference_submission>\n{exemplar}\n</reference_submission>" in (
                call.user_prompt
            )
            assert "<examples>" in call.user_prompt

        shuffles = []
        for idx in (2, 3):
            cascade_vote = cascaded.report[idx].multi_choice_votes[1]
            reference_vote = referenced.report[idx].multi_choice_votes[0]
            assert cascade_vote.judge_id == reference_vote.judge_id == "escalation"
            assert cascade_vote.shuffle_order == reference_vote.shuffle_order
            shuffles.append(cascade_vote.shuffle_order)
        assert any(order != sorted(order) for order in shuffles)  # options were shuffled

    @pytest.mark.asyncio
    async def test_few_shot_never_reaches_the_decision_model(self, fake_sdk, make_grader):
        fake_sdk.response = response(answers(c0=noul(0.6)))
        await grade(make_grader(judge_model_config=dm(), escalation=cascade(0.5)))
        await grade(
            make_grader(
                judge_model_config=dm(),
                escalation=cascade(0.5),
                training_data=training_data(),
                few_shot_config=FewShotConfig(n_examples=2),
            )
        )
        without, with_few_shot = fake_sdk.calls
        assert without[0] == with_few_shot[0]  # state
        assert {k: q.model_dump() for k, q in without[1].items()} == {
            k: q.model_dump() for k, q in with_few_shot[1].items()
        }

    @pytest.mark.asyncio
    async def test_escalation_judges_see_the_rubric_guidelines(
        self, fake_sdk, make_grader, llm_log
    ):
        fake_sdk.response = response(answers(c0=noul(0.6)))
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5))
        await grade(grader, Rubric(RUBRIC, guidelines="Judge at a grade-8 level."))

        assert fake_sdk.calls[0][0]["guidelines"] == "Judge at a grade-8 level."
        (call,) = llm_log
        assert call.user_prompt.startswith("<guidelines>\nJudge at a grade-8 level.")


# =============================================================================
# One decision-model request per item, whatever escalates
# =============================================================================


class TestOneRequestPerItem:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("dm_answers", "n_escalated"),
        [
            (SURE, 0),
            (answers(c0=noul(0.6), c3=pick("Casual", TONE_LABELS, 0.3)), 2),
            ({qid: noul(0.5) if qid in ("c0", "c1", "c4") else a for qid, a in SURE.items()}, 3),
            (
                {
                    "c0": noul(0.5),
                    "c1": noul(0.5),
                    "c2": pick("Unclear", CLARITY_LABELS, 0.25),
                    "c3": pick(NA, TONE_LABELS, 0.99),
                    "c4": noul(0.5),
                },
                5,
            ),
        ],
        ids=["none", "some", "binary", "all"],
    )
    async def test_one_request_per_item_and_escalations_only_to_the_llms(
        self, fake_sdk, make_grader, llm_log, dm_answers, n_escalated
    ):
        fake_sdk.response = response(dm_answers)
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5, judges=panel()))
        report = await grade(grader)

        assert len(fake_sdk.calls) == 1
        assert sum(escalated_flags(report)) == n_escalated
        # Every question of the rubric went out in the one request.
        assert list(fake_sdk.calls[0][1]) == ["c0", "c1", "c2", "c3", "c4"]
        assert len(llm_log) == 2 * n_escalated
        assert {call.model for call in llm_log} <= {"a", "b"}

    @pytest.mark.asyncio
    async def test_an_evaluate_run_makes_one_request_per_item(
        self, fake_sdk, make_grader, llm_log, tmp_path
    ):
        fake_sdk.response = response(answers(c0=noul(0.6)))
        data = RubricDataset(prompt=QUERY, rubric=Rubric(RUBRIC), name="cascade-run")
        for i in range(3):
            data.add_item(f"{SUBMISSION} ({i})", f"item {i}")
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5))
        result = await evaluate(
            data, grader, show_progress=False, experiments_dir=tmp_path, experiment_name="c"
        )

        assert result.successful_items == 3
        assert len(fake_sdk.calls) == 3
        assert len(llm_log) == 3  # light escalates on every item
        for item in result.item_results:
            assert [cr.escalated for cr in item.report.report] == [
                True,
                False,
                False,
                False,
                False,
            ]


# =============================================================================
# Persistence: checkpoints and the manifest
# =============================================================================


class TestCascadePersistence:
    @pytest.mark.asyncio
    async def test_cascade_reports_round_trip_through_the_checkpoint(self, fake_sdk, make_grader):
        fake_sdk.response = response(answers(c0=noul(0.6), c3=pick("Formal", TONE_LABELS, 0.4)))
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5, judges=panel()))
        report = await grade(grader)
        item = DataItem(submission=SUBMISSION, description="x")

        data = json.loads(json.dumps(ItemResult(0, item, report, 1.0).to_dict()))
        criteria = data["report"]["criterion_reports"]
        assert [c["escalated"] for c in criteria] == [True, False, False, True, False]
        assert [v["superseded"] for v in criteria[0]["votes"]] == [True, False, False]
        assert [v["superseded"] for v in criteria[3]["multi_choice_votes"]] == [
            True,
            False,
            False,
        ]
        assert data["report"]["judge_scores"]["a"] is None

        restored = ItemResult.from_dict(data, item).report
        assert isinstance(restored, EnsembleEvaluationReport)
        assert restored.report == report.report
        assert restored.judge_scores == report.judge_scores
        assert restored.score == report.score

    def test_the_manifest_records_the_escalation_config(self, make_grader):
        grader = make_grader(
            judge_model_config=dm(),
            escalation=EscalationConfig(
                judges=[
                    JudgeSpec(LLMConfig(model="a", temperature=0.2), "a", 2.0),
                    JudgeSpec(LLMConfig(model="b", max_parallel_requests=4), "b"),
                ],
                threshold=0.72,
                per_criterion={"light": 0.95},
            ),
        )
        config = _serialize_grader_config(grader)

        assert [j["judge_id"] for j in config["judges"]] == ["default"]
        assert config["judges"][0]["judge_kind"] == "decision_model"
        assert config["escalation"] == {
            "judges": [
                {
                    "judge_id": "a",
                    "judge_kind": "llm",
                    "model": "a",
                    "temperature": 0.2,
                    "weight": 2.0,
                    "max_parallel_requests": None,
                },
                {
                    "judge_id": "b",
                    "judge_kind": "llm",
                    "model": "b",
                    "temperature": None,
                    "weight": 1.0,
                    "max_parallel_requests": 4,
                },
            ],
            "threshold": 0.72,
            "per_criterion": {"light": 0.95},
        }
        assert API_KEY not in json.dumps(config)

    def test_a_bare_escalation_judge_and_no_overrides(self, make_grader):
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5))
        escalation = _serialize_grader_config(grader)["escalation"]
        assert [j["judge_id"] for j in escalation["judges"]] == ["escalation"]
        assert escalation["per_criterion"] is None

    def test_a_grader_without_escalation_records_no_escalation_key(self, make_grader):
        for kwargs in ({"judge_model_config": LLM}, {"judge_model_config": dm()}):
            assert "escalation" not in _serialize_grader_config(make_grader(**kwargs))


# =============================================================================
# EvalRunner: detailed progress lists every judge
# =============================================================================


class TestEvalRunnerJudgeIds:
    def test_detailed_progress_lists_the_escalation_judges(self, make_grader):
        grader = make_grader(judge_model_config=dm(), escalation=cascade(0.5, judges=panel()))
        runner = EvalRunner(RubricDataset(prompt=QUERY, rubric=Rubric(RUBRIC)), grader)
        assert runner._judge_ids == ["default", "a", "b"]

    def test_other_graders_keep_the_judges_duck_typing(self):
        grader = MagicMock()
        grader._judges = [SimpleNamespace(judge_id="x"), SimpleNamespace(judge_id="y")]
        runner = EvalRunner(RubricDataset(prompt=QUERY, rubric=Rubric(RUBRIC)), grader)
        assert runner._judge_ids == ["x", "y"]


# =============================================================================
# Graders whose __init__ never ran have no cascade
# =============================================================================

# The instance state a CriterionGrader gained with decision models and the cascade (set by
# __init__, or by judge once it has warned). A grader restored from a pickle made before
# they existed has none of it.
STATE_ADDED_WITH_THE_CASCADE = ("_decision_clients", "_escalation", "_escalation_names_warned")


class GradeOnly(CriterionGrader):
    """A test double that overrides ``grade`` and never calls ``CriterionGrader.__init__``."""

    def __init__(self) -> None:
        pass

    async def grade(
        self,
        to_grade: Any,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> EvaluationReport:
        return EvaluationReport(score=1.0, raw_score=1.0, report=None)


def one_item_dataset() -> RubricDataset:
    data = RubricDataset(prompt=QUERY, rubric=Rubric(RUBRIC), name="one-item")
    data.add_item(SUBMISSION, "item")
    return data


class TestGradersWithoutCascadeState:
    @pytest.mark.asyncio
    async def test_a_grader_pickled_before_the_cascade_grades_and_evaluates(
        self, make_grader, tmp_path
    ):
        grader = make_grader(judge_model_config=LLM, seed=7)
        restored = CriterionGrader.__new__(CriterionGrader)
        restored.__dict__.update(
            {k: v for k, v in vars(grader).items() if k not in STATE_ADDED_WITH_THE_CASCADE}
        )

        assert restored.judge_ids == ["default"]
        assert "escalation" not in _serialize_grader_config(restored)
        assert (await grade(restored)).report == (await grade(grader)).report
        result = await evaluate(
            one_item_dataset(),
            restored,
            show_progress=False,
            experiments_dir=tmp_path,
            experiment_name="restored",
        )
        assert result.successful_items == 1

    @pytest.mark.asyncio
    async def test_a_subclass_that_skips_init_runs_through_evaluate(self, tmp_path):
        data = one_item_dataset()
        assert EvalRunner(data, GradeOnly())._judge_ids == []
        result = await evaluate(
            data, GradeOnly(), show_progress=False, experiments_dir=tmp_path, experiment_name="d"
        )
        assert [item.report.score for item in result.item_results] == [1.0]


# =============================================================================
# per_criterion names that match no criterion
# =============================================================================


def unknown_name_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "per_criterion" in str(w.message)
    ]


class TestUnknownPerCriterionNames:
    @pytest.mark.asyncio
    async def test_a_direct_grade_warns_once_per_grader(self, fake_sdk, make_grader):
        fake_sdk.response = response(SURE)
        overrides = {"light": 0.9, "factuality": 0.9, "style": 0.8}
        grader = make_grader(
            judge_model_config=dm(), escalation=cascade(0.5, per_criterion=overrides)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await grade(grader)
            await grade(grader)
        (warning,) = unknown_name_warnings(caught)
        message = str(warning.message)
        assert "'factuality', 'style'" in message and "'light'" not in message
        assert "the rubric being graded" in message
        assert warning.filename == __file__  # attributed to the caller

        # Another grader warns again.
        other = make_grader(
            judge_model_config=dm(), escalation=cascade(0.5, per_criterion=overrides)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await grade(other)
        assert len(unknown_name_warnings(caught)) == 1

    @pytest.mark.asyncio
    async def test_no_warning_when_every_name_matches(self, fake_sdk, make_grader):
        fake_sdk.response = response(SURE)
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, per_criterion={"light": 0.9, "tone": 0.1}),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await grade(grader)
        assert unknown_name_warnings(caught) == []

    @pytest.mark.asyncio
    async def test_grading_a_rubric_with_every_name_leaves_the_warning_to_come(
        self, fake_sdk, make_grader
    ):
        """The one warning per grader is used up only by being issued, so a rubric that
        has every name does not silence it for a later rubric that lacks one."""
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, per_criterion={"light": 0.9, "myth": 0.9}),
        )
        fake_sdk.response = response({"c0": noul(0.95), "c1": noul(0.05)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await grade(grader, [LIGHT, MYTH])
        assert unknown_name_warnings(caught) == []

        fake_sdk.response = response({"c0": noul(0.95)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await grade(grader, [LIGHT])
            await grade(grader, [LIGHT])
        (warning,) = unknown_name_warnings(caught)
        assert "names 'myth' match no criterion" in str(warning.message)

    @pytest.mark.asyncio
    async def test_warnings_as_errors_fail_every_direct_grade_before_any_request(
        self, fake_sdk, make_grader
    ):
        """Under warnings-as-errors the warning raises instead of being issued, so it is not
        used up: every direct grade of a rubric lacking a name fails, not only the first."""
        fake_sdk.response = response(SURE)
        grader = make_grader(
            judge_model_config=dm(), escalation=cascade(0.5, per_criterion={"nope": 0.5})
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            for _ in range(2):
                with pytest.raises(UserWarning, match="names 'nope' match no criterion"):
                    await grade(grader)
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_evaluate_checks_names_once_against_every_items_rubric(
        self, fake_sdk, make_grader, tmp_path
    ):
        """Per-item rubrics: a name absent from one item's rubric but present in another's
        is known, so only a name no item's rubric has is reported, once, at run start."""
        fake_sdk.response = response({"c0": noul(0.95)})
        data = RubricDataset(prompt=QUERY, rubric=None, name="per-item")
        data.add_item("a", "light only", rubric=Rubric([LIGHT]))
        data.add_item("b", "myth only", rubric=Rubric([MYTH]))
        data.add_item("c", "light again", rubric=Rubric([LIGHT]))
        overrides = {"light": 0.9, "myth": 0.9, "nope": 0.5}
        grader = make_grader(
            judge_model_config=dm(), escalation=cascade(0.5, per_criterion=overrides)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await evaluate(
                data, grader, show_progress=False, experiments_dir=tmp_path, experiment_name="u"
            )
        assert result.successful_items == 3
        (warning,) = unknown_name_warnings(caught)
        message = str(warning.message)
        assert "'nope'" in message and "'light'" not in message and "'myth'" not in message
        assert "any item's rubric" in message
        assert warning.filename == __file__  # attributed to the caller

        # The run's check does not stand in for a direct grade's own check.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await grade(grader, [LIGHT])
        (warning,) = unknown_name_warnings(caught)
        assert "'myth', 'nope'" in str(warning.message)

    @pytest.mark.asyncio
    async def test_evaluate_warns_nothing_when_every_name_is_in_some_items_rubric(
        self, fake_sdk, make_grader, tmp_path
    ):
        fake_sdk.response = response({"c0": noul(0.95)})
        data = RubricDataset(prompt=QUERY, rubric=None, name="per-item")
        data.add_item("a", "light only", rubric=Rubric([LIGHT]))
        data.add_item("b", "myth only", rubric=Rubric([MYTH]))
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, per_criterion={"light": 0.9, "myth": 0.9}),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await evaluate(
                data, grader, show_progress=False, experiments_dir=tmp_path, experiment_name="k"
            )
        assert unknown_name_warnings(caught) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("show_progress", [False, True])
    async def test_fill_ground_truth_checks_names_once_against_every_items_rubric(
        self, fake_sdk, make_grader, show_progress
    ):
        """Like ``evaluate``: a name in some item's rubric is known, and only a name no
        item's rubric has is reported, once, before grading."""
        fake_sdk.response = response({"c0": noul(0.95)})
        data = RubricDataset(prompt=QUERY, rubric=None, name="per-item")
        data.add_item("a", "light only", rubric=Rubric([LIGHT]))
        data.add_item("b", "myth only", rubric=Rubric([MYTH]))
        data.add_item("c", "light again", rubric=Rubric([LIGHT]))
        overrides = {"light": 0.9, "myth": 0.9, "nope": 0.5}
        grader = make_grader(
            judge_model_config=dm(), escalation=cascade(0.5, per_criterion=overrides)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labeled = await fill_ground_truth(data, grader, show_progress=show_progress)
        assert len(labeled) == 3
        (warning,) = unknown_name_warnings(caught)
        message = str(warning.message)
        assert "'nope'" in message and "'light'" not in message and "'myth'" not in message
        assert "any item's rubric" in message
        assert warning.filename == __file__  # attributed to the caller

    @pytest.mark.asyncio
    async def test_fill_ground_truth_labels_every_item_under_warnings_as_errors(
        self, fake_sdk, make_grader
    ):
        """A name absent from one item's rubric but present in another's never warns, so
        a warnings-as-errors filter cannot fail (and drop) the item that lacks it."""
        fake_sdk.response = response({"c0": noul(0.95)})
        data = RubricDataset(prompt=QUERY, rubric=None, name="per-item")
        data.add_item("a", "light only", rubric=Rubric([LIGHT]))
        data.add_item("b", "myth only", rubric=Rubric([MYTH]))
        grader = make_grader(
            judge_model_config=dm(),
            escalation=cascade(0.5, per_criterion={"light": 0.9, "myth": 0.9}),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            labeled = await fill_ground_truth(data, grader, show_progress=False)
        assert [item.description for item in labeled] == ["light only", "myth only"]


# =============================================================================
# The cannot-assess strategy applies to cascade abstentions like any other
# =============================================================================


@pytest.mark.asyncio
async def test_an_abstained_escalation_counts_under_the_cannot_assess_strategy(
    fake_sdk, make_grader
):
    fake_sdk.response = response(answers(c0=noul(0.6)))
    grader = make_grader(
        judge_model_config=dm(),
        escalation=cascade(0.5),
        cannot_assess_config=CannotAssessConfig(strategy="fail"),
        scripts={"gemini": {LIGHT.requirement: infrastructure()}},
    )
    report = await grade(grader)
    assert report.report[0].final_verdict == CANNOT_ASSESS
    # FAIL counts the abstained light as UNMET: 4 of 9, as in the scoring test above.
    assert report.score == pytest.approx(4 / 9)
