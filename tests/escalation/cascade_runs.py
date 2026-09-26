"""Scripted judges and datasets for the cascade replay and diagnostics tests.

Nothing here reaches the network or spends money:

- ``ScriptedSDK`` stands in for the TypeSafe SDK client (the recording ``FakeSDK`` of
  ``tests/decision/conftest.py``, answering each request from a script keyed by the
  submission in its state), so the real ``DecisionModelClient`` builds every request.
- ``ScriptedLLM`` stands in for ``LLMClient``. Its answers depend on the prompt it is sent:
  unless its script names an answer, a binary verdict and a multi-choice position are drawn
  from a hash of the model and the prompts, and every explanation carries that hash. Two
  calls agree only if their prompts are identical, option order included, so a replay can
  equal a live cascade only when the LLM run used the cascade's ``seed`` and ``judge_id``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import litellm
import typesafe_sdk
from decision.conftest import FakeSDK, FakeSDKClient
from typesafe_sdk import SystemOneResponse

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DecisionModelConfig,
    EvalResult,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.types import CANONICAL_NA_OPTION, CriterionJudgment, MultiChoiceJudgment

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CANNOT_ASSESS = CriterionVerdict.CANNOT_ASSESS
NA = CANONICAL_NA_OPTION.label

API_KEY = "ts-test-key-never-persisted-e5c1"
QUERY = "Explain photosynthesis."

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
RUBRIC = Rubric([LIGHT, MYTH, CLARITY, TONE, SENTENCES])

# The options a decision model is offered for each multi-choice criterion: the rubric's,
# in order, plus the NA option auto_na_option guarantees.
CLARITY_LABELS = ["Unclear", "Mostly clear", "Very clear", NA]
TONE_LABELS = ["Formal", "Casual", "Mixed", NA]

# The cascade: escalate below 0.5; tone unless the decision model is all but certain, myth
# only on an error or an abstention. The LLM runs share its seed.
SEED = 7
THRESHOLD = 0.5
PER_CRITERION = {"tone": 0.99, "myth": 0.0}
GEMINI = [JudgeSpec(LLMConfig(model="gemini"), "escalation")]
PANEL = [JudgeSpec(LLMConfig(model="a"), "a"), JudgeSpec(LLMConfig(model="b"), "b")]
REQUEST_ERROR = typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused")

DM_INPUT_TOKENS = 1000
DM_PRICE = 1e-6  # USD per input token: 0.001 per request
LLM_COST = 0.002  # USD per LLM call
LLM_USAGE = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def dm(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(
        **{"model": "jev-latest", "api_key": API_KEY, "input_cost_per_token": DM_PRICE, **overrides}
    )


# =============================================================================
# Decision-model answers (confidence in comments: clamp((K p - 1) / (K - 1), 0, 1))
# =============================================================================


def noul(p: float) -> dict[str, Any]:
    """A Noul answer: P(MET) = p, confidence 2|p - 0.5|."""
    return {"type": "noul", "noul": p}


def pick(selected: str, labels: list[str], p: float) -> dict[str, Any]:
    """A Choice answer selecting ``selected`` with probability ``p``, the rest spread evenly."""
    rest = (1 - p) / (len(labels) - 1)
    probabilities = {label: p if label == selected else rest for label in labels}
    return {"type": "choice", "choice": selected, "confidence": 0.0, "probabilities": probabilities}


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


@dataclass
class ScriptedSDK(FakeSDK):
    """A ``FakeSDK`` whose requests are answered by the submission in their state.

    ``scripts`` maps a submission to the answers for its request, keyed by question id (a
    question without an answer is left out of the response), or to an exception to raise.
    """

    scripts: dict[str, Any] = field(default_factory=dict)

    def answer(self, state: dict[str, str]) -> SystemOneResponse:
        script = self.scripts[state["submission"]]
        if isinstance(script, BaseException):
            raise script
        payload = {
            "model": "jev-test",
            "usage": {"input_tokens": DM_INPUT_TOKENS, "output_tokens": 3},
            "answers": script,
        }
        return SystemOneResponse.model_validate_json(json.dumps(payload))


class ScriptedSDKClient(FakeSDKClient):
    """A ``FakeSDKClient`` answering from its ``ScriptedSDK``'s scripts."""

    sdk: ScriptedSDK

    async def system_one(self, state: Any, questions: Any, **kwargs: Any) -> SystemOneResponse:
        self.calls.append((state, questions, kwargs))
        return self.sdk.answer(state)


# =============================================================================
# The LLM judge, whose answers depend on the prompt
# =============================================================================

_SECTION = r"<{0}>\n(.*?)\n</{0}>"
_NA_SUFFIX = " (cannot assess / not applicable)"


class ScriptedLLM:
    """Stand-in for ``LLMClient``: answers from its script, else from a hash of the prompt.

    ``script`` maps ``(submission, requirement)`` to a verdict value (binary), an option label
    (multi-choice, answered as the label's position among the options as presented), or an
    exception to raise. Any other criterion is answered from the hash of the model and the
    prompts: MET or UNMET, or the option at a hashed position among those presented, so a
    shuffled option order changes the answer. Every explanation carries the hash.
    """

    def __init__(self, model: str, script: dict[tuple[str, str], Any]) -> None:
        self.model = model
        self.script = script
        self.user_prompts: list[str] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult:
        self.user_prompts.append(user_prompt)
        multi_choice = "<options>" in user_prompt
        # A requirement can span lines (HealthBench's do).
        section = _SECTION.format("question" if multi_choice else "criterion")
        found = re.search(section, user_prompt, re.S)
        assert found is not None
        requirement = found.group(1)
        submission = re.findall(_SECTION.format("submission"), user_prompt, re.S)[-1]
        digest = hashlib.sha256(f"{self.model}\n{system_prompt}\n{user_prompt}".encode())
        hashed = int(digest.hexdigest(), 16)
        explanation = f"{self.model} read prompt {digest.hexdigest()[:12]}"

        answer = self.script.get((submission, requirement))
        if isinstance(answer, BaseException):
            raise answer
        parsed: Any
        if multi_choice:
            options = re.search(_SECTION.format("options"), user_prompt, re.S)
            assert options is not None
            presented = [
                line.removesuffix(_NA_SUFFIX)
                for line in re.findall(r"^\d+\. (.*)$", options.group(1), re.M)
            ]
            position = presented.index(answer) if answer is not None else hashed % len(presented)
            parsed = MultiChoiceJudgment(selected_option=position + 1, explanation=explanation)
        else:
            verdict = CriterionVerdict(answer) if answer is not None else [MET, UNMET][hashed % 2]
            parsed = CriterionJudgment(criterion_status=verdict, explanation=explanation)
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=LLM_USAGE,
            cost=LLM_COST,
            parsed=parsed,
        )


# =============================================================================
# The dataset: one shared rubric; the decision model's answers and ground truth per item
# =============================================================================


@dataclass(frozen=True)
class Item:
    submission: str
    dm_answers: Any  # answers by question id, or an exception the request raises
    ground_truth: list[CriterionVerdict | str]


ITEMS = [
    # Sure of everything.
    Item("Light turns water and air into sugar.", SURE, [MET, UNMET, "Very clear", "Formal", MET]),
    # Unsure of light (0.2) and of clarity (0.2).
    Item(
        "The sun feeds plants somehow.",
        answers(c0=noul(0.6), c2=pick("Very clear", CLARITY_LABELS, 0.4)),
        [MET, UNMET, "Mostly clear", "Casual", MET],
    ),
    # Abstains on tone, however confident.
    Item(
        "Plants make food from light, water and carbon dioxide.",
        answers(c3=pick(NA, TONE_LABELS, 0.97)),
        [MET, UNMET, "Very clear", "Formal", CANNOT_ASSESS],
    ),
    # No answer for myth: a parse failure.
    Item(
        "Chlorophyll absorbs light.",
        {key: value for key, value in SURE.items() if key != "c1"},
        [MET, UNMET, "Mostly clear", "Formal", UNMET],
    ),
    # The request fails: every criterion errors.
    Item(
        "Plants eat soil and light.",
        None,  # replaced by the error in ``decision_model_scripts``
        [MET, MET, "Unclear", "Mixed", MET],
    ),
    # Unsure of myth (0.1), tone (1/3) and sentences (0.4).
    Item(
        "photosynthesis = light -> sugar",
        answers(
            c1=noul(0.45),
            c3=pick("Casual", TONE_LABELS, 0.5),
            c4=noul(0.7),
        ),
        [UNMET, UNMET, "Unclear", "Casual", UNMET],
    ),
    # Exactly as confident of light as a threshold of 0.5 (0.5), which keeps it.
    Item(
        "Light is the energy source.",
        answers(c0=noul(0.75)),
        [MET, UNMET, "Mostly clear", "Formal", MET],
    ),
]
FAILED_REQUEST = ITEMS[4].submission
TERSE = ITEMS[5].submission


def decision_model_scripts(error: BaseException) -> dict[str, Any]:
    """Every item's decision-model answers, the failing item's request raising ``error``."""
    return {
        item.submission: error if item.dm_answers is None else item.dm_answers for item in ITEMS
    }


def llm_scripts() -> dict[tuple[str, str], Any]:
    """The LLM answers named ahead of the hash: two failed calls, so failures replay too.

    On the terse item, sentences fails with an unknown error (the worst-case verdict,
    ``error`` set) and tone with an infrastructure error (an NA abstention).
    """
    return {
        (TERSE, SENTENCES.requirement): RuntimeError("LLM client crashed"),
        (TERSE, TONE.requirement): litellm.RateLimitError(
            "rate limited", model="m", llm_provider="p"
        ),
    }


def dataset(items: list[Item] = ITEMS, *, name: str = "photosynthesis") -> RubricDataset:
    data = RubricDataset(prompt=QUERY, rubric=RUBRIC, name=name)
    for i, item in enumerate(items):
        data.add_item(item.submission, f"item {i}", ground_truth=list(item.ground_truth))
    return data


# =============================================================================
# A dataset with a rubric per item (no shared rubric, as HealthBench)
# =============================================================================

# Submission, rubric, ground truth and the decision model's answers (at THRESHOLD, which
# escalates below a confidence of 0.5), per item.
PER_ITEM = [
    (
        "Leaves catch light.",
        [LIGHT, CLARITY],
        [MET, "Mostly clear"],
        {
            "c0": noul(0.6),  # MET, 0.2: escalated
            "c1": pick("Very clear", CLARITY_LABELS, 0.97),  # kept, wrong
        },
    ),
    (
        "Soil feeds plants.",
        [MYTH, TONE, SENTENCES],
        [MET, "Formal", UNMET],
        {
            "c0": noul(0.9),  # MET, 0.8: kept, right
            "c1": pick("Casual", TONE_LABELS, 0.4),  # 0.2: escalated, wrong
            "c2": noul(0.3),  # UNMET, 0.4: escalated, right
        },
    ),
    (
        "Photosynthesis uses light.",
        [LIGHT, SENTENCES],
        [MET, UNMET],
        {
            "c0": noul(0.95),  # kept, right
            "c1": noul(0.95),  # kept, wrong
        },
    ),
]
# The LLM's answers on the escalated criteria of PER_ITEM.
PER_ITEM_LLM_SCRIPT: dict[tuple[str, str], Any] = {
    ("Leaves catch light.", LIGHT.requirement): "UNMET",  # wrong
    ("Soil feeds plants.", TONE.requirement): "Formal",  # right
    ("Soil feeds plants.", SENTENCES.requirement): "MET",  # wrong
}


def per_item_dataset() -> RubricDataset:
    """``PER_ITEM`` as a dataset: every item has its own rubric, and there is no shared one."""
    data = RubricDataset(prompt=QUERY, name="per-item")
    for submission, criteria, truth, _ in PER_ITEM:
        data.add_item(submission, "item", ground_truth=list(truth), rubric=Rubric(criteria))
    return data


def per_item_decision_model_scripts() -> dict[str, Any]:
    """The decision model's answers for ``PER_ITEM``, by submission."""
    return {submission: answers for submission, _, _, answers in PER_ITEM}


@dataclass
class Runs:
    """A live cascade and the two runs a replay of it is built from, on one dataset, with
    the cascade's grader and the LLM run's (whose LLM clients are ``ScriptedLLM``s)."""

    data: RubricDataset
    live: EvalResult
    dm: EvalResult
    llm: EvalResult
    cascade: CriterionGrader
    llm_grader: CriterionGrader

    def user_prompts(self, grader: CriterionGrader) -> list[str]:
        """Every user prompt ``grader``'s LLM judges were sent, in no particular order."""
        return [prompt for client in grader._clients.values() for prompt in client.user_prompts]
