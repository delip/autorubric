"""Rubric guidelines reaching decision-model judges, alone and next to LLM judges.

A decision model sees a rubric's guidelines once per request, as the first field of the
shared state; the framed binary questions add one fixed clause naming them and their
precedence, and bare Noul and multi-choice questions see them only through the state. A
rubric without guidelines sends exactly the request it sent before guidelines existed. LLM
judges of the same grader get the guidelines block in front of every user prompt.

Expected requests are written out literally. The real ``DecisionModelClient`` runs on the
recording fake SDK client of ``conftest.py`` (``fake_sdk``); LLM judges are recording fakes
patched in for ``LLMClient``. Nothing reaches the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from typesafe_sdk import SystemOneResponse

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DataItem,
    DecisionModelConfig,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
    evaluate,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.types import CANONICAL_NA_OPTION, CriterionJudgment, MultiChoiceJudgment

API_KEY = "ts-test-key-never-persisted-5d21"
NA_LABEL = CANONICAL_NA_OPTION.label
QUERY = "Explain photosynthesis."
SUBMISSION = "Plants use light to turn water and carbon dioxide into sugar."
REFERENCE = "Photosynthesis uses light energy to make glucose from CO2 and water."
GUIDELINES = (
    "Readers are grade 8 students.\n"
    "'Mentions' means names or clearly describes; {placeholders} are literal text."
)

MET_DEF = "The thing described in the criterion IS present in the submission"
UNMET_DEF = "The thing described in the criterion IS NOT present in the submission"
NEG_MET_DEF = "The submission advocates, states, or recommends the problematic thing"
NEG_UNMET_DEF = "The submission does NOT make this error, OR mentions it only to warn against it"
TASK = "Determine whether this criterion is satisfied by the `submission`."
TASK_OUTPUT = (
    "Determine whether this criterion is satisfied by the `output`; the `thinking` is context only."
)
GUIDELINES_CLAUSE = "Apply the `guidelines`; the criterion text governs."
REFERENCE_SUBMISSION = (
    "Use the `reference_submission` only to calibrate expectations; judge the `submission` on "
    "its own merits, not by its resemblance to the reference."
)
REFERENCE_OUTPUT = (
    "Use the `reference_submission` only to calibrate expectations; judge the `output` on "
    "its own merits, not by its resemblance to the reference."
)
PRECEDENCE_RULE = (
    "These guidelines apply to every criterion. The criterion text governs; the guidelines "
    "clarify how to apply it."
)

CRITERIA = [
    Criterion(name="light", weight=5.0, requirement="Mentions light"),
    Criterion(name="myth", weight=-3.0, requirement="Claims plants eat soil"),
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
    "c1": {"type": "noul", "noul": 0.2},
    "c2": {
        "type": "choice",
        "choice": "Mostly clear",
        "confidence": 0.5,
        "probabilities": {"Unclear": 0.1, "Mostly clear": 0.7, "Very clear": 0.15, NA_LABEL: 0.05},
    },
}
CLARITY_CHOICE = {
    "type": "choice",
    "instructions": "How clear is the explanation?",
    "criteria": {"Unclear": None, "Mostly clear": None, "Very clear": None, NA_LABEL: None},
}


def dm(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(**{"model": "jev-latest", "api_key": API_KEY, **overrides})


def response() -> SystemOneResponse:
    payload = {"model": "jev-test", "usage": {"input_tokens": 50}, "answers": ANSWERS}
    return SystemOneResponse.model_validate_json(json.dumps(payload))


def request_json(call: tuple[Any, Any, dict[str, Any]]) -> str:
    """One recorded request (state, then questions) as JSON text, so key order counts: the
    order of the state fields and of the questions."""
    state, questions, _ = call
    wire = {qid: q.model_dump(mode="json") for qid, q in questions.items()}
    return json.dumps([state, wire])


def request(call: tuple[Any, Any, dict[str, Any]]) -> list[Any]:
    """One recorded request as parsed JSON: ``[state, questions]``."""
    return json.loads(request_json(call))


def framed_questions(light_instructions: str, myth_instructions: str) -> dict[str, dict[str, Any]]:
    return {
        "c0": {
            "type": "noul",
            "instructions": light_instructions,
            "criteria": {"true": MET_DEF, "false": UNMET_DEF},
        },
        "c1": {
            "type": "noul",
            "instructions": myth_instructions,
            "criteria": {"true": NEG_MET_DEF, "false": NEG_UNMET_DEF},
        },
        "c2": CLARITY_CHOICE,
    }


@pytest.fixture
def make_grader() -> Iterator[Any]:
    """Builds graders and closes their decision-model caches at teardown."""
    built: list[CriterionGrader] = []

    def build(**kwargs: Any) -> CriterionGrader:
        grader = CriterionGrader(**kwargs)
        built.append(grader)
        return grader

    yield build
    for grader in built:
        for client in grader._decision_clients.values():
            client.close()


class RecordingLLM:
    """Stand-in for ``LLMClient``: records user prompts; MET / the first option."""

    def __init__(self) -> None:
        self.user_prompts: list[str] = []

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> Any:
        self.user_prompts.append(user_prompt)
        parsed: Any
        if "<options>" in user_prompt:
            parsed = MultiChoiceJudgment(selected_option=1, explanation="first")
        else:
            parsed = CriterionJudgment(criterion_status=CriterionVerdict.MET, explanation="yes")
        return GenerateResult(
            content="{}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=None,
            parsed=parsed,
        )


def block(guidelines: str) -> str:
    return f"<guidelines>\n{guidelines}\n\n{PRECEDENCE_RULE}\n</guidelines>\n\n"


# =============================================================================
# The request: goldens
# =============================================================================


class TestRequestGoldens:
    @pytest.mark.asyncio
    async def test_plain_submission(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(SUBMISSION, grader, query=QUERY)
        (call,) = fake_sdk.calls
        assert request_json(call) == json.dumps(
            [
                {"guidelines": GUIDELINES, "input": QUERY, "submission": SUBMISSION},
                framed_questions(
                    f"{TASK} {GUIDELINES_CLAUSE} Criterion: Mentions light",
                    f"{TASK} {GUIDELINES_CLAUSE} Criterion: Claims plants eat soil",
                ),
            ]
        )

    @pytest.mark.asyncio
    async def test_with_reference(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(
            SUBMISSION, grader, query=QUERY, reference_submission=REFERENCE
        )
        (call,) = fake_sdk.calls
        assert request_json(call) == json.dumps(
            [
                {
                    "guidelines": GUIDELINES,
                    "input": QUERY,
                    "reference_submission": REFERENCE,
                    "submission": SUBMISSION,
                },
                framed_questions(
                    f"{TASK} {GUIDELINES_CLAUSE} {REFERENCE_SUBMISSION} Criterion: Mentions light",
                    f"{TASK} {GUIDELINES_CLAUSE} {REFERENCE_SUBMISSION} "
                    "Criterion: Claims plants eat soil",
                ),
            ]
        )

    @pytest.mark.asyncio
    async def test_thinking_output_submission(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(
            {"thinking": "Recall the light reactions.", "output": SUBMISSION}, grader
        )
        (call,) = fake_sdk.calls
        assert request_json(call) == json.dumps(
            [
                {
                    "guidelines": GUIDELINES,
                    "thinking": "Recall the light reactions.",
                    "output": SUBMISSION,
                },
                framed_questions(
                    f"{TASK_OUTPUT} {GUIDELINES_CLAUSE} Criterion: Mentions light",
                    f"{TASK_OUTPUT} {GUIDELINES_CLAUSE} Criterion: Claims plants eat soil",
                ),
            ]
        )

    @pytest.mark.asyncio
    async def test_thinking_output_submission_with_reference(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(
            {"thinking": "Recall the light reactions.", "output": SUBMISSION},
            grader,
            query=QUERY,
            reference_submission=REFERENCE,
        )
        (call,) = fake_sdk.calls
        assert request_json(call) == json.dumps(
            [
                {
                    "guidelines": GUIDELINES,
                    "input": QUERY,
                    "reference_submission": REFERENCE,
                    "thinking": "Recall the light reactions.",
                    "output": SUBMISSION,
                },
                framed_questions(
                    f"{TASK_OUTPUT} {GUIDELINES_CLAUSE} {REFERENCE_OUTPUT} "
                    "Criterion: Mentions light",
                    f"{TASK_OUTPUT} {GUIDELINES_CLAUSE} {REFERENCE_OUTPUT} "
                    "Criterion: Claims plants eat soil",
                ),
            ]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("binary_framing", ["noul", "choice"])
    async def test_bare_noul_and_choice_framings(self, fake_sdk, make_grader, binary_framing):
        answers = dict(ANSWERS)
        if binary_framing == "choice":
            answers["c0"] = {
                "type": "choice",
                "choice": "MET",
                "confidence": 0.8,
                "probabilities": {"MET": 0.9, "UNMET": 0.05, "CANNOT_ASSESS": 0.05},
            }
            answers["c1"] = {
                "type": "choice",
                "choice": "UNMET",
                "confidence": 0.8,
                "probabilities": {"MET": 0.05, "UNMET": 0.9, "CANNOT_ASSESS": 0.05},
            }
        fake_sdk.response = SystemOneResponse.model_validate_json(
            json.dumps({"model": "jev-test", "usage": {"input_tokens": 50}, "answers": answers})
        )
        grader = make_grader(judge_model_config=dm(binary_framing=binary_framing))
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(SUBMISSION, grader, query=QUERY)
        (call,) = fake_sdk.calls
        state, questions = request(call)
        assert list(state.items()) == [
            ("guidelines", GUIDELINES),
            ("input", QUERY),
            ("submission", SUBMISSION),
        ]
        if binary_framing == "noul":
            assert questions["c0"] == {"type": "noul", "instructions": "Mentions light"}
            assert questions["c1"] == {"type": "noul", "instructions": "Claims plants eat soil"}
        else:
            assert questions["c0"]["instructions"] == (
                f"{TASK} {GUIDELINES_CLAUSE} Criterion: Mentions light"
            )
            assert questions["c1"]["instructions"] == (
                f"{TASK} {GUIDELINES_CLAUSE} Criterion: Claims plants eat soil"
            )
        assert questions["c2"] == CLARITY_CHOICE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("guidelines", [None, "", " \n "], ids=["none", "empty", "blank"])
    async def test_without_guidelines_the_request_is_unchanged(
        self, fake_sdk, make_grader, guidelines
    ):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        await Rubric(CRITERIA, guidelines=guidelines).grade(
            SUBMISSION, grader, query=QUERY, reference_submission=REFERENCE
        )
        (call,) = fake_sdk.calls
        assert request_json(call) == json.dumps(
            [
                {"input": QUERY, "reference_submission": REFERENCE, "submission": SUBMISSION},
                framed_questions(
                    f"{TASK} {REFERENCE_SUBMISSION} Criterion: Mentions light",
                    f"{TASK} {REFERENCE_SUBMISSION} Criterion: Claims plants eat soil",
                ),
            ]
        )

    @pytest.mark.asyncio
    async def test_judge_takes_guidelines_directly(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        await grader.judge(SUBMISSION, CRITERIA, QUERY, guidelines=GUIDELINES)
        await grader.judge(SUBMISSION, CRITERIA, QUERY)
        with_guidelines, without = (request(call)[0] for call in fake_sdk.calls)
        assert list(with_guidelines.items()) == [("guidelines", GUIDELINES), *without.items()]


# =============================================================================
# Cache: guidelines are part of the state, so part of the key
# =============================================================================


@pytest.mark.asyncio
async def test_guidelines_are_part_of_the_cache_key(fake_sdk, make_grader, tmp_path):
    fake_sdk.response = response()
    config = dm(cache_enabled=True, cache_dir=tmp_path / "cache")

    async def grade(guidelines: str | None) -> None:
        grader = make_grader(judge_model_config=config)
        await Rubric(CRITERIA, guidelines=guidelines).grade(SUBMISSION, grader, query=QUERY)

    await grade(None)
    await grade(GUIDELINES)
    assert len(fake_sdk.calls) == 2  # different state: no cache hit
    await grade(GUIDELINES)
    await grade(None)
    await grade("Other guidelines.")
    assert len(fake_sdk.calls) == 3  # only the new guidelines miss


# =============================================================================
# Mixed ensembles and evaluate()
# =============================================================================


@pytest.mark.asyncio
async def test_mixed_ensemble_both_judge_kinds_see_the_guidelines(fake_sdk, make_grader):
    """One decision-model request carrying the guidelines in its state, and one LLM call per
    criterion, each prompt starting with the guidelines block."""
    fake_sdk.response = response()
    llm = RecordingLLM()
    with patch("autorubric.graders.criterion_grader.LLMClient", return_value=llm):
        grader = make_grader(
            judges=[JudgeSpec(dm(), "jev"), JudgeSpec(LLMConfig(model="test-model"), "llm")],
            seed=5,
        )
    report = await Rubric(CRITERIA, guidelines=GUIDELINES).grade(SUBMISSION, grader, query=QUERY)

    (call,) = fake_sdk.calls
    state, questions = request(call)
    assert list(state.items()) == [
        ("guidelines", GUIDELINES),
        ("input", QUERY),
        ("submission", SUBMISSION),
    ]
    assert questions["c0"]["instructions"] == (
        f"{TASK} {GUIDELINES_CLAUSE} Criterion: Mentions light"
    )
    assert len(llm.user_prompts) == len(CRITERIA)
    assert all(p.startswith(block(GUIDELINES) + "<") for p in llm.user_prompts)
    assert [v.judge_id for v in report.report[0].votes] == ["jev", "llm"]


@pytest.mark.asyncio
async def test_evaluate_sends_each_items_own_guidelines(fake_sdk, make_grader, tmp_path):
    """Per-item rubrics bring their own guidelines (or none) to both judge kinds; each item
    still gets exactly one decision-model request."""
    fake_sdk.response = response()
    dataset = RubricDataset(
        prompt=QUERY,
        rubric=Rubric(CRITERIA, guidelines="Global guidelines."),
        name="per-item-guidelines",
    )
    dataset.items = [
        DataItem(submission="Item zero.", description="global rubric"),
        DataItem(
            submission="Item one.",
            description="own rubric, own guidelines",
            rubric=Rubric(CRITERIA, guidelines="Item one guidelines."),
        ),
        DataItem(
            submission="Item two.",
            description="own rubric, no guidelines",
            rubric=Rubric(CRITERIA),
        ),
    ]
    llm = RecordingLLM()
    with patch("autorubric.graders.criterion_grader.LLMClient", return_value=llm):
        grader = make_grader(
            judges=[JudgeSpec(dm(), "jev"), JudgeSpec(LLMConfig(model="test-model"), "llm")],
            seed=5,
        )
    result = await evaluate(
        dataset,
        grader,
        show_progress=False,
        experiment_name="per-item-guidelines",
        experiments_dir=tmp_path,
    )
    assert result.successful_items == 3

    states = {state["submission"]: state for state, _, _ in fake_sdk.calls}
    assert len(fake_sdk.calls) == 3
    assert states["Item zero."].get("guidelines") == "Global guidelines."
    assert states["Item one."].get("guidelines") == "Item one guidelines."
    assert "guidelines" not in states["Item two."]
    assert list(states["Item one."]) == ["guidelines", "input", "submission"]

    def prompts_for(submission: str) -> list[str]:
        return [p for p in llm.user_prompts if f"<submission>\n{submission}\n" in p]

    zero, one, two = (prompts_for(f"Item {n}.") for n in ("zero", "one", "two"))
    assert len(zero) == len(one) == len(two) == len(CRITERIA)
    assert all(p.startswith(block("Global guidelines.")) for p in zero)
    assert all(p.startswith(block("Item one guidelines.")) for p in one)
    assert not any(p.startswith("<guidelines>") for p in two)
