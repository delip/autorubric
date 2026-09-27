"""Byte-identity goldens for the prompts of ``CriterionGrader(llm_calls="per_item")``.

Mirrors ``tests/test_prompt_goldens.py`` in structure, docstring style, universal-newline
golden reading and ``--write`` regeneration script mode, but for the whole-rubric prompts an
LLM judge is sent when a grader is set to call per item instead of per criterion. It leaves
``tests/test_prompt_goldens.py`` and ``tests/golden/prompts/`` (its ``_RecordingClient``
included) completely untouched: that golden set pins the per-criterion prompts, which
``llm_calls="per_item"`` never sends.

The goldens under ``tests/golden/prompts_per_item/`` pin three layers:

- ``constants/<NAME>.txt``: every ``RUBRIC_JUDGMENT_*`` prompt constant of
  ``autorubric.prompts``.
- ``builders/<case>.txt``: ``build_rubric_system_prompt`` over binary-only, multi-choice-only
  and mixed rubrics, with and without ``with_examples``, and with custom guides;
  ``build_rubric_user_prompt`` over binary-only, multi-choice-only, mixed, forced-choice
  (no NA option), a query and reference submission, an opaque and a genuinely rendered
  few-shot ``examples_text``, and negative- and zero-weight criteria; and
  ``_format_rubric_examples`` over binary-only, multi-choice-only with a shuffled option
  presentation, mixed, with and without ``include_reason``, and an example missing a reason.
- ``grader_calls.json``: the exact (system prompt, user prompt, response format) triples a
  ``CriterionGrader(llm_calls="per_item", seed=<fixed>)`` sends for a single judge and for a
  two-judge panel over a mixed rubric, without and with item-level few-shot training data.

A separate set of tests, independent of these goldens, checks the reviewed prompt text by
writing it out literally: goldens captured from the implementation prove the implementation
is stable, not that it is correct, so those tests do not read this module's own constants or
read back a golden file.

Regenerate from a checkout whose prompts are the intended reference by running this module as
a script (as ``tests/test_prompt_goldens.py`` is run)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout>/src \\
        uv run --frozen python tests/test_per_item_prompt_goldens.py --write

Golden text files are read with universal newlines, so a CRLF checkout compares equal to the
LF bytes the library produces.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import autorubric.prompts as prompts_module
from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DataItem,
    FewShotConfig,
    FewShotExample,
    Rubric,
    RubricDataset,
    TokenUsage,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.prompts import (
    GRADER_SYSTEM_PROMPT_DEFAULT,
    MULTI_CHOICE_SYSTEM_PROMPT,
    _format_few_shot_examples,
    _format_multi_choice_examples,
    _format_rubric_examples,
    build_rubric_system_prompt,
    build_rubric_user_prompt,
    build_user_prompt,
)
from autorubric.types import RubricJudgment

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "prompts_per_item"
CONSTANTS_DIR = GOLDEN_DIR / "constants"
BUILDERS_DIR = GOLDEN_DIR / "builders"
GRADER_CALLS_PATH = GOLDEN_DIR / "grader_calls.json"

# Every ``RUBRIC_JUDGMENT_*`` prompt constant of ``autorubric.prompts`` at capture time.
# Listed explicitly (not discovered), as ``tests/test_prompt_goldens.py`` lists its own, so a
# constant added later does not need a pre-existing golden.
PROMPT_CONSTANT_NAMES = (
    "RUBRIC_JUDGMENT_PREAMBLE",
    "RUBRIC_JUDGMENT_BINARY_KIND",
    "RUBRIC_JUDGMENT_MULTI_CHOICE_KIND",
    "RUBRIC_JUDGMENT_EXAMPLES",
    "RUBRIC_JUDGMENT_BINARY_GUIDE",
    "RUBRIC_JUDGMENT_MULTI_CHOICE_GUIDE",
    "RUBRIC_JUDGMENT_RESPONSE_FORMAT",
    "RUBRIC_JUDGMENT_BINARY_RULE",
    "RUBRIC_JUDGMENT_MULTI_CHOICE_RULE",
)

# ---------------------------------------------------------------------------
# Representative inputs
# ---------------------------------------------------------------------------

POSITIVE = Criterion(
    name="boiling_point",
    weight=3.0,
    requirement="States the boiling point of water at sea level",
)
NEGATIVE = Criterion(
    name="unsafe_advice",
    weight=-2.0,
    requirement="Recommends drinking untreated river water",
)
# A zero weight is not negative: the criterion is posed as a positive one.
ZERO_WEIGHT = Criterion(
    name="altitude",
    weight=0.0,
    requirement="Mentions that the boiling point drops at altitude",
)
ORDINAL_AUTO_NA = Criterion(
    name="clarity",
    weight=4.0,
    requirement="How clear is the explanation?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Very unclear", value=0.0),
        CriterionOption(label="Somewhat clear", value=0.5),
        CriterionOption(label="Very clear", value=1.0),
        CriterionOption(label="Cannot assess / not applicable", value=0.0, na=True),
    ],
)
ORDINAL_FORCED_CHOICE = Criterion(
    name="clarity_forced",
    weight=4.0,
    requirement="How clear is the explanation?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Very unclear", value=0.0),
        CriterionOption(label="Somewhat clear", value=0.5),
        CriterionOption(label="Very clear", value=1.0),
    ],
)

PLAIN = "Water boils at 100 degrees Celsius at sea level."
QUERY = "At what temperature does water boil?"
REFERENCE = "At sea level, water boils at 100 °C (212 °F)."

# Rendered few-shot examples are opaque to the user-prompt builder, which places them.
EXAMPLES_TEXT = (
    "<examples>\n<example_1>\n<example_submission>Water boils at 90 degrees.</example_submission>\n"
    "</example_1>\n</examples>"
)

CUSTOM_BINARY_GUIDE = "Judge this criterion as a strict binary check.\nPrefer UNMET when unsure."
CUSTOM_MULTI_CHOICE_GUIDE = "Pick the single best-fitting option; never hedge."

# ---------------------------------------------------------------------------
# System-prompt builder cases
# ---------------------------------------------------------------------------

SYSTEM_CASES: dict[str, Callable[[], str]] = {
    "system__binary_only": lambda: build_rubric_system_prompt(GRADER_SYSTEM_PROMPT_DEFAULT, None),
    "system__binary_only_with_examples": lambda: build_rubric_system_prompt(
        GRADER_SYSTEM_PROMPT_DEFAULT, None, with_examples=True
    ),
    "system__multi_choice_only": lambda: build_rubric_system_prompt(
        None, MULTI_CHOICE_SYSTEM_PROMPT
    ),
    "system__multi_choice_only_with_examples": lambda: build_rubric_system_prompt(
        None, MULTI_CHOICE_SYSTEM_PROMPT, with_examples=True
    ),
    "system__mixed": lambda: build_rubric_system_prompt(
        GRADER_SYSTEM_PROMPT_DEFAULT, MULTI_CHOICE_SYSTEM_PROMPT
    ),
    "system__mixed_with_examples": lambda: build_rubric_system_prompt(
        GRADER_SYSTEM_PROMPT_DEFAULT, MULTI_CHOICE_SYSTEM_PROMPT, with_examples=True
    ),
    "system__custom_guides": lambda: build_rubric_system_prompt(
        CUSTOM_BINARY_GUIDE, CUSTOM_MULTI_CHOICE_GUIDE
    ),
}

# ---------------------------------------------------------------------------
# User-prompt builder cases
# ---------------------------------------------------------------------------
# Each case takes extra builder keywords (``guidelines``), as ``tests/test_prompt_goldens.py``
# does, so the same inputs serve the guidelines tests below; the goldens are captured with
# none.

# A genuinely rendered few-shot block (as opposed to the opaque ``EXAMPLES_TEXT`` above),
# composing ``_format_rubric_examples`` with ``build_rubric_user_prompt`` the way
# ``_judge_rubric_in_one_call`` does.
RENDERED_EXAMPLES_TEXT = _format_rubric_examples(
    [
        (
            "Water boils around 100 degrees; not sure about altitude effects.",
            [
                ("c0", POSITIVE, CriterionVerdict.MET, "States the boiling point."),
                ("c1", ORDINAL_AUTO_NA, 2, None),
            ],
        )
    ],
    include_reason=True,
)

USER_CASES: dict[str, Callable[..., str]] = {
    "user__binary_only": lambda **kw: build_rubric_user_prompt([("c0", POSITIVE)], PLAIN, **kw),
    "user__multi_choice_only": lambda **kw: build_rubric_user_prompt(
        [("c0", ORDINAL_AUTO_NA)], PLAIN, **kw
    ),
    "user__mixed": lambda **kw: build_rubric_user_prompt(
        [("c0", POSITIVE), ("c1", ORDINAL_AUTO_NA)], PLAIN, **kw
    ),
    "user__forced_choice": lambda **kw: build_rubric_user_prompt(
        [("c0", ORDINAL_FORCED_CHOICE)], PLAIN, **kw
    ),
    "user__query_reference": lambda **kw: build_rubric_user_prompt(
        [("c0", POSITIVE), ("c1", ORDINAL_AUTO_NA)], PLAIN, QUERY, REFERENCE, **kw
    ),
    "user__negative_weight": lambda **kw: build_rubric_user_prompt([("c0", NEGATIVE)], PLAIN, **kw),
    "user__zero_weight": lambda **kw: build_rubric_user_prompt([("c0", ZERO_WEIGHT)], PLAIN, **kw),
    "user__examples_query_reference": lambda **kw: build_rubric_user_prompt(
        [("c0", POSITIVE), ("c1", ORDINAL_AUTO_NA)],
        PLAIN,
        QUERY,
        REFERENCE,
        examples_text=EXAMPLES_TEXT,
        **kw,
    ),
    "user__rendered_examples_query_reference": lambda **kw: build_rubric_user_prompt(
        [("c0", POSITIVE), ("c1", ORDINAL_AUTO_NA)],
        PLAIN,
        QUERY,
        REFERENCE,
        examples_text=RENDERED_EXAMPLES_TEXT,
        **kw,
    ),
}

# ---------------------------------------------------------------------------
# _format_rubric_examples cases
# ---------------------------------------------------------------------------
# Criteria as a call would *show* them: EXAMPLE_CLARITY_SHOWN's options are deliberately
# ordered differently from any per-criterion criterion above, to exercise a multi-choice
# judgment whose option index refers to a shuffled presentation, the way a call already
# renumbers a shuffled multi-choice criterion's options before this function ever sees it.

EXAMPLE_CAPITAL = Criterion(
    name="capital", weight=1.0, requirement="States that the capital of France is Paris"
)
EXAMPLE_FLUENT = Criterion(name="fluent", weight=1.0, requirement="The response reads fluently")
EXAMPLE_CLARITY_SHOWN = Criterion(
    name="clarity_shown",
    weight=1.0,
    requirement="How clear is the explanation?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Very clear", value=1.0),
        CriterionOption(label="Cannot assess / not applicable", value=0.0, na=True),
        CriterionOption(label="Somewhat clear", value=0.5),
        CriterionOption(label="Unclear", value=0.0),
    ],
)

# One example with a binary and a multi-choice judgment; shared by the "mixed" case and its
# "without_reason" twin, so the only difference between the two golden files is the
# ``include_reason`` flag, never the underlying judgments.
_MIXED_EXAMPLE_JUDGMENTS: list[
    tuple[str, list[tuple[str, Criterion, CriterionVerdict | int, str | None]]]
] = [
    (
        "Paris is the capital; the explanation was crystal clear.",
        [
            ("c0", EXAMPLE_CAPITAL, CriterionVerdict.MET, "Names Paris."),
            ("c1", EXAMPLE_CLARITY_SHOWN, 0, "Very clear wording."),
        ],
    ),
    (
        "London is the capital, in a somewhat rambling explanation.",
        [
            ("c0", EXAMPLE_CAPITAL, CriterionVerdict.UNMET, "Names London instead."),
            ("c1", EXAMPLE_CLARITY_SHOWN, 2, "Rambling but decipherable."),
        ],
    ),
]

EXAMPLES_CASES: dict[str, Callable[[], str]] = {
    "examples__binary_only": lambda: _format_rubric_examples(
        [
            (
                "Paris is the capital of France.",
                [("c0", EXAMPLE_CAPITAL, CriterionVerdict.MET, "Names Paris.")],
            ),
            (
                "The response is a jumbled mess of clauses.",
                [("c1", EXAMPLE_FLUENT, CriterionVerdict.UNMET, "Hard to follow.")],
            ),
        ],
        include_reason=True,
    ),
    "examples__multi_choice_shuffled": lambda: _format_rubric_examples(
        [
            (
                "The explanation was crystal clear.",
                [("c1", EXAMPLE_CLARITY_SHOWN, 0, "Reader understood immediately.")],
            ),
        ],
        include_reason=True,
    ),
    "examples__mixed": lambda: _format_rubric_examples(
        _MIXED_EXAMPLE_JUDGMENTS, include_reason=True
    ),
    "examples__without_reason": lambda: _format_rubric_examples(
        _MIXED_EXAMPLE_JUDGMENTS, include_reason=False
    ),
    "examples__missing_reason": lambda: _format_rubric_examples(
        [
            (
                "Paris is the capital; the explanation was crystal clear.",
                [
                    ("c0", EXAMPLE_CAPITAL, CriterionVerdict.MET, None),
                    ("c1", EXAMPLE_CLARITY_SHOWN, 0, "Very clear wording."),
                ],
            ),
        ],
        include_reason=True,
    ),
}

BUILDER_CASES: dict[str, Callable[..., str]] = {**SYSTEM_CASES, **USER_CASES, **EXAMPLES_CASES}

# ---------------------------------------------------------------------------
# Grader-level capture (mocked clients; no network)
# ---------------------------------------------------------------------------

MIXED_RUBRIC = [POSITIVE, NEGATIVE, ORDINAL_AUTO_NA]

# Matches a criterion block's opening tag and the tag that starts its content, so the
# recording client below can tell a binary criterion (``<criterion_type>``) from a
# multi-choice one (``<question>``) by its id alone.
_CRITERION_BLOCK_RE = re.compile(r'<rubric_criterion id="(c\d+)">\n<(criterion_type|question)>')

# Training data for the item-level few-shot grader configurations below: each item is fully
# labelled on MIXED_RUBRIC's three criteria, with a written reason for every label, so
# ``FewShotConfig(include_reason=True)`` has something to show on every criterion.
FEW_SHOT_TRAINING_DATA = RubricDataset(
    prompt=QUERY,
    rubric=Rubric(MIXED_RUBRIC),
    items=[
        DataItem(
            submission="Water boils at 100 degrees Celsius; never drink from untreated rivers.",
            description="fully compliant",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET, "Very clear"],
            ground_truth_reasons=[
                "States the boiling point.",
                "Warns against river water.",
                "Straightforward explanation.",
            ],
        ),
        DataItem(
            submission="It's warm outside; river water is refreshing to drink.",
            description="off-topic and recommends unsafe advice",
            ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET, "Very unclear"],
            ground_truth_reasons=[
                "Never states the boiling point.",
                "Recommends drinking untreated river water.",
                "Rambling and unfocused.",
            ],
        ),
        DataItem(
            submission="Boiling occurs near 100C at sea level; altitude effects are not covered.",
            description="on-topic but incomplete",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET, "Somewhat clear"],
            ground_truth_reasons=[
                "States the boiling point.",
                "No unsafe advice.",
                "Mostly clear but incomplete.",
            ],
        ),
        DataItem(
            submission="Not sure of the temperature; river water is fine boiled or not.",
            description="ambiguous and risky",
            ground_truth=[
                CriterionVerdict.CANNOT_ASSESS,
                CriterionVerdict.MET,
                "Cannot assess / not applicable",
            ],
            ground_truth_reasons=[
                None,
                "Endorses drinking untreated river water.",
                None,
            ],
        ),
    ],
)
FEW_SHOT_CONFIG = FewShotConfig(seed=0, include_reason=True)


def _grader_configurations() -> dict[str, dict[str, Any]]:
    """Fixed-seed ``llm_calls="per_item"`` grader configurations and what each grades."""
    judge = LLMConfig(model="golden-model")
    return {
        "single_judge_per_item": {
            "grader": lambda: CriterionGrader(
                judges=[JudgeSpec(judge, "golden")], llm_calls="per_item", seed=0
            ),
            "rubric": MIXED_RUBRIC,
            "to_grade": PLAIN,
            "query": QUERY,
            "reference_submission": REFERENCE,
        },
        "panel_per_item": {
            "grader": lambda: CriterionGrader(
                judges=[
                    JudgeSpec(judge, "alpha"),
                    JudgeSpec(LLMConfig(model="golden-model-2"), "beta", 2.0),
                ],
                aggregation="weighted",
                llm_calls="per_item",
                seed=0,
            ),
            "rubric": MIXED_RUBRIC,
            "to_grade": PLAIN,
            "query": QUERY,
            "reference_submission": None,
        },
        "single_judge_per_item_few_shot": {
            "grader": lambda: CriterionGrader(
                judges=[JudgeSpec(judge, "golden")],
                llm_calls="per_item",
                seed=0,
                training_data=FEW_SHOT_TRAINING_DATA,
                few_shot_config=FEW_SHOT_CONFIG,
            ),
            "rubric": MIXED_RUBRIC,
            "to_grade": PLAIN,
            "query": QUERY,
            "reference_submission": REFERENCE,
        },
        "panel_per_item_few_shot": {
            "grader": lambda: CriterionGrader(
                judges=[
                    JudgeSpec(judge, "alpha"),
                    JudgeSpec(LLMConfig(model="golden-model-2"), "beta", 2.0),
                ],
                aggregation="weighted",
                llm_calls="per_item",
                seed=0,
                training_data=FEW_SHOT_TRAINING_DATA,
                few_shot_config=FEW_SHOT_CONFIG,
            ),
            "rubric": MIXED_RUBRIC,
            "to_grade": PLAIN,
            "query": QUERY,
            "reference_submission": None,
        },
    }


class _RubricRecordingClient:
    """Stands in for ``LLMClient`` under ``llm_calls="per_item"``.

    Records every call and answers a ``RubricJudgment`` built from the ``id="cK"``
    attributes the prompt's ``<rubric_criterion>`` tags carry: a binary criterion (its block
    opens with ``<criterion_type>``) gets ``criterion_status="MET"``, and a multi-choice one
    (``<question>``) gets ``selected_option=1``. A first-match mock keyed on
    ``response_format`` alone, such as ``create_per_criterion_mock_client``, cannot read a
    multi-criterion prompt this way.
    """

    def __init__(self, judge_id: str, calls: list[dict[str, str]]) -> None:
        self._judge_id = judge_id
        self._calls = calls

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        assert response_format is RubricJudgment
        self._calls.append(
            {
                "judge_id": self._judge_id,
                "response_format": response_format.__name__,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        judgments = []
        for criterion_id, kind_tag in _CRITERION_BLOCK_RE.findall(user_prompt):
            if kind_tag == "criterion_type":
                judgments.append(
                    {
                        "criterion_id": criterion_id,
                        "criterion_status": "MET",
                        "explanation": "golden",
                    }
                )
            else:
                judgments.append(
                    {"criterion_id": criterion_id, "selected_option": 1, "explanation": "golden"}
                )
        parsed = RubricJudgment.model_validate({"judgments": judgments})
        return GenerateResult(
            content="{}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=None,
            parsed=parsed,
        )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _capture_grader_calls() -> dict[str, Any]:
    """Grade each configuration once with recording clients.

    System prompts are stored once under their SHA-256 and referenced by hash, so the file
    stays readable. Calls are sorted, since concurrent judges may interleave.
    """
    system_prompts: dict[str, str] = {}
    configurations: dict[str, list[dict[str, str]]] = {}
    for name, spec in _grader_configurations().items():
        grader = spec["grader"]()
        calls: list[dict[str, str]] = []
        for judge_id in list(grader._clients):
            grader._clients[judge_id] = _RubricRecordingClient(judge_id, calls)
        await grader.grade(
            spec["to_grade"],
            spec["rubric"],
            query=spec["query"],
            reference_submission=spec["reference_submission"],
        )
        records = []
        for call in calls:
            digest = _sha256(call["system_prompt"])
            system_prompts[digest] = call["system_prompt"]
            records.append(
                {
                    "judge_id": call["judge_id"],
                    "response_format": call["response_format"],
                    "system_prompt_sha256": digest,
                    "user_prompt": call["user_prompt"],
                }
            )
        records.sort(key=lambda r: (r["judge_id"], r["response_format"], r["user_prompt"]))
        configurations[name] = records
    return {"system_prompts": system_prompts, "configurations": configurations}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _read_golden(path: Path) -> str:
    # Universal newlines: a CRLF checkout reads back as the LF text the library emits.
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", PROMPT_CONSTANT_NAMES)
def test_prompt_constant_matches_golden(name: str) -> None:
    """Every ``RUBRIC_JUDGMENT_*`` constant is byte-identical to the captured reference."""
    assert getattr(prompts_module, name) == _read_golden(CONSTANTS_DIR / f"{name}.txt")


@pytest.mark.parametrize("case", sorted(BUILDER_CASES))
def test_builder_matches_golden(case: str) -> None:
    """Every system- and user-prompt builder output is byte-identical to the reference."""
    assert BUILDER_CASES[case]() == _read_golden(BUILDERS_DIR / f"{case}.txt")


def test_golden_files_cover_exactly_the_cases() -> None:
    """No stale or missing golden files (a renamed case must not silently pass)."""
    assert sorted(p.stem for p in CONSTANTS_DIR.glob("*.txt")) == sorted(PROMPT_CONSTANT_NAMES)
    assert sorted(p.stem for p in BUILDERS_DIR.glob("*.txt")) == sorted(BUILDER_CASES)


@pytest.mark.asyncio
async def test_grader_sends_golden_prompts() -> None:
    """A fixed-seed ``llm_calls="per_item"`` grader sends exactly the captured prompts."""
    expected = json.loads(GRADER_CALLS_PATH.read_text(encoding="utf-8"))
    actual = await _capture_grader_calls()
    assert sorted(actual["configurations"]) == sorted(expected["configurations"])
    for name, calls in expected["configurations"].items():
        assert actual["configurations"][name] == calls, name
    assert actual["system_prompts"] == expected["system_prompts"]


def test_build_rubric_system_prompt_raises_for_no_guides() -> None:
    """Both guides ``None`` is a configuration error, not an empty prompt."""
    with pytest.raises(ValueError, match="needs the guide of at least one kind"):
        build_rubric_system_prompt(None, None)


def test_build_rubric_user_prompt_raises_for_empty_criteria() -> None:
    """An empty rubric has no ``<criteria>`` to build a prompt for."""
    with pytest.raises(ValueError, match="Cannot build a rubric prompt without criteria"):
        build_rubric_user_prompt([], PLAIN)


# ---------------------------------------------------------------------------
# Rubric guidelines
# ---------------------------------------------------------------------------
# The whole-rubric user prompt starts with the rubric's guidelines exactly as a per-criterion
# prompt does (``_guidelines_block``): a block, then the prompt the builder produces without
# them, byte for byte.

GUIDELINES = (
    "Writers are English-language learners in grades 8-12; judge against that level.\n"
    "'Cited' means any attribution, e.g. {author, year}, not formal citation style."
)
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


@pytest.mark.parametrize("case", sorted(USER_CASES))
def test_user_prompt_builder_without_guidelines_matches_golden(case: str) -> None:
    """``guidelines=None`` (the default) leaves every builder's output byte-identical."""
    assert USER_CASES[case](guidelines=None) == _read_golden(BUILDERS_DIR / f"{case}.txt")


@pytest.mark.parametrize("case", sorted(USER_CASES))
def test_user_prompt_builder_with_guidelines_starts_with_the_block(case: str) -> None:
    """With guidelines, the prompt is the block, a blank line, then the golden prompt."""
    expected = GUIDELINES_PREFIX + _read_golden(BUILDERS_DIR / f"{case}.txt")
    assert USER_CASES[case](guidelines=GUIDELINES) == expected


# ---------------------------------------------------------------------------
# Literal-text tests: the reviewed prompt text, written out by hand, not read from this
# module's own golden files or from ``autorubric.prompts`` constants, so they check the text
# itself where the goldens check only that it is stable. A deliberate change to the prompt
# updates these literals and the goldens together; a change the literals do not describe
# fails here.
# ---------------------------------------------------------------------------

_SPEC_ROLE_PARAGRAPH = (
    "You are an expert evaluation judge. Your task is to judge every criterion in the "
    "<criteria> list against one submission, in a single response. Be precise, "
    "evidence-based, and consistent."
)
_SPEC_ID_PARAGRAPH_INTRO = (
    "Each criterion in <criteria> is a <rubric_criterion> with an id (c0, c1, ...)."
)
_SPEC_BINARY_BULLET = (
    "- A binary criterion holds a <criterion_type> and a <criterion>. Judge it as the "
    "binary criterion guide below describes."
)
_SPEC_MULTI_CHOICE_BULLET = (
    "- A multi-choice criterion holds a <question> and numbered <options>. Judge it as the "
    "multi-choice criterion guide below describes."
)
_SPEC_INDEPENDENCE_PARAGRAPH = (
    "Each guide is written for judging one criterion at a time. Apply it to each criterion "
    "of its kind separately, as if that criterion were the only one: judge every criterion "
    "independently, and never let your judgment of one criterion influence another. The "
    "<submission>, and any <guidelines>, <input> and <reference_submission>, apply to every "
    "criterion."
)
_SPEC_EXAMPLES_SENTENCE = (
    "The <examples> show earlier submissions with the correct judgment of each criterion, "
    "each under the criterion's id; use them as the guides describe."
)
_SPEC_FORMAT_LEAD_SENTENCE = (
    "A guide's RESPONSE FORMAT and EXAMPLES show the fields of one criterion's judgment. "
    "Your response holds one judgment per criterion, as the RESPONSE FORMAT at the end of "
    "these instructions says."
)
_SPEC_RESPONSE_FORMAT_HEADER = (
    "RESPONSE FORMAT:\n"
    "Respond with valid JSON holding exactly one judgment per criterion, in the order the "
    "criteria are listed:"
)
_SPEC_BOTH_SKELETON = (
    '{"judgments": [{"criterion_id": "c0", "criterion_status": "MET", "selected_option": '
    'null, "explanation": "..."}, {"criterion_id": "c1", "criterion_status": null, '
    '"selected_option": 2, "explanation": "..."}]}'
)
_SPEC_BINARY_SKELETON = (
    '{"judgments": [{"criterion_id": "c0", "criterion_status": "MET", "selected_option": '
    'null, "explanation": "..."}]}'
)
_SPEC_MULTI_CHOICE_SKELETON = (
    '{"judgments": [{"criterion_id": "c0", "criterion_status": null, "selected_option": 2, '
    '"explanation": "..."}]}'
)
_SPEC_ID_RULE = '- "criterion_id" is the criterion\'s id.'
_SPEC_BINARY_RULE = (
    '- For a binary criterion, "criterion_status" is "MET", "UNMET" or "CANNOT_ASSESS", and '
    '"selected_option" is null.'
)
_SPEC_MULTI_CHOICE_RULE = (
    '- For a multi-choice criterion, "selected_option" is the number of the chosen option, '
    'as that criterion\'s <options> number it, and "criterion_status" is null.'
)
_SPEC_EXPLANATION_RULE = (
    '- "explanation" is the 1-2 sentence explanation the criterion\'s guide asks for.'
)
_SPEC_CLOSING_LINE = "Return only raw JSON starting with {, no back-ticks, no 'json' prefix."


def test_spec_mixed_system_prompt_non_guide_text() -> None:
    """The mixed system prompt's every line outside the two guide bodies, written out."""
    expected = "\n\n".join(
        [
            "\n".join(
                [
                    _SPEC_ROLE_PARAGRAPH,
                    "",
                    _SPEC_ID_PARAGRAPH_INTRO,
                    _SPEC_BINARY_BULLET,
                    _SPEC_MULTI_CHOICE_BULLET,
                ]
            ),
            _SPEC_INDEPENDENCE_PARAGRAPH,
            _SPEC_FORMAT_LEAD_SENTENCE,
            f"<binary_criterion_guide>\n{GRADER_SYSTEM_PROMPT_DEFAULT}\n</binary_criterion_guide>",
            (
                "<multi_choice_criterion_guide>\n"
                f"{MULTI_CHOICE_SYSTEM_PROMPT}\n"
                "</multi_choice_criterion_guide>"
            ),
            "\n".join(
                [
                    _SPEC_RESPONSE_FORMAT_HEADER,
                    _SPEC_BOTH_SKELETON,
                    "",
                    _SPEC_ID_RULE,
                    _SPEC_BINARY_RULE,
                    _SPEC_MULTI_CHOICE_RULE,
                    _SPEC_EXPLANATION_RULE,
                    "",
                    _SPEC_CLOSING_LINE,
                ]
            ),
        ]
    )
    actual = build_rubric_system_prompt(GRADER_SYSTEM_PROMPT_DEFAULT, MULTI_CHOICE_SYSTEM_PROMPT)
    assert actual == expected


def test_spec_binary_only_system_prompt() -> None:
    """Binary-only: drop the multi-choice bullet, guide block and response-format bullet."""
    expected = "\n\n".join(
        [
            "\n".join([_SPEC_ROLE_PARAGRAPH, "", _SPEC_ID_PARAGRAPH_INTRO, _SPEC_BINARY_BULLET]),
            _SPEC_INDEPENDENCE_PARAGRAPH,
            _SPEC_FORMAT_LEAD_SENTENCE,
            f"<binary_criterion_guide>\n{GRADER_SYSTEM_PROMPT_DEFAULT}\n</binary_criterion_guide>",
            "\n".join(
                [
                    _SPEC_RESPONSE_FORMAT_HEADER,
                    _SPEC_BINARY_SKELETON,
                    "",
                    _SPEC_ID_RULE,
                    _SPEC_BINARY_RULE,
                    _SPEC_EXPLANATION_RULE,
                    "",
                    _SPEC_CLOSING_LINE,
                ]
            ),
        ]
    )
    assert build_rubric_system_prompt(GRADER_SYSTEM_PROMPT_DEFAULT, None) == expected


def test_spec_multi_choice_only_system_prompt() -> None:
    """Multi-choice-only: drop the binary bullet, guide block and response-format bullet."""
    expected = "\n\n".join(
        [
            "\n".join(
                [_SPEC_ROLE_PARAGRAPH, "", _SPEC_ID_PARAGRAPH_INTRO, _SPEC_MULTI_CHOICE_BULLET]
            ),
            _SPEC_INDEPENDENCE_PARAGRAPH,
            _SPEC_FORMAT_LEAD_SENTENCE,
            (
                "<multi_choice_criterion_guide>\n"
                f"{MULTI_CHOICE_SYSTEM_PROMPT}\n"
                "</multi_choice_criterion_guide>"
            ),
            "\n".join(
                [
                    _SPEC_RESPONSE_FORMAT_HEADER,
                    _SPEC_MULTI_CHOICE_SKELETON,
                    "",
                    _SPEC_ID_RULE,
                    _SPEC_MULTI_CHOICE_RULE,
                    _SPEC_EXPLANATION_RULE,
                    "",
                    _SPEC_CLOSING_LINE,
                ]
            ),
        ]
    )
    assert build_rubric_system_prompt(None, MULTI_CHOICE_SYSTEM_PROMPT) == expected


def test_spec_with_examples_paragraph() -> None:
    """With ``with_examples=True``, the examples sentence sits blank-line, sentence,
    blank-line, then the "A guide's RESPONSE FORMAT ..." sentence, right after the
    independence paragraph."""
    expected_infix = "\n\n".join(
        [_SPEC_INDEPENDENCE_PARAGRAPH, _SPEC_EXAMPLES_SENTENCE, _SPEC_FORMAT_LEAD_SENTENCE]
    )
    actual = build_rubric_system_prompt(GRADER_SYSTEM_PROMPT_DEFAULT, None, with_examples=True)
    assert expected_infix in actual
    # And it is absent without with_examples.
    without = build_rubric_system_prompt(GRADER_SYSTEM_PROMPT_DEFAULT, None)
    assert _SPEC_EXAMPLES_SENTENCE not in without


def test_spec_example_user_prompt() -> None:
    """The worked user-prompt example, reproduced with its literal criteria."""
    capital = Criterion(
        name="capital", weight=1.0, requirement="States that the capital of France is Paris"
    )
    clarity = Criterion(
        name="clarity",
        weight=1.0,
        requirement="How clear is the explanation?",
        scale_type="ordinal",
        options=[
            CriterionOption(label="Unclear", value=0.0),
            CriterionOption(label="Very clear", value=1.0),
            CriterionOption(label="Somewhat clear", value=0.5),
            CriterionOption(label="Cannot assess / not applicable", value=0.0, na=True),
        ],
    )
    to_grade = "Paris."
    expected = (
        "<criteria>\n"
        '<rubric_criterion id="c0">\n'
        "<criterion_type>\n"
        "positive\n"
        "</criterion_type>\n"
        "\n"
        "<criterion>\n"
        "States that the capital of France is Paris\n"
        "</criterion>\n"
        "</rubric_criterion>\n"
        "\n"
        '<rubric_criterion id="c1">\n'
        "<question>\n"
        "How clear is the explanation?\n"
        "</question>\n"
        "\n"
        "<options>\n"
        "1. Unclear\n"
        "2. Very clear\n"
        "3. Somewhat clear\n"
        "4. Cannot assess / not applicable\n"
        "</options>\n"
        "</rubric_criterion>\n"
        "</criteria>\n"
        "\n"
        "<submission>\n"
        f"{to_grade}\n"
        "</submission>"
    )
    actual = build_rubric_user_prompt([("c0", capital), ("c1", clarity)], to_grade)
    assert actual == expected


def test_spec_examples_sit_between_the_criteria_and_the_input() -> None:
    """Rendered examples follow ``</criteria>`` and a blank line, and a blank line separates
    them from the input, then the reference submission and the submission follow."""
    expected = (
        "<criteria>\n"
        '<rubric_criterion id="c0">\n'
        "<criterion_type>\n"
        "positive\n"
        "</criterion_type>\n"
        "\n"
        "<criterion>\n"
        "States the boiling point of water at sea level\n"
        "</criterion>\n"
        "</rubric_criterion>\n"
        "</criteria>\n"
        "\n"
        "<examples>\n"
        "...\n"
        "</examples>\n"
        "\n"
        "<input>At what temperature does water boil?</input>\n"
        "\n"
        "<reference_submission>\n"
        "At sea level, water boils at 100 °C (212 °F).\n"
        "</reference_submission>\n"
        "\n"
        "<submission>\n"
        "Water boils at 100 degrees Celsius at sea level.\n"
        "</submission>"
    )
    actual = build_rubric_user_prompt(
        [("c0", POSITIVE)], PLAIN, QUERY, REFERENCE, examples_text="<examples>\n...\n</examples>"
    )
    assert actual == expected


def test_spec_a_zero_weight_criterion_is_posed_as_positive() -> None:
    """Only a negative weight makes a criterion ``negative``, in a whole-rubric prompt and in
    the per-criterion prompt that shares its criterion block."""
    block = (
        "<criterion_type>\n"
        "positive\n"
        "</criterion_type>\n"
        "\n"
        "<criterion>\n"
        "Mentions that the boiling point drops at altitude\n"
        "</criterion>"
    )
    assert build_rubric_user_prompt([("c0", ZERO_WEIGHT)], PLAIN) == (
        f'<criteria>\n<rubric_criterion id="c0">\n{block}\n</rubric_criterion>\n</criteria>\n'
        f"\n<submission>\n{PLAIN}\n</submission>"
    )
    assert (
        build_user_prompt(ZERO_WEIGHT, PLAIN) == f"{block}\n\n<submission>\n{PLAIN}\n</submission>"
    )


def test_spec_format_rubric_examples_worked_example() -> None:
    """The worked few-shot example, reproduced with its literal text.

    Written out by hand, not read from a golden file or from ``autorubric.prompts``: a
    deliberate change to how examples render updates this text too, and any other change
    fails here.
    """
    capital = Criterion(
        name="capital", weight=1.0, requirement="States that the capital of France is Paris"
    )
    clarity = Criterion(
        name="clarity",
        weight=1.0,
        requirement="How clear is the explanation?",
        scale_type="ordinal",
        options=[
            CriterionOption(label="Unclear", value=0.0),
            CriterionOption(label="Very clear", value=1.0),
            CriterionOption(label="Somewhat clear", value=0.5),
            CriterionOption(label="Cannot assess / not applicable", value=0.0, na=True),
        ],
    )
    examples = [
        (
            "...",
            [
                ("c0", capital, CriterionVerdict.MET, "Names Paris."),
                ("c1", clarity, 1, None),
            ],
        )
    ]
    expected = (
        "<examples>\n"
        "<example_1>\n"
        "<example_submission>...</example_submission>\n"
        '<judgment id="c0">\n'
        "<verdict>MET</verdict>\n"
        "<reason>Names Paris.</reason>\n"
        "</judgment>\n"
        '<judgment id="c1">\n'
        "<selected_option>2</selected_option>\n"
        "<selected_label>Very clear</selected_label>\n"
        "</judgment>\n"
        "</example_1>\n"
        "</examples>"
    )
    assert _format_rubric_examples(examples, include_reason=True) == expected


# ---------------------------------------------------------------------------
# One rendering of an example's judgment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("include_reason", [False, True])
def test_a_judgment_is_rendered_as_a_per_criterion_example_renders_it(
    include_reason: bool,
) -> None:
    """A whole-rubric example's judgment holds exactly the lines a per-criterion example
    shows for the same label and reason, for a binary and a multi-choice criterion: the
    three formatters render a judgment with one helper, so they cannot drift apart."""
    capital = Criterion(name="capital", weight=1.0, requirement="Names the capital")
    clarity = Criterion(
        name="clarity",
        weight=1.0,
        requirement="How clear is the explanation?",
        options=[
            CriterionOption(label="Unclear", value=0.0),
            CriterionOption(label="Somewhat clear", value=0.5),
            CriterionOption(label="Very clear", value=1.0),
        ],
    )
    unmet = CriterionVerdict.UNMET
    binary_example = _format_few_shot_examples(
        [FewShotExample(submission="S", verdict=unmet, reason="Names no city.")], include_reason
    )
    option_example = _format_multi_choice_examples(
        clarity, [("S", 2, "Each step is explained.")], include_reason
    )
    whole = _format_rubric_examples(
        [
            (
                "S",
                [
                    ("c0", capital, unmet, "Names no city."),
                    ("c1", clarity, 2, "Each step is explained."),
                ],
            )
        ],
        include_reason,
    )

    def judgment_of(example: str) -> str:
        return example.split("</example_submission>\n", 1)[1].split("\n</example_1>", 1)[0]

    judgments = dict(re.findall(r'<judgment id="(c\d)">\n(.*?)\n</judgment>', whole, re.S))
    assert judgments == {"c0": judgment_of(binary_example), "c1": judgment_of(option_example)}
    assert ("<reason>" in whole) == include_reason


def test_an_option_index_for_a_binary_criterion_is_refused() -> None:
    capital = Criterion(name="capital", weight=1.0, requirement="Names the capital")
    with pytest.raises(ValueError, match="option index 1 given for a criterion without"):
        _format_rubric_examples([("S", [("c0", capital, 1, None)])], include_reason=False)


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def _write_goldens() -> None:
    CONSTANTS_DIR.mkdir(parents=True, exist_ok=True)
    BUILDERS_DIR.mkdir(parents=True, exist_ok=True)
    for name in PROMPT_CONSTANT_NAMES:
        text = getattr(prompts_module, name)
        (CONSTANTS_DIR / f"{name}.txt").write_bytes(text.encode("utf-8"))
    for case, build in BUILDER_CASES.items():
        (BUILDERS_DIR / f"{case}.txt").write_bytes(build().encode("utf-8"))
    captured = asyncio.run(_capture_grader_calls())
    GRADER_CALLS_PATH.write_text(
        json.dumps(captured, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python tests/test_per_item_prompt_goldens.py --write")
    print(f"Capturing per-item prompt goldens from {prompts_module.__file__}")
    _write_goldens()
