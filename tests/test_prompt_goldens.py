"""Byte-identity goldens for every LLM-judge prompt the library sends.

The goldens under ``tests/golden/prompts/`` were captured from the unmodified library
(``main`` at the merge of PR #17) before the prompt definitions were refactored into shared
constants. They pin three layers exactly:

- ``constants/<NAME>.txt``: every prompt constant exported by ``autorubric.prompts``.
- ``builders/<case>.txt``: every user-prompt builder, over a representative set of inputs
  (binary positive/negative weight, multi-choice ordinal and nominal with an NA option,
  few-shot variants of both, reference submissions, thinking/output structured
  submissions, query present/absent).
- ``grader_calls.json``: the exact (system prompt, user prompt, response format) triples a
  ``CriterionGrader`` sends for several fixed-seed configurations (single judge, ensemble,
  few-shot, forced-choice without shuffling), which also pins option shuffling and
  few-shot selection.

Unchanged prompts keep existing response caches (``LLMClient._cache_key`` hashes both
prompts) and provider prefix caches valid, so any intended prompt change must regenerate
these files deliberately. Regenerate from a checkout whose prompts are the intended
reference by running this module as a script (``PYTHONPATH`` selects which library checkout
is imported; the script prints the path it captured from)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout>/src \\
        uv run --frozen python tests/test_prompt_goldens.py --write

Golden text files are read with universal newlines, so a CRLF checkout (Windows with
``core.autocrlf``) compares equal to the LF bytes the library produces.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
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
    Rubric,
    RubricDataset,
    TokenUsage,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.prompts import (
    build_few_shot_user_prompt,
    build_multi_choice_few_shot_user_prompt,
    build_multi_choice_user_prompt,
    build_user_prompt,
)
from autorubric.types import CriterionJudgment, FewShotExample, MultiChoiceJudgment

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "prompts"
CONSTANTS_DIR = GOLDEN_DIR / "constants"
BUILDERS_DIR = GOLDEN_DIR / "builders"
GRADER_CALLS_PATH = GOLDEN_DIR / "grader_calls.json"

# Every prompt constant of ``autorubric.prompts`` at capture time. Listed explicitly (not
# discovered) so constants added later do not need a pre-existing golden.
PROMPT_CONSTANT_NAMES = (
    "GRADER_SYSTEM_PROMPT_DEFAULT",
    "FEW_SHOT_SYSTEM_PROMPT_ADDITION",
    "MULTI_CHOICE_SYSTEM_PROMPT",
    "MULTI_CHOICE_FEW_SHOT_ADDITION",
    "RUBRIC_REVISION_SYSTEM_PROMPT",
    "RUBRIC_REVISION_USER_PROMPT_TEMPLATE",
    "HELD_OUT_REVISION_SYSTEM_PROMPT",
    "HELD_OUT_REVISION_USER_PROMPT_TEMPLATE",
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
ORDINAL = Criterion(
    name="clarity",
    weight=4.0,
    requirement="How clear is the explanation?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Very unclear", value=0.0),
        CriterionOption(label="Somewhat clear", value=0.5),
        CriterionOption(label="Very clear", value=1.0),
    ],
)
# The effective criterion the grader builds under the default auto_na_option=True.
ORDINAL_AUTO_NA = ORDINAL.with_guaranteed_na_option()
# An author NA option whose label does not itself signal NA, so the renderer marks it.
ORDINAL_AUTHOR_NA = Criterion(
    name="specificity",
    weight=2.0,
    requirement="How specific are the safety instructions?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Vague", value=0.0),
        CriterionOption(label="Partly specific", value=0.5),
        CriterionOption(label="Fully specific", value=1.0),
        CriterionOption(label="Insufficient data", value=0.0, na=True),
    ],
)
NOMINAL = Criterion(
    name="register",
    weight=1.0,
    requirement="Which register best describes the answer?",
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Conversational", value=0.5),
        CriterionOption(label="Slang", value=0.0),
        CriterionOption(label="N/A", value=0.0, na=True),
    ],
)

PLAIN = "Water boils at 100 degrees Celsius at sea level."
# The string form ``Grader.grade`` builds from a {"thinking", "output"} submission.
STRUCTURED = (
    "<thinking>Recall the standard value at one atmosphere.</thinking>\n"
    "<output>Water boils at 100 °C at sea level.</output>"
)
QUERY = "At what temperature does water boil?"
REFERENCE = "At sea level, water boils at 100 °C (212 °F)."

BINARY_EXAMPLES = [
    FewShotExample(
        submission="Water boils at 100 C.", verdict=CriterionVerdict.MET, reason="Gives 100 C."
    ),
    FewShotExample(submission="Water is wet.", verdict=CriterionVerdict.UNMET, reason=None),
    FewShotExample(
        submission="[See attached table]",
        verdict=CriterionVerdict.CANNOT_ASSESS,
        reason="The attachment is unavailable.",
    ),
]
# (submission, 0-based option index, reason) as the grader passes them.
ORDINAL_EXAMPLES: list[tuple[str, int, str | None]] = [
    ("Short, direct, and well ordered.", 2, "Very clear."),
    ("It is hot, or maybe not, depending.", 0, None),
    ("See the attached PDF.", 3, "Nothing to assess."),
]
NOMINAL_EXAMPLES: list[tuple[str, int, str | None]] = [
    ("Dear Sir, the temperature is 100 °C.", 0, "Formal salutation."),
    ("yo it's like 100 lol", 2, None),
    ("", 3, "Empty submission."),
]

BUILDER_CASES: dict[str, Callable[[], str]] = {
    "user__binary_positive": lambda: build_user_prompt(POSITIVE, PLAIN),
    "user__binary_positive_query": lambda: build_user_prompt(POSITIVE, PLAIN, QUERY),
    "user__binary_negative_query_reference": lambda: build_user_prompt(
        NEGATIVE, PLAIN, QUERY, REFERENCE
    ),
    "user__binary_positive_structured_reference": lambda: build_user_prompt(
        POSITIVE, STRUCTURED, None, REFERENCE
    ),
    "user__binary_negative_empty_submission": lambda: build_user_prompt(NEGATIVE, ""),
    "few_shot__binary_positive": lambda: build_few_shot_user_prompt(
        criterion=POSITIVE, to_grade=PLAIN, examples=BINARY_EXAMPLES
    ),
    "few_shot__binary_negative_query_reference_reasons": lambda: build_few_shot_user_prompt(
        criterion=NEGATIVE,
        to_grade=PLAIN,
        examples=BINARY_EXAMPLES,
        query=QUERY,
        include_reason=True,
        reference_submission=REFERENCE,
    ),
    "few_shot__binary_positive_structured_query": lambda: build_few_shot_user_prompt(
        criterion=POSITIVE, to_grade=STRUCTURED, examples=BINARY_EXAMPLES, query=QUERY
    ),
    "mc__ordinal_auto_na": lambda: build_multi_choice_user_prompt(ORDINAL_AUTO_NA, PLAIN),
    "mc__ordinal_author_na_query_reference": lambda: build_multi_choice_user_prompt(
        ORDINAL_AUTHOR_NA, PLAIN, QUERY, REFERENCE
    ),
    "mc__nominal_na_structured_query": lambda: build_multi_choice_user_prompt(
        NOMINAL, STRUCTURED, QUERY
    ),
    "mc__ordinal_forced_choice": lambda: build_multi_choice_user_prompt(ORDINAL, PLAIN),
    "mc_few_shot__ordinal_auto_na": lambda: build_multi_choice_few_shot_user_prompt(
        criterion=ORDINAL_AUTO_NA, to_grade=PLAIN, examples=ORDINAL_EXAMPLES
    ),
    "mc_few_shot__nominal_na_query_reference_reasons": lambda: (
        build_multi_choice_few_shot_user_prompt(
            criterion=NOMINAL,
            to_grade=PLAIN,
            examples=NOMINAL_EXAMPLES,
            query=QUERY,
            include_reason=True,
            reference_submission=REFERENCE,
        )
    ),
    "mc_few_shot__ordinal_structured_query": lambda: build_multi_choice_few_shot_user_prompt(
        criterion=ORDINAL_AUTO_NA, to_grade=STRUCTURED, examples=ORDINAL_EXAMPLES, query=QUERY
    ),
}

# ---------------------------------------------------------------------------
# Grader-level capture (mocked clients; no network)
# ---------------------------------------------------------------------------

GRADER_RUBRIC = [POSITIVE, NEGATIVE, ORDINAL, ORDINAL_AUTHOR_NA, NOMINAL]


def _few_shot_training_data() -> RubricDataset:
    """A small labelled pool for the few-shot grader configuration."""
    labels = [
        (CriterionVerdict.MET, CriterionVerdict.UNMET, "Very clear"),
        (CriterionVerdict.UNMET, CriterionVerdict.UNMET, "Very unclear"),
        (CriterionVerdict.MET, CriterionVerdict.MET, "Somewhat clear"),
        (CriterionVerdict.CANNOT_ASSESS, CriterionVerdict.UNMET, "Very clear"),
        (CriterionVerdict.UNMET, CriterionVerdict.MET, "Somewhat clear"),
    ]
    items = [
        DataItem(
            submission=f"Training answer {i}: water boils at {95 + i} degrees.",
            description=f"training item {i}",
            ground_truth=list(gt),
        )
        for i, gt in enumerate(labels)
    ]
    return RubricDataset(
        prompt=QUERY,
        rubric=Rubric([POSITIVE, NEGATIVE, ORDINAL]),
        items=items,
        name="golden-few-shot-pool",
    )


def _grader_configurations() -> dict[str, dict[str, Any]]:
    """Fixed-seed grader configurations and what each grades.

    Judges are always given as positional ``JudgeSpec``s, a spelling every library
    version accepts without a deprecation warning.
    """
    judge = LLMConfig(model="golden-model")
    return {
        "single_judge_structured_query_reference": {
            "grader": lambda: CriterionGrader(judges=[JudgeSpec(judge, "golden")], seed=0),
            "rubric": GRADER_RUBRIC,
            "to_grade": {"thinking": "Recall the value.", "output": PLAIN},
            "query": QUERY,
            "reference_submission": REFERENCE,
        },
        "ensemble_plain_query": {
            "grader": lambda: CriterionGrader(
                judges=[
                    JudgeSpec(judge, "alpha"),
                    JudgeSpec(LLMConfig(model="golden-model-2"), "beta", 2.0),
                ],
                aggregation="weighted",
                seed=0,
            ),
            "rubric": GRADER_RUBRIC,
            "to_grade": PLAIN,
            "query": QUERY,
            "reference_submission": None,
        },
        "few_shot_plain_query": {
            "grader": lambda: CriterionGrader(
                judges=[JudgeSpec(judge, "golden")],
                training_data=_few_shot_training_data(),
                few_shot_config=FewShotConfig(n_examples=2, include_reason=True),
                seed=0,
            ),
            "rubric": [POSITIVE, NEGATIVE, ORDINAL],
            "to_grade": PLAIN,
            "query": QUERY,
            "reference_submission": None,
        },
        "forced_choice_unshuffled_plain": {
            "grader": lambda: CriterionGrader(
                judges=[JudgeSpec(judge, "golden")],
                auto_na_option=False,
                shuffle_options=False,
                seed=0,
            ),
            "rubric": GRADER_RUBRIC,
            "to_grade": PLAIN,
            "query": None,
            "reference_submission": None,
        },
    }


class _RecordingClient:
    """Stands in for ``LLMClient``: records every call and returns a fixed judgment."""

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
        assert response_format is not None
        self._calls.append(
            {
                "judge_id": self._judge_id,
                "response_format": response_format.__name__,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if issubclass(response_format, MultiChoiceJudgment):
            parsed: Any = response_format(selected_option=1, explanation="golden")
        else:
            assert issubclass(response_format, CriterionJudgment)
            parsed = response_format(criterion_status=CriterionVerdict.MET, explanation="golden")
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

    System prompts are stored once under their SHA-256 and referenced by hash, so the
    file stays readable. Calls are sorted, since concurrent criteria may interleave.
    """
    system_prompts: dict[str, str] = {}
    configurations: dict[str, list[dict[str, str]]] = {}
    for name, spec in _grader_configurations().items():
        grader = spec["grader"]()
        calls: list[dict[str, str]] = []
        for judge_id in list(grader._clients):
            grader._clients[judge_id] = _RecordingClient(judge_id, calls)
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
    """Every prompt constant is byte-identical to the captured reference."""
    assert getattr(prompts_module, name) == _read_golden(CONSTANTS_DIR / f"{name}.txt")


@pytest.mark.parametrize("case", sorted(BUILDER_CASES))
def test_user_prompt_builder_matches_golden(case: str) -> None:
    """Every user-prompt builder output is byte-identical to the captured reference."""
    assert BUILDER_CASES[case]() == _read_golden(BUILDERS_DIR / f"{case}.txt")


def test_golden_files_cover_exactly_the_cases() -> None:
    """No stale or missing golden files (a renamed case must not silently pass)."""
    assert sorted(p.stem for p in CONSTANTS_DIR.glob("*.txt")) == sorted(PROMPT_CONSTANT_NAMES)
    assert sorted(p.stem for p in BUILDERS_DIR.glob("*.txt")) == sorted(BUILDER_CASES)


@pytest.mark.asyncio
async def test_grader_sends_golden_prompts() -> None:
    """A fixed-seed grader sends exactly the captured prompts, call for call."""
    expected = json.loads(GRADER_CALLS_PATH.read_text(encoding="utf-8"))
    actual = await _capture_grader_calls()
    assert sorted(actual["configurations"]) == sorted(expected["configurations"])
    for name, calls in expected["configurations"].items():
        assert actual["configurations"][name] == calls, name
    assert actual["system_prompts"] == expected["system_prompts"]


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
        raise SystemExit("usage: python tests/test_prompt_goldens.py --write")
    print(f"Capturing prompt goldens from {prompts_module.__file__}")
    _write_goldens()
