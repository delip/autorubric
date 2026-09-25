"""Byte-identity goldens for the prompts meta-rubric evaluation and rubric revision send.

Rubric-level guidelines reach two meta prompts: the meta-judge sees them as part of the
rubric under review, and the revision LLM sees them as fixed context. A rubric without
guidelines must produce exactly the prompts the library sent before guidelines existed, so
existing response caches stay valid.

- ``tests/golden/meta/without_guidelines.json`` was captured from the unmodified library
  (``main`` at the merge of PR #17) and is compared against a rubric without guidelines.
- ``tests/golden/meta/with_guidelines.json`` pins the prompts for the same rubric with
  guidelines; it can only be captured from a library that supports them.

Each file holds the exact (system prompt, user prompt, response format) of every meta-judge
call of ``evaluate_rubric_standalone`` and ``evaluate_rubric_in_context`` (mocked clients,
calls sorted, system prompts stored once under their SHA-256), and the (system prompt, user
prompt) of ``revise_rubric`` (default and custom user template) and
``revise_rubric_held_out``. Regenerate deliberately (``PYTHONPATH`` selects which library
checkout is imported; the script prints the path it captured from)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout>/src \\
        uv run --frozen python tests/meta/test_meta_prompt_goldens.py --write
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout>/src \\
        uv run --frozen python tests/meta/test_meta_prompt_goldens.py --write-guidelines

The first writes ``without_guidelines.json``, the second ``with_guidelines.json``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import autorubric
import autorubric.meta._evaluate as evaluate_module
from autorubric import Criterion, CriterionOption, CriterionVerdict, Rubric, TokenUsage
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.meta import (
    ImprovementConfig,
    IssueDetail,
    evaluate_rubric_in_context,
    evaluate_rubric_standalone,
    revise_rubric,
    revise_rubric_held_out,
)
from autorubric.types import CriterionJudgment, MultiChoiceJudgment

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "meta"
WITHOUT_GUIDELINES_PATH = GOLDEN_DIR / "without_guidelines.json"
WITH_GUIDELINES_PATH = GOLDEN_DIR / "with_guidelines.json"

# Braces, quotes, a newline and non-ASCII text: none of it may be interpreted or mangled.
GUIDELINES = (
    "Writers are English-language learners in grades 8–12; judge against that level.\n"
    '"Cited" means any attribution, e.g. {Author, year}, not formal citation style.'
)
TASK_PROMPT = "Write a persuasive essay on school uniforms."

CRITERIA = [
    Criterion(name="thesis", weight=3.0, requirement="States a clear, arguable thesis"),
    Criterion(weight=2.0, requirement="Supports claims with cited evidence"),
    Criterion(
        name="off_topic",
        weight=-2.0,
        requirement="Discusses topics unrelated to school uniforms",
    ),
    Criterion(
        name="organization",
        weight=2.0,
        requirement="How well organized is the essay?",
        scale_type="ordinal",
        options=[
            CriterionOption(label="Disorganized", value=0.0),
            CriterionOption(label="Mostly organized", value=0.5),
            CriterionOption(label="Well organized", value=1.0),
            CriterionOption(label="Not an essay", value=0.0, na=True),
        ],
    ),
]
ISSUES = [
    IssueDetail(
        criterion_name="clear_requirements",
        requirement="Each criterion has a clear, unambiguous requirement",
        weight=10.0,
        is_antipattern=False,
        feedback="Criterion 2 leaves 'cited' undefined.",
    ),
    IssueDetail(
        criterion_name="double_barreled",
        requirement="A criterion assesses two things at once",
        weight=-8.0,
        is_antipattern=True,
        feedback="Criterion 4 mixes structure and flow.",
    ),
]
VALIDATION_TEXT = "## Validation\nSpearman rho = 0.42 over 12 items."
HISTORY_TEXT = "Iteration 0: quality 55%.\n"
DIAGNOSTICS_TEXT = "Criterion 1 (thesis): accuracy 60%, FN rate 40%."
CUSTOM_TEMPLATE = (
    "Task: {task_prompt}\nRubric:\n{original_criteria}\nIssues:\n{issues_text}\n"
    "{validation_text}\nHistory:\n{history_text}"
)
REVISED_JSON = json.dumps(
    [
        {"name": c.name, "weight": c.weight, "requirement": c.requirement + " (revised)"}
        for c in CRITERIA
    ]
)


def _rubric(with_guidelines: bool) -> Rubric:
    if with_guidelines:
        return Rubric(list(CRITERIA), guidelines=GUIDELINES)
    return Rubric(list(CRITERIA))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _RecordingClient:
    """Stands in for a grader's ``LLMClient``: records each call, returns a fixed verdict."""

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
            parsed: Any = response_format(selected_option=0, explanation="golden")
        else:
            assert issubclass(response_format, CriterionJudgment)
            parsed = response_format(criterion_status=CriterionVerdict.MET, explanation="golden")
        return GenerateResult(
            content="{}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=None,
            parsed=parsed,
        )


async def _meta_judge_calls(
    run: str, rubric: Rubric, system_prompts: dict[str, str]
) -> list[dict[str, str]]:
    """Run one meta-rubric evaluation with recording clients; return its sorted calls."""
    calls: list[dict[str, str]] = []
    real_grader = evaluate_module.CriterionGrader

    def recording_grader(*args: Any, **kwargs: Any) -> Any:
        grader = real_grader(*args, **kwargs)
        for judge_id in list(grader._clients):
            grader._clients[judge_id] = _RecordingClient(judge_id, calls)
        return grader

    llm = LLMConfig(model="golden-meta-judge")
    with patch.object(evaluate_module, "CriterionGrader", recording_grader):
        if run == "standalone":
            await evaluate_rubric_standalone(rubric, llm)
        else:
            await evaluate_rubric_in_context(rubric, TASK_PROMPT, llm)
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
    return records


async def _revision_prompts(
    run: str, rubric: Rubric, system_prompts: dict[str, str]
) -> dict[str, str]:
    """Run one revision with a mocked revision LLM; return the prompts it was sent."""
    config = ImprovementConfig(
        eval_llm=LLMConfig(model="golden-eval"),
        revision_llm=LLMConfig(model="golden-revision"),
    )
    capture: dict[str, str] = {}
    client = MagicMock()

    async def generate(system_prompt: str, user_prompt: str, **kwargs: Any) -> GenerateResult:
        return GenerateResult(content=REVISED_JSON, cost=None)

    client.generate = generate
    with patch("autorubric.meta._improve.LLMClient", return_value=client):
        if run == "revise_rubric":
            await revise_rubric(
                rubric, TASK_PROMPT, ISSUES, VALIDATION_TEXT, HISTORY_TEXT, config, _capture=capture
            )
        elif run == "revise_rubric_custom_template":
            await revise_rubric(
                rubric,
                None,
                ISSUES,
                VALIDATION_TEXT,
                HISTORY_TEXT,
                config,
                user_prompt_template=CUSTOM_TEMPLATE,
                _capture=capture,
            )
        else:
            await revise_rubric_held_out(
                rubric, TASK_PROMPT, DIAGNOSTICS_TEXT, HISTORY_TEXT, config, _capture=capture
            )
    digest = _sha256(capture["system_prompt"])
    system_prompts[digest] = capture["system_prompt"]
    return {"system_prompt_sha256": digest, "user_prompt": capture["user_prompt"]}


async def _capture(with_guidelines: bool) -> dict[str, Any]:
    rubric = _rubric(with_guidelines)
    system_prompts: dict[str, str] = {}
    meta_judge = {
        run: await _meta_judge_calls(run, rubric, system_prompts)
        for run in ("standalone", "in_context")
    }
    revision = {
        run: await _revision_prompts(run, rubric, system_prompts)
        for run in ("revise_rubric", "revise_rubric_custom_template", "revise_rubric_held_out")
    }
    return {"system_prompts": system_prompts, "meta_judge": meta_judge, "revision": revision}


def _golden(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("with_guidelines", "path"),
    [(False, WITHOUT_GUIDELINES_PATH), (True, WITH_GUIDELINES_PATH)],
    ids=["without_guidelines", "with_guidelines"],
)
async def test_meta_prompts_match_golden(with_guidelines: bool, path: Path) -> None:
    """Meta-judge and revision prompts are byte-identical to the captured reference."""
    expected = _golden(path)
    actual = await _capture(with_guidelines)
    assert actual["meta_judge"] == expected["meta_judge"]
    assert actual["revision"] == expected["revision"]
    assert actual["system_prompts"] == expected["system_prompts"]


def _write(path: Path, with_guidelines: bool) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    captured = asyncio.run(_capture(with_guidelines))
    path.write_text(
        json.dumps(captured, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    modes = {"--write": False, "--write-guidelines": True}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(
            "usage: python tests/meta/test_meta_prompt_goldens.py --write | --write-guidelines"
        )
    with_guidelines = modes[sys.argv[1]]
    target = WITH_GUIDELINES_PATH if with_guidelines else WITHOUT_GUIDELINES_PATH
    print(f"Capturing {target.name} from {autorubric.__file__}")
    _write(target, with_guidelines)
