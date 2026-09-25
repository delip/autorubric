"""Capture backward-compatibility fixtures from a library checkout.

The files next to this script were produced by the library as it was before
``judge_model_config`` and the decision-model vote/report fields existed (``main`` at the
merge of PR #17). Tests load them to prove that objects and checkpoints written by that
version still work unchanged:

- ``judge_spec.json``: ``JudgeSpec`` dataclass introspection (field names, ``repr``,
  ``asdict`` keys, ``__match_args__``) and a pickled ``JudgeSpec`` (base64).
- ``checkpoint_items.json``: ``ItemResult.to_dict()`` checkpoint records for an ensemble
  report produced by ``CriterionGrader`` with mocked judges (binary and multi-choice
  criteria, a failed judge call, an extended-thinking trace) and for a single report.
- ``pickled_results.json``: a pickled (base64) evaluation run of that ensemble grader over
  a dataset with ground truth (``EvalResult`` and ``RubricDataset``), the run's
  ``compute_metrics(per_judge=True)`` result and a single ``EvaluationReport``, beside what
  that library read back from the metrics (``model_dump``, both ``summary()`` texts and the
  ``to_dataframe()`` columns).

Only APIs present in every version are used here. Regenerate (only when deliberately
re-baselining) with the command below; the fixed ``PYTHONHASHSEED`` makes the pickle's
sets (pydantic's ``__pydantic_fields_set__``) come out in the same order on every capture::

    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=<checkout>/src \\
        uv run --frozen python tests/golden/legacy/capture_legacy_fixtures.py
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import autorubric
from autorubric import Criterion, CriterionOption, CriterionVerdict, DataItem, TokenUsage
from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.metrics import compute_metrics
from autorubric.rubric import Rubric
from autorubric.types import (
    CriterionReport,
    EvaluationReport,
    MultiChoiceJudgment,
)

HERE = Path(__file__).resolve().parent

SUBMISSION = "Water boils at 100 degrees Celsius at sea level."
DESCRIPTION = "legacy checkpoint item"

RUBRIC = [
    Criterion(name="boiling_point", weight=3.0, requirement="States the boiling point"),
    Criterion(name="unsafe_advice", weight=-2.0, requirement="Recommends untreated water"),
    Criterion(
        name="clarity",
        weight=4.0,
        requirement="How clear is the explanation?",
        scale_type="ordinal",
        options=[
            CriterionOption(label="Very unclear", value=0.0),
            CriterionOption(label="Somewhat clear", value=0.5),
            CriterionOption(label="Very clear", value=1.0),
        ],
    ),
    Criterion(
        name="register",
        weight=1.0,
        requirement="Which register best describes the answer?",
        scale_type="nominal",
        options=[
            CriterionOption(label="Formal", value=1.0),
            CriterionOption(label="Slang", value=0.0),
            CriterionOption(label="N/A", value=0.0, na=True),
        ],
    ),
]


def _judge_spec_fixture() -> dict[str, Any]:
    spec = JudgeSpec(LLMConfig(model="golden-model", max_parallel_requests=3), "gemini", 2.0)
    # A stand-in config keeps the repr independent of LLMConfig's own field list.
    stub = JudgeSpec("stub-config", "judge-a", 0.5)
    replaced = dataclasses.replace(stub, llm_config="other-config", weight=1.5)
    return {
        "field_names": [f.name for f in dataclasses.fields(JudgeSpec)],
        "field_defaults": {
            f.name: (None if f.default is dataclasses.MISSING else f.default)
            for f in dataclasses.fields(JudgeSpec)
        },
        "match_args": list(JudgeSpec.__match_args__),
        "stub_repr": repr(stub),
        "replaced_stub_repr": repr(replaced),
        "stub_asdict": dataclasses.asdict(stub),
        "asdict_keys": list(dataclasses.asdict(spec)),
        "stub_equals_positional_copy": stub == JudgeSpec("stub-config", "judge-a", 0.5),
        "hash_is_none": JudgeSpec.__hash__ is None,
        "pickle_protocol": 4,
        "pickle_b64": base64.b64encode(pickle.dumps(spec, protocol=4)).decode("ascii"),
    }


class _MockJudgeClient:
    """Deterministic stand-in for LLMClient, one behaviour per judge id."""

    def __init__(self, judge_id: str) -> None:
        self._judge_id = judge_id

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        assert response_format is not None
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        if issubclass(response_format, MultiChoiceJudgment):
            if self._judge_id == "beta" and "register" in user_prompt:
                raise ValueError("unparseable judge output")
            parsed: Any = response_format(
                selected_option=1,
                explanation=f"{self._judge_id} picked the first listed option",
            )
        else:
            status = CriterionVerdict.MET if self._judge_id == "alpha" else CriterionVerdict.UNMET
            parsed = response_format(
                criterion_status=status,
                explanation=f"{self._judge_id} says {status.value}",
                reasoning="step-by-step deliberation" if self._judge_id == "alpha" else None,
            )
        return GenerateResult(content="{}", usage=usage, cost=0.001, parsed=parsed)


def _ensemble_grader() -> CriterionGrader:
    grader = CriterionGrader(
        judges=[
            JudgeSpec(LLMConfig(model="golden-model"), "alpha"),
            JudgeSpec(LLMConfig(model="golden-model"), "beta", 2.0),
        ],
        seed=0,
    )
    for judge_id in list(grader._clients):
        grader._clients[judge_id] = _MockJudgeClient(judge_id)
    return grader


async def _ensemble_record() -> dict[str, Any]:
    report = await _ensemble_grader().grade(SUBMISSION, RUBRIC, query="At what temperature?")
    item = DataItem(submission=SUBMISSION, description=DESCRIPTION)
    return ItemResult(item_idx=0, item=item, report=report, duration_seconds=1.25).to_dict()


def _single_report() -> EvaluationReport:
    return EvaluationReport(
        score=0.75,
        raw_score=3.0,
        llm_raw_score=3.0,
        report=[
            CriterionReport(
                name="boiling_point",
                weight=3.0,
                requirement="States the boiling point",
                verdict=CriterionVerdict.MET,
                reason="States 100 C.",
                reasoning="considered the value",
            ),
            CriterionReport(
                name="unsafe_advice",
                weight=-2.0,
                requirement="Recommends untreated water",
                verdict=CriterionVerdict.CANNOT_ASSESS,
                reason="Judge call failed (infrastructure): timeout",
                error="infrastructure: timeout",
            ),
        ],
        token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        completion_cost=0.002,
    )


def _single_record() -> dict[str, Any]:
    item = DataItem(submission=SUBMISSION, description=DESCRIPTION)
    return ItemResult(
        item_idx=1, item=item, report=_single_report(), duration_seconds=0.5
    ).to_dict()


# Submissions of the pickled run, with their ground truth (binary verdicts, option labels).
PICKLED_ITEMS: list[tuple[str, list[CriterionVerdict | str]]] = [
    (SUBMISSION, [CriterionVerdict.MET, CriterionVerdict.UNMET, "Very clear", "Formal"]),
    (
        "Boil it until it bubbles, or just drink straight from the river.",
        [CriterionVerdict.UNMET, CriterionVerdict.MET, "Somewhat clear", "Slang"],
    ),
    (
        "Water boils at 100 C at sea level and at a lower temperature at altitude.",
        [CriterionVerdict.MET, CriterionVerdict.UNMET, "Very clear", "Formal"],
    ),
    ("It depends.", [CriterionVerdict.UNMET, CriterionVerdict.UNMET, "Very unclear", "N/A"]),
]


async def _graded_run() -> tuple[EvalResult, RubricDataset]:
    """The ensemble grader's run over a dataset, built without ``EvalRunner`` (whose
    timestamps and experiment files would make the pickle differ between captures)."""
    dataset = RubricDataset(prompt="At what temperature?", rubric=Rubric(RUBRIC), name="legacy")
    for submission, ground_truth in PICKLED_ITEMS:
        dataset.add_item(submission=submission, description=DESCRIPTION, ground_truth=ground_truth)
    grader = _ensemble_grader()
    item_results = [
        ItemResult(
            item_idx=idx,
            item=item,
            report=await grader.grade(item.submission, RUBRIC, query=dataset.prompt),
            duration_seconds=1.0,
        )
        for idx, item in enumerate(dataset.items)
    ]
    n_items = len(item_results)
    eval_result = EvalResult(
        item_results=item_results,
        total_items=n_items,
        successful_items=n_items,
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=EvalTimingStats.from_durations([1.0] * n_items, float(n_items)),
        started_at=datetime(2026, 9, 1, tzinfo=UTC),
        completed_at=datetime(2026, 9, 1, 0, 0, n_items, tzinfo=UTC),
    )
    return eval_result, dataset


def _pickled_results_fixture() -> dict[str, Any]:
    eval_result, dataset = asyncio.run(_graded_run())
    metrics = compute_metrics(eval_result, dataset, per_judge=True)
    objects = {
        "eval_result": eval_result,
        "dataset": dataset,
        "metrics": metrics,
        "single_report": _single_report(),
    }
    return {
        "pickle_protocol": 4,
        "pickle_b64": base64.b64encode(pickle.dumps(objects, protocol=4)).decode("ascii"),
        "metrics_dump": metrics.model_dump(mode="json"),
        "summary": metrics.summary(),
        "summary_verbose": metrics.summary(verbose=True),
        "frame_columns": list(metrics.to_dataframe().columns),
    }


def main() -> None:
    print(f"Capturing legacy fixtures from {autorubric.__file__}")
    (HERE / "judge_spec.json").write_text(
        json.dumps(_judge_spec_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    records = {
        "submission": SUBMISSION,
        "description": DESCRIPTION,
        "ensemble": asyncio.run(_ensemble_record()),
        "single": _single_record(),
    }
    (HERE / "checkpoint_items.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (HERE / "pickled_results.json").write_text(
        json.dumps(_pickled_results_fixture(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
