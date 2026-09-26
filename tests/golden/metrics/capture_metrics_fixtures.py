"""Capture full-panel ``compute_metrics`` outputs from a library checkout.

``full_panel_metrics.json`` next to this script was produced by the library as it was
before cascade-aware metrics existed (``main`` at the merge of PR #17), on CPython 3.12 on
macOS (arm64). The test ``tests/metrics/test_full_panel_metrics_golden.py`` recomputes
every entry with the current library and requires the outputs to be the same, proving
that full panels (every judge votes on every criterion of every item) get unchanged
metrics: every ``MetricsResult`` field value, both ``summary()`` texts and the
``to_dataframe()`` frame (column names, dtypes and every cell). The current
``model_dump`` adds only the ``JudgeMetrics`` fields ``coverage`` and ``n_pairs``, at
their defaults, to each per-judge entry. Both libraries give the same bits on any one
interpreter and platform, but a float's last bits vary across them (the builtin
``sum()`` of floats before Python 3.12, the platform's BLAS and libm), so the test
compares floats up to its ``FLOAT_TOLERANCE`` and everything else exactly.

Cases:

- hand-built ensemble panels (binary, multi-choice with auto-injected and author NA
  options and a forced-choice error-abstain, a mixed rubric with judge weights) and a
  single-judge run, built by :func:`build_cases` with APIs present in every version;
- a graded panel: three mocked judges graded by ``CriterionGrader`` (binary, ordinal and
  nominal criteria, parse failures), stored as ``ItemResult.to_dict`` checkpoint records
  and rebuilt by :func:`graded_case`.

Each case is computed under several ``compute_metrics`` settings (handling modes,
per-judge metrics, a fixed-seed bootstrap); a setting the library refuses is recorded as
its error. Regenerate (only when deliberately re-baselining) with::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout>/src \\
        uv run --frozen python tests/golden/metrics/capture_metrics_fixtures.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import autorubric
from autorubric import Criterion, CriterionOption, CriterionVerdict, TokenUsage
from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.metrics import MetricsResult, compute_metrics
from autorubric.rubric import Rubric
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    CriterionReport,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
    MultiChoiceJudgment,
    MultiChoiceVerdict,
)

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "full_panel_metrics.json"

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CA = CriterionVerdict.CANNOT_ASSESS

SETTINGS: list[dict[str, Any]] = [
    {"per_judge": True},
    {"per_judge": True, "cannot_assess": "as_category", "na_mode": "as_unmet"},
    {"per_judge": True, "cannot_assess": "as_unmet", "na_mode": "as_category"},
    {"per_judge": True, "bootstrap": True, "n_bootstrap": 40, "seed": 3},
]
"""``compute_metrics`` keyword arguments applied to every case."""


@dataclass
class Case:
    """One evaluation run and its dataset."""

    name: str
    eval_result: EvalResult
    dataset: RubricDataset


def _eval_result(items: list[ItemResult]) -> EvalResult:
    return EvalResult(
        item_results=items,
        total_items=len(items),
        successful_items=len(items),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=None,
        started_at=None,
        completed_at=None,
    )


def _dataset(name: str, criteria: list[Criterion], ground_truth: list[list]) -> RubricDataset:
    dataset = RubricDataset(prompt="Answer the question.", rubric=Rubric(criteria), name=name)
    for i, gt in enumerate(ground_truth):
        dataset.add_item(submission=f"submission {i}", description=f"item {i}", ground_truth=gt)
    return dataset


def _score(values: list[tuple[float, float]]) -> float:
    """Weighted score from (weight, value) pairs, clamped to [0, 1] like the grader."""
    positive = sum(w for w, _ in values if w > 0)
    raw = sum(w * v for w, v in values)
    return min(1.0, max(0.0, raw / positive)) if positive else 0.0


def _binary_value(verdict: CriterionVerdict | None) -> float:
    return 1.0 if verdict == MET else 0.0


def _majority(verdicts: list[CriterionVerdict]) -> CriterionVerdict:
    met = sum(1 for v in verdicts if v == MET)
    unmet = sum(1 for v in verdicts if v == UNMET)
    if met == 0 and unmet == 0:
        return CA
    return MET if met > unmet else UNMET


# =============================================================================
# Hand-built cases
# =============================================================================


def _binary_panel() -> Case:
    criteria = [
        Criterion(name="accurate", requirement="States the facts correctly", weight=3.0),
        Criterion(name="complete", requirement="Covers every part", weight=2.0),
        Criterion(name="unsafe", requirement="Gives unsafe advice", weight=-2.0),
    ]
    judges = ["ja", "jb", "jc"]
    rng = random.Random(11)
    ground_truth: list[list] = []
    items = []
    for i in range(10):
        gt = [MET if rng.random() < 0.55 else UNMET for _ in criteria]
        if i == 4:
            gt[1] = CA
        ground_truth.append(gt)
        crs = []
        judge_values: dict[str, list[tuple[float, float]]] = {j: [] for j in judges}
        final_values = []
        for c_idx, criterion in enumerate(criteria):
            votes = []
            for j_idx, jid in enumerate(judges):
                draw = rng.random()
                truth = gt[c_idx] if gt[c_idx] != CA else UNMET
                if draw < 0.07:
                    verdict, error = CA, None
                elif draw < 0.72:
                    verdict, error = truth, None
                else:
                    verdict, error = (UNMET if truth == MET else MET), None
                if i == 7 and j_idx == 2 and c_idx == 0:
                    verdict, error = CA, "infrastructure: timeout"
                votes.append(
                    JudgeVote(judge_id=jid, verdict=verdict, reason=f"{jid} r", error=error)
                )
                judge_values[jid].append((criterion.weight, _binary_value(verdict)))
            final = _majority([v.verdict for v in votes if v.error is None])
            final_values.append((criterion.weight, _binary_value(final)))
            crs.append(
                EnsembleCriterionReport(
                    criterion=criterion, final_verdict=final, final_reason="agg", votes=votes
                )
            )
        score = _score(final_values)
        report = EnsembleEvaluationReport(
            score=score,
            raw_score=score * 5.0,
            report=crs,
            judge_scores={jid: _score(judge_values[jid]) for jid in judges},
        )
        items.append(ItemResult(item_idx=i, item=None, report=report, duration_seconds=0.1))
    return Case("binary_panel", _eval_result(items), _dataset("binary", criteria, ground_truth))


QUALITY = Criterion(
    name="quality",
    requirement="Overall quality",
    weight=2.0,
    scale_type="ordinal",
    options=[
        CriterionOption(label="Poor", value=0.0),
        CriterionOption(label="Fair", value=0.33),
        CriterionOption(label="Good", value=0.67),
        CriterionOption(label="Excellent", value=1.0),
    ],
)
REGISTER = Criterion(
    name="register",
    requirement="Which register fits the answer?",
    weight=1.0,
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Casual", value=0.5),
        CriterionOption(label="Slang", value=0.0),
        CriterionOption(label="N/A", value=0.0, na=True),
    ],
)
AUTO_NA_LABEL = "Cannot assess / not applicable"


def _mc_vote(
    jid: str, criterion: Criterion, idx: int | None, *, weight: float = 1.0, error=None
) -> MultiChoiceJudgeVote:
    if idx is None:
        return MultiChoiceJudgeVote(
            judge_id=jid,
            selected_index=None,
            selected_label=None,
            value=0.0,
            reason=f"{jid} failed",
            weight=weight,
            na=True,
            error=error,
        )
    if idx < len(criterion.options):
        option = criterion.options[idx]
        label, value, na = option.label, option.value, option.na
    else:  # the grader's auto-injected NA option, appended after the author options
        label, value, na = AUTO_NA_LABEL, 0.0, True
    return MultiChoiceJudgeVote(
        judge_id=jid,
        selected_index=idx,
        selected_label=label,
        value=value,
        reason=f"{jid} r",
        weight=weight,
        na=na,
        error=error,
    )


def _mc_final(criterion: Criterion, votes: list[MultiChoiceJudgeVote]):
    counts: dict[int, int] = {}
    for v in votes:
        if v.error is None and v.selected_index is not None:
            counts[v.selected_index] = counts.get(v.selected_index, 0) + 1
    if not counts:
        idx = None
    else:
        idx = min(counts, key=lambda k: (-counts[k], k))
    ref = _mc_vote("final", criterion, idx)
    return AggregatedMultiChoiceVerdict(
        selected_index=ref.selected_index,
        selected_label=ref.selected_label,
        value=ref.value,
        na=ref.na,
        aggregated_value=ref.value,
    )


def _mc_report(
    criteria: list[Criterion],
    picks: dict[str, list[int | None]],
    errors: dict[tuple[str, int], str],
    weights: dict[str, float],
) -> EnsembleEvaluationReport:
    crs = []
    judge_values: dict[str, list[tuple[float, float]]] = {j: [] for j in picks}
    final_values = []
    for c_idx, criterion in enumerate(criteria):
        votes = [
            _mc_vote(
                jid,
                criterion,
                picks[jid][c_idx],
                weight=weights.get(jid, 1.0),
                error=errors.get((jid, c_idx)),
            )
            for jid in picks
        ]
        final = _mc_final(criterion, votes)
        for v in votes:
            judge_values[v.judge_id].append((criterion.weight, 0.0 if v.na else v.value))
        final_values.append((criterion.weight, 0.0 if final.na else final.value))
        crs.append(
            EnsembleCriterionReport(
                criterion=criterion,
                final_verdict=None,
                final_reason="agg",
                final_multi_choice_verdict=final,
                multi_choice_votes=votes,
            )
        )
    score = _score(final_values)
    return EnsembleEvaluationReport(
        score=score,
        raw_score=score * 3.0,
        report=crs,
        judge_scores={jid: _score(judge_values[jid]) for jid in picks},
    )


def _multi_choice_panel() -> Case:
    criteria = [QUALITY, REGISTER]
    judges = ["ja", "jb", "jc"]
    rng = random.Random(23)
    ground_truth = []
    items = []
    for i in range(10):
        gt_idx = [int(rng.random() * 4), int(rng.random() * 3)]
        if i == 6:
            gt_idx[1] = 3  # author NA in the ground truth
        ground_truth.append([criteria[c].options[gt_idx[c]].label for c in range(2)])
        picks: dict[str, list[int | None]] = {}
        for jid in judges:
            row: list[int | None] = []
            for c in range(2):
                draw = rng.random()
                if draw < 0.6:
                    row.append(gt_idx[c])
                elif draw < 0.9:
                    row.append(min(3, max(0, gt_idx[c] + (1 if draw < 0.75 else -1))))
                else:
                    row.append(3)
            picks[jid] = row
        errors: dict[tuple[str, int], str] = {}
        if i == 2:
            picks["jb"][0] = 4  # the auto-injected NA option on the ordinal criterion
        if i == 5:
            picks["jc"][0] = None  # forced-choice error-abstain: no option selected
            errors[("jc", 0)] = "parse: unparseable judge output"
        items.append(
            ItemResult(
                item_idx=i,
                item=None,
                report=_mc_report(criteria, picks, errors, {}),
                duration_seconds=0.1,
            )
        )
    return Case(
        "multi_choice_panel", _eval_result(items), _dataset("multi", criteria, ground_truth)
    )


def _mixed_rubric_panel() -> Case:
    binary = Criterion(name="cites", requirement="Cites a source", weight=1.5)
    criteria = [binary, QUALITY, REGISTER]
    rng = random.Random(37)
    weights = {"heavy": 2.0, "light": 1.0}
    ground_truth: list[list] = []
    items = []
    for i in range(8):
        gt_bin = MET if rng.random() < 0.5 else UNMET
        gt_q = int(rng.random() * 4)
        gt_r = int(rng.random() * 3)
        ground_truth.append([gt_bin, QUALITY.options[gt_q].label, REGISTER.options[gt_r].label])
        mc = _mc_report(
            [QUALITY, REGISTER],
            {
                "heavy": [gt_q, gt_r if rng.random() < 0.7 else (gt_r + 1) % 3],
                "light": [min(3, gt_q + 1), gt_r],
            },
            {},
            weights,
        )
        votes = [
            JudgeVote(
                judge_id=jid,
                verdict=gt_bin if rng.random() < 0.7 else (UNMET if gt_bin == MET else MET),
                reason=f"{jid} r",
                weight=weights[jid],
            )
            for jid in ("heavy", "light")
        ]
        heavy_final = votes[0].verdict
        binary_cr = EnsembleCriterionReport(
            criterion=binary, final_verdict=heavy_final, final_reason="agg", votes=votes
        )
        mc_values = [(c.criterion.weight, c.score_value) for c in mc.report]
        score = _score([(binary.weight, _binary_value(heavy_final)), *mc_values])
        judge_scores = {
            v.judge_id: _score(
                [
                    (binary.weight, _binary_value(v.verdict)),
                    *[
                        (
                            c.criterion.weight,
                            next(
                                mv.value for mv in c.multi_choice_votes if mv.judge_id == v.judge_id
                            ),
                        )
                        for c in mc.report
                    ],
                ]
            )
            for v in votes
        }
        report = EnsembleEvaluationReport(
            score=score,
            raw_score=score * 4.5,
            report=[binary_cr, *mc.report],
            judge_scores=judge_scores,
        )
        items.append(ItemResult(item_idx=i, item=None, report=report, duration_seconds=0.1))
    return Case(
        "mixed_rubric_panel", _eval_result(items), _dataset("mixed", criteria, ground_truth)
    )


def _single_judge() -> Case:
    binary = Criterion(name="cites", requirement="Cites a source", weight=1.0)
    criteria = [binary, QUALITY]
    rng = random.Random(41)
    ground_truth: list[list] = []
    items = []
    for i in range(6):
        gt_bin = MET if rng.random() < 0.5 else UNMET
        gt_q = int(rng.random() * 4)
        ground_truth.append([gt_bin, QUALITY.options[gt_q].label])
        pred_bin = gt_bin if rng.random() < 0.7 else (UNMET if gt_bin == MET else MET)
        pred_q = min(3, gt_q + int(rng.random() * 2))
        option = QUALITY.options[pred_q]
        report = EvaluationReport(
            score=_score([(1.0, _binary_value(pred_bin)), (2.0, option.value)]),
            raw_score=_binary_value(pred_bin) + 2.0 * option.value,
            report=[
                CriterionReport(
                    name="cites",
                    weight=1.0,
                    requirement=binary.requirement,
                    verdict=pred_bin,
                    reason="r",
                ),
                CriterionReport(
                    name="quality",
                    weight=2.0,
                    requirement=QUALITY.requirement,
                    reason="r",
                    options=QUALITY.options,
                    scale_type="ordinal",
                    multi_choice_verdict=MultiChoiceVerdict(
                        selected_index=pred_q, selected_label=option.label, value=option.value
                    ),
                ),
            ],
        )
        items.append(ItemResult(item_idx=i, item=None, report=report, duration_seconds=0.1))
    return Case("single_judge", _eval_result(items), _dataset("single", criteria, ground_truth))


def build_cases() -> list[Case]:
    """The hand-built cases, identical under every library version."""
    return [_binary_panel(), _multi_choice_panel(), _mixed_rubric_panel(), _single_judge()]


# =============================================================================
# Graded case
# =============================================================================

GRADED_CRITERIA = [
    Criterion(name="accurate", requirement="States the boiling point correctly", weight=3.0),
    Criterion(name="unsafe", requirement="Recommends drinking untreated water", weight=-1.0),
    QUALITY,
    REGISTER,
]
GRADED_GROUND_TRUTH: list[list] = [
    [MET, UNMET, "Good", "Formal"],
    [UNMET, UNMET, "Fair", "Casual"],
    [MET, MET, "Excellent", "Formal"],
    [MET, UNMET, "Poor", "Slang"],
    [UNMET, MET, "Fair", "N/A"],
    [MET, UNMET, "Good", "Casual"],
    [UNMET, UNMET, "Poor", "Formal"],
    [MET, UNMET, "Excellent", "Casual"],
]


def _graded_dataset() -> RubricDataset:
    dataset = RubricDataset(
        prompt="At what temperature does water boil?",
        rubric=Rubric(GRADED_CRITERIA),
        name="graded",
    )
    for i, gt in enumerate(GRADED_GROUND_TRUTH):
        dataset.add_item(
            submission=f"Answer {i}: water boils at {90 + 3 * i} C.",
            description=f"graded item {i}",
            ground_truth=gt,
        )
    return dataset


class _MockJudgeClient:
    """Deterministic stand-in for LLMClient: each answer is a hash of the prompt."""

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
        digest = hashlib.sha256(f"{self._judge_id}\n{user_prompt}".encode()).digest()
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        if issubclass(response_format, MultiChoiceJudgment):
            if self._judge_id == "gamma" and digest[1] < 40:
                raise ValueError("unparseable judge output")
            parsed: Any = response_format(
                selected_option=1 + digest[0] % 3,
                explanation=f"{self._judge_id} picked an option",
            )
        else:
            status = MET if digest[0] % 3 else UNMET
            parsed = response_format(
                criterion_status=status, explanation=f"{self._judge_id} says {status.value}"
            )
        return GenerateResult(content="{}", usage=usage, cost=0.001, parsed=parsed)


async def _grade_records() -> list[dict[str, Any]]:
    grader = CriterionGrader(
        judges=[
            JudgeSpec(LLMConfig(model="golden-model"), "alpha"),
            JudgeSpec(LLMConfig(model="golden-model"), "beta", 2.0),
            JudgeSpec(LLMConfig(model="golden-model"), "gamma"),
        ],
        seed=0,
    )
    for judge_id in list(grader._clients):
        grader._clients[judge_id] = _MockJudgeClient(judge_id)
    dataset = _graded_dataset()
    records = []
    for i, item in enumerate(dataset.items):
        report = await grader.grade(item.submission, GRADED_CRITERIA, query=dataset.prompt)
        records.append(
            ItemResult(item_idx=i, item=item, report=report, duration_seconds=0.5).to_dict()
        )
    return records


def graded_case(records: list[dict[str, Any]]) -> Case:
    """The graded case, rebuilt from its checkpoint records."""
    dataset = _graded_dataset()
    items = [ItemResult.from_dict(record, dataset.items[i]) for i, record in enumerate(records)]
    return Case("graded_panel", _eval_result(items), dataset)


# =============================================================================
# Snapshots
# =============================================================================


def _cell(value: Any) -> str | None:
    """A type-tagged, exactly comparable encoding of one DataFrame cell."""
    import numpy as np

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{bool(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"float:{float(value)!r}"
    if isinstance(value, str):
        return f"str:{value}"
    return f"{type(value).__name__}:{value!r}"


def snapshot(metrics: MetricsResult) -> dict[str, Any]:
    """Everything a user can read from ``metrics``, encoded for exact comparison."""
    frame = metrics.to_dataframe()
    return {
        "model_dump": json.dumps(metrics.model_dump(mode="json"), sort_keys=True),
        "summary": metrics.summary(),
        "summary_verbose": metrics.summary(verbose=True),
        "frame": {
            "columns": [str(c) for c in frame.columns],
            "dtypes": [str(t) for t in frame.dtypes],
            "rows": [[_cell(v) for v in row] for row in frame.itertuples(index=False)],
        },
    }


def outcome(case: Case, setting: dict[str, Any]) -> dict[str, Any]:
    """The snapshot of one ``compute_metrics`` call, or the error it raised."""
    try:
        metrics = compute_metrics(case.eval_result, case.dataset, **setting)
    except ValueError as exc:
        return {"error": f"ValueError: {exc}"}
    return snapshot(metrics)


def main() -> None:
    print(f"Capturing full-panel metrics from {autorubric.__file__}")
    records = asyncio.run(_grade_records())
    cases = [*build_cases(), graded_case(records)]
    golden = {
        "graded_records": records,
        "settings": SETTINGS,
        "outcomes": {case.name: [outcome(case, setting) for setting in SETTINGS] for case in cases},
    }
    GOLDEN.write_text(
        json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
