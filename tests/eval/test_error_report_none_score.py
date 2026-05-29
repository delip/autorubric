"""Issue #7: a grade-FAILURE has no score.

`score`/`raw_score` are ``None`` on the explicit error/empty constructors — never a
fabricated ``0.0`` (which is a valid catastrophic score, indistinguishable from a real
zero). The normal grading path always COMPUTES a real float.
"""

import json
from typing import Any

import pytest

from autorubric import Rubric
from autorubric.dataset import DataItem, RubricDataset
from autorubric.eval import (
    EvalRunner,
    ItemResult,
    _deserialize_ensemble_report,
)
from autorubric.graders import CriterionGrader
from autorubric.graders.base import Grader
from autorubric.llm import LLMConfig
from autorubric.types import (
    Criterion,
    EnsembleEvaluationReport,
    EvaluationReport,
    LengthPenalty,
)


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    return LLMConfig(model="test-model")


# -----------------------------------------------------------------------------
# 7b — error/empty constructors emit None (not a fabricated 0.0)
# -----------------------------------------------------------------------------


def test_create_error_report_score_is_none(mock_llm_config):
    """EvalRunner._create_error_report yields score/raw_score == None (not 0.0)."""
    dataset = RubricDataset(
        prompt="p",
        rubric=Rubric([Criterion(weight=1.0, requirement="r")]),
        items=[DataItem(submission="s", description="d")],
    )
    runner = EvalRunner(dataset=dataset, grader=CriterionGrader(llm_config=mock_llm_config))

    report = runner._create_error_report("boom")

    assert report.score is None
    assert report.raw_score is None
    assert report.error == "boom"


@pytest.mark.asyncio
async def test_no_judge_results_report_score_is_none(mock_llm_config):
    """The 'No judge results to aggregate' empty-path report has None scores."""
    grader = CriterionGrader(llm_config=mock_llm_config)
    report = await grader.aggregate([])

    assert isinstance(report, EnsembleEvaluationReport)
    assert report.score is None
    assert report.raw_score is None
    assert report.llm_raw_score is None
    assert report.error == "No judge results to aggregate"


# -----------------------------------------------------------------------------
# 7b — serialization round-trip preserves None (BOTH report types)
# -----------------------------------------------------------------------------


def test_single_report_none_score_roundtrip():
    """EvaluationReport(score=None, error=...) -> to_dict JSON null -> from_dict None."""
    item = DataItem(submission="s", description="d")
    report = EvaluationReport(score=None, raw_score=None, error="x")
    result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=0.1, error="x")

    d = result.to_dict()
    # The serialized score is JSON null, and survives a real json round-trip.
    assert d["report"]["score"] is None
    assert d["report"]["raw_score"] is None
    reparsed = json.loads(json.dumps(d))
    assert reparsed["report"]["score"] is None

    restored = ItemResult.from_dict(reparsed, item)
    assert restored.report.score is None
    assert restored.report.raw_score is None
    assert restored.report.error == "x"


def test_ensemble_report_none_score_roundtrip():
    """EnsembleEvaluationReport(score=None, error=...) survives the ensemble path."""
    report_data = {
        "score": None,
        "raw_score": None,
        "error": "x",
        "criterion_reports": [],
        "judge_scores": {},
    }
    report = _deserialize_ensemble_report(json.loads(json.dumps(report_data)), None)
    assert isinstance(report, EnsembleEvaluationReport)
    assert report.score is None
    assert report.raw_score is None
    assert report.error == "x"


def test_legacy_float_score_still_loads():
    """A legacy dict carrying a real float score still loads to that float."""
    item = DataItem(submission="s", description="d")
    data = {
        "item_idx": 0,
        "duration_seconds": 1.0,
        "error": None,
        "report": {"score": 0.73, "raw_score": 7.3, "error": None},
    }
    restored = ItemResult.from_dict(json.loads(json.dumps(data)), item)
    assert restored.report.score == 0.73
    assert restored.report.raw_score == 7.3


# -----------------------------------------------------------------------------
# base.py length-penalty guard: a None-score report is returned unchanged
# -----------------------------------------------------------------------------


class _StubGrader(Grader):
    """A grader whose aggregate() returns a score-less (errored) report."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def judge(self, to_grade, rubric, query=None, reference_submission=None):  # noqa: D102
        return []

    async def aggregate(self, judge_results, *, normalize: bool = True):  # noqa: D102
        return EvaluationReport(score=None, raw_score=None, error="No judge results to aggregate")


@pytest.mark.asyncio
async def test_length_penalty_skipped_for_none_score_report():
    """Applying a length_penalty to a None-score report returns it unchanged (no crash)."""
    grader = _StubGrader(length_penalty=LengthPenalty(free_budget=1, max_cap=10))
    report = await grader.grade(
        "a very long submission " * 50, [Criterion(weight=1.0, requirement="r")]
    )

    assert report.score is None
    assert report.raw_score is None
    assert report.error == "No judge results to aggregate"
