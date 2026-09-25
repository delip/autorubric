"""Checkpoint compatibility of the probabilistic-judge vote/report fields.

New fields (``probabilities``, ``confidence``, ``superseded``, ``escalated``), ``None``
reasons and ``None`` judge scores round-trip through ``ItemResult.to_dict`` /
``from_dict``. Checkpoints written before these fields existed (a fixture captured from that
library version) load with the field defaults, and re-serializing them only *adds* the new
keys, each holding its default.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from autorubric import Criterion, CriterionOption, CriterionVerdict, DataItem
from autorubric.eval import ItemResult
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    CriterionReport,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
    TokenUsage,
)

LEGACY_CHECKPOINTS = (
    Path(__file__).resolve().parents[1] / "golden" / "legacy" / "checkpoint_items.json"
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET

OPTIONS = [
    CriterionOption(label="Poor", value=0.0),
    CriterionOption(label="Good", value=1.0),
    CriterionOption(label="N/A", value=0.0, na=True),
]

# Keys added to each serialized object, with their defaults.
NEW_VOTE_KEYS = {"probabilities": None, "confidence": None, "superseded": False}
NEW_CRITERION_REPORT_KEYS = {"probabilities": None, "confidence": None}
NEW_ENSEMBLE_CRITERION_KEYS = {"escalated": False}


def _item() -> DataItem:
    return DataItem(submission="Water boils at 100 C.", description="item")


def _roundtrip(result: ItemResult) -> ItemResult:
    return ItemResult.from_dict(json.loads(json.dumps(result.to_dict())), result.item)


def _cascade_report() -> EnsembleEvaluationReport:
    """An ensemble report shaped like an escalated decision-model criterion."""
    binary = EnsembleCriterionReport(
        criterion=Criterion(name="facts", weight=2.0, requirement="States the facts"),
        final_verdict=UNMET,
        final_reason="escalation: missing the value",
        votes=[
            JudgeVote(
                judge_id="jev",
                verdict=MET,
                reason=None,
                probabilities={"MET": 0.55, "UNMET": 0.45},
                confidence=0.1,
                superseded=True,
            ),
            JudgeVote(judge_id="escalation", verdict=UNMET, reason="missing the value"),
        ],
        escalated=True,
    )
    mc = EnsembleCriterionReport(
        criterion=Criterion(
            name="quality",
            weight=1.0,
            requirement="How good?",
            scale_type="ordinal",
            options=OPTIONS,
        ),
        final_verdict=None,
        final_reason=None,
        final_multi_choice_verdict=AggregatedMultiChoiceVerdict(
            selected_index=1, selected_label="Good", value=1.0, aggregated_value=1.0
        ),
        multi_choice_votes=[
            MultiChoiceJudgeVote(
                judge_id="jev",
                selected_index=1,
                selected_label="Good",
                value=1.0,
                reason=None,
                probabilities={"0": 0.1, "1": 0.85, "2": 0.05},
                confidence=0.775,
            )
        ],
    )
    return EnsembleEvaluationReport(
        score=0.3333333333333333,
        raw_score=1.0,
        report=[binary, mc],
        judge_scores={"jev": 1.0, "escalation": None},
        mean_agreement=1.0,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        completion_cost=0.003,
    )


class TestNewFieldsRoundTrip:
    def test_ensemble_report_round_trips(self):
        result = ItemResult(item_idx=0, item=_item(), report=_cascade_report(), duration_seconds=1)
        restored = _roundtrip(result)
        assert isinstance(restored.report, EnsembleEvaluationReport)
        assert restored.report.report == result.report.report
        assert restored.report.judge_scores == {"jev": 1.0, "escalation": None}
        binary, mc = restored.report.report
        assert binary.escalated is True
        assert binary.votes[0].superseded is True
        assert binary.votes[0].probabilities == {"MET": 0.55, "UNMET": 0.45}
        assert binary.votes[0].confidence == pytest.approx(0.1)
        assert binary.votes[0].reason is None
        assert binary.agreement == pytest.approx(1.0)
        assert mc.final_reason is None
        assert mc.multi_choice_votes[0].probabilities == {"0": 0.1, "1": 0.85, "2": 0.05}
        # A second pass is byte-identical (agreement recompute stays idempotent).
        assert _roundtrip(restored).to_dict() == restored.to_dict() == result.to_dict()

    def test_none_judge_score_serializes_as_null(self):
        result = ItemResult(item_idx=0, item=_item(), report=_cascade_report(), duration_seconds=1)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["report"]["judge_scores"] == {"jev": 1.0, "escalation": None}

    def test_single_report_round_trips(self):
        report = EvaluationReport(
            score=1.0,
            raw_score=2.0,
            report=[
                CriterionReport(
                    name="facts",
                    weight=2.0,
                    requirement="States the facts",
                    verdict=MET,
                    reason=None,
                    probabilities={"MET": 0.9, "UNMET": 0.1},
                    confidence=0.8,
                )
            ],
        )
        result = ItemResult(item_idx=3, item=_item(), report=report, duration_seconds=0.1)
        restored = _roundtrip(result)
        assert restored.report.report == report.report
        assert restored.to_dict() == result.to_dict()


def _legacy() -> dict[str, Any]:
    return json.loads(LEGACY_CHECKPOINTS.read_text(encoding="utf-8"))


def _strip_new_keys(serialized: dict[str, Any]) -> dict[str, Any]:
    """Remove the new keys, asserting each holds its default (additive-only check)."""
    out = copy.deepcopy(serialized)

    def pop_defaults(obj: dict[str, Any], defaults: dict[str, Any]) -> None:
        for key, default in defaults.items():
            assert key in obj, key
            assert obj.pop(key) == default, key

    for cr in out["report"]["criterion_reports"]:
        if "votes" in cr:  # ensemble criterion report
            pop_defaults(cr, NEW_ENSEMBLE_CRITERION_KEYS)
            for vote in cr["votes"] + cr["multi_choice_votes"]:
                pop_defaults(vote, NEW_VOTE_KEYS)
        else:
            pop_defaults(cr, NEW_CRITERION_REPORT_KEYS)
    return out


class TestLegacyCheckpoints:
    @pytest.mark.parametrize("kind", ["ensemble", "single"])
    def test_loads_with_defaults_and_reserializes_additively(self, kind):
        legacy = _legacy()
        item = DataItem(submission=legacy["submission"], description=legacy["description"])
        record = legacy[kind]
        restored = ItemResult.from_dict(copy.deepcopy(record), item)
        assert restored.report.report is not None
        for cr in restored.report.report:
            if isinstance(cr, EnsembleCriterionReport):
                assert cr.escalated is False
                for vote in [*cr.votes, *cr.multi_choice_votes]:
                    assert vote.probabilities is None
                    assert vote.confidence is None
                    assert vote.superseded is False
                    assert isinstance(vote.reason, str)
                assert isinstance(cr.final_reason, str)
            else:
                assert cr.probabilities is None
                assert cr.confidence is None
                assert isinstance(cr.reason, str)
        assert _strip_new_keys(restored.to_dict()) == record

    def test_legacy_ensemble_agreement_scores_and_reasons_are_unchanged(self):
        legacy = _legacy()
        item = DataItem(submission=legacy["submission"], description=legacy["description"])
        record = legacy["ensemble"]
        restored = ItemResult.from_dict(copy.deepcopy(record), item)
        assert isinstance(restored.report, EnsembleEvaluationReport)
        assert restored.report.judge_scores == record["report"]["judge_scores"]
        assert restored.report.report is not None
        for ecr, raw in zip(restored.report.report, record["report"]["criterion_reports"]):
            assert ecr.agreement == raw["agreement"]
            assert ecr.final_reason == raw["final_reason"]
            assert ecr.error == raw["error"]
