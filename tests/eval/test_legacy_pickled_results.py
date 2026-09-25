"""Results pickled before the decision-model fields existed load and work unchanged.

``tests/golden/legacy/pickled_results.json`` holds objects pickled by the library as it was
before the vote/report fields ``probabilities``, ``confidence``, ``superseded`` and
``escalated`` and the ``JudgeMetrics`` fields ``coverage`` and ``n_pairs`` existed (see
``capture_legacy_fixtures.py`` next to it): an evaluation run of a two-judge
``CriterionGrader`` ensemble over a dataset with ground truth, the run's
``compute_metrics(per_judge=True)`` result and a single ``EvaluationReport``, beside what
that library read back from the metrics.

Pydantic restores a pickle's attributes exactly as they were stored, so a field added since
would be missing and reading it would raise ``AttributeError``. Each model that gained a
defaulted field gives a field its pickle lacks the field's default, so the restored objects
equal the same objects validated today and work wherever they did.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import compute_metrics
from autorubric.types import (
    CriterionReport,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
)

LEGACY_PICKLE = Path(__file__).resolve().parents[1] / "golden" / "legacy" / "pickled_results.json"

NEW_VOTE_FIELDS = {"probabilities": None, "confidence": None, "superseded": False}
NEW_CRITERION_REPORT_FIELDS = {"probabilities": None, "confidence": None}
NEW_ENSEMBLE_CRITERION_FIELDS = {"escalated": False}
NEW_JUDGE_METRICS_FIELDS = {"coverage": "full", "n_pairs": None}


@pytest.fixture
def fixture() -> dict[str, Any]:
    return json.loads(LEGACY_PICKLE.read_text(encoding="utf-8"))


@pytest.fixture
def legacy(fixture: dict[str, Any]) -> dict[str, Any]:
    """The unpickled objects: ``eval_result``, ``dataset``, ``metrics``, ``single_report``."""
    return pickle.loads(base64.b64decode(fixture["pickle_b64"]))


def _criterion_reports(eval_result: EvalResult) -> list[EnsembleCriterionReport]:
    reports = []
    for item_result in eval_result.item_results:
        assert isinstance(item_result.report, EnsembleEvaluationReport)
        reports.extend(item_result.report.report or [])
    return reports


def _votes(eval_result: EvalResult) -> list[JudgeVote | MultiChoiceJudgeVote]:
    return [
        vote
        for cr in _criterion_reports(eval_result)
        for vote in [*cr.votes, *cr.multi_choice_votes]
    ]


def _restored_models(legacy: dict[str, Any]) -> list[tuple[BaseModel, dict[str, Any]]]:
    """Every restored object of a model that gained fields, with those fields' defaults."""
    eval_result = legacy["eval_result"]
    single_reports = legacy["single_report"].report
    models: list[tuple[BaseModel, dict[str, Any]]] = [
        *((vote, NEW_VOTE_FIELDS) for vote in _votes(eval_result)),
        *((cr, NEW_ENSEMBLE_CRITERION_FIELDS) for cr in _criterion_reports(eval_result)),
        *((cr, NEW_CRITERION_REPORT_FIELDS) for cr in single_reports),
        *((jm, NEW_JUDGE_METRICS_FIELDS) for jm in legacy["metrics"].per_judge.values()),
    ]
    kinds = {type(model) for model, _ in models}
    assert {JudgeVote, MultiChoiceJudgeVote, EnsembleCriterionReport, CriterionReport} <= kinds
    assert len(legacy["metrics"].per_judge) == 2
    return models


class TestLegacyPickledResults:
    def test_fields_added_since_read_their_defaults(self, legacy):
        for model, new_fields in _restored_models(legacy):
            for name, default in new_fields.items():
                assert getattr(model, name) == default, (type(model).__name__, name)

    def test_restored_objects_equal_the_same_objects_validated_today(self, legacy):
        for model, new_fields in _restored_models(legacy):
            assert model == type(model).model_validate(model.model_dump())
            # A default the pickle lacked was not set, as validation leaves a defaulted field.
            assert not model.model_fields_set & new_fields.keys()

    @pytest.mark.parametrize("per_judge", [False, True])
    def test_compute_metrics_equals_that_of_the_run_loaded_from_its_checkpoint(
        self, legacy, per_judge
    ):
        eval_result, dataset = legacy["eval_result"], legacy["dataset"]
        reloaded = dataclasses.replace(
            eval_result,
            item_results=[
                ItemResult.from_dict(json.loads(json.dumps(ir.to_dict())), ir.item)
                for ir in eval_result.item_results
            ],
        )
        metrics = compute_metrics(eval_result, dataset, per_judge=per_judge)
        expected = compute_metrics(reloaded, dataset, per_judge=per_judge)
        assert metrics.model_dump() == expected.model_dump()
        assert metrics.summary(verbose=True) == expected.summary(verbose=True)

    def test_a_report_built_from_restored_votes_computes_the_same_agreement(self, legacy):
        for cr in _criterion_reports(legacy["eval_result"]):
            rebuilt = EnsembleCriterionReport(
                criterion=cr.criterion,
                final_verdict=cr.final_verdict,
                final_reason=cr.final_reason,
                votes=cr.votes,
                final_multi_choice_verdict=cr.final_multi_choice_verdict,
                multi_choice_votes=cr.multi_choice_votes,
                error=cr.error,
            )
            assert rebuilt.agreement == cr.agreement

    def test_the_metrics_dump_and_render_as_that_library_did(self, fixture, legacy):
        metrics = legacy["metrics"]
        assert metrics.model_dump(mode="json") == fixture["metrics_dump"]
        assert metrics.summary() == fixture["summary"]
        assert metrics.summary(verbose=True) == fixture["summary_verbose"]
        assert list(metrics.to_dataframe().columns) == fixture["frame_columns"]
