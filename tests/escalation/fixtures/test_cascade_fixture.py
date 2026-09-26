"""The recorded cascade fixture is internally consistent.

``cascade_parity.json`` (loaded by ``cascade_fixture.py``, built by
``build_cascade_fixture.py``) pairs a decision model's recorded P(MET) with an LLM judge's
recorded verdict on labelled (item, criterion) pairs, and stores the accuracy an offline
simulation of a decision-model-first cascade reports on them at a few margins ``tau``.
These tests check that the stored records can stand in for two ``EvalResult`` objects
built offline, and that the stored numbers follow from the stored records:

- every pair is a binary judgment from both judges, with no abstention or error;
- the decision model's verdict, probabilities and confidence are what the library derives
  from its P(MET);
- each expected point is the simulation's rule applied to the stored verdicts;
- at each point's library threshold ``2 * tau``, a confidence below the threshold (what
  escalates a decision model's judged, non-abstaining vote) selects exactly the pairs the
  simulation defers.

No test here makes a request.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from cascade_fixture import (
    RecordedDataset,
    RecordedPair,
    load_cascade_fixture,
    simulated_point,
)
from typesafe_sdk import SystemOneResponse

from autorubric import CriterionVerdict, DecisionModelConfig
from autorubric.decision import answer_to_report, question_id

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET

DATASETS = load_cascade_fixture()


def pairs_of(dataset: RecordedDataset) -> list[RecordedPair]:
    return [pair for item in dataset.items for pair in item.pairs]


def noul_answers(criterion_idx: int, p_met: float) -> dict[str, Any]:
    """A Noul answer for one question, decoded from its wire form like a response."""
    payload = {
        "model": "jev-test",
        "usage": {"input_tokens": 1},
        "answers": {question_id(criterion_idx): {"type": "noul", "noul": p_met}},
    }
    return SystemOneResponse.model_validate_json(json.dumps(payload)).answers


def test_the_fixture_holds_a_shared_rubric_dataset_and_a_per_item_rubric_dataset():
    assert [dataset.rubric is not None for dataset in DATASETS] == [True, False]
    for dataset in DATASETS:
        assert dataset.items
        assert dataset.expected
        digests = [item.submission_sha256 for item in dataset.items]
        assert len(set(digests)) == len(digests)
        assert all(
            len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in digests
        )
        for item in dataset.items:
            if dataset.rubric is not None:
                assert item.rubric == dataset.rubric
            assert len(item.pairs) == len(item.rubric) > 0


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda dataset: dataset.name)
def test_every_pair_is_a_binary_judgment_from_both_judges(dataset):
    for item in dataset.items:
        assert all(criterion.options is None for criterion in item.rubric)
        for pair in item.pairs:
            assert pair.ground_truth in (MET, UNMET)
            assert pair.dm_verdict in (MET, UNMET)
            assert pair.llm_verdict in (MET, UNMET)
            # At exactly 0.5 the simulation's verdict is MET and the library's the worst
            # case, so the fixture holds no such pair.
            assert 0.0 <= pair.p_met <= 1.0 and pair.p_met != 0.5


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda dataset: dataset.name)
def test_decision_model_fields_are_the_librarys_reading_of_its_probability(dataset):
    config = DecisionModelConfig(model="jev-latest", binary_framing="noul_framed")
    for item in dataset.items:
        for criterion_idx, (criterion, pair) in enumerate(zip(item.rubric, item.pairs)):
            report = answer_to_report(
                criterion, criterion_idx, noul_answers(criterion_idx, pair.p_met), config
            )
            assert report.verdict == pair.dm_verdict
            assert report.probabilities == pair.dm_probabilities
            assert report.confidence == pair.dm_confidence
            # The simulation's own verdict rule, P(MET) >= 0.5, agrees on every pair.
            assert (pair.p_met >= 0.5) == (pair.dm_verdict == MET)


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda dataset: dataset.name)
def test_expected_points_are_the_simulation_on_the_stored_verdicts(dataset):
    pairs = pairs_of(dataset)
    for point in dataset.expected:
        assert point.threshold == 2 * point.tau
        assert simulated_point(pairs, point.tau) == point
        assert point.accuracy == point.n_correct / len(pairs)


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda dataset: dataset.name)
def test_each_point_defers_some_pairs_and_changes_some_verdicts(dataset):
    """A point that deferred nothing, or everything, could not tell a replay's rule apart."""
    pairs = pairs_of(dataset)
    n_deferred = [point.n_deferred for point in dataset.expected]
    assert 0 < n_deferred[0] and n_deferred[-1] < len(pairs)
    assert n_deferred == sorted(set(n_deferred))
    deferred = [pair for pair in pairs if abs(pair.p_met - 0.5) < dataset.expected[-1].tau]
    assert any(pair.llm_verdict != pair.dm_verdict for pair in deferred)


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda dataset: dataset.name)
def test_the_library_threshold_escalates_exactly_the_deferred_pairs(dataset):
    for point in dataset.expected:
        for pair in pairs_of(dataset):
            deferred = abs(pair.p_met - 0.5) < point.tau
            escalated = pair.dm_confidence < point.threshold
            assert escalated == deferred, (pair.p_met, point.tau)
