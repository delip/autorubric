"""A replayed cascade reproduces the offline cascade simulation of the recorded experiments.

``fixtures/cascade_parity.json`` (loaded by ``fixtures/cascade_fixture.py``) holds a decision
model's recorded P(MET) and an LLM judge's recorded verdict on labelled (item, criterion)
pairs, and the accuracy an offline simulation of a decision-model-first cascade reports on
them at a few margins ``tau``: it defers a pair to the LLM judge when
``|P(MET) - 0.5| < tau``. The library escalates the same pairs at threshold ``2 * tau``.

Here the two recorded judges grade the fixture's items again, offline, through
``CriterionGrader``: the decision model is the real ``DecisionModelClient`` on a scripted SDK
client that returns each recorded P(MET) as a Noul answer, and the LLM judge a scripted
client that returns each recorded verdict. Replaying those two runs at ``2 * tau`` must
defer exactly the simulation's pairs and score exactly the simulation's accuracy, on a
dataset with one rubric (``compute_metrics``' per-criterion path) and on one with a rubric
per item (its pooled path). Nothing reaches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autorubric import (
    EvalResult,
    LLMConfig,
    Rubric,
    RubricDataset,
    calibrate_escalation,
    compute_metrics,
    escalation_stats,
    evaluate,
    replay_escalation,
)
from autorubric.decision import question_id
from autorubric.graders import JudgeSpec
from escalation.cascade_runs import dm, noul
from escalation.fixtures.cascade_fixture import RecordedDataset, load_cascade_fixture

DATASETS = load_cascade_fixture()

# The fixture records neither the items' prompts nor their submissions' text: neither
# escalation nor accuracy reads them. Each item's submission is its text's SHA-256.
PROMPT = "(not recorded)"


def recorded_dataset(recorded: RecordedDataset) -> RubricDataset:
    shared = None if recorded.rubric is None else Rubric(list(recorded.rubric))
    data = RubricDataset(prompt=PROMPT, rubric=shared, name=recorded.name)
    for item in recorded.items:
        data.add_item(
            item.submission_sha256,
            item.description,
            ground_truth=[pair.ground_truth for pair in item.pairs],
            rubric=None if shared is not None else Rubric(list(item.rubric)),
        )
    return data


async def recorded_runs(
    recorded: RecordedDataset, make_grader: Any, decision_model: Any, tmp_path: Path
) -> tuple[RubricDataset, EvalResult, EvalResult]:
    """The fixture's dataset, graded by the decision model alone and by the LLM judge alone,
    each answering as recorded."""
    data = recorded_dataset(recorded)
    decision_model.scripts = {
        item.submission_sha256: {
            question_id(c): noul(pair.p_met) for c, pair in enumerate(item.pairs)
        }
        for item in recorded.items
    }
    llm_script = {
        (item.submission_sha256, criterion.requirement): pair.llm_verdict.value
        for item in recorded.items
        for criterion, pair in zip(item.rubric, item.pairs, strict=True)
    }
    decision_model_grader = make_grader(
        judges=[JudgeSpec(dm(binary_framing="noul_framed", decision_threshold=0.5), "jev")]
    )
    llm_grader = make_grader(
        llm_script=llm_script,
        judges=[JudgeSpec(LLMConfig(model="recorded-llm"), recorded.llm_judge_id)],
    )
    runs = [
        await evaluate(
            data,
            grader,
            show_progress=False,
            experiments_dir=tmp_path,
            experiment_name=f"{recorded.name}-{kind}",
        )
        for kind, grader in (("dm", decision_model_grader), ("llm", llm_grader))
    ]
    return data, runs[0], runs[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded", DATASETS, ids=lambda recorded: recorded.name)
async def test_replay_at_twice_tau_scores_the_simulations_accuracy(
    recorded, make_grader, decision_model, tmp_path
):
    data, dm_run, llm_run = await recorded_runs(recorded, make_grader, decision_model, tmp_path)
    pairs = [pair for item in recorded.items for pair in item.pairs]

    # The runs hold the recorded judgments: the decision model's confidence as the fixture
    # records it, and the LLM judge's verdict.
    for item, dm_item, llm_item in zip(
        recorded.items, dm_run.item_results, llm_run.item_results, strict=True
    ):
        for pair, dm_cr, llm_cr in zip(
            item.pairs, dm_item.report.report, llm_item.report.report, strict=True
        ):
            (dm_vote,) = dm_cr.votes
            assert (dm_vote.verdict, dm_vote.confidence) == (pair.dm_verdict, pair.dm_confidence)
            assert dm_vote.probabilities == pair.dm_probabilities
            assert llm_cr.final_verdict == pair.llm_verdict

    for expected in recorded.expected:
        replay = replay_escalation(dm_run, llm_run, expected.threshold)
        n_escalated = sum(cr.escalated for r in replay.item_results for cr in r.report.report)
        metrics = compute_metrics(replay, data)

        assert n_escalated == expected.n_deferred, expected
        assert metrics.criterion_accuracy == expected.accuracy, expected
        stats = escalation_stats(replay, data)
        assert stats.metric == expected.accuracy
        assert stats.escalation_rate == expected.n_deferred / len(pairs)
        if recorded.rubric is None:
            # A rubric per item: the pooled path, whose pairs are all binary.
            assert metrics.per_criterion == []
            (binary,) = metrics.pooled_by_scale
            assert (binary.scale_type, binary.exact_accuracy) == ("binary", expected.accuracy)


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded", DATASETS, ids=lambda recorded: recorded.name)
async def test_calibrating_at_twice_tau_gives_the_simulations_curve(
    recorded, make_grader, decision_model, tmp_path
):
    data, dm_run, llm_run = await recorded_runs(recorded, make_grader, decision_model, tmp_path)
    n_pairs = sum(len(item.pairs) for item in recorded.items)

    curve = calibrate_escalation(
        data, dm_run, llm_run, thresholds=[expected.threshold for expected in recorded.expected]
    )

    assert [(p.escalation_rate, p.metric) for p in curve.points] == [
        (expected.n_deferred / n_pairs, expected.accuracy) for expected in recorded.expected
    ]
