"""``escalation_stats``: a cascade run, live or replayed, summarized as an ``EscalationPoint``.

The statistics are read from the reports' ``escalated`` and ``superseded`` flags over pooled
(item, criterion) pairs, so they need no shared rubric. A hand-built cascade run pins every
definition (labelled pairs, what an abstention or a failure counts as, which items count),
and small hand-built runs the deferred pairs on which the decision model and the fallback
are compared; cascades graded through ``evaluate`` check that a replay reads exactly as the
live run it reproduces, on a dataset with one rubric and on one with a rubric per item.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest

from autorubric import (
    AggregatedMultiChoiceVerdict,
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DataItem,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EscalationPoint,
    EvalResult,
    EvalTimingStats,
    EvaluationReport,
    ItemResult,
    JudgeVote,
    MetricsResult,
    MultiChoiceJudgeVote,
    Rubric,
    RubricDataset,
    compute_metrics,
    escalation_stats,
    replay_escalation,
)
from escalation.cascade_runs import (
    ITEMS,
    PER_CRITERION,
    PER_ITEM_LLM_SCRIPT,
    THRESHOLD,
    dataset,
    per_item_dataset,
    per_item_decision_model_scripts,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CANNOT_ASSESS = CriterionVerdict.CANNOT_ASSESS

# =============================================================================
# A hand-built cascade run
# =============================================================================

A = Criterion(name="a", weight=1.0, requirement="Req A")
B = Criterion(name="b", weight=1.0, requirement="Req B")
C = Criterion(
    name="c",
    weight=1.0,
    requirement="Req C",
    options=[
        CriterionOption(label="Bad", value=0.0),
        CriterionOption(label="Okay", value=0.5),
        CriterionOption(label="Good", value=1.0),
        CriterionOption(label="N/A", value=0.0, na=True),
    ],
)


def vote(
    criterion: Criterion,
    answer: Any,
    *,
    judge: str = "jev",
    confidence: float | None = None,
    error: str | None = None,
    superseded: bool = False,
) -> JudgeVote | MultiChoiceJudgeVote:
    """A vote: a verdict on a binary criterion, an option label on a multi-choice one."""
    common: dict[str, Any] = {
        "judge_id": judge,
        "reason": None if judge == "jev" else "because",
        "confidence": confidence,
        "probabilities": None if confidence is None else {"x": 1.0},
        "error": error,
        "superseded": superseded,
    }
    if criterion.options is None:
        return JudgeVote(verdict=answer, **common)
    idx = [o.label for o in criterion.options].index(answer)
    option = criterion.options[idx]
    return MultiChoiceJudgeVote(
        selected_index=idx, selected_label=option.label, value=option.value, na=option.na, **common
    )


def criterion_report(
    criterion: Criterion, votes: list[Any], answer: Any, *, escalated: bool = False
) -> EnsembleCriterionReport:
    if criterion.options is None:
        return EnsembleCriterionReport(
            criterion=criterion,
            final_verdict=answer,
            final_reason=None,
            votes=votes,
            escalated=escalated,
        )
    idx = [o.label for o in criterion.options].index(answer)
    option = criterion.options[idx]
    return EnsembleCriterionReport(
        criterion=criterion,
        final_verdict=None,
        final_reason=None,
        final_multi_choice_verdict=AggregatedMultiChoiceVerdict(
            selected_index=idx,
            selected_label=option.label,
            value=option.value,
            na=option.na,
            aggregated_value=option.value,
        ),
        multi_choice_votes=votes,
        escalated=escalated,
    )


def kept(criterion: Criterion, answer: Any) -> EnsembleCriterionReport:
    return criterion_report(criterion, [vote(criterion, answer, confidence=0.9)], answer)


def escalated(
    criterion: Criterion,
    dm_answer: Any,
    fallback: Any,
    *,
    dm_error: str | None = None,
) -> EnsembleCriterionReport:
    dm_vote = vote(
        criterion,
        dm_answer,
        confidence=None if dm_error else 0.2,
        error=dm_error,
        superseded=True,
    )
    fallback_vote = vote(criterion, fallback, judge="escalation")
    return criterion_report(criterion, [dm_vote, fallback_vote], fallback, escalated=True)


def item_result(
    idx: int, reports: list[EnsembleCriterionReport], *, cost: float | None, seconds: float
) -> ItemResult:
    report = EnsembleEvaluationReport(
        score=0.5,
        raw_score=1.5,
        report=reports,
        judge_scores={"jev": 0.5, "escalation": None},
        completion_cost=cost,
    )
    return ItemResult(
        item_idx=idx,
        item=DataItem(submission=f"answer {idx}", description=""),
        report=report,
        duration_seconds=seconds,
    )


def eval_result(item_results: list[ItemResult]) -> EvalResult:
    now = datetime.now()
    return EvalResult(
        item_results=item_results,
        total_items=len(item_results),
        successful_items=sum(r.error is None for r in item_results),
        failed_items=sum(r.error is not None for r in item_results),
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=EvalTimingStats.from_durations(
            [r.duration_seconds for r in item_results], 1.0
        ),
        started_at=now,
        completed_at=now,
    )


def hand_built() -> tuple[EvalResult, RubricDataset]:
    """A cascade run whose statistics are known: per item and criterion, what the decision
    model (jev) and the fallback said, against the truth.

    ====  ==========================  ======================  ==========================
    item  a                           b                       c
    ====  ==========================  ======================  ==========================
    0     kept: MET, right            escalated: jev UNMET,   escalated: jev abstains
                                      right; fallback right   (N/A); fallback Good, right
    1     escalated: jev abstains;    kept: UNMET; truth      escalated: jev Good, wrong;
          fallback UNMET, right       abstains                fallback Okay, wrong
    2     escalated: jev failed       kept: UNMET, wrong      kept: Good; truth N/A
          (UNMET, as the truth);
          fallback abstains
    3     escalated (no truth)        kept (no truth)         kept (no truth)
    4     grading failed (1 s, cost 5)
    5     kept: MET, right            kept: UNMET, right      kept: Okay, right
    6     not in the dataset (100 s, cost 100)
    ====  ==========================  ======================  ==========================
    """
    data = RubricDataset(prompt="Q", rubric=Rubric([A, B, C]), name="hand-built")
    for truth in (
        [MET, UNMET, "Good"],
        [UNMET, CANNOT_ASSESS, "Bad"],
        [UNMET, MET, "N/A"],
        None,
        [MET, MET, "Good"],
        [MET, UNMET, "Okay"],
    ):
        data.add_item("answer", "item", ground_truth=truth)

    failed = ItemResult(
        item_idx=4,
        item=data.items[4],
        report=EvaluationReport(score=None, error="grading crashed", completion_cost=5.0),
        duration_seconds=1.0,
        error="grading crashed",
    )
    result = eval_result(
        [
            item_result(
                0,
                [kept(A, MET), escalated(B, UNMET, UNMET), escalated(C, "N/A", "Good")],
                cost=0.01,
                seconds=2.0,
            ),
            item_result(
                1,
                [
                    escalated(A, CANNOT_ASSESS, UNMET),
                    kept(B, UNMET),
                    escalated(C, "Good", "Okay"),
                ],
                cost=0.02,
                seconds=3.0,
            ),
            item_result(
                2,
                [
                    escalated(A, UNMET, CANNOT_ASSESS, dm_error="unknown: crashed"),
                    kept(B, UNMET),
                    kept(C, "Good"),
                ],
                cost=None,
                seconds=4.0,
            ),
            item_result(
                3,
                [escalated(A, MET, MET), kept(B, MET), kept(C, "Okay")],
                cost=0.03,
                seconds=5.0,
            ),
            failed,
            item_result(5, [kept(A, MET), kept(B, UNMET), kept(C, "Okay")], cost=0.04, seconds=6.0),
            item_result(6, [kept(A, MET), kept(B, MET), kept(C, "Good")], cost=100.0, seconds=100),
        ]
    )
    return result, data


class TestHandBuiltRun:
    def test_every_statistic(self):
        result, data = hand_built()
        point = escalation_stats(result, data)
        assert point.model_dump(exclude={"cost_usd"}) == {
            "threshold": None,
            "per_criterion": None,
            # 6 of the 15 pairs of the graded items in the dataset (0 to 3 and 5).
            "escalation_rate": 6 / 15,
            # 0a, 5a, 5b, 5c right, 2b wrong; 1b and 2c are unlabelled, 3 has no truth.
            "dm_accuracy_kept": 4 / 5,
            # The pairs jev deferred, 0b and 1c: jev right on 0b and wrong on 1c, as the
            # fallback is. jev abstained on 0c and 1a and failed on 2a, which escalate at any
            # threshold, so neither judge is measured on them (the fallback was right on 0c
            # and 1a).
            "dm_accuracy_escalated": 1 / 2,
            "fallback_accuracy_escalated": 1 / 2,
            "metric": compute_metrics(result, data).criterion_accuracy,
            # Items 0 to 5, the failed one included; not the one outside the dataset.
            "compute_seconds": 21.0,
            "calibration_fingerprint": None,
        }
        # Items 0 to 3 and 5; not the failed item's, nor the one outside the dataset.
        assert point.cost_usd == pytest.approx(0.10)

    def test_a_metric_by_name_or_function(self):
        result, data = hand_built()
        metrics = compute_metrics(result, data)
        assert escalation_stats(result, data, metric="mean_kappa").metric == metrics.mean_kappa
        assert escalation_stats(result, data, metric=lambda m: m.n_items).metric == 4
        assert escalation_stats(result, data, metric=lambda m: None).metric is None

    @pytest.mark.parametrize("nan", [float("nan"), np.float64("nan")], ids=["float", "numpy"])
    def test_a_nan_metric_is_undefined(self, nan):
        """A function may give NaN for an undefined value (e.g. the mean of no values); the
        point records it as ``None``, as ``compute_metrics`` records an undefined metric."""
        result, data = hand_built()
        assert escalation_stats(result, data, metric=lambda m: nan).metric is None

    @pytest.mark.parametrize(
        ("number", "expected"),
        [(np.float32(0.5), 0.5), (np.int64(3), 3.0), (np.float64(0.25), 0.25)],
        ids=["float32", "int64", "float64"],
    )
    def test_a_numpy_number_is_a_number(self, number, expected):
        """A function computing the metric with NumPy may give any NumPy scalar (a count
        of booleans is an ``int64``, the mean of a ``float32`` array a ``float32``); the
        point records it as a float."""
        result, data = hand_built()
        metric = escalation_stats(result, data, metric=lambda m: number).metric
        assert metric == expected and type(metric) is float

    @pytest.mark.parametrize(
        ("metric", "message"),
        [
            ("accuracy_of_everything", "not an attribute of MetricsResult"),
            ("per_criterion", "must give a number or None, not list"),
            (lambda m: "0.9", "must give a number or None, not str"),
            (lambda m: True, "must give a number or None, not bool"),
        ],
    )
    def test_a_metric_must_be_a_number(self, metric, message):
        result, data = hand_built()
        with pytest.raises(ValueError, match=message):
            escalation_stats(result, data, metric=metric)

    def test_an_accuracy_over_no_pairs_is_none(self):
        _, data = hand_built()
        # Nothing escalated, and no cost recorded.
        kept_only = eval_result(
            [item_result(0, [kept(A, MET), kept(B, UNMET), kept(C, "Good")], cost=None, seconds=1)]
        )
        point = escalation_stats(kept_only, data)
        assert point.escalation_rate == 0.0
        assert point.dm_accuracy_kept == 1.0
        assert point.dm_accuracy_escalated is None
        assert point.fallback_accuracy_escalated is None
        assert point.cost_usd is None

    def test_a_run_that_is_not_a_cascade_is_refused(self):
        result, data = hand_built()
        panel = criterion_report(A, [vote(A, MET), vote(A, MET, judge="escalation")], MET)
        ensemble = item_result(0, [panel, kept(B, UNMET), kept(C, "Good")], cost=None, seconds=1)
        with pytest.raises(ValueError, match=r"item 0, criterion 0 is not escalated, but has 2"):
            escalation_stats(eval_result([ensemble]), data)
        unflagged = escalated(A, MET, UNMET).model_copy(update={"escalated": False})
        flat = item_result(0, [unflagged, kept(B, UNMET), kept(C, "Good")], cost=None, seconds=1)
        with pytest.raises(ValueError, match=r"is not escalated, but has 2 votes"):
            escalation_stats(eval_result([flat]), data)
        no_superseded = escalated(A, MET, UNMET)
        no_superseded = no_superseded.model_copy(
            update={
                "votes": [v.model_copy(update={"superseded": False}) for v in no_superseded.votes]
            }
        )
        broken = item_result(
            0, [no_superseded, kept(B, UNMET), kept(C, "Good")], cost=None, seconds=1
        )
        with pytest.raises(ValueError, match=r"is escalated, but its first vote is not the only"):
            escalation_stats(eval_result([broken]), data)
        all_superseded = escalated(A, MET, UNMET)
        all_superseded = all_superseded.model_copy(
            update={
                "votes": [v.model_copy(update={"superseded": True}) for v in all_superseded.votes]
            }
        )
        broken = item_result(
            0, [all_superseded, kept(B, UNMET), kept(C, "Good")], cost=None, seconds=1
        )
        with pytest.raises(ValueError, match=r"is escalated, but its first vote is not the only"):
            escalation_stats(eval_result([broken]), data)

    def test_a_run_whose_first_votes_are_not_a_decision_models_is_refused(self):
        """A kept criterion's vote, and an escalated one's superseded vote, are the decision
        model's: they carry a confidence, unless the judgment failed."""
        _, data = hand_built()
        llm_alone = criterion_report(A, [vote(A, MET, judge="gemini")], MET)
        run = item_result(0, [llm_alone, kept(B, UNMET), kept(C, "Good")], cost=None, seconds=1)
        with pytest.raises(
            ValueError, match=r"item 0, criterion 0: the vote of 'gemini' carries no"
        ):
            escalation_stats(eval_result([run]), data)

        unsure = vote(A, UNMET, judge="gemini", superseded=True)
        escalated_llm = criterion_report(A, [unsure, vote(A, MET, judge="b")], MET, escalated=True)
        run = item_result(0, [escalated_llm, kept(B, UNMET), kept(C, "Good")], cost=None, seconds=1)
        with pytest.raises(
            ValueError, match=r"item 0, criterion 0: the vote of 'gemini' carries no"
        ):
            escalation_stats(eval_result([run]), data)

        # A failed judgment has no confidence: it is the decision model's all the same.
        failed = escalated(A, UNMET, MET, dm_error="parse: no answer")
        run = item_result(0, [failed, kept(B, UNMET), kept(C, "Good")], cost=None, seconds=1)
        assert escalation_stats(eval_result([run]), data).escalation_rate == 1 / 3

    def test_an_item_whose_ground_truth_does_not_resolve_has_no_labels(self):
        """``compute_metrics`` skips an item whose ground truth does not resolve against its
        rubric (here an option the criterion lacks), so its pairs are unlabelled, as those
        of an item without ground truth: they count in ``escalation_rate`` alone."""
        result, data = hand_built()

        def with_item_5_truth(truth: list[Any] | None) -> RubricDataset:
            changed = RubricDataset(prompt=data.prompt, rubric=data.rubric, name=data.name)
            for i, item in enumerate(data.items):
                changed.add_item(
                    item.submission,
                    item.description,
                    ground_truth=truth if i == 5 else item.ground_truth,
                )
            return changed

        unresolvable = with_item_5_truth([MET, UNMET, "Z"])
        point = escalation_stats(result, unresolvable)

        assert point == escalation_stats(result, with_item_5_truth(None))
        assert point.escalation_rate == 6 / 15  # item 5's pairs are still pairs
        assert point.dm_accuracy_kept == 1 / 2  # 0a right, 2b wrong; not 5a, 5b, 5c
        assert point.metric == compute_metrics(result, unresolvable).criterion_accuracy

    def test_a_report_must_match_its_rubric(self):
        result, data = hand_built()
        short = item_result(0, [kept(A, MET), kept(B, UNMET)], cost=None, seconds=1)
        with pytest.raises(ValueError, match=r"result item 0 has 2 criteria; its rubric .* has 3"):
            escalation_stats(eval_result([short]), data)


def unsure(criterion: Criterion, answer: Any) -> EnsembleCriterionReport:
    """A criterion the decision model answered at confidence 0.2, and the cascade kept."""
    return criterion_report(criterion, [vote(criterion, answer, confidence=0.2)], answer)


class TestTheDeferredPairs:
    """``dm_accuracy_escalated`` and ``fallback_accuracy_escalated`` are measured on the same
    pairs: those the decision model deferred (it answered, at a confidence below the
    threshold) and the fallback answered. Comparing them tells whether the fallback beats
    the decision model where it defers."""

    @staticmethod
    def run(deferred: bool) -> tuple[EvalResult, RubricDataset]:
        """Seven items graded against [a, b]; the truth is MET on a and UNMET on b.

        - Items 0 to 3: the decision model is unsure of a (0.2) and right on items 0 and 1,
          wrong on 2 and 3; the fallback is right on all four. ``deferred`` is whether the
          cascade escalated them. It is sure of b, and right.
        - Items 4 to 6: its request failed, so both criteria escalate at any threshold; the
          fallback is wrong on both.
        """
        data = RubricDataset(prompt="Q", rubric=Rubric([A, B]), name="deferred")
        for _ in range(7):
            data.add_item("answer", "item", ground_truth=[MET, UNMET])
        failed = "infrastructure: connection refused"
        results = []
        for i in range(7):
            if i < 4:
                answer = MET if i < 2 else UNMET
                a = escalated(A, answer, MET) if deferred else unsure(A, answer)
                reports = [a, kept(B, UNMET)]
            else:
                reports = [
                    escalated(A, CANNOT_ASSESS, UNMET, dm_error=failed),
                    escalated(B, CANNOT_ASSESS, MET, dm_error=failed),
                ]
            results.append(item_result(i, reports, cost=None, seconds=1.0))
        return eval_result(results), data

    def test_forced_escalations_have_no_decision_model_answer_to_compare(self):
        """The fallback's six wrong verdicts where the decision model failed are in the run's
        metric at any threshold; they are not the pairs it deferred."""
        forced = escalation_stats(*self.run(deferred=False))
        assert forced.escalation_rate == 6 / 14
        assert forced.dm_accuracy_kept == 6 / 8
        assert forced.dm_accuracy_escalated is None
        assert forced.fallback_accuracy_escalated is None
        assert forced.metric == 6 / 14

        point = escalation_stats(*self.run(deferred=True))
        assert point.escalation_rate == 10 / 14
        assert point.dm_accuracy_kept == 4 / 4
        # Where the decision model deferred, the fallback beats it (4 of 4 against 2 of 4)
        assert (point.dm_accuracy_escalated, point.fallback_accuracy_escalated) == (2 / 4, 1.0)
        # ... and deferring pays: two wrong verdicts become right.
        assert point.metric == 8 / 14

    def test_a_deferred_pair_the_fallback_abstains_on_is_in_neither(self):
        """A final verdict that abstains is no prediction, so the pair leaves both
        accuracies, and the two stay measured on the same pairs."""
        data = RubricDataset(prompt="Q", rubric=Rubric([A, B]), name="abstaining fallback")
        for _ in range(3):
            data.add_item("answer", "item", ground_truth=[MET, UNMET])
        # The decision model deferred a on every item: wrong on 0, where the fallback
        # abstains; right on 1 and wrong on 2, where the fallback is right.
        result = eval_result(
            [
                item_result(
                    i, [escalated(A, dm_answer, fallback), kept(B, UNMET)], cost=None, seconds=1
                )
                for i, (dm_answer, fallback) in enumerate(
                    [(UNMET, CANNOT_ASSESS), (MET, MET), (UNMET, MET)]
                )
            ]
        )
        point = escalation_stats(result, data)
        assert point.escalation_rate == 3 / 6
        assert (point.dm_accuracy_escalated, point.fallback_accuracy_escalated) == (1 / 2, 1.0)


# =============================================================================
# Live cascades and their replays
# =============================================================================


def assert_same_statistics(replayed: EscalationPoint, live: EscalationPoint) -> None:
    """The same point, but for cost and compute time, which a replay estimates."""
    estimated = {"cost_usd", "compute_seconds"}
    assert replayed.model_dump(exclude=estimated) == live.model_dump(exclude=estimated)


class TestLiveAndReplayedRuns:
    @pytest.mark.asyncio
    async def test_a_replay_reads_as_the_live_cascade(self, cascade_runs):
        runs = await cascade_runs()
        live = escalation_stats(runs.live, runs.data)
        replay = replay_escalation(runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION)
        replayed = escalation_stats(replay, runs.data)

        assert_same_statistics(replayed, live)
        # 15 of the 7 items' 35 criteria escalate (see the replay tests' ESCALATED).
        assert live.escalation_rate == 15 / 35
        # All 15 are labelled. The decision model failed or abstained on 7 (item 2's tone,
        # item 3's myth, all of item 4), which escalate at any threshold: neither judge is
        # measured on them. It deferred the other 8, of which 7 have a final verdict that
        # does not abstain: the terse item's tone, whose LLM call was rate limited, ends in
        # the NA option and is left out; its sentences, whose LLM call crashed, ends in the
        # worst case (UNMET, as the truth) and counts. On those 7 the decision model is
        # right on 4 (the tone of items 0, 3 and 6, item 1's light) and so is the fallback
        # (the tone of items 0 and 3, item 1's clarity, the terse item's sentences).
        assert live.dm_accuracy_escalated == 4 / 7
        assert live.fallback_accuracy_escalated == 4 / 7
        assert live.cost_usd == pytest.approx(runs.live.total_completion_cost)
        assert live.compute_seconds == pytest.approx(
            sum(r.duration_seconds for r in runs.live.item_results)
        )
        assert replayed.cost_usd == pytest.approx(replay.total_completion_cost)
        assert replayed.compute_seconds == pytest.approx(
            sum(r.duration_seconds for r in replay.item_results)
        )

    @pytest.mark.asyncio
    async def test_the_decision_models_accuracy_where_it_kept_and_where_it_escalated(
        self, cascade_runs
    ):
        # Items 0, 1 and 6: no error, no abstention.
        data = dataset([ITEMS[0], ITEMS[1], ITEMS[6]])
        runs = await cascade_runs(data=data, per_criterion=None)

        nothing = escalation_stats(replay_escalation(runs.dm, runs.llm, 0.0), data)
        assert nothing.escalation_rate == 0.0
        assert nothing.dm_accuracy_escalated is None
        assert nothing.fallback_accuracy_escalated is None
        # The decision model is wrong on item 1's clarity (Very clear, not Mostly clear) and
        # tone (Formal, not Casual), and on item 6's clarity: 12 of the 15 pairs are right.
        assert nothing.dm_accuracy_kept == 12 / 15

        everything = escalation_stats(replay_escalation(runs.dm, runs.llm, 1.0), data)
        assert everything.escalation_rate == 1.0
        assert everything.dm_accuracy_kept is None
        # The LLM abstains (the NA option) on item 6's clarity, which leaves both
        # accuracies: the decision model is right on 12 of the other 14 pairs, the LLM on 5.
        assert everything.dm_accuracy_escalated == 12 / 14
        assert everything.fallback_accuracy_escalated == 5 / 14

        # At 0.5, item 1's light (right) and clarity (wrong) escalate.
        live = escalation_stats(runs.live, data)
        assert live.escalation_rate == 2 / 15
        assert live.dm_accuracy_escalated == 1 / 2
        assert live.dm_accuracy_kept == 11 / 13

    @pytest.mark.asyncio
    async def test_a_dataset_with_a_rubric_per_item(self, cascade_runs, decision_model):
        """Pooled pairs over items with different rubrics, where there is no per-criterion
        table (``compute_metrics`` takes its pooled path). The pairs are annotated in
        ``PER_ITEM``."""
        data = per_item_dataset()
        decision_model.scripts = {**decision_model.scripts, **per_item_decision_model_scripts()}
        runs = await cascade_runs(data=data, per_criterion=None, llm_script=PER_ITEM_LLM_SCRIPT)

        def pooled_binary(metrics: MetricsResult) -> float | None:
            return next(
                e.exact_accuracy for e in metrics.pooled_by_scale if e.scale_type == "binary"
            )

        live = escalation_stats(runs.live, data, metric=pooled_binary)
        replay = replay_escalation(runs.dm, runs.llm, THRESHOLD)
        assert_same_statistics(escalation_stats(replay, data, metric=pooled_binary), live)

        assert compute_metrics(runs.live, data).per_criterion == []  # the pooled path
        assert live.escalation_rate == 3 / 7
        assert live.dm_accuracy_kept == 2 / 4
        assert live.dm_accuracy_escalated == 2 / 3
        assert live.fallback_accuracy_escalated == 1 / 3
        assert live.metric == pooled_binary(compute_metrics(runs.live, data))
