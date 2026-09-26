"""``replay_escalation``: a confidence cascade rebuilt from two recorded runs, with no calls.

Three runs grade the same items: a live cascade (a decision model, and LLM escalation
judges behind it), the decision model alone, and the LLM judge(s) alone with the cascade's
``seed`` and ``judge_id``s. Replaying the two separate runs at the cascade's thresholds must
build exactly the reports the live cascade built: the same escalated criteria, votes,
``superseded``/``escalated`` flags, final verdicts and reasons, errors, agreement, scores
and ``judge_scores``, and so the same metrics. Only cost and time differ: the replay
estimates them, because per-criterion LLM cost and time are not recorded.

The decision model is the real ``DecisionModelClient`` on a scripted SDK client, and the LLM
judges answer from a hash of their prompts (``cascade_runs``), so a replay can equal the
live run only if every escalation judge saw the very prompt it saw in the LLM run, shuffled
option order included. Nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from collections import Counter
from typing import Any

import pytest

import autorubric
from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    Criterion,
    EnsembleEvaluationReport,
    EscalationPoint,
    EvalResult,
    EvalTimingStats,
    EvaluationReport,
    FewShotConfig,
    ItemResult,
    LLMConfig,
    RubricDataset,
    compute_metrics,
    escalation_stats,
    evaluate,
    replay_escalation,
)
from autorubric.escalation import _calibration_fingerprint
from autorubric.eval import ExperimentManifest, _compute_dataset_hash
from autorubric.graders import JudgeSpec
from escalation.cascade_runs import (
    CLARITY,
    DM_INPUT_TOKENS,
    DM_PRICE,
    ITEMS,
    LLM_COST,
    PANEL,
    PER_CRITERION,
    PER_ITEM_LLM_SCRIPT,
    RUBRIC,
    SEED,
    TERSE,
    THRESHOLD,
    TONE,
    Runs,
    dataset,
    dm,
    llm_scripts,
    per_item_dataset,
    per_item_decision_model_scripts,
)

# Which criteria the cascade escalates, per item, at THRESHOLD with PER_CRITERION.
ESCALATED = [
    [False, False, False, True, False],  # sure; tone 0.96 < 0.99
    [True, False, True, True, False],  # light 0.2, clarity 0.2, tone 0.96
    [False, False, False, True, False],  # tone abstains
    [False, True, False, True, False],  # myth unanswered (parse error), tone 0.96
    [True, True, True, True, True],  # the request failed
    [False, False, False, True, True],  # myth 0.1 kept (threshold 0), tone 1/3, sentences 0.4
    [False, False, False, True, False],  # light 0.5, at the threshold, kept; tone 0.96
]

# Report fields a replay estimates (cost) or cannot know (token usage).
ESTIMATED = {"token_usage", "completion_cost"}


def escalated_flags(result: EvalResult) -> list[list[bool]]:
    return [[cr.escalated for cr in r.report.report] for r in result.item_results]


def assert_replays(replay: EvalResult, live: EvalResult) -> None:
    """``replay`` holds exactly the live cascade's reports, but for cost and token usage."""
    assert [r.item_idx for r in replay.item_results] == [r.item_idx for r in live.item_results]
    for replayed, graded in zip(replay.item_results, live.item_results, strict=True):
        assert replayed.error == graded.error
        assert isinstance(replayed.report, EnsembleEvaluationReport)
        assert replayed.report.model_dump(exclude=ESTIMATED) == graded.report.model_dump(
            exclude=ESTIMATED
        )
        assert replayed.report.token_usage is None


def assert_sent_in_the_llm_run(runs: Runs) -> None:
    """Every prompt the cascade's escalation judges sent, the LLM run sent too."""
    sent = Counter(runs.user_prompts(runs.llm_grader))
    escalated = Counter(runs.user_prompts(runs.cascade))
    assert escalated and escalated <= sent


def replay_at_the_cascade(runs: Runs, **kwargs: Any) -> EvalResult:
    return replay_escalation(runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION, **kwargs)


# =============================================================================
# The replay equals the live cascade
# =============================================================================


class TestReplayEqualsTheLiveCascade:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("price", [DM_PRICE, None], ids=["priced", "unpriced"])
    async def test_one_llm_judge(self, cascade_runs, price):
        runs = await cascade_runs(dm_config=dm(input_cost_per_token=price))
        assert escalated_flags(runs.live) == ESCALATED

        replay = replay_at_the_cascade(runs)

        assert_replays(replay, runs.live)
        assert_sent_in_the_llm_run(runs)
        assert escalated_flags(replay) == ESCALATED
        assert compute_metrics(replay, runs.data, per_judge=True) == compute_metrics(
            runs.live, runs.data, per_judge=True
        )
        # The escalated multi-choice criteria were asked with shuffled options.
        orders = [
            vote.shuffle_order
            for r in replay.item_results
            for cr in r.report.report
            if cr.escalated
            for vote in cr.multi_choice_votes
            if not vote.superseded
        ]
        assert orders and all(order is not None for order in orders)
        assert any(order != sorted(order) for order in orders if order is not None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("judges", "settings"),
        [
            (PANEL, {}),
            (
                [JudgeSpec(LLMConfig(model="a"), "a", 2.0), JudgeSpec(LLMConfig(model="b"), "b")],
                {"aggregation": "weighted", "ordinal_aggregation": "weighted_mean"},
            ),
            (
                PANEL,
                {
                    "aggregation": "unanimous",
                    "ordinal_aggregation": "min",
                    "nominal_aggregation": "unanimous",
                },
            ),
        ],
        ids=["majority", "weighted", "unanimous"],
    )
    async def test_an_llm_panel_aggregated_as_the_llm_run_aggregated_it(
        self, cascade_runs, judges, settings
    ):
        runs = await cascade_runs(judges, **settings)
        replay = replay_at_the_cascade(runs)

        assert_replays(replay, runs.live)
        assert compute_metrics(replay, runs.data, per_judge=True) == compute_metrics(
            runs.live, runs.data, per_judge=True
        )
        # The two judges disagree somewhere, so the aggregation settings are exercised.
        assert any(
            len({v.selected_index if cr.criterion.is_multi_choice else v.verdict for v in votes})
            > 1
            for r in replay.item_results
            for cr in r.report.report
            if cr.escalated
            for votes in [[v for v in (cr.votes or cr.multi_choice_votes) if not v.superseded]]
        )

    @pytest.mark.asyncio
    async def test_a_dataset_with_a_rubric_per_item(self, cascade_runs, decision_model):
        data = per_item_dataset()
        decision_model.scripts = {**decision_model.scripts, **per_item_decision_model_scripts()}
        runs = await cascade_runs(
            PANEL, data=data, per_criterion={"tone": 0.1}, llm_script=PER_ITEM_LLM_SCRIPT
        )
        replay = replay_escalation(runs.dm, runs.llm, THRESHOLD, per_criterion={"tone": 0.1})

        assert_replays(replay, runs.live)
        # Light on the first item and sentences on the second; tone (0.2) stays below 0.5
        # but clears its own threshold of 0.1.
        assert escalated_flags(replay) == [[True, False], [False, False, True], [False, False]]
        assert compute_metrics(replay, data) == compute_metrics(runs.live, data)

    @pytest.mark.asyncio
    async def test_few_shot_examples(self, cascade_runs):
        """Few-shot examples are selected by ``seed`` and ``judge_id`` as well, so the LLM
        run's prompts, examples included, are the escalation judge's."""
        few_shot = {"training_data": dataset(), "few_shot_config": FewShotConfig(n_examples=2)}
        runs = await cascade_runs(PANEL, **few_shot)
        replay = replay_at_the_cascade(runs)
        assert_replays(replay, runs.live)

        escalation_prompts = runs.user_prompts(runs.cascade)
        assert all("<examples>" in prompt for prompt in escalation_prompts)
        assert_sent_in_the_llm_run(runs)

    @pytest.mark.asyncio
    async def test_the_scoring_settings_are_the_replays(self, cascade_runs):
        scoring = {
            "normalize": False,
            "cannot_assess_config": CannotAssessConfig(
                strategy=CannotAssessStrategy.PARTIAL, partial_credit=0.3
            ),
        }
        runs = await cascade_runs(**scoring)
        replay = replay_at_the_cascade(runs, **scoring)
        assert_replays(replay, runs.live)

    @pytest.mark.asyncio
    async def test_replaying_saved_experiments(self, cascade_runs):
        runs = await cascade_runs(PANEL)
        dm_run = EvalResult.from_experiment(runs.dm.experiment_dir)
        llm_run = EvalResult.from_experiment(runs.llm.experiment_dir)

        replay = replay_escalation(dm_run, llm_run, THRESHOLD, per_criterion=PER_CRITERION)

        assert_replays(replay, runs.live)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("saved", ["dm_result", "llm_result"])
    async def test_replaying_a_saved_experiment_with_a_live_run(self, cascade_runs, saved):
        """With one run loaded with ``EvalResult.from_experiment``, which records no
        submissions, and the other live, the items pair all the same; the replay holds the
        items the live run recorded, so a point calibrated on them is recognized."""
        runs = await cascade_runs()
        dm_run, llm_run = runs.dm, runs.llm
        if saved == "dm_result":
            dm_run = EvalResult.from_experiment(runs.dm.experiment_dir)
        else:
            llm_run = EvalResult.from_experiment(runs.llm.experiment_dir)

        point = calibrated_point(fingerprint(runs.data))
        with pytest.warns(UserWarning, match=r"calibrated on the items it replays"):
            replay = replay_escalation(dm_run, llm_run, point)

        assert_replays(replay, runs.live)
        assert [r.item for r in replay.item_results] == [r.item for r in runs.live.item_results]

    @pytest.mark.asyncio
    async def test_a_replay_round_trips_through_the_experiment_format(self, cascade_runs, tmp_path):
        """Saved as ``EvalRunner`` saves a run (a manifest, and one ``ItemResult.to_dict`` line
        per item), a replay loads back with ``EvalResult.from_experiment`` unchanged: its
        reports, and so its metrics and diagnostics."""
        runs = await cascade_runs(PANEL)
        replay = replay_at_the_cascade(runs)
        manifest = ExperimentManifest(
            experiment_name=replay.experiment_name or "",
            created_at=replay.started_at,
            dataset_name=runs.data.name,
            dataset_hash=_compute_dataset_hash(runs.data),
            total_items=replay.total_items,
            status="completed",
            completed_indices={r.item_idx for r in replay.item_results},
            started_at=replay.started_at,
            completed_at=replay.completed_at,
            total_duration_seconds=replay.timing_stats.total_duration_seconds,
        )
        (tmp_path / "manifest.json").write_text(json.dumps(manifest.to_dict()))
        (tmp_path / "items.jsonl").write_text(
            "".join(json.dumps(r.to_dict()) + "\n" for r in replay.item_results)
        )

        loaded = EvalResult.from_experiment(tmp_path)

        assert loaded.experiment_name == replay.experiment_name
        assert [r.report for r in loaded.item_results] == [r.report for r in replay.item_results]
        assert compute_metrics(loaded, runs.data, per_judge=True) == compute_metrics(
            replay, runs.data, per_judge=True
        )
        assert escalation_stats(loaded, runs.data) == escalation_stats(replay, runs.data)

    @pytest.mark.asyncio
    async def test_an_escalated_criterion_is_the_llm_runs_report_behind_the_superseded_vote(
        self, cascade_runs
    ):
        runs = await cascade_runs(PANEL)
        replay = replay_at_the_cascade(runs)

        for replayed, dm_item, llm_item in zip(
            replay.item_results, runs.dm.item_results, runs.llm.item_results, strict=True
        ):
            for cr, dm_cr, llm_cr in zip(
                replayed.report.report, dm_item.report.report, llm_item.report.report, strict=True
            ):
                if not cr.escalated:
                    assert cr == dm_cr
                    continue
                field = "multi_choice_votes" if cr.criterion.is_multi_choice else "votes"
                dm_vote, *llm_votes = getattr(cr, field)
                assert dm_vote == getattr(dm_cr, field)[0].model_copy(update={"superseded": True})
                assert llm_votes == getattr(llm_cr, field)
                assert cr.model_dump(exclude={field, "escalated"}) == llm_cr.model_dump(
                    exclude={field, "escalated"}
                )

    @pytest.mark.asyncio
    async def test_judge_scores_hold_the_decision_models_own_score_and_none_by_role(
        self, cascade_runs
    ):
        runs = await cascade_runs(PANEL)
        replay = replay_at_the_cascade(runs)
        for replayed, dm_item in zip(replay.item_results, runs.dm.item_results, strict=True):
            assert replayed.report.judge_scores == {
                "default": dm_item.report.judge_scores["default"],
                "a": None,
                "b": None,
            }

    @pytest.mark.asyncio
    async def test_an_llm_run_with_another_seed_replays_only_in_distribution(self, cascade_runs):
        runs = await cascade_runs(llm_seed=SEED + 1)
        replay = replay_at_the_cascade(runs)

        differing = {
            (r.item_idx, cr.criterion.name)
            for r, graded in zip(replay.item_results, runs.live.item_results, strict=True)
            for cr, live_cr in zip(r.report.report, graded.report.report, strict=True)
            if cr != live_cr
        }
        # Only escalated multi-choice criteria differ: their options were shuffled otherwise.
        assert differing
        assert {name for _, name in differing} <= {CLARITY.name, TONE.name}


class TestThresholds:
    @pytest.mark.asyncio
    async def test_an_escalation_point_replays_at_its_thresholds(self, cascade_runs):
        runs = await cascade_runs()
        point = EscalationPoint(
            threshold=THRESHOLD,
            per_criterion=PER_CRITERION,
            escalation_rate=None,
            dm_accuracy_kept=None,
            dm_accuracy_escalated=None,
            fallback_accuracy_escalated=None,
            metric=None,
            cost_usd=None,
            compute_seconds=0.0,
        )
        assert_replays(replay_escalation(runs.dm, runs.llm, point), runs.live)

    @pytest.mark.asyncio
    async def test_threshold_zero_escalates_only_errors_and_abstentions(self, cascade_runs):
        runs = await cascade_runs()
        replay = replay_escalation(runs.dm, runs.llm, 0.0)
        assert escalated_flags(replay) == [
            [False] * 5,
            [False] * 5,
            [False, False, False, True, False],  # tone abstained
            [False, True, False, False, False],  # myth unanswered
            [True] * 5,  # the request failed
            [False] * 5,
            [False] * 5,
        ]

    @pytest.mark.asyncio
    async def test_threshold_one_escalates_everything_short_of_certainty(self, cascade_runs):
        runs = await cascade_runs()
        replay = replay_escalation(runs.dm, runs.llm, 1.0)
        assert escalated_flags(replay) == [[True] * 5] * len(ITEMS)

    @pytest.mark.asyncio
    async def test_unnamed_criteria_use_the_global_threshold(self, cascade_runs):
        runs = await cascade_runs()
        # The unnamed criterion (sentences, 0.9 confident on item 0) cannot be named.
        replay = replay_escalation(runs.dm, runs.llm, 0.95, per_criterion={"light": 0.0})
        light, _, _, _, sentences = replay.item_results[0].report.report
        assert not light.escalated and sentences.escalated


def point_at(threshold: float, per_criterion: dict[str, float] | None) -> EscalationPoint:
    return EscalationPoint(
        threshold=threshold,
        per_criterion=per_criterion,
        escalation_rate=None,
        dm_accuracy_kept=None,
        dm_accuracy_escalated=None,
        fallback_accuracy_escalated=None,
        metric=None,
        cost_usd=None,
        compute_seconds=0.0,
    )


class TestUnknownPerCriterionNames:
    """As a live cascade does, a replay warns about ``per_criterion`` names that match no
    criterion: thresholds are looked up by name, so such a name has no effect."""

    @pytest.mark.asyncio
    async def test_a_name_no_replayed_criterion_has_warns(self, cascade_runs):
        runs = await cascade_runs()
        with pytest.warns(UserWarning) as caught:
            replay = replay_escalation(
                runs.dm,
                runs.llm,
                THRESHOLD,
                per_criterion={**PER_CRITERION, "factuality": 0.9, "style": 0.1},
            )
        (warning,) = caught
        assert str(warning.message) == (
            "per_criterion names 'factuality', 'style' match no criterion of the replayed "
            "items; thresholds are looked up by criterion name, so these have no effect"
        )
        assert warning.filename == __file__
        assert_replays(replay, runs.live)  # the names changed nothing

    @pytest.mark.asyncio
    async def test_an_escalation_points_names_are_checked(self, cascade_runs):
        runs = await cascade_runs()
        with pytest.warns(UserWarning, match=r"^EscalationPoint.per_criterion names 'nope' "):
            replay_escalation(runs.dm, runs.llm, point_at(THRESHOLD, {"nope": 0.5}))

    @pytest.mark.asyncio
    async def test_a_name_any_replayed_item_has_does_not_warn(self, cascade_runs, decision_model):
        """With a rubric per item, a name can be absent from some items' rubrics."""
        data = per_item_dataset()
        decision_model.scripts = {**decision_model.scripts, **per_item_decision_model_scripts()}
        runs = await cascade_runs(data=data, per_criterion=None, llm_script=PER_ITEM_LLM_SCRIPT)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            replay_escalation(runs.dm, runs.llm, THRESHOLD, per_criterion={"tone": 0.1})
            replay_escalation(runs.dm, runs.llm, point_at(THRESHOLD, {"clarity": 0.9}))
        with pytest.warns(UserWarning, match=r"names 'factuality' match no criterion"):
            replay_escalation(runs.dm, runs.llm, THRESHOLD, per_criterion={"factuality": 0.1})


# =============================================================================
# Estimated cost and time
# =============================================================================


def with_times(result: EvalResult, item_seconds: float, total_seconds: float) -> EvalResult:
    items = [dataclasses.replace(r, duration_seconds=item_seconds) for r in result.item_results]
    timing = EvalTimingStats.from_durations([item_seconds] * len(items), total_seconds)
    return dataclasses.replace(result, item_results=items, timing_stats=timing)


class TestEstimatedCostAndTime:
    @pytest.mark.asyncio
    async def test_llm_cost_and_time_are_pro_rated_by_the_escalated_share(self, cascade_runs):
        runs = await cascade_runs()
        dm_run = with_times(runs.dm, 0.5, 2.0)
        llm_run = with_times(runs.llm, 4.0, 10.0)

        replay = replay_escalation(dm_run, llm_run, THRESHOLD, per_criterion=PER_CRITERION)

        dm_cost = DM_INPUT_TOKENS * DM_PRICE
        for i, (r, flags) in enumerate(zip(replay.item_results, ESCALATED, strict=True)):
            share = sum(flags) / len(flags)
            llm_cost = llm_run.item_results[i].report.completion_cost
            # The failed request has no cost; every other item's is the decision model's.
            dm_part = 0.0 if ITEMS[i].dm_answers is None else dm_cost
            assert r.report.completion_cost == pytest.approx(dm_part + llm_cost * share)
            assert r.duration_seconds == pytest.approx(0.5 + 4.0 * share)
        # Five LLM calls per item, but the terse item's two failed calls cost nothing.
        assert [r.report.completion_cost for r in llm_run.item_results] == pytest.approx(
            [(3 if item.submission == TERSE else 5) * LLM_COST for item in ITEMS]
        )

        n_escalated = sum(map(sum, ESCALATED))
        assert replay.total_completion_cost == pytest.approx(
            sum(r.report.completion_cost for r in replay.item_results)
        )
        assert replay.total_token_usage is None
        assert replay.timing_stats.total_duration_seconds == pytest.approx(
            2.0 + 10.0 * n_escalated / (5 * len(ITEMS))
        )
        assert replay.timing_stats.mean_item_duration_seconds == pytest.approx(
            sum(r.duration_seconds for r in replay.item_results) / len(ITEMS)
        )

    @pytest.mark.asyncio
    async def test_the_runs_time_is_pro_rated_by_its_pooled_escalated_share(
        self, cascade_runs, decision_model
    ):
        """The run's share is its escalated criteria over its criteria, not the mean of its
        items' shares: they differ when items have different numbers of criteria."""
        data = per_item_dataset()
        decision_model.scripts = {**decision_model.scripts, **per_item_decision_model_scripts()}
        runs = await cascade_runs(data=data, per_criterion=None, llm_script=PER_ITEM_LLM_SCRIPT)
        dm_run = with_times(runs.dm, 0.5, 2.0)
        llm_run = with_times(runs.llm, 4.0, 10.0)

        replay = replay_escalation(dm_run, llm_run, THRESHOLD)

        # 1 of 2, 2 of 3 and 0 of 2 criteria escalated: 3 of 7, where the items' shares
        # average 7/18.
        shares = [1 / 2, 2 / 3, 0.0]
        assert escalated_flags(replay) == [[True, False], [False, True, True], [False, False]]
        assert [r.duration_seconds for r in replay.item_results] == pytest.approx(
            [0.5 + 4.0 * share for share in shares]
        )
        assert replay.timing_stats.total_duration_seconds == pytest.approx(2.0 + 10.0 * 3 / 7)

    @pytest.mark.asyncio
    async def test_an_item_with_no_known_cost_has_none(self, cascade_runs):
        runs = await cascade_runs(dm_config=dm(input_cost_per_token=None))
        # Nothing escalates on the sure item, and the decision model is unpriced: like the
        # live cascade, the item has no cost, not a fabricated 0.0.
        replay = replay_escalation(runs.dm, runs.llm, 0.0)
        assert replay.item_results[0].report.completion_cost is None
        assert runs.live.item_results[0].report.completion_cost is not None  # tone escalated

    @pytest.mark.asyncio
    async def test_the_result_is_marked_as_carrying_estimates(self, cascade_runs):
        runs = await cascade_runs()
        replay = replay_at_the_cascade(runs)
        assert replay.experiment_name == (
            f"replay of decision-model run {runs.dm.experiment_name!r} and LLM run "
            f"{runs.llm.experiment_name!r} (estimated cost and time)"
        )
        assert replay.experiment_dir is None
        n = len(ITEMS)
        assert (replay.total_items, replay.successful_items, replay.failed_items) == (n, n, 0)
        assert replay.errors == []

        unnamed = replay_escalation(
            dataclasses.replace(runs.dm, experiment_name=None),
            dataclasses.replace(runs.llm, experiment_name=None),
            THRESHOLD,
        )
        assert unnamed.experiment_name == (
            "replay of an unnamed decision-model run and an unnamed LLM run "
            "(estimated cost and time)"
        )

    @pytest.mark.asyncio
    async def test_an_item_the_decision_model_run_failed_stays_failed(self, cascade_runs):
        runs = await cascade_runs()
        failed = ItemResult(
            item_idx=0,
            item=runs.dm.item_results[0].item,
            report=EvaluationReport(score=None, error="grading crashed"),
            duration_seconds=0.3,
            error="grading crashed",
        )
        dm_run = dataclasses.replace(runs.dm, item_results=[failed, *runs.dm.item_results[1:]])

        replay = replay_escalation(dm_run, runs.llm, THRESHOLD, per_criterion=PER_CRITERION)

        first = replay.item_results[0]
        assert (first.error, first.report, first.duration_seconds) == (
            "grading crashed",
            failed.report,
            0.3,
        )
        assert (replay.successful_items, replay.failed_items) == (len(ITEMS) - 1, 1)
        assert replay.errors == [(0, "grading crashed")]
        assert_replays(
            dataclasses.replace(replay, item_results=replay.item_results[1:]),
            dataclasses.replace(runs.live, item_results=runs.live.item_results[1:]),
        )
        compute_metrics(replay, runs.data)  # an ordinary result: the failed item is skipped

    @pytest.mark.asyncio
    async def test_the_llm_time_of_an_item_the_decision_model_run_failed_is_not_spent(
        self, cascade_runs
    ):
        """A live cascade sends an item whose decision-model grading failed to no LLM judge.
        The LLM run judged its criteria all the same, so they count among the run's pairs,
        none escalated: the run's LLM time is pro-rated over every pair the LLM run judged,
        as the items' estimates are."""
        runs = await cascade_runs()
        dm_run = with_times(runs.dm, 0.5, 2.0)
        failed = dataclasses.replace(
            dm_run.item_results[0],
            report=EvaluationReport(score=None, error="grading crashed"),
            error="grading crashed",
        )
        dm_run = dataclasses.replace(dm_run, item_results=[failed, *dm_run.item_results[1:]])
        llm_run = with_times(runs.llm, 4.0, 10.0)

        replay = replay_escalation(dm_run, llm_run, THRESHOLD, per_criterion=PER_CRITERION)

        assert replay.item_results[0].duration_seconds == 0.5  # no LLM part
        # 14 of the LLM run's 35 pairs escalate: the failed item's criteria escalate none.
        n_escalated = sum(map(sum, ESCALATED[1:]))
        assert n_escalated == 14
        assert replay.timing_stats.total_duration_seconds == pytest.approx(
            2.0 + 10.0 * n_escalated / (5 * len(ITEMS))
        )
        # The same share of the LLM run's time as the items' LLM parts are of its items' time.
        llm_parts = sum(r.duration_seconds for r in replay.item_results) - 0.5 * len(ITEMS)
        assert (replay.timing_stats.total_duration_seconds - 2.0) / 10.0 == pytest.approx(
            llm_parts / (4.0 * len(ITEMS))
        )


# =============================================================================
# Requirements
# =============================================================================


def replace_item(result: EvalResult, idx: int, **changes: Any) -> EvalResult:
    items = [
        dataclasses.replace(r, **changes) if r.item_idx == idx else r for r in result.item_results
    ]
    return dataclasses.replace(result, item_results=items)


def replace_criterion_report(result: EvalResult, idx: int, c: int, **changes: Any) -> EvalResult:
    report = result.item_results[idx].report
    reports = list(report.report)
    reports[c] = reports[c].model_copy(update=changes)
    return replace_item(result, idx, report=report.model_copy(update={"report": reports}))


class TestRequirements:
    @pytest.mark.asyncio
    async def test_the_results_must_cover_the_same_items(self, cascade_runs):
        runs = await cascade_runs()
        fewer = dataclasses.replace(runs.llm, item_results=runs.llm.item_results[1:])
        with pytest.raises(ValueError, match=r"same items.*only in dm_result: \[0\]"):
            replay_escalation(runs.dm, fewer, THRESHOLD)

    @pytest.mark.asyncio
    async def test_an_item_index_may_appear_once(self, cascade_runs):
        runs = await cascade_runs()
        twice = dataclasses.replace(
            runs.dm, item_results=[*runs.dm.item_results, runs.dm.item_results[2]]
        )
        with pytest.raises(ValueError, match=r"dm_result holds item 2 more than once"):
            replay_escalation(twice, runs.llm, THRESHOLD)

    @pytest.mark.asyncio
    async def test_the_same_index_must_be_the_same_submission(self, cascade_runs):
        runs = await cascade_runs()
        other = dataclasses.replace(runs.dm.item_results[1].item, submission="Another answer.")
        with pytest.raises(ValueError, match=r"item 1 is not the same item"):
            replay_escalation(replace_item(runs.dm, 1, item=other), runs.llm, THRESHOLD)

    @pytest.mark.asyncio
    async def test_the_rubrics_must_be_the_same(self, cascade_runs, make_grader, tmp_path):
        runs = await cascade_runs()
        # Without the guaranteed NA option, the multi-choice criteria are graded differently.
        forced = await evaluate(
            runs.data,
            make_grader(judge_model_config=dm(), auto_na_option=False),
            show_progress=False,
            experiments_dir=tmp_path,
            experiment_name="forced-choice",
        )
        with pytest.raises(ValueError, match=r"item 0 was graded against different criteria"):
            replay_escalation(forced, runs.llm, THRESHOLD)

    @pytest.mark.asyncio
    async def test_dm_result_must_come_from_a_decision_model(self, cascade_runs):
        runs = await cascade_runs()
        with pytest.raises(ValueError, match=r"dm_result must come from one decision-model"):
            replay_escalation(runs.llm, runs.llm, THRESHOLD)

    @pytest.mark.asyncio
    async def test_dm_result_cannot_be_a_cascade(self, cascade_runs):
        runs = await cascade_runs()
        with pytest.raises(ValueError, match=r"dm_result must come from one decision-model"):
            replay_escalation(runs.live, runs.llm, THRESHOLD)

    @pytest.mark.asyncio
    async def test_dm_result_must_come_from_one_judge(self, cascade_runs, make_grader, tmp_path):
        runs = await cascade_runs()
        pair = await evaluate(
            runs.data,
            make_grader(judges=[JudgeSpec(dm(), "jev"), JudgeSpec(dm(), "jev2")]),
            show_progress=False,
            experiments_dir=tmp_path,
            experiment_name="two-decision-models",
        )
        with pytest.raises(ValueError, match=r"dm_result must come from one decision-model"):
            replay_escalation(pair, runs.llm, THRESHOLD)

    @pytest.mark.asyncio
    async def test_dm_result_needs_the_same_judge_on_every_item(self, cascade_runs):
        """A cascade has one decision-model judge, so the decision-model run must have had
        one (not, say, a judge renamed when the run was resumed)."""
        runs = await cascade_runs()
        report = runs.dm.item_results[4].report

        def renamed(votes: list[Any]) -> list[Any]:
            return [v.model_copy(update={"judge_id": "x"}) for v in votes]

        judge_x = report.model_copy(
            update={
                "judge_scores": {"x": report.judge_scores["default"]},
                "report": [
                    cr.model_copy(
                        update={
                            "votes": renamed(cr.votes),
                            "multi_choice_votes": renamed(cr.multi_choice_votes),
                        }
                    )
                    for cr in report.report
                ],
            }
        )
        changed = replace_item(runs.dm, 4, report=judge_x)
        with pytest.raises(
            ValueError, match=r"item 4 was judged by 'x', earlier items by 'default'"
        ):
            replay_escalation(changed, runs.llm, THRESHOLD)

    @pytest.mark.asyncio
    async def test_llm_result_must_come_from_llm_judges(self, cascade_runs):
        runs = await cascade_runs()
        with pytest.raises(ValueError, match=r"llm_result must come from LLM judges"):
            replay_escalation(runs.dm, runs.dm, THRESHOLD)
        with pytest.raises(ValueError, match=r"llm_result must come from LLM judges"):
            replay_escalation(runs.dm, runs.live, THRESHOLD)

    @pytest.mark.asyncio
    async def test_llm_result_needs_every_judges_vote_on_every_criterion(self, cascade_runs):
        runs = await cascade_runs(PANEL)
        votes = runs.llm.item_results[2].report.report[0].votes
        missing = replace_criterion_report(runs.llm, 2, 0, votes=votes[:1])
        with pytest.raises(ValueError, match=r"item 2, criterion 0 has votes from \['a'\]"):
            replay_escalation(runs.dm, missing, THRESHOLD)

    @pytest.mark.asyncio
    async def test_llm_result_needs_the_same_judges_on_every_item(self, cascade_runs):
        """A cascade has one set of escalation judges, so the LLM run must have had one
        (not, say, judges that changed when the run was resumed)."""
        runs = await cascade_runs(PANEL)
        report = runs.llm.item_results[4].report
        judge_a_alone = report.model_copy(
            update={
                "judge_scores": {"a": report.judge_scores["a"]},
                "report": [
                    cr.model_copy(
                        update={
                            "votes": cr.votes[:1],
                            "multi_choice_votes": cr.multi_choice_votes[:1],
                        }
                    )
                    for cr in report.report
                ],
            }
        )
        changed = replace_item(runs.llm, 4, report=judge_a_alone)
        with pytest.raises(
            ValueError, match=r"item 4 was graded by \['a'\], earlier items by \['a', 'b'\]"
        ):
            replay_escalation(runs.dm, changed, THRESHOLD)

    @pytest.mark.asyncio
    async def test_llm_result_needs_every_item(self, cascade_runs):
        runs = await cascade_runs()
        failed = replace_item(
            runs.llm, 3, error="grading crashed", report=EvaluationReport(score=None)
        )
        with pytest.raises(ValueError, match=r"llm_result has no votes on item 3"):
            replay_escalation(runs.dm, failed, THRESHOLD)

    @pytest.mark.asyncio
    async def test_judge_ids_must_be_unique(self, cascade_runs, make_grader, tmp_path):
        runs = await cascade_runs()
        default = await evaluate(
            runs.data,
            make_grader(llm_script=llm_scripts(), judge_model_config=LLMConfig(model="gemini")),
            show_progress=False,
            experiments_dir=tmp_path,
            experiment_name="llm-named-default",
        )
        with pytest.raises(ValueError, match=r"judge_ids must be unique.*'default'"):
            replay_escalation(runs.dm, default, THRESHOLD)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), True, "0.5", None])
    async def test_the_threshold_is_a_number_in_the_unit_interval(self, cascade_runs, threshold):
        runs = await cascade_runs()
        with pytest.raises(ValueError, match=r"escalation must be a number in \[0, 1\]"):
            replay_escalation(runs.dm, runs.llm, threshold)

    @pytest.mark.asyncio
    async def test_per_criterion_thresholds_are_validated(self, cascade_runs):
        runs = await cascade_runs()
        with pytest.raises(ValueError, match=r"per_criterion\['tone'\] must be a number"):
            replay_escalation(runs.dm, runs.llm, THRESHOLD, per_criterion={"tone": 2})
        with pytest.raises(ValueError, match=r"per_criterion keys are criterion names"):
            replay_escalation(runs.dm, runs.llm, THRESHOLD, per_criterion={0: 0.5})

    @pytest.mark.asyncio
    async def test_per_criterion_goes_only_with_a_float_threshold(self, cascade_runs):
        runs = await cascade_runs()
        point = EscalationPoint(
            threshold=THRESHOLD,
            escalation_rate=None,
            dm_accuracy_kept=None,
            dm_accuracy_escalated=None,
            fallback_accuracy_escalated=None,
            metric=None,
            cost_usd=None,
            compute_seconds=0.0,
        )
        with pytest.raises(ValueError, match=r"per_criterion goes with a float threshold"):
            replay_escalation(runs.dm, runs.llm, point, per_criterion={"tone": 0.9})

    @pytest.mark.asyncio
    async def test_an_escalation_point_needs_a_threshold(self, cascade_runs):
        runs = await cascade_runs()
        point = EscalationPoint(
            escalation_rate=0.5,
            dm_accuracy_kept=None,
            dm_accuracy_escalated=None,
            fallback_accuracy_escalated=None,
            metric=None,
            cost_usd=None,
            compute_seconds=0.0,
        )
        with pytest.raises(ValueError, match=r"EscalationPoint has no threshold"):
            replay_escalation(runs.dm, runs.llm, point)


# =============================================================================
# The calibration fingerprint and the same-data warning
# =============================================================================


def fingerprint(data: RubricDataset) -> str:
    return _calibration_fingerprint(
        (data.items[i], data.get_item_rubric(i).rubric) for i in range(len(data))
    )


def calibrated_point(fingerprint: str | None) -> EscalationPoint:
    return EscalationPoint(
        threshold=THRESHOLD,
        per_criterion=PER_CRITERION,
        escalation_rate=0.4,
        dm_accuracy_kept=0.9,
        dm_accuracy_escalated=0.5,
        fallback_accuracy_escalated=0.8,
        metric=0.85,
        cost_usd=0.02,
        compute_seconds=3.0,
        calibration_fingerprint=fingerprint,
    )


class TestCalibrationFingerprint:
    def test_it_is_a_sha256_over_the_items_contents_in_order(self):
        data = dataset()
        digest = fingerprint(data)
        assert len(digest) == 64 and int(digest, 16) >= 0
        reordered = dataset(list(reversed(ITEMS)))
        assert fingerprint(reordered) != digest
        renamed = RubricDataset(prompt="Another prompt.", rubric=RUBRIC, name="other name")
        for item in data.items:
            renamed.items.append(item)
        # The dataset's name and prompt are not the items' contents.
        assert fingerprint(renamed) == digest

    def test_it_changes_with_every_part_of_an_item(self):
        base = RubricDataset(rubric=RUBRIC)
        base.add_item("An answer.", "d", prompt="Q?", reference_submission="Ref.")
        digest = fingerprint(base)
        for change in (
            {"submission": "Another answer."},
            {"prompt": "Another question?"},
            {"reference_submission": "Another reference."},
        ):
            changed = RubricDataset(rubric=RUBRIC)
            changed.items.append(dataclasses.replace(base.items[0], **change))
            assert fingerprint(changed) != digest, change
        reworded = RubricDataset(rubric=RUBRIC)
        criteria = [
            Criterion(**{**dict(c), "requirement": c.requirement + "!"}) for c in RUBRIC.rubric
        ]
        reworded.add_item("An answer.", "d", prompt="Q?", reference_submission="Ref.")
        reworded.items[0].rubric = autorubric.Rubric(criteria)
        assert fingerprint(reworded) != digest

    def test_equal_halves_of_a_split_differ_where_the_dataset_hash_collides(self):
        data = dataset((ITEMS * 2)[:12])
        first, second = data.split_train_test(n_train=6, seed=1)
        assert len(first) == len(second) == 6
        assert _compute_dataset_hash(first) == _compute_dataset_hash(second)
        assert fingerprint(first) != fingerprint(second)

    @pytest.mark.asyncio
    async def test_replaying_the_calibration_items_warns(self, cascade_runs):
        runs = await cascade_runs()
        point = calibrated_point(fingerprint(runs.data))
        with pytest.warns(UserWarning, match=r"calibrated on the items it replays") as caught:
            replay = replay_escalation(runs.dm, runs.llm, point)
        assert caught[0].filename == __file__
        assert_replays(replay, runs.live)

    @pytest.mark.asyncio
    async def test_replaying_other_items_does_not_warn(self, cascade_runs):
        runs = await cascade_runs()
        other = dataset(ITEMS[:3])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            replay_escalation(runs.dm, runs.llm, calibrated_point(fingerprint(other)))
            replay_escalation(runs.dm, runs.llm, calibrated_point(None))
            replay_escalation(runs.dm, runs.llm, THRESHOLD)


# =============================================================================
# Public surface
# =============================================================================


def test_exported_from_autorubric():
    for name in ("EscalationPoint", "escalation_stats", "replay_escalation"):
        assert name in autorubric.__all__
        assert getattr(autorubric, name) is getattr(autorubric.escalation, name)


def test_an_escalation_point_is_frozen_with_the_documented_fields():
    assert list(EscalationPoint.model_fields) == [
        "threshold",
        "per_criterion",
        "escalation_rate",
        "dm_accuracy_kept",
        "dm_accuracy_escalated",
        "fallback_accuracy_escalated",
        "metric",
        "cost_usd",
        "compute_seconds",
        "calibration_fingerprint",
    ]
    point = calibrated_point("abc")
    with pytest.raises(Exception, match="frozen"):
        point.threshold = 0.1
