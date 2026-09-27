"""A confidence cascade whose escalation judges grade a whole rubric in one call per item.

Under ``llm_calls="per_item"`` an escalation judge makes no call for an item on which
nothing escalated; otherwise it makes the very call a grader without the cascade makes (with
the same ``seed``, ``judge_id`` and ``llm_calls``): every criterion, in rubric order, with the
same option permutations. It keeps the results of the escalated criteria only, the first of
them carrying the call's usage and cost. A ``per_item`` LLM run therefore holds exactly the
verdicts the live cascade's calls produce, and ``replay_escalation(..., llm_calls="per_item")``
rebuilds the live cascade from it, cost included (up to floating-point rounding): an item's
LLM cost and time are the LLM run's in full when anything escalated, and nothing otherwise.

The runs come from ``cascade_runs`` with ``llm_calls="per_item"`` for the cascade and the LLM
run. ``ScriptedLLM`` answers each criterion of a per-item call as a call about that criterion
alone would be, so scripted verdicts, hashed ones and failures carry over. Nothing reaches
the network.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, cast

import litellm
import pytest

from autorubric import (
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EscalationPoint,
    EvalResult,
    EvaluationReport,
    ItemResult,
    LLMCalls,
    Rubric,
    RubricDataset,
    calibrate_escalation,
    compute_metrics,
    escalation_stats,
    replay_escalation,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from escalation.cascade_runs import (
    CLARITY,
    DM_INPUT_TOKENS,
    DM_PRICE,
    GEMINI,
    ITEMS,
    LIGHT,
    LLM_COST,
    LLM_USAGE,
    MYTH,
    NO_JUDGMENT,
    PANEL,
    PER_CRITERION,
    PER_ITEM_LLM_SCRIPT,
    QUERY,
    RUBRIC,
    SEED,
    TERSE,
    THRESHOLD,
    UNUSABLE_JUDGMENT,
    Runs,
    ScriptedLLM,
    dm,
    llm_scripts,
    per_item_dataset,
    per_item_decision_model_scripts,
)
from escalation.test_replay_escalation import (
    ESCALATED,
    ESTIMATED,
    assert_replays,
    assert_sent_in_the_llm_run,
    escalated_flags,
    with_times,
)

PER_ITEM: LLMCalls = "per_item"
N_CRITERIA = len(RUBRIC.rubric)

# Thresholds to run a cascade at: the replay tests' (tone escalated on every item), the
# global threshold alone (nothing escalated on the sure items 0 and 6), failures and
# abstentions only, and everything short of certainty.
CASES = [(THRESHOLD, PER_CRITERION), (THRESHOLD, None), (0.0, None), (1.0, None)]
CASE_IDS = ["per-criterion", "global", "zero", "one"]

# Which criteria escalate, per item, at THRESHOLD without per-criterion thresholds.
ESCALATED_GLOBALLY = [
    [False] * 5,  # sure
    [True, False, True, False, False],  # light 0.2, clarity 0.2
    [False, False, False, True, False],  # tone abstains
    [False, True, False, False, False],  # myth unanswered
    [True] * 5,  # the request failed
    [False, True, False, True, True],  # myth 0.1, tone 1/3, sentences 0.4
    [False] * 5,  # light 0.5, at the threshold, kept
]

# A failure of each category, as ``classify_grading_error`` routes it.
FAILURES = {
    "infrastructure": lambda: litellm.Timeout("timed out", model="m", llm_provider="p"),
    "parse": lambda: ValueError("bad json"),
    "unknown": lambda: RuntimeError("boom"),
}

LLM_CALLS_MESSAGE = "llm_calls must be one of ('per_criterion', 'per_item'); got {!r}"

GUIDELINES = "Judge at a grade-8 level."
REFERENCE = "Light turns water and carbon dioxide into sugar."


def any_escalated(flags: list[list[bool]]) -> list[bool]:
    return [any(item) for item in flags]


def scripted(grader: CriterionGrader) -> dict[str, ScriptedLLM]:
    """``grader``'s LLM clients by judge id, the ``ScriptedLLM``s ``make_grader`` built."""
    return {judge_id: cast(ScriptedLLM, client) for judge_id, client in grader._clients.items()}


def rubric_calls(grader: CriterionGrader) -> list[tuple[str, str]]:
    """Every ``(system_prompt, user_prompt)`` ``grader``'s LLM judges were sent."""
    return [prompt for client in scripted(grader).values() for prompt in client.prompts]


def calls_by_judge(grader: CriterionGrader) -> dict[str, int]:
    return {judge_id: len(client.prompts) for judge_id, client in scripted(grader).items()}


def criteria(r: ItemResult) -> list[EnsembleCriterionReport]:
    """An item's criterion reports, as a cascade, or a replay of one, reports them."""
    assert isinstance(r.report, EnsembleEvaluationReport) and r.report.report is not None
    return r.report.report


def votes(cr: Any) -> list[Any]:
    return cr.multi_choice_votes if cr.criterion.is_multi_choice else cr.votes


def escalation_votes(cr: Any) -> list[Any]:
    """The votes of the escalation judges on a criterion (none unless it escalated)."""
    return [vote for vote in votes(cr) if vote.judge_id != "default"]


def dataset_with(*, guidelines: str | None, reference: str | None) -> RubricDataset:
    """The items of ``cascade_runs.dataset()``, graded against ``RUBRIC`` with
    ``guidelines``, each with ``reference`` as its reference submission."""
    data = RubricDataset(prompt=QUERY, rubric=Rubric(RUBRIC.rubric, guidelines=guidelines))
    for i, item in enumerate(ITEMS):
        data.add_item(
            item.submission,
            f"item {i}",
            ground_truth=list(item.ground_truth),
            reference_submission=reference,
        )
    return data


# =============================================================================
# The scripted judge answers a per-item call as it answers each criterion alone
# =============================================================================


class TestTheScriptedJudge:
    @pytest.mark.asyncio
    async def test_a_per_item_call_gives_the_per_criterion_calls_judgments(self, make_grader):
        """The two modes differ only in how the calls are made, so every criterion's report
        is the same: verdicts, reasons and option orders (hashed from each rebuilt
        per-criterion prompt). Only where the usage rides differs."""
        script = {(ITEMS[1].submission, CLARITY.requirement): "Unclear"}
        per_criterion = make_grader(llm_script=script, judges=PANEL, seed=SEED)
        per_item = make_grader(llm_script=script, judges=PANEL, seed=SEED, llm_calls=PER_ITEM)

        for item in ITEMS:
            expected = await per_criterion.judge(item.submission, RUBRIC.rubric, QUERY)
            before = calls_by_judge(per_item)
            graded = await per_item.judge(item.submission, RUBRIC.rubric, QUERY)
            assert calls_by_judge(per_item) == {j: n + 1 for j, n in before.items()}
            for judge, judged in zip(expected, graded, strict=True):
                assert judged.reports == judge.reports
                assert judged.total_usage == LLM_USAGE and judged.total_cost == LLM_COST
        assert all(p.count("<rubric_criterion ") == N_CRITERIA for _, p in rubric_calls(per_item))


# =============================================================================
# The replay equals the live cascade, cost included
# =============================================================================


class TestReplayEqualsTheLiveCascade:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("judges", [GEMINI, PANEL], ids=["one-judge", "panel"])
    @pytest.mark.parametrize(("threshold", "per_criterion"), CASES, ids=CASE_IDS)
    async def test_at_every_threshold(self, cascade_runs, judges, threshold, per_criterion):
        runs = await cascade_runs(
            judges, threshold=threshold, per_criterion=per_criterion, llm_calls=PER_ITEM
        )
        replay = replay_escalation(
            runs.dm, runs.llm, threshold, per_criterion=per_criterion, llm_calls=PER_ITEM
        )

        assert_replays(replay, runs.live)
        # A per-item call is billed whole, so the cost estimate is the live cost itself, up
        # to rounding: a panel's costs are summed in another order than the live report's.
        assert [r.report.completion_cost for r in replay.item_results] == pytest.approx(
            [r.report.completion_cost for r in runs.live.item_results]
        )
        assert replay.total_completion_cost == pytest.approx(runs.live.total_completion_cost)
        assert compute_metrics(replay, runs.data, per_judge=True) == compute_metrics(
            runs.live, runs.data, per_judge=True
        )
        # Every call the escalation judges made, system prompt included, the LLM run made.
        assert_sent_in_the_llm_run(runs)
        assert Counter(rubric_calls(runs.cascade)) <= Counter(rubric_calls(runs.llm_grader))

    @pytest.mark.asyncio
    async def test_the_escalated_criteria_are_the_per_criterion_cascades(self, cascade_runs):
        runs = await cascade_runs(llm_calls=PER_ITEM)
        assert escalated_flags(runs.live) == ESCALATED
        replay = replay_escalation(runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION)
        assert escalated_flags(replay) == ESCALATED

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "settings",
        [
            {"aggregation": "weighted", "ordinal_aggregation": "weighted_mean"},
            {
                "aggregation": "unanimous",
                "ordinal_aggregation": "min",
                "nominal_aggregation": "unanimous",
            },
        ],
        ids=["weighted", "unanimous"],
    )
    async def test_a_panel_aggregated_as_the_llm_run_aggregated_it(self, cascade_runs, settings):
        judges = [JudgeSpec(PANEL[0].llm_config, "a", 2.0), PANEL[1]]
        runs = await cascade_runs(judges, **settings, llm_calls=PER_ITEM)
        replay = replay_escalation(
            runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION, llm_calls=PER_ITEM
        )
        assert_replays(replay, runs.live)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
    async def test_multi_choice_criteria_with_the_calls_option_order(self, cascade_runs, shuffle):
        runs = await cascade_runs(PANEL, shuffle_options=shuffle, llm_calls=PER_ITEM)
        replay = replay_escalation(
            runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION, llm_calls=PER_ITEM
        )
        assert_replays(replay, runs.live)

        orders = [
            vote.shuffle_order
            for r in replay.item_results
            for cr in criteria(r)
            if cr.escalated and cr.criterion.is_multi_choice
            for vote in escalation_votes(cr)
        ]
        assert orders
        if shuffle:
            assert all(order is not None for order in orders)
            assert any(order != sorted(order) for order in orders if order is not None)
        else:
            assert orders == [None] * len(orders)

    @pytest.mark.asyncio
    async def test_an_escalation_point_with_per_criterion_thresholds(self, cascade_runs):
        per_criterion = {"myth": 0.0, "clarity": 0.1}
        runs = await cascade_runs(PANEL, per_criterion=per_criterion, llm_calls=PER_ITEM)
        point = EscalationPoint(
            threshold=THRESHOLD,
            per_criterion=per_criterion,
            escalation_rate=None,
            dm_accuracy_kept=None,
            dm_accuracy_escalated=None,
            fallback_accuracy_escalated=None,
            metric=None,
            cost_usd=None,
            compute_seconds=0.0,
        )
        replay = replay_escalation(runs.dm, runs.llm, point, llm_calls=PER_ITEM)
        assert_replays(replay, runs.live)
        # Clarity (0.2 on item 1) clears its own threshold; myth escalates on failures only.
        clarity = [criteria(r)[2].escalated for r in replay.item_results]
        assert clarity == [False, False, False, False, True, False, False]

    @pytest.mark.asyncio
    async def test_a_dataset_with_a_rubric_per_item(self, cascade_runs, decision_model):
        data = per_item_dataset()
        decision_model.scripts = {**decision_model.scripts, **per_item_decision_model_scripts()}
        runs = await cascade_runs(
            PANEL, data=data, per_criterion=None, llm_script=PER_ITEM_LLM_SCRIPT, llm_calls=PER_ITEM
        )
        replay = replay_escalation(runs.dm, runs.llm, THRESHOLD, llm_calls=PER_ITEM)

        assert_replays(replay, runs.live)
        assert escalated_flags(replay) == [[True, False], [False, True, True], [False, False]]
        assert [r.report.completion_cost for r in replay.item_results] == pytest.approx(
            [r.report.completion_cost for r in runs.live.item_results]
        )
        # Each item's call lists that item's own rubric.
        sizes = sorted(p.count("<rubric_criterion ") for _, p in rubric_calls(runs.cascade))
        assert sizes == [2, 2, 3, 3]  # two judges, on the two items with an escalation
        assert compute_metrics(replay, data) == compute_metrics(runs.live, data)

    @pytest.mark.asyncio
    async def test_replaying_saved_experiments(self, cascade_runs):
        runs = await cascade_runs(PANEL, llm_calls=PER_ITEM)
        dm_run = EvalResult.from_experiment(runs.dm.experiment_dir)
        llm_run = EvalResult.from_experiment(runs.llm.experiment_dir)

        replay = replay_escalation(
            dm_run, llm_run, THRESHOLD, per_criterion=PER_CRITERION, llm_calls=PER_ITEM
        )

        assert_replays(replay, runs.live)
        assert [r.report.completion_cost for r in replay.item_results] == pytest.approx(
            [r.report.completion_cost for r in runs.live.item_results]
        )


# =============================================================================
# The escalation judges' calls
# =============================================================================


class TestCalls:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("judges", [GEMINI, PANEL], ids=["one-judge", "panel"])
    async def test_one_call_per_escalation_judge_per_item_with_an_escalation(
        self, cascade_runs, judges
    ):
        runs = await cascade_runs(judges, per_criterion=None, llm_calls=PER_ITEM)
        assert escalated_flags(runs.live) == ESCALATED_GLOBALLY
        escalating = sum(any_escalated(ESCALATED_GLOBALLY))
        assert 0 < escalating < len(ITEMS)

        # One call per judge for each item with an escalation, none for the sure items;
        # the LLM run made one per judge for every item.
        assert calls_by_judge(runs.cascade) == dict.fromkeys(runs.cascade._clients, escalating)
        assert calls_by_judge(runs.llm_grader) == dict.fromkeys(runs.llm_grader._clients, 7)
        # Each is a call about the whole rubric.
        assert all(
            p.count("<rubric_criterion ") == N_CRITERIA for _, p in rubric_calls(runs.cascade)
        )
        sure = {ITEMS[0].submission, ITEMS[6].submission}
        assert not [p for _, p in rubric_calls(runs.cascade) if any(s in p for s in sure)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("judges", [GEMINI, PANEL], ids=["one-judge", "panel"])
    async def test_results_at_the_escalated_criteria_only_the_first_carrying_the_usage(
        self, cascade_runs, judges
    ):
        runs = await cascade_runs(judges, per_criterion=None, llm_script={}, llm_calls=PER_ITEM)
        grader = runs.cascade
        for item, flags in zip(ITEMS, ESCALATED_GLOBALLY, strict=True):
            before = calls_by_judge(grader)
            primary, *escalation = await grader.judge(item.submission, RUBRIC.rubric, QUERY)

            assert primary.role == "primary"
            assert [r is not None for r in primary.criterion_results] == [True] * N_CRITERIA
            assert [jr.judge_id for jr in escalation] == [j.judge_id for j in judges]
            escalated = [c for c, flag in enumerate(flags) if flag]
            made = int(bool(escalated))
            assert calls_by_judge(grader) == {j: n + made for j, n in before.items()}
            for jr in escalation:
                assert jr.role == "escalation"
                assert len(jr.criterion_results) == N_CRITERIA
                assert [r is not None for r in jr.criterion_results] == flags
                billed = [
                    c
                    for c, r in enumerate(jr.criterion_results)
                    if r is not None and (r.usage is not None or r.cost is not None)
                ]
                assert billed == escalated[:1]
                if escalated:
                    first = jr.criterion_results[escalated[0]]
                    assert first is not None
                    assert (first.usage, first.cost) == (LLM_USAGE, LLM_COST)
                    assert (jr.total_usage, jr.total_cost) == (LLM_USAGE, LLM_COST)
                else:
                    assert (jr.total_usage, jr.total_cost) == (None, None)

    @pytest.mark.asyncio
    async def test_the_escalated_results_are_the_non_cascade_calls(self, cascade_runs):
        """An escalation judge's results are those of the LLM run's judge on the same item:
        the same call, of which only the escalated criteria's results are kept."""
        runs = await cascade_runs(PANEL, per_criterion=None, llm_script={}, llm_calls=PER_ITEM)
        for item, flags in zip(ITEMS, ESCALATED_GLOBALLY, strict=True):
            _, *escalation = await runs.cascade.judge(item.submission, RUBRIC.rubric, QUERY)
            alone = await runs.llm_grader.judge(item.submission, RUBRIC.rubric, QUERY)
            for jr, judge in zip(escalation, alone, strict=True):
                expected = [
                    r.report if flag else None
                    for r, flag in zip(judge.criterion_results, flags, strict=True)
                    if r is not None
                ]
                assert [None if r is None else r.report for r in jr.criterion_results] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("guidelines", "reference"),
        [(GUIDELINES, None), (None, REFERENCE), (GUIDELINES, REFERENCE)],
        ids=["guidelines", "reference", "both"],
    )
    async def test_the_call_carries_the_items_context(self, cascade_runs, guidelines, reference):
        """Every part of the LLM run's prompt is in the escalation call: the rubric's
        guidelines first, the query and the item's reference submission."""
        data = dataset_with(guidelines=guidelines, reference=reference)
        runs = await cascade_runs(PANEL, data=data, llm_calls=PER_ITEM)

        calls = rubric_calls(runs.cascade)
        # Tone escalates on every item, so each judge calls on each.
        assert len(calls) == len(PANEL) * len(ITEMS)
        for _, user_prompt in calls:
            assert user_prompt.startswith(f"<guidelines>\n{GUIDELINES}\n") == bool(guidelines)
            assert f"<input>{QUERY}</input>" in user_prompt
            assert (
                f"<reference_submission>\n{REFERENCE}\n</reference_submission>" in user_prompt
            ) == bool(reference)
        assert Counter(calls) <= Counter(rubric_calls(runs.llm_grader))
        assert_replays(
            replay_escalation(
                runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION, llm_calls=PER_ITEM
            ),
            runs.live,
        )

    @pytest.mark.asyncio
    async def test_an_item_pays_for_each_escalation_call_once(self, cascade_runs):
        """The live report's totals: the decision model's request, plus one call per
        escalation judge when anything escalated."""
        runs = await cascade_runs(PANEL, per_criterion=None, llm_script={}, llm_calls=PER_ITEM)
        dm_cost = DM_INPUT_TOKENS * DM_PRICE
        for item, r, flags in zip(ITEMS, runs.live.item_results, ESCALATED_GLOBALLY, strict=True):
            dm_part = 0.0 if item.dm_answers is None else dm_cost
            llm_part = len(PANEL) * LLM_COST if any(flags) else 0.0
            assert r.report.completion_cost == pytest.approx(dm_part + llm_part)


# =============================================================================
# Failures
# =============================================================================

# On item 1, light, clarity and tone escalate at THRESHOLD with PER_CRITERION; myth and
# sentences do not.
ITEM = 1
ESCALATED_ON_ITEM = [0, 2, 3]


class TestFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("category", list(FAILURES))
    async def test_a_failed_call_fails_only_the_escalated_votes(self, cascade_runs, category):
        """The call fails as a whole (here scripted on myth, which did not escalate): the
        escalated criteria's votes carry its error, and the criteria that did not escalate
        keep the decision model's verdict, as if no call had been made."""
        script = {(ITEMS[ITEM].submission, MYTH.requirement): FAILURES[category]()}
        runs = await cascade_runs(PANEL, llm_script=script, llm_calls=PER_ITEM)
        replay = replay_escalation(
            runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION, llm_calls=PER_ITEM
        )
        assert_replays(replay, runs.live)

        # The LLM run's call failed: every criterion of the item carries the error.
        llm_report = runs.llm.item_results[ITEM].report
        assert all(cr.error.startswith(f"{category}:") for cr in llm_report.report)
        assert llm_report.error.startswith("Every criterion's judgment failed")

        live = runs.live.item_results[ITEM].report
        dm_report = runs.dm.item_results[ITEM].report
        for c, (cr, dm_cr) in enumerate(zip(live.report, dm_report.report, strict=True)):
            if c in ESCALATED_ON_ITEM:
                assert cr.escalated
                failed = escalation_votes(cr)
                assert [v.judge_id for v in failed] == ["a", "b"]
                assert all(v.error.startswith(f"{category}:") for v in failed)
                assert cr.error is not None and cr.error.startswith(f"{category}:")
                if cr.criterion.is_multi_choice:
                    assert all(v.shuffle_order is not None for v in failed)
            else:
                assert not cr.escalated and not escalation_votes(cr)
                assert cr == dm_cr
        # A failed call is not billed: the item costs the decision model's request alone.
        assert live.completion_cost == pytest.approx(dm_report.completion_cost)
        assert live.token_usage == dm_report.token_usage

    @pytest.mark.asyncio
    async def test_a_failure_scripted_on_the_terse_item_fails_its_call(self, cascade_runs):
        """Of the default script's two failures on the terse item, the call raises the
        first listed (tone's rate limit), so sentences is an infrastructure failure too."""
        runs = await cascade_runs(llm_script=llm_scripts(), llm_calls=PER_ITEM)
        terse = next(i for i, item in enumerate(ITEMS) if item.submission == TERSE)
        live = runs.live.item_results[terse].report.report
        for c in (3, 4):  # tone and sentences escalate
            assert live[c].escalated
            assert live[c].error.startswith("infrastructure:")
        assert_replays(
            replay_escalation(
                runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION, llm_calls=PER_ITEM
            ),
            runs.live,
        )

    @pytest.mark.asyncio
    async def test_a_judgment_missing_or_unusable_fails_its_criterion_alone(self, cascade_runs):
        script = {
            # Escalated, answered unusably; clarity and tone, escalated too, are answered.
            (ITEMS[1].submission, LIGHT.requirement): UNUSABLE_JUDGMENT,
            # Escalated (the decision model gave no answer), and no judgment with its id.
            (ITEMS[3].submission, MYTH.requirement): NO_JUDGMENT,
            # Answered unusably, but not escalated: the cascade keeps the decision model's.
            (ITEMS[0].submission, CLARITY.requirement): UNUSABLE_JUDGMENT,
        }
        runs = await cascade_runs(llm_script=script, llm_calls=PER_ITEM)
        replay = replay_escalation(
            runs.dm, runs.llm, THRESHOLD, per_criterion=PER_CRITERION, llm_calls=PER_ITEM
        )
        assert_replays(replay, runs.live)
        live = [r.report for r in runs.live.item_results]

        light, _, clarity, tone, _ = live[1].report
        assert light.escalated and light.error.startswith(
            "parse: no usable judgment for criterion c0 ('light'): criterion_status:"
        )
        assert clarity.escalated and clarity.error is None
        assert tone.escalated and tone.error is None
        # The first escalated result carries the call's usage though its criterion failed.
        dm_cost = DM_INPUT_TOKENS * DM_PRICE
        assert live[1].completion_cost == pytest.approx(dm_cost + LLM_COST)
        _, escalation = await runs.cascade.judge(ITEMS[1].submission, RUBRIC.rubric, QUERY)
        first = escalation.criterion_results[0]
        assert first is not None and first.report.error is not None
        assert (first.usage, first.cost) == (LLM_USAGE, LLM_COST)

        myth = live[3].report[1]
        assert myth.escalated and myth.error == (
            "parse: no usable judgment for criterion c1 ('myth'): the reply has no judgment "
            "with its id"
        )

        clarity = live[0].report[2]
        assert not clarity.escalated and clarity.error is None
        assert runs.llm.item_results[0].report.report[2].error.startswith("parse:")


# =============================================================================
# Estimated cost and time
# =============================================================================


class TestEstimatedCostAndTime:
    @pytest.mark.asyncio
    async def test_an_items_llm_cost_and_time_count_in_full_when_anything_escalated(
        self, cascade_runs
    ):
        runs = await cascade_runs(per_criterion=None, llm_calls=PER_ITEM)
        dm_run = with_times(runs.dm, 0.5, 2.0)
        llm_run = with_times(runs.llm, 4.0, 10.0)

        replay = replay_escalation(dm_run, llm_run, THRESHOLD, llm_calls=PER_ITEM)

        assert escalated_flags(replay) == ESCALATED_GLOBALLY
        escalating = any_escalated(ESCALATED_GLOBALLY)
        dm_cost = DM_INPUT_TOKENS * DM_PRICE
        for i, (r, anything) in enumerate(zip(replay.item_results, escalating, strict=True)):
            llm_cost = llm_run.item_results[i].report.completion_cost or 0.0
            dm_part = 0.0 if ITEMS[i].dm_answers is None else dm_cost
            assert r.report.completion_cost == pytest.approx(
                dm_part + (llm_cost if anything else 0.0)
            )
            assert r.duration_seconds == pytest.approx(0.5 + (4.0 if anything else 0.0))
        # One call per item, but the terse item's failed call cost nothing.
        assert [r.report.completion_cost for r in llm_run.item_results] == [
            None if item.submission == TERSE else LLM_COST for item in ITEMS
        ]
        # The run's time: its items with an escalation over its items.
        assert replay.timing_stats.total_duration_seconds == pytest.approx(
            2.0 + 10.0 * sum(escalating) / len(ITEMS)
        )
        assert replay.timing_stats.mean_item_duration_seconds == pytest.approx(
            sum(r.duration_seconds for r in replay.item_results) / len(ITEMS)
        )
        assert replay.total_completion_cost == pytest.approx(
            sum(r.report.completion_cost or 0.0 for r in replay.item_results)
        )
        assert replay.total_token_usage is None

    @pytest.mark.asyncio
    async def test_the_per_criterion_estimates_differ_and_nothing_else(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None, llm_calls=PER_ITEM)
        dm_run = with_times(runs.dm, 0.5, 2.0)
        llm_run = with_times(runs.llm, 4.0, 10.0)

        per_item = replay_escalation(dm_run, llm_run, THRESHOLD, llm_calls=PER_ITEM)
        per_criterion = replay_escalation(dm_run, llm_run, THRESHOLD)

        for a, b in zip(per_item.item_results, per_criterion.item_results, strict=True):
            assert a.report.model_dump(exclude=ESTIMATED) == b.report.model_dump(exclude=ESTIMATED)
            assert a.error == b.error
        # Per criterion, the run's share is its escalated pairs over its pairs.
        n_escalated = sum(map(sum, ESCALATED_GLOBALLY))
        assert per_criterion.timing_stats.total_duration_seconds == pytest.approx(
            2.0 + 10.0 * n_escalated / (N_CRITERIA * len(ITEMS))
        )
        assert [r.duration_seconds for r in per_criterion.item_results] == pytest.approx(
            [0.5 + 4.0 * sum(flags) / N_CRITERIA for flags in ESCALATED_GLOBALLY]
        )
        assert per_item.experiment_name == per_criterion.experiment_name

    @pytest.mark.asyncio
    async def test_the_runs_time_uses_the_share_of_items(self, cascade_runs, decision_model):
        """With a rubric per item, the per-item share (items with an escalation over items)
        differs from the per-criterion one (escalated pairs over pairs)."""
        data = per_item_dataset()
        decision_model.scripts = {**decision_model.scripts, **per_item_decision_model_scripts()}
        runs = await cascade_runs(
            data=data, per_criterion=None, llm_script=PER_ITEM_LLM_SCRIPT, llm_calls=PER_ITEM
        )
        dm_run = with_times(runs.dm, 0.5, 2.0)
        llm_run = with_times(runs.llm, 4.0, 10.0)

        replay = replay_escalation(dm_run, llm_run, THRESHOLD, llm_calls=PER_ITEM)

        assert escalated_flags(replay) == [[True, False], [False, True, True], [False, False]]
        assert [r.duration_seconds for r in replay.item_results] == pytest.approx([4.5, 4.5, 0.5])
        assert replay.timing_stats.total_duration_seconds == pytest.approx(2.0 + 10.0 * 2 / 3)
        per_criterion = replay_escalation(dm_run, llm_run, THRESHOLD)
        assert per_criterion.timing_stats.total_duration_seconds == pytest.approx(
            2.0 + 10.0 * 3 / 7
        )

    @pytest.mark.asyncio
    async def test_an_item_with_an_empty_rubric_makes_no_call_in_either_run(
        self, cascade_runs, decision_model
    ):
        """The unit of the estimates is the LLM call: an item with no criteria is asked
        nothing by the LLM run or the cascade, so it counts among neither run's calls, as it
        has no pairs when the judges are called per criterion."""
        data = per_item_dataset()
        data.add_item("Nothing to judge here.", "empty", ground_truth=[], rubric=Rubric([]))
        decision_model.scripts = {**decision_model.scripts, **per_item_decision_model_scripts()}
        runs = await cascade_runs(
            data=data, per_criterion=None, llm_script=PER_ITEM_LLM_SCRIPT, llm_calls=PER_ITEM
        )
        dm_run = with_times(runs.dm, 0.5, 2.0)
        llm_run = with_times(runs.llm, 4.0, 10.0)

        replay = replay_escalation(dm_run, llm_run, THRESHOLD, llm_calls=PER_ITEM)

        assert_replays(replay, runs.live)
        assert escalated_flags(replay) == [[True, False], [False, True, True], [False, False], []]
        # The LLM run called on the three items with criteria, the cascade on two of them.
        assert calls_by_judge(runs.llm_grader) == {"escalation": 3}
        assert calls_by_judge(runs.cascade) == {"escalation": 2}
        empty = replay.item_results[3]
        assert (empty.error, empty.duration_seconds) == (None, 0.5)
        # Neither judge was asked anything, so the item has no cost, as in the live cascade.
        assert empty.report.completion_cost is None
        assert runs.live.item_results[3].report.completion_cost is None
        assert [r.duration_seconds for r in replay.item_results] == pytest.approx(
            [4.5, 4.5, 0.5, 0.5]
        )
        # Two of the LLM run's three calls, as without the empty item.
        assert replay.timing_stats.total_duration_seconds == pytest.approx(2.0 + 10.0 * 2 / 3)
        per_criterion = replay_escalation(dm_run, llm_run, THRESHOLD)
        assert per_criterion.timing_stats.total_duration_seconds == pytest.approx(
            2.0 + 10.0 * 3 / 7
        )

    @pytest.mark.asyncio
    async def test_an_item_the_decision_model_run_failed_counts_in_the_denominator_only(
        self, cascade_runs
    ):
        """A live cascade sends an item whose decision-model grading raised to no LLM judge,
        but the LLM run made its call: it counts among the run's calls, none escalated."""
        runs = await cascade_runs(llm_calls=PER_ITEM)
        dm_run = with_times(runs.dm, 0.5, 2.0)
        failed = dataclasses.replace(
            dm_run.item_results[0],
            report=EvaluationReport(score=None, error="grading crashed"),
            error="grading crashed",
        )
        dm_run = dataclasses.replace(dm_run, item_results=[failed, *dm_run.item_results[1:]])
        llm_run = with_times(runs.llm, 4.0, 10.0)

        replay = replay_escalation(
            dm_run, llm_run, THRESHOLD, per_criterion=PER_CRITERION, llm_calls=PER_ITEM
        )

        first = replay.item_results[0]
        assert (first.error, first.report, first.duration_seconds) == (
            "grading crashed",
            failed.report,
            0.5,
        )
        # Tone escalates on every other item: 6 of the LLM run's 7 calls.
        rest = dataclasses.replace(replay, item_results=replay.item_results[1:])
        assert all(any_escalated(escalated_flags(rest)))
        assert replay.timing_stats.total_duration_seconds == pytest.approx(2.0 + 10.0 * 6 / 7)
        assert [r.duration_seconds for r in replay.item_results[1:]] == pytest.approx([4.5] * 6)

    @pytest.mark.asyncio
    async def test_an_item_with_no_known_cost_has_none(self, cascade_runs):
        """With an unpriced decision model, an item on which nothing escalated has no cost,
        as in the live cascade; the others cost their escalation calls."""
        runs = await cascade_runs(
            threshold=0.0,
            per_criterion=None,
            dm_config=dm(input_cost_per_token=None),
            llm_calls=PER_ITEM,
        )
        replay = replay_escalation(runs.dm, runs.llm, 0.0, llm_calls=PER_ITEM)

        costs = [r.report.completion_cost for r in replay.item_results]
        assert costs == [r.report.completion_cost for r in runs.live.item_results]
        escalating = any_escalated(escalated_flags(replay))
        terse = [item.submission == TERSE for item in ITEMS]
        assert costs == [
            LLM_COST if anything and not failed else None
            for anything, failed in zip(escalating, terse, strict=True)
        ]
        assert costs[0] is None


# =============================================================================
# Calibration
# =============================================================================

COST_AND_TIME = {"cost_usd", "compute_seconds"}


class TestCalibrate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("fit", [False, True], ids=["global", "per-criterion"])
    async def test_only_cost_and_time_change(self, cascade_runs, fit):
        runs = await cascade_runs(per_criterion=None, llm_calls=PER_ITEM)
        dm_run = with_times(runs.dm, 0.5, 2.0)
        llm_run = with_times(runs.llm, 4.0, 10.0)
        settings: dict[str, Any] = {"per_criterion": fit, "min_pairs_per_criterion": 2}

        with warnings.catch_warnings():
            # Thin criteria and scikit-learn's small-data warnings are not at issue here.
            warnings.simplefilter("ignore")
            per_criterion = calibrate_escalation(runs.data, dm_run, llm_run, **settings)
            per_item = calibrate_escalation(
                runs.data, dm_run, llm_run, **settings, llm_calls=PER_ITEM
            )

        assert [p.model_dump(exclude=COST_AND_TIME) for p in per_item.points] == [
            p.model_dump(exclude=COST_AND_TIME) for p in per_criterion.points
        ]
        assert [(p.cost_usd, p.compute_seconds) for p in per_item.points] != [
            (p.cost_usd, p.compute_seconds) for p in per_criterion.points
        ]
        assert per_item.best() == per_item.points[per_criterion.points.index(per_criterion.best())]
        # Each point's estimates are the per-item replay's.
        for p in per_item.points:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the point was calibrated on these items
                replay = replay_escalation(dm_run, llm_run, p, llm_calls=PER_ITEM)
            stats = escalation_stats(replay, runs.data)
            assert (p.cost_usd, p.compute_seconds) == (stats.cost_usd, stats.compute_seconds)

    @pytest.mark.asyncio
    async def test_the_point_at_the_cascades_threshold_costs_what_the_live_cascade_did(
        self, cascade_runs
    ):
        runs = await cascade_runs(per_criterion=None, llm_calls=PER_ITEM)
        (point,) = calibrate_escalation(
            runs.data, runs.dm, runs.llm, thresholds=[THRESHOLD], llm_calls=PER_ITEM
        ).points

        live = escalation_stats(runs.live, runs.data)
        # A live run records neither its configuration nor its fingerprint, and its time is
        # measured; its cost, summed in another order, is the point's up to rounding.
        not_equal = {"threshold", "calibration_fingerprint", "compute_seconds", "cost_usd"}
        assert point.model_dump(exclude=not_equal) == live.model_dump(exclude=not_equal)
        assert point.cost_usd == pytest.approx(live.cost_usd)
        assert live.cost_usd == pytest.approx(runs.live.total_completion_cost)


# =============================================================================
# The llm_calls keyword
# =============================================================================


class TestTheKeyword:
    @pytest.mark.parametrize("function", [replay_escalation, calibrate_escalation])
    def test_keyword_only_defaulting_to_per_criterion(self, function):
        parameter = inspect.signature(function).parameters["llm_calls"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == "per_criterion"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["per_call", "PER_ITEM", None, 1])
    async def test_an_invalid_value_raises_the_graders_message(
        self, cascade_runs, make_grader, value
    ):
        runs = await cascade_runs()
        with pytest.raises(ValueError) as refused:
            make_grader(judge_model_config=dm(), llm_calls=value)
        message = LLM_CALLS_MESSAGE.format(value)
        assert str(refused.value) == message

        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            replay_escalation(runs.dm, runs.llm, THRESHOLD, llm_calls=value)
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            calibrate_escalation(runs.data, runs.dm, runs.llm, llm_calls=value)

    @pytest.mark.asyncio
    async def test_the_default_replay_is_per_criterion(self, cascade_runs):
        runs = await cascade_runs()
        dm_run = with_times(runs.dm, 0.5, 2.0)
        llm_run = with_times(runs.llm, 4.0, 10.0)

        default = replay_escalation(dm_run, llm_run, THRESHOLD, per_criterion=PER_CRITERION)
        explicit = replay_escalation(
            dm_run, llm_run, THRESHOLD, per_criterion=PER_CRITERION, llm_calls="per_criterion"
        )

        def summary(result: EvalResult) -> list[Any]:
            return [
                (r.item_idx, r.error, r.duration_seconds, r.report.model_dump())
                for r in result.item_results
            ]

        assert summary(explicit) == summary(default)
        assert explicit.timing_stats == default.timing_stats
        assert explicit.total_completion_cost == default.total_completion_cost
        assert_replays(default, runs.live)
        # The per-criterion estimate: the escalated share of each item's criteria.
        assert [r.duration_seconds for r in default.item_results] == pytest.approx(
            [0.5 + 4.0 * sum(flags) / N_CRITERIA for flags in ESCALATED]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = calibrate_escalation(runs.data, dm_run, llm_run, thresholds=[0.2, THRESHOLD])
            explicit_curve = calibrate_escalation(
                runs.data, dm_run, llm_run, thresholds=[0.2, THRESHOLD], llm_calls="per_criterion"
            )
        assert explicit_curve == curve


# =============================================================================
# The manifest
# =============================================================================


def grader_config(result: EvalResult) -> dict[str, Any]:
    assert result.experiment_dir is not None
    manifest = json.loads((Path(result.experiment_dir) / "manifest.json").read_text())
    return manifest["grader_config"]


class TestManifest:
    @pytest.mark.asyncio
    async def test_a_per_item_cascade_records_llm_calls_and_its_escalation_as_before(
        self, cascade_runs
    ):
        per_item: Runs = await cascade_runs(PANEL, llm_calls=PER_ITEM)
        default: Runs = await cascade_runs(PANEL)

        cascade = grader_config(per_item.live)
        before = grader_config(default.live)
        assert cascade["llm_calls"] == "per_item"
        assert "llm_calls" not in before
        assert cascade["escalation"] == before["escalation"]
        assert {key: value for key, value in cascade.items() if key != "llm_calls"} == before
        # The LLM run records it too; the decision-model run, graded per request, does not.
        assert grader_config(per_item.llm)["llm_calls"] == "per_item"
        assert "llm_calls" not in grader_config(per_item.dm)
