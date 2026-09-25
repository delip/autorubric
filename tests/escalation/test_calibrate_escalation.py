"""``calibrate_escalation`` and ``EscalationCurve``: choosing a cascade's thresholds offline.

``calibrate_escalation`` replays a cascade at every threshold of a sweep
(``replay_escalation``) and measures each replay (``escalation_stats`` over
``compute_metrics``); ``EscalationCurve.best`` picks a point by one rule. The rule is pinned
on hand-built curves. Calibration runs on the scripted runs of ``cascade_runs``: the same
decision model and LLM judges whose live cascade the replay tests reproduce, so a
calibrated point reads as a live cascade at its thresholds would, but for its estimated
cost and time and the configuration a run does not record. Per-criterion
fitting runs on a dataset built so that each fitted threshold is known in advance.
Nothing reaches the network.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest

import autorubric
from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    Criterion,
    CriterionVerdict,
    EscalationCurve,
    EscalationPoint,
    EvalResult,
    MetricsResult,
    Rubric,
    RubricDataset,
    calibrate_escalation,
    compute_metrics,
    escalation_stats,
    replay_escalation,
)
from autorubric.escalation import _calibration_fingerprint
from escalation.cascade_runs import (
    CLARITY,
    CLARITY_LABELS,
    ITEMS,
    LIGHT,
    MYTH,
    NA,
    PER_ITEM_LLM_SCRIPT,
    QUERY,
    SENTENCES,
    THRESHOLD,
    dataset,
    noul,
    per_item_dataset,
    per_item_decision_model_scripts,
    pick,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CANNOT_ASSESS = CriterionVerdict.CANNOT_ASSESS

GRID = [i / 50 for i in range(51)]

# What a calibrated point holds that a live run's statistics do not: its configuration and
# fingerprint (an EvalResult records neither), and cost and time as a replay estimates them.
NOT_ON_A_LIVE_RUN = {"threshold", "calibration_fingerprint", "cost_usd", "compute_seconds"}


def point(
    threshold: float | None,
    escalation_rate: float | None,
    metric: float | None,
    **fields: Any,
) -> EscalationPoint:
    return EscalationPoint(
        threshold=threshold,
        escalation_rate=escalation_rate,
        dm_accuracy_kept=None,
        dm_accuracy_escalated=None,
        fallback_accuracy_escalated=None,
        metric=metric,
        cost_usd=None,
        compute_seconds=0.0,
        **fields,
    )


def fingerprint(data: RubricDataset) -> str:
    return _calibration_fingerprint(
        (data.items[i], data.get_item_rubric(i).rubric) for i in range(len(data))
    )


def calibration_warnings(caught: Any) -> list[Any]:
    """The warnings ``calibrate_escalation`` issued among those caught (``compute_metrics``
    lets scikit-learn's through on small data)."""
    return [w for w in caught if str(w.message).startswith("calibrate_escalation")]


def measured(
    result: EvalResult, data: RubricDataset, like: EscalationPoint, **kwargs: Any
) -> EscalationPoint:
    """``escalation_stats`` of ``result``, with the configuration and fingerprint of ``like``
    (which a run does not record)."""
    return escalation_stats(result, data, **kwargs).model_copy(
        update={
            "threshold": like.threshold,
            "per_criterion": like.per_criterion,
            "calibration_fingerprint": like.calibration_fingerprint,
        }
    )


# =============================================================================
# EscalationCurve.best: the selection rule, on hand-built curves
# =============================================================================

# Metrics and rates are dyadic, so "within tolerance" is exact arithmetic.
CURVE = EscalationCurve(
    points=[
        point(0.0, 0.0, 0.75),
        point(0.2, 0.125, 0.8125),
        point(0.4, 0.25, 0.875),
        point(0.6, 0.375, 0.875),
        point(0.8, 0.5, 0.8125),
        point(1.0, 1.0, 0.75),
    ]
)


def best_threshold(curve: EscalationCurve, **kwargs: Any) -> float | None:
    return curve.best(**kwargs).threshold


class TestBest:
    def test_the_least_escalating_of_the_best_points(self):
        # 0.875 at 0.4 and 0.6; 0.4 escalates less.
        assert best_threshold(CURVE) == 0.4
        assert CURVE.best() is CURVE.points[2]

    @pytest.mark.parametrize(
        ("tolerance", "expected"), [(0.0, 0.4), (0.0625, 0.2), (0.125, 0.0), (math.inf, 0.0)]
    )
    def test_tolerance_trades_metric_for_less_escalation(self, tolerance, expected):
        """Within ``tolerance`` of the best metric, the least escalating point wins."""
        assert best_threshold(CURVE, tolerance=tolerance) == expected

    @pytest.mark.parametrize(
        ("budget", "tolerance", "expected"),
        [
            (0.0, 0.0, 0.0),
            (0.125, 0.0, 0.2),  # the best within the budget, not overall
            (0.25, 0.0, 0.4),  # the budget is inclusive
            (0.3, 0.0625, 0.2),  # the tolerance is relative to the best within the budget
            (0.125, 0.0625, 0.0),
            (1.0, 0.0, 0.4),
        ],
    )
    def test_the_escalation_budget(self, budget, tolerance, expected):
        assert best_threshold(CURVE, max_escalation_rate=budget, tolerance=tolerance) == expected

    def test_no_point_within_the_budget(self):
        curve = EscalationCurve(points=[point(0.5, 0.25, 0.875), point(0.7, 0.5, 0.9375)])
        with pytest.raises(
            ValueError, match=r"no point .* max_escalation_rate=0.125.*the lowest is 0.25"
        ):
            curve.best(max_escalation_rate=0.125)

    def test_an_empty_curve(self):
        with pytest.raises(ValueError, match=r"has no points"):
            EscalationCurve(points=[]).best()

    def test_points_without_a_metric_are_passed_over(self):
        curve = EscalationCurve(points=[point(0.0, 0.0, None), point(0.5, 0.5, 0.75)])
        assert best_threshold(curve) == 0.5
        with pytest.raises(ValueError, match=r"no point .* has a metric"):
            curve.best(max_escalation_rate=0.25)
        with pytest.raises(ValueError, match=r"no point .* has a metric"):
            EscalationCurve(points=[point(0.0, 0.0, None)]).best()

    def test_points_without_an_escalation_rate(self):
        """A run with no pairs has no escalation rate: it is outside any budget, and it
        escalates no less than a point that has one."""
        curve = EscalationCurve(points=[point(0.0, None, 0.875), point(0.5, 0.5, 0.875)])
        assert best_threshold(curve) == 0.5
        assert best_threshold(curve, max_escalation_rate=1.0) == 0.5
        only = EscalationCurve(points=[point(0.0, None, 0.875)])
        assert best_threshold(only) == 0.0
        with pytest.raises(ValueError, match=r"no point .* has an escalation rate"):
            only.best(max_escalation_rate=1.0)

    @pytest.mark.parametrize(
        ("best_metric", "close_metric", "tolerance"),
        [(0.4, 0.35, 0.05), (0.13, 0.12, 0.01), (0.2, 0.15, 0.05), (0.13, 0.125, 0.005)],
    )
    def test_a_gap_equal_to_the_tolerance_is_within_it(self, best_metric, close_metric, tolerance):
        """Decimal metrics and tolerances are not exact in binary floating point (``0.4 -
        0.05`` exceeds ``0.35``): a gap equal to the tolerance up to rounding is within it."""
        curve = EscalationCurve(
            points=[point(0.6, 0.5, best_metric), point(0.2, 0.125, close_metric)]
        )
        assert best_threshold(curve, tolerance=tolerance) == 0.2
        # A gap measurably wider than the tolerance is not.
        wider = EscalationCurve(
            points=[point(0.6, 0.5, best_metric), point(0.2, 0.125, close_metric - 1e-6)]
        )
        assert best_threshold(wider, tolerance=tolerance) == 0.6

    @pytest.mark.parametrize("nan_first", [True, False], ids=["nan-first", "nan-last"])
    def test_a_nan_metric_is_passed_over_as_undefined(self, nan_first):
        """A NaN metric is an undefined one, as ``None`` is, wherever it stands."""
        undefined, defined = point(0.2, 0.125, float("nan")), point(0.6, 0.5, 0.75)
        curve = EscalationCurve(points=[undefined, defined] if nan_first else [defined, undefined])
        assert best_threshold(curve) == 0.6
        assert best_threshold(curve, tolerance=math.inf) == 0.6
        with pytest.raises(ValueError, match=r"no point .* has a metric"):
            EscalationCurve(points=[undefined]).best()

    def test_equally_escalating_points_prefer_the_higher_metric(self):
        curve = EscalationCurve(points=[point(0.1, 0.25, 0.8125), point(0.5, 0.25, 0.875)])
        assert best_threshold(curve, tolerance=0.0625) == 0.5

    def test_equal_points_prefer_the_lower_threshold_then_the_first(self):
        curve = EscalationCurve(
            points=[point(0.3, 0.25, 0.875), point(None, 0.25, 0.875), point(0.1, 0.25, 0.875)]
        )
        assert best_threshold(curve) == 0.1
        twins = EscalationCurve(points=[point(0.1, 0.25, 0.875), point(0.1, 0.25, 0.875)])
        assert twins.best() is twins.points[0]

    @pytest.mark.parametrize("tolerance", [-0.01, float("nan"), True, "0.1", None])
    def test_the_tolerance_is_a_number_at_least_zero(self, tolerance):
        with pytest.raises(ValueError, match=r"tolerance must be a number >= 0"):
            CURVE.best(tolerance=tolerance)

    @pytest.mark.parametrize("budget", [-0.01, 1.5, float("nan"), True, "0.1"])
    def test_the_budget_is_a_number_in_the_unit_interval(self, budget):
        with pytest.raises(ValueError, match=r"max_escalation_rate must be a number in \[0, 1\]"):
            CURVE.best(max_escalation_rate=budget)


class TestCurveType:
    def test_frozen_with_its_points(self):
        assert list(EscalationCurve.model_fields) == ["points"]
        with pytest.raises(Exception, match="frozen"):
            CURVE.points = []

    def test_to_dataframe_has_a_row_per_point_and_a_column_per_field(self):
        pd = pytest.importorskip("pandas")
        calibrated = EscalationCurve(
            points=[
                point(0.2, 0.125, 0.8125, per_criterion={"tone": 0.5}, calibration_fingerprint="f"),
                point(0.4, 0.25, None, calibration_fingerprint="f"),
            ]
        )
        frame = calibrated.to_dataframe()
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == list(EscalationPoint.model_fields)
        assert frame.to_dict("records")[0] == calibrated.points[0].model_dump()
        assert frame["threshold"].tolist() == [0.2, 0.4]
        assert frame["metric"].isna().tolist() == [False, True]

        empty = EscalationCurve(points=[]).to_dataframe()
        assert empty.empty and list(empty.columns) == list(EscalationPoint.model_fields)


# =============================================================================
# calibrate_escalation on the scripted cascade runs
# =============================================================================


class TestCalibrate:
    @pytest.mark.asyncio
    async def test_a_point_per_threshold_each_a_measured_replay(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None)
        curve = calibrate_escalation(runs.data, runs.dm, runs.llm)

        assert isinstance(curve, EscalationCurve)
        # The default sweep: 0.00, 0.02, ..., 1.00.
        assert [p.threshold for p in curve.points] == GRID
        for p in curve.points:
            assert p.per_criterion is None
            assert p.calibration_fingerprint == fingerprint(runs.data)
            replay = replay_escalation(runs.dm, runs.llm, p.threshold)
            assert p == measured(replay, runs.data, p)

    @pytest.mark.asyncio
    async def test_the_point_at_the_cascades_threshold_reads_as_the_live_cascade(
        self, cascade_runs
    ):
        runs = await cascade_runs(per_criterion=None)
        curve = calibrate_escalation(runs.data, runs.dm, runs.llm)

        (at_threshold,) = [p for p in curve.points if p.threshold == THRESHOLD]
        live = escalation_stats(runs.live, runs.data)
        assert at_threshold.model_dump(exclude=NOT_ON_A_LIVE_RUN) == live.model_dump(
            exclude=NOT_ON_A_LIVE_RUN
        )
        assert at_threshold.metric == compute_metrics(runs.live, runs.data).criterion_accuracy

    @pytest.mark.asyncio
    async def test_replays_are_scored_as_the_cascades_grader_scores(self, cascade_runs):
        """With the cascade's scoring settings, a score-level metric reads as the live
        cascade's; with the defaults it would not."""
        scoring: dict[str, Any] = {
            "normalize": False,
            "cannot_assess_config": CannotAssessConfig(
                strategy=CannotAssessStrategy.PARTIAL, partial_credit=0.3
            ),
        }
        runs = await cascade_runs(per_criterion=None, **scoring)
        curve = calibrate_escalation(
            runs.data,
            runs.dm,
            runs.llm,
            thresholds=[0.0, THRESHOLD],
            metric="score_rmse",
            **scoring,
        )

        live = escalation_stats(runs.live, runs.data, metric="score_rmse")
        assert curve.points[1].model_dump(exclude=NOT_ON_A_LIVE_RUN) == live.model_dump(
            exclude=NOT_ON_A_LIVE_RUN
        )
        for p in curve.points:
            replay = replay_escalation(runs.dm, runs.llm, p.threshold, **scoring)
            assert p == measured(replay, runs.data, p, metric="score_rmse")
        unscored = calibrate_escalation(
            runs.data, runs.dm, runs.llm, thresholds=[THRESHOLD], metric="score_rmse"
        )
        assert unscored.points[0].metric != live.metric

        # The fit's replays are scored the same way.
        sweep = [0.0, 0.15, 0.3, 0.45, 0.5, 0.6, 0.95, 0.97, 1.0]
        names = ["light", "myth", "clarity", "tone"]

        def lower_rmse(metrics: MetricsResult) -> float | None:
            return None if metrics.score_rmse is None else -metrics.score_rmse

        fit = {"thresholds": sweep, "per_criterion": True, "min_pairs_per_criterion": 1}
        fitted = calibrate_escalation(
            runs.data, runs.dm, runs.llm, metric=lower_rmse, **fit, **scoring
        ).points[0]
        assert fitted.per_criterion == expected_fit(
            runs, sweep, names, metric=lower_rmse, **scoring
        )
        assert fitted.per_criterion != (
            calibrate_escalation(runs.data, runs.dm, runs.llm, metric=lower_rmse, **fit)
            .points[0]
            .per_criterion
        )

    @pytest.mark.asyncio
    async def test_the_sweep_escalates_more_as_the_threshold_rises(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None)
        curve = calibrate_escalation(runs.data, runs.dm, runs.llm)
        rates = [p.escalation_rate for p in curve.points]
        assert rates == sorted(rates)
        # Errors and abstentions escalate at 0; everything short of certainty at 1.
        failures_only = replay_escalation(runs.dm, runs.llm, 0.0)
        assert rates[0] == escalation_stats(failures_only, runs.data).escalation_rate > 0
        assert rates[-1] == 1.0

    @pytest.mark.asyncio
    async def test_thresholds_are_swept_in_the_order_given(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None)
        curve = calibrate_escalation(runs.data, runs.dm, runs.llm, thresholds=[0.9, 0.1, 0.5])
        assert [p.threshold for p in curve.points] == [0.9, 0.1, 0.5]
        replay = replay_escalation(runs.dm, runs.llm, 0.1)
        assert curve.points[1] == measured(replay, runs.data, curve.points[1])

    @pytest.mark.asyncio
    async def test_a_metric_by_name_or_function(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None)
        by_name = calibrate_escalation(
            runs.data, runs.dm, runs.llm, thresholds=[0.3, 0.7], metric="mean_kappa"
        )
        by_function = calibrate_escalation(
            runs.data, runs.dm, runs.llm, thresholds=[0.3, 0.7], metric=lambda m: m.macro_accuracy
        )
        for threshold, named, computed in zip(
            [0.3, 0.7], by_name.points, by_function.points, strict=True
        ):
            metrics = compute_metrics(replay_escalation(runs.dm, runs.llm, threshold), runs.data)
            assert named.metric == metrics.mean_kappa
            assert computed.metric == metrics.macro_accuracy
        with pytest.raises(ValueError, match=r"not an attribute of MetricsResult"):
            calibrate_escalation(runs.data, runs.dm, runs.llm, metric="accuracy_of_everything")

    @pytest.mark.asyncio
    async def test_a_dataset_with_a_rubric_per_item(self, cascade_runs, decision_model):
        """``compute_metrics`` takes its pooled path: ``criterion_accuracy`` is the exact
        accuracy over every rubric point, and a per-scale metric reads ``pooled_by_scale``."""
        data = per_item_dataset()
        decision_model.scripts = {**decision_model.scripts, **per_item_decision_model_scripts()}
        runs = await cascade_runs(data=data, per_criterion=None, llm_script=PER_ITEM_LLM_SCRIPT)

        def pooled_binary(metrics: MetricsResult) -> float | None:
            return next(
                e.exact_accuracy for e in metrics.pooled_by_scale if e.scale_type == "binary"
            )

        pooled = calibrate_escalation(data, runs.dm, runs.llm, thresholds=[0.0, THRESHOLD, 1.0])
        binary = calibrate_escalation(
            data, runs.dm, runs.llm, thresholds=[0.0, THRESHOLD, 1.0], metric=pooled_binary
        )
        for p, q in zip(pooled.points, binary.points, strict=True):
            metrics = compute_metrics(replay_escalation(runs.dm, runs.llm, p.threshold), data)
            assert metrics.per_criterion == []
            assert p.metric == metrics.criterion_accuracy
            assert q.metric == pooled_binary(metrics)
        at_threshold = binary.points[1]
        live = escalation_stats(runs.live, data, metric=pooled_binary)
        assert at_threshold.model_dump(exclude=NOT_ON_A_LIVE_RUN) == live.model_dump(
            exclude=NOT_ON_A_LIVE_RUN
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("thresholds", "message"),
        [
            ([], r"thresholds must hold at least one threshold"),
            ([0.5, 1.5], r"thresholds\[1\] must be a number in \[0, 1\]; got 1.5"),
            ([True], r"thresholds\[0\] must be a number in \[0, 1\]"),
            ([float("nan")], r"thresholds\[0\] must be a number in \[0, 1\]"),
        ],
    )
    async def test_thresholds_are_numbers_in_the_unit_interval(
        self, cascade_runs, thresholds, message
    ):
        runs = await cascade_runs(per_criterion=None)
        with pytest.raises(ValueError, match=message):
            calibrate_escalation(runs.data, runs.dm, runs.llm, thresholds=thresholds)

    @pytest.mark.asyncio
    async def test_the_runs_requirements_are_the_replays(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None)
        with pytest.raises(ValueError, match=r"dm_result must come from one decision-model"):
            calibrate_escalation(runs.data, runs.llm, runs.llm)


# =============================================================================
# Per-criterion thresholds
# =============================================================================


async def graded(
    cascade_runs: Any,
    decision_model: Any,
    rubric: Rubric,
    rows: list[tuple[list[dict[str, Any]], list[Any], list[Any]]],
    name: str,
) -> Any:
    """``rows`` graded as ``cascade_runs`` grades: per item, the decision model's answer to
    each criterion (``noul``, ``pick``), the LLM judge's (a verdict value or an option label)
    and the ground truth."""
    data = RubricDataset(prompt=QUERY, rubric=rubric, name=name)
    dm_scripts: dict[str, Any] = {}
    llm_script: dict[tuple[str, str], Any] = {}
    for i, (dm_answers, llm_answers, truth) in enumerate(rows):
        submission = f"{name}: answer {i}."
        data.add_item(submission, f"item {i}", ground_truth=list(truth))
        dm_scripts[submission] = {f"c{c}": answer for c, answer in enumerate(dm_answers)}
        for criterion, answer in zip(rubric.rubric, llm_answers, strict=True):
            llm_script[(submission, criterion.requirement)] = answer
    decision_model.scripts = {**decision_model.scripts, **dm_scripts}
    return await cascade_runs(data=data, per_criterion=None, llm_script=llm_script)


def expected_fit(
    runs: Any, sweep: list[float], names: list[str], **settings: Any
) -> dict[str, float]:
    """The documented fit, step by step through the public API: from the global threshold
    ``best`` picks on the sweep alone, each name's threshold in turn is the one ``best``
    picks on the whole run's metric, every other threshold held where it is."""
    metric = settings.pop("metric", "criterion_accuracy")
    start = calibrate_escalation(
        runs.data, runs.dm, runs.llm, thresholds=sweep, metric=metric, **settings
    ).best()
    fitted: dict[str, float] = {}
    for name in names:
        candidates = [
            escalation_stats(
                replay_escalation(
                    runs.dm,
                    runs.llm,
                    start.threshold,
                    per_criterion={**fitted, name: t},
                    **settings,
                ),
                runs.data,
                metric=metric,
            ).model_copy(update={"threshold": t})
            for t in sweep
        ]
        threshold = EscalationCurve(points=candidates).best().threshold
        assert threshold is not None
        fitted[name] = threshold
    return fitted


SOURCES = Criterion(name="sources", weight=1.0, requirement="Cites a source")
FIT_RUBRIC = Rubric([LIGHT, MYTH, SOURCES, SENTENCES])

# Per item and criterion (light, myth, sources, sentences): the decision model's P(MET)
# (confidence 2|p - 0.5|), the LLM judge's verdict, and the truth. With one threshold for
# every criterion, the best is 18 of the 21 labelled pairs right, first at the grid
# threshold 0.26 (0.25 < t <= 0.5), where the fit starts.
#
# - light: the decision model says MET throughout and is wrong on items 2 (0.25) and 3
#   (0.125); the LLM is right throughout. All 6 are right once both escalate, first at
#   0.26, which escalates 2 of its 6 pairs.
# - myth: the decision model is right throughout; the LLM is wrong on items 1 (0.25) and 2
#   (0.125). All 6 are right until item 2 escalates, and least escalating at 0.
# - sources: labelled on 3 items only, fewer than the 5 pairs a fit needs; its two judges
#   agree on every labelled pair.
# - sentences: unnamed, so it can have no threshold of its own.
FIT_ITEMS = [
    ([0.9375, 0.0625, 0.9375, 0.9375], [MET, UNMET, MET, MET], [MET, UNMET, MET, MET]),
    ([0.875, 0.375, 0.75, 0.75], [MET, MET, MET, MET], [MET, UNMET, CANNOT_ASSESS, MET]),
    (
        [0.625, 0.4375, 0.625, 0.625],
        [UNMET, MET, UNMET, UNMET],
        [UNMET, UNMET, CANNOT_ASSESS, UNMET],
    ),
    ([0.5625, 0.0625, 0.9375, 0.9375], [UNMET, UNMET, MET, MET], [UNMET, UNMET, MET, MET]),
    ([0.9375, 0.0625, 0.5625, 0.5625], [MET, UNMET, UNMET, MET], [MET, UNMET, CANNOT_ASSESS, MET]),
    ([0.9375, 0.0625, 0.9375, 0.9375], [MET, UNMET, MET, MET], [MET, UNMET, UNMET, MET]),
]
FITTED = {"light": 0.26, "myth": 0.0}


async def fit_runs(cascade_runs: Any, decision_model: Any) -> Any:
    """``FIT_ITEMS`` graded as ``cascade_runs`` grades."""
    rows = [
        ([noul(p) for p in ps], [verdict.value for verdict in verdicts], truth)
        for ps, verdicts, truth in FIT_ITEMS
    ]
    return await graded(cascade_runs, decision_model, FIT_RUBRIC, rows, "per-criterion fit")


class TestPerCriterion:
    @pytest.mark.asyncio
    async def test_each_well_labelled_named_criterion_gets_its_own_threshold(
        self, cascade_runs, decision_model
    ):
        runs = await fit_runs(cascade_runs, decision_model)
        with pytest.warns(UserWarning, match=r"'sources' \(3 labelled pairs\)") as caught:
            curve = calibrate_escalation(
                runs.data, runs.dm, runs.llm, per_criterion=True, min_pairs_per_criterion=5
            )
        (warning,) = calibration_warnings(caught)
        assert warning.filename == __file__
        assert "sentences" not in str(warning.message)  # unnamed: never fitted, never warned

        # Each point is a global threshold (for sources and sentences) plus the fitted ones.
        assert [p.threshold for p in curve.points] == GRID
        assert all(p.per_criterion == FITTED for p in curve.points)
        # Fitted on each criterion's own pairs, the thresholds beat any one global threshold:
        # 20 of the 21 labelled pairs right (sources is wrong on item 5 whatever escalates),
        # against 18 at best with one threshold for every criterion.
        best = curve.best()
        assert (best.threshold, best.metric) == (0.26, 20 / 21)
        assert best.escalation_rate == 6 / 24
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="calibrate_escalation")
            single = calibrate_escalation(runs.data, runs.dm, runs.llm)
        assert single.best().metric == 18 / 21

    @pytest.mark.asyncio
    async def test_every_point_replays_to_itself(self, cascade_runs, decision_model):
        runs = await fit_runs(cascade_runs, decision_model)
        with pytest.warns(UserWarning, match=r"'sources'"):
            curve = calibrate_escalation(
                runs.data, runs.dm, runs.llm, per_criterion=True, min_pairs_per_criterion=5
            )
        for p in curve.points:
            with pytest.warns(UserWarning, match=r"calibrated on the items it replays"):
                replay = replay_escalation(runs.dm, runs.llm, p)
            assert p == measured(replay, runs.data, p)
            # The same replay, by its thresholds.
            by_thresholds = replay_escalation(
                runs.dm, runs.llm, p.threshold, per_criterion=p.per_criterion
            )
            assert [r.report for r in by_thresholds.item_results] == [
                r.report for r in replay.item_results
            ]

    @pytest.mark.asyncio
    async def test_the_fit_is_the_selection_rule_on_the_runs_metric(
        self, cascade_runs, decision_model
    ):
        """From the global threshold ``best`` picks, each criterion's threshold in turn is the
        least escalating of those that make the whole run's metric best."""
        runs = await fit_runs(cascade_runs, decision_model)
        sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.8, 0.9]
        with pytest.warns(UserWarning, match=r"'sources'"):
            curve = calibrate_escalation(
                runs.data,
                runs.dm,
                runs.llm,
                thresholds=sweep,
                per_criterion=True,
                min_pairs_per_criterion=5,
            )
        # The start is 0.3 (18 of 21). light: 16 of 21 up to 0.125, 17 up to 0.25, then 18
        # (2 escalated up to 0.75, 3 up to 0.875, 6 above); then myth: 20 up to 0.125
        # (nothing escalated at 0 and 0.1), 19 up to 0.25, 18 above.
        assert curve.points[0].per_criterion == {"light": 0.3, "myth": 0.0}
        assert curve.points[0].per_criterion == expected_fit(runs, sweep, ["light", "myth"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("order", "fitted", "metrics"),
        [
            (("light", "myth"), {"light": 0.5, "myth": 0.5}, [9 / 14, 4 / 14]),
            (("myth", "light"), {"myth": 0.0, "light": 0.5}, [10 / 16, 5 / 16]),
        ],
        ids=["light-first", "myth-first"],
    )
    async def test_each_fit_holds_the_thresholds_already_fitted(
        self, cascade_runs, decision_model, order, fitted, metrics
    ):
        """The fit is coordinate-wise, in order of first appearance: where escalating one
        criterion changes which pairs another's accuracy counts, a threshold fitted with the
        earlier fits in place differs from one fitted alone, and only the former is sure
        not to lower the metric.

        Here escalating myth moves two of its pairs into the LLM's abstentions, which leave
        ``criterion_accuracy``'s denominator. With light still at the global threshold that
        costs a right pair (7 of 16 falls to 6 of 14); with light fitted first it drops a
        wrong one (10 of 16 rises to 9 of 14). A fit of each criterion alone, or of myth
        before light, keeps myth at 0.0 and ends at 10 of 16. The rubric's order decides
        which comes first: light first in [light, myth, sentences], myth first in [myth,
        light, sentences].
        """
        # Per item and criterion: the decision model's answer, the LLM's, and the truth.
        # - light: unsure (0.2) and wrong on items 0 to 2, where the LLM is right; sure and
        #   unlabelled elsewhere.
        # - myth: on items 0 to 2 (truth UNMET), sure and right on item 0, unsure (0.2) on
        #   items 1 (right) and 2 (wrong), where the LLM abstains; sure and unlabelled
        #   elsewhere.
        # - sentences (unnamed, truth MET): unsure (0.2) throughout and right on items 0 to
        #   4; the LLM is wrong throughout.
        labelled_myth = [
            (noul(0.05), "UNMET", UNMET),
            (noul(0.4), "CANNOT_ASSESS", UNMET),
            (noul(0.6), "CANNOT_ASSESS", UNMET),
        ]
        named = {"light": LIGHT, "myth": MYTH}
        rows = []
        for i in range(10):
            columns = {
                "light": (noul(0.4), "MET", MET) if i < 3 else (noul(0.95), "MET", CANNOT_ASSESS),
                "myth": labelled_myth[i] if i < 3 else (noul(0.05), "UNMET", CANNOT_ASSESS),
            }
            sentences = (noul(0.6) if i < 5 else noul(0.4), "UNMET", MET)
            dm_answers, llm_answers, truth = zip(
                *(columns[name] for name in order), sentences, strict=True
            )
            rows.append((list(dm_answers), list(llm_answers), list(truth)))
        rubric = Rubric([*(named[name] for name in order), SENTENCES])
        runs = await graded(cascade_runs, decision_model, rubric, rows, "coordinates")
        sweep = [0.0, 0.5]

        def accuracy(**per_criterion: float) -> float | None:
            replay = replay_escalation(runs.dm, runs.llm, 0.0, per_criterion=per_criterion)
            return compute_metrics(replay, runs.data).criterion_accuracy

        assert [accuracy(), accuracy(myth=0.5)] == [7 / 16, 6 / 14]
        assert [accuracy(light=0.5), accuracy(light=0.5, myth=0.5)] == [10 / 16, 9 / 14]

        curve = calibrate_escalation(
            runs.data,
            runs.dm,
            runs.llm,
            thresholds=sweep,
            per_criterion=True,
            min_pairs_per_criterion=3,
        )
        per_criterion = curve.points[0].per_criterion
        assert per_criterion == fitted
        assert per_criterion is not None and list(per_criterion) == list(order)
        assert [p.metric for p in curve.points] == metrics
        assert curve.best().threshold == 0.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("min_pairs", "fitted", "thin"),
        [
            (6, FITTED, ["'sources' (3 labelled pairs)"]),
            # sources' judges agree on its labelled pairs: it escalates least at 0.
            (3, {**FITTED, "sources": 0.0}, []),
        ],
        ids=["six", "three"],
    )
    async def test_a_criterion_with_exactly_min_pairs_labelled_pairs_is_fitted(
        self, cascade_runs, decision_model, min_pairs, fitted, thin
    ):
        runs = await fit_runs(cascade_runs, decision_model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            curve = calibrate_escalation(
                runs.data,
                runs.dm,
                runs.llm,
                per_criterion=True,
                min_pairs_per_criterion=min_pairs,
            )
        assert curve.points[0].per_criterion == fitted
        messages = [str(w.message) for w in calibration_warnings(caught)]
        assert len(messages) == len(thin)
        for message, names in zip(messages, thin, strict=True):
            assert names in message and "light" not in message and "myth" not in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("metric", ["criterion_accuracy", "macro_accuracy", "mean_kappa"])
    async def test_fitting_never_lowers_the_best_metric(self, cascade_runs, metric):
        """On the scripted cascade's rubric (binary and multi-choice criteria, failed and
        abstaining votes), each fit keeps or raises the run's metric from the start."""
        runs = await cascade_runs(per_criterion=None)
        # Between and at the decision model's confidences (0.1 to 0.96).
        sweep = [0.0, 0.15, 0.3, 0.45, 0.5, 0.6, 0.95, 0.97, 1.0]
        global_only = calibrate_escalation(
            runs.data, runs.dm, runs.llm, thresholds=sweep, metric=metric
        )
        curve = calibrate_escalation(
            runs.data,
            runs.dm,
            runs.llm,
            thresholds=sweep,
            metric=metric,
            per_criterion=True,
            min_pairs_per_criterion=1,
        )
        start = global_only.best()
        fitted = curve.points[0].per_criterion
        assert fitted == expected_fit(
            runs, sweep, ["light", "myth", "clarity", "tone"], metric=metric
        )
        (at_start,) = [p for p in curve.points if p.threshold == start.threshold]
        assert start.metric is not None and at_start.metric is not None
        assert at_start.metric >= start.metric
        assert curve.best().metric >= global_only.best().metric

    @pytest.mark.asyncio
    async def test_every_criterion_thin_is_the_global_sweep(self, cascade_runs, decision_model):
        runs = await fit_runs(cascade_runs, decision_model)
        with pytest.warns(UserWarning) as caught:
            curve = calibrate_escalation(
                runs.data, runs.dm, runs.llm, per_criterion=True, min_pairs_per_criterion=7
            )
        (warning,) = calibration_warnings(caught)
        message = str(warning.message)
        for name, n in [("light", 6), ("myth", 6), ("sources", 3)]:
            assert f"'{name}' ({n} labelled pairs)" in message
        assert "min_pairs_per_criterion=7" in message
        assert curve == calibrate_escalation(runs.data, runs.dm, runs.llm)

    @pytest.mark.asyncio
    async def test_a_metric_undefined_at_every_threshold_fits_nothing(
        self, cascade_runs, decision_model
    ):
        """The fit starts from the point ``best`` picks on the global sweep; with no metric
        anywhere there is none, and every criterion keeps the global threshold."""
        runs = await fit_runs(cascade_runs, decision_model)
        with pytest.warns(UserWarning) as caught:
            curve = calibrate_escalation(
                runs.data,
                runs.dm,
                runs.llm,
                metric=lambda m: None,
                per_criterion=True,
                min_pairs_per_criterion=5,
            )
        thin, undefined = (str(w.message) for w in calibration_warnings(caught))
        assert "'sources' (3 labelled pairs)" in thin
        assert undefined.startswith(
            "calibrate_escalation fits no threshold for 'light' (6 labelled pairs), 'myth' "
            "(6 labelled pairs): the metric is undefined (None) at every threshold of the sweep"
        )
        assert undefined.endswith("; they use the swept global threshold")
        assert all(p.per_criterion is None and p.metric is None for p in curve.points)

    @pytest.mark.asyncio
    async def test_a_fit_never_trades_the_runs_metric_for_a_criterions_ratio(
        self, cascade_runs, decision_model
    ):
        """Where the fallback abstains, escalating a criterion's pairs shrinks its share of
        the run's accuracy: its own ratio of right to predicted pairs can rise while the
        run's falls. The fit reads the run's metric, so it keeps the criterion's pairs."""
        # light: the decision model is right on items 0 to 2 and wrong on item 3, each at
        # confidence 0.3; the LLM abstains on items 0, 1 and 3 and is right on item 2.
        # sentences (unnamed, confidence 0.9): the decision model is right on items 0 and 1.
        rows = [
            ([noul(0.65), noul(0.95)], ["CANNOT_ASSESS", "MET"], [MET, MET]),
            ([noul(0.65), noul(0.95)], ["CANNOT_ASSESS", "MET"], [MET, MET]),
            ([noul(0.65), noul(0.05)], ["MET", "MET"], [MET, MET]),
            ([noul(0.35), noul(0.05)], ["CANNOT_ASSESS", "MET"], [MET, MET]),
        ]
        runs = await graded(
            cascade_runs, decision_model, Rubric([LIGHT, SENTENCES]), rows, "abstaining fallback"
        )
        sweep = [0.0, 0.5]
        light_at = [
            compute_metrics(replay_escalation(runs.dm, runs.llm, t), runs.data).per_criterion[0]
            for t in sweep
        ]
        assert [m.accuracy for m in light_at] == [3 / 4, 1.0]  # light's own ratio

        # light has exactly the 4 labelled pairs a fit needs.
        curve = calibrate_escalation(
            runs.data,
            runs.dm,
            runs.llm,
            thresholds=sweep,
            per_criterion=True,
            min_pairs_per_criterion=4,
        )
        assert curve.points[0].per_criterion == {"light": 0.0}
        global_only = calibrate_escalation(runs.data, runs.dm, runs.llm, thresholds=sweep)
        assert [p.metric for p in global_only.points] == [5 / 8, 3 / 5]
        assert [p.metric for p in curve.points] == [5 / 8, 5 / 8]
        assert curve.best().metric == global_only.best().metric == 5 / 8

    @pytest.mark.asyncio
    async def test_a_multi_choice_criterion_is_fitted_by_what_the_metric_reads(
        self, cascade_runs, decision_model
    ):
        """``criterion_accuracy`` on a rubric with binary criteria reads those alone, so it
        cannot tell a multi-choice criterion's thresholds apart and the fit escalates it
        least; ``macro_accuracy`` reads it, and the fit escalates where the LLM is right."""
        # light: sure and right. clarity (truth "Very clear"): the decision model is wrong at
        # confidence 1/3 on items 0 and 1, right at 0.96 on items 2 and 3; the LLM is right.
        unsure_wrong = pick("Unclear", CLARITY_LABELS, 0.5)
        sure_right = pick("Very clear", CLARITY_LABELS, 0.97)
        rows = [
            ([noul(0.95), clarity], ["MET", "Very clear"], [MET, "Very clear"])
            for clarity in (unsure_wrong, unsure_wrong, sure_right, sure_right)
        ]
        runs = await graded(
            cascade_runs, decision_model, Rubric([LIGHT, CLARITY]), rows, "mixed rubric"
        )
        sweep = [0.0, 0.5]
        fit = {"per_criterion": True, "min_pairs_per_criterion": 4, "thresholds": sweep}

        by_accuracy = calibrate_escalation(runs.data, runs.dm, runs.llm, **fit)
        assert by_accuracy.points[0].per_criterion == {"light": 0.0, "clarity": 0.0}
        assert by_accuracy.best().escalation_rate == 0.0

        by_macro = calibrate_escalation(
            runs.data, runs.dm, runs.llm, metric="macro_accuracy", **fit
        )
        assert by_macro.points[0].per_criterion == {"light": 0.0, "clarity": 0.5}
        assert by_macro.best().metric == 1.0
        assert by_macro.points[0].per_criterion == expected_fit(
            runs, sweep, ["light", "clarity"], metric="macro_accuracy"
        )

    @pytest.mark.asyncio
    async def test_na_answers_count_as_the_metric_counts_them(self, cascade_runs, decision_model):
        """On a rubric without binary criteria, ``criterion_accuracy`` counts an NA
        prediction as a wrong option; escalating into the LLM's NA answers lowers it, even
        though the criterion's answered pairs all become right."""
        rows = [
            ([pick("Unclear", CLARITY_LABELS, 0.5)], [NA], ["Very clear"]),
            ([pick("Very clear", CLARITY_LABELS, 0.5)], [NA], ["Very clear"]),
            ([pick("Very clear", CLARITY_LABELS, 0.97)], ["Very clear"], ["Very clear"]),
            ([pick("Very clear", CLARITY_LABELS, 0.97)], ["Very clear"], ["Very clear"]),
        ]
        runs = await graded(cascade_runs, decision_model, Rubric([CLARITY]), rows, "na answers")
        sweep = [0.0, 0.5]
        escalated = compute_metrics(replay_escalation(runs.dm, runs.llm, 0.5), runs.data)
        assert escalated.per_criterion[0].exact_accuracy == 1.0  # NA answers left out
        assert escalated.criterion_accuracy == 2 / 4

        curve = calibrate_escalation(
            runs.data,
            runs.dm,
            runs.llm,
            thresholds=sweep,
            per_criterion=True,
            min_pairs_per_criterion=4,
        )
        assert curve.points[0].per_criterion == {"clarity": 0.0}
        assert curve.best().metric == 3 / 4

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [0, -1, 2.5, True, "30", None])
    async def test_min_pairs_is_a_positive_integer(self, cascade_runs, value):
        runs = await cascade_runs(per_criterion=None)
        with pytest.raises(ValueError, match=r"min_pairs_per_criterion must be a positive int"):
            calibrate_escalation(
                runs.data, runs.dm, runs.llm, per_criterion=True, min_pairs_per_criterion=value
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [{"tone": 0.9}, 1, None])
    async def test_per_criterion_is_a_bool(self, cascade_runs, value):
        """Thresholds by name go to ``replay_escalation``; here ``per_criterion`` asks for
        them to be fitted."""
        runs = await cascade_runs(per_criterion=None)
        with pytest.raises(ValueError, match=r"per_criterion must be True or False"):
            calibrate_escalation(runs.data, runs.dm, runs.llm, per_criterion=value)


# =============================================================================
# The calibration fingerprint and held-out replay
# =============================================================================


class TestSameData:
    @pytest.mark.asyncio
    async def test_replaying_the_calibrated_point_on_its_calibration_items_warns(
        self, cascade_runs
    ):
        runs = await cascade_runs(per_criterion=None)
        best = calibrate_escalation(runs.data, runs.dm, runs.llm).best()
        with pytest.warns(UserWarning, match=r"calibrated on the items it replays") as caught:
            replay_escalation(runs.dm, runs.llm, best)
        assert caught[0].filename == __file__

    @pytest.mark.asyncio
    async def test_a_point_calibrated_on_one_split_replays_the_other_silently(self, cascade_runs):
        calibration, held_out = dataset(ITEMS).split_train_test(n_train=4, seed=3)
        calibrated = await cascade_runs(data=calibration, per_criterion=None)
        evaluated = await cascade_runs(data=held_out, per_criterion=None)

        best = calibrate_escalation(calibration, calibrated.dm, calibrated.llm).best()
        assert best.calibration_fingerprint == fingerprint(calibration)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            replay = replay_escalation(evaluated.dm, evaluated.llm, best)
        assert escalation_stats(replay, held_out).escalation_rate is not None

    @pytest.mark.asyncio
    async def test_the_fingerprint_covers_the_items_both_runs_graded(self, cascade_runs):
        """The calibration items are the dataset's items the runs graded (by ``item_idx``)."""
        data = dataset()
        runs = await cascade_runs(data=data, per_criterion=None)
        larger = dataset([*ITEMS, ITEMS[0]], name="larger")
        curve = calibrate_escalation(larger, runs.dm, runs.llm, thresholds=[THRESHOLD])
        assert curve.points[0].calibration_fingerprint == fingerprint(data)


# =============================================================================
# Public surface
# =============================================================================


def test_exported_from_autorubric():
    for name in ("EscalationCurve", "calibrate_escalation"):
        assert name in autorubric.__all__
        assert getattr(autorubric, name) is getattr(autorubric.escalation, name)
