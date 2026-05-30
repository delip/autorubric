"""Unit tests for the shared scoring core (``autorubric.scoring.score_reports``).

These tests pin the exact arithmetic of every ``CannotAssessStrategy`` across the
{binary, multi-choice} x {positive, negative weight} matrix, mirroring the live-grading
expectations in ``tests/graders/test_cannot_assess.py`` but exercising ``score_reports``
directly on hand-built ``CriterionReport`` objects (no LLM / grader involved).
"""

import pytest

from autorubric.scoring import score_reports
from autorubric.types import (
    CannotAssessConfig,
    CannotAssessStrategy,
    CriterionOption,
    CriterionReport,
    CriterionVerdict,
    MultiChoiceVerdict,
)

# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def binary_report(weight: float, verdict: CriterionVerdict) -> CriterionReport:
    """Build a binary CriterionReport with the given weight and verdict."""
    return CriterionReport(
        requirement="r",
        name="bin",
        weight=weight,
        verdict=verdict,
        reason="",
    )


# Standard ordinal scale: values 0.0 / 0.5 / 1.0 plus an NA option.
STANDARD_OPTIONS = [
    CriterionOption(label="low", value=0.0),
    CriterionOption(label="mid", value=0.5),
    CriterionOption(label="high", value=1.0),
    CriterionOption(label="na", value=0.0, na=True),
]

# Shifted ordinal scale: the lowest *scored* value is 0.3 (not 0.0). This
# distinguishes worst_scored_option() (-> 0.3) from a naive binary-style 0.0.
SHIFTED_OPTIONS = [
    CriterionOption(label="low", value=0.3),
    CriterionOption(label="mid", value=0.6),
    CriterionOption(label="high", value=1.0),
    CriterionOption(label="na", value=0.0, na=True),
]


def mc_report(
    weight: float,
    options: list[CriterionOption],
    selected_index: int,
) -> CriterionReport:
    """Build a multi-choice (ordinal) CriterionReport selecting ``selected_index``."""
    opt = options[selected_index]
    return CriterionReport(
        requirement="r",
        name="mc",
        weight=weight,
        options=options,
        scale_type="ordinal",
        multi_choice_verdict=MultiChoiceVerdict(
            selected_index=selected_index,
            selected_label=opt.label,
            value=opt.value,
            na=opt.na,
        ),
        reason="",
    )


def mc_na_report(weight: float, options: list[CriterionOption]) -> CriterionReport:
    """Build a multi-choice CriterionReport that selected the NA option."""
    na_idx = next(i for i, o in enumerate(options) if o.na)
    return mc_report(weight, options, na_idx)


SKIP = CannotAssessConfig(strategy=CannotAssessStrategy.SKIP)
ZERO = CannotAssessConfig(strategy=CannotAssessStrategy.ZERO)
FAIL = CannotAssessConfig(strategy=CannotAssessStrategy.FAIL)
PARTIAL = CannotAssessConfig(strategy=CannotAssessStrategy.PARTIAL, partial_credit=0.5)


# ===========================================================================
# Binary: genuine MET / UNMET (no abstain) -- baseline arithmetic
# ===========================================================================


@pytest.mark.parametrize(
    "reports,expected",
    [
        # +weight MET -> 1.0 (10/10)
        ([binary_report(10.0, CriterionVerdict.MET)], 1.0),
        # +weight UNMET -> 0.0
        ([binary_report(10.0, CriterionVerdict.UNMET)], 0.0),
        # +weight MET plus -weight MET (penalty applies, clamps at 0):
        # weighted_sum = 10 - 5 = 5; total_positive = 10 -> 0.5
        (
            [
                binary_report(10.0, CriterionVerdict.MET),
                binary_report(-5.0, CriterionVerdict.MET),
            ],
            0.5,
        ),
        # -weight UNMET contributes 0 (no penalty):
        # weighted_sum = 10 + 0 = 10; total_positive = 10 -> 1.0
        (
            [
                binary_report(10.0, CriterionVerdict.MET),
                binary_report(-5.0, CriterionVerdict.UNMET),
            ],
            1.0,
        ),
    ],
)
def test_binary_genuine_verdict_baseline_arithmetic(reports, expected):
    """Baseline binary genuine MET/UNMET arithmetic (no abstain) under SKIP.

    Pins the normalize numerator/denominator across the +/- weight x MET/UNMET cells.
    """
    assert score_reports(reports, SKIP) == pytest.approx(expected)


# ===========================================================================
# Binary CANNOT_ASSESS across all strategies, +/- weight
# ===========================================================================


@pytest.mark.parametrize(
    "config,expected",
    [
        # SKIP: CA(w=10) excluded from numerator AND denominator -> 5/5 = 1.0
        (SKIP, 1.0),
        # ZERO: CA contributes 0, stays in denominator -> 5 / 15
        (ZERO, 5.0 / 15.0),
        # PARTIAL(0.5): CA(+10) contributes 0.5*10=5 -> (5+5)/15
        (PARTIAL, 10.0 / 15.0),
        # FAIL: positive CA -> UNMET (0 contribution), stays in denom -> 5/15
        (FAIL, 5.0 / 15.0),
    ],
)
def test_binary_positive_cannot_assess_all_strategies(config, expected):
    """Mirror tests/graders/test_cannot_assess.py: +weight CA across strategies.

    Rubric: CA(w=10) + MET(w=5) + UNMET(w=-3).
    """
    reports = [
        binary_report(10.0, CriterionVerdict.CANNOT_ASSESS),
        binary_report(5.0, CriterionVerdict.MET),
        binary_report(-3.0, CriterionVerdict.UNMET),
    ]
    assert score_reports(reports, config) == pytest.approx(expected)


@pytest.mark.parametrize(
    "config,expected",
    [
        # FAIL: negative CA -> MET (assume the error is present), penalizing.
        # Mirrors test_fail_strategy_negative_criterion_cannot_assess:
        # MET(w=10) + CA(w=-5) under FAIL -> 10 - 5 = 5; total_positive = 10 -> 0.5.
        (FAIL, 0.5),
        # ZERO: negative CA contributes 0 (no penalty) but stays in denominator.
        # weighted_sum = 10 + 0 = 10; total_positive = 10 -> 1.0
        (ZERO, 1.0),
        # PARTIAL only awards partial credit to +weight criteria; -weight CA -> 0.
        # weighted_sum = 10 -> 1.0
        (PARTIAL, 1.0),
        # SKIP: negative CA excluded from numerator and denominator.
        # weighted_sum = 10; total_positive = 10 -> 1.0
        (SKIP, 1.0),
    ],
)
def test_binary_negative_cannot_assess_all_strategies(config, expected):
    """Mirror test_binary_positive_cannot_assess_all_strategies: -weight CA across strategies.

    Rubric: MET(w=10) + CANNOT_ASSESS(w=-5).
    """
    reports = [
        binary_report(10.0, CriterionVerdict.MET),
        binary_report(-5.0, CriterionVerdict.CANNOT_ASSESS),
    ]
    assert score_reports(reports, config) == pytest.approx(expected)


# ===========================================================================
# Multi-choice NA under FAIL -- the worst_scored_option() cases
# ===========================================================================


def test_mc_positive_na_fail_standard_scale_contributes_lowest_value():
    """+weight NA under FAIL on the standard 0/0.5/1 scale -> worst value 0.0.

    Here worst_scored_option() coincides with the binary-style 0 contribution.
    """
    reports = [mc_na_report(10.0, STANDARD_OPTIONS)]
    # NA -> worst scored option value 0.0; weighted_sum = 0; total_positive = 10 -> 0.0
    assert score_reports(reports, FAIL) == pytest.approx(0.0)


def test_mc_negative_na_fail_standard_scale_penalizes_not_zero():
    """-weight NA under FAIL -> highest value (1.0) * w, which PENALIZES.

    This is the previously-broken case: the old code left NA at value 0 (no penalty).
    With worst_scored_option(), a negative-weight NA picks the highest-value option,
    so it contributes a real penalty and the result is NOT 0 contribution.
    """
    reports = [
        mc_report(10.0, STANDARD_OPTIONS, 2),  # high -> 1.0 * 10 = 10 (assessable +)
        mc_na_report(-5.0, STANDARD_OPTIONS),  # NA -> highest value 1.0 * -5 = -5
    ]
    # weighted_sum = 10 - 5 = 5; total_positive = 10 -> 0.5 (clearly NOT 1.0)
    result = score_reports(reports, FAIL)
    assert result == pytest.approx(0.5)
    assert result != pytest.approx(1.0)

    # And isolate the contribution sign via the raw (unnormalized) sum: it is negative.
    raw = score_reports([mc_na_report(-5.0, STANDARD_OPTIONS)], FAIL, normalize=False)
    assert raw == pytest.approx(-5.0)
    assert raw != 0.0


def test_mc_positive_na_fail_shifted_scale_uses_worst_scored_option():
    """+weight NA under FAIL on shifted scale -> 0.3 * w, NOT 0.

    The shifted scale's lowest *scored* option is 0.3 (the 0.0 option is NA).
    A naive binary-style "0 for positive FAIL" would give 0.0; worst_scored_option()
    correctly gives 0.3.
    """
    raw = score_reports([mc_na_report(10.0, SHIFTED_OPTIONS)], FAIL, normalize=False)
    assert raw == pytest.approx(0.3 * 10.0)
    assert raw != 0.0


# ===========================================================================
# Multi-choice NA under ZERO / PARTIAL / SKIP
# ===========================================================================


@pytest.mark.parametrize(
    "config,expected",
    [
        # ZERO: +weight NA contributes 0 but stays in the denominator.
        # weighted_sum = 5; total_positive = 15 -> 5/15
        (ZERO, 5.0 / 15.0),
        # PARTIAL(0.5): +weight NA contributes partial_credit * w = 0.5 * 10 = 5.
        # weighted_sum = 5 + 5 = 10; total_positive = 15 -> 10/15
        (PARTIAL, 10.0 / 15.0),
        # SKIP: +weight NA excluded from numerator and denominator.
        # weighted_sum = 5; total_positive = 5 (NA's 10 excluded) -> 1.0
        (SKIP, 1.0),
    ],
)
def test_mc_positive_na_zero_partial_skip_strategies(config, expected):
    """+weight multi-choice NA across ZERO / PARTIAL / SKIP.

    Rubric: high-MC(w=5, value 1.0) + NA-MC(w=10).
    """
    reports = [
        mc_report(5.0, STANDARD_OPTIONS, 2),  # high -> 1.0 * 5 = 5
        mc_na_report(10.0, STANDARD_OPTIONS),  # NA
    ]
    assert score_reports(reports, config) == pytest.approx(expected)


def test_mc_negative_na_partial_no_credit():
    """-weight NA under PARTIAL gets no partial credit (0 contribution)."""
    reports = [
        mc_report(10.0, STANDARD_OPTIONS, 2),  # high -> 10
        mc_na_report(-5.0, STANDARD_OPTIONS),  # NA -> 0 (no partial for -weight)
    ]
    # weighted_sum = 10; total_positive = 10 -> 1.0
    assert score_reports(reports, PARTIAL) == pytest.approx(1.0)


# ===========================================================================
# SKIP denominator correctness (the Rubric bug being fixed at the core level)
# ===========================================================================


def test_skip_denominator_excludes_na_positive_weight():
    """SKIP must not count an NA criterion's weight in the denominator.

    NA(w=10) + MET(w=5) + UNMET(w=3) under SKIP -> 5/8 = 0.625 (NOT 0.0, NOT 5/18).
    This is exactly the dual-denominator bug the unified core fixes.
    """
    reports = [
        binary_report(10.0, CriterionVerdict.CANNOT_ASSESS),
        binary_report(5.0, CriterionVerdict.MET),
        binary_report(3.0, CriterionVerdict.UNMET),
    ]
    result = score_reports(reports, SKIP)
    assert result == pytest.approx(5.0 / 8.0)
    assert result == pytest.approx(0.625)
    assert result != pytest.approx(0.0)


def test_skip_all_na_returns_zero():
    """SKIP with every criterion abstaining -> no positive weight -> 0.0."""
    reports = [
        binary_report(10.0, CriterionVerdict.CANNOT_ASSESS),
        binary_report(5.0, CriterionVerdict.CANNOT_ASSESS),
    ]
    assert score_reports(reports, SKIP) == pytest.approx(0.0)


def test_skip_mixed_binary_and_mc_na_denominator():
    """SKIP excludes both binary CA and multi-choice NA weights from the denominator."""
    reports = [
        binary_report(10.0, CriterionVerdict.CANNOT_ASSESS),  # excluded
        mc_na_report(7.0, STANDARD_OPTIONS),  # excluded
        binary_report(5.0, CriterionVerdict.MET),  # 5
        mc_report(3.0, STANDARD_OPTIONS, 2),  # high -> 3
    ]
    # weighted_sum = 5 + 3 = 8; total_positive = 5 + 3 = 8 -> 1.0
    assert score_reports(reports, SKIP) == pytest.approx(1.0)


# ===========================================================================
# normalize=False -- raw weighted sum
# ===========================================================================


def test_normalize_false_returns_raw_weighted_sum():
    """normalize=False returns the raw weighted sum, unclamped."""
    reports = [
        binary_report(10.0, CriterionVerdict.MET),  # +10
        binary_report(5.0, CriterionVerdict.UNMET),  # 0
        binary_report(-3.0, CriterionVerdict.MET),  # -3
    ]
    # raw = 10 + 0 - 3 = 7 (no clamping, no division)
    assert score_reports(reports, SKIP, normalize=False) == pytest.approx(7.0)


def test_normalize_false_partial_includes_partial_contribution():
    """normalize=False with PARTIAL includes the partial-credit contribution in the sum."""
    reports = [
        binary_report(10.0, CriterionVerdict.CANNOT_ASSESS),  # 0.5 * 10 = 5
        binary_report(5.0, CriterionVerdict.MET),  # 5
        binary_report(-3.0, CriterionVerdict.UNMET),  # 0
    ]
    # raw = 5 + 5 + 0 = 10 (mirrors test_raw_score_with_cannot_assess)
    assert score_reports(reports, PARTIAL, normalize=False) == pytest.approx(10.0)


# ===========================================================================
# negative-weight-only normalize fallback (1 + sum/neg_weight)
# ===========================================================================


def test_negative_weight_only_fallback_unmet_full_score():
    """No positive weight, all -weight UNMET -> 1 + 0/neg = 1.0."""
    reports = [
        binary_report(-5.0, CriterionVerdict.UNMET),
        binary_report(-3.0, CriterionVerdict.UNMET),
    ]
    # weighted_sum = 0; total_negative = 8 -> 1 + 0/8 = 1.0
    assert score_reports(reports, SKIP) == pytest.approx(1.0)


def test_negative_weight_only_fallback_one_met():
    """No positive weight; one -weight MET subtracts -> 1 + (-5)/8 = 0.375."""
    reports = [
        binary_report(-5.0, CriterionVerdict.MET),  # -5
        binary_report(-3.0, CriterionVerdict.UNMET),  # 0
    ]
    # weighted_sum = -5; total_negative = 8 -> 1 + (-5/8) = 0.375
    assert score_reports(reports, SKIP) == pytest.approx(0.375)


def test_negative_weight_only_fallback_clamps_at_zero():
    """No positive weight, all -weight MET -> 1 + (-8)/8 = 0.0 (clamped)."""
    reports = [
        binary_report(-5.0, CriterionVerdict.MET),
        binary_report(-3.0, CriterionVerdict.MET),
    ]
    # weighted_sum = -8; total_negative = 8 -> 1 + (-1) = 0.0
    assert score_reports(reports, SKIP) == pytest.approx(0.0)


def test_no_weight_at_all_returns_zero():
    """Empty reports / all-abstained -> 0.0 (no positive or negative weight)."""
    assert score_reports([], SKIP) == pytest.approx(0.0)
