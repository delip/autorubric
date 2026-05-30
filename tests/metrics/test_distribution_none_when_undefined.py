"""Tests for the "statistic is None when genuinely undefined" principle in the
distribution / bias utilities (Issue #4: distribution stats; Issue #5: bias).

A statistic is ``None`` when genuinely undefined for the sample size, NEVER a fake
``0.0``. A value that is actually computable from the data (e.g. ``mean_bias`` at n=1,
or the IQR of a single point = 0.0) MUST be computed, not zeroed. Counts (``n``,
``n_samples``) stay real.
"""

import pytest

from autorubric.metrics._types import BiasResult, _fmt_opt
from autorubric.metrics.distribution import (
    earth_movers_distance,
    score_distribution,
    systematic_bias,
)

# =============================================================================
# score_distribution — None for stats undefined at the given sample size
# =============================================================================


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        # n == 0: every stat field is None (no fake 0.0); n is the real count 0.
        pytest.param(
            [],
            {
                "n": 0,
                "mean": None,
                "std": None,
                "variance": None,
                "min": None,
                "max": None,
                "median": None,
                "q25": None,
                "q75": None,
                "iqr": None,
                "skewness": None,
                "kurtosis": None,
                "histogram": None,
            },
            id="n0_all_none",
        ),
        # n == 1: mean/min/max/median/q25/q75 defined; iqr is the true 0.0 of one point
        # (q75 - q25), not undefined; std/variance/skewness/kurtosis need >=2/>=3/>=4
        # samples respectively → None.
        pytest.param(
            [0.5],
            {
                "n": 1,
                "mean": pytest.approx(0.5),
                "min": pytest.approx(0.5),
                "max": pytest.approx(0.5),
                "median": pytest.approx(0.5),
                "q25": pytest.approx(0.5),
                "q75": pytest.approx(0.5),
                "iqr": pytest.approx(0.0),
                "std": None,
                "variance": None,
                "skewness": None,
                "kurtosis": None,
            },
            id="n1_single_point",
        ),
        # n == 2: std/variance computable (std > 0); skewness needs >=3, kurtosis >=4 → None.
        pytest.param(
            [0.2, 0.8],
            {
                "n": 2,
                "std": "float_positive",
                "variance": "float",
                "skewness": None,
                "kurtosis": None,
            },
            id="n2_two_points",
        ),
        # n == 3: skewness computable; kurtosis needs >=4 → None.
        pytest.param(
            [0.1, 0.2, 0.3],
            {
                "n": 3,
                "skewness": "float",
                "kurtosis": None,
            },
            id="n3_three_points",
        ),
        # n == 4: kurtosis is now computable (boundary: the threshold is >3, not >4);
        # skewness stays defined (lower threshold not regressed).
        pytest.param(
            [0.1, 0.2, 0.3, 0.4],
            {
                "n": 4,
                "skewness": "float",
                "kurtosis": "float",
            },
            id="n4_kurtosis_defined",
        ),
    ],
)
def test_score_distribution_by_sample_size(data, expected):
    """A statistic is None when undefined at the given sample size, a real float when
    computable (the std>=2, skewness>=3, kurtosis>=4 thresholds; the iqr of one point is
    the true 0.0)."""
    result = score_distribution(data)
    for field, want in expected.items():
        got = getattr(result, field)
        if want is None:
            assert got is None, field
        elif want == "float":
            assert got is not None, field
            assert isinstance(got, float), field
        elif want == "float_positive":
            assert got is not None, field
            assert isinstance(got, float), field
            assert got > 0.0, field
        else:
            assert got == want, field


# =============================================================================
# earth_movers_distance — None for an empty distribution
# =============================================================================


@pytest.mark.parametrize(
    ("dist1", "dist2"),
    [
        # Both sides empty.
        ([], []),
        # One empty side also yields the None early-return.
        ([0.5, 0.6], []),
    ],
)
def test_emd_empty_inputs_none(dist1, dist2):
    """Empty input (either side): emd / mean_diff / std_diff / bias_magnitude are None
    (no fake 0.0); bias_direction stays 'none' and interpretation 'insufficient data'."""
    result = earth_movers_distance(dist1, dist2)
    assert result.emd is None
    assert result.mean_diff is None
    assert result.std_diff is None
    assert result.bias_magnitude is None
    assert result.bias_direction == "none"
    assert result.interpretation == "insufficient data"


def test_emd_nonempty_still_floats():
    """Non-empty input still produces real floats (no regression)."""
    result = earth_movers_distance([0.8, 0.7, 0.9], [0.7, 0.6, 0.8])
    assert isinstance(result.emd, float)
    assert isinstance(result.mean_diff, float)
    assert isinstance(result.std_diff, float)
    assert isinstance(result.bias_magnitude, float)


# =============================================================================
# systematic_bias — compute what's computable; None what isn't
# =============================================================================


@pytest.mark.parametrize(
    ("y_pred", "y_true", "paired", "expected_mean_bias", "expected_direction"),
    [
        # Paired positive: mean_bias is the single difference; direction from its sign.
        ([0.8], [0.6], True, pytest.approx(0.2), "positive"),
        # Paired negative: pred < true → negative direction.
        ([0.4], [0.6], True, pytest.approx(-0.2), "negative"),
        # Unpaired positive: mean_bias = mean(pred) - mean(true); same None semantics.
        ([0.8], [0.5], False, pytest.approx(0.3), "positive"),
    ],
)
def test_systematic_bias_single_pair(
    y_pred, y_true, paired, expected_mean_bias, expected_direction
):
    """n == 1: mean_bias IS computable (the single difference / mean diff); std_bias /
    effect_size / ci / p_value undefined → None; direction from the sign of the
    difference."""
    result = systematic_bias(y_pred, y_true, paired=paired)
    assert result.mean_bias == expected_mean_bias
    assert result.direction == expected_direction
    assert result.std_bias is None
    assert result.effect_size is None
    assert result.ci is None
    assert result.p_value is None
    assert result.is_significant is False
    assert result.n_samples == 1


def test_systematic_bias_empty():
    """n == 0: mean_bias and std_bias both None; direction 'none'; counts real."""
    result = systematic_bias([], [], paired=True)
    assert result.mean_bias is None
    assert result.std_bias is None
    assert result.effect_size is None
    assert result.ci is None
    assert result.p_value is None
    assert result.is_significant is False
    assert result.direction == "none"
    assert result.n_samples == 0


@pytest.mark.parametrize(
    ("y_pred", "y_true", "paired", "expected_mean_bias"),
    [
        # Paired: constant nonzero differences (diffs = [2.0, 2.0]) → std exactly 0.
        # mean_bias / std_bias themselves stay real (computable). Inputs are bit-exact
        # (integer-valued floats) so the element-wise differences are identical and
        # np.std is genuinely 0.0 (no FP residue).
        ([3.0, 4.0], [1.0, 2.0], True, pytest.approx(2.0)),
        # Unpaired analog: both groups constant → pooled std 0.
        ([5.0, 5.0], [3.0, 3.0], False, None),
    ],
)
def test_systematic_bias_zero_variance_effect_size_none(y_pred, y_true, paired, expected_mean_bias):
    """n >= 2 with std_bias == 0 ⇒ Cohen's d undefined → effect_size None (Issue #5/B15)."""
    result = systematic_bias(y_pred, y_true, paired=paired)
    assert result.std_bias == pytest.approx(0.0)
    assert result.effect_size is None
    if expected_mean_bias is not None:
        assert result.mean_bias == expected_mean_bias


def test_systematic_bias_normal_case_floats():
    """n >= 2 with real variance: every field is a real float (no regression)."""
    result = systematic_bias([0.8, 0.7, 0.9], [0.7, 0.6, 0.8], paired=True)
    assert isinstance(result.mean_bias, float)
    assert isinstance(result.std_bias, float)
    assert isinstance(result.effect_size, float)
    assert result.ci is not None
    assert result.p_value is not None
    assert result.direction == "positive"


# =============================================================================
# Rendering: a BiasResult with mean_bias=None must not crash on summary render
# =============================================================================


def test_fmt_opt_none_bias_does_not_crash():
    """The summary() bias line renders mean_bias via _fmt_opt, so a None (n=0) bias
    renders 'n/a' rather than raising on the '+.4f' format spec."""
    bias = BiasResult(
        mean_bias=None,
        std_bias=None,
        is_significant=False,
        p_value=None,
        direction="none",
        effect_size=None,
        ci=None,
        n_samples=0,
    )
    rendered = _fmt_opt(bias.mean_bias, "+.4f")
    assert rendered == "n/a"
    # And a real value still formats with sign.
    assert _fmt_opt(0.2, "+.4f") == "+0.2000"


def test_systematic_bias_unpaired_n2_ci_none_not_nan():
    """Unpaired n=2 has df = n-2 = 0, so the t critical value is undefined: the CI must be
    None, never a NaN-valued ConfidenceInterval. mean_bias stays a real (non-NaN) float."""
    import math

    result = systematic_bias([0.8, 0.6], [0.3, 0.5], paired=False)
    assert result.ci is None
    assert result.mean_bias is not None and not math.isnan(result.mean_bias)
    # n=3 unpaired (df=1) is defined → a real, non-NaN CI.
    ok = systematic_bias([0.8, 0.6, 0.7], [0.3, 0.5, 0.4], paired=False)
    assert ok.ci is not None and not math.isnan(ok.ci.lower)
