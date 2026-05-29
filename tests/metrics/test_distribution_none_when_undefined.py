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


def test_score_distribution_empty_all_none():
    """n == 0: every stat field is None (no fake 0.0); n is the real count 0."""
    result = score_distribution([])
    assert result.n == 0
    assert result.mean is None
    assert result.std is None
    assert result.variance is None
    assert result.min is None
    assert result.max is None
    assert result.median is None
    assert result.q25 is None
    assert result.q75 is None
    assert result.iqr is None
    assert result.skewness is None
    assert result.kurtosis is None
    assert result.histogram is None


def test_score_distribution_single_point():
    """n == 1: mean/min/max/median defined; iqr is the true 0.0 of one point;
    std/variance/skewness/kurtosis undefined → None."""
    result = score_distribution([0.5])
    assert result.n == 1
    assert result.mean == pytest.approx(0.5)
    assert result.min == pytest.approx(0.5)
    assert result.max == pytest.approx(0.5)
    assert result.median == pytest.approx(0.5)
    assert result.q25 == pytest.approx(0.5)
    assert result.q75 == pytest.approx(0.5)
    # IQR of a single point is genuinely 0.0 (q75 - q25), not undefined.
    assert result.iqr == pytest.approx(0.0)
    # These need >=2 / >=3 / >=4 samples respectively.
    assert result.std is None
    assert result.variance is None
    assert result.skewness is None
    assert result.kurtosis is None


def test_score_distribution_two_points():
    """n == 2: std/variance computable; skewness needs >=3, kurtosis >=4 → None."""
    result = score_distribution([0.2, 0.8])
    assert result.n == 2
    assert result.std is not None
    assert result.variance is not None
    assert isinstance(result.std, float)
    assert isinstance(result.variance, float)
    assert result.std > 0.0
    assert result.skewness is None
    assert result.kurtosis is None


def test_score_distribution_three_points():
    """n == 3: skewness computable; kurtosis needs >=4 → None."""
    result = score_distribution([0.1, 0.2, 0.3])
    assert result.n == 3
    assert result.skewness is not None
    assert isinstance(result.skewness, float)
    assert result.kurtosis is None


def test_score_distribution_four_points_kurtosis_defined():
    """n == 4: kurtosis is now computable (sanity that the threshold is >3, not >4)."""
    result = score_distribution([0.1, 0.2, 0.3, 0.4])
    assert result.n == 4
    assert result.skewness is not None
    assert result.kurtosis is not None
    assert isinstance(result.kurtosis, float)


# =============================================================================
# earth_movers_distance — None for an empty distribution
# =============================================================================


def test_emd_empty_inputs_none():
    """Empty input: emd / mean_diff / std_diff / bias_magnitude are None (no fake 0.0);
    bias_direction stays 'none' and interpretation 'insufficient data'."""
    result = earth_movers_distance([], [])
    assert result.emd is None
    assert result.mean_diff is None
    assert result.std_diff is None
    assert result.bias_magnitude is None
    assert result.bias_direction == "none"
    assert result.interpretation == "insufficient data"


def test_emd_one_side_empty_none():
    """One empty side also yields the None early-return."""
    result = earth_movers_distance([0.5, 0.6], [])
    assert result.emd is None
    assert result.mean_diff is None
    assert result.std_diff is None
    assert result.bias_magnitude is None


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


def test_systematic_bias_single_pair():
    """n == 1: mean_bias IS computable (the single difference); std_bias / effect_size /
    ci / p_value undefined → None; direction from the sign of the difference."""
    result = systematic_bias([0.8], [0.6], paired=True)
    assert result.mean_bias == pytest.approx(0.2)
    assert result.std_bias is None
    assert result.effect_size is None
    assert result.ci is None
    assert result.p_value is None
    assert result.is_significant is False
    assert result.direction == "positive"
    assert result.n_samples == 1


def test_systematic_bias_single_pair_negative_direction():
    """n == 1 with pred < true → negative direction; mean_bias is the single diff."""
    result = systematic_bias([0.4], [0.6], paired=True)
    assert result.mean_bias == pytest.approx(-0.2)
    assert result.direction == "negative"
    assert result.std_bias is None


def test_systematic_bias_single_pair_unpaired():
    """n == 1 unpaired: mean_bias = mean(pred) - mean(true); same None semantics."""
    result = systematic_bias([0.8], [0.5], paired=False)
    assert result.mean_bias == pytest.approx(0.3)
    assert result.std_bias is None
    assert result.effect_size is None
    assert result.direction == "positive"
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


def test_systematic_bias_zero_variance_effect_size_none():
    """n >= 2 paired with constant nonzero differences → std_bias == 0 ⇒ Cohen's d
    undefined → effect_size None (Issue #5/B15). mean_bias / std_bias themselves stay
    real (computable). Inputs are bit-exact (integer-valued floats) so the element-wise
    differences are identical and np.std is genuinely 0.0 (no FP residue)."""
    # diffs = [2.0, 2.0] → std exactly 0
    result = systematic_bias([3.0, 4.0], [1.0, 2.0], paired=True)
    assert result.mean_bias == pytest.approx(2.0)
    assert result.std_bias == pytest.approx(0.0)
    assert result.effect_size is None


def test_systematic_bias_zero_variance_unpaired_effect_size_none():
    """Unpaired analog: both groups constant → pooled std 0 ⇒ effect_size None."""
    result = systematic_bias([5.0, 5.0], [3.0, 3.0], paired=False)
    assert result.std_bias == pytest.approx(0.0)
    assert result.effect_size is None


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
