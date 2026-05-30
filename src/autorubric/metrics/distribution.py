"""Distribution comparison metrics.

This module provides metrics for comparing score distributions,
detecting systematic bias, and analyzing score distribution characteristics.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ._types import (
    BiasResult,
    ConfidenceInterval,
    DistributionResult,
    EMDResult,
    KSTestResult,
)


def _to_array(x: ArrayLike) -> np.ndarray:
    """Convert input to numpy array."""
    return np.asarray(x)


def _validate_same_length(x: np.ndarray, y: np.ndarray) -> None:
    """Validate that arrays have the same length."""
    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length, got {len(x)} and {len(y)}")


# =============================================================================
# Distribution Distance Metrics
# =============================================================================


def earth_movers_distance(
    dist1: ArrayLike,
    dist2: ArrayLike,
    *,
    normalize: bool = True,
) -> EMDResult:
    """Compute Earth Mover's Distance (Wasserstein distance) between two distributions.

    EMD measures the minimum "work" required to transform one distribution
    into another. Unlike correlation, it captures both shift (systematic bias)
    and shape differences (variance, skew).

    Args:
        dist1: First set of values (e.g., LLM scores).
        dist2: Second set of values (e.g., human scores).
        normalize: If True, normalize both distributions to [0, 1] before
            computing EMD. This makes EMD comparable across different scales.

    Returns:
        EMDResult with EMD value and interpretive statistics.

    Interpretation:
        - EMD = 0: Identical distributions
        - EMD < 0.05: Very similar distributions
        - EMD 0.05-0.10: Minor distributional differences
        - EMD 0.10-0.20: Moderate differences (may need attention)
        - EMD > 0.20: Substantial differences (likely systematic bias)

    Example:
        >>> result = earth_movers_distance([0.8, 0.7, 0.9], [0.7, 0.6, 0.8])
        >>> result.emd
        0.1
    """
    d1 = _to_array(dist1).astype(float)
    d2 = _to_array(dist2).astype(float)

    if len(d1) == 0 or len(d2) == 0:
        # No data to transport ⇒ every distance/diff statistic is genuinely undefined.
        return EMDResult(
            emd=None,
            mean_diff=None,
            std_diff=None,
            bias_direction="none",
            bias_magnitude=None,
            interpretation="insufficient data",
        )

    # Normalize if requested
    if normalize:
        all_vals = np.concatenate([d1, d2])
        min_val = all_vals.min()
        max_val = all_vals.max()
        if max_val > min_val:
            d1 = (d1 - min_val) / (max_val - min_val)
            d2 = (d2 - min_val) / (max_val - min_val)

    # Compute EMD using scipy
    emd = float(stats.wasserstein_distance(d1, d2))

    # Compute statistics
    mean1, mean2 = np.mean(d1), np.mean(d2)
    std1, std2 = np.std(d1), np.std(d2)
    mean_diff = float(mean1 - mean2)
    std_diff = float(std1 - std2)
    bias_magnitude = abs(mean_diff)

    # Determine bias direction
    if mean_diff > 0.01:
        bias_direction: Literal["higher", "lower", "none"] = "higher"
    elif mean_diff < -0.01:
        bias_direction = "lower"
    else:
        bias_direction = "none"

    return EMDResult(
        emd=emd,
        mean_diff=mean_diff,
        std_diff=std_diff,
        bias_direction=bias_direction,
        bias_magnitude=bias_magnitude,
        interpretation=EMDResult.interpret_emd(emd),
    )


def wasserstein_distance(
    dist1: ArrayLike,
    dist2: ArrayLike,
    *,
    normalize: bool = True,
) -> float | None:
    """Compute Wasserstein distance (alias for EMD).

    This is a convenience function that returns just the distance value.

    Args:
        dist1: First set of values.
        dist2: Second set of values.
        normalize: If True, normalize both distributions to [0, 1].

    Returns:
        Wasserstein distance value, or ``None`` when either distribution is empty (the
        distance is genuinely undefined with no data to transport).
    """
    result = earth_movers_distance(dist1, dist2, normalize=normalize)
    return result.emd


def ks_test(
    sample1: ArrayLike,
    sample2: ArrayLike | None = None,
) -> KSTestResult:
    """Kolmogorov-Smirnov test comparing two samples (or one sample to normal).

    The KS test measures the maximum difference between cumulative distribution
    functions. It tests whether two samples come from the same distribution.

    Args:
        sample1: First sample of values.
        sample2: Second sample. If None, tests against normal distribution.

    Returns:
        KSTestResult with test statistic and p-value.

    Example:
        >>> result = ks_test([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        >>> result.is_significant
        False
    """
    s1 = _to_array(sample1).astype(float)

    if sample2 is None:
        # One-sample test against normal distribution
        stat, p_value = stats.kstest(s1, "norm", args=(np.mean(s1), np.std(s1)))
    else:
        # Two-sample test
        s2 = _to_array(sample2).astype(float)
        stat, p_value = stats.ks_2samp(s1, s2)

    return KSTestResult(
        statistic=float(stat),
        p_value=float(p_value),
        is_significant=p_value < 0.05,
    )


# =============================================================================
# Bias Detection
# =============================================================================


def systematic_bias(
    y_pred: ArrayLike,
    y_true: ArrayLike,
    *,
    paired: bool = True,
    confidence: float = 0.95,
) -> BiasResult:
    """Detect and quantify systematic bias between predictions and ground truth.

    Systematic bias occurs when one set consistently scores higher or lower
    than another, independent of the item being rated.

    Args:
        y_pred: Predicted values (e.g., LLM scores).
        y_true: Ground truth values (e.g., human scores).
        paired: If True, assumes values are paired (same items).
            Uses paired t-test. If False, uses independent t-test.
        confidence: Confidence level for interval estimation.

    Returns:
        BiasResult with bias magnitude, direction, and statistical tests.

    Example:
        >>> result = systematic_bias([0.8, 0.7, 0.9], [0.7, 0.6, 0.8])
        >>> result.mean_bias
        0.1
        >>> result.direction
        'positive'
    """
    y_pred = _to_array(y_pred).astype(float)
    y_true = _to_array(y_true).astype(float)

    n = len(y_pred)
    if n == 0:
        # Nothing to compare ⇒ mean_bias is genuinely undefined (not 0.0).
        return BiasResult(
            mean_bias=None,
            std_bias=None,
            is_significant=False,
            p_value=None,
            direction="none",
            effect_size=None,
            ci=None,
            n_samples=0,
        )

    if n == 1:
        # mean_bias IS computable from a single pair (the one difference); the
        # dispersion-dependent statistics (std_bias, Cohen's d, CI, p-value) are not.
        if paired:
            _validate_same_length(y_pred, y_true)
            single = float(y_pred[0] - y_true[0])
        else:
            single = float(np.mean(y_pred) - np.mean(y_true))

        single_direction: Literal["positive", "negative", "none"]
        if single > 0.001:
            single_direction = "positive"
        elif single < -0.001:
            single_direction = "negative"
        else:
            single_direction = "none"

        return BiasResult(
            mean_bias=single,
            std_bias=None,
            is_significant=False,
            p_value=None,
            direction=single_direction,
            effect_size=None,
            ci=None,
            n_samples=1,
        )

    if paired:
        _validate_same_length(y_pred, y_true)
        differences = y_pred - y_true
        mean_bias = float(np.mean(differences))
        std_bias = float(np.std(differences, ddof=1))

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(y_pred, y_true)

        # Cohen's d for paired samples (undefined when std_bias == 0).
        effect_size = mean_bias / std_bias if std_bias > 0 else None
    else:
        mean_bias = float(np.mean(y_pred) - np.mean(y_true))

        # Pooled standard deviation
        var_pred = np.var(y_pred, ddof=1)
        var_true = np.var(y_true, ddof=1)
        pooled_std = np.sqrt((var_pred + var_true) / 2)
        std_bias = float(pooled_std)

        # Independent t-test
        t_stat, p_value = stats.ttest_ind(y_pred, y_true)

        # Cohen's d (undefined when pooled std == 0).
        effect_size = mean_bias / std_bias if std_bias > 0 else None

    # Determine direction
    if mean_bias > 0.001:
        direction: Literal["positive", "negative", "none"] = "positive"
    elif mean_bias < -0.001:
        direction = "negative"
    else:
        direction = "none"

    # Confidence interval for mean bias. The t critical value is undefined when the
    # degrees of freedom drop below 1 (e.g. n=2 unpaired → df=0 → stats.t.ppf returns
    # NaN); the interval is then genuinely undefined → ci=None, never a NaN-valued CI.
    df = (n - 1) if paired else (n - 2)
    if df < 1:
        ci = None
    else:
        if paired:
            se = std_bias / np.sqrt(n)
        else:
            se = std_bias * np.sqrt(1 / len(y_pred) + 1 / len(y_true))

        t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df)
        ci = ConfidenceInterval(
            lower=float(mean_bias - t_crit * se),
            upper=float(mean_bias + t_crit * se),
            confidence=confidence,
            method="t",
        )

    return BiasResult(
        mean_bias=mean_bias,
        std_bias=std_bias,
        is_significant=p_value < 0.05,
        p_value=float(p_value),
        direction=direction,
        effect_size=float(effect_size) if effect_size is not None else None,
        ci=ci,
        n_samples=n,
    )


# =============================================================================
# Distribution Statistics
# =============================================================================


def score_distribution(
    scores: ArrayLike,
    *,
    bins: int | Sequence[float] = 10,
    include_histogram: bool = True,
) -> DistributionResult:
    """Compute descriptive statistics for a score distribution.

    Args:
        scores: Sequence of scores to analyze.
        bins: Number of bins or explicit bin edges for histogram.
        include_histogram: If True, include histogram counts and edges.

    Returns:
        DistributionResult with summary statistics and optional histogram.

    Example:
        >>> result = score_distribution([0.1, 0.5, 0.8, 0.9])
        >>> 0.5 < result.mean < 0.6
        True
    """
    scores = _to_array(scores).astype(float)

    n = len(scores)
    if n == 0:
        # No data ⇒ every descriptive statistic is genuinely undefined (not 0.0).
        return DistributionResult(
            n=0,
            mean=None,
            std=None,
            variance=None,
            min=None,
            max=None,
            median=None,
            q25=None,
            q75=None,
            iqr=None,
            skewness=None,
            kurtosis=None,
            histogram=None,
        )

    # Basic statistics. A single point still has a defined mean/min/max/median/quartiles
    # (and iqr = q75 - q25 = 0.0, the true IQR of one point). Dispersion- and
    # shape-dependent stats need more points and are None below their thresholds:
    #   std/variance need >=2, skewness needs >=3, kurtosis needs >=4.
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if n > 1 else None
    variance = float(np.var(scores, ddof=1)) if n > 1 else None
    min_val = float(np.min(scores))
    max_val = float(np.max(scores))
    median = float(np.median(scores))
    q25 = float(np.percentile(scores, 25))
    q75 = float(np.percentile(scores, 75))
    iqr = q75 - q25

    # Skewness and kurtosis
    skewness = float(stats.skew(scores)) if n > 2 else None
    kurtosis = float(stats.kurtosis(scores)) if n > 3 else None

    # Histogram
    histogram = None
    if include_histogram:
        counts, edges = np.histogram(scores, bins=bins)
        histogram = (counts.tolist(), edges.tolist())

    return DistributionResult(
        n=n,
        mean=mean,
        std=std,
        variance=variance,
        min=min_val,
        max=max_val,
        median=median,
        q25=q25,
        q75=q75,
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurtosis,
        histogram=histogram,
    )
