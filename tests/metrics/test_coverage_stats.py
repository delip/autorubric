"""Tests for the CoverageStats type's undefined->None rate convention.

CoverageStats records how much of the raw paired sample survived abstention/error
exclusion. Every rate is None when its denominator (the raw total) is zero; counts
stay int 0.
"""

from autorubric.metrics import CoverageStats


class TestCoverageStatsRates:
    """Coverage / abstain / error rates honour undefined->None at a zero denominator."""

    def test_all_rates_none_when_n_total_zero(self):
        cs = CoverageStats(n_total=0, n_covered=0)
        assert cs.coverage is None
        assert cs.judge_abstain_rate is None
        assert cs.gt_abstain_rate is None
        assert cs.union_exclusion_rate is None
        assert cs.error_rate is None

    def test_counts_stay_int_zero(self):
        cs = CoverageStats(n_total=0, n_covered=0)
        assert cs.n_total == 0
        assert cs.n_covered == 0
        assert cs.n_errored == 0

    def test_rates_carried_through_when_provided(self):
        # The type itself does not compute the rates (compute layer does); it must
        # faithfully carry whatever the compute layer supplies.
        cs = CoverageStats(
            n_total=10,
            n_covered=8,
            coverage=0.8,
            judge_abstain_rate=0.1,
            gt_abstain_rate=0.1,
            union_exclusion_rate=0.2,
            n_errored=1,
            error_rate=0.1,
        )
        assert cs.coverage == 0.8
        assert cs.judge_abstain_rate == 0.1
        assert cs.gt_abstain_rate == 0.1
        assert cs.union_exclusion_rate == 0.2
        assert cs.n_errored == 1
        assert cs.error_rate == 0.1


class TestCoverageStatsFrozen:
    """CoverageStats is frozen (immutable)."""

    def test_cannot_mutate(self):
        import pytest
        from pydantic import ValidationError

        cs = CoverageStats(n_total=4, n_covered=4)
        with pytest.raises(ValidationError):
            cs.n_total = 5  # type: ignore[misc]
