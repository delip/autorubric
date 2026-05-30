"""Tests for the new MetricsResult type surface added in the metrics-consistency work.

Covers:
- ``AgreementSummary`` is gone (not importable, not in ``__all__``).
- ``ConfusionMatrix`` / ``CoverageStats`` are exported.
- ``MetricsResult`` carries the frozen handling-mode fields and new aggregate fields with
  legacy-tolerant defaults, and round-trips them through ``model_dump_json``.
"""

import pytest

import autorubric.metrics as metrics_pkg
from autorubric.metrics import (
    BiasResult,
    ConfusionMatrix,
    CorrelationResult,
    CoverageStats,
    MetricsResult,
)


class TestAgreementSummaryRemoved:
    """The dead ``AgreementSummary`` class is deleted everywhere."""

    def test_not_importable_from_package(self):
        with pytest.raises(ImportError):
            from autorubric.metrics import AgreementSummary  # noqa: F401

    def test_not_in_dunder_all(self):
        assert "AgreementSummary" not in metrics_pkg.__all__

    def test_attribute_absent(self):
        assert not hasattr(metrics_pkg, "AgreementSummary")


class TestNewTypesExported:
    """The new typed diagnostics are part of the public surface."""

    def test_confusion_matrix_exported(self):
        assert "ConfusionMatrix" in metrics_pkg.__all__
        assert metrics_pkg.ConfusionMatrix is ConfusionMatrix

    def test_coverage_stats_exported(self):
        assert "CoverageStats" in metrics_pkg.__all__
        assert metrics_pkg.CoverageStats is CoverageStats


def _minimal_metrics_result(**overrides) -> MetricsResult:
    """Build a MetricsResult with only the required fields, accepting overrides."""

    def _corr() -> CorrelationResult:
        return CorrelationResult(
            coefficient=None,
            p_value=None,
            interpretation="insufficient data",
            n_samples=0,
            method="spearman",
        )

    kwargs = dict(
        criterion_accuracy=None,
        criterion_precision=None,
        criterion_recall=None,
        criterion_f1=None,
        mean_kappa=None,
        per_criterion=[],
        score_rmse=0.0,
        score_mae=0.0,
        score_spearman=_corr(),
        score_kendall=_corr(),
        score_pearson=_corr(),
        bias=BiasResult(
            mean_bias=None,
            std_bias=None,
            is_significant=False,
            direction="none",
            n_samples=0,
        ),
        n_items=0,
        n_criteria=0,
    )
    kwargs.update(overrides)
    return MetricsResult(**kwargs)


class TestMetricsResultHandlingModeDefaults:
    """The handling-mode fields exist, default for legacy checkpoints, and are frozen."""

    def test_defaults(self):
        m = _minimal_metrics_result()
        assert m.cannot_assess_mode == "exclude"
        assert m.na_mode == "exclude"
        assert m.n_samples is None
        assert m.mean_krippendorff_alpha is None
        assert m.criterion_phi is None
        assert m.macro_accuracy is None
        assert m.micro_kappa is None
        assert m.coverage_stats is None

    def test_modes_settable(self):
        m = _minimal_metrics_result(cannot_assess_mode="as_category", na_mode="as_unmet")
        assert m.cannot_assess_mode == "as_category"
        assert m.na_mode == "as_unmet"

    def test_frozen(self):
        from pydantic import ValidationError

        m = _minimal_metrics_result()
        with pytest.raises(ValidationError):
            m.cannot_assess_mode = "as_unmet"  # type: ignore[misc]


class TestMetricsResultRoundTrip:
    """New fields survive a JSON round-trip (additive, non-breaking serialization)."""

    def test_round_trip_preserves_modes_and_new_fields(self):
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
        m = _minimal_metrics_result(
            cannot_assess_mode="as_unmet",
            na_mode="as_category",
            n_samples=42,
            mean_krippendorff_alpha=0.5,
            criterion_phi=0.4,
            macro_accuracy=0.9,
            micro_kappa=0.3,
            coverage_stats=cs,
        )
        restored = MetricsResult.model_validate_json(m.model_dump_json())
        assert restored.cannot_assess_mode == "as_unmet"
        assert restored.na_mode == "as_category"
        assert restored.n_samples == 42
        assert restored.mean_krippendorff_alpha == 0.5
        assert restored.criterion_phi == 0.4
        assert restored.macro_accuracy == 0.9
        assert restored.micro_kappa == 0.3
        assert restored.coverage_stats is not None
        assert restored.coverage_stats.coverage == 0.8
        assert restored.coverage_stats.n_errored == 1
