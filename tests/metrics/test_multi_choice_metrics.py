"""Tests for multi-choice (ordinal and nominal) metrics computation."""

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric
from autorubric.dataset import DataItem, RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import (
    classify_criteria,
    classify_criterion,
    compute_metrics,
    extract_all_verdicts_from_report,
    filter_na_multi_choice,
    get_option_value,
    is_na_option,
    resolve_ground_truth,
)
from autorubric.metrics._compute import (
    _compute_nominal_criterion_metrics,
    _compute_ordinal_criterion_metrics,
    _compute_per_option_metrics,
)
from autorubric.types import CriterionReport, EvaluationReport, MultiChoiceVerdict


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ordinal_criterion() -> Criterion:
    """Create an ordinal criterion (satisfaction scale)."""
    return Criterion(
        name="satisfaction",
        weight=10.0,
        requirement="How satisfied are you?",
        scale_type="ordinal",
        options=[
            {"label": "Very dissatisfied", "value": 0.0},
            {"label": "Dissatisfied", "value": 0.33},
            {"label": "Satisfied", "value": 0.67},
            {"label": "Very satisfied", "value": 1.0},
        ],
    )


@pytest.fixture
def nominal_criterion() -> Criterion:
    """Create a nominal criterion (response length)."""
    return Criterion(
        name="length",
        weight=5.0,
        requirement="Is the response length appropriate?",
        scale_type="nominal",
        options=[
            {"label": "Too brief", "value": 0.0},
            {"label": "Too verbose", "value": 0.0},
            {"label": "Just right", "value": 1.0},
        ],
    )


@pytest.fixture
def nominal_criterion_with_na() -> Criterion:
    """Create a nominal criterion with an NA option."""
    return Criterion(
        name="specificity",
        weight=6.0,
        requirement="How specific are the recommendations?",
        scale_type="ordinal",
        options=[
            {"label": "Vague", "value": 0.0},
            {"label": "Somewhat specific", "value": 0.5},
            {"label": "Very specific", "value": 1.0},
            {"label": "N/A", "value": 0.0, "na": True},
        ],
    )


@pytest.fixture
def binary_criterion() -> Criterion:
    """Create a binary criterion."""
    return Criterion(
        name="accuracy",
        weight=10.0,
        requirement="Is the response factually accurate?",
    )


@pytest.fixture
def hybrid_rubric(binary_criterion, ordinal_criterion, nominal_criterion) -> Rubric:
    """Create a rubric with mixed criterion types."""
    return Rubric([binary_criterion, ordinal_criterion, nominal_criterion])


# =============================================================================
# Test classify_criterion and classify_criteria
# =============================================================================


class TestClassifyCriterion:
    """Tests for classify_criterion helper."""

    def test_binary_criterion(self, binary_criterion):
        """Binary criteria are classified as 'binary'."""
        assert classify_criterion(binary_criterion) == "binary"

    def test_ordinal_criterion(self, ordinal_criterion):
        """Ordinal criteria are classified as 'ordinal'."""
        assert classify_criterion(ordinal_criterion) == "ordinal"

    def test_nominal_criterion(self, nominal_criterion):
        """Nominal criteria are classified as 'nominal'."""
        assert classify_criterion(nominal_criterion) == "nominal"


class TestClassifyCriteria:
    """Tests for classify_criteria helper."""

    def test_all_binary(self):
        """Classify all binary criteria."""
        criteria = [
            Criterion(weight=1.0, requirement="R1"),
            Criterion(weight=1.0, requirement="R2"),
        ]
        assert classify_criteria(criteria) == ["binary", "binary"]

    def test_mixed_types(self, binary_criterion, ordinal_criterion, nominal_criterion):
        """Classify mixed criteria types."""
        criteria = [binary_criterion, ordinal_criterion, nominal_criterion]
        types = classify_criteria(criteria)
        assert types == ["binary", "ordinal", "nominal"]


# =============================================================================
# Test resolve_ground_truth
# =============================================================================


class TestResolveGroundTruth:
    """Tests for resolve_ground_truth helper."""

    def test_binary_passthrough(self, binary_criterion):
        """Binary criteria keep CriterionVerdict unchanged."""
        ground_truth = [CriterionVerdict.MET]
        resolved = resolve_ground_truth(ground_truth, [binary_criterion])
        assert resolved == [CriterionVerdict.MET]

    def test_binary_string_to_verdict(self, binary_criterion):
        """Binary criteria accept string verdicts."""
        ground_truth = ["MET"]
        resolved = resolve_ground_truth(ground_truth, [binary_criterion])
        assert resolved == [CriterionVerdict.MET]

    def test_multi_choice_label_to_index(self, ordinal_criterion):
        """Multi-choice criteria resolve labels to indices."""
        ground_truth = ["Very satisfied"]
        resolved = resolve_ground_truth(ground_truth, [ordinal_criterion])
        assert resolved == [3]  # Index of "Very satisfied"

    def test_multi_choice_index_passthrough(self, ordinal_criterion):
        """Multi-choice criteria pass through integer indices."""
        ground_truth = [2]
        resolved = resolve_ground_truth(ground_truth, [ordinal_criterion])
        assert resolved == [2]

    def test_mixed_hybrid(self, binary_criterion, ordinal_criterion):
        """Mixed binary and multi-choice criteria."""
        criteria = [binary_criterion, ordinal_criterion]
        ground_truth = [CriterionVerdict.MET, "Satisfied"]
        resolved = resolve_ground_truth(ground_truth, criteria)
        assert resolved[0] == CriterionVerdict.MET
        assert resolved[1] == 2  # Index of "Satisfied"

    def test_invalid_binary_verdict_raises(self, binary_criterion):
        """Invalid binary verdict raises ValueError."""
        with pytest.raises(ValueError, match="Invalid binary verdict"):
            resolve_ground_truth(["INVALID"], [binary_criterion])

    def test_mismatched_length_raises(self, binary_criterion):
        """Mismatched ground truth length raises ValueError."""
        with pytest.raises(ValueError, match="doesn't match"):
            resolve_ground_truth([CriterionVerdict.MET, CriterionVerdict.UNMET], [binary_criterion])


# =============================================================================
# Test filter_na_multi_choice
# =============================================================================


class TestFilterNaMultiChoice:
    """Tests for filter_na_multi_choice helper."""

    def test_no_na_options(self, ordinal_criterion):
        """No NA options returns data unchanged."""
        pred = [0, 1, 2, 3]
        true = [1, 2, 3, 0]
        filtered_pred, filtered_true, na_agree, na_fp, na_fn = filter_na_multi_choice(
            pred, true, ordinal_criterion
        )
        assert filtered_pred == pred
        assert filtered_true == true
        assert na_agree == 0
        assert na_fp == 0
        assert na_fn == 0

    def test_exclude_na(self, nominal_criterion_with_na):
        """Exclude mode removes NA pairs."""
        pred = [0, 3, 2, 3]  # 3 is NA index
        true = [1, 3, 2, 0]  # 3 is NA index
        filtered_pred, filtered_true, na_agree, na_fp, na_fn = filter_na_multi_choice(
            pred, true, nominal_criterion_with_na, mode="exclude"
        )
        # First pair: both non-NA, keep
        # Second pair: both NA, skip (NA agreement)
        # Third pair: both non-NA, keep
        # Fourth pair: pred NA, true non-NA, skip (NA FP)
        assert len(filtered_pred) == 2
        assert len(filtered_true) == 2
        assert na_agree == 1
        assert na_fp == 1
        assert na_fn == 0


# =============================================================================
# Test get_option_value and is_na_option
# =============================================================================


class TestOptionHelpers:
    """Tests for option helper functions."""

    def test_get_option_value(self, ordinal_criterion):
        """Get option value by index."""
        assert get_option_value(ordinal_criterion, 0) == 0.0
        assert get_option_value(ordinal_criterion, 3) == 1.0

    def test_get_option_value_binary_raises(self, binary_criterion):
        """get_option_value raises for binary criteria."""
        with pytest.raises(ValueError, match="Cannot get option value for binary"):
            get_option_value(binary_criterion, 0)

    def test_is_na_option(self, nominal_criterion_with_na):
        """Check if option is NA."""
        assert is_na_option(nominal_criterion_with_na, 3) is True
        assert is_na_option(nominal_criterion_with_na, 0) is False


# =============================================================================
# Test per-criterion metric functions
# =============================================================================


class TestComputePerOptionMetrics:
    """Tests for _compute_per_option_metrics."""

    def test_perfect_predictions(self, ordinal_criterion):
        """Perfect predictions give F1=1 for all options."""
        pred = [0, 1, 2, 3, 0, 1, 2, 3]
        true = [0, 1, 2, 3, 0, 1, 2, 3]
        metrics = _compute_per_option_metrics(pred, true, ordinal_criterion)

        assert len(metrics) == 4
        for m in metrics:
            assert m.f1 == 1.0
            assert m.precision == 1.0
            assert m.recall == 1.0


class TestComputeOrdinalCriterionMetrics:
    """Tests for _compute_ordinal_criterion_metrics."""

    def test_perfect_predictions(self, ordinal_criterion):
        """Perfect predictions give exact accuracy 1.0."""
        pred = [0, 1, 2, 3]
        true = [0, 1, 2, 3]
        metrics = _compute_ordinal_criterion_metrics(pred, true, ordinal_criterion, 0)

        assert metrics.exact_accuracy == 1.0
        assert metrics.adjacent_accuracy == 1.0
        assert metrics.weighted_kappa == 1.0
        assert metrics.rmse == 0.0

    def test_adjacent_accuracy(self, ordinal_criterion):
        """Adjacent accuracy for off-by-one predictions."""
        pred = [1, 2, 3, 2]  # Off by 1 from true
        true = [0, 1, 2, 3]
        metrics = _compute_ordinal_criterion_metrics(pred, true, ordinal_criterion, 0)

        assert metrics.exact_accuracy == 0.0
        assert metrics.adjacent_accuracy == 1.0  # All within ±1

    def test_correlation(self, ordinal_criterion):
        """Ordinal metrics include correlations."""
        pred = [0, 1, 2, 3, 0, 1, 2, 3]
        true = [0, 1, 2, 3, 0, 1, 2, 3]
        metrics = _compute_ordinal_criterion_metrics(pred, true, ordinal_criterion, 0)

        assert metrics.spearman.coefficient == 1.0
        assert metrics.kendall.coefficient == 1.0


class TestComputeNominalCriterionMetrics:
    """Tests for _compute_nominal_criterion_metrics."""

    def test_perfect_predictions(self, nominal_criterion):
        """Perfect predictions give accuracy 1.0."""
        pred = [0, 1, 2, 0, 1, 2]
        true = [0, 1, 2, 0, 1, 2]
        metrics = _compute_nominal_criterion_metrics(pred, true, nominal_criterion, 0)

        assert metrics.exact_accuracy == 1.0
        assert metrics.kappa == 1.0

    def test_confusion_matrix(self, nominal_criterion):
        """Nominal metrics include confusion matrix."""
        pred = [0, 1, 2]
        true = [0, 1, 2]
        metrics = _compute_nominal_criterion_metrics(pred, true, nominal_criterion, 0)

        assert len(metrics.confusion_matrix) == 3
        assert metrics.confusion_matrix[0][0] == 1  # True 0, Pred 0


# =============================================================================
# Test compute_metrics with multi-choice
# =============================================================================


@pytest.fixture
def ordinal_dataset() -> RubricDataset:
    """Create a dataset with ordinal criteria."""
    rubric = Rubric([
        Criterion(
            name="satisfaction",
            weight=10.0,
            requirement="Satisfaction level",
            scale_type="ordinal",
            options=[
                {"label": "1", "value": 0.0},
                {"label": "2", "value": 0.33},
                {"label": "3", "value": 0.67},
                {"label": "4", "value": 1.0},
            ],
        ),
    ])
    dataset = RubricDataset(prompt="Test", rubric=rubric)
    # Add items with ground truth as string labels
    dataset.add_item(submission="A", description="D1", ground_truth=["4"])
    dataset.add_item(submission="B", description="D2", ground_truth=["3"])
    dataset.add_item(submission="C", description="D3", ground_truth=["2"])
    return dataset


@pytest.fixture
def hybrid_dataset(hybrid_rubric) -> RubricDataset:
    """Create a dataset with mixed criterion types."""
    dataset = RubricDataset(prompt="Test", rubric=hybrid_rubric)
    # Binary: MET, Ordinal: "Very satisfied" (index 3), Nominal: "Just right" (index 2)
    dataset.add_item(
        submission="A",
        description="D1",
        ground_truth=[CriterionVerdict.MET, "Very satisfied", "Just right"],
    )
    dataset.add_item(
        submission="B",
        description="D2",
        ground_truth=[CriterionVerdict.UNMET, "Dissatisfied", "Too brief"],
    )
    return dataset


def _make_ordinal_report(selected_index: int) -> EvaluationReport:
    """Create a mock EvaluationReport for ordinal criterion."""
    from autorubric.types import CriterionReport, MultiChoiceVerdict

    return EvaluationReport(
        score=0.5,
        raw_score=5.0,
        report=[
            CriterionReport(
                weight=10.0,
                requirement="Satisfaction level",
                name="satisfaction",
                verdict=CriterionVerdict.MET if selected_index >= 2 else CriterionVerdict.UNMET,
                reason="Test",
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=selected_index,
                    selected_label=str(selected_index + 1),
                    value=selected_index * 0.33,
                ),
            ),
        ],
    )


def _make_hybrid_report(
    binary_verdict: CriterionVerdict,
    ordinal_index: int,
    nominal_index: int,
) -> EvaluationReport:
    """Create a mock EvaluationReport for hybrid rubric."""
    from autorubric.types import CriterionReport, MultiChoiceVerdict

    return EvaluationReport(
        score=0.5,
        raw_score=10.0,
        report=[
            # Binary criterion
            CriterionReport(
                weight=10.0,
                requirement="Is accurate?",
                name="accuracy",
                verdict=binary_verdict,
                reason="Test",
            ),
            # Ordinal criterion
            CriterionReport(
                weight=10.0,
                requirement="Satisfaction",
                name="satisfaction",
                verdict=CriterionVerdict.MET,
                reason="Test",
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=ordinal_index,
                    selected_label="Test",
                    value=ordinal_index * 0.33,
                ),
            ),
            # Nominal criterion
            CriterionReport(
                weight=5.0,
                requirement="Length",
                name="length",
                verdict=CriterionVerdict.MET,
                reason="Test",
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=nominal_index,
                    selected_label="Test",
                    value=0.5,
                ),
            ),
        ],
    )


class TestComputeMetricsOrdinal:
    """Tests for compute_metrics with ordinal criteria."""

    def test_perfect_ordinal_predictions(self, ordinal_dataset):
        """Perfect ordinal predictions give accuracy 1.0."""
        # Create perfect predictions matching ground truth
        item_results = [
            ItemResult(
                item_idx=0,
                item=ordinal_dataset.items[0],
                report=_make_ordinal_report(3),  # Ground truth is "4" = index 3
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=ordinal_dataset.items[1],
                report=_make_ordinal_report(2),  # Ground truth is "3" = index 2
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=2,
                item=ordinal_dataset.items[2],
                report=_make_ordinal_report(1),  # Ground truth is "2" = index 1
                duration_seconds=0.1,
            ),
        ]

        from datetime import datetime

        eval_result = EvalResult(
            item_results=item_results,
            total_items=3,
            successful_items=3,
            failed_items=0,
            total_token_usage=None,
            total_completion_cost=None,
            timing_stats=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=[],
            experiment_name=None,
            experiment_dir=None,
        )

        metrics = compute_metrics(eval_result, ordinal_dataset)

        assert metrics.n_ordinal_criteria == 1
        assert metrics.n_binary_criteria == 0
        assert metrics.criterion_accuracy == 1.0
        assert len(metrics.per_criterion) == 1
        assert metrics.per_criterion[0].criterion_type == "ordinal"


class TestComputeMetricsHybrid:
    """Tests for compute_metrics with hybrid (mixed) rubrics."""

    def test_hybrid_metrics_has_all_types(self, hybrid_dataset):
        """Hybrid metrics include all criterion types."""
        item_results = [
            ItemResult(
                item_idx=0,
                item=hybrid_dataset.items[0],
                report=_make_hybrid_report(CriterionVerdict.MET, 3, 2),
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=hybrid_dataset.items[1],
                report=_make_hybrid_report(CriterionVerdict.UNMET, 1, 0),
                duration_seconds=0.1,
            ),
        ]

        from datetime import datetime

        eval_result = EvalResult(
            item_results=item_results,
            total_items=2,
            successful_items=2,
            failed_items=0,
            total_token_usage=None,
            total_completion_cost=None,
            timing_stats=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=[],
            experiment_name=None,
            experiment_dir=None,
        )

        metrics = compute_metrics(eval_result, hybrid_dataset)

        assert metrics.n_binary_criteria == 1
        assert metrics.n_ordinal_criteria == 1
        assert metrics.n_nominal_criteria == 1
        assert len(metrics.per_criterion) == 3

        # Check criterion types
        types = [cm.criterion_type for cm in metrics.per_criterion]
        assert "binary" in types
        assert "ordinal" in types
        assert "nominal" in types


class TestMetricsSummaryMultiChoice:
    """Tests for summary() method with multi-choice criteria."""

    def test_summary_shows_type_breakdown(self, hybrid_dataset):
        """Summary shows criterion type breakdown."""
        item_results = [
            ItemResult(
                item_idx=0,
                item=hybrid_dataset.items[0],
                report=_make_hybrid_report(CriterionVerdict.MET, 3, 2),
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=hybrid_dataset.items[1],
                report=_make_hybrid_report(CriterionVerdict.UNMET, 1, 0),
                duration_seconds=0.1,
            ),
        ]

        from datetime import datetime

        eval_result = EvalResult(
            item_results=item_results,
            total_items=2,
            successful_items=2,
            failed_items=0,
            total_token_usage=None,
            total_completion_cost=None,
            timing_stats=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=[],
            experiment_name=None,
            experiment_dir=None,
        )

        metrics = compute_metrics(eval_result, hybrid_dataset)
        summary = metrics.summary()

        # Check that summary contains type info
        assert "binary" in summary.lower()
        assert "ordinal" in summary.lower()
        assert "nominal" in summary.lower()


class TestBackwardsCompatibility:
    """Tests that binary-only rubrics work unchanged."""

    def test_binary_only_unchanged(self):
        """Binary-only rubrics produce identical results to before."""
        rubric = Rubric([
            Criterion(name="C1", weight=10.0, requirement="R1"),
            Criterion(name="C2", weight=5.0, requirement="R2"),
        ])
        dataset = RubricDataset(prompt="Test", rubric=rubric)
        dataset.add_item(
            submission="A",
            description="D1",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
        )
        dataset.add_item(
            submission="B",
            description="D2",
            ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET],
        )

        # Create matching predictions
        item_results = [
            ItemResult(
                item_idx=0,
                item=dataset.items[0],
                report=EvaluationReport(
                    score=1.0,
                    raw_score=15.0,
                    report=[
                        CriterionReport(
                            weight=10.0,
                            requirement="R1",
                            name="C1",
                            verdict=CriterionVerdict.MET,
                            reason="Test",
                        ),
                        CriterionReport(
                            weight=5.0,
                            requirement="R2",
                            name="C2",
                            verdict=CriterionVerdict.MET,
                            reason="Test",
                        ),
                    ],
                ),
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=dataset.items[1],
                report=EvaluationReport(
                    score=0.33,
                    raw_score=5.0,
                    report=[
                        CriterionReport(
                            weight=10.0,
                            requirement="R1",
                            name="C1",
                            verdict=CriterionVerdict.UNMET,
                            reason="Test",
                        ),
                        CriterionReport(
                            weight=5.0,
                            requirement="R2",
                            name="C2",
                            verdict=CriterionVerdict.MET,
                            reason="Test",
                        ),
                    ],
                ),
                duration_seconds=0.1,
            ),
        ]

        from datetime import datetime

        eval_result = EvalResult(
            item_results=item_results,
            total_items=2,
            successful_items=2,
            failed_items=0,
            total_token_usage=None,
            total_completion_cost=None,
            timing_stats=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=[],
            experiment_name=None,
            experiment_dir=None,
        )

        metrics = compute_metrics(eval_result, dataset)

        # Check backwards compatibility
        assert metrics.n_binary_criteria == 2
        assert metrics.n_ordinal_criteria == 0
        assert metrics.n_nominal_criteria == 0
        assert metrics.criterion_accuracy == 1.0
        assert metrics.criterion_precision > 0
        assert metrics.criterion_recall > 0
        assert metrics.criterion_f1 > 0

        # All per_criterion should be binary
        for cm in metrics.per_criterion:
            assert cm.criterion_type == "binary"
