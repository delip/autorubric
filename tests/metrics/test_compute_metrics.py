"""Tests for compute_metrics function."""

import pytest
from unittest.mock import MagicMock

from autorubric.dataset import DataItem, RubricDataset
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    CriterionVerdict,
    CriterionReport,
    EvaluationReport,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
)
from autorubric.metrics import compute_metrics, MetricsResult
from datetime import datetime


def create_mock_dataset():
    """Create a simple mock dataset for testing."""
    rubric = Rubric([
        Criterion(name="Accuracy", weight=10.0, requirement="Be accurate"),
        Criterion(name="Clarity", weight=5.0, requirement="Be clear"),
    ])

    dataset = RubricDataset(
        prompt="Test prompt",
        rubric=rubric,
        name="test",
    )

    # Add items with ground truth
    dataset.add_item(
        submission="Response 1",
        description="Item 1",
        ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
    )
    dataset.add_item(
        submission="Response 2",
        description="Item 2",
        ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
    )
    dataset.add_item(
        submission="Response 3",
        description="Item 3",
        ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET],
    )
    dataset.add_item(
        submission="Response 4",
        description="Item 4",
        ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.UNMET],
    )

    return dataset


def create_mock_eval_result(dataset: RubricDataset, predictions: list[list[CriterionVerdict]]):
    """Create a mock EvalResult from predictions."""
    item_results = []

    for idx, pred_verdicts in enumerate(predictions):
        # Calculate score based on verdicts
        score = 0.0
        for c_idx, verdict in enumerate(pred_verdicts):
            if verdict == CriterionVerdict.MET:
                score += dataset.rubric.rubric[c_idx].weight
        score = score / dataset.total_positive_weight

        report = EvaluationReport(
            score=score,
            raw_score=score * dataset.total_positive_weight,
            report=[
                CriterionReport(
                    weight=dataset.rubric.rubric[i].weight,
                    requirement=dataset.rubric.rubric[i].requirement,
                    verdict=v,
                    reason="Test reason",
                )
                for i, v in enumerate(pred_verdicts)
            ],
        )

        item_results.append(ItemResult(
            item_idx=idx,
            item=dataset.items[idx],
            report=report,
            duration_seconds=0.5,
        ))

    return EvalResult(
        item_results=item_results,
        total_items=len(predictions),
        successful_items=len(predictions),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=EvalTimingStats(
            total_duration_seconds=2.0,
            mean_item_duration_seconds=0.5,
            min_item_duration_seconds=0.4,
            max_item_duration_seconds=0.6,
            p50_item_duration_seconds=0.5,
            p95_item_duration_seconds=0.55,
            items_per_second=2.0,
        ),
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


class TestComputeMetricsPerfect:
    """Test compute_metrics with perfect predictions."""

    def test_perfect_predictions(self):
        dataset = create_mock_dataset()

        # Perfect predictions match ground truth
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]

        eval_result = create_mock_eval_result(dataset, predictions)
        metrics = compute_metrics(eval_result, dataset)

        assert metrics.criterion_accuracy == 1.0
        assert metrics.score_rmse == 0.0
        assert metrics.n_items == 4
        assert metrics.n_criteria == 2

    def test_per_criterion_perfect(self):
        dataset = create_mock_dataset()

        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]

        eval_result = create_mock_eval_result(dataset, predictions)
        metrics = compute_metrics(eval_result, dataset)

        assert len(metrics.per_criterion) == 2
        assert metrics.per_criterion[0].name == "Accuracy"
        assert metrics.per_criterion[0].accuracy == 1.0
        assert metrics.per_criterion[1].name == "Clarity"
        assert metrics.per_criterion[1].accuracy == 1.0


class TestComputeMetricsImperfect:
    """Test compute_metrics with imperfect predictions."""

    def test_half_correct(self):
        dataset = create_mock_dataset()

        # All predictions are MET - half are wrong
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],  # 2/2 correct
            [CriterionVerdict.MET, CriterionVerdict.MET],  # 1/2 correct
            [CriterionVerdict.MET, CriterionVerdict.MET],  # 1/2 correct
            [CriterionVerdict.MET, CriterionVerdict.MET],  # 0/2 correct
        ]

        eval_result = create_mock_eval_result(dataset, predictions)
        metrics = compute_metrics(eval_result, dataset)

        # 4 out of 8 correct = 50%
        assert metrics.criterion_accuracy == 0.5

    def test_bias_detection(self):
        dataset = create_mock_dataset()

        # All predictions are MET - consistently overestimates
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.MET],
        ]

        eval_result = create_mock_eval_result(dataset, predictions)
        metrics = compute_metrics(eval_result, dataset)

        # Should detect positive bias (predicting higher scores)
        assert metrics.bias.mean_bias > 0
        assert metrics.bias.direction == "positive"


class TestComputeMetricsOptions:
    """Test compute_metrics options."""

    def test_bootstrap_disabled(self):
        dataset = create_mock_dataset()
        predictions = [[CriterionVerdict.MET, CriterionVerdict.MET]] * 4
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(eval_result, dataset, bootstrap=False)

        assert metrics.bootstrap is None

    def test_bootstrap_enabled(self):
        dataset = create_mock_dataset()
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(
            eval_result, dataset, bootstrap=True, n_bootstrap=100, seed=42
        )

        assert metrics.bootstrap is not None
        assert metrics.bootstrap.n_bootstrap == 100
        assert metrics.bootstrap.accuracy_ci[0] <= metrics.criterion_accuracy
        assert metrics.bootstrap.accuracy_ci[1] >= metrics.criterion_accuracy


class TestComputeMetricsEdgeCases:
    """Test edge cases."""

    def test_missing_items_warning(self):
        dataset = create_mock_dataset()

        # Only provide predictions for first 2 items
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
        ]

        eval_result = create_mock_eval_result(dataset, predictions)

        # Remove items 2 and 3 from eval_result (simulate partial evaluation)
        eval_result.item_results = eval_result.item_results[:2]

        metrics = compute_metrics(eval_result, dataset)

        # Should have warnings about missing items
        assert len(metrics.warnings) > 0
        assert "not found" in metrics.warnings[0]
        assert metrics.n_items == 2

    def test_no_common_items_raises(self):
        dataset = create_mock_dataset()
        predictions = [[CriterionVerdict.MET, CriterionVerdict.MET]]
        eval_result = create_mock_eval_result(dataset, predictions)

        # Change item indices to be out of range
        eval_result.item_results[0].item_idx = 100

        with pytest.raises(ValueError, match="No common items"):
            compute_metrics(eval_result, dataset)


class TestMetricsResultMethods:
    """Test MetricsResult methods."""

    def test_summary(self):
        dataset = create_mock_dataset()
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(eval_result, dataset)
        summary = metrics.summary()

        assert "METRICS SUMMARY" in summary
        assert "Criterion-Level Metrics" in summary
        assert "Score-Level Metrics" in summary
        assert "Accuracy" in summary

    def test_to_dataframe(self):
        dataset = create_mock_dataset()
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(eval_result, dataset)
        df = metrics.to_dataframe()

        # Should have aggregate row + 2 criterion rows
        assert len(df) == 3
        assert "level" in df.columns
        assert "aggregate" in df["level"].values
        assert "criterion" in df["level"].values

    def test_to_file(self, tmp_path):
        dataset = create_mock_dataset()
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(eval_result, dataset)
        output_path = tmp_path / "metrics.json"
        metrics.to_file(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "criterion_accuracy" in content
        assert "score_rmse" in content
        assert "per_criterion" in content


class TestEvalResultMethod:
    """Test EvalResult.compute_metrics method."""

    def test_method_works(self):
        dataset = create_mock_dataset()
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]
        eval_result = create_mock_eval_result(dataset, predictions)

        # Call via method on EvalResult
        metrics = eval_result.compute_metrics(dataset)

        assert isinstance(metrics, MetricsResult)
        assert metrics.criterion_accuracy == 1.0
