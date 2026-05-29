"""Tests for compute_metrics function."""

from datetime import datetime

import pytest

from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.metrics import MetricsResult, compute_metrics
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    CriterionReport,
    CriterionVerdict,
    EvaluationReport,
)


def create_mock_dataset():
    """Create a simple mock dataset for testing."""
    rubric = Rubric(
        [
            Criterion(name="Accuracy", weight=10.0, requirement="Be accurate"),
            Criterion(name="Clarity", weight=5.0, requirement="Be clear"),
        ]
    )

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

        item_results.append(
            ItemResult(
                item_idx=idx,
                item=dataset.items[idx],
                report=report,
                duration_seconds=0.5,
            )
        )

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

        metrics = compute_metrics(eval_result, dataset, bootstrap=True, n_bootstrap=100, seed=42)

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
        pytest.importorskip("pandas")  # to_dataframe() requires the optional pandas dependency
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


def create_single_criterion_dataset(ground_truths: list[list[CriterionVerdict]]):
    """Create a single binary-criterion dataset with explicit ground truths.

    Each element of ``ground_truths`` is a per-item list of one verdict.
    """
    rubric = Rubric([Criterion(name="Accuracy", weight=10.0, requirement="Be accurate")])
    dataset = RubricDataset(prompt="Test prompt", rubric=rubric, name="test-single")
    for idx, gt in enumerate(ground_truths):
        dataset.add_item(
            submission=f"Response {idx}",
            description=f"Item {idx}",
            ground_truth=gt,
        )
    return dataset


class TestComputeMetricsCannotAssess:
    """Tests for CANNOT_ASSESS handling, including the as_category mode (Issue #1)."""

    CA = CriterionVerdict.CANNOT_ASSESS
    MET = CriterionVerdict.MET
    UNMET = CriterionVerdict.UNMET

    def test_as_category_accepted_at_runtime(self):
        """as_category must be accepted and return a MetricsResult (was unreachable)."""
        dataset = create_single_criterion_dataset([[self.MET], [self.UNMET], [self.CA], [self.MET]])
        predictions = [[self.MET], [self.CA], [self.CA], [self.UNMET]]
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(eval_result, dataset, cannot_assess="as_category")

        assert isinstance(metrics, MetricsResult)

    def test_ca_vs_ca_counts_as_correct_under_as_category(self):
        """A CA prediction matching a CA ground truth counts as a correct 3-class match."""
        # Items: (MET,MET) match, (CA,CA) match under as_category, (UNMET,UNMET) match.
        dataset = create_single_criterion_dataset([[self.MET], [self.CA], [self.UNMET]])
        predictions = [[self.MET], [self.CA], [self.UNMET]]
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(eval_result, dataset, cannot_assess="as_category")

        # All three pairs agree as distinct classes -> perfect 3-class accuracy.
        assert metrics.criterion_accuracy == 1.0
        cm = metrics.per_criterion[0]
        assert cm.accuracy == 1.0
        assert cm.n_samples == 3

    def test_as_category_differs_from_as_unmet(self):
        """When truth is a real verdict but pred is CA, the two modes diverge."""
        # truth: MET, UNMET ; pred: CA, CA
        dataset = create_single_criterion_dataset([[self.MET], [self.UNMET]])
        predictions = [[self.CA], [self.CA]]
        eval_result = create_mock_eval_result(dataset, predictions)

        cat = compute_metrics(eval_result, dataset, cannot_assess="as_category")
        unmet = compute_metrics(eval_result, dataset, cannot_assess="as_unmet")

        # as_category: pred=CA vs truth=MET (miss), pred=CA vs truth=UNMET (miss) -> 0.0
        assert cat.criterion_accuracy == 0.0
        # as_unmet: CA->UNMET, so (UNMET vs MET)=miss, (UNMET vs UNMET)=hit -> 0.5
        assert unmet.criterion_accuracy == 0.5
        # Distinct results prove the 3-class treatment is real.
        assert cat.criterion_accuracy != unmet.criterion_accuracy

    def test_three_class_kappa_and_accuracy_under_as_category(self):
        """Per-criterion kappa/accuracy reflect three classes under as_category."""
        # A mix exercising all three classes in both truth and pred.
        gts = [[self.MET], [self.UNMET], [self.CA], [self.MET], [self.CA], [self.UNMET]]
        preds = [[self.MET], [self.UNMET], [self.CA], [self.UNMET], [self.MET], [self.UNMET]]
        dataset = create_single_criterion_dataset(gts)
        eval_result = create_mock_eval_result(dataset, preds)

        metrics = compute_metrics(eval_result, dataset, cannot_assess="as_category")
        cm = metrics.per_criterion[0]

        # No filtering -> all 6 samples retained.
        assert cm.n_samples == 6
        # 3-class accuracy: matches at indices 0,1,2,5 -> 4/6.
        assert cm.accuracy == pytest.approx(4 / 6)
        # Kappa is computed over 3 classes; it must be finite (not the undefined 0 fallback
        # path) and reflect the partial agreement.
        assert -1.0 <= cm.kappa <= 1.0
        assert cm.kappa != 0.0

    def test_exclude_unchanged(self):
        """Regression: exclude drops CA pairs; numbers match hand computation."""
        # truth: MET, UNMET, CA, MET ; pred: MET, CA, UNMET, UNMET
        # exclude drops item1 (pred CA) and item2 (truth CA), leaving:
        #   item0: MET/MET (hit), item3: truth MET / pred UNMET (miss) -> 2 samples, acc 0.5
        dataset = create_single_criterion_dataset([[self.MET], [self.UNMET], [self.CA], [self.MET]])
        predictions = [[self.MET], [self.CA], [self.UNMET], [self.UNMET]]
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(eval_result, dataset, cannot_assess="exclude")
        cm = metrics.per_criterion[0]

        assert cm.n_samples == 2
        assert cm.accuracy == 0.5
        assert metrics.criterion_accuracy == 0.5
        # MET-vs-rest support: truth MET among kept = item0,item3 -> 2; pred MET = item0 -> 1
        assert cm.support_true == 2
        assert cm.support_pred == 1

    def test_as_unmet_unchanged(self):
        """Regression: as_unmet collapses CA->UNMET; numbers match hand computation."""
        # truth: MET, UNMET, CA, MET ; pred: MET, CA, UNMET, UNMET
        # as_unmet: truth->[MET,UNMET,UNMET,MET], pred->[MET,UNMET,UNMET,UNMET]
        #   matches: item0 hit, item1 hit, item2 hit, item3 miss -> 3/4 = 0.75
        dataset = create_single_criterion_dataset([[self.MET], [self.UNMET], [self.CA], [self.MET]])
        predictions = [[self.MET], [self.CA], [self.UNMET], [self.UNMET]]
        eval_result = create_mock_eval_result(dataset, predictions)

        metrics = compute_metrics(eval_result, dataset, cannot_assess="as_unmet")
        cm = metrics.per_criterion[0]

        assert cm.n_samples == 4
        assert cm.accuracy == 0.75
        assert metrics.criterion_accuracy == 0.75
        # MET-vs-rest: truth MET = item0,item3 -> 2 ; pred MET = item0 -> 1
        assert cm.support_true == 2
        assert cm.support_pred == 1


class TestCaKappa:
    """CANNOT_ASSESS stats (T2-C): the binary parallel of multi-choice NAStats.

    Cohen's kappa on the dichotomized {CANNOT_ASSESS, not-CANNOT_ASSESS} decision plus
    counts/FP/FN, mirroring TestNaKappa. CANNOT_ASSESS is a distinct kind of abstention
    from NA but gets the same structural diagnostic block.
    """

    CA = CriterionVerdict.CANNOT_ASSESS
    MET = CriterionVerdict.MET
    UNMET = CriterionVerdict.UNMET

    def test_cannot_assess_stats_populated_for_binary(self):
        """A binary rubric with CANNOT_ASSESS in truth/pred populates cannot_assess_stats."""
        gts = [[self.MET], [self.CA], [self.UNMET], [self.CA]]
        preds = [[self.MET], [self.CA], [self.UNMET], [self.MET]]
        dataset = create_single_criterion_dataset(gts)
        eval_result = create_mock_eval_result(dataset, preds)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.cannot_assess_stats is not None

    def test_ca_kappa_perfect_agreement_is_one(self):
        """Perfect {CA, not-CA} agreement => ca_kappa=1.0, interpretation 'almost perfect'.

        items 0,1: both pred and truth CA; items 2,3: both not-CA.
        A=2, fp=0, fn=0, N=2 => P_o=1.0, P_e=0.5, kappa=1.0.
        """
        gts = [[self.CA], [self.CA], [self.MET], [self.UNMET]]
        preds = [[self.CA], [self.CA], [self.MET], [self.UNMET]]
        dataset = create_single_criterion_dataset(gts)
        eval_result = create_mock_eval_result(dataset, preds)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.cannot_assess_stats is not None
        assert metrics.cannot_assess_stats.ca_kappa == pytest.approx(1.0)
        assert metrics.cannot_assess_stats.ca_kappa_interpretation == "almost perfect"

    def test_ca_kappa_no_ca_observed_is_none(self):
        """No CA in either side => single class => ca_kappa undefined => None."""
        gts = [[self.MET], [self.UNMET], [self.MET]]
        preds = [[self.MET], [self.UNMET], [self.MET]]
        dataset = create_single_criterion_dataset(gts)
        eval_result = create_mock_eval_result(dataset, preds)

        metrics = compute_metrics(eval_result, dataset)

        # The block is still populated for a binary rubric (counts 0), kappa undefined.
        assert metrics.cannot_assess_stats is not None
        assert metrics.cannot_assess_stats.ca_kappa is None
        assert metrics.cannot_assess_stats.ca_kappa_interpretation is None
        assert metrics.cannot_assess_stats.ca_count_true == 0
        assert metrics.cannot_assess_stats.ca_count_pred == 0

    def test_ca_kappa_disagreement_below_perfect(self):
        """6-item 2x2 mix gives ca_kappa = 1/3 (mirrors the NA disagreement test)."""
        gts = [[self.CA], [self.CA], [self.MET], [self.CA], [self.MET], [self.UNMET]]
        preds = [[self.CA], [self.CA], [self.CA], [self.MET], [self.MET], [self.UNMET]]
        dataset = create_single_criterion_dataset(gts)
        eval_result = create_mock_eval_result(dataset, preds)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.cannot_assess_stats is not None
        assert metrics.cannot_assess_stats.ca_kappa == pytest.approx(1 / 3, abs=1e-9)

    def test_ca_counts_preserved(self):
        """ca_count_* and ca_false_* populate correctly for the disagreement layout."""
        gts = [[self.CA], [self.CA], [self.MET], [self.CA], [self.MET], [self.UNMET]]
        preds = [[self.CA], [self.CA], [self.CA], [self.MET], [self.MET], [self.UNMET]]
        dataset = create_single_criterion_dataset(gts)
        eval_result = create_mock_eval_result(dataset, preds)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.cannot_assess_stats is not None
        ca = metrics.cannot_assess_stats
        assert ca.ca_count_true == 3
        assert ca.ca_count_pred == 3
        assert ca.ca_false_positive == 1
        assert ca.ca_false_negative == 1

    def test_ca_stats_mode_independent(self):
        """The CA stats come from raw verdicts, so they are identical across cannot_assess
        modes (exclude/as_unmet/as_category)."""
        gts = [[self.CA], [self.CA], [self.MET], [self.CA], [self.MET], [self.UNMET]]
        preds = [[self.CA], [self.CA], [self.CA], [self.MET], [self.MET], [self.UNMET]]
        dataset = create_single_criterion_dataset(gts)
        eval_result = create_mock_eval_result(dataset, preds)

        for mode in ("exclude", "as_unmet", "as_category"):
            ca = compute_metrics(eval_result, dataset, cannot_assess=mode).cannot_assess_stats
            assert ca is not None
            assert ca.ca_count_true == 3
            assert ca.ca_count_pred == 3
            assert ca.ca_false_positive == 1
            assert ca.ca_false_negative == 1
            assert ca.ca_kappa == pytest.approx(1 / 3, abs=1e-9)

    def test_summary_renders_cannot_assess_section(self):
        """summary() shows a CANNOT_ASSESS Handling section for a binary rubric with CA."""
        gts = [[self.CA], [self.CA], [self.MET], [self.UNMET]]
        preds = [[self.CA], [self.CA], [self.MET], [self.UNMET]]
        dataset = create_single_criterion_dataset(gts)
        eval_result = create_mock_eval_result(dataset, preds)

        text = compute_metrics(eval_result, dataset).summary()

        assert "CANNOT_ASSESS Handling:" in text
        assert "CA in Ground Truth:" in text
        assert "CA Kappa:" in text


class TestErroredReportExcludedFromScoreAggregation:
    """Issue #7a: a report-level error (no item-level error) is EXCLUDED from the
    score-level RMSE/correlation/bias arrays — not substituted with a fabricated 0.0."""

    def _eval_with_one_errored_report(self, dataset, predictions):
        """Build an EvalResult where item index 1's report carries error + score=None
        WITHOUT an item-level error (so it passes the item_result.error is None gate
        and reaches the score-aggregation logic)."""
        base = create_mock_eval_result(dataset, predictions)
        broken = base.item_results[1]
        broken.report = EvaluationReport(
            score=None,
            raw_score=None,
            report=broken.report.report,  # keep per-criterion verdicts intact
            error="No judge results to aggregate",
        )
        # Crucially: NO item-level error, so it is not skipped at the item level.
        assert broken.error is None
        return base

    def test_errored_report_excluded_from_score_rmse_and_bias(self):
        dataset = create_mock_dataset()
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]

        # Full eval but with item 1's report errored (score=None).
        eval_with_error = self._eval_with_one_errored_report(dataset, predictions)

        # Reference: a dataset+eval containing ONLY the 3 non-errored items.
        good_dataset = create_mock_dataset()
        # Drop item index 1 from both dataset and predictions.
        del good_dataset.items[1]
        good_preds = [predictions[0], predictions[2], predictions[3]]
        good_eval = create_mock_eval_result(good_dataset, good_preds)

        m_err = compute_metrics(eval_with_error, dataset)
        m_good = compute_metrics(good_eval, good_dataset)

        # Score-level metrics are computed over the SAME 3 good items in both, so they
        # must match. If the errored item had been substituted with a fabricated 0.0,
        # the RMSE / bias would differ.
        assert m_err.score_rmse == pytest.approx(m_good.score_rmse)
        assert m_err.bias.mean_bias == pytest.approx(m_good.bias.mean_bias)

    def test_errored_item_still_counts_for_per_criterion(self):
        """Per-criterion verdict handling is unchanged: the errored report's verdicts
        still contribute (only the score-level arrays exclude it)."""
        dataset = create_mock_dataset()
        predictions = [
            [CriterionVerdict.MET, CriterionVerdict.MET],
            [CriterionVerdict.MET, CriterionVerdict.UNMET],
            [CriterionVerdict.UNMET, CriterionVerdict.MET],
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        ]
        eval_with_error = self._eval_with_one_errored_report(dataset, predictions)

        m = compute_metrics(eval_with_error, dataset)
        # All 4 items have ground truth, so n_items (items_with_ground_truth) is 4.
        assert m.n_items == 4
        # Per-criterion accuracy still sees the errored item's verdicts (perfect preds).
        assert m.criterion_accuracy == 1.0
