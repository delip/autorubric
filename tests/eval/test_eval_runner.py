"""Tests for EvalRunner and evaluation infrastructure."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from autorubric import (
    Criterion,
    CriterionVerdict,
    EvalConfig,
    EvalResult,
    EvalRunner,
    EvalTimingStats,
    ExperimentManifest,
    ItemResult,
    Rubric,
    TokenUsage,
    evaluate,
)
from autorubric.dataset import DataItem, RubricDataset
from autorubric.llm import LLMConfig
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    CriterionOption,
    CriterionReport,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
    MultiChoiceVerdict,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_dataset() -> RubricDataset:
    """Create a sample dataset for testing."""
    rubric = Rubric(
        [
            Criterion(name="accuracy", weight=10.0, requirement="Is factually accurate"),
            Criterion(name="clarity", weight=5.0, requirement="Is clearly written"),
        ]
    )
    dataset = RubricDataset(
        prompt="Explain a concept",
        rubric=rubric,
        name="test-dataset",
    )
    dataset.add_item(
        submission="A good explanation.",
        description="Good response",
        ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
    )
    dataset.add_item(
        submission="A mediocre explanation.",
        description="Average response",
        ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
    )
    dataset.add_item(
        submission="A poor explanation.",
        description="Poor response",
        ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.UNMET],
    )
    return dataset


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create a mock LLMConfig for testing."""
    return LLMConfig(model="test-model", max_parallel_requests=5)


def create_mock_grader(mock_reports: list[EvaluationReport] | None = None) -> MagicMock:
    """Create a mock grader that returns predefined reports."""
    if mock_reports is None:
        mock_reports = [
            EvaluationReport(
                score=1.0,
                raw_score=15.0,
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                completion_cost=0.001,
            ),
            EvaluationReport(
                score=0.67,
                raw_score=10.0,
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                completion_cost=0.001,
            ),
            EvaluationReport(
                score=0.0,
                raw_score=0.0,
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                completion_cost=0.001,
            ),
        ]

    call_idx = 0

    async def mock_grade(
        to_grade: str,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
    ):
        nonlocal call_idx
        report = mock_reports[call_idx % len(mock_reports)]
        call_idx += 1
        return report

    mock = MagicMock()
    mock.grade = AsyncMock(side_effect=mock_grade)
    mock._judges = []  # Empty list indicates non-ensemble mode
    return mock


# -----------------------------------------------------------------------------
# EvalConfig Tests
# -----------------------------------------------------------------------------


class TestEvalConfig:
    """Tests for EvalConfig dataclass."""

    def test_default_values(self):
        """Test that EvalConfig has sensible defaults."""
        config = EvalConfig()
        assert config.fail_fast is False
        assert config.show_progress is True
        assert config.progress_style == "simple"
        assert config.max_concurrent_items is None
        assert config.experiment_name is None
        assert config.experiments_dir == "experiments"
        assert config.resume is True

    def test_custom_values(self):
        """Test EvalConfig with custom values."""
        config = EvalConfig(
            fail_fast=True,
            show_progress=False,
            progress_style="detailed",
            max_concurrent_items=10,
            experiment_name="my-experiment",
            experiments_dir="/custom/path",
            resume=False,
        )
        assert config.fail_fast is True
        assert config.show_progress is False
        assert config.progress_style == "detailed"
        assert config.max_concurrent_items == 10
        assert config.experiment_name == "my-experiment"
        assert config.experiments_dir == "/custom/path"
        assert config.resume is False


# -----------------------------------------------------------------------------
# EvalTimingStats Tests
# -----------------------------------------------------------------------------


class TestEvalTimingStats:
    """Tests for EvalTimingStats computation."""

    def test_from_durations_basic(self):
        """Test computing timing stats from durations."""
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        total_duration = 10.0

        stats = EvalTimingStats.from_durations(durations, total_duration)

        assert stats.total_duration_seconds == 10.0
        assert stats.mean_item_duration_seconds == 3.0
        assert stats.min_item_duration_seconds == 1.0
        assert stats.max_item_duration_seconds == 5.0
        assert stats.p50_item_duration_seconds == 3.0
        assert stats.items_per_second == 0.5

    def test_from_durations_empty(self):
        """Test timing stats with empty durations list."""
        stats = EvalTimingStats.from_durations([], 10.0)

        assert stats.total_duration_seconds == 10.0
        assert stats.mean_item_duration_seconds == 0.0
        assert stats.min_item_duration_seconds == 0.0
        assert stats.max_item_duration_seconds == 0.0
        assert stats.p50_item_duration_seconds == 0.0
        assert stats.items_per_second == 0.0

    def test_from_durations_single_item(self):
        """Test timing stats with single duration."""
        stats = EvalTimingStats.from_durations([2.5], 2.5)

        assert stats.mean_item_duration_seconds == 2.5
        assert stats.min_item_duration_seconds == 2.5
        assert stats.max_item_duration_seconds == 2.5
        assert stats.p50_item_duration_seconds == 2.5
        assert stats.items_per_second == 0.4

    def test_to_dict(self):
        """Test serialization of timing stats."""
        stats = EvalTimingStats(
            total_duration_seconds=10.0,
            mean_item_duration_seconds=2.0,
            min_item_duration_seconds=1.0,
            max_item_duration_seconds=3.0,
            p50_item_duration_seconds=2.0,
            p95_item_duration_seconds=2.9,
            items_per_second=0.5,
        )

        d = stats.to_dict()

        assert d["total_duration_seconds"] == 10.0
        assert d["mean_item_duration_seconds"] == 2.0
        assert d["items_per_second"] == 0.5


# -----------------------------------------------------------------------------
# ItemResult Tests
# -----------------------------------------------------------------------------


class TestItemResult:
    """Tests for ItemResult serialization."""

    def test_to_dict(self):
        """Test ItemResult serialization to dict."""
        item = DataItem(submission="Test text", description="Test item")
        report = EvaluationReport(
            score=0.85,
            raw_score=12.75,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            completion_cost=0.002,
        )
        result = ItemResult(
            item_idx=0,
            item=item,
            report=report,
            duration_seconds=1.5,
        )

        d = result.to_dict()

        assert d["item_idx"] == 0
        assert d["duration_seconds"] == 1.5
        assert d["error"] is None
        assert d["report"]["score"] == 0.85
        assert d["report"]["raw_score"] == 12.75
        assert d["report"]["token_usage"]["prompt_tokens"] == 100

    def test_from_dict(self):
        """Test ItemResult deserialization from dict."""
        data = {
            "item_idx": 1,
            "duration_seconds": 2.0,
            "error": None,
            "report": {
                "score": 0.75,
                "raw_score": 11.25,
                "error": None,
                "cannot_assess_count": 0,
                "token_usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
                "completion_cost": 0.003,
            },
        }
        item = DataItem(submission="Test", description="Test")

        result = ItemResult.from_dict(data, item)

        assert result.item_idx == 1
        assert result.duration_seconds == 2.0
        assert result.error is None
        assert result.report.score == 0.75
        assert result.report.token_usage.prompt_tokens == 200

    def test_to_dict_with_error(self):
        """Test ItemResult serialization when there's an error."""
        item = DataItem(submission="Test", description="Test")
        report = EvaluationReport(score=0.0, raw_score=0.0, error="Parse error")
        result = ItemResult(
            item_idx=0,
            item=item,
            report=report,
            duration_seconds=0.5,
            error="Parse error",
        )

        d = result.to_dict()

        assert d["error"] == "Parse error"
        assert d["report"]["error"] == "Parse error"

    def test_roundtrip_single_binary(self):
        """Single-judge binary criterion reports survive roundtrip."""
        item = DataItem(submission="Test", description="Test")
        criterion_reports = [
            CriterionReport(
                weight=10.0,
                requirement="Is accurate",
                verdict=CriterionVerdict.MET,
                reason="Looks good",
            ),
            CriterionReport(
                weight=5.0,
                requirement="Is concise",
                verdict=CriterionVerdict.UNMET,
                reason="Too verbose",
            ),
        ]
        report = EvaluationReport(score=0.67, raw_score=10.0, report=criterion_reports)
        result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=1.0)

        d = result.to_dict()
        restored = ItemResult.from_dict(d, item)

        assert restored.report.report is not None
        assert len(restored.report.report) == 2
        assert restored.report.report[0].verdict == CriterionVerdict.MET
        assert restored.report.report[0].reason == "Looks good"
        assert restored.report.report[1].verdict == CriterionVerdict.UNMET
        assert restored.report.score == 0.67

    def test_roundtrip_single_multi_choice(self):
        """Single-judge multi-choice criterion reports survive roundtrip."""
        item = DataItem(submission="Test", description="Test")
        options = [
            CriterionOption(label="Bad", value=0.0),
            CriterionOption(label="OK", value=0.5),
            CriterionOption(label="Good", value=1.0),
        ]
        criterion_reports = [
            CriterionReport(
                weight=10.0,
                requirement="Quality?",
                options=options,
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=2, selected_label="Good", value=1.0
                ),
                reason="High quality",
            ),
        ]
        report = EvaluationReport(score=1.0, raw_score=10.0, report=criterion_reports)
        result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=1.0)

        d = result.to_dict()
        restored = ItemResult.from_dict(d, item)

        assert restored.report.report is not None
        cr = restored.report.report[0]
        assert cr.multi_choice_verdict is not None
        assert cr.multi_choice_verdict.selected_index == 2
        assert cr.multi_choice_verdict.selected_label == "Good"

    def test_roundtrip_ensemble_binary(self):
        """Ensemble binary criterion reports survive roundtrip."""
        item = DataItem(submission="Test", description="Test")
        criterion = Criterion(weight=10.0, requirement="Is accurate")
        ecr = EnsembleCriterionReport(
            criterion=criterion,
            final_verdict=CriterionVerdict.MET,
            final_reason="Both agree",
            votes=[
                JudgeVote(judge_id="j1", verdict=CriterionVerdict.MET, reason="Yes", weight=1.0),
                JudgeVote(judge_id="j2", verdict=CriterionVerdict.MET, reason="Agree", weight=1.0),
            ],
            agreement=1.0,
        )
        report = EnsembleEvaluationReport(
            score=1.0,
            raw_score=10.0,
            report=[ecr],
            judge_scores={"j1": 1.0, "j2": 1.0},
            mean_agreement=1.0,
        )
        result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=1.0)

        d = result.to_dict()
        restored = ItemResult.from_dict(d, item)

        assert isinstance(restored.report, EnsembleEvaluationReport)
        assert restored.report.report is not None
        assert len(restored.report.report) == 1
        assert restored.report.report[0].final_verdict == CriterionVerdict.MET
        assert len(restored.report.report[0].votes) == 2
        assert restored.report.report[0].votes[0].judge_id == "j1"
        assert restored.report.judge_scores == {"j1": 1.0, "j2": 1.0}

    def test_roundtrip_ensemble_multi_choice(self):
        """Ensemble multi-choice criterion reports survive roundtrip."""
        item = DataItem(submission="Test", description="Test")
        options = [
            CriterionOption(label="Low", value=0.0),
            CriterionOption(label="High", value=1.0),
        ]
        criterion = Criterion(weight=10.0, requirement="Quality?", options=options)
        ecr = EnsembleCriterionReport(
            criterion=criterion,
            final_verdict=None,
            final_reason="Aggregated",
            final_multi_choice_verdict=AggregatedMultiChoiceVerdict(
                selected_index=1,
                selected_label="High",
                value=1.0,
                aggregated_value=0.8,
            ),
            multi_choice_votes=[
                MultiChoiceJudgeVote(
                    judge_id="j1",
                    selected_index=1,
                    selected_label="High",
                    value=1.0,
                    reason="Clearly high",
                ),
            ],
        )
        report = EnsembleEvaluationReport(
            score=1.0, raw_score=10.0, report=[ecr], judge_scores={"j1": 1.0}
        )
        result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=1.0)

        d = result.to_dict()
        restored = ItemResult.from_dict(d, item)

        assert isinstance(restored.report, EnsembleEvaluationReport)
        ecr_restored = restored.report.report[0]
        assert ecr_restored.final_multi_choice_verdict is not None
        assert ecr_restored.final_multi_choice_verdict.selected_index == 1
        assert ecr_restored.final_multi_choice_verdict.aggregated_value == 0.8
        assert len(ecr_restored.multi_choice_votes) == 1
        assert ecr_restored.multi_choice_votes[0].judge_id == "j1"

    def test_backward_compat_no_criterion_reports(self):
        """Old format without criterion_reports loads with report=None."""
        data = {
            "item_idx": 0,
            "duration_seconds": 1.0,
            "error": None,
            "report": {"score": 0.5, "raw_score": 5.0, "error": None},
        }
        item = DataItem(submission="Test", description="Test")
        result = ItemResult.from_dict(data, item)

        assert result.report.report is None
        assert result.report.score == 0.5


# -----------------------------------------------------------------------------
# ExperimentManifest Tests
# -----------------------------------------------------------------------------


class TestExperimentManifest:
    """Tests for ExperimentManifest serialization."""

    def test_to_dict(self):
        """Test manifest serialization."""
        from datetime import datetime

        manifest = ExperimentManifest(
            experiment_name="test-experiment",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            dataset_name="test-dataset",
            dataset_hash="abc123",
            total_items=10,
            status="running",
            completed_indices={0, 1, 2},
            started_at=datetime(2024, 1, 15, 10, 30, 0),
        )

        d = manifest.to_dict()

        assert d["experiment_name"] == "test-experiment"
        assert d["dataset_name"] == "test-dataset"
        assert d["dataset_hash"] == "abc123"
        assert d["total_items"] == 10
        assert d["status"] == "running"
        assert sorted(d["completed_indices"]) == [0, 1, 2]

    def test_from_dict(self):
        """Test manifest deserialization."""
        data = {
            "experiment_name": "loaded-experiment",
            "created_at": "2024-01-15T12:00:00",
            "dataset_name": "my-dataset",
            "dataset_hash": "xyz789",
            "total_items": 20,
            "status": "completed",
            "completed_indices": [0, 1, 2, 3, 4],
            "started_at": "2024-01-15T12:00:00",
            "completed_at": "2024-01-15T12:05:00",
        }

        manifest = ExperimentManifest.from_dict(data)

        assert manifest.experiment_name == "loaded-experiment"
        assert manifest.dataset_name == "my-dataset"
        assert manifest.status == "completed"
        assert manifest.completed_indices == {0, 1, 2, 3, 4}


# -----------------------------------------------------------------------------
# EvalRunner Tests
# -----------------------------------------------------------------------------


class TestEvalRunner:
    """Tests for EvalRunner execution."""

    @pytest.mark.asyncio
    async def test_basic_evaluation(self, sample_dataset, mock_llm_config):
        """Test basic evaluation run."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvalConfig(
                show_progress=False,
                experiments_dir=tmp_dir,
            )
            mock_grader = create_mock_grader()

            runner = EvalRunner(
                dataset=sample_dataset,
                grader=mock_grader,
                config=config,
            )
            result = await runner.run()

            assert result.total_items == 3
            assert result.successful_items == 3
            assert result.failed_items == 0
            assert len(result.item_results) == 3
            assert result.experiment_dir is not None
            assert result.experiment_dir.exists()

    @pytest.mark.asyncio
    async def test_evaluation_with_errors(self, sample_dataset, mock_llm_config):
        """Test evaluation handles errors gracefully."""
        call_idx = 0

        async def mock_grade_with_error(
            to_grade: str,
            rubric: list[Criterion],
            query: str | None = None,
            reference_submission: str | None = None,
        ):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 2:  # Fail on second item
                raise ValueError("Simulated error")
            return EvaluationReport(score=1.0, raw_score=15.0)

        mock_grader = MagicMock()
        mock_grader.grade = AsyncMock(side_effect=mock_grade_with_error)
        mock_grader._judges = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvalConfig(
                show_progress=False,
                experiments_dir=tmp_dir,
                fail_fast=False,
            )

            runner = EvalRunner(
                dataset=sample_dataset,
                grader=mock_grader,
                config=config,
            )
            result = await runner.run()

            assert result.total_items == 3
            assert result.successful_items == 2
            assert result.failed_items == 1
            assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_fail_fast_stops_on_error(self, sample_dataset, mock_llm_config):
        """Test fail_fast=True stops evaluation on first error."""
        call_idx = 0

        async def mock_grade_with_error(
            to_grade: str,
            rubric: list[Criterion],
            query: str | None = None,
            reference_submission: str | None = None,
        ):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:  # Fail on first item
                raise ValueError("Simulated error")
            return EvaluationReport(score=1.0, raw_score=15.0)

        mock_grader = MagicMock()
        mock_grader.grade = AsyncMock(side_effect=mock_grade_with_error)
        mock_grader._judges = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvalConfig(
                show_progress=False,
                experiments_dir=tmp_dir,
                fail_fast=True,
            )

            runner = EvalRunner(
                dataset=sample_dataset,
                grader=mock_grader,
                config=config,
            )

            with pytest.raises(RuntimeError, match="Evaluation failed"):
                await runner.run()

    @pytest.mark.asyncio
    async def test_checkpointing_creates_files(self, sample_dataset, mock_llm_config):
        """Test that checkpointing creates manifest and items files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvalConfig(
                show_progress=False,
                experiments_dir=tmp_dir,
                experiment_name="checkpoint-test",
            )
            mock_grader = create_mock_grader()

            runner = EvalRunner(
                dataset=sample_dataset,
                grader=mock_grader,
                config=config,
            )
            await runner.run()

            exp_dir = Path(tmp_dir) / "checkpoint-test"
            assert exp_dir.exists()
            assert (exp_dir / "manifest.json").exists()
            assert (exp_dir / "items.jsonl").exists()

            # Verify manifest content
            with open(exp_dir / "manifest.json") as f:
                manifest = json.load(f)
            assert manifest["experiment_name"] == "checkpoint-test"
            assert manifest["status"] == "completed"
            assert manifest["total_items"] == 3
            assert len(manifest["completed_indices"]) == 3

            # Verify items file has all results
            with open(exp_dir / "items.jsonl") as f:
                items = [json.loads(line) for line in f if line.strip()]
            assert len(items) == 3

    @pytest.mark.asyncio
    async def test_max_concurrent_items(self, sample_dataset, mock_llm_config):
        """Test that max_concurrent_items limits concurrency."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvalConfig(
                show_progress=False,
                experiments_dir=tmp_dir,
                max_concurrent_items=1,  # Process one at a time
            )
            mock_grader = create_mock_grader()

            runner = EvalRunner(
                dataset=sample_dataset,
                grader=mock_grader,
                config=config,
            )
            result = await runner.run()

            assert result.total_items == 3
            assert result.successful_items == 3

    @pytest.mark.asyncio
    async def test_aggregated_usage_and_cost(self, sample_dataset, mock_llm_config):
        """Test that usage and cost are aggregated correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvalConfig(
                show_progress=False,
                experiments_dir=tmp_dir,
            )
            mock_grader = create_mock_grader()

            runner = EvalRunner(
                dataset=sample_dataset,
                grader=mock_grader,
                config=config,
            )
            result = await runner.run()

            # Each item has 150 tokens and costs 0.001
            assert result.total_token_usage is not None
            assert result.total_token_usage.total_tokens == 450  # 3 * 150
            assert result.total_completion_cost == pytest.approx(0.003)  # 3 * 0.001


# -----------------------------------------------------------------------------
# evaluate() Convenience Function Tests
# -----------------------------------------------------------------------------


class TestEvaluateFunction:
    """Tests for the evaluate() convenience function."""

    @pytest.mark.asyncio
    async def test_evaluate_basic(self, sample_dataset, mock_llm_config):
        """Test basic evaluation with convenience function."""
        mock_grader = create_mock_grader()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await evaluate(
                dataset=sample_dataset,
                grader=mock_grader,
                show_progress=False,
                experiments_dir=tmp_dir,
            )

            assert result.total_items == 3
            assert result.successful_items == 3
            assert len(result.get_scores()) == 3

    @pytest.mark.asyncio
    async def test_evaluate_with_custom_config(self, sample_dataset, mock_llm_config):
        """Test evaluate with various config options."""
        mock_grader = create_mock_grader()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await evaluate(
                dataset=sample_dataset,
                grader=mock_grader,
                show_progress=False,
                progress_style="detailed",
                fail_fast=False,
                max_concurrent_items=2,
                experiment_name="custom-eval",
                experiments_dir=tmp_dir,
                resume=True,
            )

            assert result.total_items == 3
            assert result.experiment_name == "custom-eval"


# -----------------------------------------------------------------------------
# EvalResult Methods Tests
# -----------------------------------------------------------------------------


class TestEvalResultMethods:
    """Tests for EvalResult helper methods."""

    @pytest.mark.asyncio
    async def test_get_scores(self, sample_dataset, mock_llm_config):
        """Test get_scores returns correct scores."""
        mock_grader = create_mock_grader()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await evaluate(
                dataset=sample_dataset,
                grader=mock_grader,
                show_progress=False,
                experiments_dir=tmp_dir,
            )

            scores = result.get_scores()
            assert len(scores) == 3
            assert scores[0] == 1.0
            assert scores[1] == pytest.approx(0.67)
            assert scores[2] == 0.0

    @pytest.mark.asyncio
    async def test_get_reports(self, sample_dataset, mock_llm_config):
        """Test get_reports returns all reports."""
        mock_grader = create_mock_grader()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await evaluate(
                dataset=sample_dataset,
                grader=mock_grader,
                show_progress=False,
                experiments_dir=tmp_dir,
            )

            reports = result.get_reports()
            assert len(reports) == 3
            assert all(isinstance(r, EvaluationReport) for r in reports)

    @pytest.mark.asyncio
    async def test_filter_successful_and_failed(self, sample_dataset, mock_llm_config):
        """Test filter methods work correctly."""
        call_idx = 0

        async def mock_grade_with_error(
            to_grade: str,
            rubric: list[Criterion],
            query: str | None = None,
            reference_submission: str | None = None,
        ):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 2:
                raise ValueError("Test error")
            return EvaluationReport(score=1.0, raw_score=15.0)

        mock_grader = MagicMock()
        mock_grader.grade = AsyncMock(side_effect=mock_grade_with_error)
        mock_grader._judges = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await evaluate(
                dataset=sample_dataset,
                grader=mock_grader,
                show_progress=False,
                experiments_dir=tmp_dir,
            )

            successful = result.filter_successful()
            failed = result.filter_failed()

            assert len(successful) == 2
            assert len(failed) == 1


# -----------------------------------------------------------------------------
# EvalResult.from_experiment Tests
# -----------------------------------------------------------------------------


class TestEvalResultFromExperiment:
    """Tests for loading results from experiment directories."""

    @pytest.mark.asyncio
    async def test_load_completed_experiment(self, sample_dataset, mock_llm_config):
        """Test loading a completed experiment from disk."""
        mock_grader = create_mock_grader()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Run evaluation
            result = await evaluate(
                dataset=sample_dataset,
                grader=mock_grader,
                show_progress=False,
                experiment_name="load-test",
                experiments_dir=tmp_dir,
            )

            # Load from disk
            loaded = EvalResult.from_experiment(Path(tmp_dir) / "load-test")

            assert loaded.total_items == result.total_items
            assert loaded.successful_items == result.successful_items
            assert len(loaded.item_results) == len(result.item_results)

    def test_from_experiment_not_found(self):
        """Test loading non-existent experiment raises error."""
        with pytest.raises(FileNotFoundError):
            EvalResult.from_experiment("/nonexistent/path")

    def test_from_experiment_missing_manifest(self):
        """Test loading experiment without manifest raises error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir) / "no-manifest"
            exp_dir.mkdir()

            with pytest.raises(FileNotFoundError, match="Manifest not found"):
                EvalResult.from_experiment(exp_dir)


class TestManifestSeedPersistence:
    """Tests for master seed persistence in experiment manifests."""

    @pytest.fixture
    def sample_dataset(self):
        items = [DataItem(submission=f"Answer {i}", description=f"Item {i}") for i in range(2)]
        rubric = Rubric(
            rubric=[
                Criterion(weight=10.0, requirement="Is the answer correct?"),
            ]
        )
        return RubricDataset(
            name="seed-test",
            prompt="Test prompt",
            items=items,
            rubric=rubric,
        )

    @pytest.mark.asyncio
    async def test_manifest_contains_master_seed(self, sample_dataset):
        """Manifest grader_config includes master_seed after evaluation."""
        mock_grader = create_mock_grader()
        mock_grader._seed = 42
        mock_grader._shuffle_options = True
        mock_grader._auto_na_option = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvalConfig(
                show_progress=False,
                experiments_dir=tmp_dir,
                experiment_name="seed-test",
            )
            runner = EvalRunner(
                dataset=sample_dataset,
                grader=mock_grader,
                config=config,
            )
            await runner.run()

            with open(Path(tmp_dir) / "seed-test" / "manifest.json") as f:
                manifest = json.load(f)

            assert manifest["grader_config"]["master_seed"] == 42
            assert manifest["grader_config"]["shuffle_options"] is True
            assert manifest["grader_config"]["auto_na_option"] is False
