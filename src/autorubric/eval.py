"""Evaluation runner for batch grading with rate limiting and progress tracking.

This module provides infrastructure for running batch evaluations of datasets
against rubrics, with support for:
- Parallel execution with configurable concurrency
- Rate limiting via LLMConfig.max_parallel_requests
- Progress display with rich progress bars
- Checkpointing and resumption from failures
- Result aggregation with timing statistics
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from autorubric.dataset import DataItem, RubricDataset
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    CriterionReport,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
    TokenUsage,
)
from autorubric.utils import aggregate_completion_cost, aggregate_token_usage

if TYPE_CHECKING:
    from autorubric.graders.base import Grader
    from autorubric.metrics import MetricsResult, NAMode

logger = logging.getLogger(__name__)


def _generate_experiment_name() -> str:
    """Generate a random experiment name using coolname."""
    try:
        import coolname

        return coolname.generate_slug(2)
    except ImportError:
        # Fallback to timestamp if coolname not installed
        return datetime.now().strftime("%Y%m%d-%H%M%S")


def _compute_dataset_hash(dataset: RubricDataset) -> str:
    """Compute a hash of the dataset for integrity verification."""
    try:
        num_criteria = dataset.num_criteria
    except ValueError:
        # Per-item rubrics — use sum of all item rubric lengths as a stable fingerprint
        num_criteria = sum(
            len(r.rubric)
            for i in range(len(dataset))
            if (r := dataset.get_item_rubric(i)) is not None
        )
    content = json.dumps(
        {
            "name": dataset.name,
            "prompt": dataset.prompt,
            "num_items": len(dataset),
            "num_criteria": num_criteria,
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _serialize_grader_config(grader: Grader) -> dict[str, Any]:
    """Serialize grader configuration for manifest storage.

    Captures key configuration for reproducibility without storing sensitive data.
    Gracefully handles mocks and missing attributes.
    """
    config: dict[str, Any] = {
        "grader_class": grader.__class__.__name__,
    }

    # Get normalize setting, handling mocks
    normalize = getattr(grader, "_normalize", None)
    if isinstance(normalize, bool):
        config["normalize"] = normalize

    # For CriterionGrader, capture judges and aggregation
    try:
        judges = getattr(grader, "_judges", None)
        if judges and isinstance(judges, list) and len(judges) > 0:
            first_judge = judges[0]
            # Verify first element is actually a JudgeSpec by checking types of attributes
            judge_id = getattr(first_judge, "judge_id", None)
            llm_config = getattr(first_judge, "llm_config", None)
            if isinstance(judge_id, str) and llm_config is not None:
                model = getattr(llm_config, "model", None)
                if isinstance(model, str):
                    config["judges"] = []
                    for j in judges:
                        jid = getattr(j, "judge_id", None)
                        jcfg = getattr(j, "llm_config", None)
                        if isinstance(jid, str) and jcfg is not None:
                            jmodel = getattr(jcfg, "model", None)
                            if isinstance(jmodel, str):
                                weight = getattr(j, "weight", 1.0)
                                mpr = getattr(jcfg, "max_parallel_requests", None)
                                config["judges"].append(
                                    {
                                        "judge_id": jid,
                                        "model": jmodel,
                                        "weight": weight
                                        if isinstance(weight, (int, float))
                                        else 1.0,
                                        "max_parallel_requests": mpr
                                        if isinstance(mpr, int)
                                        else None,
                                    }
                                )
                    aggregation = getattr(grader, "_aggregation", None)
                    if isinstance(aggregation, str):
                        config["aggregation"] = aggregation
    except (TypeError, AttributeError):
        pass  # Gracefully skip if attributes aren't proper objects

    # Capture few-shot config if present
    try:
        fsc = getattr(grader, "_few_shot_config", None)
        # Check that n_examples is actually an integer, not a Mock
        if fsc and hasattr(fsc, "n_examples") and isinstance(fsc.n_examples, int):
            config["few_shot_config"] = {
                "n_examples": fsc.n_examples,
                "balance_verdicts": bool(getattr(fsc, "balance_verdicts", True)),
                "include_reason": bool(getattr(fsc, "include_reason", False)),
            }
    except (TypeError, AttributeError):
        pass

    # Capture master seed and shuffle_options
    try:
        master_seed = getattr(grader, "_seed", None)
        if isinstance(master_seed, int):
            config["master_seed"] = master_seed
        shuffle_opts = getattr(grader, "_shuffle_options", None)
        if isinstance(shuffle_opts, bool):
            config["shuffle_options"] = shuffle_opts
        auto_na = getattr(grader, "_auto_na_option", None)
        if isinstance(auto_na, bool):
            config["auto_na_option"] = auto_na
    except (TypeError, AttributeError):
        pass

    # Capture cannot_assess_config if present
    try:
        cac = getattr(grader, "_cannot_assess_config", None)
        # Check that strategy is actually a proper object with value attribute
        if cac and hasattr(cac, "strategy"):
            strat = cac.strategy
            # Verify it's an actual enum or string, not a Mock
            if hasattr(strat, "value") and isinstance(strat.value, str):
                strategy = strat.value
            elif isinstance(strat, str):
                strategy = strat
            else:
                strategy = None

            if strategy is not None:
                partial = getattr(cac, "partial_credit", 0.5)
                if isinstance(partial, (int, float)):
                    config["cannot_assess_config"] = {
                        "strategy": strategy,
                        "partial_credit": partial,
                    }
    except (TypeError, AttributeError):
        pass

    return config


def _serialize_eval_config(config: EvalConfig) -> dict[str, Any]:
    """Serialize EvalConfig for manifest storage."""
    return {
        "fail_fast": config.fail_fast,
        "show_progress": config.show_progress,
        "progress_style": config.progress_style,
        "max_concurrent_items": config.max_concurrent_items,
        "experiment_name": config.experiment_name,
        "experiments_dir": str(config.experiments_dir),
        "resume": config.resume,
    }


def _serialize_ensemble_criterion_report(ecr: EnsembleCriterionReport) -> dict[str, Any]:
    """Serialize an EnsembleCriterionReport dataclass to a dict."""
    d: dict[str, Any] = {
        "criterion": ecr.criterion.model_dump(mode="json"),
        "final_verdict": ecr.final_verdict.value if ecr.final_verdict else None,
        "final_reason": ecr.final_reason,
        "agreement": ecr.agreement,
        "error": ecr.error,
    }
    if ecr.votes:
        d["votes"] = [
            {
                "judge_id": v.judge_id,
                "verdict": v.verdict.value,
                "reason": v.reason,
                "weight": v.weight,
                "error": v.error,
            }
            for v in ecr.votes
        ]
    if ecr.final_multi_choice_verdict is not None:
        d["final_multi_choice_verdict"] = ecr.final_multi_choice_verdict.model_dump(mode="json")
    if ecr.multi_choice_votes:
        d["multi_choice_votes"] = [
            {
                "judge_id": v.judge_id,
                "selected_index": v.selected_index,
                "selected_label": v.selected_label,
                "value": v.value,
                "reason": v.reason,
                "weight": v.weight,
                "na": v.na,
                "shuffle_order": v.shuffle_order,
                "error": v.error,
            }
            for v in ecr.multi_choice_votes
        ]
    return d


def _deserialize_single_report(
    report_data: dict[str, Any], token_usage: TokenUsage | None
) -> EvaluationReport:
    """Reconstruct EvaluationReport with full criterion reports."""
    criterion_reports = [
        CriterionReport.model_validate(cr) for cr in report_data["criterion_reports"]
    ]
    return EvaluationReport(
        score=report_data["score"],
        raw_score=report_data.get("raw_score"),
        llm_raw_score=report_data.get("raw_score"),
        report=criterion_reports,
        token_usage=token_usage,
        completion_cost=report_data.get("completion_cost"),
        error=report_data.get("error"),
        cannot_assess_count=report_data.get("cannot_assess_count", 0),
    )


def _deserialize_ensemble_report(
    report_data: dict[str, Any], token_usage: TokenUsage | None
) -> EnsembleEvaluationReport:
    """Reconstruct EnsembleEvaluationReport with full criterion reports."""
    from autorubric.types import Criterion

    ensemble_reports = []
    for ecr_data in report_data["criterion_reports"]:
        criterion = Criterion.model_validate(ecr_data["criterion"])
        final_verdict = (
            CriterionVerdict(ecr_data["final_verdict"]) if ecr_data.get("final_verdict") else None
        )

        votes = [
            JudgeVote(
                judge_id=v["judge_id"],
                verdict=CriterionVerdict(v["verdict"]),
                reason=v["reason"],
                weight=v.get("weight", 1.0),
                error=v.get("error"),
            )
            for v in ecr_data.get("votes", [])
        ]

        final_mc_verdict = None
        if ecr_data.get("final_multi_choice_verdict"):
            final_mc_verdict = AggregatedMultiChoiceVerdict.model_validate(
                ecr_data["final_multi_choice_verdict"]
            )

        mc_votes = [
            MultiChoiceJudgeVote(
                judge_id=v["judge_id"],
                # None for a genuine no-option abstain (T2-B); use .get for robustness.
                selected_index=v.get("selected_index"),
                selected_label=v.get("selected_label"),
                value=v["value"],
                reason=v["reason"],
                weight=v.get("weight", 1.0),
                na=v.get("na", False),
                shuffle_order=v.get("shuffle_order"),
                error=v.get("error"),
            )
            for v in ecr_data.get("multi_choice_votes", [])
        ]

        ensemble_reports.append(
            EnsembleCriterionReport(
                criterion=criterion,
                final_verdict=final_verdict,
                final_reason=ecr_data["final_reason"],
                votes=votes,
                agreement=ecr_data.get("agreement", 0.0),
                final_multi_choice_verdict=final_mc_verdict,
                multi_choice_votes=mc_votes,
                error=ecr_data.get("error"),
            )
        )

    return EnsembleEvaluationReport(
        score=report_data["score"],
        raw_score=report_data.get("raw_score"),
        llm_raw_score=report_data.get("raw_score"),
        report=ensemble_reports,
        judge_scores=report_data.get("judge_scores", {}),
        mean_agreement=report_data.get("mean_agreement", 0.0),
        token_usage=token_usage,
        completion_cost=report_data.get("completion_cost"),
        error=report_data.get("error"),
        cannot_assess_count=report_data.get("cannot_assess_count", 0),
    )


@dataclass
class EvalConfig:
    """Configuration for evaluation runs.

    Attributes:
        fail_fast: If True, stop on first error. Default False continues all items.
        show_progress: Whether to display progress bars. Default True.
        progress_style: Style of progress display.
            - "simple": Single overall progress bar
            - "detailed": Shows per-judge progress for ensemble mode
        max_concurrent_items: Maximum items to grade concurrently.
            None = grade all items in parallel (default).
            Set this to limit memory usage for very large datasets.
        experiment_name: Name for this experiment run.
            If None, auto-generates using coolname.
        experiments_dir: Root directory for experiment outputs.
            Default is "./experiments".
        resume: If True and experiment exists, resume from checkpoint.
            Default True.
    """

    fail_fast: bool = False
    show_progress: bool = True
    progress_style: Literal["simple", "detailed"] = "simple"
    max_concurrent_items: int | None = None
    experiment_name: str | None = None
    experiments_dir: Path | str = "experiments"
    resume: bool = True


@dataclass
class ItemResult:
    """Result for a single evaluated item."""

    item_idx: int
    item: DataItem
    report: EvaluationReport | EnsembleEvaluationReport
    duration_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        report_dict: dict[str, Any] = {
            "score": self.report.score,
            "raw_score": self.report.raw_score,
            "error": self.report.error,
        }
        if hasattr(self.report, "cannot_assess_count"):
            report_dict["cannot_assess_count"] = self.report.cannot_assess_count
        if hasattr(self.report, "mean_agreement"):
            report_dict["mean_agreement"] = self.report.mean_agreement
        if self.report.token_usage:
            report_dict["token_usage"] = {
                "prompt_tokens": self.report.token_usage.prompt_tokens,
                "completion_tokens": self.report.token_usage.completion_tokens,
                "total_tokens": self.report.token_usage.total_tokens,
            }
        if self.report.completion_cost is not None:
            report_dict["completion_cost"] = self.report.completion_cost

        if self.report.report is not None:
            if isinstance(self.report, EnsembleEvaluationReport):
                report_dict["report_type"] = "ensemble"
                report_dict["criterion_reports"] = [
                    _serialize_ensemble_criterion_report(ecr) for ecr in self.report.report
                ]
                report_dict["judge_scores"] = self.report.judge_scores
            else:
                report_dict["report_type"] = "single"
                report_dict["criterion_reports"] = [
                    cr.model_dump(mode="json") for cr in self.report.report
                ]

        return {
            "item_idx": self.item_idx,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "report": report_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], item: DataItem) -> ItemResult:
        """Deserialize from dictionary."""
        report_data = data["report"]

        # Reconstruct TokenUsage if present
        token_usage = None
        if "token_usage" in report_data:
            from autorubric.types import TokenUsage as TU

            token_usage = TU(
                prompt_tokens=report_data["token_usage"].get("prompt_tokens", 0),
                completion_tokens=report_data["token_usage"].get("completion_tokens", 0),
                total_tokens=report_data["token_usage"].get("total_tokens", 0),
            )

        report_type = report_data.get("report_type")
        criterion_reports_data = report_data.get("criterion_reports")

        if report_type == "ensemble" and criterion_reports_data is not None:
            report = _deserialize_ensemble_report(report_data, token_usage)
        elif report_type == "single" and criterion_reports_data is not None:
            report = _deserialize_single_report(report_data, token_usage)
        else:
            report = EvaluationReport(
                score=report_data["score"],
                raw_score=report_data.get("raw_score"),
                llm_raw_score=report_data.get("raw_score"),
                token_usage=token_usage,
                completion_cost=report_data.get("completion_cost"),
                error=report_data.get("error"),
                cannot_assess_count=report_data.get("cannot_assess_count", 0),
            )

        return cls(
            item_idx=data["item_idx"],
            item=item,
            report=report,
            duration_seconds=data["duration_seconds"],
            error=data.get("error"),
        )


@dataclass
class EvalTimingStats:
    """Timing statistics for the evaluation run."""

    total_duration_seconds: float
    mean_item_duration_seconds: float
    min_item_duration_seconds: float
    max_item_duration_seconds: float
    p50_item_duration_seconds: float
    p95_item_duration_seconds: float
    items_per_second: float

    @classmethod
    def from_durations(
        cls,
        durations: list[float],
        total_duration: float,
    ) -> EvalTimingStats:
        """Compute timing stats from a list of item durations."""
        if not durations:
            return cls(
                total_duration_seconds=total_duration,
                mean_item_duration_seconds=0.0,
                min_item_duration_seconds=0.0,
                max_item_duration_seconds=0.0,
                p50_item_duration_seconds=0.0,
                p95_item_duration_seconds=0.0,
                items_per_second=0.0,
            )

        sorted_durations = sorted(durations)
        n = len(sorted_durations)

        return cls(
            total_duration_seconds=total_duration,
            mean_item_duration_seconds=sum(durations) / n,
            min_item_duration_seconds=sorted_durations[0],
            max_item_duration_seconds=sorted_durations[-1],
            p50_item_duration_seconds=sorted_durations[n // 2],
            p95_item_duration_seconds=sorted_durations[min(int(n * 0.95), n - 1)],
            items_per_second=n / total_duration if total_duration > 0 else 0.0,
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize to dictionary."""
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "mean_item_duration_seconds": self.mean_item_duration_seconds,
            "min_item_duration_seconds": self.min_item_duration_seconds,
            "max_item_duration_seconds": self.max_item_duration_seconds,
            "p50_item_duration_seconds": self.p50_item_duration_seconds,
            "p95_item_duration_seconds": self.p95_item_duration_seconds,
            "items_per_second": self.items_per_second,
        }


@dataclass
class EvalResult:
    """Complete result from an evaluation run."""

    # Core results
    item_results: list[ItemResult]

    # Aggregated metrics
    total_items: int
    successful_items: int
    failed_items: int

    # Usage and cost
    total_token_usage: TokenUsage | None
    total_completion_cost: float | None

    # Timing
    timing_stats: EvalTimingStats
    started_at: datetime
    completed_at: datetime

    # Error details
    errors: list[tuple[int, str]] = field(default_factory=list)

    # Experiment info
    experiment_name: str | None = None
    experiment_dir: Path | None = None

    def get_scores(self) -> list[float]:
        """Extract scores from all successful results."""
        return [r.report.score for r in self.item_results if r.error is None]

    def get_reports(self) -> list[EvaluationReport | EnsembleEvaluationReport]:
        """Extract reports from all successful results."""
        return [r.report for r in self.item_results if r.error is None]

    def filter_successful(self) -> list[ItemResult]:
        """Get only successful item results."""
        return [r for r in self.item_results if r.error is None]

    def filter_failed(self) -> list[ItemResult]:
        """Get only failed item results."""
        return [r for r in self.item_results if r.error is not None]

    def compute_metrics(
        self,
        dataset: RubricDataset,
        *,
        bootstrap: bool = False,
        n_bootstrap: int = 1000,
        per_judge: bool = False,
        cannot_assess: Literal["exclude", "as_unmet"] = "exclude",
        na_mode: NAMode = "exclude",
        confidence_level: float = 0.95,
        seed: int | None = None,
    ) -> MetricsResult:
        """Compute comprehensive evaluation metrics against ground truth.

        This method compares predicted verdicts and scores against ground truth
        from the dataset, computing criterion-level agreement metrics, score
        correlations, and bias analysis.

        If eval_result does not contain all items from the dataset, metrics
        are computed only for the intersection, and a warning is included
        in the result.

        Args:
            dataset: The dataset with ground truth labels.
            bootstrap: If True, compute bootstrap confidence intervals (expensive).
            n_bootstrap: Number of bootstrap samples if bootstrap=True.
            per_judge: If True and ensemble, compute per-judge metrics.
            cannot_assess: How to handle CANNOT_ASSESS verdicts:
                - "exclude": Skip pairs where either is CANNOT_ASSESS (default)
                - "as_unmet": Treat CANNOT_ASSESS as UNMET
            na_mode: How to handle NA options in multi-choice criteria.
                Mirrors ``cannot_assess`` for binary:

                - "exclude": Skip pairs where either is NA (default).
                - "as_unmet": Remap NA → the score-minimizing non-NA option,
                  weight-sign aware (shares ``Criterion.worst_scored_option()``
                  with the grader's ``unknown``-error path).
                - "as_category": Keep NA as a distinct categorical column.
                  Refused for ordinal criteria with an NA option (raises
                  ``ValueError``).
            confidence_level: Confidence level for bootstrap CIs (default 0.95).
            seed: Random seed for bootstrap reproducibility.

        Returns:
            MetricsResult with comprehensive metrics. Use .summary() for
            formatted output or .to_dataframe() for export.

        Example:
            >>> result = await evaluate(dataset, grader)
            >>> metrics = result.compute_metrics(dataset)
            >>> print(metrics.summary())
            >>> print(f"Accuracy: {metrics.criterion_accuracy:.1%}")
            >>> df = metrics.to_dataframe()
        """
        from autorubric.metrics._compute import compute_metrics as _compute

        return _compute(
            self,
            dataset,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            per_judge=per_judge,
            cannot_assess=cannot_assess,
            na_mode=na_mode,
            confidence_level=confidence_level,
            seed=seed,
        )

    @classmethod
    def from_experiment(cls, experiment_path: Path | str) -> EvalResult:
        """Load EvalResult from a completed experiment directory.

        Args:
            experiment_path: Path to the experiment directory.

        Returns:
            EvalResult with loaded item results and statistics.

        Raises:
            FileNotFoundError: If experiment directory doesn't exist.
            ValueError: If manifest is invalid or experiment is incomplete.
        """
        exp_dir = Path(experiment_path)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        items_path = exp_dir / "items.jsonl"

        # Load manifest
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        # Load items
        item_results: list[ItemResult] = []
        if items_path.exists():
            with open(items_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Create a minimal DataItem for reconstruction
                        item = DataItem(
                            submission="",  # Submission not stored in items.jsonl
                            description=f"Item {data['item_idx']}",
                        )
                        item_results.append(ItemResult.from_dict(data, item))

        # Sort by item_idx
        item_results.sort(key=lambda r: r.item_idx)

        # Compute aggregated stats
        reports = [r.report for r in item_results if r.error is None]
        usages = [r.token_usage for r in reports if r.token_usage]
        costs = [r.completion_cost for r in reports if r.completion_cost is not None]

        total_usage = aggregate_token_usage(usages)
        total_cost = aggregate_completion_cost(costs)

        durations = [r.duration_seconds for r in item_results]
        timing_stats = EvalTimingStats.from_durations(
            durations, manifest.get("total_duration_seconds", 0.0)
        )

        errors = [(r.item_idx, r.error) for r in item_results if r.error]

        return cls(
            item_results=item_results,
            total_items=manifest["total_items"],
            successful_items=len(item_results) - len(errors),
            failed_items=len(errors),
            total_token_usage=total_usage,
            total_completion_cost=total_cost,
            timing_stats=timing_stats,
            started_at=datetime.fromisoformat(manifest["started_at"]),
            completed_at=datetime.fromisoformat(
                manifest.get("completed_at", manifest["started_at"])
            ),
            errors=errors,
            experiment_name=manifest["experiment_name"],
            experiment_dir=exp_dir,
        )


@dataclass
class ExperimentManifest:
    """Manifest for experiment checkpointing.

    Contains metadata about an evaluation run for reproducibility and resumption.
    """

    experiment_name: str
    created_at: datetime
    dataset_name: str | None
    dataset_hash: str
    total_items: int
    status: Literal["running", "completed", "failed"]
    completed_indices: set[int]
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_duration_seconds: float | None = None
    # Optional fields for reproducibility
    dataset_path: str | None = None  # Path if loaded from file
    grader_config: dict[str, Any] | None = None  # Serialized grader configuration
    eval_config: dict[str, Any] | None = None  # Serialized EvalConfig

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "experiment_name": self.experiment_name,
            "created_at": self.created_at.isoformat(),
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "total_items": self.total_items,
            "status": self.status,
            "completed_indices": list(self.completed_indices),
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "dataset_path": self.dataset_path,
            "grader_config": self.grader_config,
            "eval_config": self.eval_config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentManifest:
        """Deserialize from dictionary."""
        return cls(
            experiment_name=data["experiment_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            dataset_name=data.get("dataset_name"),
            dataset_hash=data["dataset_hash"],
            total_items=data["total_items"],
            status=data["status"],
            completed_indices=set(data.get("completed_indices", [])),
            error=data.get("error"),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            total_duration_seconds=data.get("total_duration_seconds"),
            dataset_path=data.get("dataset_path"),
            grader_config=data.get("grader_config"),
            eval_config=data.get("eval_config"),
        )


class EvalProgressDisplay:
    """Manages rich progress bars for evaluation runs."""

    def __init__(
        self,
        total_items: int,
        style: Literal["simple", "detailed"] = "simple",
        judge_ids: list[str] | None = None,
    ):
        self.total_items = total_items
        self.style = style
        self.judge_ids = judge_ids or []

        self._console = Console()
        self._progress: Progress | None = None
        self._live: Live | None = None

        # Task IDs
        self._main_task: TaskID | None = None
        self._judge_tasks: dict[str, TaskID] = {}

    def __enter__(self) -> EvalProgressDisplay:
        """Start the progress display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("({task.fields[rate]:.2f}/s)"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            expand=False,
        )

        # Create main progress task
        self._main_task = self._progress.add_task(
            "Evaluating",
            total=self.total_items,
            rate=0.0,
        )

        # Create per-judge tasks in detailed mode
        if self.style == "detailed" and self.judge_ids:
            for judge_id in self.judge_ids:
                self._judge_tasks[judge_id] = self._progress.add_task(
                    f"  {judge_id}",
                    total=self.total_items,
                    rate=0.0,
                )

        self._live = Live(
            self._progress,
            console=self._console,
            refresh_per_second=4,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the progress display."""
        if self._live:
            self._live.__exit__(*args)

    def advance(self, rate: float = 0.0) -> None:
        """Advance the main progress bar by one."""
        if self._progress and self._main_task is not None:
            self._progress.update(
                self._main_task,
                advance=1,
                rate=rate,
            )

    def set_status(self, status: str) -> None:
        """Update the main task description."""
        if self._progress and self._main_task is not None:
            self._progress.update(self._main_task, description=status)


class EvalRunner:
    """Runs batch evaluations with rate limiting and progress tracking.

    This class orchestrates the evaluation of a RubricDataset using a grader,
    handling:
    - Concurrent execution with configurable parallelism
    - Rate limiting via LLMConfig.max_parallel_requests
    - Progress display with rich progress bars
    - Checkpointing and resumption from failures
    - Result aggregation with timing statistics

    Example:
        >>> from autorubric import LLMConfig, RubricDataset
        >>> from autorubric.graders import CriterionGrader
        >>> from autorubric.eval import EvalRunner, EvalConfig
        >>>
        >>> dataset = RubricDataset.from_file("data.json")
        >>> grader = CriterionGrader(
        ...     llm_config=LLMConfig(
        ...         model="openai/gpt-4",
        ...         max_parallel_requests=10,
        ...     )
        ... )
        >>>
        >>> runner = EvalRunner(dataset=dataset, grader=grader)
        >>> result = await runner.run()
        >>> print(f"Evaluated {result.successful_items}/{result.total_items}")
    """

    def __init__(
        self,
        dataset: RubricDataset,
        grader: Grader,
        config: EvalConfig | None = None,
    ):
        """Initialize the evaluation runner.

        Args:
            dataset: The dataset to evaluate.
            grader: The grader to use for evaluation.
            config: Optional configuration. Uses defaults if not provided.
        """
        self.dataset = dataset
        self.grader = grader
        self.config = config or EvalConfig()

        # Extract judge IDs if using ensemble grader
        self._judge_ids: list[str] = []
        if hasattr(grader, "_judges"):
            self._judge_ids = [j.judge_id for j in grader._judges]

        # Resolve experiment name
        self._experiment_name = self.config.experiment_name or _generate_experiment_name()
        self._exp_dir = Path(self.config.experiments_dir) / self._experiment_name

    async def run(self) -> EvalResult:
        """Run the evaluation and return aggregated results.

        Returns:
            EvalResult with all item results, aggregated usage/cost,
            and timing statistics.

        Raises:
            RuntimeError: If fail_fast=True and any item fails.
        """
        started_at = datetime.now()
        start_time = time.perf_counter()

        # Set up experiment directory and load checkpoint if resuming
        completed_indices, previous_results = self._setup_experiment(started_at)

        # Determine pending items
        pending_items = [
            (idx, item) for idx, item in enumerate(self.dataset) if idx not in completed_indices
        ]

        item_results: list[ItemResult] = list(previous_results)
        errors: list[tuple[int, str]] = []
        completed_count = len(completed_indices)

        # Create progress display
        progress: EvalProgressDisplay | None = None
        if self.config.show_progress:
            progress = EvalProgressDisplay(
                total_items=len(self.dataset),
                style=self.config.progress_style,
                judge_ids=self._judge_ids,
            )

        try:
            if progress:
                progress.__enter__()
                # Update progress to show already completed items
                for _ in range(completed_count):
                    progress.advance()

            # Process remaining results as they complete
            async for result in self._run_with_streaming(pending_items):
                item_results.append(result)
                completed_count += 1

                # Persist result immediately
                self._append_item_result(result)
                self._update_manifest_indices(result.item_idx)

                if result.error:
                    errors.append((result.item_idx, result.error))
                    if self.config.fail_fast:
                        self._update_manifest_status("failed", error=result.error)
                        raise RuntimeError(
                            f"Evaluation failed at item {result.item_idx}: {result.error}"
                        )

                # Update progress
                if progress:
                    elapsed = time.perf_counter() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0.0
                    progress.advance(rate=rate)

        finally:
            if progress:
                progress.__exit__(None, None, None)

        # Sort results by item index
        item_results.sort(key=lambda r: r.item_idx)

        # Compute final metrics
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        completed_at = datetime.now()

        # Aggregate usage and cost
        reports = [r.report for r in item_results if r.error is None]
        usages = [r.token_usage for r in reports if r.token_usage]
        costs = [r.completion_cost for r in reports if r.completion_cost is not None]

        total_usage = aggregate_token_usage(usages)
        total_cost = aggregate_completion_cost(costs)

        # Compute timing stats
        durations = [r.duration_seconds for r in item_results]
        timing_stats = EvalTimingStats.from_durations(durations, total_duration)

        # Update manifest to completed
        self._update_manifest_status(
            "completed",
            completed_at=completed_at,
            total_duration=total_duration,
        )

        return EvalResult(
            item_results=item_results,
            total_items=len(self.dataset),
            successful_items=len(item_results) - len(errors),
            failed_items=len(errors),
            total_token_usage=total_usage,
            total_completion_cost=total_cost,
            timing_stats=timing_stats,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            experiment_name=self._experiment_name,
            experiment_dir=self._exp_dir,
        )

    def _setup_experiment(self, started_at: datetime) -> tuple[set[int], list[ItemResult]]:
        """Set up experiment directory and load checkpoint if resuming.

        Returns:
            Tuple of (completed_indices, previous_results).
        """
        completed_indices: set[int] = set()
        previous_results: list[ItemResult] = []

        if self._exp_dir.exists() and self.config.resume:
            # Load existing manifest
            manifest_path = self._exp_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = ExperimentManifest.from_dict(json.load(f))

                # Verify dataset hash
                current_hash = _compute_dataset_hash(self.dataset)
                if manifest.dataset_hash != current_hash:
                    logger.warning(
                        f"Dataset hash mismatch. Expected {manifest.dataset_hash}, "
                        f"got {current_hash}. Starting fresh."
                    )
                else:
                    completed_indices = manifest.completed_indices
                    # Load previous results
                    items_path = self._exp_dir / "items.jsonl"
                    if items_path.exists():
                        with open(items_path, encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    data = json.loads(line)
                                    idx = data["item_idx"]
                                    if idx < len(self.dataset):
                                        item = self.dataset[idx]
                                        previous_results.append(ItemResult.from_dict(data, item))
                    logger.info(
                        f"Resuming experiment {self._experiment_name} with "
                        f"{len(completed_indices)} completed items"
                    )
        else:
            # Create new experiment directory
            self._exp_dir.mkdir(parents=True, exist_ok=True)

            # Write initial manifest with full config for reproducibility
            manifest = ExperimentManifest(
                experiment_name=self._experiment_name,
                created_at=started_at,
                dataset_name=self.dataset.name,
                dataset_hash=_compute_dataset_hash(self.dataset),
                total_items=len(self.dataset),
                status="running",
                completed_indices=set(),
                started_at=started_at,
                grader_config=_serialize_grader_config(self.grader),
                eval_config=_serialize_eval_config(self.config),
            )
            self._write_manifest(manifest)

        return completed_indices, previous_results

    def _write_manifest(self, manifest: ExperimentManifest) -> None:
        """Write manifest to experiment directory."""
        manifest_path = self._exp_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def _update_manifest_indices(self, item_idx: int) -> None:
        """Update manifest with newly completed item index."""
        manifest_path = self._exp_dir / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        completed = set(data.get("completed_indices", []))
        completed.add(item_idx)
        data["completed_indices"] = list(completed)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _update_manifest_status(
        self,
        status: str,
        error: str | None = None,
        completed_at: datetime | None = None,
        total_duration: float | None = None,
    ) -> None:
        """Update manifest status."""
        manifest_path = self._exp_dir / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        data["status"] = status
        if error:
            data["error"] = error
        if completed_at:
            data["completed_at"] = completed_at.isoformat()
        if total_duration is not None:
            data["total_duration_seconds"] = total_duration

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _append_item_result(self, result: ItemResult) -> None:
        """Append item result to items.jsonl."""
        items_path = self._exp_dir / "items.jsonl"
        with open(items_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    async def _run_with_streaming(
        self, pending_items: list[tuple[int, DataItem]]
    ) -> AsyncIterator[ItemResult]:
        """Execute evaluation tasks and yield results as they complete.

        Uses asyncio.as_completed() to handle stragglers - results are
        yielded as soon as they finish rather than waiting for all tasks.
        """
        if not pending_items:
            return

        # Create all tasks
        tasks: dict[asyncio.Task[ItemResult], int] = {}

        if self.config.max_concurrent_items:
            # Use semaphore to limit concurrent items
            semaphore = asyncio.Semaphore(self.config.max_concurrent_items)

            async def limited_grade(idx: int, item: DataItem) -> ItemResult:
                async with semaphore:
                    return await self._grade_item(idx, item)

            for idx, item in pending_items:
                task = asyncio.create_task(limited_grade(idx, item))
                tasks[task] = idx
        else:
            # Run all in parallel
            for idx, item in pending_items:
                task = asyncio.create_task(self._grade_item(idx, item))
                tasks[task] = idx

        # Yield results as they complete (straggler handling)
        for coro in asyncio.as_completed(tasks.keys()):
            try:
                result = await coro
                yield result
            except Exception as e:
                # Find which task failed by checking done tasks
                for task, idx in tasks.items():
                    if task.done() and not task.cancelled():
                        try:
                            task.result()
                        except Exception:
                            yield ItemResult(
                                item_idx=idx,
                                item=self.dataset[idx],
                                report=self._create_error_report(str(e)),
                                duration_seconds=0.0,
                                error=str(e),
                            )
                            break

    async def _grade_item(self, idx: int, item: DataItem) -> ItemResult:
        """Grade a single item and wrap in ItemResult."""
        start = time.perf_counter()
        error: str | None = None

        # Use per-item rubric if available, otherwise fall back to global
        effective_rubric = self.dataset.get_item_rubric(idx)
        # Get effective reference submission (item-level takes precedence)
        reference = self.dataset.get_item_reference_submission(idx)

        try:
            report = await effective_rubric.grade(
                to_grade=item.submission,
                grader=self.grader,
                query=self.dataset.get_item_prompt(idx),
                reference_submission=reference,
            )
        except Exception as e:
            logger.warning(f"Error grading item {idx}: {e}")
            report = self._create_error_report(str(e))
            error = str(e)

        duration = time.perf_counter() - start

        return ItemResult(
            item_idx=idx,
            item=item,
            report=report,
            duration_seconds=duration,
            error=error,
        )

    def _create_error_report(self, error_msg: str) -> EvaluationReport:
        """Create an error report for failed items."""
        return EvaluationReport(
            score=0.0,
            raw_score=0.0,
            error=error_msg,
        )


async def evaluate(
    dataset: RubricDataset,
    grader: Grader,
    *,
    fail_fast: bool = False,
    show_progress: bool = True,
    progress_style: Literal["simple", "detailed"] = "simple",
    max_concurrent_items: int | None = None,
    experiment_name: str | None = None,
    experiments_dir: Path | str = "experiments",
    resume: bool = True,
) -> EvalResult:
    """Evaluate a dataset with a grader.

    Convenience wrapper around EvalRunner.

    Args:
        dataset: The dataset to evaluate.
        grader: The grader to use.
        fail_fast: Stop on first error if True.
        show_progress: Display progress bars if True.
        progress_style: "simple" or "detailed" progress display.
        max_concurrent_items: Limit concurrent items (None = unlimited).
        experiment_name: Name for this experiment run.
        experiments_dir: Root directory for experiment outputs.
        resume: If True and experiment exists, resume from checkpoint.

    Returns:
        EvalResult with all results and aggregated statistics.

    Example:
        >>> from autorubric.eval import evaluate
        >>> result = await evaluate(dataset, grader, show_progress=True)
        >>> print(f"Evaluated {result.successful_items}/{result.total_items}")
    """
    config = EvalConfig(
        fail_fast=fail_fast,
        show_progress=show_progress,
        progress_style=progress_style,
        max_concurrent_items=max_concurrent_items,
        experiment_name=experiment_name,
        experiments_dir=experiments_dir,
        resume=resume,
    )
    runner = EvalRunner(dataset=dataset, grader=grader, config=config)
    return await runner.run()
