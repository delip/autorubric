"""Validation-aware rubric improvement engine.

Iteratively improves rubrics by optimizing for meta-rubric quality (validity)
and validation reliability. Supports two validation modes:

- **Ground-truth mode**: Items have ``ground_truth`` verdicts; measures
  Spearman rank correlation between rubric scores and expected scores.
- **Multi-judge mode**: Items lack ``ground_truth``; requires an ensemble
  of judges (``eval_llm`` as ``list[JudgeSpec]``) to measure inter-judge
  agreement.

Uses a Pareto constraint to reject revisions that improve quality but
decrease validation reliability.

Two-tier API:
    - ``improve_rubric()``: Convenience function with keyword shortcuts
    - ``ImprovementRunner``: Full control class following the EvalRunner pattern

Public building blocks for custom loops:
    - ``extract_issues``, ``validate_agreement``, ``revise_rubric``, ``diff_issues``
    - ``format_issues_for_prompt``, ``format_agreement_for_prompt``
    - ``format_ground_truth_for_prompt``, ``compute_expected_scores``
    - ``validate_ground_truth``, ``build_revision_history``, ``pareto_accept``
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader
from autorubric.graders.criterion_grader import JudgeSpec
from autorubric.llm import LLMClient, LLMConfig
from autorubric.rubric import Rubric
from autorubric.types import (
    CriterionVerdict,
    EnsembleEvaluationReport,
    TokenUsage,
)

from ._evaluate import evaluate_rubric_in_context, evaluate_rubric_standalone

logger = logging.getLogger(__name__)

ConvergenceFn = Callable[["IterationResult", list["IterationResult"]], str | None]
"""Custom convergence function type.

Called after each iteration with (current_result, all_results).
Returns a convergence reason string to stop, or None to continue.
When provided in ImprovementConfig, replaces the built-in convergence logic.
"""


# ============================================================================
# Types
# ============================================================================


@dataclass
class IssueDetail:
    """A single issue identified in a rubric by meta-rubric evaluation.

    Attributes:
        criterion_name: Name of the meta-rubric criterion that flagged this issue.
        requirement: The meta-rubric criterion's requirement text.
        weight: Weight of the meta-rubric criterion.
        is_antipattern: True if this is a negative criterion (anti-pattern detected).
        feedback: The judge's explanation for why this issue was flagged.
    """

    criterion_name: str
    requirement: str
    weight: float
    is_antipattern: bool
    feedback: str


@dataclass
class IterationResult:
    """Result from a single improvement iteration.

    Attributes:
        iteration: Zero-based iteration number.
        rubric: The rubric at this iteration.
        quality_score: Meta-rubric quality score (0-1).
        agreement: Mean inter-judge agreement (0-1), or None if not tested.
        per_criterion_agreement: Per-criterion agreement scores, or None.
        issues: List of issues identified in this iteration.
        issues_fixed: Names of issues present in previous iteration but not this one.
        issues_introduced: Names of issues not in previous iteration but present now.
        accepted: Whether this revision was accepted (Pareto check passed).
        rejection_reason: Why the revision was rejected, if applicable.
        quality_report: Full meta-rubric evaluation report.
        token_usage: Token usage for this iteration.
        completion_cost: Cost in USD for this iteration.
    """

    iteration: int
    rubric: Rubric
    quality_score: float
    agreement: float | None
    per_criterion_agreement: dict[str, float] | None
    issues: list[IssueDetail]
    issues_fixed: list[str]
    issues_introduced: list[str]
    accepted: bool
    rejection_reason: str | None
    quality_report: EnsembleEvaluationReport
    token_usage: TokenUsage | None
    completion_cost: float | None


@dataclass
class ImprovementResult:
    """Final result from the rubric improvement process.

    Attributes:
        original_rubric: The rubric before any improvements.
        final_rubric: The rubric after the last accepted iteration.
        iterations: All iteration results (including rejected ones).
        best_rubric: The rubric with the best combined quality+agreement.
        best_iteration: Index of the best iteration.
        convergence_reason: Why the improvement loop stopped.
        total_completion_cost: Total cost across all iterations.
    """

    original_rubric: Rubric
    final_rubric: Rubric
    iterations: list[IterationResult]
    best_rubric: Rubric
    best_iteration: int
    convergence_reason: str
    total_completion_cost: float | None


@dataclass
class ImprovementConfig:
    """Configuration for the rubric improvement process.

    Attributes:
        eval_llm: LLM configuration for meta-rubric evaluation and validation.
            - ``LLMConfig``: single judge (used in both ground-truth mode and
              meta-rubric evaluation).
            - ``list[JudgeSpec]``: ensemble (required for multi-judge mode;
              meta-rubric evaluation uses the first judge's config).
        revision_llm: LLM configuration for rubric revision.
        mode: Evaluation mode - "standalone" or "in_context".
        validation_data: Optional dataset for validation. When items have
            ``ground_truth``, uses ground-truth mode (Spearman ρ). When items
            lack ``ground_truth``, requires ``eval_llm`` as ``list[JudgeSpec]``
            for multi-judge agreement mode.
        max_iterations: Maximum number of improvement iterations.
        min_quality_score: Stop if quality score reaches this threshold.
        min_agreement: Stop if agreement/correlation reaches this threshold.
        score_plateau_threshold: Minimum score improvement to avoid plateau detection.
        plateau_patience: Number of iterations with no improvement before stopping.
        history_window: Number of recent iterations to include in revision prompt.
        reject_agreement_regression: Whether to reject revisions that decrease
            validation reliability.
        save_artifacts: Whether to save rubric JSONs and reports to disk.
        artifacts_dir: Directory for saved artifacts. Auto-generated if None.
        display: Output mode - "stdout", "html", or None.
        show_progress: Whether to show Rich progress indicators.
        max_total_cost: Stop if total cost exceeds this amount (USD).
        convergence_fn: Custom convergence function. When provided, replaces
            the built-in convergence logic entirely. Called after each iteration
            with (current_result, all_results). Returns a reason string to stop,
            or None to continue.
        revision_system_prompt: Custom system prompt for rubric revision LLM calls.
            Falls back to the default from prompts.py if None.
        revision_user_prompt_template: Custom user prompt template for revision.
            Must contain {task_prompt}, {original_criteria}, {issues_text},
            {validation_text}, {history_text} placeholders.
            Falls back to the default from prompts.py if None.
    """

    eval_llm: LLMConfig | list[JudgeSpec]
    revision_llm: LLMConfig
    mode: Literal["standalone", "in_context"] = "in_context"

    validation_data: RubricDataset | None = None

    max_iterations: int = 10
    min_quality_score: float = 0.95
    min_agreement: float = 0.85
    score_plateau_threshold: float = 0.02
    plateau_patience: int = 2

    history_window: int = 3
    reject_agreement_regression: bool = True

    save_artifacts: bool = True
    artifacts_dir: Path | str | None = None
    display: Literal["stdout", "html", None] = None
    show_progress: bool = True
    max_total_cost: float | None = None

    convergence_fn: ConvergenceFn | None = None
    revision_system_prompt: str | None = None
    revision_user_prompt_template: str | None = None


# ============================================================================
# Progress Display
# ============================================================================


def _rubric_to_lines(rubric: Rubric) -> list[str]:
    """Convert rubric criteria to text lines for diffing."""
    lines: list[str] = []
    for i, c in enumerate(rubric.rubric, 1):
        sign = "+" if c.weight > 0 else ""
        lines.append(f"{i}. [{sign}{c.weight}] {c.requirement}")
    return lines


def _build_inline_diff_line(
    prefix: str, line: str, other_line: str, *, is_add: bool
) -> Text:
    """Build a Rich Text line with two-tone background highlighting.

    Uses dark background for the full line and brighter background on the
    specific character spans that changed (VS Code-style inline diff).
    """
    base_style = "on #1f3d1f" if is_add else "on #3d1f1f"
    highlight_style = "on #306e30" if is_add else "on #6e3030"

    result = Text(prefix, style=base_style)

    if is_add:
        sm = difflib.SequenceMatcher(None, other_line, line)
        for op, _i1, _i2, j1, j2 in sm.get_opcodes():
            style = base_style if op == "equal" else highlight_style
            result.append(line[j1:j2], style=style)
    else:
        sm = difflib.SequenceMatcher(None, line, other_line)
        for op, i1, i2, _j1, _j2 in sm.get_opcodes():
            style = base_style if op == "equal" else highlight_style
            result.append(line[i1:i2], style=style)

    result.append("\n")
    return result


class ImprovementProgressDisplay:
    """Rich-based progress display for the improvement loop.

    Shows a progress bar during evaluation/agreement phases, prints one-line
    iteration summaries with issues tables and rubric panels, and renders
    a Rich Table at the end.
    """

    def __init__(self) -> None:
        self._console = Console(stderr=True)
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def begin_iteration(
        self, iteration: int, max_iterations: int, total_steps: int
    ) -> None:
        """Start a progress bar for one iteration."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn(
                f"[bold blue]Iteration {iteration + 1}/{max_iterations} "
                "[not bold]· {task.fields[phase_name]}"
            ),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            "", total=total_steps, phase_name="Evaluating quality"
        )

    def advance(self, phase_name: str | None = None) -> None:
        """Advance by 1 step, optionally updating the phase label."""
        if self._progress is not None and self._task_id is not None:
            update_kwargs: dict = {"advance": 1}
            if phase_name is not None:
                update_kwargs["phase_name"] = phase_name
            self._progress.update(self._task_id, **update_kwargs)

    def end_iteration(self) -> None:
        """Stop the progress bar (disappears due to transient=True)."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    @contextmanager
    def phase(self, iteration: int, max_iterations: int, phase_name: str):
        """Spinner-only context for single-step atomic phases (e.g. revision)."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn(
                f"[bold blue]Iteration {iteration + 1}/{max_iterations} "
                f"[not bold]· {phase_name}"
            ),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        with progress:
            progress.add_task("")
            yield

    def log_iteration(self, result: IterationResult) -> None:
        """Print a one-line iteration summary."""
        parts = [f"  Iter {result.iteration}:"]
        parts.append(f"quality={result.quality_score:.1%}")
        if result.agreement is not None:
            parts.append(f"agreement={result.agreement:.0%}")
        if not result.accepted:
            parts.append(f"[REJECTED: {result.rejection_reason}]")
        self._console.print(" ".join(parts))

    def log_issues_table(
        self,
        issues: list["IssueDetail"],
        *,
        rubric: "Rubric | None" = None,
    ) -> None:
        """Print a Rich table of issues found in this iteration."""
        if not issues:
            self._console.print("  [green]No issues found[/green]")
            return

        table = Table(
            title=f"{len(issues)} Issue{'s' if len(issues) != 1 else ''} Found",
            show_lines=False,
            padding=(0, 1),
        )
        table.add_column("Metarubric Criterion", style="cyan", max_width=30, no_wrap=True)
        table.add_column("Type", no_wrap=True)
        if rubric is not None:
            table.add_column("Rubric #", style="dim", no_wrap=True)
        table.add_column("Feedback", max_width=80)

        for issue in issues:
            if issue.is_antipattern:
                type_text = Text("ANTI-PATTERN", style="red")
            else:
                type_text = Text("QUALITY GAP", style="yellow")
            feedback = issue.feedback
            if len(feedback) > 80:
                feedback = feedback[:77] + "..."
            if rubric is not None:
                refs = _match_issue_to_criteria(issue, rubric)
                ref_str = ", ".join(f"#{r}" for r in refs) if refs else "--"
                table.add_row(issue.criterion_name, type_text, ref_str, feedback)
            else:
                table.add_row(issue.criterion_name, type_text, feedback)

        self._console.print(table)

    def log_rubric(self, rubric: "Rubric", iteration: int) -> None:
        """Print a Rich panel showing the rubric criteria."""
        table = Table(show_header=True, show_lines=False, padding=(0, 1))
        table.add_column("#", style="dim", justify="right", width=3)
        table.add_column("Weight", style="magenta", justify="right", width=7)
        table.add_column("Requirement", max_width=70)

        for i, c in enumerate(rubric.rubric, 1):
            sign = "+" if c.weight > 0 else ""
            table.add_row(str(i), f"{sign}{c.weight}", c.requirement)

        panel = Panel(
            table,
            title=f"Rubric · Iteration {iteration}",
            border_style="blue",
        )
        self._console.print(panel)

    def log_rubric_diff(
        self,
        prev_rubric: "Rubric",
        curr_rubric: "Rubric",
        iteration: int,
    ) -> None:
        """Print a paired before/after diff of rubric criteria between iterations.

        Changed lines are shown as adjacent old/new pairs with character-level
        highlighting of the exact changes.
        """
        prev_lines = _rubric_to_lines(prev_rubric)
        curr_lines = _rubric_to_lines(curr_rubric)

        sm = difflib.SequenceMatcher(None, prev_lines, curr_lines)
        opcodes = sm.get_opcodes()

        if all(tag == "equal" for tag, *_ in opcodes):
            self._console.print(
                Panel(
                    "[dim]No changes[/dim]",
                    title=f"Rubric · Iteration {iteration}",
                    border_style="blue",
                )
            )
            return

        styled = Text()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                for line in prev_lines[i1:i2]:
                    styled.append(f"  {line}\n", style="dim")
            elif tag == "replace":
                old_block = prev_lines[i1:i2]
                new_block = curr_lines[j1:j2]
                n_pairs = max(len(old_block), len(new_block))
                for k in range(n_pairs):
                    if k < len(old_block) and k < len(new_block):
                        styled.append_text(_build_inline_diff_line(
                            "- ", old_block[k], new_block[k], is_add=False,
                        ))
                        styled.append_text(_build_inline_diff_line(
                            "+ ", new_block[k], old_block[k], is_add=True,
                        ))
                    elif k < len(old_block):
                        styled.append(f"- {old_block[k]}\n", style="on #3d1f1f")
                    else:
                        styled.append(f"+ {new_block[k]}\n", style="on #1f3d1f")
            elif tag == "delete":
                for line in prev_lines[i1:i2]:
                    styled.append(f"- {line}\n", style="on #3d1f1f")
            elif tag == "insert":
                for line in curr_lines[j1:j2]:
                    styled.append(f"+ {line}\n", style="on #1f3d1f")

        panel = Panel(
            styled,
            title=f"Rubric · Iteration {iteration}",
            border_style="blue",
        )
        self._console.print(panel)

    def log_convergence(self, reason: str) -> None:
        """Print convergence reason."""
        self._console.print(f"  Converged: {reason}")

    def print_summary(
        self,
        iterations: list[IterationResult],
        convergence_reason: str,
        total_cost: float,
        artifacts_dir: Path | None,
    ) -> None:
        """Print a Rich Table summary and final statistics."""
        table = Table(title="Improvement Summary", show_lines=False)
        table.add_column("Iter", justify="right", style="cyan", no_wrap=True)
        table.add_column("Quality", justify="right")
        table.add_column("Agreement", justify="right")
        table.add_column("Issues", justify="right")
        table.add_column("Fixed", justify="right")
        table.add_column("New", justify="right")
        table.add_column("Status")

        for it in iterations:
            agreement_str = (
                f"{it.agreement:.0%}" if it.agreement is not None else "N/A"
            )
            fixed_str = str(len(it.issues_fixed)) if it.issues_fixed else "-"
            intro_str = str(len(it.issues_introduced)) if it.issues_introduced else "-"
            status = "Accepted" if it.accepted else f"Rejected: {it.rejection_reason}"
            table.add_row(
                str(it.iteration),
                f"{it.quality_score:.1%}",
                agreement_str,
                str(len(it.issues)),
                fixed_str,
                intro_str,
                status,
            )

        self._console.print()
        self._console.print(table)

        if len(iterations) >= 2:
            initial = iterations[0]
            final_accepted = next(
                (it for it in reversed(iterations) if it.accepted), iterations[-1]
            )
            self._console.print(
                f"\n  Quality: {initial.quality_score:.1%} -> "
                f"{final_accepted.quality_score:.1%}"
            )
            if (
                initial.agreement is not None
                and final_accepted.agreement is not None
            ):
                self._console.print(
                    f"  Agreement: {initial.agreement:.0%} -> "
                    f"{final_accepted.agreement:.0%}"
                )
            self._console.print(
                f"  Issues: {len(initial.issues)} -> "
                f"{len(final_accepted.issues)}"
            )

        self._console.print(f"  Convergence: {convergence_reason}")
        if total_cost > 0:
            self._console.print(f"  Total cost: ${total_cost:.4f}")
        if artifacts_dir:
            self._console.print(f"  Artifacts: {artifacts_dir}")


# ============================================================================
# Public Building Blocks
# ============================================================================


def extract_issues(report: EnsembleEvaluationReport) -> list[IssueDetail]:
    """Extract actionable issues from a meta-rubric evaluation report.

    An issue is either a positive criterion that is UNMET (quality gap)
    or a negative criterion that is MET (anti-pattern detected).
    """
    issues: list[IssueDetail] = []
    if report.report is None:
        return issues

    for criterion_report in report.report:
        weight = criterion_report.criterion.weight
        verdict = criterion_report.final_verdict

        is_issue = (weight > 0 and verdict == CriterionVerdict.UNMET) or (
            weight < 0 and verdict == CriterionVerdict.MET
        )

        if is_issue:
            issues.append(
                IssueDetail(
                    criterion_name=criterion_report.criterion.name or "unnamed",
                    requirement=criterion_report.criterion.requirement,
                    weight=weight,
                    is_antipattern=weight < 0,
                    feedback=criterion_report.final_reason,
                )
            )

    return issues


def diff_issues(
    prev_issues: list[IssueDetail], curr_issues: list[IssueDetail]
) -> tuple[list[str], list[str]]:
    """Compare issue sets to track which were fixed and which were introduced.

    Returns:
        Tuple of (issues_fixed, issues_introduced) as lists of criterion names.
    """
    prev_names = {i.criterion_name for i in prev_issues}
    curr_names = {i.criterion_name for i in curr_issues}

    fixed = sorted(prev_names - curr_names)
    introduced = sorted(curr_names - prev_names)
    return fixed, introduced


def format_issues_for_prompt(issues: list[IssueDetail]) -> str:
    """Format issues into text for the revision prompt."""
    if not issues:
        return "No issues found."

    lines: list[str] = []
    for i, issue in enumerate(issues, 1):
        issue_type = (
            "ANTI-PATTERN DETECTED" if issue.is_antipattern else "QUALITY GAP"
        )
        lines.append(f"{i}. [{issue_type}] {issue.criterion_name}")
        lines.append(f"   Feedback: {issue.feedback}")
        lines.append("")

    return "\n".join(lines)


def format_agreement_for_prompt(
    per_criterion_agreement: dict[str, float] | None,
) -> str:
    """Format per-criterion agreement data as a self-contained prompt section."""
    if not per_criterion_agreement:
        return ""

    lines: list[str] = ["## Inter-Judge Agreement"]
    for name, agreement in sorted(
        per_criterion_agreement.items(), key=lambda x: x[1]
    ):
        if agreement >= 0.8:
            level = "HIGH"
        elif agreement >= 0.6:
            level = "MEDIUM"
        else:
            level = "LOW"
        lines.append(f"  - {name}: {agreement:.0%} ({level})")
    lines.append(
        "Pay special attention to criteria with LOW agreement — "
        "these need clearer, more specific language."
    )

    return "\n".join(lines)


def compute_expected_scores(validation_data: RubricDataset) -> list[float]:
    """Compute expected scores from ground-truth verdicts and the rubric weights.

    Args:
        validation_data: Dataset whose items all have ``ground_truth``.

    Returns:
        List of expected scores, one per item.
    """
    rubric = validation_data.rubric
    if rubric is None:
        raise ValueError("validation_data must have a rubric to compute expected scores")
    return [
        rubric.compute_score(item.ground_truth)  # type: ignore[arg-type]
        for item in validation_data.items
    ]


async def validate_ground_truth(
    rubric: Rubric,
    validation_data: RubricDataset,
    expected_scores: list[float],
    grader: CriterionGrader,
    task_prompt: str | None = None,
    *,
    on_item_complete: Callable[[], None] | None = None,
    _capture: list | None = None,
    _item_reports: list | None = None,
) -> tuple[float, list[tuple[float, float]], float | None]:
    """Grade validation items with the current rubric and compare against expected scores.

    Uses Spearman rank correlation when n >= 3, falls back to ``1 - MAE``
    when n < 3.

    Args:
        rubric: Current rubric to evaluate.
        validation_data: Dataset with ground-truth verdicts.
        expected_scores: Pre-computed expected scores from ``compute_expected_scores``.
        grader: Grader configured from ``eval_llm``.
        task_prompt: Optional task prompt for grading context.
        on_item_complete: Callback invoked after each item is graded.
        _capture: When provided, per-item results are appended for artifact persistence.
        _item_reports: When provided, each item's ``EnsembleEvaluationReport``
            is appended for downstream diagnostics (e.g. grading reasons).

    Returns:
        Tuple of (correlation_metric, per_item_pairs, total_cost) where
        per_item_pairs is a list of (rubric_score, expected_score) tuples.
    """
    from scipy.stats import spearmanr

    rubric_scores: list[float] = []
    total_cost: float = 0.0

    for i, item in enumerate(validation_data.items):
        result = await rubric.grade(
            to_grade=item.submission,
            grader=grader,
            query=task_prompt or validation_data.prompt,
            reference_submission=(
                item.reference_submission
                or validation_data.reference_submission
            ),
        )
        rubric_scores.append(result.score)
        total_cost += result.completion_cost or 0.0

        if _item_reports is not None:
            _item_reports.append(result)

        if _capture is not None:
            _capture.append({
                "submission": item.submission[:200],
                "description": item.description,
                "rubric_score": result.score,
                "expected_score": expected_scores[i],
                "gap": result.score - expected_scores[i],
            })

        if on_item_complete is not None:
            on_item_complete()

    per_item = list(zip(rubric_scores, expected_scores))

    if len(rubric_scores) >= 3:
        corr, _ = spearmanr(rubric_scores, expected_scores)
        # Handle NaN (constant arrays) as 0
        if corr != corr:  # NaN check
            corr = 0.0
        metric = float(corr)
    else:
        mae = sum(abs(r - e) for r, e in per_item) / len(per_item)
        metric = 1.0 - mae

    return metric, per_item, total_cost if total_cost > 0 else None


def _select_diagnostic_items(
    per_item: list[tuple[float, float]],
    item_reports: list[EnsembleEvaluationReport],
    n_per_direction: int = 3,
) -> tuple[
    list[tuple[int, float, float, EnsembleEvaluationReport]],
    list[tuple[int, float, float, EnsembleEvaluationReport]],
]:
    """Select the items with the largest scoring gaps for diagnostics.

    Returns:
        (over_scored, under_scored) lists of
        ``(index, rubric_score, expected_score, report)`` tuples,
        sorted by descending ``|gap|``, capped at *n_per_direction* each.
    """
    over_scored: list[tuple[int, float, float, EnsembleEvaluationReport]] = []
    under_scored: list[tuple[int, float, float, EnsembleEvaluationReport]] = []

    for i, ((rubric_score, expected_score), report) in enumerate(
        zip(per_item, item_reports)
    ):
        gap = rubric_score - expected_score
        if gap > 0:
            over_scored.append((i, rubric_score, expected_score, report))
        elif gap < 0:
            under_scored.append((i, rubric_score, expected_score, report))

    over_scored.sort(key=lambda t: abs(t[1] - t[2]), reverse=True)
    under_scored.sort(key=lambda t: abs(t[1] - t[2]), reverse=True)

    return over_scored[:n_per_direction], under_scored[:n_per_direction]


def _format_error_criteria(
    report: EnsembleEvaluationReport,
    over_scored: bool,
) -> list[str]:
    """Extract per-criterion lines that explain the scoring error direction.

    For over-scored items: criteria that inflated the score
    (MET positive-weight or UNMET negative-weight).
    For under-scored items: criteria that deflated the score
    (UNMET positive-weight or MET negative-weight).
    """
    lines: list[str] = []
    if report.report is None:
        return lines

    for ecr in report.report:
        verdict = ecr.final_verdict
        if verdict is None or verdict == CriterionVerdict.CANNOT_ASSESS:
            continue

        weight = ecr.criterion.weight
        is_met = verdict == CriterionVerdict.MET

        if over_scored:
            include = (is_met and weight > 0) or (not is_met and weight < 0)
        else:
            include = (not is_met and weight > 0) or (is_met and weight < 0)

        if include:
            sign = "+" if weight > 0 else ""
            verdict_str = verdict.value
            name = ecr.criterion.name or ecr.criterion.requirement[:40]
            lines.append(
                f"    [w={sign}{weight}, {verdict_str}] {name}: "
                f"{ecr.final_reason}"
            )

    return lines


def format_ground_truth_for_prompt(
    correlation: float,
    per_item: list[tuple[float, float]],
    *,
    item_reports: list[EnsembleEvaluationReport] | None = None,
    n_diagnostic: int = 3,
) -> str:
    """Format ground-truth validation results as a self-contained prompt section.

    Args:
        correlation: Spearman ρ or 1-MAE metric.
        per_item: List of (rubric_score, expected_score) pairs.
        item_reports: Per-item grading reports from ``validate_ground_truth``.
            When provided, a diagnostics section is appended showing per-criterion
            reasons for the items with the largest scoring gaps.
        n_diagnostic: Number of items per direction (over/under-scored) to include
            in the diagnostics section.

    Returns:
        Formatted section string with header, data, and instructions.
    """
    lines: list[str] = [f"## Validation Against Ground Truth (Spearman ρ = {correlation:.2f})"]
    for i, (rubric_score, expected_score) in enumerate(per_item, 1):
        gap = rubric_score - expected_score
        lines.append(
            f"  - Submission {i}: rubric={rubric_score:.2f}, "
            f"expected={expected_score:.2f} (gap: {gap:+.2f})"
        )
    lines.append(
        "Adjust criteria to reduce scoring gaps on submissions "
        "that are over- or under-scored."
    )

    if item_reports is not None and len(item_reports) == len(per_item):
        over_scored, under_scored = _select_diagnostic_items(
            per_item, item_reports, n_per_direction=n_diagnostic,
        )
        if over_scored or under_scored:
            lines.append("")
            lines.append("## Grading Diagnostics for Largest Gaps")
            lines.append(
                "The following shows why the rubric over- or under-scored "
                "specific submissions."
            )
            lines.append(
                "For over-scored submissions, criteria listed were MET but "
                "likely shouldn't have been (or negative"
            )
            lines.append(
                "criteria were UNMET when they should catch errors). "
                "For under-scored submissions, criteria listed"
            )
            lines.append(
                "were UNMET but likely should have been MET "
                "(or negative criteria incorrectly penalized)."
            )

            if over_scored:
                lines.append("")
                lines.append("### Over-scored (rubric too high):")
                for idx, rscore, escore, report in over_scored:
                    gap = rscore - escore
                    lines.append(
                        f"  Submission {idx + 1} (rubric={rscore:.2f}, "
                        f"expected={escore:.2f}, gap={gap:+.2f}):"
                    )
                    criteria_lines = _format_error_criteria(report, over_scored=True)
                    lines.extend(criteria_lines)
                    if not criteria_lines:
                        lines.append("    (no error-direction criteria identified)")

            if under_scored:
                lines.append("")
                lines.append("### Under-scored (rubric too low):")
                for idx, rscore, escore, report in under_scored:
                    gap = rscore - escore
                    lines.append(
                        f"  Submission {idx + 1} (rubric={rscore:.2f}, "
                        f"expected={escore:.2f}, gap={gap:+.2f}):"
                    )
                    criteria_lines = _format_error_criteria(report, over_scored=False)
                    lines.extend(criteria_lines)
                    if not criteria_lines:
                        lines.append("    (no error-direction criteria identified)")

            lines.append("")
            lines.append(
                "Use these diagnostics to identify criteria whose wording "
                "is too lenient or too strict."
            )

    return "\n".join(lines)


def build_revision_history(
    iterations: list[IterationResult], window: int
) -> str:
    """Format recent iteration history for the revision prompt."""
    if not iterations:
        return "No previous iterations."

    recent = iterations[-window:]
    lines: list[str] = []
    for it in recent:
        agreement_str = f"{it.agreement:.0%}" if it.agreement is not None else "N/A"
        lines.append(f"Iteration {it.iteration}:")
        lines.append(f"  Quality: {it.quality_score:.1%}")
        lines.append(f"  Agreement: {agreement_str}")
        lines.append(f"  Issues: {len(it.issues)}")
        if it.issues_fixed:
            lines.append(f"  Fixed: {', '.join(it.issues_fixed)}")
        if it.issues_introduced:
            lines.append(f"  Introduced: {', '.join(it.issues_introduced)}")
        if not it.accepted:
            lines.append(f"  REJECTED: {it.rejection_reason}")
        lines.append("")

    return "\n".join(lines)


async def validate_agreement(
    rubric: Rubric,
    samples: list[str],
    judges: list[JudgeSpec],
    task_prompt: str | None = None,
    *,
    on_sample_complete: Callable[[], None] | None = None,
    _capture: list | None = None,
) -> tuple[float, dict[str, float], float | None]:
    """Test inter-judge agreement by grading samples with an ensemble.

    Args:
        rubric: Rubric to test.
        samples: Sample submissions to grade.
        judges: Judge specifications for the ensemble.
        task_prompt: Optional task prompt for context.
        on_sample_complete: Optional callback invoked after each sample is graded.
        _capture: When provided, per-sample ensemble reports are appended
            as serialized dicts for artifact persistence.

    Returns:
        Tuple of (mean_agreement, per_criterion_agreement, total_cost).
    """
    grader = CriterionGrader(judges=judges, aggregation="majority")

    all_agreements: list[float] = []
    per_criterion_totals: dict[str, list[float]] = {}
    total_cost: float = 0.0

    for sample in samples:
        result = await rubric.grade(
            to_grade=sample, grader=grader, query=task_prompt
        )
        if isinstance(result, EnsembleEvaluationReport) and result.report:
            all_agreements.append(result.mean_agreement)
            total_cost += result.completion_cost or 0.0
            for cr in result.report:
                name = cr.criterion.name or cr.criterion.requirement[:30]
                per_criterion_totals.setdefault(name, []).append(cr.agreement)

            if _capture is not None:
                from autorubric.eval import _serialize_ensemble_criterion_report

                _capture.append({
                    "submission": sample,
                    "score": result.score,
                    "mean_agreement": result.mean_agreement,
                    "completion_cost": result.completion_cost,
                    "criterion_reports": [
                        _serialize_ensemble_criterion_report(ecr)
                        for ecr in result.report
                    ],
                })

        if on_sample_complete is not None:
            on_sample_complete()

    if not all_agreements:
        return 0.0, {}, total_cost if total_cost > 0 else None

    mean_agreement = sum(all_agreements) / len(all_agreements)
    per_criterion_agreement = {
        name: sum(vals) / len(vals)
        for name, vals in per_criterion_totals.items()
    }
    return mean_agreement, per_criterion_agreement, total_cost if total_cost > 0 else None


def pareto_accept(
    curr_agreement: float | None,
    prev_agreement: float | None,
    reject_regression: bool,
    consecutive_rejections: int,
    epsilon: float = 0.03,
) -> tuple[bool, str | None]:
    """Check if a revision should be accepted under the Pareto constraint.

    A revision is accepted if agreement >= prev_agreement - epsilon.
    After 2 consecutive rejections, the constraint is relaxed.

    Returns:
        Tuple of (accepted, rejection_reason).
    """
    if not reject_regression:
        return True, None

    if curr_agreement is None or prev_agreement is None:
        return True, None

    if consecutive_rejections >= 2:
        return True, None

    if curr_agreement < prev_agreement - epsilon:
        return (
            False,
            f"Agreement regressed: {prev_agreement:.0%} -> {curr_agreement:.0%}",
        )

    return True, None


async def revise_rubric(
    rubric: Rubric,
    task_prompt: str | None,
    issues: list[IssueDetail],
    validation_text: str,
    history_text: str,
    config: ImprovementConfig,
    *,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    _capture: dict | None = None,
) -> tuple[Rubric, float | None]:
    """Use an LLM to revise the rubric based on evaluation feedback and validation data.

    Args:
        rubric: Current rubric to revise.
        task_prompt: The task the rubric evaluates.
        issues: Issues identified by meta-rubric evaluation.
        validation_text: Formatted validation data (ground-truth or agreement).
        history_text: Formatted revision history.
        config: Improvement configuration (provides revision_llm).
        system_prompt: Override system prompt. Falls back to
            config.revision_system_prompt, then the default from prompts.py.
        user_prompt_template: Override user prompt template. Falls back to
            config.revision_user_prompt_template, then the default from prompts.py.
        _capture: When provided, populated with the system prompt, user prompt,
            and LLM response text for artifact persistence.

    Returns:
        Tuple of (revised Rubric, completion cost or None).
    """
    from autorubric.prompts import (
        RUBRIC_REVISION_SYSTEM_PROMPT,
        RUBRIC_REVISION_USER_PROMPT_TEMPLATE,
    )

    effective_system_prompt = (
        system_prompt
        or config.revision_system_prompt
        or RUBRIC_REVISION_SYSTEM_PROMPT
    )
    effective_user_template = (
        user_prompt_template
        or config.revision_user_prompt_template
        or RUBRIC_REVISION_USER_PROMPT_TEMPLATE
    )

    original_criteria = json.dumps(
        [
            {
                "weight": c.weight,
                "requirement": c.requirement,
                **({"name": c.name} if c.name else {}),
            }
            for c in rubric.rubric
        ],
        indent=2,
    )

    user_prompt = effective_user_template.format(
        task_prompt=task_prompt or "(No specific task — standalone evaluation)",
        original_criteria=original_criteria,
        issues_text=format_issues_for_prompt(issues),
        validation_text=validation_text,
        history_text=history_text,
    )

    client = LLMClient(config.revision_llm)
    gen_result = await client.generate(
        effective_system_prompt, user_prompt, return_result=True
    )

    text = gen_result.content
    revision_cost = gen_result.cost

    if _capture is not None:
        _capture["system_prompt"] = effective_system_prompt
        _capture["user_prompt"] = user_prompt
        _capture["llm_response"] = text

    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Could not find JSON array in LLM response: {text[:200]}")

    criteria_data = json.loads(text[start:end])
    return Rubric.from_dict(criteria_data), revision_cost


# Backward-compatible private aliases
_extract_issues = extract_issues
_diff_issues = diff_issues
_format_issues_for_prompt = format_issues_for_prompt
_format_agreement_for_prompt = format_agreement_for_prompt
_build_revision_history = build_revision_history
_validate_agreement = validate_agreement
_pareto_accept = pareto_accept
_revise_rubric = revise_rubric


# ============================================================================
# Private Helpers
# ============================================================================


def _match_issue_to_criteria(issue: IssueDetail, rubric: Rubric) -> list[int]:
    """Best-effort match of a meta-rubric issue to specific rubric criteria.

    Primary method: parse the structured ``[Affects: #1, #3]`` tag that the
    meta-rubric judge is instructed to append.  Falls back to text heuristics
    (name matching, requirement substring, ``#N`` patterns) when the tag is
    absent.

    Returns:
        Sorted list of 1-based criterion indices the issue likely refers to.
    """
    n = len(rubric.rubric)
    feedback_lower = issue.feedback.lower()

    # --- Primary: structured [Affects: ...] tag ---
    affects_match = re.search(r"\[affects:\s*(.+?)\]", feedback_lower)
    if affects_match:
        content = affects_match.group(1).strip()
        if content == "all":
            return list(range(1, n + 1))
        indices = set()
        for num_match in re.finditer(r"#?(\d+)", content):
            num = int(num_match.group(1))
            if 1 <= num <= n:
                indices.add(num)
        if indices:
            return sorted(indices)

    # --- Fallback heuristics ---
    matches: set[int] = set()

    for i, criterion in enumerate(rubric.rubric, 1):
        if criterion.name and criterion.name.lower() in feedback_lower:
            matches.add(i)

        req_lower = criterion.requirement.lower()
        if len(req_lower) >= 20:
            for start in range(0, len(req_lower) - 19, 10):
                if req_lower[start : start + 20] in feedback_lower:
                    matches.add(i)
                    break

    for m in re.finditer(r"(?:criteri(?:on|a)\s+|#|index\s+)(\d+)", feedback_lower):
        num = int(m.group(1))
        if 1 <= num <= n:
            matches.add(num)

    return sorted(matches)


def _get_eval_llm_config(eval_llm: LLMConfig | list[JudgeSpec]) -> LLMConfig:
    """Extract a single LLMConfig from eval_llm for meta-rubric evaluation."""
    if isinstance(eval_llm, list):
        return eval_llm[0].llm_config
    return eval_llm


async def _evaluate_quality(
    rubric: Rubric,
    task_prompt: str | None,
    config: ImprovementConfig,
    *,
    html_path: str | None = None,
) -> EnsembleEvaluationReport:
    """Evaluate rubric quality using the appropriate meta-rubric."""
    display = "html" if html_path else config.display
    llm_config = _get_eval_llm_config(config.eval_llm)
    if config.mode == "in_context":
        if task_prompt is None:
            raise ValueError("task_prompt is required for in_context mode")
        return await evaluate_rubric_in_context(
            rubric,
            task_prompt,
            llm_config,
            display=display,
            output_html_path=html_path,
        )
    else:
        return await evaluate_rubric_standalone(
            rubric,
            llm_config,
            display=display,
            output_html_path=html_path,
        )


@dataclass
class _ConvergenceState:
    """Tracks convergence detection state across iterations."""

    plateau_count: int = 0
    prev_score: float = 0.0
    consecutive_rejections: int = 0


def _check_convergence(
    iteration: int,
    issues: list[IssueDetail],
    quality_score: float,
    agreement: float | None,
    config: ImprovementConfig,
    state: _ConvergenceState,
    total_cost: float | None,
) -> str | None:
    """Check if the improvement loop should stop.

    Returns:
        Convergence reason string, or None to continue.
    """
    if not issues:
        return "no_issues"

    quality_met = quality_score >= config.min_quality_score
    agreement_met = (
        agreement is not None and agreement >= config.min_agreement
    ) or config.validation_data is None
    if quality_met and agreement_met:
        return "thresholds_met"

    if iteration >= config.max_iterations - 1:
        return "max_iterations"

    improvement = quality_score - state.prev_score
    if abs(improvement) < config.score_plateau_threshold:
        state.plateau_count += 1
    else:
        state.plateau_count = 0

    if state.plateau_count >= config.plateau_patience:
        return "score_plateau"

    if state.consecutive_rejections >= 3:
        return "pareto_stuck"

    if config.max_total_cost is not None and total_cost is not None:
        if total_cost >= config.max_total_cost:
            return "cost_limit"

    state.prev_score = quality_score
    return None


def _save_rubric(rubric: Rubric, path: Path) -> None:
    """Save rubric criteria to a JSON file."""
    criteria = [
        {
            "weight": c.weight,
            "requirement": c.requirement,
            **({"name": c.name} if c.name else {}),
        }
        for c in rubric.rubric
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(criteria, f, indent=2)


def _serialize_iteration(iter_result: IterationResult) -> dict:
    """Serialize an IterationResult to a JSON-safe dict for artifact persistence."""
    from autorubric.eval import _serialize_ensemble_criterion_report

    qr = iter_result.quality_report
    quality_report_data: dict = {
        "score": qr.score,
        "raw_score": qr.raw_score,
        "mean_agreement": qr.mean_agreement,
        "completion_cost": qr.completion_cost,
    }
    if qr.report:
        quality_report_data["criterion_reports"] = [
            _serialize_ensemble_criterion_report(ecr) for ecr in qr.report
        ]
    if qr.judge_scores:
        quality_report_data["judge_scores"] = qr.judge_scores

    token_usage_data = None
    if iter_result.token_usage:
        tu = iter_result.token_usage
        token_usage_data = {
            "prompt_tokens": tu.prompt_tokens,
            "completion_tokens": tu.completion_tokens,
            "total_tokens": tu.total_tokens,
        }

    rubric_criteria = [
        {
            "weight": c.weight,
            "requirement": c.requirement,
            **({"name": c.name} if c.name else {}),
        }
        for c in iter_result.rubric.rubric
    ]

    return {
        "iteration": iter_result.iteration,
        "quality_score": iter_result.quality_score,
        "agreement": iter_result.agreement,
        "per_criterion_agreement": iter_result.per_criterion_agreement,
        "issues": [asdict(issue) for issue in iter_result.issues],
        "issues_fixed": iter_result.issues_fixed,
        "issues_introduced": iter_result.issues_introduced,
        "accepted": iter_result.accepted,
        "rejection_reason": iter_result.rejection_reason,
        "quality_report": quality_report_data,
        "rubric_criteria": rubric_criteria,
        "token_usage": token_usage_data,
        "completion_cost": iter_result.completion_cost,
    }


# ============================================================================
# ImprovementRunner
# ============================================================================


class ImprovementRunner:
    """Runs iterative rubric improvement with progress tracking.

    Following the EvalRunner pattern, this class orchestrates the full
    improvement loop: evaluate quality, test agreement, check convergence,
    revise rubric, repeat.

    Example:
        >>> from autorubric import LLMConfig, Rubric
        >>> from autorubric.meta import ImprovementRunner, ImprovementConfig
        >>>
        >>> config = ImprovementConfig(
        ...     eval_llm=LLMConfig(model="gpt-4o"),
        ...     revision_llm=LLMConfig(model="gpt-4o"),
        ... )
        >>> runner = ImprovementRunner(rubric, "Your task prompt", config=config)
        >>> result = await runner.run()
    """

    def __init__(
        self,
        rubric: Rubric,
        task_prompt: str | None = None,
        *,
        config: ImprovementConfig | None = None,
    ) -> None:
        if config is None:
            raise ValueError("config is required for ImprovementRunner")
        self.rubric = rubric
        self.task_prompt = task_prompt
        self.config = config

    async def run(self) -> ImprovementResult:
        """Run the improvement loop and return the result.

        Returns:
            ImprovementResult with the original, final, and best rubrics
            plus all iteration details.

        Raises:
            ValueError: If task_prompt is required but not provided.
        """
        config = self.config

        if config.mode == "in_context" and self.task_prompt is None:
            raise ValueError("task_prompt is required for in_context mode")

        # Set up artifacts directory
        artifacts_dir = self._setup_artifacts_dir()

        # Set up validation mode
        has_gt = False
        expected_scores: list[float] | None = None
        validation_grader: CriterionGrader | None = None
        n_validation_items = 0

        if config.validation_data is not None:
            has_gt = all(
                item.ground_truth is not None
                for item in config.validation_data.items
            )
            has_no_gt = all(
                item.ground_truth is None
                for item in config.validation_data.items
            )
            if not has_gt and not has_no_gt:
                raise ValueError(
                    "validation_data items must ALL have ground_truth or "
                    "NONE have ground_truth (mixed is not supported)"
                )
            if has_gt:
                expected_scores = compute_expected_scores(config.validation_data)
            else:
                if isinstance(config.eval_llm, LLMConfig):
                    raise ValueError(
                        "eval_llm must be list[JudgeSpec] with >= 2 judges "
                        "for multi-judge validation mode (items lack ground_truth)"
                    )
                if len(config.eval_llm) < 2:
                    raise ValueError(
                        "eval_llm must have >= 2 judges for multi-judge "
                        "validation mode"
                    )

            # Build grader from eval_llm
            if isinstance(config.eval_llm, list):
                validation_grader = CriterionGrader(judges=config.eval_llm)
            else:
                validation_grader = CriterionGrader(llm_config=config.eval_llm)
            n_validation_items = len(config.validation_data.items)

        # Set up progress display
        progress: ImprovementProgressDisplay | None = None
        if config.show_progress:
            progress = ImprovementProgressDisplay()

        original_rubric = self.rubric
        current_rubric = self.rubric
        iterations: list[IterationResult] = []
        convergence_state = _ConvergenceState()
        total_cost: float = 0.0
        best_score: float = -1.0
        best_iteration: int = 0
        best_rubric = self.rubric
        convergence_reason = "max_iterations"
        active_agreement: float | None = None

        for iteration in range(config.max_iterations):
            # Capture containers for artifact persistence
            revision_capture: dict | None = {} if artifacts_dir else None
            validation_capture: list | None = [] if artifacts_dir else None

            if not progress and config.display == "stdout":
                print(f"\n{'=' * 60}")
                print(f"ITERATION {iteration}: EVALUATING RUBRIC...")
                print(f"{'=' * 60}")

            # --- Evaluate quality ---
            html_path = (
                str(artifacts_dir / f"eval-iter-{iteration:02d}.html")
                if artifacts_dir
                else None
            )

            if progress:
                progress.begin_iteration(
                    iteration,
                    config.max_iterations,
                    1 + n_validation_items,
                )

            quality_report = await _evaluate_quality(
                current_rubric,
                self.task_prompt,
                config,
                html_path=html_path,
            )

            issues = extract_issues(quality_report)
            quality_score = quality_report.score
            iter_cost = quality_report.completion_cost or 0.0
            iter_usage = quality_report.token_usage

            if progress:
                next_phase = (
                    "Validating" if config.validation_data is not None
                    else "Done"
                )
                progress.advance(phase_name=next_phase)

            # --- Validation ---
            agreement: float | None = None
            per_criterion_agreement: dict[str, float] | None = None
            validation_text: str = ""
            validation_cost: float = 0.0

            if config.validation_data is not None and validation_grader is not None:
                if has_gt and expected_scores is not None:
                    item_reports: list[EnsembleEvaluationReport] = []
                    correlation, per_item, val_cost = await validate_ground_truth(
                        current_rubric,
                        config.validation_data,
                        expected_scores,
                        validation_grader,
                        task_prompt=self.task_prompt,
                        on_item_complete=(
                            progress.advance if progress else None
                        ),
                        _capture=validation_capture,
                        _item_reports=item_reports,
                    )
                    agreement = correlation
                    validation_text = format_ground_truth_for_prompt(
                        correlation, per_item, item_reports=item_reports,
                    )
                    validation_cost = val_cost or 0.0
                else:
                    # Multi-judge mode
                    samples = [
                        item.submission
                        for item in config.validation_data.items
                    ]
                    assert isinstance(config.eval_llm, list)
                    agr, per_crit, val_cost = await validate_agreement(
                        current_rubric,
                        samples,
                        config.eval_llm,
                        task_prompt=self.task_prompt,
                        on_sample_complete=(
                            progress.advance if progress else None
                        ),
                        _capture=validation_capture,
                    )
                    agreement = agr
                    per_criterion_agreement = per_crit
                    validation_text = format_agreement_for_prompt(per_crit)
                    validation_cost = val_cost or 0.0

            if progress:
                progress.end_iteration()

            # --- Diff issues ---
            prev_issues = iterations[-1].issues if iterations else []
            issues_fixed, issues_introduced = diff_issues(prev_issues, issues)

            # --- Pareto acceptance check ---
            accepted, rejection_reason = pareto_accept(
                agreement,
                active_agreement,
                config.reject_agreement_regression,
                convergence_state.consecutive_rejections,
            )

            if not accepted:
                convergence_state.consecutive_rejections += 1
            else:
                convergence_state.consecutive_rejections = 0
                active_agreement = agreement

            # --- Record iteration ---
            iter_cost += validation_cost
            iter_result = IterationResult(
                iteration=iteration,
                rubric=current_rubric,
                quality_score=quality_score,
                agreement=agreement,
                per_criterion_agreement=per_criterion_agreement,
                issues=issues,
                issues_fixed=issues_fixed,
                issues_introduced=issues_introduced,
                accepted=accepted,
                rejection_reason=rejection_reason,
                quality_report=quality_report,
                token_usage=iter_usage,
                completion_cost=iter_cost if iter_cost else None,
            )
            iterations.append(iter_result)
            total_cost += iter_cost

            # --- Save artifacts ---
            if artifacts_dir:
                _save_rubric(
                    current_rubric,
                    artifacts_dir / f"rubric-iter-{iteration:02d}.json",
                )

            # --- Display iteration results (persistent output) ---
            prev_rubric = (
                iterations[-2].rubric if len(iterations) >= 2 else None
            )
            if progress:
                progress.log_iteration(iter_result)
                if prev_rubric is not None:
                    progress.log_rubric_diff(
                        prev_rubric, current_rubric, iteration
                    )
                else:
                    progress.log_rubric(current_rubric, iteration)
                progress.log_issues_table(
                    iter_result.issues, rubric=current_rubric
                )
            elif config.display == "stdout":
                print(f"\n  Quality: {quality_score:.1%}")
                if agreement is not None:
                    print(f"  Agreement: {agreement:.0%}")
                if not accepted:
                    print(f"  REJECTED: {rejection_reason}")
                if prev_rubric is not None:
                    prev_lines = _rubric_to_lines(prev_rubric)
                    curr_lines = _rubric_to_lines(current_rubric)
                    diff = difflib.unified_diff(
                        prev_lines,
                        curr_lines,
                        fromfile=f"Iteration {iteration - 1}",
                        tofile=f"Iteration {iteration}",
                        lineterm="",
                    )
                    print(f"  Rubric diff:")
                    for line in diff:
                        print(f"    {line}")
                else:
                    print(f"  Rubric ({len(current_rubric.rubric)} criteria):")
                    for i, c in enumerate(current_rubric.rubric, 1):
                        sign = "+" if c.weight > 0 else ""
                        print(f"    {i}. [{sign}{c.weight}] {c.requirement}")
                for issue in issues:
                    tag = "ANTI-PATTERN" if issue.is_antipattern else "QUALITY GAP"
                    refs = _match_issue_to_criteria(issue, current_rubric)
                    ref_str = (
                        f" (#{', #'.join(str(r) for r in refs)})" if refs else ""
                    )
                    print(
                        f"  [{tag}] {issue.criterion_name}{ref_str}: "
                        f"{issue.feedback}"
                    )

            # --- Track best ---
            combined = quality_score
            if agreement is not None:
                combined = (quality_score + agreement) / 2
            if accepted and combined > best_score:
                best_score = combined
                best_iteration = iteration
                best_rubric = current_rubric

            # --- Check convergence ---
            if config.convergence_fn is not None:
                reason = config.convergence_fn(iter_result, iterations)
            else:
                reason = _check_convergence(
                    iteration,
                    issues,
                    quality_score,
                    agreement,
                    config,
                    convergence_state,
                    total_cost,
                )
            if reason is not None:
                convergence_reason = reason
                if progress:
                    progress.log_convergence(reason)
                elif config.display == "stdout":
                    print(f"\nConverged: {reason}")
                # Write per-iteration JSON (no revision for final iteration)
                if artifacts_dir:
                    iter_data = _serialize_iteration(iter_result)
                    if validation_capture:
                        iter_data["validation_samples"] = validation_capture
                    with open(
                        artifacts_dir / f"iter-{iteration:02d}.json",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(iter_data, f, indent=2, default=str)
                break

            # --- Revise rubric ---
            if not progress and config.display == "stdout":
                print(f"\nITERATION {iteration}: REVISING RUBRIC...")

            history_text = build_revision_history(
                iterations, config.history_window
            )

            # If the revision was rejected, revert to the last accepted rubric
            if not accepted:
                for prev in reversed(iterations[:-1]):
                    if prev.accepted:
                        current_rubric = prev.rubric
                        break

            if progress:
                with progress.phase(
                    iteration, config.max_iterations, "Revising rubric"
                ):
                    current_rubric, revision_cost = await revise_rubric(
                        current_rubric,
                        self.task_prompt,
                        issues,
                        validation_text,
                        history_text,
                        config,
                        _capture=revision_capture,
                    )
            else:
                current_rubric, revision_cost = await revise_rubric(
                    current_rubric,
                    self.task_prompt,
                    issues,
                    validation_text,
                    history_text,
                    config,
                    _capture=revision_capture,
                )
            total_cost += revision_cost or 0.0

            # --- Write per-iteration JSON ---
            if artifacts_dir:
                iter_data = _serialize_iteration(iter_result)
                if validation_capture:
                    iter_data["validation_samples"] = validation_capture
                if revision_capture:
                    iter_data["revision"] = revision_capture
                with open(
                    artifacts_dir / f"iter-{iteration:02d}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(iter_data, f, indent=2, default=str)

        # --- HTML improvement report ---
        if artifacts_dir:
            from ._display import render_improvement_report_html

            html = render_improvement_report_html(
                iterations,
                convergence_reason,
                total_cost,
                original_rubric,
                best_rubric,
            )
            (artifacts_dir / "improvement_report.html").write_text(
                html, encoding="utf-8"
            )

        # --- summary.json ---
        if artifacts_dir:
            def _rubric_to_criteria_list(rubric: Rubric) -> list[dict]:
                return [
                    {
                        "weight": c.weight,
                        "requirement": c.requirement,
                        **({"name": c.name} if c.name else {}),
                    }
                    for c in rubric.rubric
                ]

            summary = {
                "original_rubric": _rubric_to_criteria_list(original_rubric),
                "final_rubric": _rubric_to_criteria_list(best_rubric),
                "task_prompt": self.task_prompt,
                "convergence_reason": convergence_reason,
                "best_iteration": best_iteration,
                "total_iterations": len(iterations),
                "total_completion_cost": total_cost if total_cost > 0 else None,
                "config": {
                    "mode": config.mode,
                    "max_iterations": config.max_iterations,
                    "min_quality_score": config.min_quality_score,
                    "min_agreement": config.min_agreement,
                    "score_plateau_threshold": config.score_plateau_threshold,
                    "plateau_patience": config.plateau_patience,
                    "has_validation_data": config.validation_data is not None,
                    "validation_mode": (
                        "ground_truth" if has_gt
                        else "multi_judge" if config.validation_data is not None
                        else None
                    ),
                    "reject_agreement_regression": config.reject_agreement_regression,
                    "history_window": config.history_window,
                    "eval_llm_model": (
                        _get_eval_llm_config(config.eval_llm).model
                    ),
                    "revision_llm_model": config.revision_llm.model,
                },
                "iterations_summary": [
                    {
                        "iteration": it.iteration,
                        "quality_score": it.quality_score,
                        "agreement": it.agreement,
                        "num_issues": len(it.issues),
                        "num_fixed": len(it.issues_fixed),
                        "num_introduced": len(it.issues_introduced),
                        "accepted": it.accepted,
                        "rejection_reason": it.rejection_reason,
                        "completion_cost": it.completion_cost,
                    }
                    for it in iterations
                ],
            }
            with open(
                artifacts_dir / "summary.json", "w", encoding="utf-8"
            ) as f:
                json.dump(summary, f, indent=2, default=str)

        # --- Summary ---
        if progress:
            progress.print_summary(
                iterations, convergence_reason, total_cost, artifacts_dir
            )
        elif config.display == "stdout":
            print(f"\n{'=' * 60}")
            print("IMPROVEMENT SUMMARY")
            print(f"{'=' * 60}")
            self._print_summary_table_stdout(iterations)

            if len(iterations) >= 2:
                initial = iterations[0]
                final_accepted = next(
                    (it for it in reversed(iterations) if it.accepted),
                    iterations[-1],
                )
                print(
                    f"\n  Quality: {initial.quality_score:.1%} -> "
                    f"{final_accepted.quality_score:.1%}"
                )
                if (
                    initial.agreement is not None
                    and final_accepted.agreement is not None
                ):
                    print(
                        f"  Agreement: {initial.agreement:.0%} -> "
                        f"{final_accepted.agreement:.0%}"
                    )
                print(
                    f"  Issues: {len(initial.issues)} -> "
                    f"{len(final_accepted.issues)}"
                )

            print(f"  Convergence: {convergence_reason}")
            if total_cost > 0:
                print(f"  Total cost: ${total_cost:.4f}")
            if artifacts_dir:
                print(f"  Artifacts: {artifacts_dir}")

        final_rubric = best_rubric

        return ImprovementResult(
            original_rubric=original_rubric,
            final_rubric=final_rubric,
            iterations=iterations,
            best_rubric=best_rubric,
            best_iteration=best_iteration,
            convergence_reason=convergence_reason,
            total_completion_cost=total_cost if total_cost > 0 else None,
        )

    def _setup_artifacts_dir(self) -> Path | None:
        """Set up and return the artifacts directory, or None if disabled."""
        if not self.config.save_artifacts:
            return None
        if self.config.artifacts_dir is not None:
            artifacts_dir = Path(self.config.artifacts_dir)
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifacts_dir = Path("experiments") / f"rubric_improvement_{timestamp}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir

    @staticmethod
    def _print_summary_table_stdout(iterations: list[IterationResult]) -> None:
        """Fallback plain-text summary table when Rich progress is disabled."""
        header = (
            f"{'Iter':<6} {'Quality':<10} {'Agreement':<12} "
            f"{'Issues':<8} {'Fixed':<8} {'Introduced':<12}"
        )
        sep = (
            f"{'─' * 6} {'─' * 10} {'─' * 12} "
            f"{'─' * 8} {'─' * 8} {'─' * 12}"
        )
        print(f"\n{header}")
        print(sep)
        for it in iterations:
            agreement_str = (
                f"{it.agreement:.0%}" if it.agreement is not None else "N/A"
            )
            fixed_str = str(len(it.issues_fixed)) if it.issues_fixed else "-"
            intro_str = (
                str(len(it.issues_introduced)) if it.issues_introduced else "-"
            )
            status = "" if it.accepted else " [REJECTED]"
            print(
                f"{it.iteration:<6} {it.quality_score:<10.1%} {agreement_str:<12} "
                f"{len(it.issues):<8} {fixed_str:<8} {intro_str:<12}{status}"
            )


# ============================================================================
# Convenience API
# ============================================================================


def _build_config(
    config: ImprovementConfig | None,
    *,
    eval_llm: LLMConfig | list[JudgeSpec] | None = None,
    revision_llm: LLMConfig | None = None,
    validation_data: RubricDataset | None = None,
    max_iterations: int | None = None,
    display: Literal["stdout", "html", None] = None,
    artifacts_dir: Path | str | None = None,
    save_artifacts: bool | None = None,
    show_progress: bool | None = None,
    mode: Literal["standalone", "in_context"] | None = None,
    max_total_cost: float | None = None,
) -> ImprovementConfig:
    """Build an ImprovementConfig by merging kwargs over an optional base config.

    When ``config`` is provided, non-None kwargs override its fields.
    When ``config`` is None, creates a new config from kwargs (requires
    eval_llm and revision_llm).
    """
    overrides_map = {
        "eval_llm": eval_llm,
        "revision_llm": revision_llm,
        "validation_data": validation_data,
        "max_iterations": max_iterations,
        "display": display,
        "artifacts_dir": artifacts_dir,
        "save_artifacts": save_artifacts,
        "show_progress": show_progress,
        "mode": mode,
        "max_total_cost": max_total_cost,
    }
    overrides = {k: v for k, v in overrides_map.items() if v is not None}

    if config is not None:
        return replace(config, **overrides) if overrides else config

    if "eval_llm" not in overrides or "revision_llm" not in overrides:
        raise ValueError(
            "Either config or both eval_llm and revision_llm must be provided"
        )
    return ImprovementConfig(**overrides)


async def improve_rubric(
    rubric: Rubric,
    task_prompt: str | None = None,
    *,
    config: ImprovementConfig | None = None,
    eval_llm: LLMConfig | list[JudgeSpec] | None = None,
    revision_llm: LLMConfig | None = None,
    validation_data: RubricDataset | None = None,
    max_iterations: int | None = None,
    display: Literal["stdout", "html", None] = None,
    artifacts_dir: Path | str | None = None,
    save_artifacts: bool | None = None,
    show_progress: bool | None = None,
    mode: Literal["standalone", "in_context"] | None = None,
    max_total_cost: float | None = None,
) -> ImprovementResult:
    """Iteratively improve a rubric using meta-rubric evaluation and validation.

    Convenience wrapper around ImprovementRunner. Optimizes for both quality
    (meta-rubric score) and validation reliability. Uses a Pareto constraint
    to reject revisions that improve quality but decrease reliability.

    Args:
        rubric: The rubric to improve.
        task_prompt: The task the rubric evaluates (required for in_context mode).
        config: Full configuration. If provided, keyword shortcuts override
            its fields (non-None values only).
        eval_llm: LLM for meta-rubric evaluation and validation. Can be a
            single ``LLMConfig`` or ``list[JudgeSpec]`` for ensemble.
        revision_llm: LLM for rubric revision.
        validation_data: Dataset for validation (ground-truth or multi-judge mode).
        max_iterations: Maximum number of iterations.
        display: Output mode - "stdout", "html", or None.
        artifacts_dir: Directory for saved artifacts.
        save_artifacts: Whether to save rubric JSONs and reports to disk.
        show_progress: Whether to show Rich progress indicators.
        mode: Evaluation mode - "standalone" or "in_context".
        max_total_cost: Stop if total cost exceeds this amount (USD).

    Returns:
        ImprovementResult with the original, final, and best rubrics plus
        all iteration details.

    Raises:
        ValueError: If neither config nor eval_llm/revision_llm are provided.
    """
    cfg = _build_config(
        config,
        eval_llm=eval_llm,
        revision_llm=revision_llm,
        validation_data=validation_data,
        max_iterations=max_iterations,
        display=display,
        artifacts_dir=artifacts_dir,
        save_artifacts=save_artifacts,
        show_progress=show_progress,
        mode=mode,
        max_total_cost=max_total_cost,
    )
    runner = ImprovementRunner(rubric, task_prompt, config=cfg)
    return await runner.run()
