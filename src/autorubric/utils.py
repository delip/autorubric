"""Utility functions for autorubric."""

from __future__ import annotations

import asyncio
import re
import warnings
from typing import TYPE_CHECKING

from autorubric.types import LengthPenalty, ThinkingOutputDict, ToGradeInput, TokenUsage

if TYPE_CHECKING:
    from autorubric.dataset import DataItem, RubricDataset
    from autorubric.graders.base import Grader
    from autorubric.types import (
        Criterion,
        CriterionReport,
        CriterionVerdict,
        EnsembleCriterionReport,
        EnsembleEvaluationReport,
        EvaluationReport,
    )


def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in text.

    This is the default counting function used by LengthPenalty.
    For more accurate token counting with a specific model, provide a custom
    count_fn that uses a tokenizer.
    """
    return len(text.split())


def parse_thinking_output(text: str) -> ThinkingOutputDict:
    """Parse thinking and output sections from text with XML-style markers.

    Looks for <thinking>...</thinking> and <output>...</output> markers.
    If markers are not found, treats the entire text as output.

    Args:
        text: Text potentially containing thinking/output markers.

    Returns:
        Dict with 'thinking' and 'output' keys. Empty strings if sections not found.

    Examples:
        >>> parse_thinking_output("<thinking>ABC</thinking><output>DEF</output>")
        {'thinking': 'ABC', 'output': 'DEF'}

        >>> parse_thinking_output("Just output text")
        {'thinking': '', 'output': 'Just output text'}

        >>> parse_thinking_output("<thinking>Think</thinking>Rest")
        {'thinking': 'Think', 'output': 'Rest'}
    """
    # Try to extract thinking section
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
    thinking = thinking_match.group(1).strip() if thinking_match else ""

    # Try to extract output section
    output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL | re.IGNORECASE)

    if output_match:
        # Explicit output markers found
        output = output_match.group(1).strip()
    elif thinking_match:
        # Has thinking but no output markers - treat rest as output
        # Remove the thinking section and use remainder
        output = re.sub(
            r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
    else:
        # No markers at all - treat entire text as output
        output = text

    return ThinkingOutputDict(thinking=thinking, output=output)


def normalize_to_grade_input(to_grade: ToGradeInput) -> ThinkingOutputDict:
    """Normalize to_grade input to dict format.

    Args:
        to_grade: Either a string (with optional markers) or a dict.

    Returns:
        Dict with 'thinking' and 'output' keys.

    Raises:
        ValueError: If dict format is invalid (missing keys, wrong types).
    """
    if isinstance(to_grade, str):
        return parse_thinking_output(to_grade)

    # Handle dict input
    if not isinstance(to_grade, dict):
        raise ValueError(f"to_grade must be a string or dict, got {type(to_grade).__name__}")

    # Validate dict has correct keys
    thinking = to_grade.get("thinking", "")
    output = to_grade.get("output", "")

    # Validate types
    if not isinstance(thinking, str):
        raise ValueError(f"'thinking' must be a string, got {type(thinking).__name__}")
    if not isinstance(output, str):
        raise ValueError(f"'output' must be a string, got {type(output).__name__}")

    # Warn if dict has unexpected keys
    expected_keys = {"thinking", "output"}
    extra_keys = set(to_grade.keys()) - expected_keys
    if extra_keys:
        warnings.warn(
            f"Unexpected keys in to_grade dict: {extra_keys}. "
            f"Only 'thinking' and 'output' are used.",
            UserWarning,
        )

    return ThinkingOutputDict(thinking=thinking, output=output)


def compute_length_penalty(text: str | ThinkingOutputDict, config: LengthPenalty) -> float:
    """Compute the length penalty for the given text based on the config.

    The penalty follows an exponential curve:
    - Returns 0 if word/token count is at or below free_budget
    - Returns penalty_at_cap if count is at or above max_cap
    - Returns an interpolated value between those bounds using the exponent

    Args:
        text: Either a string (backwards compatible) or a dict with 'thinking'
            and 'output' keys. When a string is provided, it's treated as
            all output (no thinking section).
        config: LengthPenalty configuration specifying thresholds, penalty,
            and which sections to count based on penalty_type.

    Returns:
        A penalty value between 0 and penalty_at_cap to subtract from the score.
    """
    # Normalize input to dict format
    if isinstance(text, str):
        # Backwards compatibility: treat string as output only
        text_dict = ThinkingOutputDict(thinking="", output=text)
    else:
        text_dict = text

    # Select which text to count based on penalty_type
    if config.penalty_type == "ALL":
        # Concatenate both sections (with space to avoid word merging)
        text_to_count = text_dict.get("thinking", "") + " " + text_dict.get("output", "")
    elif config.penalty_type == "OUTPUT_ONLY":
        text_to_count = text_dict.get("output", "")
    elif config.penalty_type == "THINKING_ONLY":
        text_to_count = text_dict.get("thinking", "")
    else:
        raise ValueError(
            f"Invalid penalty_type: {config.penalty_type}. "
            f"Must be 'ALL', 'OUTPUT_ONLY', or 'THINKING_ONLY'."
        )

    # Count tokens/words
    count_fn = config.count_fn if config.count_fn is not None else word_count
    count = count_fn(text_to_count)

    # Apply penalty curve
    if count <= config.free_budget:
        return 0.0
    if count >= config.max_cap:
        return config.penalty_at_cap

    frac = (count - config.free_budget) / float(config.max_cap - config.free_budget)
    return config.penalty_at_cap * (frac**config.exponent)


# ============================================================================
# Usage and Cost Aggregation Helpers
# ============================================================================


def aggregate_token_usage(usages: list[TokenUsage | None]) -> TokenUsage | None:
    """Aggregate multiple TokenUsage objects into a single total.

    Useful for combining usage from multiple LLM calls or multiple grading operations.

    Args:
        usages: List of TokenUsage objects (None values are filtered out).

    Returns:
        A single TokenUsage with summed values, or None if all inputs are None.

    Example:
        >>> from autorubric import TokenUsage
        >>> usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        >>> usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        >>> total = aggregate_token_usage([usage1, usage2])
        >>> print(f"Total tokens: {total.total_tokens}")
        Total tokens: 450
    """
    valid_usages = [u for u in usages if u is not None]
    if not valid_usages:
        return None
    return sum(valid_usages, TokenUsage())


def aggregate_completion_cost(costs: list[float | None]) -> float | None:
    """Aggregate multiple completion costs into a single total.

    Useful for combining costs from multiple LLM calls or multiple grading operations.

    Args:
        costs: List of cost values in USD (None values are filtered out).

    Returns:
        Total cost in USD, or None if all inputs are None.

    Example:
        >>> costs = [0.001, 0.002, None, 0.003]
        >>> total = aggregate_completion_cost(costs)
        >>> print(f"Total cost: ${total:.4f}")
        Total cost: $0.0060
    """
    valid_costs = [c for c in costs if c is not None]
    if not valid_costs:
        return None
    return sum(valid_costs)


def aggregate_evaluation_usage(
    reports: list["EvaluationReport"],
) -> tuple[TokenUsage | None, float | None]:
    """Aggregate usage and cost from multiple EvaluationReports.

    Useful for batch grading operations where you want to track total resource usage.

    Args:
        reports: List of EvaluationReport objects from grading operations.

    Returns:
        Tuple of (total_token_usage, total_completion_cost).
        Either value may be None if no usage data was available.

    Example:
        >>> # After batch grading
        >>> results = await asyncio.gather(*[rubric.grade(...) for item in items])
        >>> total_usage, total_cost = aggregate_evaluation_usage(results)
        >>> if total_usage:
        ...     print(f"Total tokens used: {total_usage.total_tokens}")
        >>> if total_cost:
        ...     print(f"Total cost: ${total_cost:.4f}")
    """
    usages = [r.token_usage for r in reports]
    costs = [r.completion_cost for r in reports]
    return aggregate_token_usage(usages), aggregate_completion_cost(costs)


# ============================================================================
# Ground Truth Generation
# ============================================================================


def _extract_ground_truth_from_report(
    report: "EvaluationReport | EnsembleEvaluationReport",
    criteria: list["Criterion"],
) -> list["CriterionVerdict | str"]:
    """Extract ground truth values from an evaluation report.

    Converts grading report verdicts to the ground_truth format expected by DataItem.

    Args:
        report: The evaluation report from grading.
        criteria: List of criteria from the rubric.

    Returns:
        List of ground truth values:
        - CriterionVerdict for binary criteria
        - str (option label) for multi-choice criteria

    Raises:
        ValueError: If report is missing or malformed.
    """
    from autorubric.types import CriterionVerdict

    if report.report is None:
        raise ValueError("Report has no criterion-level breakdown")

    if len(report.report) != len(criteria):
        raise ValueError(
            f"Report has {len(report.report)} criteria but rubric has {len(criteria)}"
        )

    ground_truth: list[CriterionVerdict | str] = []

    for i, cr in enumerate(report.report):
        criterion = criteria[i]

        if criterion.is_binary:
            # Binary criterion: extract CriterionVerdict
            if hasattr(cr, "final_verdict") and cr.final_verdict is not None:
                # EnsembleCriterionReport
                ground_truth.append(cr.final_verdict)
            elif hasattr(cr, "verdict") and cr.verdict is not None:
                # CriterionReport
                ground_truth.append(cr.verdict)
            else:
                raise ValueError(
                    f"Could not extract binary verdict for criterion {i} "
                    f"({criterion.name or 'unnamed'})"
                )
        else:
            # Multi-choice criterion: extract label string (not index)
            if (
                hasattr(cr, "final_multi_choice_verdict")
                and cr.final_multi_choice_verdict is not None
            ):
                # EnsembleCriterionReport
                ground_truth.append(cr.final_multi_choice_verdict.selected_label)
            elif (
                hasattr(cr, "multi_choice_verdict")
                and cr.multi_choice_verdict is not None
            ):
                # CriterionReport
                ground_truth.append(cr.multi_choice_verdict.selected_label)
            else:
                raise ValueError(
                    f"Could not extract multi-choice verdict for criterion {i} "
                    f"({criterion.name or 'unnamed'})"
                )

    return ground_truth


async def fill_ground_truth(
    dataset: "RubricDataset",
    grader: "Grader",
    *,
    force: bool = False,
    show_progress: bool = True,
    max_concurrent_items: int | None = None,
) -> "RubricDataset":
    """Generate ground truth labels for dataset items using an LLM grader.

    Uses the provided grader to evaluate each item and extracts the verdicts
    to populate ground_truth. This is useful for creating synthetic ground
    truth labels when manual annotation is impractical.

    Args:
        dataset: The dataset to fill ground truth for.
        grader: The grader to use for generating verdicts.
        force: If True, re-grade all items. If False (default), only grade items
            where ground_truth is None.
        show_progress: Whether to display progress bars. Default True.
        max_concurrent_items: Maximum items to grade concurrently.
            None = grade all items in parallel (default).

    Returns:
        A new RubricDataset with ground_truth filled in. Items that fail to
        grade are excluded from the returned dataset. Items with existing
        ground_truth (when force=False) are included unchanged.

    Raises:
        ValueError: If dataset has no items.

    Example:
        >>> from autorubric import RubricDataset, LLMConfig
        >>> from autorubric.graders import CriterionGrader
        >>> from autorubric.utils import fill_ground_truth
        >>>
        >>> dataset = RubricDataset.from_file("unlabeled.json")
        >>> grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4o"))
        >>> labeled = await fill_ground_truth(dataset, grader)
        >>> labeled.to_file("labeled.json")
    """
    from autorubric.dataset import DataItem, RubricDataset

    if len(dataset) == 0:
        raise ValueError("Dataset has no items")

    # Partition items
    items_to_grade: list[tuple[int, DataItem]] = []
    preserved_items: dict[int, DataItem] = {}

    for idx, item in enumerate(dataset.items):
        if force or item.ground_truth is None:
            items_to_grade.append((idx, item))
        else:
            preserved_items[idx] = item

    # Grade items that need it
    graded_items: dict[int, DataItem] = {}

    if items_to_grade:

        async def grade_item(
            idx: int, item: DataItem
        ) -> tuple[int, DataItem | None, str | None]:
            try:
                # Use per-item rubric if available, otherwise fall back to global
                effective_rubric = dataset.get_item_rubric(idx)
                # Get effective reference submission (item-level takes precedence)
                reference = dataset.get_item_reference_submission(idx)
                report = await effective_rubric.grade(
                    to_grade=item.submission,
                    grader=grader,
                    query=dataset.prompt,
                    reference_submission=reference,
                )
                gt = _extract_ground_truth_from_report(report, effective_rubric.rubric)
                new_item = DataItem(
                    submission=item.submission,
                    description=item.description,
                    ground_truth=gt,
                    rubric=item.rubric,  # Preserve per-item rubric
                    reference_submission=item.reference_submission,  # Preserve reference
                )
                return (idx, new_item, None)
            except Exception as e:
                return (idx, None, str(e))

        # Create tasks with optional concurrency limit
        if max_concurrent_items:
            semaphore = asyncio.Semaphore(max_concurrent_items)

            async def limited_grade(
                idx: int, item: DataItem
            ) -> tuple[int, DataItem | None, str | None]:
                async with semaphore:
                    return await grade_item(idx, item)

            tasks = [limited_grade(idx, item) for idx, item in items_to_grade]
        else:
            tasks = [grade_item(idx, item) for idx, item in items_to_grade]

        # Execute with optional progress
        if show_progress:
            try:
                from rich.console import Console
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                )

                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Filling ground truth"),
                    BarColumn(bar_width=40),
                    MofNCompleteColumn(),
                    console=Console(stderr=True),
                )

                with progress:
                    task_id = progress.add_task("Grading", total=len(tasks))
                    for coro in asyncio.as_completed(tasks):
                        idx, new_item, error = await coro
                        if new_item is not None:
                            graded_items[idx] = new_item
                        progress.update(task_id, advance=1)
            except ImportError:
                # Fall back to no progress if rich is not available
                results = await asyncio.gather(*tasks)
                for idx, new_item, error in results:
                    if new_item is not None:
                        graded_items[idx] = new_item
        else:
            results = await asyncio.gather(*tasks)
            for idx, new_item, error in results:
                if new_item is not None:
                    graded_items[idx] = new_item

    # Combine preserved and graded items, maintaining order
    all_items: dict[int, DataItem] = {**preserved_items, **graded_items}
    ordered_items = [all_items[i] for i in sorted(all_items.keys())]

    return RubricDataset(
        prompt=dataset.prompt,
        rubric=dataset.rubric,
        items=ordered_items,
        name=dataset.name,
        reference_submission=dataset.reference_submission,
    )
