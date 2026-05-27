#!/usr/bin/env python3
"""
Standalone demonstration of the autorubric library with criterion-level accuracy evaluation.

Dataset: 11 student essay responses to the prompt:
    "Explain the causes and effects of the Industrial Revolution."

Rubric: 5 criteria evaluating historical accuracy, structure, and analysis.

This demo shows the new streamlined metrics API:
- result.compute_metrics(dataset) computes all metrics in one call
- metrics.summary() provides formatted output
- metrics.to_dataframe() exports to pandas
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    LLMConfig,
    evaluate,
)
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader

load_dotenv()

# Path to the shared dataset
DATASET_PATH = Path(__file__).parent / "data" / "essay_grading_dataset.json"


async def main():
    # Load the dataset from JSON file
    dataset = RubricDataset.from_file(DATASET_PATH)

    # Configure LLM with thinking enabled and rate limiting
    llm_config = LLMConfig(
        model="gemini/gemini-2.5-flash",
        temperature=0.0,
        thinking="medium",  # Enable reasoning for better evaluation quality
        cache_enabled=True,
        max_parallel_requests=10,  # Rate limit to prevent API throttling
    )

    # Create grader with CANNOT_ASSESS handling
    grader = CriterionGrader(
        llm_config=llm_config,
        normalize=True,
        cannot_assess_config=CannotAssessConfig(
            strategy=CannotAssessStrategy.SKIP,  # Exclude unassessable from scoring
        ),
    )

    print("=" * 80)
    print("AutoRubric Demo: Criterion-Level Accuracy Evaluation")
    print(f"Model: {llm_config.model}")
    print(f"Thinking: {llm_config.thinking}")
    print(f"Max Parallel Requests: {llm_config.max_parallel_requests}")
    print("=" * 80)
    print(f"\nPrompt: {dataset.prompt}\n")
    print(f"Rubric Criteria (total positive weight: {dataset.total_positive_weight}):")
    for i, (criterion, name) in enumerate(zip(dataset.rubric.rubric, dataset.criterion_names)):
        weight_str = f"+{criterion.weight}" if criterion.weight > 0 else str(criterion.weight)
        print(f"  {i + 1}. {name:10} [{weight_str:>6}]")
    print("\n" + "-" * 80)

    # Run evaluation
    print(f"\nGrading {len(dataset)} items with EvalRunner...")
    print("(Checkpoints saved to experiments/ directory for resumption)\n")

    eval_result = await evaluate(
        dataset=dataset,
        grader=grader,
        show_progress=True,
        progress_style="simple",
        experiment_name=None,  # Auto-generate name
        resume=True,  # Resume from checkpoint if exists
    )

    if eval_result.experiment_dir:
        print(f"\nExperiment saved to: {eval_result.experiment_dir}")

    # Compute metrics with the new streamlined API
    metrics = eval_result.compute_metrics(
        dataset,
        bootstrap=True,  # Include confidence intervals
    )
    metrics.to_file(Path(eval_result.experiment_dir) / "metrics.json")

    # Print the formatted summary
    print("\n" + metrics.summary())

    # Show individual results
    print("\n" + "=" * 80)
    print("INDIVIDUAL RESULTS")
    print("=" * 80)

    for item_result in eval_result.item_results:
        result = item_result.report
        item = item_result.item

        true_score = dataset.compute_weighted_score(item.ground_truth)
        score_error = abs(result.score - true_score)

        print(f"\nItem {item_result.item_idx + 1}: {item.description}")
        print(
            f"  Scores: Predicted={result.score:.3f} | Actual={true_score:.3f} "
            f"| Error={score_error:.3f}"
        )
        if result.cannot_assess_count > 0:
            print(f"  Could not assess: {result.cannot_assess_count} criteria")
        print(f"  Duration: {item_result.duration_seconds:.2f}s")

        if result.token_usage:
            usage = result.token_usage
            cost_str = f"${result.completion_cost:.6f}" if result.completion_cost else "N/A"
            print(
                f"  LLM Usage: {usage.total_tokens:,} tokens "
                f"(prompt: {usage.prompt_tokens:,}, completion: {usage.completion_tokens:,})"
            )
            print(f"  Cost: {cost_str}")

    # Score comparison table
    print("\n" + "-" * 80)
    print("Score Comparison:")
    print(f"{'Item':<10} {'Predicted':>12} {'Actual':>12} {'Error':>12}")
    print("-" * 50)
    for item_result in eval_result.item_results:
        item = item_result.item
        pred = item_result.report.score
        actual = dataset.compute_weighted_score(item.ground_truth)
        error = abs(pred - actual)
        print(f"{item_result.item_idx + 1:<10} {pred:>12.3f} {actual:>12.3f} {error:>12.3f}")

    # EvalRunner timing stats
    print("\n" + "-" * 80)
    print("EVALUATION RUNNER STATS")
    print("-" * 80)

    timing = eval_result.timing_stats
    print(f"\nThroughput: {timing.items_per_second:.2f} items/second")
    print(f"Total Duration: {timing.total_duration_seconds:.2f}s")
    print(f"Mean Item Duration: {timing.mean_item_duration_seconds:.2f}s")
    print(f"P50 Item Duration: {timing.p50_item_duration_seconds:.2f}s")
    print(f"P95 Item Duration: {timing.p95_item_duration_seconds:.2f}s")

    # LLM Usage summary
    print("\n" + "-" * 80)
    print("LLM USAGE AND COST")
    print("-" * 80)

    total_usage = eval_result.total_token_usage
    total_cost = eval_result.total_completion_cost

    if total_usage:
        print("\nTotal Token Usage:")
        print(f"  Prompt tokens:     {total_usage.prompt_tokens:>12,}")
        print(f"  Completion tokens: {total_usage.completion_tokens:>12,}")
        print(f"  Total tokens:      {total_usage.total_tokens:>12,}")
        if total_usage.cache_creation_input_tokens > 0:
            print(f"  Cache created:     {total_usage.cache_creation_input_tokens:>12,}")
        if total_usage.cache_read_input_tokens > 0:
            print(f"  Cache hits:        {total_usage.cache_read_input_tokens:>12,}")
        print(f"\nTotal Cost: ${total_cost:.6f}" if total_cost else "\nTotal Cost: N/A")
        if total_cost:
            cost_per_item = total_cost / len(dataset)
            print(f"Cost per Item: ${cost_per_item:.6f}")
    else:
        print("\nNo usage data available (provider may not support usage tracking)")


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set. Set it to run the demo.")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nAlternatively, modify llm_config to use a different provider.")
        exit(1)

    asyncio.run(main())
