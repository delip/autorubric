#!/usr/bin/env python3
"""
Ensemble grading demonstration using multiple LLM judges.

Uses the same Industrial Revolution essay dataset as essay_grading_demo.py,
loaded from examples/data/essay_grading_dataset.json.

Judges:
- Gemini 2.5 Flash (fast, cost-effective)
- Claude Sonnet 4.5 (balanced quality)
- GPT-5.2 (high quality)

This demonstrates:
- Multi-model ensemble grading
- The streamlined metrics API with per_judge=True
- Agreement metrics across judges
- Cost comparison vs. single-model grading
"""

import asyncio
import os
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    LLMConfig,
    evaluate,
)
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader, JudgeSpec

load_dotenv()

litellm.suppress_debug_info = True

# Path to the shared dataset
DATASET_PATH = Path(__file__).parent / "data" / "essay_grading_dataset.json"


async def main():
    # Load the dataset from JSON file
    dataset = RubricDataset.from_file(DATASET_PATH)

    # Configure ensemble judges with rate limiting
    judges = [
        JudgeSpec(
            llm_config=LLMConfig(
                model="gemini/gemini-2.5-flash",
                temperature=0.0,
                max_parallel_requests=10,
            ),
            judge_id="gemini-flash",
            weight=1.0,
        ),
        JudgeSpec(
            llm_config=LLMConfig(
                model="anthropic/claude-sonnet-4-5-20250929",
                temperature=0.0,
                max_parallel_requests=10,
            ),
            judge_id="claude-sonnet",
            weight=1.0,
        ),
        JudgeSpec(
            llm_config=LLMConfig(
                model="openai/gpt-5.2",
                temperature=0.0,
                max_parallel_requests=10,
            ),
            judge_id="gpt-5.2",
            weight=1.2,
        ),
    ]

    grader = CriterionGrader(
        judges=judges,
        aggregation="majority",  # Use majority voting
        normalize=True,
        cannot_assess_config=CannotAssessConfig(
            strategy=CannotAssessStrategy.SKIP,
        ),
    )

    print("=" * 80)
    print("Ensemble Grading Demo: Multi-LLM Panel Evaluation")
    print("=" * 80)
    print("\nJudges:")
    for spec in judges:
        print(f"  - {spec.judge_id} (weight: {spec.weight})")
    print("\nAggregation Strategy: majority")
    print(f"\nPrompt: {dataset.prompt}\n")
    print(f"Rubric Criteria (total positive weight: {dataset.total_positive_weight}):")
    for i, (criterion, name) in enumerate(
        zip(dataset.rubric.rubric, dataset.criterion_names)
    ):
        weight_str = (
            f"+{criterion.weight}" if criterion.weight > 0 else str(criterion.weight)
        )
        print(f"  {i + 1}. {name:10} [{weight_str:>6}]")
    print("\n" + "-" * 80)

    # Grade all items using EvalRunner
    print(f"\nGrading {len(dataset)} items with ensemble of {len(judges)} judges...")
    llm_calls = len(judges) * dataset.num_criteria
    print(
        f"(Each item requires {len(judges)} x {dataset.num_criteria} = {llm_calls} LLM calls)\n"
    )

    eval_result = await evaluate(
        dataset=dataset,
        grader=grader,
        show_progress=True,
        progress_style="detailed",  # Shows per-judge progress
    )

    if eval_result.experiment_dir:
        print(f"\nExperiment saved to: {eval_result.experiment_dir}")

    # Compute metrics with per-judge breakdown
    metrics = eval_result.compute_metrics(
        dataset,
        bootstrap=True,
        per_judge=True,  # Include per-judge metrics
    )
    metrics.to_file(Path(eval_result.experiment_dir) / "metrics.json")

    # Print the formatted summary (includes per-judge section)
    print("\n" + metrics.summary())

    # Show individual results
    print("\n" + "=" * 80)
    print("INDIVIDUAL RESULTS")
    print("=" * 80)

    for item_result in eval_result.item_results:
        item = item_result.item
        report = item_result.report

        true_score = dataset.compute_weighted_score(item.ground_truth)
        score_error = abs(report.score - true_score)

        print(f"\nItem {item_result.item_idx + 1}: {item.description}")
        print(
            f"  Scores: Ensemble={report.score:.3f} | Actual={true_score:.3f} "
            f"| Error={score_error:.3f}"
        )
        print(f"  Agreement: {report.mean_agreement:.1%}")

        # Show per-judge scores
        print("  Judge Scores: ", end="")
        for judge_id, score in sorted(report.judge_scores.items()):
            print(f"{judge_id}={score:.3f}  ", end="")
        print()

    # Show per-judge comparison table
    if metrics.per_judge:
        print("\n" + "=" * 80)
        print("PER-JUDGE COMPARISON")
        print("=" * 80)
        print(
            f"\n{'Judge':<15} {'Accuracy':>10} {'Kappa':>10} {'RMSE':>10} "
            f"{'Spearman':>10} {'Bias':>10}"
        )
        print("-" * 70)

        for judge_id, jm in sorted(metrics.per_judge.items()):
            print(
                f"{judge_id:<15} {jm.criterion_accuracy:>10.1%} {jm.mean_kappa:>10.3f} "
                f"{jm.score_rmse:>10.4f} {jm.score_spearman.coefficient:>10.4f} "
                f"{jm.bias.mean_bias:>+10.4f}"
            )

        print("-" * 70)
        print(
            f"{'ENSEMBLE':<15} {metrics.criterion_accuracy:>10.1%} {metrics.mean_kappa:>10.3f} "
            f"{metrics.score_rmse:>10.4f} {metrics.score_spearman.coefficient:>10.4f} "
            f"{metrics.bias.mean_bias:>+10.4f}"
        )

    # Cost & Timing Summary
    print("\n" + "-" * 80)
    print("COST & TIMING ANALYSIS")
    print("-" * 80)

    if eval_result.total_token_usage:
        print("\nTotal Token Usage:")
        print(
            f"  Prompt tokens:     {eval_result.total_token_usage.prompt_tokens:>12,}"
        )
        print(
            f"  Completion tokens: {eval_result.total_token_usage.completion_tokens:>12,}"
        )
        print(f"  Total tokens:      {eval_result.total_token_usage.total_tokens:>12,}")

    if eval_result.total_completion_cost and eval_result.total_completion_cost > 0:
        print(f"\nTotal Cost: ${eval_result.total_completion_cost:.6f}")
        cost_per_item = eval_result.total_completion_cost / len(dataset)
        print(f"Cost per Item: ${cost_per_item:.6f}")
        print(f"Cost per Item per Judge: ${cost_per_item / len(judges):.6f}")

    ts = eval_result.timing_stats
    print("\nTiming Stats:")
    print(f"  Total duration:    {ts.total_duration_seconds:.2f}s")
    print(f"  Throughput:        {ts.items_per_second:.2f} items/s")
    print(f"  Mean item time:    {ts.mean_item_duration_seconds:.2f}s")
    print(f"  P95 item time:     {ts.p95_item_duration_seconds:.2f}s")

    print("\n" + "=" * 80)
    print(
        f"Note: This ensemble grading uses {len(judges)}x the LLM calls of single-model grading,"
    )
    print(
        "but can improve reliability through consensus and highlight ambiguous cases."
    )
    print("=" * 80)


def check_api_keys() -> list[str]:
    """Check which API keys are available."""
    missing = []
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    return missing


if __name__ == "__main__":
    missing_keys = check_api_keys()
    if missing_keys:
        print("Warning: The following API keys are not set:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nSet them to run the ensemble demo:")
        print("  export GEMINI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
        print("\nAlternatively, modify the judges list to use available providers.")
        sys.exit(1)

    asyncio.run(main())
