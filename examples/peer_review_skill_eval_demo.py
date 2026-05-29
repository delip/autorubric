#!/usr/bin/env python3
"""
Demonstration of evaluating agent skill efficacy using AutoRubric.

Dataset: 30 peer reviews of research papers generated under 3 conditions:
  - Without skill: No guidance document provided to the agent
  - Poor skill (SKILL_v1.md): Early-draft skill with gaps
  - Good skill (SKILL.md): Polished skill with comprehensive guidance

Items are grouped by paper: items 0-2 = Paper 1 (without/poor/good),
items 3-5 = Paper 2, etc. — 10 papers total, 3 reviews each.

Rubric: 10 criteria covering outcome quality, review style, and efficiency,
organized into 3 evaluation dimensions:
  - Outcome (55%): paper_summary, methodology_assessment, statistical_evaluation,
                    strengths_and_weaknesses
  - Style (25%): constructive_tone, structured_format, specific_references
  - Efficiency (20%): concise_review, clear_recommendation
  - Penalty: factual_misrepresentation (negative weight, tracked separately)

This demo runs 5 phases:
  1. Evaluate all 30 items with a single LLM grader
  2. Dimension analysis — average accuracy per dimension
  3. Three-condition comparison — mean scores and per-criterion pass rates
  4. Failure mode analysis — top failure criteria per condition
  5. (Optional) Rubric improvement via --improve flag
"""

import argparse
import asyncio
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    CriterionVerdict,
    LLMConfig,
    evaluate,
)
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader

load_dotenv()

DATASET_PATH = Path(__file__).parent / "data" / "peer_review_skill_eval.json"

# Dimension definitions: dimension name -> (weight label, list of criterion names)
DIMENSIONS = {
    "Outcome": (
        "55%",
        [
            "paper_summary",
            "methodology_assessment",
            "statistical_evaluation",
            "strengths_and_weaknesses",
        ],
    ),
    "Style": (
        "25%",
        ["constructive_tone", "structured_format", "specific_references"],
    ),
    "Efficiency": (
        "20%",
        ["concise_review", "clear_recommendation"],
    ),
}

PENALTY_CRITERION = "factual_misrepresentation"

CONDITION_TAGS = {
    "without-skill": "[without-skill]",
    "poor-skill": "[poor-skill]",
    "good-skill": "[good-skill]",
}


def parse_condition(description: str) -> str:
    """Extract condition tag from item description."""
    for key, tag in CONDITION_TAGS.items():
        if tag in description:
            return key
    return "unknown"


def compute_pass_rates(items, dataset):
    """Compute per-criterion pass rates from ground truth verdicts."""
    criterion_names = dataset.criterion_names
    pass_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for item in items:
        if item.ground_truth is None:
            continue
        for j, verdict in enumerate(item.ground_truth):
            name = criterion_names[j]
            total_counts[name] += 1
            if isinstance(verdict, CriterionVerdict) and verdict == CriterionVerdict.MET:
                pass_counts[name] += 1

    return {
        name: pass_counts[name] / total_counts[name] if total_counts[name] > 0 else 0.0
        for name in criterion_names
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate agent skill efficacy with AutoRubric")
    parser.add_argument(
        "--improve",
        action="store_true",
        help="Run rubric improvement (Phase 5) after analysis",
    )
    args = parser.parse_args()

    # Load the dataset
    dataset = RubricDataset.from_file(DATASET_PATH)

    # Configure LLM with thinking enabled and rate limiting
    llm_config = LLMConfig(
        model="gemini/gemini-3-flash-preview",
        temperature=1.0,
        thinking="medium",
        max_parallel_requests=10,
    )

    # Create grader with CANNOT_ASSESS handling
    grader = CriterionGrader(
        llm_config=llm_config,
        normalize=True,
        cannot_assess_config=CannotAssessConfig(
            strategy=CannotAssessStrategy.SKIP,
        ),
    )

    print("=" * 80)
    print("AutoRubric Demo: Peer Review Skill Efficacy Evaluation")
    print(f"Model: {llm_config.model}")
    print(f"Thinking: {llm_config.thinking}")
    print(f"Max Parallel Requests: {llm_config.max_parallel_requests}")
    print("=" * 80)

    print(f"\nDataset: {len(dataset)} items (10 papers x 3 conditions)")
    print(
        f"Rubric: {dataset.num_criteria} criteria "
        f"(total positive weight: {dataset.total_positive_weight})"
    )
    for i, (criterion, name) in enumerate(zip(dataset.rubric.rubric, dataset.criterion_names)):
        weight_str = f"+{criterion.weight}" if criterion.weight > 0 else str(criterion.weight)
        print(f"  {i + 1}. {name:30} [{weight_str:>6}]")

    # =========================================================================
    # Phase 1: Evaluate All Items
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Evaluate All Items")
    print("=" * 80)
    print(f"\nGrading {len(dataset)} items...")

    eval_result = await evaluate(
        dataset=dataset,
        grader=grader,
        show_progress=True,
        progress_style="simple",
        experiment_name=None,
        resume=True,
    )

    if eval_result.experiment_dir:
        print(f"\nExperiment saved to: {eval_result.experiment_dir}")

    metrics = eval_result.compute_metrics(dataset, bootstrap=True)
    print("\n" + metrics.summary())

    # =========================================================================
    # Phase 2: Dimension Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Dimension Analysis")
    print("=" * 80)

    per_criterion_map = {cm.name: cm for cm in metrics.per_criterion}

    print(f"\n{'Dimension':<15} {'Weight':<10} {'Criteria':>10} {'Avg Accuracy':>15}")
    print("-" * 55)

    for dim_name, (weight_label, criteria_names) in DIMENSIONS.items():
        accuracies = []
        for cname in criteria_names:
            if cname in per_criterion_map:
                cm = per_criterion_map[cname]
                if cm.accuracy is not None:
                    accuracies.append(cm.accuracy)
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else None
        avg_acc_str = f"{avg_acc:>14.1%}" if avg_acc is not None else f"{'N/A':>14}"
        print(f"{dim_name:<15} {weight_label:<10} {len(criteria_names):>10} {avg_acc_str}")

    # Penalty criterion
    if PENALTY_CRITERION in per_criterion_map:
        cm = per_criterion_map[PENALTY_CRITERION]
        pen_acc_str = f"{cm.accuracy:>14.1%}" if cm.accuracy is not None else f"{'N/A':>14}"
        print(f"{'Penalty':<15} {'---':<10} {'1':>10} {pen_acc_str}")

    # =========================================================================
    # Phase 3: Three-Condition Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Three-Condition Comparison")
    print("=" * 80)

    # Partition items by condition
    condition_items = defaultdict(list)
    condition_results = defaultdict(list)

    for item_result in eval_result.item_results:
        condition = parse_condition(item_result.item.description)
        condition_items[condition].append(item_result.item)
        condition_results[condition].append(item_result)

    # Mean scores per condition
    print(f"\n{'Condition':<20} {'N':>5} {'Mean Score':>12} {'Delta vs None':>15}")
    print("-" * 55)

    condition_means = {}
    for cond in ["without-skill", "poor-skill", "good-skill"]:
        results = condition_results[cond]
        scores = [r.report.score for r in results if r.report.score is not None]
        mean_score = sum(scores) / len(scores) if scores else None
        condition_means[cond] = mean_score

    baseline = condition_means.get("without-skill")
    for cond in ["without-skill", "poor-skill", "good-skill"]:
        mean_score = condition_means[cond]
        mean_str = f"{mean_score:>12.3f}" if mean_score is not None else f"{'N/A':>12}"
        if cond == "without-skill":
            delta_str = "---"
        elif mean_score is not None and baseline is not None:
            delta_str = f"{mean_score - baseline:+.3f}"
        else:
            delta_str = "N/A"
        print(f"{cond:<20} {len(condition_results[cond]):>5} {mean_str} {delta_str:>15}")

    # Per-criterion pass rates by condition (from ground truth)
    print("\nPer-Criterion Ground Truth Pass Rates by Condition:")
    criterion_names = dataset.criterion_names
    print(f"\n{'Criterion':<30} {'Without':>10} {'Poor':>10} {'Good':>10}")
    print("-" * 65)

    pass_rates = {}
    for cond in ["without-skill", "poor-skill", "good-skill"]:
        pass_rates[cond] = compute_pass_rates(condition_items[cond], dataset)

    for cname in criterion_names:
        wo = pass_rates["without-skill"].get(cname, 0.0)
        po = pass_rates["poor-skill"].get(cname, 0.0)
        go = pass_rates["good-skill"].get(cname, 0.0)
        print(f"{cname:<30} {wo:>10.0%} {po:>10.0%} {go:>10.0%}")

    # Paper triplet comparison (papers 1-3)
    print("\nDirect Comparison — Selected Paper Triplets:")
    print(f"{'Paper':<8} {'Condition':<20} {'Predicted':>10} {'Expected':>10}")
    print("-" * 52)

    for paper_idx in range(3):
        for offset, cond in enumerate(["without-skill", "poor-skill", "good-skill"]):
            item_idx = paper_idx * 3 + offset
            if item_idx < len(eval_result.item_results):
                ir = eval_result.item_results[item_idx]
                item = ir.item
                pred = ir.report.score
                pred_str = f"{pred:>10.3f}" if pred is not None else f"{'N/A':>10}"
                expected = dataset.compute_weighted_score(item.ground_truth)
                print(f"P{paper_idx + 1:<7} {cond:<20} {pred_str} {expected:>10.3f}")
        if paper_idx < 2:
            print()

    # =========================================================================
    # Phase 4: Failure Mode Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: Failure Mode Analysis")
    print("=" * 80)

    for cond in ["without-skill", "poor-skill", "good-skill"]:
        items = condition_items[cond]
        fail_counts = defaultdict(int)
        total_counts = defaultdict(int)

        for item in items:
            if item.ground_truth is None:
                continue
            for j, verdict in enumerate(item.ground_truth):
                name = criterion_names[j]
                total_counts[name] += 1
                if isinstance(verdict, CriterionVerdict) and verdict == CriterionVerdict.UNMET:
                    fail_counts[name] += 1

        # Sort by failure rate descending
        failure_rates = [
            (name, fail_counts[name] / total_counts[name] if total_counts[name] > 0 else 0.0)
            for name in criterion_names
        ]
        failure_rates.sort(key=lambda x: x[1], reverse=True)

        # Show top failures (those with > 0% failure rate)
        top_failures = [(n, r) for n, r in failure_rates if r > 0]

        print(f"\n[{cond}] Top Failure Criteria:")
        if top_failures:
            for name, rate in top_failures[:5]:
                bar = "#" * int(rate * 20)
                print(f"  {name:<30} {rate:>6.0%}  {bar}")
        else:
            print("  (no failures)")

    # =========================================================================
    # Phase 5 (Optional): Rubric Improvement
    # =========================================================================
    if args.improve:
        print("\n" + "=" * 80)
        print("PHASE 5: Rubric Improvement")
        print("=" * 80)

        from autorubric.meta import improve_rubric

        revision_llm = LLMConfig(
            model="gemini/gemini-3-flash-preview",
            temperature=1.0,
            thinking="medium",
            max_parallel_requests=5,
        )

        print("\nRunning iterative rubric improvement...")
        print(f"  Eval LLM: {llm_config.model}")
        print(f"  Revision LLM: {revision_llm.model}")

        result = await improve_rubric(
            dataset.rubric,
            eval_llm=llm_config,
            revision_llm=revision_llm,
            max_iterations=5,
            mode="standalone",
            display="stdout",
            save_artifacts=True,
        )

        print("\nImprovement complete:")
        print(f"  Iterations:       {len(result.iterations)}")
        print(f"  Best iteration:   {result.best_iteration}")
        print(f"  Convergence:      {result.convergence_reason}")
        print(f"  Total cost:       ${result.total_completion_cost:.6f}")

        print("\nOriginal rubric criteria:")
        for c in result.original_rubric.rubric:
            print(f"  - {c.name}: {c.requirement[:60]}...")

        print("\nBest rubric criteria:")
        for c in result.best_rubric.rubric:
            print(f"  - {c.name}: {c.requirement[:60]}...")

    # =========================================================================
    # Cost & Timing Summary
    # =========================================================================
    print("\n" + "-" * 80)
    print("COST & TIMING SUMMARY")
    print("-" * 80)

    timing = eval_result.timing_stats
    print(f"\nThroughput: {timing.items_per_second:.2f} items/second")
    print(f"Total Duration: {timing.total_duration_seconds:.2f}s")
    print(f"Mean Item Duration: {timing.mean_item_duration_seconds:.2f}s")
    print(f"P50 Item Duration: {timing.p50_item_duration_seconds:.2f}s")
    print(f"P95 Item Duration: {timing.p95_item_duration_seconds:.2f}s")

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
