#!/usr/bin/env python3
"""Grade ResearcherBench outputs using AutoRubric with a single LLM judge.

Loads a RubricDataset from researcher_bench/output/ and grades each item's
response against its per-item rubric, producing coverage scores.

Usage:
    # Grade test dataset (2 items, quick validation)
    python examples/researcher_bench_demo.py

    # Grade a specific model's responses
    python examples/researcher_bench_demo.py researcher_bench/output/Claude_rubric_dataset.json

    # Grade a random subset
    python examples/researcher_bench_demo.py researcher_bench/output/OpenAI_rubric_dataset.json --max-items 10

    # Adjust parallelism
    python examples/researcher_bench_demo.py researcher_bench/output/Claude_rubric_dataset.json --max-parallel-requests 5
"""

import argparse
import asyncio
import hashlib
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    LLMConfig,
    RubricDataset,
    evaluate,
)
from autorubric.graders import CriterionGrader
from autorubric.types import CriterionVerdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = (
    PROJECT_ROOT / "researcher_bench" / "output" / "test_rubric_dataset.json"
)
# MODEL = "anthropic/claude-sonnet-4-5-20250929"
MODEL = "gemini/gemini-3-flash-preview"
DEFAULT_MAX_PARALLEL_REQUESTS = 3
CATEGORY_PATTERN = re.compile(r"\[(.+?)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade ResearcherBench outputs with AutoRubric."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a *_rubric_dataset.json file (default: test dataset).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Grade a random subset of N items (seed=42).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from a previous checkpoint.",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=DEFAULT_MAX_PARALLEL_REQUESTS,
        help=f"Max concurrent LLM requests (default: {DEFAULT_MAX_PARALLEL_REQUESTS}).",
    )
    return parser.parse_args()


def extract_category(description: str) -> str:
    """Extract the category from a description like 'Q1 [Open Consulting] Synthetic Data'."""
    match = CATEGORY_PATTERN.search(description)
    return match.group(1) if match else "Unknown"


def subsample_dataset(dataset: RubricDataset, max_items: int) -> RubricDataset:
    """Return a new dataset with a random subset of items (seed=42)."""
    if max_items >= len(dataset.items):
        return dataset
    rng = random.Random(42)
    sampled = rng.sample(dataset.items, max_items)
    return RubricDataset(
        name=dataset.name,
        prompt=dataset.prompt,
        rubric=dataset.rubric,
        items=sampled,
    )


async def main() -> None:
    args = parse_args()

    dataset = RubricDataset.from_file(args.dataset)
    if args.max_items:
        dataset = subsample_dataset(dataset, args.max_items)

    n_items = len(dataset.items)
    dataset_name = dataset.name or args.dataset.stem

    print(f"\n=== ResearcherBench Grading: {dataset_name} ===")
    print(f"Judge model: {MODEL}")
    print(f"Items: {n_items}")

    llm_config = LLMConfig(
        model=MODEL,
        temperature=0.0,
        max_parallel_requests=args.max_parallel_requests,
    )

    grader = CriterionGrader(
        llm_config=llm_config,
        normalize=True,
        cannot_assess_config=CannotAssessConfig(strategy=CannotAssessStrategy.SKIP),
    )

    config_str = f"{MODEL}:no_thinking"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    experiment_name = f"researcher_bench_{dataset_name}_{config_hash}"

    eval_result = await evaluate(
        dataset=dataset,
        grader=grader,
        show_progress=True,
        progress_style="simple",
        experiment_name=experiment_name,
        resume=not args.no_resume,
    )

    # -- Per-item results --
    print(
        f"\n--- Per-Item Results ({eval_result.successful_items}/{n_items} succeeded) ---"
    )

    scores: list[float] = []
    category_scores: dict[str, list[float]] = defaultdict(list)

    for ir in eval_result.item_results:
        desc = ir.item.description or f"Item {ir.item_idx}"
        if ir.error is not None:
            print(f"  {desc:<55s}  ERROR: {ir.error}")
            continue

        score = ir.report.score
        scores.append(score)

        category = extract_category(desc)
        category_scores[category].append(score)

        # Count MET criteria
        met = 0
        total = 0
        if ir.report.report is not None:
            for cr in ir.report.report:
                verdict = (
                    cr.final_verdict if hasattr(cr, "final_verdict") else cr.verdict
                )
                if verdict == CriterionVerdict.CANNOT_ASSESS:
                    continue
                total += 1
                if verdict == CriterionVerdict.MET:
                    met += 1

        print(f"  {desc:<55s}  Score: {score:.2f}  ({met}/{total} criteria met)")

    if not scores:
        print("\nNo successful results to summarize.")
        return

    # -- Summary statistics --
    min_score = min(scores)
    max_score = max(scores)
    min_idx = scores.index(min_score)
    max_idx = scores.index(max_score)
    successful = eval_result.filter_successful()
    min_desc = successful[min_idx].item.description or f"Item {min_idx}"
    max_desc = successful[max_idx].item.description or f"Item {max_idx}"

    print("\n--- Summary ---")
    print(f"  Mean Coverage Score:  {statistics.mean(scores):.3f}")
    print(f"  Median:              {statistics.median(scores):.3f}")
    if len(scores) > 1:
        print(f"  Std Dev:             {statistics.stdev(scores):.3f}")
    print(f"  Min:                 {min_score:.3f}  ({min_desc})")
    print(f"  Max:                 {max_score:.3f}  ({max_desc})")

    # -- Per-category breakdown --
    if category_scores:
        print("\n--- By Category ---")
        for cat in sorted(category_scores):
            cat_vals = category_scores[cat]
            mean = statistics.mean(cat_vals)
            std = statistics.stdev(cat_vals) if len(cat_vals) > 1 else 0.0
            print(f"  {cat} ({len(cat_vals)}):".ljust(40) + f"{mean:.3f} ± {std:.3f}")

    # -- Cost & timing --
    print("\n--- Cost & Timing ---")
    ts = eval_result.timing_stats
    print(f"  Total time:          {ts.total_duration_seconds:.1f}s")
    print(f"  Mean time per item:  {ts.mean_item_duration_seconds:.1f}s")

    if eval_result.total_completion_cost is not None:
        total_cost = eval_result.total_completion_cost
        print(f"  Total cost:          ${total_cost:.4f}")
        print(f"  Cost per item:       ${total_cost / len(scores):.4f}")

    if eval_result.total_token_usage is not None:
        tu = eval_result.total_token_usage
        print(f"  Total tokens:        {tu.total_tokens:,}")

    if eval_result.failed_items > 0:
        print(f"\n  Errors: {eval_result.failed_items}")
        for idx, err in eval_result.errors:
            print(f"    Item {idx}: {err}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
