#!/usr/bin/env python3
"""Demo: two-track rubric improvement using ricechem ground-truth data.

Workflow:
1. Load the ricechem q1 dataset (327 student responses on ionization energy)
2. Split 30/70 into validation (98 items) and held-out test (229 items)
3. Track A — Baseline: generate a rubric via a single LLM call, measure test rho,
   then improve it iteratively and measure again
4. Track B — Original: take the human-authored rubric from the dataset, measure
   test rho, then improve it iteratively and measure again
5. Print four-way comparison: baseline vs improved-baseline vs original vs
   improved-original

Usage:
    python rubric_improvement_demo.py
"""

import asyncio

from pydantic import BaseModel, Field

from autorubric import LLMConfig, Rubric
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader
from autorubric.llm import LLMClient
from autorubric.meta import ImprovementConfig, ImprovementRunner, IterationResult
from autorubric.meta._improve import compute_expected_scores, validate_ground_truth

MAX_ITERATIONS = 10
N_VAL = 98  # ~30% of 327 items


class CriterionSpec(BaseModel):
    weight: int = Field(
        description="Importance weight (5-15 for positive, negative for errors)"
    )
    requirement: str = Field(description="Clear, specific evaluation criterion")
    name: str | None = Field(default=None, description="Short identifier")


class GeneratedRubric(BaseModel):
    criteria: list[CriterionSpec] = Field(description="List of evaluation criteria")


async def generate_baseline_rubric(
    task_prompt: str, llm_config: LLMConfig
) -> tuple[Rubric, float]:
    """Generate a baseline rubric via a single LLM call (no iterative improvement)."""
    client = LLMClient(llm_config)
    gen = await client.generate(
        system_prompt="You are an expert rubric designer for educational assessment.",
        user_prompt=(
            "Write a detailed rubric to grade the following chemistry exam question:\n\n"
            f"Question: {task_prompt}"
        ),
        response_format=GeneratedRubric,
        return_result=True,
    )
    rubric = Rubric.from_dict(
        [c.model_dump(exclude_none=True) for c in gen.parsed.criteria]
    )
    return rubric, gen.cost or 0.0


def quality_plateau(
    current: IterationResult, history: list[IterationResult]
) -> str | None:
    """Stop when no issues remain or quality score plateaus for 4 iterations."""
    if not current.issues:
        return "no_issues"
    if len(history) >= MAX_ITERATIONS:
        return "max_iterations"
    if len(history) < 4:
        return None
    prev = max(r.quality_score for r in history[-4:-1])
    if current.quality_score - prev < 0.02:
        return "quality_plateau"
    return None


async def measure_test_correlation(
    rubric: Rubric,
    test: RubricDataset,
    expected_scores: list[float],
    grader: CriterionGrader,
    task_prompt: str,
    label: str = "Grading test set",
) -> tuple[float, float]:
    """Grade the held-out test set and return (Spearman rho, cost)."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{label}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    )
    with progress:
        task = progress.add_task("", total=len(test.items))
        correlation, _per_item, cost = await validate_ground_truth(
            rubric,
            test,
            expected_scores,
            grader,
            task_prompt=task_prompt,
            on_item_complete=lambda: progress.advance(task),
        )
    return correlation, cost or 0.0


def _print_rubric(title: str, rubric: Rubric) -> None:
    """Print a rubric's criteria with weights."""
    print(f"\n--- {title} ---")
    for i, c in enumerate(rubric.rubric, 1):
        print(f"  {i}. [w={c.weight}] {c.requirement}")


async def main() -> None:
    """Run the two-track rubric improvement demo with ricechem data."""
    print("Rubric Improvement Demo (ricechem q1 -- ionization energy)")
    print("=" * 64)

    # -- LLM configs --
    eval_llm = LLMConfig(
        model="gemini/gemini-3-flash-preview",
        temperature=1.0,
        max_parallel_requests=10,
    )
    revision_llm = LLMConfig(
        model="gemini/gemini-3-pro-preview",
        temperature=1.0,
        thinking="medium",
        timeout=300,
    )

    # -- Load dataset and prepare splits --
    dataset = RubricDataset.from_file("examples/data/ricechem/q1.json")
    task_prompt = dataset.prompt
    original_rubric = dataset.rubric

    val, test = dataset.split_train_test(n_train=N_VAL, seed=42)
    test_expected = compute_expected_scores(test)

    grader = CriterionGrader(llm_config=eval_llm)

    # ================================================================
    # Track A — Baseline (zero-shot LLM-generated rubric)
    # ================================================================
    print("\n" + "-" * 64)
    print("TRACK A: Baseline (zero-shot LLM-generated rubric)")
    print("-" * 64)

    print("\nGenerating baseline rubric via single LLM call ...")
    baseline_rubric, baseline_gen_cost = await generate_baseline_rubric(
        task_prompt, revision_llm
    )
    baseline_eval_cost = 0.0

    baseline_rho, cost = await measure_test_correlation(
        baseline_rubric,
        test,
        test_expected,
        grader,
        task_prompt,
        label="[Baseline] Grading test set",
    )
    baseline_eval_cost += cost
    print(f"[Baseline] Spearman rho on test set: {baseline_rho:.3f}")

    print(f"\nRunning improvement loop (max {MAX_ITERATIONS} iterations) ...")
    baseline_config = ImprovementConfig(
        eval_llm=eval_llm,
        revision_llm=revision_llm,
        validation_data=val,
        artifacts_dir="experiments/ricechem_improvement/baseline",
        display="stdout",
        convergence_fn=quality_plateau,
        max_iterations=MAX_ITERATIONS,
    )
    baseline_runner = ImprovementRunner(
        baseline_rubric, task_prompt, config=baseline_config
    )
    baseline_result = await baseline_runner.run()
    improved_baseline = baseline_result.final_rubric

    improved_baseline_rho, cost = await measure_test_correlation(
        improved_baseline,
        test,
        test_expected,
        grader,
        task_prompt,
        label="[Baseline improved] Grading test set",
    )
    baseline_eval_cost += cost
    print(f"[Baseline improved] Spearman rho on test set: {improved_baseline_rho:.3f}")

    # ================================================================
    # Track B — Original (human-authored rubric from dataset)
    # ================================================================
    print("\n" + "-" * 64)
    print("TRACK B: Original (human-authored rubric)")
    print("-" * 64)

    original_eval_cost = 0.0

    original_rho, cost = await measure_test_correlation(
        original_rubric,
        test,
        test_expected,
        grader,
        task_prompt,
        label="[Original] Grading test set",
    )
    original_eval_cost += cost
    print(f"[Original] Spearman rho on test set: {original_rho:.3f}")

    print(f"\nRunning improvement loop (max {MAX_ITERATIONS} iterations) ...")
    original_config = ImprovementConfig(
        eval_llm=eval_llm,
        revision_llm=revision_llm,
        validation_data=val,
        artifacts_dir="experiments/ricechem_improvement/original",
        display="stdout",
        convergence_fn=quality_plateau,
        max_iterations=MAX_ITERATIONS,
    )
    original_runner = ImprovementRunner(
        original_rubric, task_prompt, config=original_config
    )
    original_result = await original_runner.run()
    improved_original = original_result.final_rubric

    improved_original_rho, cost = await measure_test_correlation(
        improved_original,
        test,
        test_expected,
        grader,
        task_prompt,
        label="[Original improved] Grading test set",
    )
    original_eval_cost += cost
    print(f"[Original improved] Spearman rho on test set: {improved_original_rho:.3f}")

    # ================================================================
    # Summary
    # ================================================================
    baseline_improvement_cost = baseline_result.total_completion_cost or 0.0
    original_improvement_cost = original_result.total_completion_cost or 0.0
    baseline_track_cost = baseline_gen_cost + baseline_eval_cost + baseline_improvement_cost
    original_track_cost = original_eval_cost + original_improvement_cost
    grand_total = baseline_track_cost + original_track_cost

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print("  Spearman rho (test set):")
    print(f"    Baseline (zero-shot LLM):      {baseline_rho:.3f}")
    print(f"    Baseline improved (iterative):  {improved_baseline_rho:.3f}")
    print(f"    Original (human-authored):      {original_rho:.3f}")
    print(f"    Original improved (iterative):  {improved_original_rho:.3f}")
    baseline_delta = improved_baseline_rho - baseline_rho
    original_delta = improved_original_rho - original_rho
    print(f"  Delta (baseline -> improved baseline): {baseline_delta:+.3f}")
    print(f"  Delta (original -> improved original): {original_delta:+.3f}")

    print(f"\n  Baseline track:")
    print(f"    Improvement iterations:  {len(baseline_result.iterations)}")
    print(f"    Convergence reason:      {baseline_result.convergence_reason}")
    print(f"    Improvement loop cost:   ${baseline_improvement_cost:.4f}")
    print(f"    Generation + eval cost:  ${baseline_gen_cost + baseline_eval_cost:.4f}")
    print(f"    Track total cost:        ${baseline_track_cost:.4f}")

    print(f"\n  Original track:")
    print(f"    Improvement iterations:  {len(original_result.iterations)}")
    print(f"    Convergence reason:      {original_result.convergence_reason}")
    print(f"    Improvement loop cost:   ${original_improvement_cost:.4f}")
    print(f"    Eval cost:               ${original_eval_cost:.4f}")
    print(f"    Track total cost:        ${original_track_cost:.4f}")

    print(f"\n  Grand total cost:          ${grand_total:.4f}")

    _print_rubric("Baseline rubric (zero-shot LLM)", baseline_rubric)
    _print_rubric("Improved baseline rubric (iterative)", improved_baseline)
    _print_rubric("Original rubric (human-authored)", original_rubric)
    _print_rubric("Improved original rubric (iterative)", improved_original)


if __name__ == "__main__":
    asyncio.run(main())
