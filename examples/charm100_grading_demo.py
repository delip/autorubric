#!/usr/bin/env python3
"""
Demonstration of multi-choice criterion evaluation with the autorubric library.

Dataset: 100 chatbot responses evaluated on a hybrid rubric with:
    - 4 ordinal criteria (satisfaction, helpfulness, naturalness, specificity)
    - 1 nominal criterion (response_length)
    - 1 binary criterion (factual_accuracy)

This demo showcases the new multi-choice metrics:
    - Weighted kappa for ordinal scales
    - Adjacent accuracy (within ±1) for ordinal
    - Spearman/Kendall correlations for ordinal
    - Per-option precision/recall/F1
    - Confusion matrices
    - NA option handling
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
from autorubric.metrics import (
    NominalCriterionMetrics,
    OrdinalCriterionMetrics,
)
from autorubric.metrics._helpers import classify_criterion

load_dotenv()

# Path to the multi-choice dataset
DATASET_PATH = Path(__file__).parent / "data" / "charm100.json"


def fmt(value: float | None, spec: str, na: str = "N/A") -> str:
    """Format an optional metric value, rendering None as N/A."""
    return format(value, spec) if value is not None else na


def format_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> str:
    """Format a confusion matrix for display."""
    lines = []

    # Truncate labels for display
    max_label_len = 12
    short_labels = [lbl[:max_label_len] for lbl in labels]

    # Header
    col_width = max(len(lbl) for lbl in short_labels) + 2
    header = " " * (col_width + 2) + "Predicted".center(col_width * len(labels))
    lines.append(header)

    label_header = " " * (col_width + 2) + "".join(lbl.center(col_width) for lbl in short_labels)
    lines.append(label_header)
    lines.append(" " * (col_width) + "-" * (col_width * len(labels) + 2))

    # Rows
    for i, row_label in enumerate(short_labels):
        prefix = "Actual " if i == len(labels) // 2 else "       "
        row_str = f"{prefix}{row_label:>{col_width - 2}} |"
        for val in matrix[i]:
            row_str += f"{val:^{col_width}}"
        lines.append(row_str)

    return "\n".join(lines)


async def main():
    # Load the dataset from JSON file
    dataset = RubricDataset.from_file(DATASET_PATH)

    # Classify criteria by type
    criterion_types = [classify_criterion(c) for c in dataset.rubric.rubric]
    n_ordinal = sum(1 for t in criterion_types if t == "ordinal")
    n_nominal = sum(1 for t in criterion_types if t == "nominal")
    n_binary = sum(1 for t in criterion_types if t == "binary")

    # Configure LLM with thinking enabled
    llm_config = LLMConfig(
        model="gemini/gemini-3-flash-preview",
        temperature=1.0,
        cache_enabled=False,
        max_parallel_requests=10,
    )

    # Create grader
    grader = CriterionGrader(
        llm_config=llm_config,
        normalize=True,
        cannot_assess_config=CannotAssessConfig(
            strategy=CannotAssessStrategy.SKIP,
        ),
    )

    print("=" * 80)
    print("AutoRubric Demo: Multi-Choice Criterion Evaluation")
    print(f"Model: {llm_config.model}")
    print(f"Thinking: {llm_config.thinking}")
    print("=" * 80)
    print(f"\nPrompt: {dataset.prompt}\n")
    print(
        f"Rubric: {dataset.num_criteria} criteria ({n_ordinal} ordinal, {n_nominal} nominal, {n_binary} binary)"
    )
    print(f"Total positive weight: {dataset.total_positive_weight}")
    print()

    # Display criteria with their types
    print("Criteria:")
    for i, criterion in enumerate(dataset.rubric.rubric):
        c_type = classify_criterion(criterion)
        weight_str = f"+{criterion.weight}" if criterion.weight > 0 else str(criterion.weight)

        if criterion.is_binary:
            print(f"  {i + 1}. {criterion.name:20} [{weight_str:>6}] (binary)")
        else:
            scale_info = f"{c_type}, {len(criterion.options)} options"
            na_opts = [opt for opt in criterion.options if opt.na]
            if na_opts:
                scale_info += ", 1 NA"
            print(f"  {i + 1}. {criterion.name:20} [{weight_str:>6}] ({scale_info})")

            # Show options for multi-choice
            for j, opt in enumerate(criterion.options):
                na_marker = " [NA]" if opt.na else ""
                print(f"        {j}: {opt.label} (value={opt.value:.2f}){na_marker}")

    print("\n" + "-" * 80)

    # Run evaluation
    print(f"\nGrading {len(dataset)} items...")
    print("(Checkpoints saved to experiments/ directory)\n")

    eval_result = await evaluate(
        dataset=dataset,
        grader=grader,
        show_progress=True,
        progress_style="simple",
        experiment_name=None,  # Auto-generate name
        resume=True,
    )

    if eval_result.experiment_dir:
        print(f"\nExperiment saved to: {eval_result.experiment_dir}")

    # Compute metrics with the multi-choice support
    metrics = eval_result.compute_metrics(
        dataset,
        bootstrap=True,
        na_mode="exclude",  # Exclude NA from metrics
    )

    # Save metrics
    if eval_result.experiment_dir:
        metrics.to_file(Path(eval_result.experiment_dir) / "metrics.json")

    # Print the summary
    print("\n" + metrics.summary())

    # Detailed per-criterion breakdown
    print("\n" + "=" * 80)
    print("DETAILED PER-CRITERION METRICS")
    print("=" * 80)

    for cm in metrics.per_criterion:
        print(f"\n{'─' * 60}")
        print(f"Criterion: {cm.name} ({cm.criterion_type.upper()})")
        print(f"{'─' * 60}")

        if isinstance(cm, OrdinalCriterionMetrics):
            print(f"  Samples: {cm.n_samples}, Options: {cm.n_options}")
            print()
            print(f"  Exact Accuracy:    {fmt(cm.exact_accuracy, '.1%')}")
            print(f"  Adjacent Accuracy: {fmt(cm.adjacent_accuracy, '.1%')} (within ±1)")
            print()
            print(
                f"  Weighted Kappa:    {fmt(cm.weighted_kappa, '.3f')} ({cm.kappa_interpretation})"
            )
            if cm.krippendorff_alpha is not None:
                print(f"  Krippendorff α:    {cm.krippendorff_alpha:.3f}  (recommended)")
            if cm.fleiss_kappa is not None:
                print(f"  Fleiss' Kappa:     {cm.fleiss_kappa:.3f}")
            print()
            print(
                f"  Spearman rho:      {fmt(cm.spearman.coefficient, '.3f')} "
                f"(p={fmt(cm.spearman.p_value, '.4f')})"
            )
            print(
                f"  Kendall tau:       {fmt(cm.kendall.coefficient, '.3f')} "
                f"(p={fmt(cm.kendall.p_value, '.4f')})"
            )
            print()
            print(f"  RMSE:              {fmt(cm.rmse, '.4f')}")
            print(f"  MAE:               {fmt(cm.mae, '.4f')}")

            # Per-option metrics
            print("\n  Per-Option Metrics:")
            print(f"    {'Option':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
            print(f"    {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
            for opt in cm.per_option:
                label = opt.label[:30]
                prec = fmt(opt.precision, ">8.2f", f"{'N/A':>8}")
                rec = fmt(opt.recall, ">8.2f", f"{'N/A':>8}")
                f1 = fmt(opt.f1, ">8.2f", f"{'N/A':>8}")
                print(f"    {label:<30} {prec} {rec} {f1} {opt.support_true:>8}")

            # Confusion matrix
            print("\n  Confusion Matrix:")
            conf_str = format_confusion_matrix(cm.confusion_matrix, cm.option_labels)
            for line in conf_str.split("\n"):
                print(f"    {line}")

        elif isinstance(cm, NominalCriterionMetrics):
            print(f"  Samples: {cm.n_samples}, Options: {cm.n_options}")
            print()
            print(f"  Exact Accuracy:    {fmt(cm.exact_accuracy, '.1%')}")
            print(f"  Cohen's Kappa:     {fmt(cm.kappa, '.3f')} ({cm.kappa_interpretation})")
            if cm.krippendorff_alpha is not None:
                print(f"  Krippendorff α:    {cm.krippendorff_alpha:.3f}  (recommended)")
            if cm.fleiss_kappa is not None:
                print(f"  Fleiss' Kappa:     {cm.fleiss_kappa:.3f}")

            # Per-option metrics
            print("\n  Per-Option Metrics:")
            print(f"    {'Option':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
            print(f"    {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
            for opt in cm.per_option:
                label = opt.label[:30]
                prec = fmt(opt.precision, ">8.2f", f"{'N/A':>8}")
                rec = fmt(opt.recall, ">8.2f", f"{'N/A':>8}")
                f1 = fmt(opt.f1, ">8.2f", f"{'N/A':>8}")
                print(f"    {label:<30} {prec} {rec} {f1} {opt.support_true:>8}")

            # Confusion matrix
            print("\n  Confusion Matrix:")
            conf_str = format_confusion_matrix(cm.confusion_matrix, cm.option_labels)
            for line in conf_str.split("\n"):
                print(f"    {line}")

        else:
            # Binary criterion (BinaryCriterionMetrics)
            print(f"  Samples: {cm.n_samples}")
            print()
            print(f"  Accuracy:          {fmt(cm.accuracy, '.1%')}")
            print(f"  Precision:         {fmt(cm.precision, '.2f')}")
            print(f"  Recall:            {fmt(cm.recall, '.2f')}")
            print(f"  F1:                {fmt(cm.f1, '.2f')}")
            print(f"  Cohen's Kappa:     {fmt(cm.kappa, '.3f')} ({cm.kappa_interpretation})")
            if cm.krippendorff_alpha is not None:
                print(f"  Krippendorff α:    {cm.krippendorff_alpha:.3f}  (recommended)")
            if cm.fleiss_kappa is not None:
                print(f"  Fleiss' Kappa:     {cm.fleiss_kappa:.3f}")
            print(f"  MET support:       {cm.support_true} (pred: {cm.support_pred})")

    # Individual results
    print("\n" + "=" * 80)
    print("INDIVIDUAL RESULTS")
    print("=" * 80)

    for item_result in eval_result.item_results:
        result = item_result.report
        item = item_result.item

        # Compute true score
        true_score = dataset.compute_weighted_score(item.ground_truth)
        score_error = abs(result.score - true_score) if result.score is not None else None

        print(f"\nItem {item_result.item_idx + 1}: {item.description}")
        print(f"  Predicted Score: {fmt(result.score, '.3f')}")
        print(f"  Actual Score:    {true_score:.3f}")
        print(f"  Error:           {fmt(score_error, '.3f')}")

        if result.report:
            # Show per-criterion verdicts
            print("  Verdicts:")
            for i, cr in enumerate(result.report):
                criterion = dataset.rubric.rubric[i]
                gt = item.ground_truth[i]
                if cr.verdict is not None:
                    verdict = cr.verdict.value
                    match = "✓" if str(verdict) == str(gt) else "✗"
                    print(f"    {criterion.name}: {verdict} (GT: {gt}) {match}")
                elif cr.multi_choice_verdict is not None:
                    mc = cr.multi_choice_verdict
                    match = "✓" if mc.selected_label == gt else "✗"
                    print(f"    {criterion.name}: {mc.selected_label} (GT: {gt}) {match}")

    # Timing stats
    print("\n" + "-" * 80)
    print("EVALUATION STATS")
    print("-" * 80)

    timing = eval_result.timing_stats
    print(f"\nThroughput:          {timing.items_per_second:.2f} items/second")
    print(f"Total Duration:      {timing.total_duration_seconds:.2f}s")
    print(f"Mean Item Duration:  {timing.mean_item_duration_seconds:.2f}s")
    print(f"P95 Item Duration:   {timing.p95_item_duration_seconds:.2f}s")

    # Cost summary
    if eval_result.total_completion_cost:
        print(f"\nTotal Cost: ${eval_result.total_completion_cost:.6f}")
        cost_per_item = eval_result.total_completion_cost / len(dataset)
        print(f"Cost per Item: ${cost_per_item:.6f}")


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set. Set it to run the demo.")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nAlternatively, modify llm_config to use a different provider.")
        exit(1)

    asyncio.run(main())
