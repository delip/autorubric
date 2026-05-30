#!/usr/bin/env python3
"""
Evaluate Gemini 3 Flash on the RiceChem dataset using an 80-10-10
train/val/test split, following the protocol from Sonkar et al. (2024).

When FEW_SHOT=False (zero-shot), the training split is unused and only the
10% test set is evaluated. When FEW_SHOT=True, the 80% training split
provides few-shot examples and the 10% test set is evaluated. The val split
is unused in this demo but the split ratios match the paper.

Paper baseline (GPT-4 zero-shot, 10% test): 70.9% accuracy, 0.689 F1.

Usage:
    export GEMINI_API_KEY='your-key-here'
    python examples/ricechem_demo.py
"""

import asyncio
import hashlib
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from autorubric import FewShotConfig, LLMConfig, evaluate
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.types import CriterionVerdict

load_dotenv()

DATA_DIR = Path(__file__).parent / "data" / "ricechem"


USE_ENSEMBLE = False
FEW_SHOT = False
FEW_SHOT_N_EXAMPLES = 3
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 1 - (TRAIN_FRACTION + VAL_FRACTION)
SPLIT_SEED = 42

MODEL = "gemini/gemini-3-flash-preview"
ENSEMBLE_MODELS = [
    "gemini/gemini-3-flash-preview",
    "openai/gpt-5-mini-2025-08-07",
    "anthropic/claude-haiku-4-5-20251001",
]


def micro_average_from_verdicts(
    all_pred: list[CriterionVerdict], all_true: list[CriterionVerdict]
) -> dict:
    """Compute micro-averaged accuracy, precision, recall, F1 from verdict lists."""
    tp = fp = fn = tn = 0
    for p, t in zip(all_pred, all_true):
        if t == CriterionVerdict.MET:
            if p == CriterionVerdict.MET:
                tp += 1
            else:
                fn += 1
        else:
            if p == CriterionVerdict.MET:
                fp += 1
            else:
                tn += 1
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }


def extract_verdicts(eval_result, dataset) -> tuple[list, list]:
    """Extract predicted and ground truth verdict lists from an eval result."""
    all_pred = []
    all_true = []
    for item_result in eval_result.item_results:
        if item_result.error is not None:
            continue
        item = dataset.items[item_result.item_idx]
        for i, cr in enumerate(item_result.report.report):
            pred = cr.final_verdict if hasattr(cr, "final_verdict") else cr.verdict
            true = item.ground_truth[i]
            if isinstance(true, str):
                true = CriterionVerdict(true)
            if pred == CriterionVerdict.CANNOT_ASSESS:
                continue
            all_pred.append(pred)
            all_true.append(true)
    return all_pred, all_true


def build_grader(
    training_data: RubricDataset | None = None,
) -> tuple[CriterionGrader, str, str]:
    """Build grader and return (grader, config_str, model_label)."""
    few_shot_config = None
    if training_data is not None:
        few_shot_config = FewShotConfig(
            n_examples=FEW_SHOT_N_EXAMPLES,
            balance_verdicts=True,
            seed=SPLIT_SEED,
        )

    if USE_ENSEMBLE:
        judges = [
            JudgeSpec(
                llm_config=LLMConfig(
                    model=m,
                    temperature=1.0,
                    max_parallel_requests=5,
                    cache_enabled=True,
                    cache_dir=".autorubric_cache",
                ),
                judge_id=m.split("/")[-1],
            )
            for m in ENSEMBLE_MODELS
        ]
        grader = CriterionGrader(
            judges=judges,
            aggregation="majority",
            normalize=True,
            training_data=training_data,
            few_shot_config=few_shot_config,
        )
        config_str = f"ensemble:{':'.join(ENSEMBLE_MODELS)}"
        model_label = f"Ensemble ({', '.join(m.split('/')[-1] for m in ENSEMBLE_MODELS)})"
    else:
        llm_config = LLMConfig(
            model=MODEL,
            temperature=1.0,
            max_parallel_requests=10,
            cache_enabled=True,
            cache_dir=".autorubric_cache",
            cache_ttl=None,
            thinking=None,
        )
        grader = CriterionGrader(
            llm_config=llm_config,
            normalize=True,
            training_data=training_data,
            few_shot_config=few_shot_config,
        )
        config_str = f"{MODEL}:t={llm_config.temperature}:thinking={llm_config.thinking}"
        model_label = MODEL

    if training_data is not None:
        config_str += f":fewshot={FEW_SHOT_N_EXAMPLES}"

    return grader, config_str, model_label


async def evaluate_question(
    q: int,
    grader: CriterionGrader | None,
    config_hash: str,
):
    """Load dataset, split 80-10-10, evaluate on test set, and return results."""
    full_dataset = RubricDataset.from_file(DATA_DIR / f"q{q}.json")

    # 80-10-10 train/val/test split (paper protocol)
    n_test = max(1, math.floor(len(full_dataset) * TEST_FRACTION))
    n_remaining = len(full_dataset) - n_test
    remaining, test_dataset = full_dataset.split_train_test(
        n_train=n_remaining, stratify=False, seed=SPLIT_SEED
    )
    n_train = max(1, math.floor(len(full_dataset) * TRAIN_FRACTION))
    train_dataset, _val_dataset = remaining.split_train_test(
        n_train=n_train, stratify=False, seed=SPLIT_SEED
    )

    if FEW_SHOT:
        grader, _, _ = build_grader(training_data=train_dataset)

    print(f"\n{'=' * 70}")
    print(
        f"Q{q}: {full_dataset.name} — {len(full_dataset)} total, "
        f"{len(train_dataset)} train, {len(_val_dataset)} val, "
        f"{len(test_dataset)} test, {len(full_dataset.rubric.rubric)} criteria"
    )
    print(f"{'=' * 70}")

    eval_result = await evaluate(
        dataset=test_dataset,
        grader=grader,
        show_progress=True,
        progress_style="simple",
        experiment_name=f"ricechem-q{q}-{config_hash}",
        resume=False,
    )

    metrics = eval_result.compute_metrics(test_dataset)
    print(metrics.summary())

    return eval_result, metrics, test_dataset


async def main():
    if FEW_SHOT:
        # Grader is built per-question inside evaluate_question()
        grader = None
        # Compute config_hash using the few-shot config string
        _, config_str, model_label = build_grader()
        config_str += f":fewshot={FEW_SHOT_N_EXAMPLES}"
    else:
        grader, config_str, model_label = build_grader()

    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]

    shot_label = f"{FEW_SHOT_N_EXAMPLES}-Shot" if FEW_SHOT else "Zero-Shot"
    print(f"RiceChem {shot_label} Evaluation")
    print(f"Model: {model_label}")
    print(
        f"Split: {TRAIN_FRACTION:.0%} train / {VAL_FRACTION:.0%} val / "
        f"{TEST_FRACTION:.0%} test (seed={SPLIT_SEED})"
    )
    print("Paper baseline (GPT-4 Zero-Shot, 10% test): 70.9% accuracy, 0.689 F1")

    all_results = []
    for q in range(1, 5):
        result = await evaluate_question(q, grader, config_hash)
        all_results.append(result)

    # Micro-averaged metrics across all questions
    global_pred = []
    global_true = []
    for eval_result, _, dataset in all_results:
        pred, true = extract_verdicts(eval_result, dataset)
        global_pred.extend(pred)
        global_true.extend(true)

    agg = micro_average_from_verdicts(global_pred, global_true)

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    # Per-question table
    print(
        f"\n{'Question':<10} {'Accuracy':>10} {'F1':>10} {'Precision':>10} "
        f"{'Recall':>10} {'Kappa':>10} {'Pairs':>8}"
    )
    print("-" * 68)
    for q, (eval_result, metrics, dataset) in enumerate(all_results, 1):
        pred, true = extract_verdicts(eval_result, dataset)
        qm = micro_average_from_verdicts(pred, true)
        kappa_str = (
            f"{metrics.mean_kappa:>10.3f}" if metrics.mean_kappa is not None else f"{'N/A':>10}"
        )
        print(
            f"Q{q:<9} {qm['accuracy']:>10.3f} {qm['f1']:>10.3f} "
            f"{qm['precision']:>10.3f} {qm['recall']:>10.3f} "
            f"{kappa_str} {qm['total']:>8}"
        )

    print("-" * 68)
    print(
        f"{'Aggregate':<10} {agg['accuracy']:>10.3f} {agg['f1']:>10.3f} "
        f"{agg['precision']:>10.3f} {agg['recall']:>10.3f} "
        f"{'':>10} {agg['total']:>8}"
    )

    print(f"\nPaper (GPT-4 Zero-Shot):      {'0.709':>10} {'0.689':>10}")

    # Cost summary
    total_cost = sum(r[0].total_completion_cost or 0 for r in all_results)
    total_tokens = sum(
        (r[0].total_token_usage.total_tokens if r[0].total_token_usage else 0) for r in all_results
    )
    errors = sum(sum(1 for ir in r[0].item_results if ir.error is not None) for r in all_results)
    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Total tokens: {total_tokens:,}")
    if errors:
        print(f"Errors: {errors} items failed")


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set.")
        print("  export GEMINI_API_KEY='your-key-here'")
        exit(1)

    asyncio.run(main())
