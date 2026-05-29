#!/usr/bin/env python3
"""
Reliability ablation experiment for the autorubric paper.

Runs 4 cumulative configurations on the Hashemi et al. 2024 dataset (223 items,
9 multi-choice criteria) and outputs a results table showing the incremental
benefit of each reliability technique.

Configurations (cumulative, ordered cheapest-first):
  1. Baseline         — single judge (Gemini Flash), no shuffle
  2. +Shuffling       — single judge, shuffle_options=True
  3. +Few-shot        — single judge, shuffle, 3-shot balanced examples
  4. Full (+Ensemble) — 3 judges, shuffle, 3-shot balanced, majority vote

All configurations evaluate on the same 203-item test set (20 items held out
for few-shot training) for fair comparison.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import litellm
from dotenv import load_dotenv

from autorubric import FewShotConfig, LLMConfig, evaluate
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader, JudgeSpec

load_dotenv()
litellm.suppress_debug_info = True

DATASET_PATH = Path(__file__).parent / "data" / "hashemi_etal_2024_dataset.json"
SEED = 42
N_TRAIN = 20


def build_judges() -> list[JudgeSpec]:
    """Build the 3-judge ensemble panel."""
    return [
        JudgeSpec(
            llm_config=LLMConfig(
                model="groq/openai/gpt-oss-120b",
                temperature=0.0,
                max_parallel_requests=1,
                cache_enabled=True,
                cache_dir=".autorubric_paper_experiments_cache",
            ),
            judge_id="gpt-oss-120b",
        ),
        JudgeSpec(
            llm_config=LLMConfig(
                model="groq/moonshotai/kimi-k2-instruct-0905",
                temperature=0.0,
                max_parallel_requests=2,
                cache_enabled=True,
                cache_dir=".autorubric_paper_experiments_cache",
            ),
            judge_id="kimi-k2-instruct",
        ),
        JudgeSpec(
            llm_config=LLMConfig(
                model="gemini/gemini-3-flash-preview",
                temperature=0.0,
                max_parallel_requests=10,
                cache_enabled=True,
                cache_dir=".autorubric_paper_experiments_cache",
            ),
            judge_id="gemini-3-flash",
        ),
    ]


def build_configurations(
    train_ds: RubricDataset,
) -> dict[str, CriterionGrader]:
    """Build the 4 cumulative ablation configurations."""
    judges = build_judges()
    baseline_llm = judges[0].llm_config

    return {
        "Baseline": CriterionGrader(
            llm_config=baseline_llm,
            shuffle_options=False,
        ),
        "+Shuffling": CriterionGrader(
            llm_config=baseline_llm,
            shuffle_options=True,
        ),
        "+Few-shot": CriterionGrader(
            llm_config=baseline_llm,
            shuffle_options=True,
            training_data=train_ds,
            few_shot_config=FewShotConfig(
                n_examples=3,
                balance_verdicts=True,
                seed=SEED,
            ),
        ),
        "Full (+Ensemble)": CriterionGrader(
            judges=judges,
            aggregation="majority",
            shuffle_options=True,
            training_data=train_ds,
            few_shot_config=FewShotConfig(
                n_examples=3,
                balance_verdicts=True,
                seed=SEED,
            ),
        ),
    }


async def run_ablation() -> None:
    dataset = RubricDataset.from_file(DATASET_PATH)
    train_ds, test_ds = dataset.split_train_test(n_train=N_TRAIN, stratify=True, seed=SEED)

    print(f"Dataset: {dataset.name} ({len(dataset)} items, {dataset.num_criteria} criteria)")
    print(f"Split: {len(train_ds)} train / {len(test_ds)} test\n")

    configs = build_configurations(train_ds)
    results: list[dict] = []

    for name, grader in configs.items():
        print(f"Running configuration: {name} ...")
        eval_result = await evaluate(
            dataset=test_ds,
            grader=grader,
            experiment_name=f"reliability-ablation-{name.lower().replace('+', 'plus_').replace(' ', '_')}",
            show_progress=True,
            resume=False,
        )

        metrics = eval_result.compute_metrics(test_ds)

        # Compute mean agreement across items (only meaningful for ensemble configs)
        mean_agreement = None
        if grader.is_ensemble:
            agreements = []
            for ir in eval_result.filter_successful():
                if hasattr(ir.report, "mean_agreement") and ir.report.mean_agreement is not None:
                    agreements.append(ir.report.mean_agreement)
            if agreements:
                mean_agreement = sum(agreements) / len(agreements)

        cost = eval_result.total_completion_cost

        results.append(
            {
                "config": name,
                "accuracy": metrics.criterion_accuracy,
                "kappa": metrics.mean_kappa,
                "agreement": mean_agreement,
                "cost": cost,
            }
        )

        acc_str = (
            f"{metrics.criterion_accuracy:.1%}" if metrics.criterion_accuracy is not None else "N/A"
        )
        kappa_str = f"{metrics.mean_kappa:.3f}" if metrics.mean_kappa is not None else "N/A"
        print(
            f"  accuracy={acc_str}  "
            f"kappa={kappa_str}  "
            f"agreement={f'{mean_agreement:.1%}' if mean_agreement is not None else 'N/A':>6}  "
            f"cost=${cost:.2f}"
            if cost
            else ""
        )
        print()

    # Print results table
    print("=" * 70)
    print(
        f"{'Configuration':<16} | {'Accuracy':>8} | {'Kappa':>6} | {'Agreement':>9} | {'Cost':>7}"
    )
    print("-" * 70)
    for r in results:
        acc = f"{r['accuracy']:>7.1%}" if r["accuracy"] is not None else f"{'N/A':>7}"
        kap = f"{r['kappa']:>6.3f}" if r["kappa"] is not None else f"{'N/A':>6}"
        agr = f"{r['agreement']:.1%}" if r["agreement"] is not None else "N/A"
        cst = f"${r['cost']:.2f}" if r["cost"] else "N/A"
        print(f"{r['config']:<16} | {acc} | {kap} | {agr:>9} | {cst:>7}")
    print("=" * 70)

    # Save JSON results
    output_path = Path("experiments") / "reliability_ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "dataset": str(DATASET_PATH),
                "n_train": N_TRAIN,
                "n_test": len(test_ds),
                "seed": SEED,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


def check_api_keys() -> list[str]:
    missing = []
    for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if not os.environ.get(key):
            missing.append(key)
    return missing


if __name__ == "__main__":
    missing = check_api_keys()
    if missing:
        print("Missing API keys:")
        for key in missing:
            print(f"  - {key}")
        sys.exit(1)
    asyncio.run(run_ablation())
