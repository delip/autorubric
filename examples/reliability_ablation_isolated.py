#!/usr/bin/env python3
"""
Isolated reliability ablation experiment for the autorubric paper.

Runs 4 configurations on the Hashemi et al. 2024 dataset (223 items,
9 multi-choice criteria). Each configuration adds exactly ONE technique
over the zero-shot baseline, isolating each technique's effect.

Configurations:
  1. Baseline         — single judge (Gemini Flash), no shuffle, no few-shot
  2. +Ensemble only   — 3 judges, majority vote (no shuffle, no few-shot)
  3. +Shuffling only  — single judge, shuffle_options=True (no ensemble, no few-shot)
  4. +Few-shot only   — single judge, 3-shot balanced (no ensemble, no shuffle)

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
                model="gemini/gemini-2.5-flash",
                temperature=0.0,
                cache_enabled=True,
                cache_dir=".autorubric_paper_experiments_cache",
                max_parallel_requests=10,
            ),
            judge_id="gemini-flash",
        ),
        JudgeSpec(
            llm_config=LLMConfig(
                model="anthropic/claude-sonnet-4-5-20250929",
                temperature=0.0,
                cache_enabled=True,
                cache_dir=".autorubric_paper_experiments_cache",
                max_parallel_requests=10,
            ),
            judge_id="claude-sonnet",
        ),
        JudgeSpec(
            llm_config=LLMConfig(
                model="openai/gpt-5.2",
                temperature=0.0,
                cache_enabled=True,
                cache_dir=".autorubric_paper_experiments_cache",
                max_parallel_requests=10,
            ),
            judge_id="gpt-5.2",
        ),
    ]


def build_configurations(
    train_ds: RubricDataset,
) -> dict[str, CriterionGrader]:
    """Build 4 configurations: baseline + 3 isolated single-technique additions."""
    judges = build_judges()
    baseline_llm = judges[0].llm_config

    return {
        "Baseline": CriterionGrader(
            llm_config=baseline_llm,
            shuffle_options=False,
        ),
        "+Ensemble only": CriterionGrader(
            judges=judges,
            aggregation="majority",
            shuffle_options=False,
        ),
        "+Shuffle only": CriterionGrader(
            llm_config=baseline_llm,
            shuffle_options=True,
        ),
        "+Few-shot only": CriterionGrader(
            llm_config=baseline_llm,
            shuffle_options=False,
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
    train_ds, test_ds = dataset.split_train_test(
        n_train=N_TRAIN, stratify=True, seed=SEED
    )

    print(
        f"Dataset: {dataset.name} ({len(dataset)} items, {dataset.num_criteria} criteria)"
    )
    print(f"Split: {len(train_ds)} train / {len(test_ds)} test\n")

    configs = build_configurations(train_ds)
    results: list[dict] = []

    for name, grader in configs.items():
        print(f"Running configuration: {name} ...")
        eval_result = await evaluate(
            dataset=test_ds,
            grader=grader,
            experiment_name=f"reliability-ablation-isolated-{name.lower().replace('+', 'plus_').replace(' ', '_')}",
            show_progress=True,
            resume=True,
        )

        metrics = eval_result.compute_metrics(test_ds)

        # Compute mean agreement across items (only meaningful for ensemble configs)
        mean_agreement = None
        if grader.is_ensemble:
            agreements = []
            for ir in eval_result.filter_successful():
                if (
                    hasattr(ir.report, "mean_agreement")
                    and ir.report.mean_agreement is not None
                ):
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

        print(
            f"  accuracy={metrics.criterion_accuracy:.1%}  "
            f"kappa={metrics.mean_kappa:.3f}  "
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
        agr = f"{r['agreement']:.1%}" if r["agreement"] is not None else "N/A"
        cst = f"${r['cost']:.2f}" if r["cost"] else "N/A"
        print(
            f"{r['config']:<16} | {r['accuracy']:>7.1%} | {r['kappa']:>6.3f} | {agr:>9} | {cst:>7}"
        )
    print("=" * 70)

    # Save JSON results
    output_path = Path("experiments") / "reliability_ablation_isolated_results.json"
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
