#!/usr/bin/env python3
"""Demo script for evaluating rubric quality using the meta-rubric.

This script demonstrates how to use the meta-rubrics to evaluate the quality
of grading rubrics. It supports two evaluation modes:

1. Standalone mode: Evaluate rubric quality in isolation
2. In-context mode: Evaluate rubric given task prompt

Usage:
    python meta_rubric_demo.py
"""

import asyncio
import json
from pathlib import Path

from autorubric import LLMConfig, Rubric
from autorubric.meta import (
    evaluate_rubric_in_context,
    evaluate_rubric_standalone,
    get_in_context_meta_rubric,
    get_standalone_meta_rubric,
)


def load_sample_rubric() -> Rubric:
    """Load a sample rubric from the researcher_bench dataset for testing."""
    dataset_path = (
        Path(__file__).parent.parent / "researcher_bench/output/Claude_rubric_dataset.json"
    )
    if not dataset_path.exists():
        return Rubric.from_dict(
            [
                {"weight": 10, "requirement": "Response is clear and well-organized"},
                {"weight": 8, "requirement": "Response addresses the main question"},
                {"weight": 5, "requirement": "Response is accurate and factual"},
            ]
        )

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("items"):
        first_item = data["items"][0]
        if first_item.get("rubric"):
            return Rubric.from_dict(first_item["rubric"])

    return Rubric.from_dict(
        [
            {"weight": 10, "requirement": "Response is clear and well-organized"},
            {"weight": 8, "requirement": "Response addresses the main question"},
        ]
    )


async def main() -> None:
    """Run the meta-rubric demo."""
    print("Meta-Rubric Demo: Evaluating Rubric Quality")
    print("=" * 60)

    llm_config = LLMConfig(
        model="gemini/gemini-2.5-flash",
        temperature=0.0,
        thinking="medium",
        max_parallel_requests=10,
    )

    sample_rubric = load_sample_rubric()
    print(f"Sample rubric has {len(sample_rubric.rubric)} criteria\n")

    # Standalone evaluation (terminal output)
    await evaluate_rubric_standalone(sample_rubric, llm_config, display="stdout")

    # In-context evaluation
    task_prompt = (
        "Write a comprehensive research summary on the applications of "
        "large language models in healthcare, including benefits, risks, "
        "and ethical considerations."
    )
    await evaluate_rubric_in_context(sample_rubric, task_prompt, llm_config, display="stdout")


def verify_rubric_loading() -> None:
    """Verify both meta-rubrics load correctly."""
    print("Verifying meta-rubric files load correctly...")

    standalone = get_standalone_meta_rubric()
    print(f"  Standalone: {len(standalone.rubric)} criteria loaded")

    positive_weight = sum(c.weight for c in standalone.rubric if c.weight > 0)
    negative_weight = sum(c.weight for c in standalone.rubric if c.weight < 0)
    print(f"    Positive weight total: {positive_weight}")
    print(f"    Negative weight total: {negative_weight}")

    in_context = get_in_context_meta_rubric()
    print(f"  In-context: {len(in_context.rubric)} criteria loaded")

    positive_weight = sum(c.weight for c in in_context.rubric if c.weight > 0)
    negative_weight = sum(c.weight for c in in_context.rubric if c.weight < 0)
    print(f"    Positive weight total: {positive_weight}")
    print(f"    Negative weight total: {negative_weight}")

    print("\nAll meta-rubrics loaded successfully!")


if __name__ == "__main__":
    verify_rubric_loading()
    print()
    asyncio.run(main())
