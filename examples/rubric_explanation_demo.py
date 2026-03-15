#!/usr/bin/env python3
"""
Demonstrates accessing per-criterion explanations from rubric grading.

Grades a single essay from the essay dataset and displays the judge's
reasoning for each criterion verdict.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from autorubric import LLMConfig
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader

load_dotenv()

DATASET_PATH = Path(__file__).parent / "data" / "ricechem" / "q1.json"


async def main():
    dataset = RubricDataset.from_file(DATASET_PATH)

    grader = CriterionGrader(
        llm_config=LLMConfig(
            model="openai/gpt-4.1-mini",
            temperature=0.0,
        )
    )

    # Grade just the first item
    item = dataset.items[1]
    rubric = dataset.get_item_rubric(0)
    prompt = dataset.get_item_prompt(0)

    print(f"Prompt: {prompt}\n")
    print(f"Submission: {item.description}")
    print("=" * 70)

    result = await rubric.grade(
        to_grade=item.submission,
        grader=grader,
        query=prompt,
    )

    print(f"\nScore: {result.score:.2f}\n")

    # Display per-criterion explanations
    for cr in result.report:
        verdict = cr.verdict.value if cr.verdict else "N/A"
        name = cr.name or "unnamed"
        weight = cr.weight
        sign = "+" if weight > 0 else ""

        print(f"[{verdict}] {name} ({sign}{weight})")
        print(f"  Reason: {cr.reason}")
        print()


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY='your-key-here'")
        exit(1)

    asyncio.run(main())
