#!/usr/bin/env python3
"""Convert Sharma et al. (2025) ResearchRubrics to RubricGenerationDataset.

Reads the raw ResearchRubrics JSON (101 deep-research tasks with expert-written
rubrics from Scale AI) and produces a single RubricGenerationDataset JSON
in ``data/autorubriclm/converted/sharma_etal_2025.json``.

Each item becomes one RubricGenerationExample:
  - grading_problem_description: the original prompt, framed as a grading task
  - criteria: binary criteria with positive weights normalised to sum to 100;
    negative weights (penalties) scaled by the same factor
  - domain: extracted from the item's description metadata
  - reference_response: None (no gold-standard answer in the raw data)
  - responses: None (no sample submissions in the raw data)

Source: Sharma et al. (2025), "ResearchRubrics: Evaluating Long-Form Research
with Fine-Grained Expert Rubrics", arXiv:2511.07685.
HuggingFace: ScaleAI/researchrubrics
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from autorubric.generation.dataset import (
    RubricGenerationDataset,
    RubricGenerationExample,
)
from autorubric.types import Criterion

RAW_PATH = ROOT / "examples" / "data" / "sharma_etal_2025_research_rubrics.json"
OUT_PATH = ROOT / "data" / "autorubriclm" / "converted" / "sharma_etal_2025.json"

# Domain slug mapping for cleaner domain tags
DOMAIN_SLUGS: dict[str, str] = {
    "AI & ML": "ai_ml",
    "Business Planning & Research": "business_planning",
    "Creative Writing": "creative_writing",
    "Current Events": "current_events",
    "General Consumer Research": "consumer_research",
    "Historical Analysis": "historical_analysis",
    "Hypotheticals & Philosophy": "philosophy",
    "Other": "other",
    "STEM": "stem",
    "Technical Documentation": "technical_documentation",
}


def extract_domain(description: str) -> str:
    """Extract the domain tag from the item description metadata string."""
    match = re.search(r"domain:\s*([^|]+)", description)
    if match:
        raw = match.group(1).strip()
        return DOMAIN_SLUGS.get(raw, raw.lower().replace(" ", "_"))
    return "unknown"


def normalise_criteria(raw_criteria: list[dict]) -> list[Criterion]:
    """Normalise positive weights to sum to 100; scale negatives by the same factor.

    Negative-weight criteria are penalties and are NOT included in the sum-to-100
    target, but they are scaled by the same normalisation factor so the relative
    magnitudes are preserved.
    """
    positive_sum = sum(c["weight"] for c in raw_criteria if c["weight"] > 0)
    if positive_sum == 0:
        factor = 1.0
    else:
        factor = 100.0 / positive_sum

    criteria = []
    for c in raw_criteria:
        criteria.append(
            Criterion(
                name=c["name"],
                weight=round(c["weight"] * factor, 2),
                requirement=c["requirement"],
            )
        )
    return criteria


def frame_grading_problem(prompt: str, domain: str) -> str:
    """Wrap the original user prompt in a grading-task framing."""
    domain_labels = {
        "ai_ml": "an AI/ML research",
        "business_planning": "a business planning",
        "creative_writing": "a creative writing",
        "current_events": "a current events",
        "consumer_research": "a consumer research",
        "historical_analysis": "a historical analysis",
        "philosophy": "a philosophical/hypothetical",
        "other": "a general knowledge",
        "stem": "a STEM",
        "technical_documentation": "a technical documentation",
    }
    label = domain_labels.get(domain, "a deep research")
    return (
        f"Evaluate the quality of a deep-research response to the following "
        f"{label} query:\n\n{prompt}"
    )


def convert_item(item: dict, idx: int) -> RubricGenerationExample:
    """Convert a single ResearchRubrics item to a RubricGenerationExample."""
    domain = extract_domain(item["description"])
    criteria = normalise_criteria(item["rubric"])
    grading_problem = frame_grading_problem(item["prompt"], domain)

    positive_sum = sum(c.weight for c in criteria if c.weight > 0)
    negative_sum = sum(c.weight for c in criteria if c.weight < 0)
    num_positive = sum(1 for c in criteria if c.weight > 0)
    num_negative = sum(1 for c in criteria if c.weight < 0)

    print(
        f"  [{idx + 1:3d}] {domain:<25s} "
        f"criteria={len(criteria):2d} "
        f"(+{num_positive}/-{num_negative})  "
        f"pos_sum={positive_sum:6.1f}  "
        f"neg_sum={negative_sum:6.1f}"
    )

    example = RubricGenerationExample(
        grading_problem_description=grading_problem,
        criteria=criteria,
        source_dataset="sharma_etal_2025",
        is_human_authored=True,
        domain=domain,
        language="en",
    )
    # Set criterion_mix from the computed property rather than hardcoding
    example = example.model_copy(update={"criterion_mix": example.effective_criterion_mix})
    return example


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw["items"]
    print(f"Loaded {len(items)} items from {RAW_PATH.name}\n")

    seen_ids: set[str] = set()
    examples: list[RubricGenerationExample] = []

    for idx, item in enumerate(items):
        example = convert_item(item, idx)
        if example.id in seen_ids:
            print(f"  WARNING: duplicate content hash {example.id}, skipping")
            continue
        seen_ids.add(example.id)
        examples.append(example)

    dataset = RubricGenerationDataset(
        name="sharma_etal_2025",
        version="0.1.0",
        examples=examples,
    )

    dataset.to_json(str(OUT_PATH))

    # Summary statistics
    all_criteria = [c for ex in examples for c in ex.criteria]
    domains = {ex.domain for ex in examples}
    print(f"\n{'=' * 60}")
    print(f"Wrote {len(dataset)} examples to {OUT_PATH}")
    print(f"  Domains: {sorted(domains)}")
    print(f"  Total criteria: {len(all_criteria)}")
    print(
        f"  Criteria per example: "
        f"min={min(ex.num_criteria for ex in examples)}, "
        f"max={max(ex.num_criteria for ex in examples)}, "
        f"mean={sum(ex.num_criteria for ex in examples) / len(examples):.1f}"
    )
    print(
        f"  Negative criteria: "
        f"{sum(1 for c in all_criteria if c.weight < 0)} "
        f"({sum(1 for c in all_criteria if c.weight < 0) / len(all_criteria) * 100:.1f}%)"
    )

    # Round-trip verification
    loaded = RubricGenerationDataset.from_json(str(OUT_PATH))
    assert len(loaded) == len(dataset), (
        f"Round-trip length mismatch: {len(loaded)} != {len(dataset)}"
    )
    for orig, rt in zip(dataset.examples, loaded.examples):
        assert orig.id == rt.id, f"ID mismatch: {orig.id} != {rt.id}"
        assert orig.num_criteria == rt.num_criteria
        assert orig.grading_problem_description == rt.grading_problem_description
        for co, cr in zip(orig.criteria, rt.criteria):
            assert co.weight == cr.weight, f"Weight mismatch: {co.weight} != {cr.weight}"
            assert co.requirement == cr.requirement
    print("Round-trip verification passed.")


if __name__ == "__main__":
    main()
