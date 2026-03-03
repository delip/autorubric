#!/usr/bin/env python3
"""Convert ResearcherBench data to AutoRubric RubricDataset format.

This script reads:
- eval_data/questions.json: Question metadata (id, question, category, Subject)
- eval_data/rubric.json: Per-question rubrics (id, question, rubric[{point, weight}])
- user_data/*.json: System responses (id, question, response)

And produces:
- output/*.json: Single RubricDataset JSON file per system

Each output file is ONE RubricDataset with:
- prompt: Generic shared prompt
- rubric: null (per-item rubrics are used instead)
- items: List of all responses (65 items typically, 2 for test)
  - submission: Serialized JSON of {question, response}
  - description: Question metadata (category, subject, id)
  - rubric: Per-item rubric (criteria specific to this question)
"""

import json
from pathlib import Path
from typing import Any


GENERIC_PROMPT = "Provide a response to the question."


def load_json(path: Path) -> Any:
    """Load JSON file with UTF-8 encoding."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_questions_map(questions_path: Path) -> dict[int, dict]:
    """Load questions and index by ID."""
    questions = load_json(questions_path)
    return {q["id"]: q for q in questions}


def build_rubrics_map(rubrics_path: Path) -> dict[int, list[dict]]:
    """Load rubrics and index by question ID."""
    rubrics = load_json(rubrics_path)
    return {r["id"]: r["rubric"] for r in rubrics}


def convert_rubric_to_criteria(rubric_data: list[dict]) -> list[dict]:
    """Convert ResearcherBench rubric format to AutoRubric criteria format.

    Args:
        rubric_data: List of {point, weight} dicts from rubric.json

    Returns:
        List of Criterion-compatible dicts with name, weight, requirement
    """
    criteria = []
    for i, item in enumerate(rubric_data):
        criteria.append({
            "name": f"C{i + 1}",
            "weight": item["weight"],
            "requirement": item["point"],
        })
    return criteria


def create_item(
    question_id: int,
    question_data: dict,
    response: str,
    rubric_data: list[dict],
) -> dict:
    """Create a single DataItem-compatible entry with per-item rubric.

    Args:
        question_id: The question ID
        question_data: Question metadata (question, category, Subject)
        response: System's response text
        rubric_data: Per-question rubric from rubric.json

    Returns:
        DataItem-compatible dict with submission, description, and per-item rubric
    """
    submission_content = json.dumps({
        "question": question_data["question"],
        "response": response,
    }, ensure_ascii=False)

    category = question_data.get("category", "Unknown")
    subject = question_data.get("Subject", "Unknown")

    return {
        "submission": submission_content,
        "description": f"Q{question_id} [{category}] {subject}",
        "ground_truth": None,
        "rubric": convert_rubric_to_criteria(rubric_data),
    }


def process_user_data_file(
    user_data_path: Path,
    questions_map: dict[int, dict],
    rubrics_map: dict[int, list[dict]],
) -> dict:
    """Process a single user_data file and create ONE RubricDataset.

    Args:
        user_data_path: Path to the user data JSON file
        questions_map: Question metadata indexed by ID
        rubrics_map: Rubrics indexed by question ID

    Returns:
        Single RubricDataset-compatible dict with all items (per-item rubrics)
    """
    system_name = user_data_path.stem
    responses = load_json(user_data_path)

    items = []
    for response_entry in responses:
        question_id = response_entry["id"]
        response_text = response_entry["response"]

        if question_id not in questions_map:
            print(f"  Warning: Question ID {question_id} not found in questions.json")
            continue
        if question_id not in rubrics_map:
            print(f"  Warning: Question ID {question_id} not found in rubric.json")
            continue

        item = create_item(
            question_id=question_id,
            question_data=questions_map[question_id],
            response=response_text,
            rubric_data=rubrics_map[question_id],
        )
        items.append((question_id, item))

    items.sort(key=lambda x: x[0])
    sorted_items = [item for _, item in items]

    return {
        "name": system_name,
        "prompt": GENERIC_PROMPT,
        "rubric": None,  # Per-item rubrics are used instead of global rubric
        "items": sorted_items,
    }


def main():
    """Main conversion function."""
    base_dir = Path(__file__).parent
    eval_data_dir = base_dir / "eval_data"
    user_data_dir = base_dir / "user_data"
    output_dir = base_dir / "output"

    output_dir.mkdir(exist_ok=True)

    print("Loading questions and rubrics...")
    questions_map = build_questions_map(eval_data_dir / "questions.json")
    rubrics_map = build_rubrics_map(eval_data_dir / "rubric.json")
    print(f"  Loaded {len(questions_map)} questions and {len(rubrics_map)} rubrics")

    user_data_files = sorted(user_data_dir.glob("*.json"))
    print(f"\nFound {len(user_data_files)} user data files to process")

    for user_data_path in user_data_files:
        print(f"\nProcessing {user_data_path.name}...")

        dataset = process_user_data_file(
            user_data_path,
            questions_map,
            rubrics_map,
        )

        output_path = output_dir / f"{user_data_path.stem}_rubric_dataset.json"
        output_path.write_text(
            json.dumps(dataset, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Created {output_path.name} with {len(dataset['items'])} items")

    print("\nConversion complete!")
    print(f"Output files written to: {output_dir}")


if __name__ == "__main__":
    main()
