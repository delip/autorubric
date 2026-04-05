"""Rubric generation module for AutorubricLM training and evaluation.

Provides data models for rubric generation datasets: collections of
(input, output) pairs where the input is a grading problem description
and the output is a list[Criterion] rubric.

Example usage:
    from autorubric.generation import RubricGenerationExample, RubricGenerationDataset
    from autorubric.types import Criterion

    example = RubricGenerationExample(
        grading_problem_description="Evaluate a student essay on the causes of WWI.",
        criteria=[
            Criterion(name="thesis", weight=30.0, requirement="Has a clear thesis statement"),
            Criterion(name="evidence", weight=40.0, requirement="Supports claims with evidence"),
            Criterion(name="organization", weight=30.0, requirement="Logically organized argument"),
        ],
        domain="writing",
        source_dataset="manual",
        is_human_authored=True,
    )

    dataset = RubricGenerationDataset(name="v0.1", examples=[example])
    dataset.to_json("training_data.json")
"""

from .dataset import (
    CriterionMix,
    RubricGenerationDataset,
    RubricGenerationExample,
)

__all__ = [
    "CriterionMix",
    "RubricGenerationDataset",
    "RubricGenerationExample",
]
