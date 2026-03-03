"""Demo: Using reference_submission to calibrate grading expectations.

This example shows how a reference submission provides context for the grader
to understand what a high-quality response looks like, without requiring
strict comparison.
"""

import asyncio

from autorubric import (
    Criterion,
    CriterionGrader,
    LLMConfig,
    Rubric,
    RubricDataset,
)


async def main():
    # Define evaluation criteria
    rubric = Rubric([
        Criterion(
            name="completeness",
            weight=1.0,
            requirement="The explanation covers all key steps of photosynthesis",
        ),
        Criterion(
            name="accuracy",
            weight=1.0,
            requirement="Scientific facts are accurate and correctly stated",
        ),
        Criterion(
            name="clarity",
            weight=1.0,
            requirement="The explanation is clear and understandable",
        ),
    ])

    # Reference submission: exemplar of a high-quality response
    reference = """Photosynthesis is the process by which plants convert light energy
into chemical energy. It occurs primarily in the chloroplasts of plant cells.
The process has two main stages: the light-dependent reactions, which occur in
the thylakoid membranes and produce ATP and NADPH, and the light-independent
reactions (Calvin cycle), which occur in the stroma and use ATP and NADPH to
convert CO2 into glucose. The overall equation is:
6CO2 + 6H2O + light energy -> C6H12O6 + 6O2."""

    # Dataset with global reference_submission
    dataset = RubricDataset(
        prompt="Explain photosynthesis",
        rubric=rubric,
        name="photosynthesis_eval",
        reference_submission=reference,  # Global reference for all items
    )

    # Add student submissions to evaluate
    dataset.add_item(
        submission="Plants use sunlight to make food. They take in carbon dioxide "
        "and water and produce oxygen and sugar.",
        description="Student A - Basic understanding",
    )

    dataset.add_item(
        submission="Photosynthesis happens in chloroplasts. Light reactions make ATP "
        "and NADPH in thylakoids, then the Calvin cycle uses these to fix CO2 into "
        "sugar in the stroma.",
        description="Student B - More detailed",
    )

    # Configure grader
    grader = CriterionGrader(
        llm_config=LLMConfig(model="claude-sonnet-4-20250514"),
    )

    # Evaluate each submission
    for idx, item in enumerate(dataset.items):
        print(f"\n{'='*60}")
        print(f"Evaluating: {item.description}")
        print(f"Submission: {item.submission[:100]}...")

        # Get the effective reference (global in this case)
        ref = dataset.get_item_reference_submission(idx)
        print(f"Using reference: {ref[:50]}..." if ref else "No reference")

        report = await rubric.grade(
            to_grade=item.submission,
            grader=grader,
            query=dataset.prompt,
            reference_submission=ref,
        )

        print(f"\nScore: {report.score:.2f}")
        for cr in report.report:
            print(f"  - {cr.criterion.name}: {cr.final_verdict.value} ({cr.final_reason})")


if __name__ == "__main__":
    asyncio.run(main())
