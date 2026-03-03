# Working with Grading Explanations

Learn how to access, display, and use per-criterion explanations from rubric grading.

## The Scenario

You're building an automated essay feedback system. Students submit essays, and you want to provide not just a score, but per-criterion feedback explaining why each requirement was met or not met. AutoRubric's grading produces these explanations automatically — you just need to access them.

## What You'll Learn

- Accessing `final_reason` from grading results
- Formatting explanations for student feedback
- Working with ensemble explanations (combined judge reasons)
- Filtering and categorizing reasons programmatically

## The Solution

### Step 1: Grade and Access Explanations

Every grading result contains a `report` — a list of `EnsembleCriterionReport` objects, each with a `final_reason` field:

```python
import asyncio
from autorubric import Rubric, LLMConfig
from autorubric.graders import CriterionGrader

rubric = Rubric.from_dict([
    {"name": "causes", "weight": 30.0, "requirement": "Identifies at least 2 major causes of the Industrial Revolution"},
    {"name": "effects", "weight": 30.0, "requirement": "Describes at least 2 major effects of the Industrial Revolution"},
    {"name": "structure", "weight": 12.0, "requirement": "Clear essay structure with introduction and logical flow"},
    {"name": "errors", "weight": -15.0, "requirement": "Contains significant factual errors"},
])

grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)
)

async def main():
    result = await rubric.grade(
        to_grade="The Industrial Revolution began in Britain around 1760...",
        grader=grader,
        query="Explain the causes and effects of the Industrial Revolution.",
    )

    for cr in result.report:
        verdict = cr.final_verdict.value if cr.final_verdict else "N/A"
        name = cr.criterion.name or "unnamed"
        print(f"[{verdict}] {name}: {cr.final_reason}")

asyncio.run(main())
```

### Step 2: Format as Student Feedback

Structure the explanations into a readable feedback report:

```python
def format_feedback(result):
    lines = [f"Overall Score: {result.score:.0%}\n"]

    met = [cr for cr in result.report if cr.final_verdict and cr.final_verdict.value == "MET" and cr.criterion.weight > 0]
    unmet = [cr for cr in result.report if cr.final_verdict and cr.final_verdict.value == "UNMET" and cr.criterion.weight > 0]
    errors = [cr for cr in result.report if cr.final_verdict and cr.final_verdict.value == "MET" and cr.criterion.weight < 0]

    if met:
        lines.append("Strengths:")
        for cr in met:
            lines.append(f"  + {cr.criterion.name}: {cr.final_reason}")

    if unmet:
        lines.append("\nAreas for Improvement:")
        for cr in unmet:
            lines.append(f"  - {cr.criterion.name}: {cr.final_reason}")

    if errors:
        lines.append("\nErrors Found:")
        for cr in errors:
            lines.append(f"  ! {cr.criterion.name}: {cr.final_reason}")

    return "\n".join(lines)
```

### Step 3: Ensemble Explanations

When using ensemble judging, `final_reason` combines all judges' explanations with a pipe (`|`) separator:

```python
from autorubric.graders import CriterionGrader, JudgeSpec

grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "gpt"),
        JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
    ],
    aggregation="majority",
)

result = await rubric.grade(to_grade=essay, grader=grader, query=prompt)

for cr in result.report:
    # Individual judge reasons are pipe-separated
    judge_reasons = cr.final_reason.split(" | ")
    print(f"[{cr.final_verdict.value}] {cr.criterion.name}")
    for i, reason in enumerate(judge_reasons):
        print(f"  Judge {i + 1}: {reason}")

    # Individual votes are also available
    for vote in cr.votes:
        print(f"  {vote.judge_id}: {vote.verdict.value} — {vote.reason}")
```

### Step 4: Programmatic Filtering

Extract specific explanations for downstream use:

```python
def get_unmet_feedback(result):
    """Extract reasons for criteria that were not met."""
    return {
        cr.criterion.name: cr.final_reason
        for cr in result.report
        if cr.final_verdict and cr.final_verdict.value == "UNMET" and cr.criterion.weight > 0
    }

def get_error_explanations(result):
    """Extract explanations for detected errors (negative-weight criteria that were MET)."""
    return {
        cr.criterion.name: cr.final_reason
        for cr in result.report
        if cr.final_verdict and cr.final_verdict.value == "MET" and cr.criterion.weight < 0
    }
```

## Key Takeaways

- Every `EnsembleCriterionReport` has a `final_reason` explaining the verdict
- For single-judge grading, `final_reason` is the judge's direct explanation
- For ensemble grading, `final_reason` concatenates all judges' reasons with ` | `; individual votes are in `cr.votes`
- Use `final_verdict.value` to get the string ("MET", "UNMET", "CANNOT_ASSESS")
- Negative-weight criteria that are MET indicate detected problems — their reasons explain the issue

## Going Further

- [Ensemble Judging](ensemble-judging.md) — Get multiple perspectives on each criterion
- [Extended Thinking](extended-thinking.md) — Enable deeper reasoning for complex evaluations
- [API Reference: Core Grading](../api/core-grading.md) — Full `CriterionReport` and `EnsembleCriterionReport` docs

---

## Appendix: Complete Code

```python
"""Working with Grading Explanations - Essay Feedback System"""

import asyncio
from pathlib import Path

from autorubric import LLMConfig
from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader

DATASET_PATH = Path(__file__).parent / "examples" / "data" / "essay_grading_dataset.json"


def format_feedback(result):
    """Format grading result as student-readable feedback."""
    lines = [f"Overall Score: {result.score:.0%}\n"]

    met = [cr for cr in result.report if cr.final_verdict and cr.final_verdict.value == "MET" and cr.criterion.weight > 0]
    unmet = [cr for cr in result.report if cr.final_verdict and cr.final_verdict.value == "UNMET" and cr.criterion.weight > 0]
    errors = [cr for cr in result.report if cr.final_verdict and cr.final_verdict.value == "MET" and cr.criterion.weight < 0]

    if met:
        lines.append("Strengths:")
        for cr in met:
            lines.append(f"  + {cr.criterion.name}: {cr.final_reason}")

    if unmet:
        lines.append("\nAreas for Improvement:")
        for cr in unmet:
            lines.append(f"  - {cr.criterion.name}: {cr.final_reason}")

    if errors:
        lines.append("\nErrors Found:")
        for cr in errors:
            lines.append(f"  ! {cr.criterion.name}: {cr.final_reason}")

    return "\n".join(lines)


async def main():
    dataset = RubricDataset.from_file(DATASET_PATH)

    grader = CriterionGrader(
        llm_config=LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)
    )

    item = dataset.items[0]
    rubric = dataset.get_item_rubric(0)
    prompt = dataset.get_item_prompt(0)

    print(f"Prompt: {prompt}")
    print(f"Submission: {item.description}")
    print("=" * 70)

    result = await rubric.grade(
        to_grade=item.submission,
        grader=grader,
        query=prompt,
    )

    print(format_feedback(result))


if __name__ == "__main__":
    asyncio.run(main())
```
