# Core Grading

Fundamental types for rubric-based evaluation: criteria, rubrics, verdicts, and evaluation reports.

## Overview

The core grading module provides the foundational types for defining evaluation criteria and receiving grading results. A `Rubric` contains multiple `Criterion` objects, each with a weight and requirement. Grading produces an `EvaluationReport` with per-criterion verdicts and explanations.

## Quick Example

```python
from autorubric import Rubric, Criterion, CriterionVerdict, LLMConfig
from autorubric.graders import CriterionGrader

# Define criteria
rubric = Rubric([
    Criterion(name="accuracy", weight=10.0, requirement="States the correct answer"),
    Criterion(name="clarity", weight=5.0, requirement="Explains reasoning clearly"),
    Criterion(weight=-15.0, requirement="Contains factual errors"),  # name optional
])

# Or from dict/file
rubric = Rubric.from_dict([
    {"weight": 10.0, "requirement": "States the correct answer"},
    {"requirement": "Explains reasoning clearly"},  # weight defaults to 10.0
])
rubric = Rubric.from_file("rubric.yaml")

# Grade
grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4.1-mini"))
result = await rubric.grade(to_grade="...", grader=grader)

# result.score is `float | None` (None if the grade failed); guard before formatting.
print(f"Score: {result.score:.2f}" if result.score is not None else "Score: n/a (grade failed)")
for cr in result.report:
    print(f"  [{cr.verdict}] {cr.criterion.requirement}")
    print(f"    Reason: {cr.reason}")
```

## Score Calculation

For each criterion $i$:

- If verdict = MET, contribution = $w_i$
- If verdict = UNMET, contribution = 0

Final score:

$$
\text{score} = \max\left(0, \min\left(1, \frac{\sum_{i=1}^{n} \mathbb{1}[\text{verdict}_i = \text{MET}] \cdot w_i}{\sum_{i=1}^{n} \max(0, w_i)}\right)\right)
$$

---

## Criterion

A single evaluation criterion with weight and requirement.

::: autorubric.Criterion
    options:
      show_source: true
      members_order: source

---

## CriterionVerdict

Enum representing the verdict for a criterion.

::: autorubric.CriterionVerdict
    options:
      show_source: true

---

## CriterionReport

Per-criterion result with verdict and explanation.

::: autorubric.CriterionReport
    options:
      show_source: true
      members_order: source

---

## CriterionJudgment

Structured output from LLM judge for a single criterion.

::: autorubric.CriterionJudgment
    options:
      show_source: true
      members_order: source

---

## Rubric

Collection of criteria for evaluation.

::: autorubric.Rubric
    options:
      show_source: true
      members_order: source

---

## EvaluationReport

Complete grading result with score and per-criterion reports.

::: autorubric.EvaluationReport
    options:
      show_source: true
      members_order: source

---

## TokenUsage

Token usage tracking for LLM calls.

::: autorubric.TokenUsage
    options:
      show_source: true
      members_order: source

---

## ToGradeInput

Type alias for the input format accepted by `rubric.grade()`.

::: autorubric.ToGradeInput
    options:
      show_source: true

---

## ThinkingOutputDict

TypedDict for responses with separate thinking and output sections.

::: autorubric.ThinkingOutputDict
    options:
      show_source: true

---

## ScaleType

Enum for criterion scale types (binary, ordinal, nominal).

::: autorubric.ScaleType
    options:
      show_source: true
