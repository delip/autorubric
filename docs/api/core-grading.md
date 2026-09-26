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
grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"))
result = await rubric.grade(to_grade="...", grader=grader)

# result.score is `float | None` (None if the grade failed); guard before formatting.
print(f"Score: {result.score:.2f}" if result.score is not None else "Score: n/a (grade failed)")
for cr in result.report:
    # `final_verdict` is None on error/multi-choice criteria; guard before printing.
    verdict = cr.final_verdict.value if cr.final_verdict is not None else "n/a"
    print(f"  [{verdict}] {cr.criterion.requirement}")
    print(f"    Reason: {cr.final_reason}")
```

## Score Calculation

For each criterion $i$:

- If verdict = MET, contribution = $w_i$
- If verdict = UNMET, contribution = 0

Final score:

$$
\text{score} = \max\left(0, \min\left(1, \frac{\sum_{i=1}^{n} \mathbb{1}[\text{verdict}_i = \text{MET}] \cdot w_i}{\sum_{i=1}^{n} \max(0, w_i)}\right)\right)
$$

## Rubric Guidelines

Guidelines are optional free text that applies to every criterion of a rubric: grading conventions, definitions, the audience, scale anchors. Pass them as the keyword-only `guidelines`:

```python
rubric = Rubric(
    [
        Criterion(name="thesis", weight=3, requirement="States a clear, arguable thesis"),
        Criterion(name="evidence", weight=2, requirement="Supports claims with cited evidence"),
    ],
    guidelines="'Cited' means any attribution, not formal citation style.",
)
```

Blank text (empty or whitespace only) means no guidelines and is stored as `None`. In a file, a rubric with guidelines is a dict; the list form still loads unchanged:

```json
{
  "guidelines": "'Cited' means any attribution, not formal citation style.",
  "criteria": [
    {"name": "thesis", "weight": 3, "requirement": "States a clear, arguable thesis"}
  ]
}
```

`Rubric.from_dict`, `from_json`, `from_yaml` and `from_file` read this form, and `"guidelines"` also combines with the `"sections"` and `"rubric"` forms. A dataset writes a rubric in the dict form only when it has guidelines, and a per-item rubric carries its own.

What each judge sees:

- **LLM judges** get a `<guidelines>` block at the start of every per-criterion prompt, stating that the criterion text governs and the guidelines clarify how to apply it. It sits in the user prompt, so a custom `system_prompt` keeps it. A rubric without guidelines produces exactly the prompts it produced before guidelines existed.
- **Decision models** get them once per request as `state["guidelines"]`; the framed questions add a sentence telling the model to apply them (see [Decision-Model Judges](../cookbook/decision-models.md#rubric-guidelines-shared-context)).
- **Custom graders** receive them when their `judge` accepts a `guidelines` keyword or `**kwargs`; otherwise grading proceeds without them and warns once per grader that it ignores rubric guidelines.
- **Meta-rubric evaluation** shows them to the meta-judge as part of the rubric under review. **Rubric improvement** keeps them unchanged on every revised rubric; the revision LLM sees them but does not revise them.

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

Literal type alias for multi-choice criterion scale types (ordinal, nominal).

::: autorubric.ScaleType
    options:
      show_source: true
