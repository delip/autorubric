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

# result.score is `float | None` (None if the grade failed or no criterion was left to
# score); guard before formatting.
print(f"Score: {result.score:.2f}" if result.score is not None else "Score: n/a")
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

A criterion that abstains (`CANNOT_ASSESS`, or a multi-choice NA option) is left out of both sums under the default `SKIP` strategy (see [CANNOT_ASSESS Handling](cannot-assess.md)). Some items have no score, and then `score` and `raw_score` are `None`, never a fabricated 0.0:

- **No criterion left to score.** Under `SKIP`, every criterion abstained. Unless every judgment failed (next case), this is not a failed grade, and `error` is `None`.
- **The grade failed.** `error` says why. When the judgment of every criterion failed (for example, every judge call failed), each criterion keeps its stand-in verdict and its own `error`, and the report's `error` begins `Every criterion's judgment failed:`. This holds under every strategy.

`Rubric.compute_score`, which scores ground-truth verdicts, follows the same rule: it returns `None` when no criterion is left to score.

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

## Submissions with Images

A submission, a query and a reference submission are each `Content`. This is a string, an `Image`, or a list (or tuple) of strings and images read in order. A list such as a figure followed by its caption is called a document:

```python
from autorubric import Image

result = await rubric.grade(
    to_grade=[Image("figures/fig1.png"), "Figure 1: accuracy by model size."],
    grader=grader,
    query="Make a figure that shows how accuracy grows with model size.",
)
```

`Image(path)` refers to a file and reads nothing when it is made. `Image(data)` holds image bytes and checks their format at once. An image is PNG, JPEG, GIF or WebP. A list holding only strings is the same content as their concatenation, so text grades exactly as before.

`Grader.grade` reads every image of the submission, the query and the reference submission once, before any judge is called. A missing or unreadable file, or a file of another format, raises from the grade. It is never turned into a verdict. A list holding a part that is neither a string nor an `Image` raises `TypeError`, and so does a query or reference submission that is not `Content`. The `{"thinking": ..., "output": ...}` dict form stays text only. Any other submission is read as that dict form, as it was before images existed, so a dict with neither key grades as an empty submission and a number raises `AttributeError`. The length penalty counts the words of the text and ignores the images.

LLM judges receive the images in the prompt, where the text would be (see [LLM Infrastructure](llm.md#images-in-prompts)). Decision models receive them as assets fitted to the model's limits (see [Decision Models](decision-models.md#images)). Before any judge is called, `CriterionGrader` checks that every judge that could be sent the images accepts them. An LLM judge must have a model that LiteLLM's model map lists as accepting images, or `LLMConfig(images=True)`, and a decision model must have a registry entry, or declared `capabilities`, that accepts images. Otherwise the grade raises `ValueError` naming the judge, and no judge sends a request. For a walk-through, provider notes and dataset files, see the [Image Inputs](../cookbook/image-inputs.md) recipe.

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

## Image

An image in submission content, read from a file or held as bytes.

::: autorubric.Image
    options:
      show_source: true
      members_order: source

---

## Content

Type alias for a submission, query or reference submission: a string, an `Image`, or a sequence of both.

::: autorubric.Content
    options:
      show_source: true

---

## ToGradeInput

Type alias for the input format accepted by `rubric.grade()`: `Content`, or a dict with thinking and output sections.

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
