# Utilities

Helper functions for token aggregation, text processing, and data manipulation.

## Overview

Utility functions for common operations like aggregating token usage across multiple evaluations, parsing thinking/output responses, and generating synthetic ground truth labels.

## Token Aggregation

```python
from autorubric import (
    aggregate_token_usage,
    aggregate_completion_cost,
    aggregate_evaluation_usage,
)

# After batch grading
results = await asyncio.gather(*[rubric.grade(r, grader) for r in responses])

# Aggregate usage and cost
total_usage, total_cost = aggregate_evaluation_usage(results)

# Or aggregate manually
usages = [r.token_usage for r in results]
costs = [r.completion_cost for r in results]

total_usage = aggregate_token_usage(usages)
total_cost = aggregate_completion_cost(costs)

if total_usage:
    print(f"Total tokens: {total_usage.total_tokens}")
if total_cost:
    print(f"Total cost: ${total_cost:.4f}")
```

## Thinking Output Parsing

```python
from autorubric import parse_thinking_output, normalize_to_grade_input

# Parse string with markers
text = "<thinking>Reasoning here</thinking><output>Final answer</output>"
parsed = parse_thinking_output(text)
# {'thinking': 'Reasoning here', 'output': 'Final answer'}

# Normalize any input format
input1 = "plain text"
input2 = {"thinking": "...", "output": "..."}
input3 = "<thinking>...</thinking><output>...</output>"

normalized = normalize_to_grade_input(input1)  # {'thinking': '', 'output': 'plain text'}
normalized = normalize_to_grade_input(input2)  # passes through
normalized = normalize_to_grade_input(input3)  # parses markers
```

## Synthetic Ground Truth

```python
from autorubric import RubricDataset, LLMConfig
from autorubric.graders import CriterionGrader
from autorubric import fill_ground_truth

async def generate_labels():
    dataset = RubricDataset.from_file("unlabeled.json")

    # Use strong model for ground truth
    grader = CriterionGrader(
        llm_config=LLMConfig(
            model="anthropic/claude-sonnet-4-5-20250929",
            max_parallel_requests=10,
        )
    )

    labeled = await fill_ground_truth(
        dataset,
        grader,
        force=False,        # Only label items without ground_truth
        show_progress=True,
    )

    labeled.to_file("labeled.json")
```

## Verdict Helpers

```python
from autorubric import (
    CriterionVerdict,
    extract_verdicts_from_report,
    filter_cannot_assess,
    verdict_to_binary,
    verdict_to_string,
)

# Extract verdicts from an evaluation report (pass the report object plus a
# criterion count)
pred_verdicts = extract_verdicts_from_report(result.report, num_criteria=len(rubric.rubric))
true_verdicts = extract_verdicts_from_report(reference.report, num_criteria=len(rubric.rubric))

# Filter out CANNOT_ASSESS (drops pairs where either side abstains; returns a tuple)
filtered_pred, filtered_true = filter_cannot_assess(pred_verdicts, true_verdicts)

# Convert to binary (for metrics) — takes a sequence, returns a list
binary = verdict_to_binary([CriterionVerdict.MET, CriterionVerdict.UNMET])  # [1, 0]

# Convert to string — takes a sequence, returns a list
strings = verdict_to_string([CriterionVerdict.MET])  # ["MET"]
```

---

## aggregate_token_usage

Aggregate token usage from multiple evaluations.

::: autorubric.aggregate_token_usage
    options:
      show_source: true

---

## aggregate_completion_cost

Aggregate completion costs from multiple evaluations.

::: autorubric.aggregate_completion_cost
    options:
      show_source: true

---

## aggregate_evaluation_usage

Aggregate both usage and cost from evaluation reports.

::: autorubric.aggregate_evaluation_usage
    options:
      show_source: true

---

## fill_ground_truth

Generate synthetic ground truth labels for unlabeled datasets.

::: autorubric.fill_ground_truth
    options:
      show_source: true

---

## parse_thinking_output

Parse text with thinking/output markers.

::: autorubric.parse_thinking_output
    options:
      show_source: true

---

## normalize_to_grade_input

Normalize any input format to ThinkingOutputDict.

::: autorubric.normalize_to_grade_input
    options:
      show_source: true

---

## word_count

Count words in text (default length penalty function).

::: autorubric.word_count
    options:
      show_source: true

---

## extract_verdicts_from_report

Extract verdicts from criterion reports.

::: autorubric.extract_verdicts_from_report
    options:
      show_source: true

---

## filter_cannot_assess

Filter out CANNOT_ASSESS verdicts.

::: autorubric.filter_cannot_assess
    options:
      show_source: true

---

## verdict_to_binary

Convert verdict to binary value.

::: autorubric.verdict_to_binary
    options:
      show_source: true

---

## verdict_to_string

Convert verdict to string representation.

::: autorubric.verdict_to_string
    options:
      show_source: true
