# Meta-Rubric Evaluation

Utilities for assessing the quality of grading rubrics using meta-rubrics (rubrics for evaluating rubrics).

## Overview

The `autorubric.meta` module provides tools to evaluate rubric quality before using them for actual grading. This helps identify issues like vague criteria, anti-patterns, or misalignment with the task being evaluated.

Two evaluation modes are supported:

| Mode | Function | Use Case |
|------|----------|----------|
| **Standalone** | `evaluate_rubric_standalone()` | Evaluate rubric quality in isolation (clarity, structure, LLM-friendliness) |
| **In-Context** | `evaluate_rubric_in_context()` | Evaluate rubric quality relative to a specific task prompt |

## Quick Example

```python
from autorubric import LLMConfig, Rubric
from autorubric.meta import evaluate_rubric_standalone, evaluate_rubric_in_context

llm_config = LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)
rubric = Rubric.from_file("my_rubric.json")

# Standalone evaluation with terminal output. result.score is `float | None`
# (None if the meta-rubric grade failed); guard before formatting.
result = await evaluate_rubric_standalone(rubric, llm_config, display="stdout")
print(
    f"Rubric quality score: {result.score:.2f}"
    if result.score is not None
    else "Rubric quality score: n/a (grade failed)"
)

# In-context evaluation with HTML report
result = await evaluate_rubric_in_context(
    rubric,
    task_prompt="Write a comprehensive summary of the research paper.",
    llm_config=llm_config,
    display="html",
    output_html_path="rubric_evaluation.html",
)
```

## Display Modes

Both evaluation functions accept a `display` parameter:

| Value | Behavior |
|-------|----------|
| `None` | No display output (default) |
| `"stdout"` | Rich terminal output with colored tables |
| `"html"` | Generates styled HTML report (requires `output_html_path`) |

### Terminal Output

```python
result = await evaluate_rubric_standalone(rubric, llm_config, display="stdout")
```

Produces formatted output with:

- Score summary panel
- Per-section criterion tables with color-coded statuses
- Issues found section with full feedback

### HTML Output

```python
result = await evaluate_rubric_in_context(
    rubric, task_prompt, llm_config,
    display="html",
    output_html_path="report.html"
)
```

Generates a self-contained HTML file with:

- Dark theme styling
- Visual grouping by rubric sections
- Color-coded status badges (green for MET/CLEAR, red for UNMET/DETECTED)
- Full issue descriptions

## Meta-Rubric Criteria

### Standalone Meta-Rubric

Evaluates intrinsic rubric quality across these sections:

| Section | Focus |
|---------|-------|
| **Clarity & Precision** | Clear requirements, specific language, unidimensional criteria |
| **Structure & Design** | Appropriate count, balanced weights, orthogonal criteria |
| **LLM-Friendliness** | Independent verification, objective assessment |
| **Anti-Patterns** | Double-barreled criteria, vague wording, circular definitions, excessive overlap |

### In-Context Meta-Rubric

Includes all standalone criteria plus:

| Section | Focus |
|---------|-------|
| **Construct Alignment** | Task alignment, coverage of key aspects, appropriate emphasis |
| **Discriminative Power** | Distinguishes quality levels, avoids trivial criteria |
| **Anti-Patterns (In-Context)** | Irrelevant criteria, missing critical aspects |

## Accessing Meta-Rubrics Directly

You can load the meta-rubrics as `Rubric` objects for inspection or custom use:

```python
from autorubric.meta import get_standalone_meta_rubric, get_in_context_meta_rubric

standalone = get_standalone_meta_rubric()
print(f"Standalone meta-rubric: {len(standalone.rubric)} criteria")

in_context = get_in_context_meta_rubric()
print(f"In-context meta-rubric: {len(in_context.rubric)} criteria")
```

---

## evaluate_rubric_standalone

Evaluate a rubric's quality in isolation.

::: autorubric.meta.evaluate_rubric_standalone
    options:
      show_source: false
      members_order: source

---

## evaluate_rubric_in_context

Evaluate a rubric's quality relative to a task prompt.

::: autorubric.meta.evaluate_rubric_in_context
    options:
      show_source: false
      members_order: source

---

## get_standalone_meta_rubric

Load the standalone meta-rubric.

::: autorubric.meta.get_standalone_meta_rubric
    options:
      show_source: false

---

## get_in_context_meta_rubric

Load the in-context meta-rubric.

::: autorubric.meta.get_in_context_meta_rubric
    options:
      show_source: false

---

## Rubric Improvement

For the iterative rubric improvement API (`improve_rubric`, `ImprovementRunner`, building blocks, etc.), see the dedicated **[Rubric Improvement](meta-improvement.md)** page.
