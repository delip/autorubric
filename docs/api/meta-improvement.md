# Meta-Rubric Improvement

Iterative rubric improvement engine that optimizes for meta-rubric quality (validity) and validation reliability.

## Overview

The improvement API provides a two-tier interface for iteratively refining rubrics:

| Level | Entry Point | Use Case |
|-------|-------------|----------|
| **Convenience** | `improve_rubric()` | Quick start with keyword arguments |
| **Full Control** | `ImprovementRunner` | Custom convergence, callbacks, fine-grained config |

Two **validation modes** are supported:

| Mode | Trigger | Metric |
|------|---------|--------|
| **Ground-truth** | `validation_data` items have `ground_truth` | Spearman rank correlation (ρ) between rubric scores and expected scores |
| **Multi-judge** | `validation_data` items lack `ground_truth` + `eval_llm` is `list[JudgeSpec]` | Mean inter-judge agreement |

A **Pareto constraint** rejects revisions that improve quality but decrease validation reliability.

## Quick Example

### Using `improve_rubric()`

```python
import asyncio
from autorubric import LLMConfig, Rubric
from autorubric.dataset import RubricDataset
from autorubric.meta import improve_rubric

async def main():
    eval_llm = LLMConfig(model="openai/gpt-4.1", temperature=0.0)
    revision_llm = LLMConfig(model="openai/gpt-4.1", temperature=0.3)

    rubric = Rubric.from_file("my_rubric.json")
    validation_data = RubricDataset.from_file("validation_data.json")

    result = await improve_rubric(
        rubric,
        "Your task prompt here",
        eval_llm=eval_llm,
        revision_llm=revision_llm,
        validation_data=validation_data,
        artifacts_dir="experiments/my_improvement",
        display="stdout",
    )

    print(f"Quality: {result.iterations[-1].quality_score:.0%}")
    print(f"Convergence: {result.convergence_reason}")
    result.final_rubric.to_file("improved_rubric.json")

asyncio.run(main())
```

### Using `ImprovementRunner`

```python
from autorubric.meta import ImprovementRunner, ImprovementConfig

config = ImprovementConfig(
    eval_llm=eval_llm,
    revision_llm=revision_llm,
    validation_data=validation_data,
    max_iterations=15,
    min_quality_score=0.95,
    show_progress=True,
)
runner = ImprovementRunner(rubric, "Your task prompt", config=config)
result = await runner.run()
```

## Validation Modes

### Ground-Truth Mode

When `validation_data` items have `ground_truth` verdicts, the loop computes expected scores from the rubric weights and measures Spearman ρ against the actual graded scores.

```python
# Items with ground_truth → ground-truth mode
dataset = RubricDataset.from_file("labeled_data.json")
result = await improve_rubric(
    rubric, prompt,
    eval_llm=LLMConfig(model="openai/gpt-4.1"),
    revision_llm=LLMConfig(model="openai/gpt-4.1"),
    validation_data=dataset,
)
```

### Multi-Judge Mode

When items lack `ground_truth`, provide an ensemble of judges to measure inter-judge agreement:

```python
from autorubric.graders import JudgeSpec

judges = [
    JudgeSpec(LLMConfig(model="openai/gpt-4.1"), "gpt"),
    JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
]
result = await improve_rubric(
    rubric, prompt,
    eval_llm=judges,
    revision_llm=LLMConfig(model="openai/gpt-4.1"),
    validation_data=dataset,  # items without ground_truth
)
```

## Artifact Persistence

When `save_artifacts=True` and `artifacts_dir` is set, the improvement loop writes:

| File | Contents |
|------|----------|
| `rubric-iter-{NN}.json` | Criteria array per iteration |
| `eval-iter-{NN}.html` | Meta-rubric eval report (always generated) |
| `iter-{NN}.json` | Rich per-iteration JSON (quality report, issues, validation samples, revision prompts/response) |
| `improvement_report.html` | Consolidated report (always generated) |
| `summary.json` | Full run metadata, config snapshot, and per-iteration summary |

## Custom Convergence

Replace the built-in convergence logic with a custom function:

```python
from autorubric.meta import ConvergenceFn, IterationResult

def my_convergence(current: IterationResult, history: list[IterationResult]) -> str | None:
    if current.quality_score > 0.9 and len(current.issues) == 0:
        return "perfect quality with no issues"
    if len(history) >= 5:
        return "max iterations reached"
    return None  # continue

config = ImprovementConfig(
    eval_llm=eval_llm,
    revision_llm=revision_llm,
    convergence_fn=my_convergence,
)
```

---

## improve_rubric

Convenience wrapper for iterative rubric improvement.

::: autorubric.meta.improve_rubric
    options:
      show_source: false
      members_order: source

---

## ImprovementRunner

Full-control runner class following the `EvalRunner` pattern.

::: autorubric.meta.ImprovementRunner
    options:
      show_source: false
      members_order: source

---

## ImprovementConfig

Configuration for the rubric improvement process.

::: autorubric.meta.ImprovementConfig
    options:
      show_source: false
      members_order: source

---

## ImprovementResult

Final result from the rubric improvement process.

::: autorubric.meta.ImprovementResult
    options:
      show_source: false
      members_order: source

---

## IterationResult

Result from a single improvement iteration.

::: autorubric.meta.IterationResult
    options:
      show_source: false
      members_order: source

---

## IssueDetail

A single issue identified in a rubric by meta-rubric evaluation.

::: autorubric.meta.IssueDetail
    options:
      show_source: false
      members_order: source

---

## ConvergenceFn

Custom convergence function type alias.

::: autorubric.meta.ConvergenceFn

---

## ImprovementProgressDisplay

Rich-based progress display for the improvement loop.

::: autorubric.meta.ImprovementProgressDisplay
    options:
      show_source: false
      members_order: source

---

## Building Blocks

These functions can be used independently to compose custom improvement loops.

### extract_issues

Extract actionable issues from a meta-rubric evaluation report.

::: autorubric.meta.extract_issues
    options:
      show_source: false

---

### diff_issues

Track fixed and introduced issues between iterations.

::: autorubric.meta.diff_issues
    options:
      show_source: false

---

### format_issues_for_prompt

Format issues into text for the revision prompt.

::: autorubric.meta.format_issues_for_prompt
    options:
      show_source: false

---

### format_agreement_for_prompt

Format per-criterion agreement data as a self-contained prompt section.

::: autorubric.meta.format_agreement_for_prompt
    options:
      show_source: false

---

### format_ground_truth_for_prompt

Format ground-truth validation results as a prompt section.

::: autorubric.meta.format_ground_truth_for_prompt
    options:
      show_source: false

---

### build_revision_history

Format recent iteration history for the revision prompt.

::: autorubric.meta.build_revision_history
    options:
      show_source: false

---

### validate_agreement

Test inter-judge agreement on validation data.

::: autorubric.meta.validate_agreement
    options:
      show_source: false

---

### validate_ground_truth

Grade validation items and compute Spearman ρ against expected scores.

::: autorubric.meta.validate_ground_truth
    options:
      show_source: false

---

### compute_expected_scores

Compute expected scores from ground-truth verdicts and rubric weights.

::: autorubric.meta.compute_expected_scores
    options:
      show_source: false

---

### pareto_accept

Check revision acceptance under the Pareto constraint.

::: autorubric.meta.pareto_accept
    options:
      show_source: false

---

### revise_rubric

Revise a rubric via LLM based on identified issues.

::: autorubric.meta.revise_rubric
    options:
      show_source: false
