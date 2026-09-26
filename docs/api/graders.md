# Graders

Grader implementations for evaluating responses against rubrics.

## Overview

Graders evaluate responses against rubrics and return structured reports. The main implementation is `CriterionGrader`, which supports single LLM, ensemble, and few-shot modes. All combinations work orthogonally.

## Quick Example

```python
from autorubric import LLMConfig, FewShotConfig
from autorubric.graders import CriterionGrader, JudgeSpec, Grader

# Single LLM mode
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
)

# With custom system prompt
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
    system_prompt="You are evaluating technical documentation...",
)

# Ensemble mode
grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini", weight=1.0),
        JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude", weight=1.2),
    ],
    aggregation="weighted",
)

# Single LLM + few-shot
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
    training_data=train_data,
    few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=True),
)

# Grade
result = await rubric.grade(to_grade=response, grader=grader)
```

`judge_model_config` sets up a single judge and `judges` sets up an ensemble. Pass exactly one of them.

Either accepts a `DecisionModelConfig` as well as an `LLMConfig`: a decision-model judge grades each item with one request for the whole rubric. `escalation=EscalationConfig(...)` makes a grader with one decision-model judge a confidence cascade that escalates uncertain criteria to LLM judges. See [Decision Models](decision-models.md) for the API, and the [LLM Judges](../cookbook/llm-judges.md) and [Decision-Model Judges](../cookbook/decision-models.md) chapters for guided introductions to the two judge kinds.

!!! note "`llm_config` is deprecated"
    `llm_config` is a deprecated alias of `judge_model_config`. It builds the same judge but emits a
    `DeprecationWarning`. To migrate, rename the keyword; nothing else changes. Passing both raises
    `ValueError`. `llm_config` will not be removed before the next major version.

## Grading Options

```python
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),

    # Score normalization
    normalize=True,          # True: 0-1 range, False: raw weighted sum

    # CANNOT_ASSESS handling
    cannot_assess_config=CannotAssessConfig(strategy=CannotAssessStrategy.SKIP),

    # Length penalty
    length_penalty=LengthPenalty(free_budget=6000, max_cap=8000),

    # Position bias mitigation (for multi-choice)
    shuffle_options=True,    # Default: enabled

    # Multi-choice abstain channel
    auto_na_option=True,     # Default: True — guarantees every multi-choice criterion a
                             # first-class NA/abstain option (auto-injected if absent).
                             # Set False for forced-choice (no auto NA option).

    # Reproducibility — pins all non-LLM randomness (shuffles, few-shot selection)
    seed=42,                 # Default: auto-generated
)
```

With `auto_na_option=True`, an auto-injected NA option is appended at the end (highest index), so existing option indices are preserved; an author-supplied NA option is never stripped.

---

## CriterionGrader

Main grader with support for single LLM, ensemble, and few-shot modes.

::: autorubric.graders.CriterionGrader
    options:
      show_source: true
      members_order: source

---

## Grader

Abstract base class for grader implementations.

`Grader` is generic in the report type it returns. `CriterionGrader` is a
`Grader[EnsembleEvaluationReport]`, so a type checker types `await rubric.grade(..., grader=grader)`
as `EnsembleEvaluationReport`. A custom grader declares its own, for example
`class MyGrader(Grader[EvaluationReport])`; one that names no type still works.

::: autorubric.graders.Grader
    options:
      show_source: true
      members_order: source

---

## JudgeSpec

Configuration for a single judge in an ensemble.

::: autorubric.graders.JudgeSpec
    options:
      show_source: true
      members_order: source
