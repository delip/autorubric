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
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
)

# With custom system prompt
grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
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
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
    training_data=train_data,
    few_shot_config=FewShotConfig(n_examples=3, balance_verdicts=True),
)

# Grade
result = await rubric.grade(to_grade=response, grader=grader)
```

## Grading Options

```python
grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),

    # Score normalization
    normalize=True,          # True: 0-1 range, False: raw weighted sum

    # CANNOT_ASSESS handling
    cannot_assess_config=CannotAssessConfig(strategy=CannotAssessStrategy.SKIP),

    # Length penalty
    length_penalty=LengthPenalty(free_budget=6000, max_cap=8000),

    # Position bias mitigation (for multi-choice)
    shuffle_options=True,    # Default: enabled
)
```

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
