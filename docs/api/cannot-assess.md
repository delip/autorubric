# CANNOT_ASSESS Handling

Configuration for handling criteria that cannot be assessed due to insufficient evidence.

## Overview

When a judge lacks evidence to determine whether a criterion is met, it may return `CANNOT_ASSESS` instead of `MET` or `UNMET`. This module provides configuration options for how these uncertain verdicts affect scoring.

!!! tip "Research Background"

    A recurring recommendation across LLM-as-a-judge research is to include an explicit "cannot assess / insufficient information" option. Forcing binary verdicts when evidence is insufficient leads to unreliable evaluations. Min et al. (2023) demonstrate in FActScore that atomic fact verification must explicitly handle cases where claims cannot be verified.

## Usage

```python
from autorubric import CannotAssessConfig, CannotAssessStrategy, LLMConfig
from autorubric.graders import CriterionGrader

# Default: skip unassessable criteria (adjust denominator)
grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
)

# Be conservative: treat cannot-assess as failure
grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
    cannot_assess_config=CannotAssessConfig(strategy=CannotAssessStrategy.FAIL),
)

# Give partial credit (30%)
grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
    cannot_assess_config=CannotAssessConfig(
        strategy=CannotAssessStrategy.PARTIAL,
        partial_credit=0.3
    ),
)
```

## Strategies

| Strategy | Description |
|----------|-------------|
| `SKIP` | Exclude from scoring (adjust denominator) - default |
| `ZERO` | Treat as 0 contribution (same as UNMET) |
| `PARTIAL` | Treat as partial credit (configurable fraction) |
| `FAIL` | Treat as worst case (UNMET for positive, MET for negative weights) |

---

## CannotAssessConfig

::: autorubric.CannotAssessConfig
    options:
      show_source: true
      members_order: source

---

## CannotAssessStrategy

::: autorubric.CannotAssessStrategy
    options:
      show_source: true

---

## CannotAssessMode

Used in metrics computation to specify how CANNOT_ASSESS verdicts should be handled when comparing against ground truth.

::: autorubric.CannotAssessMode
    options:
      show_source: true

---

## References

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., and Hajishirzi, H. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 12076–12100.
