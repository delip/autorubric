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
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
)

# Be conservative: treat cannot-assess as failure
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
    cannot_assess_config=CannotAssessConfig(strategy=CannotAssessStrategy.FAIL),
)

# Give partial credit (30%)
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
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

The strategies apply to multi-choice NA options too. Under `SKIP`, an item every criterion of which abstains has nothing left to score: its `score` and `raw_score` are `None`, not 0.0. This is not a failed grade, so its `error` is `None` (unless every judgment failed, below). Zero-weight criteria still count as scored. `ZERO`, `PARTIAL` and `FAIL` keep abstentions in the score, so abstentions never leave them without one. Ground truth follows the same rule: `Rubric.compute_score` and `RubricDataset.compute_weighted_score` return `None` when every label abstains under `SKIP`.

A judge call that fails with an API or parse error also yields `CANNOT_ASSESS` (or the NA option), and one that fails for another reason yields the worst case. These stand-in verdicts are not judgments, though. When the judgment of every criterion of an item failed, the item has no score under any strategy, and its report's `error` begins `Every criterion's judgment failed:`.

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
