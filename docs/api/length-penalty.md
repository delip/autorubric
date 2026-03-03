# Length Penalty

Control verbosity bias by penalizing excessively long responses.

## Overview

LLM judges often prefer longer answers, a phenomenon known as verbosity bias. The length penalty feature provides a configurable mechanism to penalize excessively verbose outputs without requiring changes to the rubric itself.

!!! tip "Research Background"

    Dubois et al. (2024) document verbosity bias extensively in their length-controlled AlpacaEval work. Length penalty helps reduce verbosity-driven score inflation by adding conciseness as an implicit scoring dimension.

## Quick Example

```python
from autorubric import Rubric, LLMConfig, LengthPenalty
from autorubric.graders import CriterionGrader

grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
    length_penalty=LengthPenalty(
        free_budget=6000,        # No penalty below this count
        max_cap=8000,            # Maximum penalty at/above this count
        penalty_at_cap=0.5,      # Max penalty to subtract from score
        exponent=1.6,            # Curve steepness
        penalty_type="ALL",      # "ALL", "OUTPUT_ONLY", "THINKING_ONLY"
    ),
)

result = await rubric.grade(to_grade=response, grader=grader)
```

## Penalty Formula

```
if count <= free_budget:
    penalty = 0
elif count >= max_cap:
    penalty = penalty_at_cap
else:
    frac = (count - free_budget) / (max_cap - free_budget)
    penalty = penalty_at_cap * (frac ** exponent)

final_score = max(0.0, base_score - penalty)
```

## Custom Count Functions

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
    length_penalty=LengthPenalty(
        free_budget=8000,
        max_cap=10000,
        count_fn=lambda t: len(tokenizer.encode(t)),  # Token count
    ),
)
```

## Thinking/Output Separation

For models with separate thinking and output sections:

```python
# Dict format
await rubric.grade(
    to_grade={
        "thinking": "Let me reason through this...",
        "output": "The final answer is 42"
    },
    grader=grader
)

# String with markers
await rubric.grade(
    to_grade="<thinking>My reasoning...</thinking><output>Final answer</output>",
    grader=grader
)

# Penalty type selection
penalty = LengthPenalty(
    free_budget=8000,
    max_cap=10000,
    penalty_at_cap=0.5,
    penalty_type="OUTPUT_ONLY"  # Only count output, not thinking
)
```

## Training/RL Use Cases

For reinforcement learning, use unnormalized scores with absolute penalties:

```python
grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
    normalize=False,  # Raw weighted sums
    length_penalty=LengthPenalty(
        free_budget=8000,
        max_cap=10000,
        penalty_at_cap=50.0,  # Absolute penalty
        exponent=1.6,
        count_fn=lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    ),
)
```

---

## LengthPenalty

Configuration for length-based score penalty.

::: autorubric.LengthPenalty
    options:
      show_source: true
      members_order: source

---

## compute_length_penalty

Compute the penalty value for a given text.

::: autorubric.compute_length_penalty
    options:
      show_source: true

---

## PenaltyType

Enum for which sections to count for length penalty.

::: autorubric.PenaltyType
    options:
      show_source: true

---

## CountFn

Type alias for custom counting functions.

::: autorubric.CountFn
    options:
      show_source: true

---

## word_count

Default word counting function.

::: autorubric.word_count
    options:
      show_source: true

---

## References

Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B. (2024). Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. arXiv:2404.04475.
