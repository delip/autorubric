# Few-Shot

Calibrate LLM judges with labeled examples for improved grading consistency.

## Overview

Few-shot learning provides the judge with graded examples before evaluation, helping calibrate its understanding of the rubric criteria. This is particularly effective for subjective criteria or domain-specific evaluation.

!!! tip "Research Background"

    Casabianca et al. (2025) and Ashktorab et al. (2025) recommend graded exemplars ("gold anchors") including negative examples of common failure modes for both human and LLM judge calibration. Few-shot examples reduce rater error and improve agreement metrics.

## Quick Example

```python
from autorubric import LLMConfig, FewShotConfig, RubricDataset
from autorubric.graders import CriterionGrader

# Load dataset with ground truth
dataset = RubricDataset.from_file("labeled_data.json")

# Split into training (for few-shot) and test
train_data, test_data = dataset.split_train_test(n_train=100, stratify=True, seed=42)

# Configure few-shot grader
grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini"),
    training_data=train_data,
    few_shot_config=FewShotConfig(
        n_examples=3,
        balance_verdicts=True,  # Balance examples across label classes (verdicts for binary, option indices for multi-choice)
        include_reason=True,
        seed=42,
    ),
)

# Grade with few-shot calibration
result = await rubric.grade(to_grade=response, grader=grader)
```

## Ensemble + Few-Shot

Few-shot works orthogonally with ensemble mode. All judges receive the same examples:

```python
from autorubric.graders import JudgeSpec

grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini"),
        JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
    ],
    aggregation="majority",
    training_data=train_data,
    few_shot_config=FewShotConfig(n_examples=3),
)
```

---

## FewShotConfig

Configuration for few-shot example selection.

::: autorubric.FewShotConfig
    options:
      show_source: true
      members_order: source

---

## FewShotExample

A single few-shot example with submission and ground truth verdict.

::: autorubric.FewShotExample
    options:
      show_source: true
      members_order: source

---

## References

Ashktorab, Z., Daly, E. M., Miehling, E., Geyer, W., Santillán Cooper, M., Pedapati, T., Desmond, M., Pan, Q., and Do, H. J. (2025). EvalAssist: A Human-Centered Tool for LLM-as-a-Judge. arXiv:2507.02186.

Casabianca, J., McCaffrey, D. F., Johnson, M. S., Alper, N., and Zubenko, V. (2025). Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications. arXiv:2501.02334.
