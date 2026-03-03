# Dataset

Dataset management classes for organizing evaluation data with optional ground truth labels.

## Overview

The `RubricDataset` class provides structured storage for evaluation datasets, including submissions, optional ground truth verdicts, per-item rubrics, and reference submissions. Datasets can be serialized to JSON/YAML for sharing and reproducibility.

## Quick Example

```python
from autorubric import Rubric, Criterion, CriterionVerdict, DataItem, RubricDataset

# Create a rubric
rubric = Rubric([
    Criterion(name="accuracy", weight=10.0, requirement="Factually correct"),
    Criterion(name="clarity", weight=5.0, requirement="Clear and concise"),
])

# Create a dataset (prompt is now optional)
dataset = RubricDataset(
    name="photosynthesis-eval",
    prompt="Explain photosynthesis",  # Optional global prompt
    rubric=rubric,
)

# Add items with ground truth
dataset.add_item(
    submission="Photosynthesis is the process by which plants convert sunlight...",
    description="Good response",
    ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET]
)

# Serialize
dataset.to_file("dataset.json")

# Load
loaded = RubricDataset.from_file("dataset.json")
```

## Per-Item Rubrics

For datasets where each item requires a unique rubric (e.g., question-specific evaluation):

```python
item = DataItem(
    submission="Answer to question 1...",
    description="Q1",
    rubric=Rubric([
        Criterion(weight=1.0, requirement="Correct answer for Q1"),
    ])
)

dataset = RubricDataset(
    prompt="Answer the question",
    rubric=None,  # No global rubric
    items=[item],
)

# Get effective rubric for an item
rubric = dataset.get_item_rubric(0)  # Returns item's rubric
```

## Per-Item Prompts

Different items can have different prompts:

```python
# Create items with individual prompts
item1 = DataItem(
    submission="The answer is 42.",
    description="Math problem",
    prompt="Evaluate this answer to: What is 6 x 7?"
)

item2 = DataItem(
    submission="Analysis of Hamlet's soliloquy...",
    description="Literary analysis"
    # No prompt specified, will use global prompt
)

# Create dataset with optional global prompt
dataset = RubricDataset(
    prompt="Evaluate this response",  # Optional global prompt (fallback)
    rubric=rubric,
    items=[item1, item2],
)

# Get effective prompt for an item
prompt = dataset.get_item_prompt(0)  # Returns item1's prompt
prompt = dataset.get_item_prompt(1)  # Returns global prompt (item2 has none)

# Raises ValueError if neither item nor dataset has a prompt
try:
    prompt = dataset.get_item_prompt(0)
except ValueError:
    print("No prompt available for this item")
```

Use in grading:

```python
for i, item in enumerate(dataset):
    result = await rubric.grade(
        to_grade=item.submission,
        grader=grader,
        query=dataset.get_item_prompt(i),
    )
```

## Reference Submissions

Provide exemplar responses for judge calibration:

```python
# Global reference for all items
dataset = RubricDataset(
    prompt="Explain photosynthesis",
    rubric=rubric,
    reference_submission="Detailed explanation of photosynthesis...",
)

# Per-item reference (overrides global)
dataset.add_item(
    submission="Student answer...",
    description="Q1",
    reference_submission="Custom reference for this item",
)

# Get effective reference
ref = dataset.get_item_reference_submission(0)
```

## Train/Test Split

```python
train_data, test_data = dataset.split_train_test(
    n_train=100,
    stratify=True,  # Balance by ground truth verdicts
    seed=42,
)
```

---

## DataItem

A single item in an evaluation dataset.

::: autorubric.DataItem
    options:
      show_source: true
      members_order: source

---

## RubricDataset

Container for evaluation datasets with optional ground truth.

::: autorubric.RubricDataset
    options:
      show_source: true
      members_order: source
