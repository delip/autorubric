# Per-Item Rubrics for Diverse Tasks

Use different evaluation criteria for different items in the same dataset.

## The Scenario

You're evaluating coding interview responses. Each problem has unique requirements: a sorting problem needs different criteria than a system design question. Rather than creating separate datasets, you want one dataset with problem-specific rubrics while maintaining a global baseline.

## What You'll Learn

- Setting per-item rubrics with `DataItem.rubric`
- Combining global and item-level rubrics
- Using `get_item_rubric()` to resolve effective rubrics
- Per-item reference submissions for comparison
- Mixed rubric types (binary + multi-choice) per item

## The Solution

### Step 1: Create a Dataset with Global Rubric

Start with baseline criteria that apply to all items:

```python
from autorubric import Rubric, RubricDataset, DataItem

# Global rubric with baseline criteria
global_rubric = Rubric.from_dict([
    {
        "name": "correct_solution",
        "weight": 15.0,
        "requirement": "Solution produces correct output for all test cases"
    },
    {
        "name": "code_clarity",
        "weight": 8.0,
        "requirement": "Code is readable with meaningful variable names"
    },
    {
        "name": "handles_edge_cases",
        "weight": 10.0,
        "requirement": "Handles edge cases (empty input, nulls, boundaries)"
    }
])

dataset = RubricDataset(
    prompt="Evaluate this coding interview solution.",
    rubric=global_rubric,
    name="coding-interviews"
)
```

The resolution logic follows a simple fallback chain:

| Scenario | Rubric Source | Example |
|----------|--------------|---------|
| Item has per-item rubric | Item's own `DataItem.rubric` | Sorting problem with complexity criteria |
| Item has no per-item rubric, dataset has global rubric | Dataset's `RubricDataset.rubric` | Simple string reversal using baseline criteria |
| Neither item nor dataset has a rubric | Error raised at evaluation time | Misconfigured dataset with no rubric at any level |

### Step 2: Add Items with Problem-Specific Rubrics

Override rubrics for items that need specialized criteria:

```python
# Problem 1: Sorting algorithm
# Uses global rubric + problem-specific criteria
sorting_rubric = Rubric.from_dict([
    {
        "name": "correct_solution",
        "weight": 15.0,
        "requirement": "Solution correctly sorts the array in ascending order"
    },
    {
        "name": "optimal_complexity",
        "weight": 12.0,
        "requirement": "Achieves O(n log n) time complexity or better"
    },
    {
        "name": "in_place",
        "weight": 5.0,
        "requirement": "Sorts in-place without using extra O(n) space"
    },
    {
        "name": "handles_edge_cases",
        "weight": 8.0,
        "requirement": "Handles empty arrays and single-element arrays"
    }
])

dataset.add_item(
    submission="""
def sort_array(nums):
    # Quick sort implementation
    if len(nums) <= 1:
        return nums
    pivot = nums[len(nums) // 2]
    left = [x for x in nums if x < pivot]
    middle = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]
    return sort_array(left) + middle + sort_array(right)
""",
    description="Sorting problem - quick sort solution",
    rubric=sorting_rubric,  # Problem-specific rubric
)

# Problem 2: Two Sum
# Uses different criteria focused on data structures
two_sum_rubric = Rubric.from_dict([
    {
        "name": "correct_solution",
        "weight": 15.0,
        "requirement": "Returns correct indices that sum to target"
    },
    {
        "name": "optimal_complexity",
        "weight": 12.0,
        "requirement": "Achieves O(n) time using hash map"
    },
    {
        "name": "single_pass",
        "weight": 5.0,
        "requirement": "Solves in single pass through the array"
    },
    {
        "name": "code_clarity",
        "weight": 6.0,
        "requirement": "Code is readable with meaningful variable names"
    }
])

dataset.add_item(
    submission="""
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
""",
    description="Two Sum problem - hash map solution",
    rubric=two_sum_rubric,
)
```

!!! tip "When to Use Per-Item Rubrics"
    Per-item rubrics are most useful when evaluation items require fundamentally different
    criteria, not just different prompts. A sorting problem and a system design question
    need distinct criteria (time complexity vs. architecture quality), so a single global
    rubric cannot adequately cover both. If items differ only in prompt wording but share
    the same quality dimensions, a global rubric with per-item prompts is simpler.

### Step 3: Items Using Global Rubric

Some items can use the global rubric without customization:

```python
# Problem 3: Uses global rubric
dataset.add_item(
    submission="""
def reverse_string(s):
    return s[::-1]
""",
    description="String reversal - simple problem",
    # No rubric specified = uses global rubric
)
```

### Step 4: Use Reference Submissions

Provide ideal solutions for comparison:

```python
# Problem with reference submission
dataset.add_item(
    submission="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
    description="Fibonacci - recursive solution",
    reference_submission="""
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",  # Optimal iterative solution for comparison
)
```

!!! note "Reference Submissions"
    A reference submission is an ideal or expected solution attached to a data item. When
    provided, the grader can compare the candidate submission against the reference to make
    more informed judgments. Reference submissions are optional and do not affect rubric
    resolution -- they serve purely as additional context for the LLM judge.

### Step 5: Resolve Effective Rubrics

Access the rubric that applies to each item:

```python
for idx, item in enumerate(dataset):
    # get_item_rubric returns item's rubric if set, else global rubric
    effective_rubric = dataset.get_item_rubric(idx)

    print(f"\nItem {idx}: {item.description}")
    print(f"  Criteria: {[c.name for c in effective_rubric.criteria]}")
    print(f"  Has custom rubric: {item.rubric is not None}")

    # Get reference submission if available
    reference = dataset.get_item_reference_submission(idx)
    if reference:
        print(f"  Has reference: Yes")
```

### Step 6: Evaluate with Per-Item Rubrics

The grader automatically uses the effective rubric for each item:

```python
from autorubric import LLMConfig, evaluate
from autorubric.graders import CriterionGrader

grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini")
)

# evaluate() automatically uses the correct rubric per item
result = await evaluate(dataset, grader, show_progress=True)

# Results reflect per-item criteria
for item_result in result.item_results:
    print(f"\n{item_result.item.description}")
    # report.score is `float | None` (None if that item's grade failed).
    score = item_result.report.score
    print(f"  Score: {score:.2f}" if score is not None else "  Score: n/a")
    print(f"  Criteria evaluated: {len(item_result.report.report)}")
```

### Step 7: Mix Binary and Multi-Choice Criteria

Different items can use different criterion types:

```python
from autorubric import Criterion, CriterionOption

# System design problem with multi-choice criteria
system_design_rubric = Rubric([
    Criterion(
        name="architecture_quality",
        weight=15.0,
        requirement="Quality of system architecture design",
        scale_type="ordinal",
        options=[
            CriterionOption(label="Poor - Major flaws", value=0.0),
            CriterionOption(label="Basic - Works but not scalable", value=0.4),
            CriterionOption(label="Good - Solid design", value=0.7),
            CriterionOption(label="Excellent - Production ready", value=1.0),
        ]
    ),
    Criterion(
        name="handles_scale",
        weight=10.0,
        requirement="Design handles 10x traffic increase"
    ),
    Criterion(
        name="failure_handling",
        weight=8.0,
        requirement="Addresses failure modes and recovery"
    ),
])

dataset.add_item(
    submission="[System design answer...]",
    description="Design a URL shortener",
    rubric=system_design_rubric,
)
```

!!! warning "Rubric Compatibility with Ground-Truth Verdicts"
    When using `ground_truth` verdicts on a `DataItem`, those verdicts must correspond to
    the criteria in the item's per-item rubric, not the global rubric. A mismatch between
    verdict keys and rubric criteria will produce incorrect metrics. Always verify that
    ground-truth labels align with the effective rubric for each item.

| Problem | Custom Rubric? | Criteria Count |
|---------|---------------|----------------|
| Sorting algorithm | Yes | 4 (correctness, complexity, space, edge cases) |
| Two Sum | Yes | 4 (correctness, complexity, single pass, clarity) |
| String reversal | No (global) | 3 (correctness, clarity, edge cases) |
| Fibonacci | No (global) | 3 (correctness, clarity, edge cases) |
| System design | Yes | 3 (architecture, scale, failure handling) |

## Key Takeaways

- **Per-item rubrics** allow heterogeneous evaluation in one dataset
- **`DataItem.rubric`** overrides the global rubric for that item
- **`get_item_rubric(idx)`** resolves the effective rubric
- **Per-item prompts** follow the same pattern: `DataItem.prompt` overrides the global prompt
- **`get_item_prompt(idx)`** resolves the effective prompt (returns item prompt if set, else dataset prompt)
- **Reference submissions** provide comparison baselines
- **Mix criterion types** (binary/multi-choice) per item as needed
- **Global rubric** provides baseline for items without custom rubrics

## Going Further

- [Managing Datasets](managing-datasets.md) - Dataset fundamentals
- [Multi-Choice Rubrics](multi-choice-rubrics.md) - Advanced criterion types
- [API Reference: Dataset](../api/dataset.md) - Full DataItem documentation

---

## Appendix: Complete Code

```python
"""Per-Item Rubrics - Coding Interview Evaluation"""

import asyncio
from autorubric import (
    Rubric, Criterion, CriterionOption,
    RubricDataset, LLMConfig, evaluate
)
from autorubric.graders import CriterionGrader


def create_coding_interview_dataset() -> RubricDataset:
    """Create a coding interview dataset with per-item rubrics."""

    # Global baseline rubric
    global_rubric = Rubric.from_dict([
        {
            "name": "correct_solution",
            "weight": 15.0,
            "requirement": "Solution produces correct output for standard inputs"
        },
        {
            "name": "code_clarity",
            "weight": 8.0,
            "requirement": "Code is readable with meaningful variable names"
        },
        {
            "name": "handles_edge_cases",
            "weight": 10.0,
            "requirement": "Handles edge cases appropriately"
        }
    ])

    dataset = RubricDataset(
        prompt="Evaluate this coding interview solution.",
        rubric=global_rubric,
        name="coding-interviews-v1"
    )

    # Problem 1: Sorting (custom rubric)
    sorting_rubric = Rubric.from_dict([
        {
            "name": "correct_sorting",
            "weight": 15.0,
            "requirement": "Correctly sorts array in ascending order"
        },
        {
            "name": "time_complexity",
            "weight": 12.0,
            "requirement": "Achieves O(n log n) time complexity"
        },
        {
            "name": "space_efficiency",
            "weight": 6.0,
            "requirement": "Uses O(1) extra space (in-place sorting)"
        },
        {
            "name": "handles_empty",
            "weight": 5.0,
            "requirement": "Handles empty and single-element arrays"
        }
    ])

    dataset.add_item(
        submission="""
def sort_array(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[len(nums) // 2]
    left = [x for x in nums if x < pivot]
    middle = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]
    return sort_array(left) + middle + sort_array(right)
""",
        description="Sorting - Quick sort (not in-place)",
        rubric=sorting_rubric,
    )

    # Problem 2: Two Sum (custom rubric)
    two_sum_rubric = Rubric.from_dict([
        {
            "name": "correct_indices",
            "weight": 15.0,
            "requirement": "Returns correct pair of indices summing to target"
        },
        {
            "name": "optimal_time",
            "weight": 12.0,
            "requirement": "Achieves O(n) time using hash-based approach"
        },
        {
            "name": "handles_no_solution",
            "weight": 6.0,
            "requirement": "Handles case when no valid pair exists"
        }
    ])

    dataset.add_item(
        submission="""
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
""",
        description="Two Sum - Hash map solution",
        rubric=two_sum_rubric,
    )

    # Problem 3: Fibonacci with reference (global rubric)
    dataset.add_item(
        submission="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        description="Fibonacci - Naive recursive",
        reference_submission="""
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
        # Uses global rubric
    )

    # Problem 4: System design with multi-choice (custom)
    design_rubric = Rubric([
        Criterion(
            name="architecture_quality",
            weight=15.0,
            requirement="Overall quality of system architecture",
            scale_type="ordinal",
            options=[
                CriterionOption(label="Poor - Missing key components", value=0.0),
                CriterionOption(label="Basic - Works but issues", value=0.4),
                CriterionOption(label="Good - Solid design", value=0.7),
                CriterionOption(label="Excellent - Production ready", value=1.0),
            ]
        ),
        Criterion(
            name="scalability",
            weight=10.0,
            requirement="Design scales to 10x current load"
        ),
        Criterion(
            name="fault_tolerance",
            weight=8.0,
            requirement="Handles component failures gracefully"
        ),
    ])

    dataset.add_item(
        submission="""
Design: URL Shortener

Components:
- Web servers behind load balancer
- Redis cache for hot URLs
- PostgreSQL for persistence
- Base62 encoding for short codes

Flow:
1. Generate unique ID (snowflake)
2. Encode as base62
3. Store mapping in DB
4. Cache popular URLs

Scale: Shard by hash of short code
""",
        description="System Design - URL Shortener",
        rubric=design_rubric,
    )

    # Problem 5: Simple string problem (global rubric)
    dataset.add_item(
        submission="""
def reverse_string(s):
    return ''.join(reversed(s))
""",
        description="String Reversal - Simple solution",
        # Uses global rubric
    )

    return dataset


async def main():
    # Create dataset
    dataset = create_coding_interview_dataset()
    print(f"Dataset: {dataset.name}")
    print(f"Total items: {len(dataset)}")

    # Show per-item rubric configuration
    print("\n" + "=" * 60)
    print("PER-ITEM RUBRIC CONFIGURATION")
    print("=" * 60)

    for idx, item in enumerate(dataset):
        effective_rubric = dataset.get_item_rubric(idx)
        reference = dataset.get_item_reference_submission(idx)

        print(f"\n[{idx}] {item.description}")
        print(f"    Custom rubric: {'Yes' if item.rubric else 'No (uses global)'}")
        print(f"    Criteria: {[c.name for c in effective_rubric.criteria]}")
        print(f"    Has reference: {'Yes' if reference else 'No'}")

    # Configure grader
    grader = CriterionGrader(
        llm_config=LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    result = await evaluate(dataset, grader, show_progress=True)

    for item_result in result.item_results:
        print(f"\n{item_result.item.description}")
        # report.score is `float | None` (None if that item's grade failed).
        score = item_result.report.score
        print(f"  Score: {score:.2f}" if score is not None else "  Score: n/a")
        print(f"  Criteria evaluated: {len(item_result.report.report or [])}")

        if item_result.report.report:
            for cr in item_result.report.report:
                if cr.verdict:
                    status = cr.verdict.value
                else:
                    status = cr.multi_choice_verdict.selected_label if cr.multi_choice_verdict else "?"
                print(f"    [{status}] {cr.name}")


if __name__ == "__main__":
    asyncio.run(main())
```
