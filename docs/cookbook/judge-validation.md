# Validating Your Judge Against Human Labels

Measure how well your LLM judge agrees with human evaluators.

## The Scenario

You've deployed an LLM judge for content moderation. Before trusting it at scale, you need to validate it against human moderator decisions. You have 100 items with human labels and want comprehensive metrics: accuracy, precision, recall, Cohen's kappa, and analysis of systematic biases.

## What You'll Learn

- Using `compute_metrics()` with ground truth datasets
- Interpreting accuracy, precision, recall, F1, and kappa
- Detecting systematic bias with `metrics.bias`
- Per-criterion breakdown for targeted improvements
- Bootstrap confidence intervals for statistical rigor

## The Solution

```mermaid
flowchart LR
    A[Ground Truth Dataset\nwith human labels] --> B[LLM Grader]
    B --> C[Predictions]
    A --> D{Compare}
    C --> D
    D --> E[Metrics\naccuracy, kappa,\nprecision, recall]
    E --> F[Per-Criterion\nAnalysis]
```

### Step 1: Prepare Your Validation Dataset

Load a dataset with human-labeled ground truth:

```python
from autorubric import RubricDataset

# Load dataset with ground truth labels
dataset = RubricDataset.from_file("content_moderation_labeled.json")

print(f"Dataset: {dataset.name}")
print(f"Items: {len(dataset)}")
print(f"Criteria: {dataset.criterion_names}")

# Verify ground truth coverage
items_with_gt = sum(1 for item in dataset if item.ground_truth is not None)
print(f"Items with ground truth: {items_with_gt}/{len(dataset)}")
```

### Step 2: Run Evaluation

Evaluate the dataset with your grader:

```python
from autorubric import LLMConfig, evaluate
from autorubric.graders import CriterionGrader

grader = CriterionGrader(
    llm_config=LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)
)

# Run evaluation
result = await evaluate(
    dataset,
    grader,
    show_progress=True,
    experiment_name="judge-validation-v1"
)

print(f"Evaluated: {result.successful_items}/{result.total_items}")
print(f"Cost: ${result.total_completion_cost:.4f}")
```

### Step 3: Compute Validation Metrics

Use `compute_metrics()` to compare predictions against ground truth:

```python
metrics = result.compute_metrics(dataset)

# Overall metrics
print("=" * 50)
print("OVERALL VALIDATION METRICS")
print("=" * 50)
print(f"Criterion Accuracy:  {metrics.criterion_accuracy:.1%}")
print(f"Cohen's Kappa:       {metrics.mean_kappa:.3f}")
print(f"Precision (MET):     {metrics.criterion_precision:.1%}")
print(f"Recall (MET):        {metrics.criterion_recall:.1%}")
print(f"F1 Score:            {metrics.criterion_f1:.3f}")
```

### Interpreting the Metrics

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| **Accuracy** | % of verdicts matching ground truth | >85% |
| **Cohen's Kappa** | Agreement beyond chance | >0.6 (substantial) |
| **Precision** | Of predicted MET, % actually MET | Depends on use case |
| **Recall** | Of actual MET, % predicted MET | Depends on use case |
| **F1** | Harmonic mean of precision/recall | >0.7 |

!!! tip "Kappa Interpretation"
    | Kappa | Agreement Level |
    |-------|-----------------|
    | < 0.2 | Slight |
    | 0.2–0.4 | Fair |
    | 0.4–0.6 | Moderate |
    | 0.6–0.8 | Substantial |
    | > 0.8 | Near perfect |

### Step 4: Per-Criterion Analysis

Identify which criteria need improvement:

```python
print("\nPER-CRITERION BREAKDOWN")
print("-" * 50)
print(f"{'Criterion':<25} {'Acc':>8} {'Kappa':>8} {'F1':>8}")
print("-" * 50)

for cr_metrics in metrics.per_criterion:
    print(f"{cr_metrics.name:<25} {cr_metrics.accuracy:>7.1%} {cr_metrics.kappa:>8.3f} "
          f"{cr_metrics.f1:>8.3f}")
```

Sample output:

```
PER-CRITERION BREAKDOWN
--------------------------------------------------
Criterion                      Acc    Kappa       F1
--------------------------------------------------
hate_speech                  92.0%    0.812    0.891
harassment                   87.0%    0.714    0.823
misinformation               78.0%    0.521    0.742
self_harm                    95.0%    0.876    0.923
spam                         89.0%    0.761    0.856
```

![Per-criterion validation metrics for content moderation](../images/judge-validation-per-criterion.png)

The `misinformation` criterion has lower kappa—consider few-shot calibration or clearer criteria definition.

### Step 5: Detect Systematic Bias

Check if the judge systematically over- or under-predicts MET:

```python
bias = metrics.bias

print(f"\nSystematic Bias Analysis:")
print(f"  Mean bias: {bias.mean_bias:.3f}")
print(f"  Bias direction: {bias.direction}")  # "permissive" or "strict"
print(f"  Statistically significant: {bias.is_significant}")
print(f"  P-value: {bias.p_value:.4f}")
print(f"  Effect size: {bias.effect_size:.3f}")
```

- **Permissive bias**: Judge marks MET more often than humans
- **Strict bias**: Judge marks MET less often than humans

!!! note "Remediation strategies for judge bias"
    When bias is detected, several options can bring the judge closer to human agreement.
    Adding few-shot examples of correctly labeled edge cases often fixes borderline misclassifications.
    Rewording ambiguous criteria reduces disagreement that stems from unclear requirements rather than model error.
    Upgrading to a more capable model or using ensemble judging across multiple models both reduce individual model bias at the cost of higher inference spend.

### Step 6: Bootstrap Confidence Intervals

Get statistical confidence in your metrics:

```python
metrics = result.compute_metrics(
    dataset,
    bootstrap=True,
    n_bootstrap=1000,
    confidence_level=0.95,
    seed=42
)

print(f"\nAccuracy: {metrics.criterion_accuracy:.1%}")
print(f"  95% CI: [{metrics.bootstrap.accuracy_ci[0]:.1%}, {metrics.bootstrap.accuracy_ci[1]:.1%}]")

print(f"\nKappa: {metrics.mean_kappa:.3f}")
print(f"  95% CI: [{metrics.bootstrap.kappa_ci[0]:.3f}, {metrics.bootstrap.kappa_ci[1]:.3f}]")
```

!!! warning "Bootstrap Cost"
    Bootstrap analysis is computationally expensive. For quick iteration,
    use `bootstrap=False` during development, then enable for final validation.

### Step 7: Score Correlation

Check how well predicted scores correlate with ground truth scores:

```python
print(f"\nScore Correlation:")
print(f"  Pearson:  {metrics.score_pearson.coefficient:.3f}")
print(f"  Spearman: {metrics.score_spearman.coefficient:.3f}")
print(f"  RMSE:     {metrics.score_rmse:.3f}")
```

High correlation (>0.8) indicates the judge ranks items similarly to humans, even if individual verdicts differ.

### Step 8: Export Results

Save metrics for reporting:

```python
# Get summary as text
print(metrics.summary())

# Export to DataFrame for analysis
df = metrics.to_dataframe()
df.to_csv("validation_results.csv")
```

## Key Takeaways

- **`compute_metrics()`** provides comprehensive validation against ground truth
- **Cohen's kappa** is more meaningful than accuracy for imbalanced labels
- **Per-criterion analysis** identifies weak spots for targeted improvement
- **Systematic bias** detection reveals if judge is too permissive or strict
- **Bootstrap CIs** provide statistical confidence in your metrics
- **Score correlation** shows overall ranking agreement

## Going Further

- [Few-Shot Calibration](few-shot-calibration.md) - Improve accuracy on weak criteria
- [Ensemble Judging](ensemble-judging.md) - Reduce bias with multiple judges
- [API Reference: Metrics](../api/metrics.md) - Full metrics documentation

---

## Appendix: Complete Code

```python
"""Judge Validation - Content Moderation Evaluation"""

import asyncio
from autorubric import (
    Rubric, RubricDataset, CriterionVerdict,
    LLMConfig, evaluate
)
from autorubric.graders import CriterionGrader


def create_moderation_dataset() -> RubricDataset:
    """Create a content moderation dataset with human labels."""

    rubric = Rubric.from_dict([
        {
            "name": "hate_speech",
            "weight": -15.0,
            "requirement": "Contains hate speech targeting protected groups"
        },
        {
            "name": "harassment",
            "weight": -12.0,
            "requirement": "Contains harassment or bullying directed at individuals"
        },
        {
            "name": "misinformation",
            "weight": -10.0,
            "requirement": "Contains demonstrably false claims about health/safety"
        },
        {
            "name": "appropriate_content",
            "weight": 10.0,
            "requirement": "Content is appropriate for general audience"
        }
    ])

    dataset = RubricDataset(
        prompt="Moderate this user-generated content.",
        rubric=rubric,
        name="content-moderation-v1"
    )

    # Sample content with human moderator labels
    # In practice, these would come from your human annotation process
    items = [
        {
            "submission": "I love this product! Works great for my family.",
            "description": "Positive review",
            "ground_truth": [CriterionVerdict.UNMET, CriterionVerdict.UNMET,
                           CriterionVerdict.UNMET, CriterionVerdict.MET]
        },
        {
            "submission": "This group of people are all criminals and should be deported.",
            "description": "Hate speech",
            "ground_truth": [CriterionVerdict.MET, CriterionVerdict.UNMET,
                           CriterionVerdict.UNMET, CriterionVerdict.UNMET]
        },
        {
            "submission": "The vaccine contains microchips for mind control.",
            "description": "Health misinformation",
            "ground_truth": [CriterionVerdict.UNMET, CriterionVerdict.UNMET,
                           CriterionVerdict.MET, CriterionVerdict.UNMET]
        },
        {
            "submission": "Just tried the new restaurant downtown. The pasta was amazing!",
            "description": "Normal review",
            "ground_truth": [CriterionVerdict.UNMET, CriterionVerdict.UNMET,
                           CriterionVerdict.UNMET, CriterionVerdict.MET]
        },
        {
            "submission": "@username you're so stupid, everyone hates you, just quit already",
            "description": "Direct harassment",
            "ground_truth": [CriterionVerdict.UNMET, CriterionVerdict.MET,
                           CriterionVerdict.UNMET, CriterionVerdict.UNMET]
        },
        {
            "submission": "Studies show 5G towers cause cancer. Wake up sheeple!",
            "description": "Health misinformation",
            "ground_truth": [CriterionVerdict.UNMET, CriterionVerdict.UNMET,
                           CriterionVerdict.MET, CriterionVerdict.UNMET]
        },
        {
            "submission": "Thanks for the help! Your customer service team is wonderful.",
            "description": "Positive feedback",
            "ground_truth": [CriterionVerdict.UNMET, CriterionVerdict.UNMET,
                           CriterionVerdict.UNMET, CriterionVerdict.MET]
        },
        {
            "submission": "Anyone else notice the weather has been weird lately?",
            "description": "Neutral observation",
            "ground_truth": [CriterionVerdict.UNMET, CriterionVerdict.UNMET,
                           CriterionVerdict.UNMET, CriterionVerdict.MET]
        },
        {
            "submission": "[Religious group] are taking over the country. We need to stop them.",
            "description": "Religious discrimination",
            "ground_truth": [CriterionVerdict.MET, CriterionVerdict.UNMET,
                           CriterionVerdict.UNMET, CriterionVerdict.UNMET]
        },
        {
            "submission": "The product arrived damaged. Very disappointed with shipping.",
            "description": "Negative but appropriate review",
            "ground_truth": [CriterionVerdict.UNMET, CriterionVerdict.UNMET,
                           CriterionVerdict.UNMET, CriterionVerdict.MET]
        }
    ]

    for item in items:
        dataset.add_item(**item)

    return dataset


async def main():
    # Create/load dataset
    dataset = create_moderation_dataset()
    print(f"Dataset: {dataset.name}")
    print(f"Items: {len(dataset)}")
    print(f"Criteria: {dataset.criterion_names}")

    # Configure grader
    grader = CriterionGrader(
        llm_config=LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)
    )

    # Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    result = await evaluate(
        dataset,
        grader,
        show_progress=True,
        experiment_name="moderation-validation"
    )

    print(f"\nEvaluated: {result.successful_items}/{result.total_items}")
    print(f"Cost: ${result.total_completion_cost or 0:.4f}")

    # Compute validation metrics
    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)

    metrics = result.compute_metrics(dataset)

    print(f"\nOverall Metrics:")
    print(f"  Criterion Accuracy:  {metrics.criterion_accuracy:.1%}")
    print(f"  Cohen's Kappa:       {metrics.mean_kappa:.3f}")
    print(f"  Precision (MET):     {metrics.criterion_precision:.1%}")
    print(f"  Recall (MET):        {metrics.criterion_recall:.1%}")
    print(f"  F1 Score:            {metrics.criterion_f1:.3f}")

    # Per-criterion breakdown
    print("\n" + "-" * 60)
    print("PER-CRITERION METRICS")
    print("-" * 60)
    print(f"{'Criterion':<20} {'Acc':>8} {'Kappa':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 60)

    for cr_metrics in metrics.per_criterion:
        print(f"{cr_metrics.name:<20} {cr_metrics.accuracy:>7.1%} {cr_metrics.kappa:>8.3f} "
              f"{cr_metrics.precision:>7.1%} {cr_metrics.recall:>7.1%}")

    # Score correlation
    print("\n" + "-" * 60)
    print("SCORE CORRELATION")
    print("-" * 60)
    print(f"  Pearson r:   {metrics.score_pearson.coefficient:.3f}")
    print(f"  Spearman ρ:  {metrics.score_spearman.coefficient:.3f}")
    print(f"  RMSE:        {metrics.score_rmse:.3f}")

    # Bias analysis
    print("\n" + "-" * 60)
    print("BIAS ANALYSIS")
    print("-" * 60)

    # Count predicted vs ground truth MET rates
    pred_met = 0
    gt_met = 0
    total = 0

    for item_result in result.item_results:
        if item_result.error:
            continue
        item = item_result.item
        if item.ground_truth is None:
            continue

        for j, cr in enumerate(item_result.report.report or []):
            total += 1
            if cr.verdict == CriterionVerdict.MET:
                pred_met += 1
            if item.ground_truth[j] == CriterionVerdict.MET:
                gt_met += 1

    if total > 0:
        pred_rate = pred_met / total
        gt_rate = gt_met / total
        direction = "permissive" if pred_rate > gt_rate else "strict"
        magnitude = abs(pred_rate - gt_rate)

        print(f"  Predicted MET rate:    {pred_rate:.1%}")
        print(f"  Ground truth MET rate: {gt_rate:.1%}")
        print(f"  Bias direction:        {direction}")
        print(f"  Bias magnitude:        {magnitude:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
```
