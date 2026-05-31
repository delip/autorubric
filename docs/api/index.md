# API Reference

Complete API documentation for AutoRubric, organized by functional area.

## Overview

AutoRubric exports 91 public items across the main module and `autorubric.graders`, plus 20 additional building-block functions and types available from `autorubric.meta`. This reference is organized into thematic chapters for easier navigation.

## Quick Links

| Chapter | Key Exports | Description |
|---------|-------------|-------------|
| [CANNOT_ASSESS Handling](cannot-assess.md) | `CannotAssessConfig`, `CannotAssessStrategy` | Configure handling of uncertain verdicts |
| [Core Grading](core-grading.md) | `Criterion`, `Rubric`, `EvaluationReport` | Fundamental types for rubric-based evaluation |
| [Dataset](dataset.md) | `DataItem`, `RubricDataset` | Dataset management and serialization |
| [Distribution Metrics](distribution-metrics.md) | `earth_movers_distance`, `ks_test` | Statistical distribution comparisons |
| [Ensemble](ensemble.md) | `EnsembleEvaluationReport`, `JudgeVote` | Multi-judge aggregation |
| [Eval Runner](eval-runner.md) | `EvalRunner`, `evaluate()` | Batch evaluation with checkpointing |
| [Few-Shot](few-shot.md) | `FewShotConfig`, `FewShotExample` | Calibration with labeled examples |
| [Graders](graders.md) | `CriterionGrader`, `Grader`, `JudgeSpec` | Grader implementations |
| [Length Penalty](length-penalty.md) | `LengthPenalty`, `compute_length_penalty` | Verbosity control |
| [LLM Infrastructure](llm.md) | `LLMConfig`, `LLMClient`, `generate()` | LLM client and configuration |
| [Meta-Rubric Evaluation](meta.md) | `evaluate_rubric_standalone`, `evaluate_rubric_in_context` | Assess rubric quality |
| [Rubric Improvement](meta-improvement.md) | `improve_rubric`, `ImprovementRunner`, `ImprovementConfig` | Iterative rubric improvement |
| [Metrics](metrics.md) | `MetricsResult`, `compute_metrics` | Agreement and correlation metrics |
| [Multi-Choice](multi-choice.md) | `CriterionOption`, `MultiChoiceVerdict` | Ordinal and nominal scales |
| [Utilities](utilities.md) | `aggregate_token_usage`, `word_count` | Helper functions |

## Import Patterns

### Main Module

```python
from autorubric import (
    # Core types
    Criterion,
    Rubric,
    CriterionVerdict,
    EvaluationReport,

    # LLM configuration
    LLMConfig,
    LLMClient,

    # Dataset
    DataItem,
    RubricDataset,

    # Evaluation
    EvalRunner,
    evaluate,

    # Metrics
    compute_metrics,
    MetricsResult,
)
```

### Graders Module

```python
from autorubric.graders import (
    CriterionGrader,
    Grader,
    JudgeSpec,
)
```

### Meta Module

```python
from autorubric.meta import (
    # Evaluation
    evaluate_rubric_standalone,
    evaluate_rubric_in_context,
    get_standalone_meta_rubric,
    get_in_context_meta_rubric,

    # Improvement - convenience API & runner
    improve_rubric,
    ImprovementRunner,

    # Improvement - types
    ConvergenceFn,
    CriterionErrorReport,
    CriterionExemplar,
    HeldOutValidationResult,
    ImprovementConfig,
    ImprovementProgressDisplay,
    ImprovementResult,
    IssueDetail,
    IterationResult,

    # Improvement - building blocks
    build_revision_history,
    compute_expected_scores,
    diff_issues,
    extract_issues,
    format_agreement_for_prompt,
    format_ground_truth_for_prompt,
    format_held_out_for_prompt,
    format_issues_for_prompt,
    pareto_accept,
    revise_rubric,
    revise_rubric_held_out,
    validate_agreement,
    validate_criteria_structure,
    validate_ground_truth,
    validate_held_out,
)
```

## Type Hierarchy

```
Grader (ABC)
└── CriterionGrader

BaseModel (Pydantic)
├── EvaluationReport
├── EnsembleEvaluationReport
├── CriterionReport
├── EnsembleCriterionReport
└── MetricsResult
    has ──▶ list[CriterionMetrics | OrdinalCriterionMetrics | NominalCriterionMetrics]
    has ──▶ dict[str, JudgeMetrics]  (optional)
    has ──▶ BiasResult
    has ──▶ BootstrapResults  (optional)
```

## Architecture

### Grading Flow

1. `Rubric.grade()` delegates to grader's `grade()` method
2. `CriterionGrader` treats single LLM as "ensemble of 1"
3. Makes concurrent LLM calls per criterion per judge via `asyncio.gather()`
4. Aggregates votes using configurable strategy
5. Returns `EnsembleEvaluationReport` (consistent interface)

### Score Calculation

```python
# Positive criteria: MET earns weight, UNMET earns 0
# Negative criteria: MET subtracts weight, UNMET contributes 0
weighted_sum = sum(verdict_value * criterion.weight for each criterion)
score = clamp(weighted_sum / total_positive_weight, 0, 1)  # if normalized
# Length penalty subtracted after base calculation
```

## Conventions

- All graders return `EnsembleEvaluationReport` for consistent interface
- `raw_score` (the unnormalized weighted sum) is populated regardless of the `normalize` setting on a successful grade, but is `None` on a failed/error report (consumers should filter on `error is not None`)
- Judge-call failures route via `classify_grading_error`: infrastructure/parse failures become `CANNOT_ASSESS` (`na=True`, excluded from scoring under the default SKIP strategy); only `unknown` errors fall back to the conservative worst-case verdict (UNMET for positive weight, MET for negative weight). Failed reports carry a category-prefixed `error` and `is_error`
- Filter `error is not None` results in training pipelines
- Rate limiting via `LLMConfig.max_parallel_requests` (per-provider semaphore)
