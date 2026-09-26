# Eval Runner

High-throughput batch evaluation with checkpointing, resumption, and timing statistics.

## Overview

`EvalRunner` and the `evaluate()` convenience function provide infrastructure for evaluating datasets at scale. Features include parallel execution with rate limiting, progress tracking, automatic checkpointing for long-running jobs, and comprehensive timing/cost statistics.

!!! tip "Research Background"

    Casabianca et al. (2025) recommend maintaining a "gold set" of human-graded examples and sampling 1-5% of production traffic for continuous validation. EvalRunner provides the infrastructure for systematic evaluation with checkpointing for long-running jobs and cost tracking for budget management.

## Quick Example

```python
from autorubric import RubricDataset, LLMConfig, evaluate
from autorubric.graders import CriterionGrader

async def main():
    dataset = RubricDataset.from_file("essays.json")
    grader = CriterionGrader(
        judge_model_config=LLMConfig(
            model="openai/gpt-4.1-mini",
            max_parallel_requests=10,
        )
    )

    result = await evaluate(dataset, grader, show_progress=True)

    print(f"Evaluated {result.successful_items}/{result.total_items}")
    print(f"Throughput: {result.timing_stats.items_per_second:.2f} items/s")
    print(f"Total cost: ${result.total_completion_cost or 0:.4f}")
```

## Checkpointing and Resumption

```python
from autorubric import EvalRunner, EvalConfig, EvalResult

# First run (may be interrupted)
config = EvalConfig(
    experiment_name="my-essay-eval",
    experiments_dir="./experiments",
    show_progress=True,
)
runner = EvalRunner(dataset=dataset, grader=grader, config=config)
result = await runner.run()
# Saves to: experiments/my-essay-eval/manifest.json + items.jsonl

# Resume after crash
runner = EvalRunner(dataset=dataset, grader=grader, config=config)
result = await runner.run()  # Skips already-completed items

# Load results later
result = EvalResult.from_experiment("experiments/my-essay-eval")
```

A run resumes only from a checkpoint of the same dataset. With a changed dataset, or with
`resume=False`, it starts fresh and replaces the directory's checkpoint, so an experiment
directory always holds exactly one run.

## Failed Items

An item fails when its grading raises, or when the grade returns a report whose `error` is
set. The second case covers an item whose every criterion's judgment failed, for example
because every judge call failed: the report has no score and its `error` begins
`Every criterion's judgment failed:`. The runner records the error in the item's
`ItemResult.error` and logs a warning. A failed item:

- counts in `failed_items` and `errors`;
- stops the run when `fail_fast=True`;
- is left out of `get_scores()`, `get_reports()` and `filter_successful()` (and returned by
  `filter_failed()`);
- still counts in `total_token_usage` and `total_completion_cost`: a billed call counts
  whatever became of the grade (a decision-model request answered without a usable answer,
  say);
- is not graded again when the run resumes, like any item already done, but still counts in
  the resumed run's `failed_items`.

A run saved by v1.5.3 or earlier recorded an item whose every judgment failed as successful,
with a fabricated score (for example 0.0 under `SKIP`). `EvalResult.from_experiment` loads it
as it was saved, and `compute_metrics` still leaves it out (see
[Errored Items and Score Pairs](metrics.md#errored-items-and-score-pairs)).

An item whose judges answered `CANNOT_ASSESS` on every criterion has nothing left to score
under the default `SKIP` strategy. Its `score` is `None`, but nothing failed: its `error` is
`None` and it counts as successful, though `get_scores()` skips it.

```python
for item_result in result.filter_failed():
    print(f"Item {item_result.item_idx}: {item_result.error}")
```

## Rate Limiting

```python
from autorubric.graders import CriterionGrader, JudgeSpec

grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="openai/gpt-4.1", max_parallel_requests=10), "gpt"),
        JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929", max_parallel_requests=5), "claude"),
    ],
    aggregation="majority",
)
```

Rate limiting uses a global per-provider semaphore, so all `openai/*` models share the same limit.

---

## evaluate

Convenience function for batch evaluation.

::: autorubric.evaluate
    options:
      show_source: true

---

## EvalRunner

Runner class for batch evaluation with checkpointing.

::: autorubric.EvalRunner
    options:
      show_source: true
      members_order: source

---

## EvalConfig

Configuration options for evaluation runs.

::: autorubric.EvalConfig
    options:
      show_source: true
      members_order: source

---

## EvalResult

Results from a completed evaluation run.

::: autorubric.EvalResult
    options:
      show_source: true
      members_order: source

---

## ItemResult

Result for a single evaluated item.

::: autorubric.ItemResult
    options:
      show_source: true
      members_order: source

---

## EvalTimingStats

Timing statistics for an evaluation run.

::: autorubric.EvalTimingStats
    options:
      show_source: true
      members_order: source

---

## ExperimentManifest

Metadata for a saved experiment.

::: autorubric.ExperimentManifest
    options:
      show_source: true
      members_order: source

---

## References

Casabianca, J., McCaffrey, D. F., Johnson, M. S., Alper, N., and Zubenko, V. (2025). Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications. arXiv:2501.02334.
