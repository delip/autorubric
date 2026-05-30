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
        llm_config=LLMConfig(
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
