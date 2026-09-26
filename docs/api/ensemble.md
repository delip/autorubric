# Ensemble

Multi-judge evaluation with configurable aggregation strategies.

## Overview

Ensemble judging combines verdicts from multiple LLM judges to improve robustness and reduce individual model biases. All graders return `EnsembleEvaluationReport` for a consistent interface (single LLM is treated as "ensemble of 1").

!!! tip "Research Background"

    Verga et al. (2024) demonstrate in "Replacing Judges with Juries" that aggregating independent judgments from diverse models reduces systematic errors. Cross-family judging (using models from different providers) is particularly effective at mitigating self-preference bias documented by He et al. (2025).

## Quick Example

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader, JudgeSpec

# Ensemble with multiple judges
grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini", weight=1.0),
        JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude", weight=1.2),
        JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "gpt", weight=1.0),
    ],
    aggregation="weighted",
)

result = await rubric.grade(to_grade=response, grader=grader)

# Ensemble-specific fields. score / mean_agreement are `float | None`
# (None on a failed grade or an empty rubric); guard before formatting.
print(f"Score: {result.score:.3f}" if result.score is not None else "Score: n/a (grade failed)")
print(
    f"Mean Agreement: {result.mean_agreement:.1%}"
    if result.mean_agreement is not None
    else "Mean Agreement: n/a"
)
print(f"Judge Scores: {result.judge_scores}")

# Per-criterion vote breakdown
for cr in result.report:
    print(f"{cr.criterion.requirement}")
    for vote in cr.votes:
        print(f"  {vote.judge_id}: {vote.verdict} ({vote.reason[:50]}...)")
```

## Keyword Form

`JudgeSpec` takes the judge's config, its `judge_id`, and an optional `weight`, in that order. A
weight must be a positive, finite number (default `1.0`); zero, negative, NaN and infinite weights
raise `ValueError`, because the weighted and threshold strategies sum and compare judge weights. When
you pass the config by keyword, call it `judge_model_config`:

```python
spec = JudgeSpec(
    judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"),
    judge_id="gpt",
    weight=1.0,
)
spec.judge_model_config  # read/write; the same object as spec.llm_config
```

The underlying dataclass field is still named `llm_config`, so `JudgeSpec(llm_config=...)` and
`spec.llm_config` keep working without a warning. Passing both keywords raises `ValueError`.

## Judge IDs

Give every judge in `judges=` its own `judge_id`. A `judge_id` is the key that separates judges,
so judges that share one are merged. Judges of the same kind all call the model of the last one.
They also share a single `judge_scores` entry, one set of per-judge metrics, and the same option
order and few-shot examples. `CriterionGrader` still accepts a repeated `judge_id` for now, but
it emits a `FutureWarning` naming the repeated ids. A repeated `judge_id` will raise `ValueError`
in the next major version. A cascade already rejects repeats (see
[Decision Models](../cookbook/decision-models.md)).

To poll one model several times, give each copy its own id. Each copy then also gets its own
option shuffle, and with `cache_enabled=True` its own response-cache entries, so a rerun replays
each copy's own answers:

```python
judges = [JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), f"gpt-{i}") for i in range(3)]
```

## Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `majority` | Head count of MET vs UNMET votes; the larger count wins |
| `weighted` | Weighted vote using judge weights |
| `unanimous` | All non-abstaining judges must vote MET |
| `any` | Any judge voting MET results in MET |

These apply to **binary** criteria only and are independent of multi-choice aggregation
(`ordinal_aggregation` / `nominal_aggregation`). Conceptually, binary `unanimous` ≡ the
**min** over the {0, 1} option values and `any` ≡ the **max**; the ordinal analogs are the
`min` / `max` strategies (see the [multi-choice cookbook](../cookbook/multi-choice-rubrics.md)).

Every strategy counts only MET and UNMET votes. CANNOT_ASSESS votes, including those of judge
calls that failed with an API or parse error, are set aside first, and a criterion whose votes
all abstain is CANNOT_ASSESS. A `majority` or `weighted` tie goes to the verdict that scores
lowest for the criterion's weight sign: UNMET for a positive (or zero) weight, MET for a
negative one.

---

## AggregationStrategy

Enum for binary verdict aggregation strategies.

::: autorubric.AggregationStrategy
    options:
      show_source: true

---

## EnsembleEvaluationReport

Evaluation result from ensemble grading with per-judge breakdown.

::: autorubric.EnsembleEvaluationReport
    options:
      show_source: true
      members_order: source

---

## EnsembleCriterionReport

Per-criterion result with individual judge votes.

::: autorubric.EnsembleCriterionReport
    options:
      show_source: true
      members_order: source

---

## JudgeVote

Individual judge's verdict for a criterion.

::: autorubric.JudgeVote
    options:
      show_source: true
      members_order: source

---

## References

He, J., Shi, J., Zhuo, T. Y., Treude, C., Sun, J., Xing, Z., Du, X., and Lo, D. (2025). LLM-as-a-Judge for Software Engineering: Literature Review, Vision, and the Road Ahead. arXiv:2510.24367.

Verga, P., Hofstatter, S., Althammer, S., Su, Y., Piktus, A., Arkhangorodsky, A., Xu, M., White, N., and Lewis, P. (2024). Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models. arXiv:2404.18796.
