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

# Ensemble-specific fields
print(f"Score: {result.score:.3f}")
print(f"Mean Agreement: {result.mean_agreement:.1%}")
print(f"Judge Scores: {result.judge_scores}")

# Per-criterion vote breakdown
for cr in result.report:
    print(f"{cr.criterion.requirement}")
    for vote in cr.votes:
        print(f"  {vote.judge_id}: {vote.verdict} ({vote.reason[:50]}...)")
```

## Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `majority` | > 50% of judges must vote MET |
| `weighted` | Weighted vote using judge weights |
| `unanimous` | All judges must vote MET |
| `any` | Any judge voting MET results in MET |

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
