# Decision Models

Decision-model judges, the confidence cascade, and the offline tools that calibrate and replay a cascade.

## Overview

A decision model answers typed questions with probabilities instead of generating text, and grades an item with one request for the whole rubric. `DecisionModelConfig` configures one; it is accepted wherever a judge config is (`CriterionGrader(judge_model_config=...)`, `JudgeSpec`). `EscalationConfig` turns a grader with one decision-model judge into a confidence cascade that escalates uncertain criteria to LLM judges. `replay_escalation`, `calibrate_escalation` and `escalation_stats` choose and check a cascade's threshold offline, without new calls.

The decision-model client needs the optional TypeSafe SDK: `pip install 'autorubric[typesafe]'`.

For a guided introduction see [Decision-Model Judges](../decision-models.md); for the end-to-end calibration workflow see the [cascade recipe](../cookbook/cascade-grading.md).

## Quick Example

```python
from autorubric import (
    DecisionModelConfig, EscalationConfig, LLMConfig,
    calibrate_escalation, replay_escalation,
)
from autorubric.graders import CriterionGrader, JudgeSpec

jev = DecisionModelConfig(model="jev-latest", cache_enabled=True)
gemini = LLMConfig(model="gemini/gemini-3-flash-preview")

# Standalone
grader = CriterionGrader(judge_model_config=jev)

# Cascade
grader = CriterionGrader(
    judge_model_config=jev,
    escalation=EscalationConfig(judges=gemini, threshold=0.72),
    seed=7,
)

# Offline calibration from a decision-model run and an LLM run over the same items
curve = calibrate_escalation(calib, dm_calib, llm_calib)
best = curve.best(tolerance=0.005)
hybrid = replay_escalation(dm_test, llm_test, best)
```

---

## DecisionModelConfig

Configuration of a decision-model judge: endpoint, framings, threshold, cache and price.

::: autorubric.DecisionModelConfig
    options:
      show_source: true
      members_order: source

---

## EscalationConfig

Configuration of a confidence cascade: the LLM escalation judges and the thresholds.

::: autorubric.graders.EscalationConfig
    options:
      show_source: true
      members_order: source

---

## replay_escalation

Build the `EvalResult` a cascade would have produced from a decision-model run and an LLM run, with no calls.

::: autorubric.replay_escalation
    options:
      show_source: true

---

## calibrate_escalation

Replay and measure a cascade at every threshold of a sweep.

::: autorubric.calibrate_escalation
    options:
      show_source: true

---

## escalation_stats

Diagnostics of a live or replayed cascade run.

::: autorubric.escalation_stats
    options:
      show_source: true

---

## EscalationCurve

The points of a threshold sweep and the rule that picks one.

::: autorubric.EscalationCurve
    options:
      show_source: true
      members_order: source

---

## EscalationPoint

One operating point of a cascade: its thresholds and what they achieve.

::: autorubric.EscalationPoint
    options:
      show_source: true
      members_order: source
