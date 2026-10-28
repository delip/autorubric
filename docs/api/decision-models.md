# Decision Models

Decision-model judges, the confidence cascade, and the offline tools that calibrate and replay a cascade.

## Overview

A decision model answers typed questions with probabilities instead of generating text, and grades an item with one request for the whole rubric. `DecisionModelConfig` configures one; it is accepted wherever a judge config is (`CriterionGrader(judge_model_config=...)`, `JudgeSpec`). `EscalationConfig` turns a grader with one decision-model judge into a confidence cascade that escalates uncertain criteria to LLM judges. `replay_escalation`, `calibrate_escalation` and `escalation_stats` choose and check a cascade's threshold offline, without new calls.

The decision-model client needs the optional TypeSafe SDK: `pip install 'autorubric[typesafe]'`.

For a guided introduction see [Decision-Model Judges](../cookbook/decision-models.md); for the end-to-end calibration workflow see the [cascade recipe](../cookbook/cascade-grading.md).

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

## Model Capabilities

Any decision model grades text. Sending images needs a model that accepts them, and autorubric looks this up in its decision-model registry, the file `src/autorubric/data/decision_models.json` shipped with the package. The registry is read only when a submission, query or reference submission holds an image, so text grading never depends on it.

| Model | Images | Formats | Pixels per side | Per image | Per request |
| --- | --- | --- | --- | --- | --- |
| `jev-1.13.0` | no | | | | |
| `jev-1.14.0` | yes | PNG, JPEG | 280 to 1568 | 16 MiB | 32 MiB |
| `jev-latest` | yes | PNG, JPEG | 280 to 1568 | 16 MiB | 32 MiB |

`jev-latest` follows the newest Jev release, so its entry is re-checked whenever Jev releases a new version. A model can be named with or without its provider prefix (`"typesafe/jev-1.14.0"` or `"jev-1.14.0"`). The prefix is removed before the model id is sent, and an experiment's manifest records the name as you gave it.

When the content holds an image, every decision-model judge's model is looked up before any judge is called. A model the registry does not list, or lists as not accepting images, raises `ValueError` naming the judge and the model, and no request is sent, not even by the LLM judges of a mixed panel. `fill_ground_truth` raises it too rather than leaving the item out, because it would fail every item with images. Each image is fitted into the model's limits before it is sent, and an item whose fitted images together exceed the per-request limit raises `ValueError` before its request.

To send images to another model, such as a new Jev release or a self-hosted endpoint, add an entry keyed by the model id as sent to the API:

```json
"my-org/rubric-dm-7b": {
  "provider": "my-org",
  "images": true,
  "image_formats": ["png", "jpeg"],
  "min_side_px": 280,
  "max_side_px": 1568,
  "max_image_bytes": 16777216,
  "max_request_bytes": 33554432,
  "notes": "Self-hosted; limits from the endpoint's documentation."
}
```

Every entry needs `provider` and `images`, and may have a `notes` string (JSON has no comments). An entry with `"images": true` also needs the five limits. `image_formats` lists `"png"` and `"jpeg"`, the formats System One reads. The sizes are inclusive bounds in pixels, and the byte limits apply to one image and to all of a request's images. A model without images has only `provider`, `images` and `notes`. The file is checked on first use, and a malformed entry raises `ValueError` naming the file and the entry.

::: autorubric.decision.decision_model_capabilities
    options:
      show_source: true

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
