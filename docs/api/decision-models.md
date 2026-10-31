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

Any decision model grades text. Sending images needs a model that accepts them, and autorubric looks this up in its decision-model registry, the file `autorubric/data/decision_models.json` inside the installed package (`src/autorubric/data/decision_models.json` in the repository). `decision_model_capabilities(model)` returns a model's entry, or `None` for a model the registry does not list. A judge can also declare its model's entry itself, with `DecisionModelConfig(capabilities=...)`, which then takes the registry's place for that judge. Either is read only when a submission, query or reference submission holds an image, so text grading never depends on them.

| Model | Images | Formats | Pixels per side | Per image | Per request |
| --- | --- | --- | --- | --- | --- |
| `jev-1.13.0` | no | | | | |
| `jev-1.14.0` | yes | PNG, JPEG | 280 to 1568 | 16 MiB | 32 MiB |
| `jev-latest` | yes | PNG, JPEG | 280 to 1568 | 16 MiB | 32 MiB |

`jev-latest` follows the newest Jev release, so its entry is re-checked whenever Jev releases a new version. A model can be named with or without its provider prefix (`"typesafe/jev-1.14.0"` or `"jev-1.14.0"`). The prefix is removed before the model id is sent, and an experiment's manifest records the name as you gave it.

When the content holds an image, every decision-model judge's model is looked up before any judge is called. A model with neither a registry entry nor declared capabilities, or one whose entry says it does not accept images, raises `ValueError` naming the judge and the model, and no request is sent, not even by the LLM judges of a mixed panel. `fill_ground_truth` raises it too rather than leaving the item out, because it would fail every item with images. Each image is fitted into the model's limits before it is sent (see *Images* below).

### Declaring a model's capabilities

To send images to a model the registry does not list, such as a new Jev release or a self-hosted endpoint, declare what it accepts on the judge's config. You do not need to edit the installed package.

```python
from autorubric import DecisionModelConfig
from autorubric.graders import CriterionGrader

judge = DecisionModelConfig(
    model="my-org/rubric-dm-7b",
    api_base="https://xyz.endpoints.huggingface.cloud",
    capabilities={
        "provider": "my-org",
        "images": True,
        "image_formats": ["png", "jpeg"],
        "min_side_px": 64,
        "max_side_px": 1024,
        "max_image_bytes": 4_000_000,
        "max_request_bytes": 8_000_000,
    },
)
grader = CriterionGrader(judge_model_config=judge)
```

`capabilities` has the shape of a registry entry, described below, and is checked by the same rules when the config is built. A malformed one raises `ValueError` naming `DecisionModelConfig.capabilities`, with the message a malformed registry entry would give. When it is set, it replaces the registry entry for that judge entirely, so it also works for a model the registry lists. You can start from a listed entry and change one limit, as in `capabilities={**decision_model_capabilities("jev-latest"), "max_side_px": 1024}`, or turn images off for one judge with `{"provider": "typesafe", "images": False}`. The config keeps a plain `dict` copy, so it still pickles and compares, and an experiment's manifest records it under the judge's `capabilities` key. Judges without it have the manifest entries they had before.

### Registry entries

In a checkout of the repository, a model can instead be added to the registry, keyed by the model id as sent to the API:

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

::: autorubric.decision_model_capabilities
    options:
      show_source: true

---

## Images

A decision model sees the item as one state. Each image in it is replaced by a reference, `<image:{sha256}>`, the SHA-256 of the image's original bytes, and the request carries the image as the asset of that name. An image that occurs twice in an item is sent once. A request that carries images also sends the header `TypeSafe-Danger-Zone-Images: true`, which System One requires on it. A text-only request carries no assets and no such header, so it is the request autorubric sent before it supported images. Sending images needs `typesafe-sdk` 0.7.5 or later, which `pip install -U 'autorubric[typesafe]'` installs. With an older SDK the grade raises `ImportError`.

System One does not resize images, so autorubric fits each image into the model's entry (its declared `capabilities`, else its registry entry) before the request:

- **Passed through.** A still PNG, or a baseline JPEG in grayscale or RGB, that has no EXIF rotation, whose sides are all within the entry's bounds and whose size is within the per-image limit, is sent as its original bytes.
- **Re-encoded.** Any other image is decoded and re-encoded. An animated image, or a JPEG holding several pictures, is sent as its first frame. EXIF rotation is applied, and samples deeper than 8 bits are reduced to 8-bit grayscale. An image from a PNG, GIF or WebP, or one with a palette, 1-bit pixels or transparency, becomes a PNG. Any other becomes a baseline JPEG (quality 95).
- **Resized.** An image with a side over the maximum is scaled down so that its longest side is the maximum (1568 pixels for Jev). One with a side under the minimum is scaled up, keeping its aspect ratio, until its shortest side is the minimum (280 pixels for Jev) or its longest side reaches the maximum. Resampling is Lanczos. An image that scaling up would change by at most a pixel per side keeps its size and is only padded. Scaling down always resamples, so a 1569 x 1000 image is sent at 1568 x 999.
- **Padded.** A side still under the minimum, as in a very wide or very tall image, is padded with white to the minimum, with the image centered on the canvas. An image with transparency is padded with transparent white.

Fitting is deterministic for a given Pillow version. The cache key of a request with images names each image by the SHA-256 of its original bytes and also records the limits the images were fitted into, so a re-run reuses cached answers, while a change to those limits, in the registry or in `capabilities`, asks the model again. A text-only request's cache key is unchanged. Colours are not converted to sRGB, and a colour profile is not carried into a re-encoded or padded image, so an image in another colour space (Display P3, say) is judged with shifted colours once it is re-encoded. To keep an image exactly as it is, give it as a PNG or baseline JPEG within the model's limits.

Two failures raise `ValueError` before the request and fail only their item. One is an image that cannot be decoded, or that is still over the per-image limit once fitted. The other is an item whose fitted images together exceed the per-request limit (32 MiB for Jev). `EvalRunner` records such an item as failed, and `fill_ground_truth` leaves it out.

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
