# Cheap First-Pass Grading with Jev and an LLM Fallback

Let a decision model grade every criterion and send only its uncertain criteria to an LLM judge, with a threshold chosen on held-out data.

## The Scenario

You grade thousands of short answers a week against a ten-criterion rubric. An LLM judge is accurate enough, but it makes one call per criterion and re-sends the answer each time, so the bill grows with every criterion. TypeSafe's Jev, a [decision model](decision-models.md), grades the whole rubric in one request for a small fraction of the price, and it reports how confident it is. You want Jev to handle the criteria it is sure about and the LLM to handle the rest, at the LLM's accuracy and a lower price. You have a few hundred labelled answers to check that the trade holds.

## What You'll Learn

- Grading a labelled calibration split with each judge alone
- Sweeping escalation thresholds offline with `calibrate_escalation`
- Picking a threshold with `EscalationCurve.best(tolerance=...)`
- Checking the choice on a held-out split with `replay_escalation` and `escalation_stats`, without running the cascade
- Deploying the cascade so that its LLM prompts match the ones you calibrated on

## The Solution

A cascade is a cost tool. It can match the LLM judge's accuracy for less money when two conditions hold on your data: Jev is accurate on the criteria it keeps, and the LLM beats Jev on the criteria Jev defers. This recipe measures both before anything is deployed.

### Step 1: Configure the Two Judges

```python
from autorubric import DecisionModelConfig, LLMConfig
from autorubric.graders import CriterionGrader, JudgeSpec

SEED = 7

jev = DecisionModelConfig(
    model="jev-latest",             # reads TYPESAFE_API_KEY
    cache_enabled=True,             # Jev is not bit-deterministic; the cache makes re-runs identical
    input_cost_per_token=0.042e-6,  # your price, so that costs are reported
)
gemini = LLMConfig(model="gemini/gemini-3-flash-preview", cache_enabled=True)

dm_grader = CriterionGrader(judge_model_config=jev)
llm_grader = CriterionGrader(judges=[JudgeSpec(gemini, "escalation")], seed=SEED)
```

The LLM grader is built the way the cascade will call its fallback. `EscalationConfig(judges=gemini, ...)` names a bare `LLMConfig` `"escalation"`, and the cascade will use `seed=SEED`. With the same `judge_id` and `seed`, every prompt of the LLM run, option shuffles and few-shot examples included, is the prompt the live cascade sends for that criterion, so the offline replay reproduces the cascade exactly.

### Step 2: Split the Labelled Data

```python
from autorubric import RubricDataset

dataset = RubricDataset.from_file("short_answers.json")  # items with ground truth
calib, test = dataset.split_train_test(n_train=len(dataset) // 2, seed=42)
```

Choose the threshold on `calib` and measure it on `test`. A threshold chosen and measured on the same items looks better than it is.

### Step 3: Grade the Calibration Split with Each Judge Alone

```python
from autorubric import evaluate

dm_calib = await evaluate(calib, dm_grader, experiment_name="calib-jev")
llm_calib = await evaluate(calib, llm_grader, experiment_name="calib-gemini")
```

This step is where calibration costs money: the LLM grades every criterion of every calibration item, whichever criteria a threshold would later escalate. Everything after it makes no calls until the held-out check.

### Step 4: Sweep Thresholds and Pick One

```python
from autorubric import calibrate_escalation, compute_metrics

curve = calibrate_escalation(calib, dm_calib, llm_calib)  # thresholds 0.00, 0.02, ..., 1.00
print(curve.to_dataframe()[["threshold", "escalation_rate", "metric", "cost_usd"]])

best = curve.best(tolerance=0.005)
print(f"threshold={best.threshold}, escalation_rate={best.escalation_rate:.2f}")
print(f"LLM alone: {compute_metrics(llm_calib, calib).criterion_accuracy:.3f}")
```

Each point replays the cascade at one threshold and measures it: its `escalation_rate`, its `metric` (`criterion_accuracy` by default) and its estimated `cost_usd`. `best(tolerance=0.005)` returns the point that escalates least among those within half a point of the best accuracy, which is the "same accuracy for less money" choice. Add `max_escalation_rate=0.3` to cap the share of criteria that go to the LLM.

Before going further, read the diagnostics of the chosen point:

- `dm_accuracy_kept` should be close to the LLM's accuracy: Jev keeps these criteria.
- `fallback_accuracy_escalated` should beat `dm_accuracy_escalated`: on the deferred criteria, the LLM must do better than Jev would have.

If the second condition fails, escalating spends money without gaining accuracy, and Jev alone is the better deal. If the first fails, a cascade that keeps many criteria cannot reach the LLM's accuracy.

### Step 5: Check the Threshold on the Held-Out Split

```python
from autorubric import escalation_stats, replay_escalation

dm_test = await evaluate(test, dm_grader, experiment_name="test-jev")
llm_test = await evaluate(test, llm_grader, experiment_name="test-gemini")

hybrid = replay_escalation(dm_test, llm_test, best)  # the cascade's run, with no calls
print(compute_metrics(hybrid, test).summary())

stats = escalation_stats(hybrid, test)
llm_alone = compute_metrics(llm_test, test).criterion_accuracy
print(f"cascade {stats.metric:.3f} vs LLM alone {llm_alone:.3f}")
print(f"escalated {stats.escalation_rate:.0%}, estimated cost ${stats.cost_usd:.4f}")
```

`replay_escalation` builds the `EvalResult` the cascade would have produced on the test items. Its cost is an estimate, pro-rated from per-item totals. Compare it with the LLM-alone accuracy on the same items. Replaying `best` on the calibration items instead would warn with a `UserWarning`, since the point records a fingerprint of the items it was calibrated on.

### Step 6: Deploy the Cascade

```python
from autorubric import EscalationConfig

cascade = CriterionGrader(
    judge_model_config=jev,
    escalation=EscalationConfig(
        judges=gemini,  # judge_id "escalation", as in the LLM runs
        threshold=best.threshold,
        per_criterion=best.per_criterion,
    ),
    seed=SEED,
)

result = await evaluate(new_batch, cascade, experiment_name="weekly-cascade")
```

For each item, Jev answers every criterion in one request; the criteria whose judgment failed, that Jev abstained on, or that it answered with a confidence below the threshold go to Gemini. An escalated criterion's report has `escalated=True` and keeps Jev's vote with `superseded=True`; its verdict comes from Gemini alone. When a batch has labels, `escalation_stats(result, labelled_batch)` reports the same diagnostics as Step 4 on the live run, with exact costs.

## Key Takeaways

- **A cascade saves money; it is not meant to raise accuracy.** Judge it against the LLM alone on held-out items, at equal accuracy.
- **Calibrate on one split, measure on another.** `EscalationCurve.best` picks from many measurements on the same items, so its metric is optimistic.
- **Use `tolerance` to trade a little accuracy for fewer LLM calls**, and `max_escalation_rate` to cap the LLM share.
- **Match the seed and `judge_id`.** Grade the LLM runs with `JudgeSpec(gemini, "escalation")` and the cascade's `seed`, so replays equal the live cascade exactly.
- **Check both conditions:** Jev must be accurate where it keeps criteria, and the LLM must beat Jev where it defers.

## Going Further

- [Decision-Model Judges](decision-models.md): framings, confidence, what escalates, per-criterion thresholds
- [LLM Judges](llm-judges.md): configuring the fallback judge, and how the two judge kinds compare
- [Judge Validation](judge-validation.md): agreement metrics against human labels
- [Cost Optimization](cost-optimization.md): response caching and model choice
- [API Reference: Decision Models](../api/decision-models.md)

---

## Appendix: Complete Code

```python
"""Cheap first-pass grading: Jev for every criterion, Gemini for the uncertain ones."""

import asyncio

from autorubric import (
    DecisionModelConfig,
    EscalationConfig,
    LLMConfig,
    RubricDataset,
    calibrate_escalation,
    compute_metrics,
    escalation_stats,
    evaluate,
    replay_escalation,
)
from autorubric.graders import CriterionGrader, JudgeSpec

SEED = 7


async def main():
    jev = DecisionModelConfig(
        model="jev-latest", cache_enabled=True, input_cost_per_token=0.042e-6
    )
    gemini = LLMConfig(model="gemini/gemini-3-flash-preview", cache_enabled=True)
    dm_grader = CriterionGrader(judge_model_config=jev)
    llm_grader = CriterionGrader(judges=[JudgeSpec(gemini, "escalation")], seed=SEED)

    dataset = RubricDataset.from_file("short_answers.json")
    calib, test = dataset.split_train_test(n_train=len(dataset) // 2, seed=42)

    # Calibrate on one split (the LLM run is the paid part).
    dm_calib = await evaluate(calib, dm_grader, experiment_name="calib-jev")
    llm_calib = await evaluate(calib, llm_grader, experiment_name="calib-gemini")
    curve = calibrate_escalation(calib, dm_calib, llm_calib)
    print(curve.to_dataframe()[["threshold", "escalation_rate", "metric", "cost_usd"]])
    best = curve.best(tolerance=0.005)
    print(
        f"threshold={best.threshold}: kept {best.dm_accuracy_kept}, "
        f"deferred Jev {best.dm_accuracy_escalated} vs "
        f"Gemini {best.fallback_accuracy_escalated}"
    )

    # Measure the chosen threshold on the other split, without running the cascade.
    dm_test = await evaluate(test, dm_grader, experiment_name="test-jev")
    llm_test = await evaluate(test, llm_grader, experiment_name="test-gemini")
    hybrid = replay_escalation(dm_test, llm_test, best)
    stats = escalation_stats(hybrid, test)
    llm_alone = compute_metrics(llm_test, test).criterion_accuracy
    print(f"cascade {stats.metric} vs LLM alone {llm_alone}")
    print(f"escalated {stats.escalation_rate}, estimated cost {stats.cost_usd}")

    # The grader to deploy: same seed, and the bare LLMConfig is judge "escalation".
    cascade = CriterionGrader(
        judge_model_config=jev,
        escalation=EscalationConfig(
            judges=gemini, threshold=best.threshold, per_criterion=best.per_criterion
        ),
        seed=SEED,
    )
    return cascade


if __name__ == "__main__":
    asyncio.run(main())
```
