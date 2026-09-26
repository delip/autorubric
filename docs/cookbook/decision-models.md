# Decision-Model Judges

A decision model is a judge that answers typed questions with probabilities instead of generating text. AutoRubric supports TypeSafe's Jev and any other served model that speaks the same System One protocol. You can use one as a standalone judge, as a member of a mixed ensemble, or as the cheap first stage of a confidence cascade that sends only its uncertain criteria to an LLM.

Decision models are one of AutoRubric's two kinds of judges; the other is the [LLM judge](llm-judges.md). [Choosing a judge kind](llm-judges.md#choosing-a-judge-kind) compares them.

## What a decision model does

An LLM judge makes one call per criterion. A decision model grades an item with **one request for the whole rubric**:

- **Shared state, sent once.** The request's `state` holds the material to judge, and it is billed once per request: the submission, the prompt (`input`), a reference submission when the dataset has one, and the rubric's [guidelines](#rubric-guidelines-shared-context). A structured `{"thinking", "output"}` submission is sent as `thinking` and `output` fields instead of `submission`.
- **One typed question per criterion.** Each criterion becomes one question. The criterion's `requirement` and option labels are sent verbatim; only the structure around them depends on the [framing](#framings).
- **Probabilities, not text.** For each question the model returns a probability distribution over the answers. The vote records it as `probabilities`, derives a `confidence` from it, and has no explanation: `reason` is `None`.

Because the submission is sent once rather than once per criterion, a decision model costs far less per item than an LLM judge, and the gap grows with the number of criteria and the length of the submission.

Every grader still returns an `EnsembleEvaluationReport`, and scoring, `CannotAssessConfig`, length penalties, `evaluate()`/`EvalRunner` and `compute_metrics()` work unchanged.

## Installation

The TypeSafe SDK is an optional extra:

```bash
pip install 'autorubric[typesafe]'
```

Constructing a `DecisionModelConfig` does not need the SDK; building a grader with one does, and raises an `ImportError` naming the extra when it is missing.

## Configuration

```python
from autorubric import DecisionModelConfig

jev = DecisionModelConfig(
    model="jev-latest",             # api_key defaults to the TYPESAFE_API_KEY environment variable
    cache_enabled=True,             # reproducible re-runs (see below)
    input_cost_per_token=0.042e-6,  # your price per input token, in USD
)
```

Every field after `model` is keyword-only. Field names match `LLMConfig` wherever the concept is the same.

| Field | Default | Meaning |
|---|---|---|
| `model` | required | Model name sent with each request, e.g. `"jev-latest"` |
| `api_key` | `None` | Bearer token; `None` reads `TYPESAFE_API_KEY`. A missing or malformed key raises `ValueError` when the grader is built. The key is never recorded. |
| `api_base` | `None` | Base URL of a System One-compatible endpoint (requests go to `{api_base}/v1/systemone`). `None` reads `TYPESAFE_BASE_URL`, then falls back to `https://api.typesafe.ai`. |
| `timeout` | `60.0` | Per-request timeout in seconds |
| `max_retries` | `3` | Total attempts per request, the first included, as in `LLMConfig` |
| `max_parallel_requests` | `None` | Concurrent requests, shared by every decision-model config whose base URL has the same host |
| `cache_enabled` / `cache_dir` / `cache_ttl` | `False` / `".autorubric_cache"` / `None` | Disk response cache, as in `LLMConfig` |
| `extra_headers` | `{}` | Additional HTTP headers (not the ones the SDK sets itself, such as `Authorization`) |
| `binary_framing` | `"noul_framed"` | How binary criteria are posed |
| `ordinal_framing` | `"choice"` | How ordinal multi-choice criteria are posed |
| `decision_threshold` | `0.5` | P(MET) cut for the Noul framings |
| `input_cost_per_token` | `None` | Price in USD per input token |

### Self-hosted endpoints

Any served model that implements the System One protocol works. Point `api_base` at it:

```python
import os

my_dm = DecisionModelConfig(
    model="my-org/rubric-dm-7b",
    api_base="https://xyz.endpoints.huggingface.cloud",
    api_key=os.environ["HF_TOKEN"],
)
```

The base URL must be an absolute http(s) URL with no query, fragment or credentials; anything else raises `ValueError` when the grader is built. Requests honour the environment's TLS and proxy settings (`SSL_CERT_FILE`, `HTTPS_PROXY` and the like). An experiment's manifest records only the host of the base URL, never the key.

### Framings

**Binary criteria** (`binary_framing`):

| Value | Question | Verdict | Can abstain |
|---|---|---|---|
| `"noul_framed"` (default) | Yes/no. The requirement is wrapped in a fixed sentence, and the yes/no outcomes carry the same MET/UNMET definitions as the LLM judge's system prompt (the negative-criterion definitions for negative weights). | P(MET) vs `decision_threshold` | No |
| `"noul"` | Yes/no whose instructions are the bare requirement | P(MET) vs `decision_threshold` | No |
| `"choice"` | A choice among `MET`, `UNMET` and `CANNOT_ASSESS`, with the definitions of `"noul_framed"` | the selected option | Yes (`CANNOT_ASSESS`) |

**The tie rule.** Above `decision_threshold` the verdict is MET and below it UNMET. At exactly the threshold the verdict is the worst case for the criterion's weight sign (UNMET for a positive weight, MET for a negative one), the same tie rule as ensemble aggregation.

**Multi-choice criteria.** Nominal criteria are always posed as a choice. For ordinal criteria, `ordinal_framing` picks:

| Value | Question | Verdict | Can abstain |
|---|---|---|---|
| `"choice"` (default) | A choice over the option labels, the NA option included. Needs unique labels and at most 255 options. | the selected option | Yes (the NA option) |
| `"score"` | An ordered scale over the non-NA options (2 to 10 of them) | the expected level `s`, snapped to `round(s)` and clamped to the scale | No: the NA option is not offered |

Python's `round` rounds half to even, so an expected level exactly halfway between two levels snaps to the even one. Options are always offered in the rubric's order: `shuffle_options` does not apply to decision models. A criterion its framing cannot express (duplicate labels for a choice, too few or too many levels for a score) is left out of the item's request and fails alone as a `parse` error; the other criteria are still asked.

### Confidence

AutoRubric computes each vote's confidence itself, from the probability `p` of the outcome the judge selected among the `K` it was offered:

```
confidence = clamp((K * p - 1) / (K - 1), 0, 1)
```

It is 1.0 when `p` is 1 and 0.0 at chance (`p = 1/K`) or below. For a Noul verdict at the default `decision_threshold=0.5` it equals `2 * |p - 0.5|`; for a choice it equals the confidence TypeSafe reports. A selected outcome can have less than chance probability (a snapped Score level, or a Noul verdict under a threshold other than 0.5); its confidence is then 0.0. Computing confidence from the distribution gives every framing and every endpoint one number on one scale, which is what makes a single cascade threshold meaningful.

LLM votes have `probabilities=None` and `confidence=None`.

### Cost

A request's cost is its input tokens times `input_cost_per_token`. There is no built-in price table, because vendor prices change. With the default `None` the cost is unknown, and the decision model contributes nothing to a report's `completion_cost`: a standalone decision-model grader then reports `completion_cost=None`, and a cascade reports only its LLM part. Set the price whenever cost matters. The request's usage and cost are recorded once per item.

### Reproducibility

Decision-model answers are not bit-deterministic: repeating an identical request can return slightly different probabilities, so a verdict whose probability lies near `decision_threshold` can change between runs. With `cache_enabled=True`, re-running an experiment returns the stored answers, usage included. The cache key is the model, the resolved base URL, the state and the questions, plus the `judge_id` of any judge other than a lone `judge_model_config` judge, so that judges of one decision model keep their own answers. `decision_threshold` is not part of it, so changing the threshold re-reads cached answers instead of asking again. Failed requests are never cached.

## Standalone use

```python
from autorubric.graders import CriterionGrader

grader = CriterionGrader(judge_model_config=jev)
report = await rubric.grade(to_grade=submission, grader=grader, query=prompt)  # one request

vote = report.report[0].votes[0]
print(vote.verdict, vote.probabilities, vote.confidence, vote.reason)
# CriterionVerdict.MET {'MET': 0.83, 'UNMET': 0.17} 0.66 None
```

`probabilities` is keyed by verdict for binary criteria (`"MET"`, `"UNMET"`, and `"CANNOT_ASSESS"` for the `"choice"` framing), and by the original option index as a string (`"0"`, `"1"`, ...) for multi-choice criteria.

**Failures.** A failed request abstains every criterion of that item for that judge, and each records the `error`: `infrastructure` for authentication, rate-limit, timeout, connection and server errors, `parse` for a request the endpoint rejected. An answer that is missing or of the wrong type abstains only its own criterion, as a `parse` error. Under the default `CannotAssessStrategy.SKIP` abstentions are left out of the score.

## Mixed ensembles

A decision model is one more `JudgeSpec` in `judges=[...]`:

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader, JudgeSpec

grader = CriterionGrader(
    judges=[
        JudgeSpec(jev, "jev"),
        JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini"),
        JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "gpt"),
    ],
    aggregation="majority",
)
```

Votes are hard verdicts, aggregated like any ensemble's; probabilities are recorded, not fused. `final_reason` joins only the explanations that exist, so the decision model contributes none. A mixed ensemble is an ordinary full panel, so inter-judge agreement (`krippendorff_alpha`, `fleiss_kappa`) and per-judge metrics behave as for an all-LLM ensemble.

## The confidence cascade

A cascade lets the decision model judge every criterion and sends only the criteria it is unsure of to LLM judges:

```python
from autorubric import EscalationConfig, LLMConfig

grader = CriterionGrader(
    judge_model_config=jev,
    escalation=EscalationConfig(
        judges=LLMConfig(model="gemini/gemini-3-flash-preview"),  # or a list of JudgeSpec
        threshold=0.72,
        per_criterion={"factuality": 0.95},  # optional, by criterion name
    ),
    seed=7,
)
```

**Treat the cascade as a cost tool, not an accuracy booster.** It pays when the decision model is as accurate as the LLM on the criteria it is confident about, and the LLM is more accurate on the ones it defers. Then the cascade can match the LLM judge's accuracy for a fraction of its price. Whether both conditions hold depends on your data. [Measure them](#offline-calibration-and-replay) before you deploy one.

**Configuration rules.**

- The grader's one judge must be a decision model (`judge_model_config=DecisionModelConfig(...)`, or a single decision-model `JudgeSpec` in `judges`).
- Escalation judges must be LLMs. A bare `LLMConfig` becomes `[JudgeSpec(config, "escalation")]`; a list of `JudgeSpec` is aggregated with the grader's `aggregation`, `ordinal_aggregation` and `nominal_aggregation`.
- Thresholds lie in [0, 1]. `judge_id`s must be unique across the decision model and the escalation judges.
- `per_criterion` keys are criterion names; other and unnamed criteria use `threshold`. A name that matches no criterion of the rubrics being graded is warned about (once per `EvalRunner` run, or once per grader for direct `Rubric.grade` calls).

**What escalates.** A criterion is escalated when the decision model's vote:

- errored (any error category), or
- abstained (`CANNOT_ASSESS` or an NA option), or
- has a `confidence` below the criterion's threshold.

`threshold=0.0` escalates only errors and abstentions; `threshold=1.0` escalates everything short of certainty. Only escalated criteria are sent to the escalation judges, each with the same prompt it would get in an otherwise identical non-cascade grader with the same `seed` and `judge_id`.

**What the report looks like.**

```python
for cr in report.report:
    if cr.escalated:
        votes = cr.votes or cr.multi_choice_votes  # binary / multi-choice
        dm_vote = next(v for v in votes if v.superseded)
        print(cr.criterion.name, dm_vote.confidence, cr.final_verdict, cr.final_reason)
```

- A kept criterion has the decision model's vote as its only vote.
- An escalated criterion has `escalated=True`. Its vote list starts with the decision model's vote, marked `superseded=True`, followed by the escalation judges' votes. Its final verdict and `final_reason` come from the escalation judges' votes alone.
- The superseded vote is recorded, never aggregated, and never used as a fallback. If every escalation vote abstains or fails, the criterion abstains and its `error` says why.
- `judge_scores` holds the decision model's own score over the whole rubric (its superseded verdicts included) and `None` for every escalation judge, which never judges a whole rubric.

**Metrics on a cascade run.** `compute_metrics(result, dataset, per_judge=True)` measures:

- the decision model on every criterion, its superseded votes counting as its predictions;
- each escalation judge on the escalated subset only. Its `JudgeMetrics` has `coverage="escalated"` and `n_pairs` (the subset's size), its score-level fields are `None`, and `summary()` labels it "(escalated subset, N pairs)".

For inter-judge agreement, Fleiss' κ needs every judge to vote on a criterion, so it is `None` for a cascade: a cascade is not a panel. Krippendorff's α tolerates missing votes and keeps the superseded ones, so it measures agreement between the decision model and the LLM on the escalated criteria.

`escalation_stats(result, dataset)` summarizes a cascade run as an `EscalationPoint`:

| Field | Meaning |
|---|---|
| `escalation_rate` | Share of (item, criterion) pairs escalated |
| `dm_accuracy_kept` | Decision-model accuracy on the pairs it kept |
| `dm_accuracy_escalated` | Decision-model accuracy on the pairs it deferred for low confidence |
| `fallback_accuracy_escalated` | The escalation judges' accuracy on those same pairs |
| `metric` | The chosen metric over the whole run (default `criterion_accuracy`) |
| `cost_usd`, `compute_seconds` | Summed cost and grading time |

A cascade can only pay if `dm_accuracy_kept` is high and `fallback_accuracy_escalated` beats `dm_accuracy_escalated`. `escalation_stats` pools pairs, so it also works on datasets whose items have different rubrics. Accuracies are `None` when their subset has no labelled pairs.

## Offline calibration and replay

Choosing a threshold needs labelled data and no live cascade runs. Grade a labelled set once with the decision model alone and once with the LLM judge(s) alone, then sweep thresholds offline:

```python
from autorubric import calibrate_escalation, compute_metrics, evaluate, replay_escalation

SEED = 7
gemini = LLMConfig(model="gemini/gemini-3-flash-preview")
dm_grader = CriterionGrader(judge_model_config=jev)
llm_grader = CriterionGrader(judges=[JudgeSpec(gemini, "escalation")], seed=SEED)

calib, test = dataset.split_train_test(n_train=len(dataset) // 2, seed=42)
dm_calib = await evaluate(calib, dm_grader)
llm_calib = await evaluate(calib, llm_grader)

curve = calibrate_escalation(calib, dm_calib, llm_calib)  # no calls
best = curve.best(max_escalation_rate=0.5, tolerance=0.005)
print(curve.to_dataframe())

dm_test = await evaluate(test, dm_grader)
llm_test = await evaluate(test, llm_grader)
hybrid = replay_escalation(dm_test, llm_test, best)  # an EvalResult; no calls
print(compute_metrics(hybrid, test).summary())
```

- **`replay_escalation(dm_result, llm_result, escalation)`** builds, without a single call, the `EvalResult` a live cascade would have produced: each criterion keeps the decision model's recorded vote or is superseded by the LLM run's votes, by the live cascade's escalation rule. `escalation` is a threshold or an `EscalationPoint`. The result works with `compute_metrics`, `escalation_stats` and every other metric. Its cost and time are estimates, because per-criterion LLM cost is not recorded: each item's LLM cost is pro-rated by the share of its criteria that escalated. The length penalty is not re-applied.
- **`calibrate_escalation(dataset, dm_result, llm_result)`** replays the cascade at every threshold of a sweep (by default 0.00, 0.02, ..., 1.00) and returns an `EscalationCurve` of `EscalationPoint`s. `metric` picks what to optimize: the name of a `MetricsResult` attribute, or a function of the `MetricsResult`. With `per_criterion=True`, each named criterion with at least `min_pairs_per_criterion` labelled pairs gets a threshold of its own.
- **`EscalationCurve.best(max_escalation_rate=None, tolerance=0.0)`** returns the least-escalating point whose metric is within `tolerance` of the best metric under the escalation budget. `tolerance` is in the metric's units: `tolerance=0.005` trades up to half a point of accuracy for fewer LLM calls. This is the "same accuracy for less money" choice.

**Exactness.** A replay equals the live cascade exactly when the LLM run was graded the way the cascade grades escalated criteria: the same escalation judges with the same `judge_id`s (`"escalation"` for a bare `LLMConfig`) and the same `seed`, plus the same LLM settings (`shuffle_options`, system prompts, few-shot settings, `auto_na_option`, aggregation). With another `seed` or `judge_id`, shuffled multi-choice options and few-shot examples differ, and the replay matches the cascade only in distribution. Deploy the cascade with the seed you calibrated with:

```python
grader = CriterionGrader(
    judge_model_config=jev,
    escalation=EscalationConfig(
        judges=gemini, threshold=best.threshold, per_criterion=best.per_criterion
    ),
    seed=SEED,
)
```

**Calibrate on held-out data.** Every point of a curve is measured on the pairs the threshold is chosen from, so the metric of the point `best` picks is optimistic. Measure the chosen point on held-out items, as above. Each point carries a `calibration_fingerprint` of its calibration items, and `replay_escalation` warns with a `UserWarning` when you replay a point on the items it was calibrated on. The check needs the runs' submissions, which runs loaded with `EvalResult.from_experiment` do not record; with those, keep the splits apart yourself.

**Calibration costs one full LLM grading** of the calibration items: the LLM run needs a vote on every criterion, whichever criteria a threshold would escalate. Calibrate on a sample of the items the cascade is meant to grade.

See the [cookbook recipe](cascade-grading.md) for the full workflow.

## What decision models cannot do

A decision model returns probabilities, not text, so some features stay LLM-only:

- **Few-shot examples.** `training_data` and `few_shot_config` go to LLM judges only. A grader whose judges are all decision models raises `ValueError` when given them; in a mixed ensemble or a cascade they reach the LLM judges and never the decision model.
- **Custom response formats.** `binary_response_format` and `multi_choice_response_format` describe a generated judgment, so combining either with any decision-model judge raises `ValueError`.
- **Meta-rubric evaluation and rubric improvement.** Their judges must write text, so `evaluate_rubric_standalone`, `evaluate_rubric_in_context`, `improve_rubric`, `ImprovementRunner` and the improvement building blocks raise `ValueError` when given a decision model as a judge, as the evaluation or revision LLM, or in the grader they validate with.
- **LLM-only settings.** `system_prompt`, `multi_choice_system_prompt` and `shuffle_options` apply to LLM judges only. A grader whose judges are all decision models warns (`UserWarning`) when one of them differs from its default, since no judge would use it.

A decision model's calibration channels are the rubric's guidelines and, in a cascade, the escalation threshold.

## Rubric guidelines: shared context

Guidelines are free text that applies to every criterion of a rubric: grading conventions, definitions, audience, scale anchors. Every judge kind sees them.

```python
from autorubric import Criterion, Rubric

rubric = Rubric(
    [
        Criterion(name="thesis", weight=3, requirement="States a clear, arguable thesis"),
        Criterion(name="evidence", weight=2, requirement="Supports claims with cited evidence"),
    ],
    guidelines=(
        "Writers are English-language learners in grades 8-12; judge against that level. "
        "'Cited' means any attribution, not formal citation style."
    ),
)
```

- **Decision models** receive them once per request as `state["guidelines"]`, billed once like the rest of the state, so they cost almost nothing. The framed questions (`"noul_framed"`, binary `"choice"`) add one sentence telling the model to apply them, with the criterion text governing; the bare `"noul"` framing and multi-choice questions see them only through the state.
- **LLM judges** receive them in a `<guidelines>` block at the start of every per-criterion prompt.

See [Rubric guidelines](../api/core-grading.md#rubric-guidelines) for the file format and the full rules.
