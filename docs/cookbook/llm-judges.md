# LLM Judges

An LLM judge is a language model that AutoRubric prompts once per criterion; each call returns a verdict, or a selected option, with a written explanation. Through [LiteLLM](https://docs.litellm.ai/), AutoRubric reaches 100+ providers, hosted APIs and self-hosted servers alike. You can use an LLM judge on its own, as a member of an ensemble, or as the fallback in a confidence cascade that sends it only the criteria a decision model is unsure of.

LLM judges are one of AutoRubric's two kinds of judges; the other is the [decision model](decision-models.md). The [quickstart](../quickstart.md) and most recipes use LLM judges, and meta-rubric evaluation and rubric improvement accept no other kind. [Choosing a judge kind](#choosing-a-judge-kind) compares the two.

## What an LLM judge does

An LLM judge grades an item with **one call per criterion**. An item's calls run concurrently, as do the calls of every judge in an ensemble, up to any [rate limit](#rate-limits-and-retries) you set.

- **One criterion per prompt.** The system prompt defines `MET`, `UNMET` and `CANNOT_ASSESS` for positive and negative criteria or, for a multi-choice criterion, how to pick one option and when to pick the NA option, with rules and worked examples. The user prompt holds, in order: the rubric's [guidelines](#steering-an-llm-judge) when it has any, the criterion's `requirement`, the `query` that produced the submission and a reference submission when you pass them, and the submission. A binary prompt also marks the criterion positive or negative, by the sign of its weight; a multi-choice prompt numbers the options. The judge never sees the other criteria, a criterion's name or weight, or an option's value.
- **Structured output.** Each call requests JSON in a response format: `CriterionJudgment` (`criterion_status`, `explanation`) for binary criteria, `MultiChoiceJudgment` (`selected_option`, `explanation`) for multi-choice ones. The system prompt asks for a one- or two-sentence explanation that cites the submission.
- **A verdict with a reason.** The vote records the verdict and, as `reason`, the explanation. With [extended thinking](#extended-thinking), `reasoning` holds the provider's reasoning trace. `probabilities` and `confidence` are `None`: an LLM judge returns an answer, not a distribution.

A `{"thinking": ..., "output": ...}` submission reaches the judge as `<thinking>` and `<output>` sections, and the system prompt tells it to judge only the output unless a criterion concerns the reasoning. Setting `autorubric.debug = True` prints each call's prompts before it is sent.

Every call repeats the system prompt, about 1,600 tokens with OpenAI's tokenizers, and the submission. An item's input tokens therefore grow with the number of criteria times the length of the submission, and an ensemble multiplies them by its number of judges. A [decision model](decision-models.md) sends the submission once for the whole rubric instead.

`CriterionGrader` always returns an `EnsembleEvaluationReport`; a single judge is an ensemble of one, with `judge_id="default"`.

## Installation and provider keys

LLM judges need no extra: `pip install autorubric` installs LiteLLM. A judge's `model` is a LiteLLM model id, `provider/model`, such as `openai/gpt-4.1-mini`, `anthropic/claude-sonnet-4-5-20250929` or `gemini/gemini-3-flash-preview`. LiteLLM reads each provider's key from that provider's environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...), and `api_key=` overrides it for one config. [Supported providers](../quickstart.md#supported-providers) lists common providers and their variables; the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) has the rest.

When it is imported, AutoRubric loads the nearest `.env` file at or above the working directory, in scripts and notebooks alike. Variables already set in the environment take precedence.

Keys are not checked when a grader is built. With a missing or rejected key every call fails as an `infrastructure` error, so every criterion abstains (see [Failures](#standalone-use)).

## Configuration

```python
from autorubric import LLMConfig

gpt = LLMConfig(
    model="openai/gpt-4.1-mini",  # the key comes from OPENAI_API_KEY
    cache_enabled=True,           # reproducible re-runs (see below)
    max_parallel_requests=10,     # concurrent calls to this provider (see below)
)
```

Every field except `model` has a default. [LLM Infrastructure](../api/llm.md) is the full reference for `LLMConfig`, `LLMClient` and `generate()`.

| Field | Default | Meaning |
|---|---|---|
| `model` | required | LiteLLM model id, e.g. `"openai/gpt-4.1-mini"` |
| `temperature` | `None` | Sampling temperature; `None` leaves it out of the request, so the provider's default applies, and any number, `0.0` included, is sent |
| `max_tokens` | `None` | Cap on response tokens; `None` sends no cap |
| `top_p` | `None` | Nucleus sampling; `None` sends none |
| `timeout` | `60.0` | Timeout of each attempt, in seconds |
| `max_retries` | `3` | Attempts AutoRubric makes per call, the first included |
| `retry_min_wait` / `retry_max_wait` | `1.0` / `60.0` | Bounds, in seconds, on the exponential backoff between attempts |
| `max_parallel_requests` | `None` | Concurrent calls per provider, shared by every LLM judge that sets it for that provider; `None` sets no limit |
| `cache_enabled` / `cache_dir` / `cache_ttl` | `False` / `".autorubric_cache"` / `None` | Disk response cache; `cache_ttl` is in seconds, and `None` never expires |
| `api_key` | `None` | Provider key; `None` lets LiteLLM read the provider's environment variable |
| `api_base` | `None` | Base URL of the endpoint, for proxies and self-hosted servers |
| `thinking` | `None` | Extended thinking: a level (`"low"`, `"medium"`, `"high"`, `"none"`), a token budget, or a `ThinkingConfig` |
| `prompt_caching` | `True` | Mark the system prompt for Anthropic's prompt cache (model ids starting with `anthropic/` or `claude`; others are unaffected) |
| `seed` | `None` | Sampling seed sent with every call; a provider that takes none, such as Anthropic or Gemini, fails every call unless `extra_params={"drop_params": True}` |
| `extra_headers` | `{}` | Additional HTTP headers sent with every call |
| `extra_params` | `{}` | Additional LiteLLM parameters sent with every call |

### Temperature

Leaving `temperature` at `None` is recommended for reasoning models such as GPT-5.x and for Gemini 3, and Anthropic's extended thinking requires a temperature of 1, its default. A low temperature reduces run-to-run variation, but most providers do not guarantee identical outputs even at `0.0`. Up to v1.5.3 the default was `0.0`; see [Temperature](../api/llm.md#temperature).

### Extended thinking

`thinking` turns on the provider's reasoning mode through LiteLLM. A level is sent as `reasoning_effort`, which LiteLLM translates for each provider; a token budget, as an int or `ThinkingConfig(budget_tokens=...)`, is sent as an explicit thinking budget; `None`, the default, sends nothing. Set it only for models with a reasoning mode: on others, such as `openai/gpt-4.1-mini`, LiteLLM rejects every value, `"none"` included, and every call fails as `infrastructure`. The trace, when the provider returns one, is in each vote's `reasoning`. [Extended Thinking](extended-thinking.md) covers provider support and budgets.

### Cost

A call's cost is what LiteLLM's `completion_cost()` computes from the response and LiteLLM's price data; when LiteLLM cannot price the model (a self-hosted one, say), it is `None`. A report's `completion_cost` is the sum of the known costs of its successful calls, or `None` when that sum is zero or unknown, and its `token_usage` sums their tokens, Anthropic's prompt-cache reads and writes included (`cache_read_input_tokens`, `cache_creation_input_tokens`). A failed call adds nothing to either, even one the provider answered and billed (with an option that does not exist, say), so failures make both undercount. With `prompt_caching=True`, AutoRubric marks the system prompt, which repeats from call to call, for Anthropic's prompt cache. [Cost Optimization](cost-optimization.md) covers caching and model choice.

### Reproducibility

Two seeds do different jobs:

- `CriterionGrader(seed=...)` pins AutoRubric's own randomness: the order of a multi-choice criterion's options and each judge's few-shot examples. Without one, the grader draws a seed when it is built (`grader.seed`). See [Fixing Seeds](fixing-seeds.md).
- `LLMConfig.seed` is sent to the provider as its sampling seed. It changes neither option order nor few-shot selection.

Neither guarantees the same outputs on a re-run; the response cache does. With `cache_enabled=True` each response is stored on disk, and an identical call returns it without contacting the provider, with the usage and cost recorded when it was generated, so a cached re-run reports the first run's cost. The key is the model, the system and user prompts, the response format's name, `temperature`, `top_p`, `max_tokens`, `thinking` and `seed`, plus the `judge_id` of any judge other than a lone `judge_model_config` judge, so that judges polling one model keep their own answers. Any change to a prompt misses the cache, including a new grader `seed` where options are shuffled or examples drawn. `api_base`, `api_key` and `extra_params` are not in the key: pointing the same model id at another endpoint, or changing `extra_params`, re-reads stored responses, so give such runs their own `cache_dir`. Calls that raise and responses that do not parse are not cached, but a response that parses and cannot be used (one naming an option that does not exist, say) is, and a re-run replays its `parse` failure.

### Rate limits and retries

`max_parallel_requests` caps concurrent calls per provider: the part of the model id before the first `/`, such as `openai` (the whole id when it has no `/`). The cap is shared by every LLM judge in the process that sets one for that provider, whatever its grader, and the strictest value wins. It applies within one event loop, so each `asyncio.run` has its own. A judge that leaves it `None` is not limited. In a batch, `EvalConfig.max_concurrent_items` bounds how many items are graded at once; see [Concurrency Tuning](batch-evaluation.md#concurrency-tuning).

AutoRubric retries a call that raises one of LiteLLM's rate-limit, service-unavailable, connection or timeout errors, making up to `max_retries` attempts with exponential backoff between `retry_min_wait` and `retry_max_wait` seconds. It does not retry other errors, such as an internal server error, or a response that does not parse, and LiteLLM decides which error a failure raises. A provider's SDK can also retry inside each attempt (LiteLLM's OpenAI client does by default), so one attempt can make several requests.

### YAML

`LLMConfig.from_yaml(path)` loads a config and `config.to_yaml(path)` saves one. Keys that are not `LLMConfig` fields go to `extra_params`, and `thinking` may be a level, a budget or a mapping of `ThinkingConfig` fields. `to_yaml` writes every field that is set, `api_key` included, so keep keys in the environment when you commit config files. See [Configuration Management](configuration-management.md).

## Standalone use

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader

grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"))
report = await rubric.grade(to_grade=submission, grader=grader, query=prompt)  # one call per criterion

vote = report.report[0].votes[0]
print(vote.verdict, vote.probabilities, vote.confidence)
# CriterionVerdict.MET None None
print(vote.reason)
# The opening paragraph states the thesis: "Cities should fund protected bike lanes."
```

A binary criterion's votes are in `votes`, a multi-choice criterion's in `multi_choice_votes`; with shuffling on, a multi-choice vote also records the order its judge saw (`shuffle_order`). The criterion's `final_reason` joins the votes' reasons, each prefixed with its `judge_id` (a single judge's reads `"default: ..."`), except on a binary criterion every vote abstained on, where it reads `"All judges could not assess"`. [Working with Grading Explanations](explanations.md) puts the reasons to use.

**Failures.** A failed call fails only its own criterion. `classify_grading_error` sorts the exception. Every LiteLLM API error (authentication, rate limit, timeout, connection, server error, rejected request or parameter) is `infrastructure`; a response that does not parse into the response format, or names an option that does not exist, is `parse`. Both abstain: `CANNOT_ASSESS` for a binary criterion; for a multi-choice one, the NA option, or an abstention that selects no option when there is none. Any other exception is `unknown`, including the one a response with no content (`None`) raises, and gets the worst case for the criterion's weight sign: UNMET for a positive weight, MET for a negative one, the lowest-scoring option for a multi-choice criterion. The category and message are recorded in `error`, e.g. `"infrastructure: ..."`. Under the default `CannotAssessStrategy.SKIP` abstentions are left out of the score; an item whose criteria all abstain has nothing left to score and gets 0.0, which its `cannot_assess_count` and its criteria's `error` tell apart from a genuine zero. [Handling Uncertain Assessments](cannot-assess.md) covers the other strategies.

## Ensembles

An ensemble is a list of `JudgeSpec`s, each a config, a `judge_id` and an optional `weight`, a positive number (default `1.0`):

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader, JudgeSpec

grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "gpt"),
        JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
        JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini"),
    ],
    aggregation="majority",
)
```

Every judge votes on every criterion. The `judge_id` names a judge's votes, its `judge_scores` entry and its per-judge metrics, and it is part of the random seed for the judge's option order and few-shot examples. Give each judge its own: judges that share one are conflated, which warns with a `FutureWarning` and will raise `ValueError` in the next major version ([Judge IDs](../api/ensemble.md#judge-ids)).

Three independent settings combine the votes:

| Setting | Criteria | Values (default first) |
|---|---|---|
| `aggregation` | binary | `"majority"`, `"weighted"`, `"unanimous"`, `"any"` |
| `ordinal_aggregation` | ordinal multi-choice | `"mean"`, `"median"`, `"weighted_mean"`, `"mode"`, `"min"`, `"max"` |
| `nominal_aggregation` | nominal multi-choice | `"mode"`, `"weighted_mode"`, `"unanimous"` |

Abstaining votes are set aside; a criterion abstains when all its votes do, or, under nominal `"unanimous"`, when the judges disagree and the criterion has an NA option. Ties go to the outcome that scores lowest for the criterion's weight sign. A multi-choice criterion's own `aggregation` field, when set, overrides the grader's setting for it. [Ensemble Judging](ensemble-judging.md) and [Aggregation Strategies](../api/ensemble.md#aggregation-strategies) explain the binary values, and [Multi-Choice Rubrics](multi-choice-rubrics.md#step-7-ensemble-aggregation-for-multi-choice) the ordinal and nominal ones. A decision model joins an ensemble as one more `JudgeSpec`; see [Mixed ensembles](decision-models.md#mixed-ensembles).

## Steering an LLM judge

- **Rubric guidelines.** Free text on the `Rubric` that applies to every criterion: grading conventions, definitions, the audience. It opens every user prompt in a `<guidelines>` block, which states that the criterion text governs and the guidelines clarify how to apply it. See [Rubric guidelines](../api/core-grading.md#rubric-guidelines).
- **Few-shot examples.** Passing `training_data`, a `RubricDataset` with ground truth, adds graded examples to each prompt; `few_shot_config=FewShotConfig(...)` tunes them and does nothing on its own. By default a criterion gets 3 examples (`n_examples`), balanced across verdicts or options and drawn separately for each judge, with the grader's seed unless `FewShotConfig.seed` is set. Examples are matched to criteria by position in the training data's rubric, so train on the rubric you grade with. With `include_reason=True`, an example also shows its item's written reason for the criterion (`DataItem.ground_truth_reasons`), when it has one. Their submissions lengthen every call. See [Few-Shot Calibration](few-shot-calibration.md) and [Few-Shot](../api/few-shot.md).
- **System prompts.** `system_prompt` replaces the binary system prompt and `multi_choice_system_prompt` the multi-choice one. A replacement is sent as written: it must itself define the verdicts, or how to pick an option, and it gets none of the few-shot instructions the defaults gain. The response format still shapes the output, and the guidelines, being in the user prompt, still reach the judge.
- **Option order and the NA option.** With `shuffle_options=True` (the default), each multi-choice prompt lists the options in an order derived from the grader's seed, the submission, the criterion's position and the `judge_id`, to counter position bias; the answer is mapped back to the rubric's option. With `auto_na_option=True` (the default), a multi-choice criterion without an NA option gets one, "Cannot assess / not applicable", appended to its options and shuffled with them, so the judge can abstain; with `auto_na_option=False` it must pick a scored option unless the rubric supplies an NA option. See [Multi-Choice Rubrics](multi-choice-rubrics.md) and [Position Bias Mitigation](../api/multi-choice.md#position-bias-mitigation).
- **Custom response formats.** `binary_response_format` and `multi_choice_response_format` swap `CriterionJudgment` and `MultiChoiceJudgment` for a compatible Pydantic model of your own, from which the grader reads the same fields. An `affected_criteria` field, a list of criterion numbers, is appended to the reason as an `[Affects: #1, #3]` tag, which meta-rubric evaluation relies on. Neither can be combined with a decision-model judge. See [`CriterionGrader`](../api/graders.md#criteriongrader).
- **Extended thinking.** `thinking` gives the judge a reasoning phase before it answers; see [Extended thinking](#extended-thinking) and the [Extended Thinking](extended-thinking.md) recipe.

## LLM judges in a cascade

In a [confidence cascade](decision-models.md#the-confidence-cascade), a decision model judges every criterion first, and LLM judges take the criteria it errored or abstained on or answered with low confidence. Escalation judges must be LLMs: `EscalationConfig(judges=...)` takes a bare `LLMConfig`, which becomes `JudgeSpec(config, "escalation")`, or a list of `JudgeSpec`s whose votes are combined with the grader's aggregation settings, and raises `ValueError` for a decision model.

```python
from autorubric import DecisionModelConfig, EscalationConfig, LLMConfig
from autorubric.graders import CriterionGrader, JudgeSpec

grader = CriterionGrader(
    judge_model_config=DecisionModelConfig(model="jev-latest"),
    escalation=EscalationConfig(
        judges=[
            JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini"),
            JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "gpt"),
            JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
        ],
        threshold=0.72,
    ),
    aggregation="majority",  # combines the escalation judges' votes
    seed=7,
)
```

An escalation judge is called only on the escalated criteria, and for each it gets the prompt it would get in an otherwise identical grader without the cascade, with the same `seed` and `judge_id`: few-shot examples, system prompts and option shuffling apply to it as to any LLM judge. That is why a plain LLM run graded with `judges=escalation.judges`, the same `seed` and the same LLM settings can stand in for the cascade's LLM stage when you calibrate a threshold offline ([Offline calibration and replay](decision-models.md#offline-calibration-and-replay)). An escalation judge's `judge_scores` entry is `None`, since it never judges a whole rubric. [Cheap First-Pass Grading with Jev and an LLM Fallback](cascade-grading.md) calibrates and deploys a cascade end to end.

## Choosing a judge kind

| | LLM judge | Decision model |
|---|---|---|
| Configuration | `LLMConfig` | `DecisionModelConfig` |
| Requests per item | One call per criterion, for each judge | One request for the whole rubric |
| What comes back | A verdict or option with a written explanation | A probability distribution over each criterion's answers |
| Explanations | `reason` on every vote | None: an answered vote's `reason` is `None` |
| Probabilities and confidence | `None` | `probabilities` and `confidence` on every answered vote |
| What drives cost | The system prompt (about 1,600 tokens) and the submission, once per criterion, plus each call's output | One request's input tokens, with the submission sent once |
| Price | LiteLLM's price data; `None` when it cannot price the model | `input_cost_per_token`, which you set; `None` until you do |
| Steering and calibration | Rubric guidelines, few-shot examples, system prompts, extended thinking | Rubric guidelines, the framings and `decision_threshold`; in a cascade, the escalation threshold |
| Abstention | `CANNOT_ASSESS` (binary) or the NA option (multi-choice, guaranteed unless `auto_na_option=False`) | Only for criteria posed as a choice: binary ones under `binary_framing="choice"`, and multi-choice ones through the NA option, except ordinal ones under `ordinal_framing="score"` |
| Multi-choice option order | Shuffled per item, criterion and judge (`shuffle_options`) | The rubric's order |
| Determinism and caching | Outputs can vary between runs; the response cache replays them | Probabilities can vary slightly between runs; the response cache replays them |
| Few-shot examples | Yes | No: a grader whose judges are all decision models raises `ValueError` |
| Custom response formats | Yes | No: `ValueError` with any decision-model judge |
| Meta-rubric evaluation and rubric improvement | Yes | No: `ValueError` |
| Role in a cascade | Escalation judge | First stage |
| Missing credentials | Not checked when the grader is built; every call fails as `infrastructure` | `ValueError` when the grader is built |
| Providers and endpoints | 100+ providers through LiteLLM; proxies and self-hosted servers through `api_base` | TypeSafe's API, or any System One-compatible endpoint through `api_base`; needs `autorubric[typesafe]` |

An LLM judge fits when you need a written reason for each verdict, want to calibrate the judge with few-shot examples or your own prompts, need meta-rubric evaluation or rubric improvement, or want a particular provider's model. A decision model fits when the cost per item matters, since it sends the submission once for the whole rubric instead of once per criterion, and when you want a probability and a confidence for each verdict. Which kind is more accurate depends on your rubric and data: grade a labelled sample with each and compare them, as in [Validating Your Judge Against Human Labels](judge-validation.md). To combine the two, let a decision model take the first pass and escalate its uncertain criteria to an LLM, as in [Cheap First-Pass Grading with Jev and an LLM Fallback](cascade-grading.md).
