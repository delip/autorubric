# LLM Judges

An LLM judge is a language model that AutoRubric prompts to grade a rubric against a submission; each call returns a verdict, or a selected option, with a written explanation. By default it makes one call per criterion; it can instead grade an item's whole rubric in a single call (`llm_calls="per_item"`; see [Grading a whole rubric in one call](#grading-a-whole-rubric-in-one-call)). Through [LiteLLM](https://docs.litellm.ai/), AutoRubric reaches 100+ providers, hosted APIs and self-hosted servers alike. You can use an LLM judge on its own, as a member of an ensemble, or as the fallback in a confidence cascade that sends it only the criteria a decision model is unsure of.

LLM judges are one of AutoRubric's two kinds of judges; the other is the [decision model](decision-models.md). The [quickstart](../quickstart.md) and most recipes use LLM judges, and meta-rubric evaluation and rubric improvement accept no other kind. [Choosing a judge kind](#choosing-a-judge-kind) compares the two.

## What an LLM judge does

An LLM judge grades an item with **one call per criterion** by default. An item's calls run concurrently, as do the calls of every judge in an ensemble, up to any [rate limit](#rate-limits-and-retries) you set. This section describes that default; [Grading a whole rubric in one call](#grading-a-whole-rubric-in-one-call) describes the alternative, `llm_calls="per_item"`.

- **One criterion per prompt.** The system prompt defines `MET`, `UNMET` and `CANNOT_ASSESS` for positive and negative criteria or, for a multi-choice criterion, how to pick one option and when to pick the NA option, with rules and worked examples. The user prompt holds, in order: the rubric's [guidelines](#steering-an-llm-judge) when it has any, the criterion's `requirement`, the `query` that produced the submission and a reference submission when you pass them, and the submission. A binary prompt also marks the criterion positive or negative, by the sign of its weight; a multi-choice prompt numbers the options. The judge never sees the other criteria, a criterion's name or weight, or an option's value. Under `llm_calls="per_item"` the same rules are embedded, unchanged, in a single call for every criterion instead.
- **Structured output.** Each call requests JSON in a response format: `CriterionJudgment` (`criterion_status`, `explanation`) for binary criteria, `MultiChoiceJudgment` (`selected_option`, `explanation`) for multi-choice ones. The system prompt asks for a one- or two-sentence explanation that cites the submission.
- **A verdict with a reason.** The vote records the verdict and, as `reason`, the explanation. With [extended thinking](#extended-thinking), `reasoning` holds the provider's reasoning trace. `probabilities` and `confidence` are `None`: an LLM judge returns an answer, not a distribution.

A `{"thinking": ..., "output": ...}` submission reaches the judge as `<thinking>` and `<output>` sections, and the system prompt tells it to judge only the output unless a criterion concerns the reasoning. Setting `autorubric.debug = True` prints each call's prompts before it is sent.

Every call repeats the system prompt, about 1,600 tokens with OpenAI's tokenizers, and the submission. An item's input tokens therefore grow with the number of criteria times the length of the submission, and an ensemble multiplies them by its number of judges. A [decision model](decision-models.md) sends the submission once for the whole rubric instead.

`CriterionGrader` always returns an `EnsembleEvaluationReport`; a single judge is an ensemble of one, with `judge_id="default"`.

## Installation and provider keys

LLM judges need no extra: `pip install autorubric` installs LiteLLM. A judge's `model` is a LiteLLM model id, `provider/model`, such as `openai/gpt-5.6-luna`, `anthropic/claude-sonnet-4-5-20250929` or `gemini/gemini-3-flash-preview`. LiteLLM reads each provider's key from that provider's environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...), and `api_key=` overrides it for one config. [Supported providers](../quickstart.md#supported-providers) lists common providers and their variables; the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) has the rest.

When it is imported, AutoRubric loads the nearest `.env` file at or above the working directory, in scripts and notebooks alike. Variables already set in the environment take precedence.

Keys are not checked when a grader is built, since checking one takes a request. A rejected key (401), or a key, project or region without access to the model (403), fails the first call and the grade raises a `ValueError` that names the judge, as an unknown model name (404) does, and so does an account with no credits left (402) (see [Failures](#standalone-use)). OpenRouter is the exception for 403, which it answers when moderation flags an input: that is specific to the item, so it abstains. A missing key raises the same way where LiteLLM reports it as an authentication error or the OpenAI SDK refuses to build its client: OpenAI, OpenAI-compatible providers, Azure and Anthropic. For providers where LiteLLM reports a missing key as a connection error (Gemini, Vertex AI and Bedrock, for example), every call fails as an `infrastructure` error and abstains, so check those keys before a long run.

## Configuration

```python
from autorubric import LLMConfig

gpt = LLMConfig(
    model="openai/gpt-5.6-luna",  # the key comes from OPENAI_API_KEY
    cache_enabled=True,           # reproducible re-runs (see below)
    max_parallel_requests=10,     # concurrent calls to this provider (see below)
)
```

Every field except `model` has a default. [LLM Infrastructure](../api/llm.md) is the full reference for `LLMConfig`, `LLMClient` and `generate()`.

| Field | Default | Meaning |
|---|---|---|
| `model` | required | LiteLLM model id, e.g. `"openai/gpt-5.6-luna"` |
| `temperature` | `None` | Sampling temperature; `None` leaves it out of the request, so the provider's default applies, and any number, `0.0` included, is sent |
| `max_tokens` | `None` | Cap on response tokens, an integer of at least 1; `None` sends no cap |
| `top_p` | `None` | Nucleus sampling, a number in [0, 1] (`0.0` is sent); `None` sends none |
| `timeout` | `60.0` | Timeout of each attempt, in seconds |
| `max_retries` | `3` | Attempts AutoRubric makes per call, the first included |
| `retry_min_wait` / `retry_max_wait` | `1.0` / `60.0` | Bounds, in seconds, on the exponential backoff between attempts |
| `max_parallel_requests` | `None` | Concurrent calls per provider, shared by every LLM judge that sets it for that provider; `None` sets no limit |
| `cache_enabled` / `cache_dir` / `cache_ttl` | `False` / `".autorubric_cache"` / `None` | Disk response cache; `cache_ttl` is in seconds, and `None` never expires |
| `api_key` | `None` | Provider key; `None` lets LiteLLM read the provider's environment variable |
| `api_base` | `None` | Base URL of the endpoint, for proxies and self-hosted servers |
| `thinking` | `None` | Extended thinking: a level (`"low"`, `"medium"`, `"high"`, `"none"`), a token budget, or a `ThinkingConfig` |
| `prompt_caching` | `True` | Arrange per-criterion requests for the provider's prompt cache: what an item's calls share goes in its own message, marked where the provider needs it, and one call finishes before the rest. See [prompt caching](cost-optimization.md#step-5-enable-prompt-caching-anthropic) |
| `seed` | `None` | Sampling seed sent with every call; for a provider LiteLLM knows takes none, such as Anthropic or Gemini, the first call's grade raises `ValueError` unless `extra_params={"drop_params": True}` |
| `extra_headers` | `{}` | Additional HTTP headers sent with every call |
| `extra_params` | `{}` | Additional LiteLLM parameters sent with every call |
| `images` | `None` | Whether the model accepts images: `None` looks the model up in LiteLLM's model map the first time a grader is to send it an image, and `True` or `False` is your declaration, which wins over the map. Never sent, and read only for content holding images; see [Images in Prompts](../api/llm.md#images-in-prompts) |

### Temperature

Leaving `temperature` at `None` is recommended for reasoning models such as GPT-5.x and for Gemini 3, and Anthropic's extended thinking requires a temperature of 1, its default. A low temperature reduces run-to-run variation, but most providers do not guarantee identical outputs even at `0.0`. Up to v1.5.3 the default was `0.0`; see [Temperature](../api/llm.md#temperature).

### Extended thinking

`thinking` turns on the provider's reasoning mode through LiteLLM. A level is sent as `reasoning_effort`, which LiteLLM translates for each provider; a token budget, as an int or `ThinkingConfig(budget_tokens=...)`, is sent as an explicit thinking budget; `None`, the default, sends nothing. Set it only for models with a reasoning mode. On others, such as `openai/gpt-5.6-luna`, LiteLLM refuses every value, `"none"` included, so the first call's grade raises `ValueError` (see [Parameters LiteLLM refuses](#parameters-litellm-refuses)). The trace, when the provider returns one as text, is in each vote's `reasoning`. A trace in any other shape is dropped, and the verdict is kept. [Extended Thinking](extended-thinking.md) covers provider support and budgets.

### Cost

A call's cost is what LiteLLM's `completion_cost()` computes from the response and LiteLLM's price data; when LiteLLM cannot price the model (a self-hosted one, say), it is `None`. A report's `completion_cost` is the sum of the known costs of its successful calls, or `None` when that sum is zero or unknown, and its `token_usage` sums their tokens, Anthropic's prompt-cache reads and writes included (`cache_read_input_tokens`, `cache_creation_input_tokens`). A failed call adds nothing to either, even one the provider answered and billed (with an option that does not exist, say), so failures make both undercount. With `prompt_caching=True`, AutoRubric marks the system prompt, which repeats from call to call, for Anthropic's prompt cache. [Cost Optimization](cost-optimization.md) covers caching and model choice.

### Reproducibility

Two seeds do different jobs:

- `CriterionGrader(seed=...)` pins AutoRubric's own randomness: the order of a multi-choice criterion's options and each judge's few-shot examples. Without one, the grader draws a seed when it is built (`grader.seed`). See [Fixing Seeds](fixing-seeds.md).
- `LLMConfig.seed` is sent to the provider as its sampling seed. It changes neither option order nor few-shot selection.

Neither guarantees the same outputs on a re-run; the response cache does. With `cache_enabled=True` each response is stored on disk, and an identical call returns it without contacting the provider, with the usage and cost recorded when it was generated, so a cached re-run reports the first run's cost. The key is the model, the system and user prompts, the response format's name, `temperature`, `top_p`, `max_tokens`, `thinking` and `seed`, plus the `judge_id` of any judge other than a lone `judge_model_config` judge, so that judges polling one model keep their own answers. Any change to a prompt misses the cache, including a new grader `seed` where options are shuffled or examples drawn. `api_base`, `api_key` and `extra_params` are not in the key: pointing the same model id at another endpoint, or changing `extra_params`, re-reads stored responses, so give such runs their own `cache_dir`. Calls that raise and responses that do not parse are not cached, but a response that parses and cannot be used (one naming an option that does not exist, say) is, and a re-run replays its `parse` failure.

An `EvalRunner` experiment's manifest records each LLM judge's `temperature` and, when they are set, its `max_tokens`, `top_p`, `thinking` and `seed` ([Fixing Seeds](fixing-seeds.md) lists the keys).

### Rate limits and retries

`max_parallel_requests` caps concurrent calls per provider: the part of the model id before the first `/`, such as `openai` (the whole id when it has no `/`). The cap is shared by every LLM judge in the process that sets one for that provider, whatever its grader, and the strictest value wins. It applies within one event loop, so each `asyncio.run` has its own. A judge that leaves it `None` is not limited. In a batch, `EvalConfig.max_concurrent_items` bounds how many items are graded at once; see [Concurrency Tuning](batch-evaluation.md#concurrency-tuning).

AutoRubric retries a call that raises one of LiteLLM's rate-limit, service-unavailable, connection or timeout errors, making up to `max_retries` attempts with exponential backoff between `retry_min_wait` and `retry_max_wait` seconds. It does not retry other errors, such as an internal server error, or a response that does not parse, and LiteLLM decides which error a failure raises. A provider's SDK can also retry inside each attempt (LiteLLM's OpenAI client does by default), so one attempt can make several requests.

### YAML

`LLMConfig.from_yaml(path)` loads a config and `config.to_yaml(path)` saves one. Keys that are not `LLMConfig` fields go to `extra_params`, and `thinking` may be a level, a budget or a mapping of `ThinkingConfig` fields. `to_yaml` writes every field that is set, `api_key` included, so keep keys in the environment when you commit config files. See [Configuration Management](configuration-management.md).

## Standalone use

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader

grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-5.6-luna"))
report = await rubric.grade(to_grade=submission, grader=grader, query=prompt)  # one call per criterion (the default)

vote = report.report[0].votes[0]
print(vote.verdict, vote.probabilities, vote.confidence)
# CriterionVerdict.MET None None
print(vote.reason)
# The opening paragraph states the thesis: "Cities should fund protected bike lanes."
```

A binary criterion's votes are in `votes`, a multi-choice criterion's in `multi_choice_votes`; with shuffling on, a multi-choice vote also records the order its judge saw (`shuffle_order`). The criterion's `final_reason` joins the votes' reasons, each prefixed with its `judge_id` (a single judge's reads `"default: ..."`), except on a binary criterion every vote abstained on, where it reads `"All judges could not assess"`. [Working with Grading Explanations](explanations.md) puts the reasons to use.

**Failures.** A failed call fails only its own criterion, unless the judge is misconfigured (below). `classify_grading_error` sorts the exception. Every other LiteLLM API error (rate limit, timeout, connection, server error, a rejected request) is `infrastructure`; a response that does not parse into the response format (one that is not JSON, not a JSON object, has no content at all, such as a refusal, or content that is not text, such as a list of content parts, with or without [extended thinking](#extended-thinking); or a response with no reply in it at all), or names an option that does not exist, is `parse`. Both abstain: `CANNOT_ASSESS` for a binary criterion; for a multi-choice one, the NA option, or an abstention that selects no option when there is none. Any other exception is `unknown` and gets the worst case for the criterion's weight sign: UNMET for a positive weight, MET for a negative one, the lowest-scoring option for a multi-choice criterion. The category and message are recorded in `error`, e.g. `"infrastructure: ..."`. Under the default `CannotAssessStrategy.SKIP` abstentions are left out of the score; an item whose criteria all abstain has nothing left to score, so its `score` is `None`, never a fabricated 0.0. When every call for an item failed, nothing was judged: the item has no score under any strategy, its report's `error` begins `Every criterion's judgment failed:`, and an `EvalRunner` run counts it as failed. [Handling Uncertain Assessments](cannot-assess.md) covers the other strategies.

**A misconfigured judge raises.** Some errors say nothing about the item being graded and fail every call the judge makes: an invalid API key (`litellm.AuthenticationError`, 401), a missing one where LiteLLM can tell (an `AuthenticationError` for Anthropic; for OpenAI, OpenAI-compatible providers and Azure, the OpenAI SDK's refusal to build its client, which LiteLLM reports as an `InternalServerError` or, for Azure, an `APIError`), a key, project or region without access to the model (a 403, which LiteLLM reports as `PermissionDeniedError` for some providers and as another API error for others: an `APIError` for OpenAI-compatible providers, an `APIConnectionError` for Anthropic, a `BadRequestError` for Gemini; the provider's HTTP status in the exception's chain says it was a 403), an account with no credits left or a billing problem (a 402 Payment Required, which LiteLLM reports as an `APIError` for OpenAI-compatible providers and OpenRouter and as an `APIConnectionError` for Anthropic and Gemini; the raised message says the provider answered 402), an unknown model (`NotFoundError`, 404), and a parameter LiteLLM refuses for the model (`UnsupportedParamsError`). Abstaining on them would hide the problem, and in a panel the other judges would score every item while this one contributes nothing. So the grade raises a `ValueError` that names the judge, with LiteLLM's exception as its `__cause__`; the judge's other calls for the item are cancelled. `LLMClient` does not retry one either. `EvalRunner` records it as the item's error, and `fail_fast=True` stops at the first item. Some failures still abstain as `infrastructure`: OpenRouter's 403, which flags an input by moderation, a 400 that can be specific to one item, such as `ContextWindowExceededError` for an over-long submission or `ContentPolicyViolationError`, and a plain `BadRequestError`.

### Parameters LiteLLM refuses

LiteLLM decides on the client whether a model takes a call's parameters. A parameter it refuses makes it raise `UnsupportedParamsError` before it sends any request, so the judge's first call fails and the grade raises `ValueError` naming the judge, as for any [misconfigured judge](#standalone-use). This holds for a single judge, each panel member, and a cascade's escalation judges. The message carries LiteLLM's, which names the parameter and the two ways out:

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader

# LiteLLM 1.95.0 knows OpenAI reasoning models by the gpt-5 and o-series names, so it
# refuses reasoning_effort, which thinking="low" sends, for this model.
grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-6-luna", thinking="low"))
report = await rubric.grade(to_grade=submission, grader=grader, query=prompt)
# ValueError: Judge 'default' is misconfigured (UnsupportedParamsError), so every call it
# makes fails: litellm.UnsupportedParamsError: openai does not support parameters:
# ['reasoning_effort'], for model=gpt-6-luna. ...

# Send it anyway:
LLMConfig(
    model="openai/gpt-6-luna",
    thinking="low",
    extra_params={"allowed_openai_params": ["reasoning_effort"]},
)
# Or leave out whatever the model does not take (litellm.drop_params = True does it for
# every call):
LLMConfig(model="openai/gpt-6-luna", thinking="low", extra_params={"drop_params": True})
```

Other refusals in LiteLLM 1.95.0 include `thinking` for a model without a reasoning mode (`openai/gpt-4o`), `temperature=0.0` for GPT-5 and o-series models, `seed` for Anthropic and Gemini, and the structured output every judge call needs for `openai/gpt-4`. Some refuse a value rather than the parameter. `allowed_openai_params` does not make a GPT-5 model take `temperature=0.0`, so leave `temperature` unset there, or use `drop_params`.

## Ensembles

An ensemble is a list of `JudgeSpec`s, each a config, a `judge_id` and an optional `weight`, a positive number (default `1.0`):

```python
from autorubric import LLMConfig
from autorubric.graders import CriterionGrader, JudgeSpec

grader = CriterionGrader(
    judges=[
        JudgeSpec(LLMConfig(model="openai/gpt-5.6-luna"), "gpt"),
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
- **Few-shot examples.** Passing `training_data`, a `RubricDataset` with ground truth, adds graded examples to each prompt; `few_shot_config=FewShotConfig(...)` tunes them and does nothing on its own. By default a criterion gets 3 examples (`n_examples`), balanced across verdicts or options and drawn separately for each judge, with the grader's seed unless `FewShotConfig.seed` is set. Examples are matched to criteria by position in the training data's rubric, so train on the rubric you grade with: a graded criterion that differs from the training criterion at its position in requirement, criterion type or options gets no examples, and grading it warns (`UserWarning`) which criterion and why (see [Which Criteria Get Examples](../api/few-shot.md#which-criteria-get-examples)). With `include_reason=True`, an example also shows its item's written reason for the criterion (`DataItem.ground_truth_reasons`), when it has one. Their submissions lengthen every call. See [Few-Shot Calibration](few-shot-calibration.md) and [Few-Shot](../api/few-shot.md).
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
            JudgeSpec(LLMConfig(model="openai/gpt-5.6-luna"), "gpt"),
            JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
        ],
        threshold=0.72,
    ),
    aggregation="majority",  # combines the escalation judges' votes
    seed=7,
)
```

An escalation judge votes only on the escalated criteria, and is called as it would be in an otherwise identical grader without the cascade, with the same `seed` and `judge_id`: few-shot examples, system prompts and option shuffling apply to it as to any LLM judge. Called per criterion (the default), it gets one call per escalated criterion, with that grader's prompt for it; under [`llm_calls="per_item"`](#grading-a-whole-rubric-in-one-call), it makes that grader's one full-rubric call for an item with any escalated criterion and keeps only the escalated criteria's results. That is why a plain LLM run graded with `judges=escalation.judges`, the same `seed` and the same LLM settings can stand in for the cascade's LLM stage when you calibrate a threshold offline ([Offline calibration and replay](decision-models.md#offline-calibration-and-replay)). An escalation judge's `judge_scores` entry is `None`, since it votes only on the escalated criteria. [Cheap First-Pass Grading with Jev and an LLM Fallback](cascade-grading.md) calibrates and deploys a cascade end to end.

## Grading a whole rubric in one call

`CriterionGrader(llm_calls="per_item")` makes each LLM judge grade an item's whole rubric in one call instead of one call per criterion. It applies to every LLM judge of the grader — the judge of `judge_model_config`, every judge in `judges`, and a cascade's escalation judges — and never to a decision-model judge. The default, `llm_calls="per_criterion"`, is everything described above.

### Enabling it

A single judge:

```python
grader = CriterionGrader(
    judge_model_config=LLMConfig(model="openai/gpt-5.6-luna"),
    llm_calls="per_item",
)
report = await rubric.grade(to_grade=submission, grader=grader, query=prompt)  # one call for the whole rubric
```

A panel — each LLM judge makes one call per item; a decision-model judge in the same panel is unaffected:

```python
grader = CriterionGrader(
    judges=[
        JudgeSpec(DecisionModelConfig(model="jev-latest"), "jev"),
        JudgeSpec(LLMConfig(model="openai/gpt-5.6-luna"), "gpt"),
    ],
    llm_calls="per_item",
)
```

A cascade — an escalation judge makes one full-rubric call for an item on which anything escalated, and keeps only the escalated criteria's results:

```python
grader = CriterionGrader(
    judge_model_config=DecisionModelConfig(model="jev-latest"),
    escalation=EscalationConfig(judges=LLMConfig(model="openai/gpt-5.6-luna"), threshold=0.72),
    llm_calls="per_item",
)
```

`llm_calls="per_item"` cannot be combined with `binary_response_format` or `multi_choice_response_format`: those describe one criterion's judgment, and a `per_item` call answers every criterion of the rubric at once. An `EvalRunner` experiment records the setting in its manifest's `grader_config` as `"llm_calls": "per_item"`; a grader that calls per criterion records no `llm_calls` key.

### What the call looks like

The system prompt embeds the per-criterion system prompts verbatim, one per kind of criterion the rubric has, under a short preamble that tells the judge to apply each guide to each criterion of its kind separately and independently, and a closing section that asks for one JSON judgment per criterion. A rubric with only binary or only multi-choice criteria carries only that one guide, just as a per-criterion call for that kind never sees the other.

The user prompt lists every criterion of the effective rubric under an id, `c0`, `c1`, ... (the same id a decision model poses it under), each holding exactly the criterion block a per-criterion prompt would send it — its `<criterion_type>` and `<criterion>`, or its `<question>` and `<options>` — and nothing else about it: no name, no weight. A multi-choice criterion's options are shuffled by the same key a per-criterion call uses (the grader's seed, the submission, the criterion's index and the judge's id), so a criterion's option order is identical in both modes and `shuffle_order` is recorded the same way.

The judge answers with one structured judgment per criterion: `RubricJudgment`, an internal response format holding a list of per-criterion entries (a `criterion_id`, a `criterion_status` or `selected_option`, and an explanation). The grader maps that answer back to one `CriterionResult` per criterion, in rubric order, the same way it maps a decision model's answers.

### Tokens and cost

With N criteria, a submission (with its input and reference) of S tokens, a system prompt of P tokens, and about c tokens per criterion block, a per-criterion judge's input tokens per item and judge scale as N·(P + S + c); a `per_item` judge's scale as P_w + S + N·c, where P_w is the wrapped system prompt (the preamble, the guide for each kind of criterion present, and the closing format). For N = 10 and S = 1,000, that is roughly 26,700 input tokens per item and judge called per criterion, against roughly 3,400 (a rubric of one kind) to 5,000 (both kinds) under `per_item`. [Prompt caching](cost-optimization.md#step-5-enable-prompt-caching-anthropic) narrows the gap on input: a `per_criterion` prompt puts the submission before the criterion, so a provider can reuse the system prompt and submission across an item's calls. It cannot reduce output, and `per_criterion` writes more of it.

The call's token usage and cost ride on its first criterion's result, so a report's `token_usage` and `completion_cost` add up each judge's one call. A call in which some criteria fail alone is counted in full; a call that fails adds nothing, as a failed per-criterion call adds nothing.

### Latency

`per_item` cuts the number of requests N-fold, and input tokens by about (N − 1)·(P + S) per item and judge, which raises throughput under a requests-per-minute or tokens-per-minute limit. Wall-clock time for one item can move the other way: a `per_criterion` judge writes its N explanations in parallel calls, while a `per_item` call writes all N, one after another, inside a single response.

### `max_tokens` and `timeout`

Output grows with N: roughly 80-100 tokens per criterion (its id, verdict fields, and a one- or two-sentence explanation), plus any thinking budget. Leave `max_tokens` unset, or size it at about 100·N tokens plus the thinking budget; the `max_tokens=1024` used elsewhere in these docs covers about 10-12 criteria. A reply truncated at `max_tokens` is not valid JSON, which fails every criterion of the call. The default `timeout` of 60 seconds can also be too short once a thinking model has to write out every criterion's reasoning inside one response.

### Failure scoping

A criterion whose answer is missing, duplicated, or otherwise unusable fails alone, as an abstention (`CANNOT_ASSESS`, or the NA option), exactly as an unusable per-criterion answer does. A failed call fails every criterion of that call with the same error, sorted as a failed per-criterion call's error is ([Failures](#standalone-use)). An API error, or a reply that is not a valid `RubricJudgment` (one cut off at `max_tokens`, one that is not a JSON object, one with no content or with content that is not text, or a response with no reply at all, with or without [extended thinking](#one-thinking-trace-per-call)), is `infrastructure` or `parse`, so every criterion abstains; a configuration error, such as a rejected key, raises instead. Any other exception is `unknown` and gives every criterion the worst case for its weight sign. With a single judge, a failed call leaves the item with no score at all (its report's `error` reads `"Every criterion's judgment failed: ..."`). In a panel the other judges still grade the item, as they do when per-criterion calls fail, but only abstentions are set aside when votes are aggregated: an `unknown` failure's worst-case stand-ins count as votes on every criterion of the call.

### Few-shot examples

With `training_data`, an example under `per_item` is a whole training item instead of one criterion's submission: each of the judge's `n_examples` items is shown once, labelled on every graded criterion that poses the training rubric's question at its position. With `balance_verdicts=True` (the default), the items are chosen by greedy label coverage — repeatedly picking the item that adds the most (criterion, label) pairs not yet shown, until `n_examples` are picked — since balancing each criterion's labels on their own is impossible when the same k items must carry every criterion's labels.

### One thinking trace per call

A `per_item` call produces one thinking trace for the whole rubric. It is copied onto the `reasoning` of every criterion judged from a usable answer in that call (not onto a criterion that failed alone), so one thinking budget now covers the whole rubric instead of one per criterion. See [Extended Thinking](extended-thinking.md).

### The contamination caveat

The upstream `rubric` library, from which AutoRubric was forked, shipped a mode that graded a whole rubric in one call; AutoRubric dropped it at the fork, because criteria evaluations could shift each other within a shared context window. `llm_calls="per_item"` brings single-call grading back as an opt-in: the judge is told to judge each criterion as if it were the only one, and each criterion gets its own field in the structured answer, but that does not guarantee the verdicts match per-criterion grading. In the live comparison in [Holistic Rubric grading](single-call-grading.md), one call per item was cheaper and faster on both of two datasets and graded measurably stricter on one of them. Compare the two on your own labelled data before switching a production judge over; that recipe shows how.

### A naming note

`llm_calls` counts LLM calls per item. It is unrelated to `compute_metrics(per_item_metrics=...)`, a metrics-computation setting, and to per-item rubrics, where each dataset item carries its own `Rubric`.

## Choosing a judge kind

| | LLM judge | Decision model |
|---|---|---|
| Configuration | `LLMConfig` | `DecisionModelConfig` |
| Requests per item | One call per criterion by default, for each judge; one call for the whole rubric under `llm_calls="per_item"` | One request for the whole rubric |
| What comes back | A verdict or option with a written explanation | A probability distribution over each criterion's answers |
| Explanations | `reason` on every vote | None: an answered vote's `reason` is `None` |
| Probabilities and confidence | `None` | `probabilities` and `confidence` on every answered vote |
| What drives cost | By default, the system prompt (about 1,600 tokens) and the submission, once per criterion, plus each call's output; under `llm_calls="per_item"`, the submission and a wrapped system prompt once, plus each criterion and its judgment ([below](#grading-a-whole-rubric-in-one-call)) | One request's input tokens, with the submission sent once |
| Price | LiteLLM's price data; `None` when it cannot price the model | `input_cost_per_token`, which you set; `None` until you do |
| Steering and calibration | Rubric guidelines, few-shot examples, system prompts, extended thinking | Rubric guidelines, the framings and `decision_threshold`; in a cascade, the escalation threshold |
| Abstention | `CANNOT_ASSESS` (binary) or the NA option (multi-choice, guaranteed unless `auto_na_option=False`) | Only for criteria posed as a choice: binary ones under `binary_framing="choice"`, and multi-choice ones through the NA option, except ordinal ones under `ordinal_framing="score"` |
| Multi-choice option order | Shuffled per item, criterion and judge (`shuffle_options`) | The rubric's order |
| Determinism and caching | Outputs can vary between runs; the response cache replays them | Probabilities can vary slightly between runs; the response cache replays them |
| Few-shot examples | Yes | No: a grader whose judges are all decision models raises `ValueError` |
| Custom response formats | Yes, under `llm_calls="per_criterion"` only | No: `ValueError` with any decision-model judge |
| Meta-rubric evaluation and rubric improvement | Yes | No: `ValueError` |
| Role in a cascade | Escalation judge | First stage |
| Missing credentials | Not checked when the grader is built; the first call's grade raises `ValueError` for OpenAI, OpenAI-compatible providers, Azure and Anthropic, while providers whose missing key LiteLLM reports as a connection error (e.g. Gemini, Vertex AI, Bedrock) abstain on every call as `infrastructure` | `ValueError` when the grader is built |
| Providers and endpoints | 100+ providers through LiteLLM; proxies and self-hosted servers through `api_base` | TypeSafe's API, or any System One-compatible endpoint through `api_base`; needs `autorubric[typesafe]` |

An LLM judge fits when you need a written reason for each verdict, want to calibrate the judge with few-shot examples or your own prompts, need meta-rubric evaluation or rubric improvement, or want a particular provider's model. A decision model fits when the cost per item matters, since it sends the submission once for the whole rubric instead of once per criterion, and when you want a probability and a confidence for each verdict; [`llm_calls="per_item"`](#grading-a-whole-rubric-in-one-call) narrows that cost gap for an LLM judge without giving up its written explanations. Which kind is more accurate depends on your rubric and data: grade a labelled sample with each and compare them, as in [Validating Your Judge Against Human Labels](judge-validation.md). To combine the two, let a decision model take the first pass and escalate its uncertain criteria to an LLM, as in [Cheap First-Pass Grading with Jev and an LLM Fallback](cascade-grading.md).
