# LLM Infrastructure

LLM client configuration, caching, and generation utilities.

## Overview

AutoRubric uses LiteLLM for multi-provider LLM support. The `LLMConfig` class provides centralized configuration, while `LLMClient` handles request execution with caching, rate limiting, and retry logic.

For a guided introduction see [LLM Judges](../cookbook/llm-judges.md).

## Quick Example

```python
from autorubric import LLMConfig, LLMClient, generate

# Configuration
config = LLMConfig(
    model="openai/gpt-4.1-mini",
    max_tokens=1024,
    cache_enabled=True,
    max_parallel_requests=10,
)

# Direct generation (standalone function)
result = await generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain quantum computing.",
    model="openai/gpt-4.1-mini",
)
print(result)

# Or use the client
client = LLMClient(config)
result = await client.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain quantum computing.",
)
```

## Provider Configuration

| Provider | Model Format | Environment Variable |
|----------|-------------|---------------------|
| OpenAI | `openai/gpt-4.1`, `openai/gpt-4.1-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| Azure | `azure/openai/gpt-4.1` | `AZURE_API_KEY`, `AZURE_API_BASE` |
| Groq | `groq/llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| Ollama | `ollama/qwen3:14b` | (local, no key needed) |

## Temperature

`LLMConfig.temperature` defaults to `None`: `temperature` is left out of the request, so the provider's own default applies. Many current models need or recommend this: reasoning models such as the GPT-5.x family accept only their default temperature, Gemini 3 recommends keeping its default of 1.0, and Anthropic requires temperature 1 when extended thinking is enabled.

An explicit value, including `0.0`, is sent unchanged. A `temperature=` keyword argument to `LLMClient.generate()` overrides the config for that call (`temperature=None` omits it for that call). The response cache keys on the temperature actually used for the request (including a per-call override), and an unset temperature has its own key.

```python
LLMConfig(model="openai/gpt-5.2")                        # temperature omitted: provider default
LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)  # temperature=0.0 sent
```

A low temperature reduces run-to-run variance, but most providers do not guarantee identical outputs even at `0.0`.

!!! warning "Default changed in v1.6.0"
    Up to and including v1.5.3, `temperature` defaulted to `0.0` and was always sent. A config that never sets it now samples at the provider's default, so its grades may differ from earlier runs, and responses cached by those runs are not reused (an unset temperature has its own cache key). To keep the old behaviour, set `temperature=0.0` explicitly (`temperature: 0.0` in YAML). Configs that already set `temperature` explicitly send the same value and keep their cached responses.

## Output Length and Nucleus Sampling

`max_tokens` and `top_p` also default to `None`, which leaves them out of the request so the provider's own limit and default apply. A value you set is sent unchanged, including `top_p=0.0`. `LLMConfig` refuses a value no provider can mean with a `ValueError`: a `max_tokens` that is not an integer of at least 1, or a `top_p` that is not a number between 0 and 1 (a `bool` is not taken for a number).

```python
LLMConfig(model="openai/gpt-4.1-mini", max_tokens=1024, top_p=0.0)  # both sent
LLMConfig(model="openai/gpt-4.1-mini", max_tokens=0)                 # ValueError
```

!!! note "Changed behaviour"
    Earlier releases sent `max_tokens` and `top_p` only when they were truthy, so `top_p=0.0` and `max_tokens=0` were silently left out of the request. `top_p=0.0` is now sent, and responses cached for it by earlier releases, which were sampled without it, are not reused (it has a new cache key). `max_tokens=0`, a negative `max_tokens` and a `top_p` outside [0, 1] are now refused. Configs that set other values send the same requests and keep their cached responses.

## YAML Configuration

```yaml
# llm_config.yaml
model: openai/gpt-4.1
temperature: 0.0  # optional; omit to use the provider default
max_tokens: 1024
cache_enabled: true
cache_ttl: 3600
```

```python
config = LLMConfig.from_yaml("llm_config.yaml")
config.to_yaml("llm_config_backup.yaml")
```

`thinking` can be a level (`thinking: high`), a token budget (`thinking: 32000`), or a `ThinkingConfig` written as a mapping:

```yaml
thinking:
  level: high
  budget_tokens: 9000
```

`to_yaml` writes a `ThinkingLevel` enum as its string value, so it loads back as the string `"high"`. That string is equal to `ThinkingLevel.HIGH` and gives the same thinking settings.

## Extended Thinking

Enable step-by-step reasoning for complex evaluations:

```python
# Level-based (cross-provider)
config = LLMConfig(
    model="anthropic/claude-sonnet-4-5-20250929",
    thinking="high",  # "low", "medium", "high", or "none"
)

# Token budget
config = LLMConfig(
    model="anthropic/claude-opus-4-5-20251101",
    thinking=32000,  # Explicit token budget
)
```

Supported providers: Anthropic, OpenAI (o-series), Gemini (2.5+), DeepSeek.

`thinking` is checked when the `LLMConfig` is built. It must be `None`, a `ThinkingLevel` or one of the strings `"none"`, `"low"`, `"medium"`, `"high"`, an integer token budget of at least -1, or a `ThinkingConfig`; anything else raises `ValueError` there, before any request. That includes an unknown level such as `"extreme"`, `True` or `False` (a `bool` is not a budget), a budget below -1, and a float such as `32000.0`. A budget of 0 or -1 is accepted because Gemini 2.5 reads 0 as thinking off and -1 as dynamic thinking. A budget smaller than a provider's minimum passes this check. Anthropic's minimum is 1,024 tokens, and it rejects each request with a smaller budget with a 400 (`BadRequestError`). A grade does not raise that error as a configuration error, so every call of such a judge abstains (`CANNOT_ASSESS`, or NA for a multi-choice criterion). Use at least 1,024 for Anthropic models. `ThinkingConfig` checks its own `level` and `budget_tokens` the same way, except that `level=None` is accepted next to a `budget_tokens`, which is sent without reading the level. Both check their values only when they are built, so a field assigned afterwards is not checked again.

## Response Caching

```python
config = LLMConfig(
    model="openai/gpt-4.1-mini",
    cache_enabled=True,
    cache_dir=".autorubric_cache",
    cache_ttl=3600,  # 1 hour
)

client = LLMClient(config)
client.clear_cache()
stats = client.cache_stats()
# {'size': 1024, 'count': 10, 'directory': '.autorubric_cache'}
```

## Images in Prompts

`LLMClient.generate` takes the user prompt as `Content`: a string, an `Image`, or a list of strings and images. A text prompt is sent as the message's content string, exactly as before. A prompt holding an image is sent as a list of OpenAI-style content parts in the prompt's order, a `{"type": "text", "text": ...}` part for each run of text and a `{"type": "image_url", "image_url": {"url": "data:<media type>;base64,..."}}` part for each image. LiteLLM translates these parts for providers with another image format (Anthropic, Gemini). The image is sent as its own bytes, neither resized nor re-encoded, so the provider's own limits on format, size and image count apply.

```python
from autorubric import Image, LLMClient, LLMConfig

client = LLMClient(LLMConfig(model="openai/gpt-4.1-mini"))
reply = await client.generate(
    system_prompt="You describe figures.",
    user_prompt=[Image("figures/fig1.png"), "What does this figure show?"],
)
```

The response cache keys a prompt holding an image by the SHA-256 of its content, its text and each image's bytes in order, so an identical prompt hits the cache and a changed image misses it. Text prompts keep their existing cache keys. With `prompt_caching=True`, the system prompt is marked for caching as usual, and the image parts follow it in the user message. `autorubric.debug = True` prints a marker in place of each image, `[image: <file name>]` for a file and `[image: <N> bytes]` for bytes.

A provider that refuses an image (a format it does not read, or a model without vision input) usually answers with a bad-request error. A grader routes it like any other failed call, so the criterion abstains (`CANNOT_ASSESS`, or NA for a multi-choice criterion) and the error is kept on its report. See the [Image Inputs](../cookbook/image-inputs.md) recipe for the formats providers accept.

A grader does not leave the model question to the provider. `CriterionGrader` sends an image to an LLM judge only when the judge is known to accept images, and `LLMConfig.images` says how that is known:

| `images` | Judge accepts images when |
|----------|---------------------------|
| `None` (default) | LiteLLM's model map lists the model as accepting images (`litellm.supports_vision(model=...)` answers `True`). The map is asked once per grader per model, the first time the grader is to send that model an image. The lookup sends no request; it reads the model map LiteLLM loaded at import, which LiteLLM fetches from its repository unless `LITELLM_LOCAL_MODEL_COST_MAP=True` is set (then the copy installed with LiteLLM is used), and prints and logs nothing. |
| `True` | Always: your declaration, for a vision model the map does not list (for example `ollama/llava`). |
| `False` | Never: your declaration, even for a model the map lists. |

Any other value raises `ValueError` when the config is built (`1` and `"yes"` are not declarations). A judge that is not known to accept images raises `ValueError` from the grade, naming the judge and the model, before any judge of the grader sends a request. `fill_ground_truth` raises it too, rather than leaving each item with an image out. `LLMClient.generate` itself does not check, so a direct call sends what it is given. Text-only grading never reads `images` or the map, and the field is never sent, so text requests and cache keys are unchanged. An experiment manifest records `images` for a judge only when it is set.

## Prompt Caching (Anthropic)

Reduce latency and cost on repeated calls. Enabled by default:

```python
config = LLMConfig(
    model="anthropic/claude-sonnet-4-5-20250929",
    prompt_caching=True,  # Default
)
```

---

## LLMConfig

Central configuration class for LLM calls.

::: autorubric.LLMConfig
    options:
      show_source: true
      members_order: source

---

## LLMClient

Async client for LLM generation with caching and rate limiting.

::: autorubric.LLMClient
    options:
      show_source: true
      members_order: source

---

## generate

Convenience function for one-off LLM generation.

::: autorubric.generate
    options:
      show_source: true

---

## GenerateResult

Result from LLM generation.

::: autorubric.GenerateResult
    options:
      show_source: true
      members_order: source

---

## ThinkingConfig

Configuration for extended thinking/reasoning.

::: autorubric.ThinkingConfig
    options:
      show_source: true
      members_order: source

---

## ThinkingLevel

Enum for thinking level presets.

::: autorubric.ThinkingLevel
    options:
      show_source: true

---

## ThinkingLevelLiteral

Type alias for thinking level strings.

::: autorubric.ThinkingLevelLiteral
    options:
      show_source: true

---

## ThinkingParam

Type alias for thinking parameter (level or budget).

::: autorubric.ThinkingParam
    options:
      show_source: true

---

## classify_grading_error

Classify an exception raised while grading a single criterion into an [`ErrorCategory`](#errorcategory). The grading pipeline uses the result to route a failed judge call: `infrastructure` and `parse` errors are treated as abstentions (mapped to `CANNOT_ASSESS` / `na=True` and excluded from scoring under the default `SKIP` strategy), while `unknown` errors fall back to a conservative worst-case verdict. These verdicts only stand in for failed calls: an item whose every criterion's judgment failed has no score, and its report's `error` begins `Every criterion's judgment failed:`. Configuration errors (LiteLLM's `UnsupportedParamsError`, 401, 403 and 404 errors from the provider or from TypeSafe, and the OpenAI SDK's refusal to build its client, e.g. with no OpenAI or Azure API key, which LiteLLM reports as an `InternalServerError`, or an `APIError` for Azure) still classify as `infrastructure`, but the grader never routes them: they fail every call the judge makes, so the grade raises a `ValueError` naming the judge, chained from the original exception. Item-specific 400s such as `ContextWindowExceededError` still abstain.

::: autorubric.classify_grading_error
    options:
      show_source: true

---

## ErrorCategory

Type alias for the category of a grading failure: `Literal["infrastructure", "parse", "unknown"]`.

- `"infrastructure"`: API/network failure (timeout, connection, rate limit, server error). Not the submission's fault. (A configuration error, e.g. an invalid API key, is not routed at all: `CriterionGrader` raises it.)
- `"parse"`: the judge responded but its output could not be parsed/validated into the expected schema: a reply that is not JSON, is not a JSON object, has no content (e.g. a refusal) or content that is not a string (e.g. a list of content parts), with or without a thinking trace, and a response with no choices at all. Also not the submission's fault. `LLMClient.generate` raises the same `ValueError` for a plain-text call (no `response_format`) whose reply has no content or content that is not a string, rather than returning it, and does not cache it.
- `"unknown"`: an unexpected error that does not fit the above categories.

::: autorubric.ErrorCategory
    options:
      show_source: true
