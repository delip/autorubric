# LLM Infrastructure

LLM client configuration, caching, and generation utilities.

## Overview

AutoRubric uses LiteLLM for multi-provider LLM support. The `LLMConfig` class provides centralized configuration, while `LLMClient` handles request execution with caching, rate limiting, and retry logic.

## Quick Example

```python
from autorubric import LLMConfig, LLMClient, generate

# Configuration
config = LLMConfig(
    model="openai/gpt-4.1-mini",
    temperature=0.0,
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

## YAML Configuration

```yaml
# llm_config.yaml
model: openai/gpt-4.1
temperature: 0.0
max_tokens: 1024
cache_enabled: true
cache_ttl: 3600
```

```python
config = LLMConfig.from_yaml("llm_config.yaml")
config.to_yaml("llm_config_backup.yaml")
```

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

Classify an exception raised while grading a single criterion into an [`ErrorCategory`](#errorcategory). The grading pipeline uses the result to route a failed judge call: `infrastructure` and `parse` errors are treated as abstentions (mapped to `CANNOT_ASSESS` / `na=True` and excluded from scoring under the default `SKIP` strategy), while `unknown` errors fall back to a conservative worst-case verdict.

::: autorubric.classify_grading_error
    options:
      show_source: true

---

## ErrorCategory

Type alias for the category of a grading failure: `Literal["infrastructure", "parse", "unknown"]`.

- `"infrastructure"`: API/network failure (timeout, connection, rate limit, server error). Not the submission's fault.
- `"parse"`: the judge responded but its output could not be parsed/validated into the expected schema. Also not the submission's fault.
- `"unknown"`: an unexpected error that does not fit the above categories.

::: autorubric.ErrorCategory
    options:
      show_source: true
