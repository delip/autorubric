"""LLM calling infrastructure with multi-provider support.

This module provides a unified interface to 100+ LLM providers via LiteLLM,
with support for structured outputs, automatic retries, response caching,
and provider-specific features like extended thinking and prompt caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import sys
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import diskcache
import litellm
import openai
import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from autorubric.content import (
    Content,
    Image,
    _content_digest,
    _content_preview,
    _normalize_content,
)
from autorubric.rate_limit import RateLimitPool
from autorubric.types import _as_int

if TYPE_CHECKING:
    from autorubric.types import TokenUsage

# Load the nearest .env at or above the working directory, in scripts and notebooks alike;
# variables already set in the environment take precedence.
load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)

# Type variable for structured output
T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Grading error classification
# ============================================================================

ErrorCategory = Literal["infrastructure", "parse", "unknown"]
"""Category of a failure encountered while grading a single criterion.

- infrastructure: API/network failure (timeout, connection, rate limit, server error).
  Not the submission's fault; the judge never produced a usable response. (A configuration
  error, e.g. an invalid API key, is not routed at all: ``CriterionGrader`` raises it.)
- parse: the judge responded but its output could not be parsed/validated into the
  expected schema. Also not the submission's fault.
- unknown: an unexpected error that does not fit the above categories.
"""


def classify_grading_error(exc: BaseException) -> ErrorCategory:
    """Classify an exception raised while grading a criterion.

    Reuses the underlying OpenAI exception taxonomy that LiteLLM builds on: every
    transient API failure (``Timeout``, ``APIConnectionError``, ``RateLimitError``,
    ``ServiceUnavailableError``, ``InternalServerError``, status errors, etc.) subclasses
    ``openai.APIError`` (including ``litellm.APIError`` itself), while parse/validation
    failures (``json.JSONDecodeError`` -> ``ValueError``, ``pydantic.ValidationError``)
    do not. ``LLMClient.generate`` raises one of these for every structured reply it
    cannot use: one that is not JSON, not a JSON object (with or without a thinking
    trace), or that does not match the schema is a ``ValidationError`` or
    ``JSONDecodeError``. A reply with no content (e.g. a refusal) or whose content is not
    a string (e.g. a list of content parts) is a ``ValueError``, as is a response with no
    choices, structured or plain-text alike.

    Decision-model failures raised by the optional TypeSafe SDK (``typesafe-sdk``) are
    routed by the SDK's own exception types, ahead of the ``ValueError`` rule:

    - ``TypeSafeAPIResponseValidationError`` (the endpoint's answer did not validate),
      ``TypeSafeBadRequestError`` (400) and ``TypeSafeUnprocessableEntityError`` (422, e.g.
      a request over the model's input limit) are ``"parse"``: the endpoint answered, but
      no usable answer came back for the request that was built.
    - Every other ``TypeSafeError`` is ``"infrastructure"``: authentication, permission,
      not found, rate limit, server errors, and connection failures and timeouts (which
      subclass ``TypeSafeError`` directly, not ``TypeSafeAPIError``).

    Configuration errors are classified like any other API error (``"infrastructure"``),
    but ``CriterionGrader`` never routes one: an exception that says the judge is
    misconfigured (LiteLLM's ``UnsupportedParamsError``, a 401, 403 or 404 from the
    provider (but OpenRouter's 403, a moderation flag on the input) or from TypeSafe, or
    the OpenAI SDK's refusal to build its client, e.g. with no OpenAI or Azure API key,
    which LiteLLM reports as an ``InternalServerError`` or, for Azure, an ``APIError``)
    fails every call the judge makes, so the grade raises a ``ValueError`` naming the
    judge, chained from it, instead of abstaining (#40). Item-specific 400s
    (``ContextWindowExceededError``, ``ContentPolicyViolationError``) and a plain
    ``BadRequestError`` still abstain.

    Args:
        exc: The exception raised during a judge call.

    Returns:
        ``"infrastructure"`` for API/network errors, ``"parse"`` for JSON/validation
        errors, and ``"unknown"`` for anything else.
    """
    if isinstance(exc, openai.APIError):
        return "infrastructure"
    typesafe_category = _classify_typesafe_error(exc)
    if typesafe_category is not None:
        return typesafe_category
    if isinstance(exc, (ValidationError, ValueError)):
        return "parse"
    return "unknown"


def _is_configuration_error(exc: BaseException | None) -> bool:
    """Whether ``exc``, raised by a judge call, says the judge is misconfigured.

    Such an error says nothing about the item being graded: it fails every call the judge
    makes, so ``CriterionGrader`` raises it instead of turning it into an abstention, as it
    does with the failures ``classify_grading_error`` routes. The configuration errors are:

    - ``litellm.UnsupportedParamsError``: LiteLLM refuses a parameter for the model. It
      decides this on the client and raises before sending any request, so the judge's
      first call raises it.
    - ``openai.AuthenticationError`` (401: a missing or invalid API key),
      ``openai.PermissionDeniedError`` (403) and ``openai.NotFoundError`` (404: e.g. an
      unknown model), which LiteLLM's exceptions of the same names subclass.
    - Any other API error that is the provider's HTTP 403 answer (``_is_forbidden_answer``):
      a key, project or region without access to the model. LiteLLM reports few 403s as
      ``PermissionDeniedError`` (Bedrock's ``AccessDeniedException``, for one): an
      OpenAI-compatible provider's (OpenAI, Groq, DeepSeek) comes back as a plain
      ``APIError``, Anthropic's as an ``APIConnectionError``, Gemini's as a
      ``BadRequestError``. OpenRouter's 403 is not one: it flags an input by moderation,
      which is specific to the item.
    - The TypeSafe SDK's ``TypeSafeAuthenticationError``, ``TypeSafePermissionDeniedError``
      and ``TypeSafeNotFoundError`` (e.g. a base URL that names no System One endpoint),
      for decision-model judges.
    - The OpenAI SDK's refusal to build its client, before any request (e.g. no API key:
      "Missing credentials"), for OpenAI, OpenAI-compatible providers and Azure, which
      LiteLLM wraps in an ``InternalServerError`` (an ``APIError`` for Azure) with the SDK's
      bare ``openai.OpenAIError`` in its ``__cause__``/``__context__`` chain
      (``_is_sdk_client_refusal``).

    A provider for which LiteLLM reports a missing key as a connection error (e.g. Gemini,
    Vertex AI, Bedrock) is not recognized: its calls fail as ``infrastructure`` failures.

    ``litellm.BadRequestError`` (400) is not one, nor are its other subclasses: a 400 can
    be specific to an item (``ContextWindowExceededError`` for one over-long submission,
    ``ContentPolicyViolationError``), so it keeps abstaining as an ``infrastructure``
    failure. ``UnsupportedParamsError`` subclasses ``BadRequestError``, and is the one
    exception among them.

    Args:
        exc: The exception raised by a judge call, or None.
    """
    if exc is None:
        return False
    if _is_sdk_client_refusal(exc):
        return True
    if isinstance(
        exc,
        (
            litellm.UnsupportedParamsError,
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
        ),
    ):
        return True
    if _is_forbidden_answer(exc):
        return True
    sdk = sys.modules.get("typesafe_sdk")
    return sdk is not None and isinstance(
        exc,
        (
            sdk.TypeSafeAuthenticationError,
            sdk.TypeSafePermissionDeniedError,
            sdk.TypeSafeNotFoundError,
        ),
    )


# The host whose 403 answers flag an input by moderation, a failure specific to the item.
_MODERATING_HOST = "openrouter.ai"


def _is_forbidden_answer(exc: BaseException) -> bool:
    """Whether ``exc`` is an API error that is the provider's HTTP 403 answer to a call.

    LiteLLM maps a 403 to different exceptions for different providers, most of them
    without its status (an ``APIConnectionError`` for Anthropic carries 500), but it keeps
    the exception its HTTP client raised in the ``__cause__``/``__context__`` chain, and
    that one holds the provider's HTTP response (``response``): its status is what says the
    answer was a 403. A 403 from OpenRouter (``llm_provider`` ``"openrouter"``, or a response from
    ``openrouter.ai`` reached through another prefix) is left out: OpenRouter answers 403
    when moderation flags the input.
    """
    if not isinstance(exc, openai.APIError) or getattr(exc, "llm_provider", None) == "openrouter":
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if response is not None and getattr(response, "status_code", None) == 403:
            try:
                host = str(response.request.url.host)
            except (AttributeError, RuntimeError):  # a response built without its request
                host = ""
            return not (host == _MODERATING_HOST or host.endswith("." + _MODERATING_HOST))
        current = current.__cause__ or current.__context__
    return False


# The OpenAI SDK modules whose client constructors refuse a configuration that cannot work.
_SDK_CLIENT_MODULES = frozenset({"openai._client", "openai.lib.azure"})


def _is_sdk_client_refusal(exc: BaseException) -> bool:
    """Whether ``exc`` comes from the OpenAI SDK refusing to build its client.

    ``openai.OpenAI``/``openai.AsyncOpenAI`` and Azure's ``openai.AzureOpenAI``/
    ``openai.AsyncAzureOpenAI`` raise a bare ``openai.OpenAIError`` (not an
    ``openai.APIError``: no request was made) when their configuration cannot work, such as
    missing credentials or conflicting authentication options. LiteLLM builds the client for
    each call and wraps that error (an ``InternalServerError`` for OpenAI, an ``APIError``
    for Azure), so it is looked for along the ``__cause__``/``__context__`` chain. The SDK
    raises a bare ``OpenAIError`` elsewhere too, some of them after a request that can fail
    transiently (a workload identity's token exchange), so only one raised by a client
    constructor in ``openai._client`` or ``openai.lib.azure`` counts.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current) is openai.OpenAIError:
            tb = current.__traceback__
            while tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
            if (
                tb is not None
                and tb.tb_frame.f_code.co_name == "__init__"
                and tb.tb_frame.f_globals.get("__name__") in _SDK_CLIENT_MODULES
            ):
                return True
        current = current.__cause__ or current.__context__
    return False


def _classify_typesafe_error(exc: BaseException) -> ErrorCategory | None:
    """Category of a TypeSafe SDK exception, or None when ``exc`` is not one.

    The SDK is optional and is looked up, never imported: one of its exceptions can exist
    only after ``typesafe_sdk`` was imported, so this costs nothing when the SDK is absent
    or unused.
    """
    sdk = sys.modules.get("typesafe_sdk")
    if sdk is None or not isinstance(exc, sdk.TypeSafeError):
        return None
    if isinstance(
        exc,
        (
            sdk.TypeSafeAPIResponseValidationError,
            sdk.TypeSafeBadRequestError,
            sdk.TypeSafeUnprocessableEntityError,
        ),
    ):
        return "parse"
    return "infrastructure"


# ============================================================================
# Thinking/Reasoning Configuration
# ============================================================================


class ThinkingLevel(str, Enum):
    """Standardized thinking/reasoning effort levels across LLM providers.

    LiteLLM translates these to provider-specific parameters:
    - Anthropic: thinking={type, budget_tokens} with low→1024, medium→2048, high→4096
    - OpenAI: reasoning_effort parameter for o-series and GPT-5 models
    - Gemini: thinking configuration with similar token budgets
    - DeepSeek: Standardized via LiteLLM's reasoning_effort

    Higher levels mean more "thinking" tokens/steps, trading latency for quality.
    """

    NONE = "none"  # Disable thinking (where supported, e.g., Gemini)
    LOW = "low"  # Light reasoning: ~1024 tokens budget
    MEDIUM = "medium"  # Moderate reasoning: ~2048 tokens budget
    HIGH = "high"  # Deep reasoning: ~4096 tokens budget


# String literal type for convenience
ThinkingLevelLiteral = Literal["none", "low", "medium", "high"]


# The level strings, as error messages list them.
_THINKING_LEVELS = ", ".join(repr(level.value) for level in ThinkingLevel)

# Token budget mapping for providers that support explicit budgets
THINKING_LEVEL_BUDGETS: dict[str, int] = {
    "none": 0,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
}


@dataclass
class ThinkingConfig:
    """Detailed configuration for LLM thinking/reasoning.

    Provides a uniform interface across providers:
    - Anthropic: Extended thinking (claude-sonnet-4-5, claude-opus-4-5+)
    - OpenAI: Reasoning (o-series, GPT-5 models via openai/responses/ prefix)
    - Gemini: Thinking mode (2.5+, 3.0+ models)
    - DeepSeek: Reasoning content

    Attributes:
        level: High-level thinking effort. Used when budget_tokens is not set.
            Defaults to MEDIUM for a good balance of quality and latency.
        budget_tokens: Explicit token budget for thinking (provider-specific): None, or
            an integer >= -1, sent as ``thinking={"type": "enabled", "budget_tokens": ...}``.
            When set, overrides level. Recommended: 10000-50000 for complex tasks.
            Providers that don't support explicit budgets will map this to the
            nearest level. Gemini 2.5 reads 0 as thinking off and -1 as a budget the
            model sets itself (dynamic thinking); a provider with a minimum budget
            (Anthropic's is 1024) rejects a smaller one.

    Both fields are checked when the config is built: ``level`` must be a ``ThinkingLevel``
    or its string value, stored as the member, or None alongside a ``budget_tokens`` (a
    budget is sent without reading ``level``), and ``budget_tokens`` None or an integer
    >= -1 (numpy integers too, stored as ``int``; never a ``bool``). Anything else raises
    ``ValueError``.

    Examples:
        # Simple: use a thinking level
        ThinkingConfig(level=ThinkingLevel.HIGH)

        # Fine-grained: specify exact token budget
        ThinkingConfig(budget_tokens=32000)  # For complex reasoning tasks

        # Disable thinking
        ThinkingConfig(level=ThinkingLevel.NONE)
    """

    level: ThinkingLevel | ThinkingLevelLiteral = ThinkingLevel.MEDIUM
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate both fields, storing ``level`` as a ``ThinkingLevel`` member.

        Raises:
            ValueError: If ``level`` is not a ``ThinkingLevel`` or one of its string values
                (None is one only with a ``budget_tokens``), or ``budget_tokens`` is
                neither None nor an integer >= -1.
        """
        name = type(self).__name__
        if self.level is not None or self.budget_tokens is None:
            try:
                self.level = ThinkingLevel(self.level)
            except ValueError as e:
                raise ValueError(
                    f"{name}.level must be one of {_THINKING_LEVELS} or a ThinkingLevel "
                    f"(None only with a budget_tokens); got {self.level!r}"
                ) from e
        if self.budget_tokens is not None:
            budget_tokens = _as_int(self.budget_tokens)
            if budget_tokens is None or budget_tokens < -1:
                raise ValueError(
                    f"{name}.budget_tokens must be None or an integer >= -1; "
                    f"got {self.budget_tokens!r}"
                )
            self.budget_tokens = budget_tokens

    def get_effective_budget(self) -> int:
        """Get the effective token budget based on level or explicit budget."""
        if self.budget_tokens is not None:
            return self.budget_tokens
        return THINKING_LEVEL_BUDGETS.get(self.level.value, 2048)

    def get_reasoning_effort(self) -> str:
        """Get the reasoning_effort string for LiteLLM."""
        return self.level.value


# Union type for flexible thinking parameter in LLMConfig
ThinkingParam = ThinkingConfig | ThinkingLevel | ThinkingLevelLiteral | int | None
"""Type for the thinking parameter in LLMConfig.

Accepts:
- ThinkingConfig: Full configuration object
- ThinkingLevel: Enum value (e.g., ThinkingLevel.HIGH)
- str: Level as string ("low", "medium", "high", "none")
- int: Direct token budget (e.g., 32000), an integer >= -1 (never a bool)
- None: Disable thinking
"""

_THINKING_FORMS = (
    f"None, a ThinkingLevel or one of {_THINKING_LEVELS}, "
    "an integer token budget >= -1, or a ThinkingConfig"
)


def _normalize_thinking_param(
    thinking: ThinkingParam, name: str = "thinking"
) -> ThinkingConfig | None:
    """Convert various thinking parameter formats to ThinkingConfig.

    A ``ThinkingConfig`` is returned as it is: it checked its own fields when it was built.

    Raises:
        ValueError: If ``thinking`` is none of the ``ThinkingParam`` forms (a ``bool`` is
            not a budget), names an unknown level, or is a budget below -1. ``name`` names
            the setting in the message.
    """
    if thinking is None or isinstance(thinking, ThinkingConfig):
        return thinking
    try:
        if isinstance(thinking, str):  # a ThinkingLevel member is a str too
            return ThinkingConfig(level=thinking)
        budget_tokens = _as_int(thinking)
        if budget_tokens is not None:
            return ThinkingConfig(budget_tokens=budget_tokens)
    except ValueError as e:
        raise ValueError(f"{name} must be {_THINKING_FORMS}; got {thinking!r}") from e
    raise ValueError(f"{name} must be {_THINKING_FORMS}; got {thinking!r}")


@dataclass
class GenerateResult:
    """Result from LLM generation including content, thinking, and usage statistics.

    Attributes:
        content: The main response content from the LLM (raw string).
        thinking: The thinking/reasoning trace if thinking was enabled.
            None if thinking was not enabled or the provider doesn't support it.
        raw_response: The raw LiteLLM response object for advanced use cases.
        usage: Token usage statistics from this LLM call.
        cost: Completion cost in USD for this LLM call, calculated using
            LiteLLM's completion_cost() function. None if cost calculation fails.
        parsed: The parsed Pydantic model instance when response_format was provided.
            None if no response_format was used or parsing failed.
    """

    content: str
    thinking: str | None = None
    raw_response: Any = None
    usage: TokenUsage | None = None
    cost: float | None = None
    parsed: Any = None


def _extract_thinking_content(message: Any) -> str | None:
    """Extract thinking/reasoning content from LLM response message.

    LiteLLM standardizes reasoning content across providers:
    - `reasoning_content`: Unified field across all providers (preferred)
    - `thinking_blocks`: Anthropic-specific list of thinking blocks (fallback)
    - `thinking`: Legacy Anthropic field (fallback)

    A trace is text. LiteLLM types `reasoning_content` and `thinking` as strings and
    documents no other shape for them, so only a non-empty string is taken from each
    source (and from each thinking block); anything else is passed over to the next
    source. Injected into a response model's string-typed `reasoning` field, a non-string
    trace would make an otherwise usable reply fail validation.

    Args:
        message: The message object from response.choices[0].message

    Returns:
        The thinking/reasoning content as a string, or None if not present.
    """
    # Primary: LiteLLM's standardized reasoning_content field
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content:
        return reasoning_content

    # Fallback: Anthropic-specific thinking_blocks
    blocks = getattr(message, "thinking_blocks", None)
    if isinstance(blocks, list):
        thinking_parts = []
        for block in blocks:
            text = (
                block.get("thinking")
                if isinstance(block, dict)
                else getattr(block, "thinking", None)
            )
            if isinstance(text, str) and text:
                thinking_parts.append(text)
        if thinking_parts:
            return "\n".join(thinking_parts)

    # Fallback: Legacy thinking field
    thinking = getattr(message, "thinking", None)
    if isinstance(thinking, str) and thinking:
        return thinking

    return None


def _extract_usage_from_response(response: Any) -> TokenUsage:
    """Extract token usage from LiteLLM response.

    LiteLLM provides an OpenAI-compatible usage object:
    - prompt_tokens: Number of tokens in the prompt
    - completion_tokens: Number of tokens in the completion
    - total_tokens: Sum of prompt + completion tokens

    Additional fields may be present for specific providers:
    - cache_creation_input_tokens: Tokens used to create cache (Anthropic)
    - cache_read_input_tokens: Tokens read from cache (Anthropic)

    Args:
        response: The LiteLLM response object

    Returns:
        TokenUsage object with usage statistics. Returns zeros if usage not available.
    """
    # Import here to avoid circular import
    from autorubric.types import TokenUsage

    if not hasattr(response, "usage") or response.usage is None:
        return TokenUsage()

    usage = response.usage

    # Extract standard OpenAI-compatible fields
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    # Extract Anthropic prompt caching fields if present
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


def _calculate_completion_cost(response: Any) -> float | None:
    """Calculate completion cost using LiteLLM's completion_cost function.

    Uses LiteLLM's built-in cost calculation which has pricing data for
    all supported providers.

    Args:
        response: The LiteLLM response object

    Returns:
        Cost in USD, or None if cost calculation fails.
    """
    try:
        # LiteLLM's completion_cost accepts the response object directly
        cost = litellm.completion_cost(completion_response=response)
        return float(cost) if cost is not None else None
    except Exception as e:
        logger.debug(f"Could not calculate completion cost: {e}")
        return None


_debug_console = Console()


def _user_message(user_prompt: Content) -> dict[str, Any]:
    """The chat message that carries ``user_prompt`` to the model.

    Text is sent as the message's content, a plain string. A document (content holding an
    image) is sent as a list of OpenAI-style content parts in the document's order: a
    ``{"type": "text", "text": ...}`` part for each run of text and an
    ``{"type": "image_url", "image_url": {"url": ...}}`` part for each image, whose URL is
    the image's ``data:`` URL (``Image._data_url``). LiteLLM translates these parts for
    providers with another image format.
    """
    user = _normalize_content(user_prompt)
    if isinstance(user, str):
        return {"role": "user", "content": user}
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": part._data_url}}
            if isinstance(part, Image)
            else {"type": "text", "text": part}
            for part in user
        ],
    }


def _print_debug_prompt(system_prompt: str, user_prompt: Content, model: str) -> None:
    """Print the fully constructed LLM prompt using Rich formatting.

    The user prompt is shown as ``_content_preview`` renders it, with an ``[image: ...]``
    marker in place of each image.
    """
    _debug_console.print()
    _debug_console.rule(f"[bold cyan]DEBUG PROMPT — {model}[/bold cyan]")
    _debug_console.print(
        Panel(system_prompt, title="[bold green]System Prompt[/bold green]", border_style="green")
    )
    _debug_console.print(
        Panel(
            Syntax(_content_preview(user_prompt), "xml", theme="monokai", word_wrap=True),
            title="[bold yellow]User Prompt[/bold yellow]",
            border_style="yellow",
        )
    )
    _debug_console.rule("[bold cyan]END DEBUG PROMPT[/bold cyan]")
    _debug_console.print()


def _provider_response_format(response_format: type[BaseModel]) -> type[BaseModel] | dict[str, Any]:
    """Build the ``response_format`` payload sent to the provider for structured output.

    The judgment schemas carry an optional ``reasoning`` field that is never read from the
    model's output: it is injected after the fact from the provider's separate thinking
    channel (see ``LLMClient.generate``). Under OpenAI/Groq *strict* structured output, every
    property is forced into ``required``, so leaving ``reasoning`` in the schema makes models
    that follow the prompt — which never asks for ``reasoning`` — fail schema validation and
    abstain (observed for Llama and Qwen on Groq, where the verdict collapses to
    ``CANNOT_ASSESS``). Strip ``reasoning`` from what the provider must emit; parsing still
    uses the full Pydantic model, so reasoning injection when thinking is enabled is unaffected.

    Models without a ``reasoning`` field are returned unchanged. Any unexpected schema shape
    falls back to the original model so behaviour never regresses.
    """
    if "reasoning" not in getattr(response_format, "model_fields", {}):
        return response_format
    try:
        from litellm.utils import type_to_response_format_param

        param = type_to_response_format_param(response_format)
    except Exception:
        param = None
    # Unknown litellm shape — keep current behaviour (pass the model through).
    if not isinstance(param, dict):
        return response_format
    schema = param.get("json_schema", {}).get("schema")
    if not isinstance(schema, dict):
        return response_format
    schema.get("properties", {}).pop("reasoning", None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [key for key in required if key != "reasoning"]
    return param


class _LLMConfigDumper(yaml.SafeDumper):
    """SafeDumper that writes enums and paths as plain YAML scalars.

    ``asdict()`` leaves ``ThinkingLevel`` members (bare, or inside a ``ThinkingConfig``)
    and ``Path`` values in place, and ``yaml.SafeDumper`` refuses both. Writing their plain
    values keeps ``LLMConfig.to_yaml`` output loadable by ``yaml.safe_load``.
    """


_LLMConfigDumper.add_multi_representer(
    Enum, lambda dumper, value: dumper.represent_data(value.value)
)
_LLMConfigDumper.add_multi_representer(
    PurePath, lambda dumper, value: dumper.represent_str(str(value))
)


@dataclass
class LLMConfig:
    """Configuration for LLM calls.

    Attributes:
        model: Model identifier in LiteLLM format (e.g., "openai/gpt-5.2",
               "anthropic/claude-sonnet-4-5-20250929", "gemini/gemini-3-pro-preview",
               "ollama/qwen3:14b"). REQUIRED - no default.
               See LiteLLM docs for full list of supported models.
        temperature: Sampling temperature. None (default) omits it from the request so
            the provider's own default applies; this is recommended for reasoning models
            such as GPT-5.x and for Gemini 3, and required for Anthropic extended
            thinking. An explicit value, including 0.0, is sent unchanged. Temperature
            0.0 reduces but does not guarantee determinism on most providers.
            Note: the default changed from 0.0 to None in v1.6.0. To keep the previous
            behavior, pass ``temperature=0.0`` explicitly (it keeps the same cache keys).
        max_tokens: Maximum tokens in the response: None (default) omits it from the
            request, so the provider's limit applies; otherwise an integer >= 1, sent
            unchanged.
        top_p: Nucleus sampling parameter: None (default) omits it from the request, so the
            provider's default applies; otherwise a number in [0, 1], sent unchanged
            (including 0).
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for transient failures.
        retry_min_wait: Minimum wait between retries (seconds).
        retry_max_wait: Maximum wait between retries (seconds).
        max_parallel_requests: Maximum concurrent requests to this model's provider.
            When set, a global per-provider semaphore limits parallel requests. The limit
            applies within one event loop; each ``asyncio.run`` call has its own.
            None (default) means unlimited parallel requests.
        cache_enabled: Default caching behavior (can be overridden per-request).
        cache_dir: Directory for response cache.
        cache_ttl: Cache time-to-live in seconds (None = no expiration).
        api_key: Optional API key override (otherwise uses environment variables).
        api_base: Optional API base URL override.
        thinking: Enable thinking/reasoning mode (unified across providers). Accepts
            multiple formats:
            - ThinkingLevel enum: ThinkingLevel.HIGH, ThinkingLevel.MEDIUM, etc.
            - String: "low", "medium", "high", "none"
            - Int: Direct token budget (e.g., 32000), an integer >= -1 (see
              ``ThinkingConfig.budget_tokens``); a bool is not a budget
            - ThinkingConfig: Full configuration with level and/or budget_tokens
            - None: Disable thinking (default)

            Any other value raises ``ValueError`` when the config is built.

            Provider support:
            - Anthropic: Extended thinking (claude-sonnet-4-5, claude-opus-4-5+)
            - OpenAI: Reasoning for o-series and GPT-5 models
            - Gemini: Thinking mode (2.5+, 3.0+ models)
            - DeepSeek: Reasoning content
        prompt_caching: Mark the system prompt for Anthropic's prompt cache (default: True).
            Applies only to model ids that start with ``anthropic/`` or ``claude``: the
            system message is sent with ``cache_control: {"type": "ephemeral"}``. Prompt
            caching is generally available, so no beta header is added, and an
            ``anthropic-beta`` value in ``extra_headers`` is sent unchanged. Other model
            ids, Claude models served through Bedrock or Vertex AI included, are sent
            unchanged. Providers that cache long prompts on their own (OpenAI, DeepSeek) do
            so whatever this setting is.
        seed: Random seed for reproducible outputs (OpenAI, some other providers).
        extra_headers: Additional HTTP headers for provider-specific features.
        extra_params: Additional provider-specific parameters passed to LiteLLM.
        images: Whether the model accepts images in its prompts. None (default) looks
            the model up in LiteLLM's model map (the one it loaded at import; no request
            is sent) the first time a grader is to send this judge an image; ``True`` or
            ``False`` is your declaration and wins over the map. A judge whose model
            neither LiteLLM lists nor you declare as accepting images is never sent one:
            grading an item holding images refuses it before any request
            (``CriterionGrader``). Set ``images=True`` for a vision model LiteLLM does not
            map (e.g. a LLaVA or Qwen-VL model served through Ollama, vLLM or
            OpenRouter). Text-only grading never reads it, so it changes no request. Only
            ``None``, ``True`` and ``False`` are accepted.

    Examples:
        # Basic usage without thinking (provider-default temperature)
        config = LLMConfig(model="openai/gpt-5.2")

        # Explicit temperature
        config = LLMConfig(model="openai/gpt-4.1-mini", temperature=0.0)  # explicit; sent as-is

        # Enable thinking with a level
        config = LLMConfig(model="anthropic/claude-sonnet-4-5-20250929", thinking="high")
        config = LLMConfig(model="openai/responses/gpt-5-mini", thinking=ThinkingLevel.HIGH)

        # Enable thinking with explicit token budget
        config = LLMConfig(model="anthropic/claude-opus-4-5-20251101", thinking=32000)

        # Full control with ThinkingConfig
        config = LLMConfig(
            model="gemini/gemini-2.5-pro",
            thinking=ThinkingConfig(level=ThinkingLevel.HIGH, budget_tokens=50000)
        )
    """

    model: str  # REQUIRED - no default, must always be specified
    temperature: float | None = None  # None = provider default (not sent)
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 60.0
    max_parallel_requests: int | None = None
    cache_enabled: bool = False
    cache_dir: str | Path = ".autorubric_cache"
    cache_ttl: int | None = None  # None = no expiration
    api_key: str | None = None
    api_base: str | None = None

    # Thinking/Reasoning (unified across providers)
    thinking: ThinkingParam = None

    # Other provider-specific features
    prompt_caching: bool = True  # Enable prompt caching by default for supported models
    seed: int | None = None  # OpenAI reproducibility
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)

    # Whether the model accepts images: None = ask LiteLLM's model map (last, so that no
    # positional argument moves)
    images: bool | None = None

    def __post_init__(self) -> None:
        """Validate ``max_tokens``, ``top_p``, ``thinking`` and ``images``.

        The first three are sent with every request whenever they are not None, so a value
        no provider can mean is refused here rather than sent or silently dropped.
        ``images`` is never sent; it says whether the judge may be sent images, which only
        a ``bool`` says, so ``1`` or ``"yes"`` is refused rather than read as a
        declaration. A ``bool`` is never taken for a number. Any ``numbers.Integral``
        (numpy integers too) is a ``max_tokens``, stored as ``int``; a ``top_p`` that is a
        real number other than an ``int`` or ``float`` (a numpy float, say) is stored as
        ``float``.

        ``thinking`` is normalized with ``get_thinking_config()`` here, so a value it
        cannot normalize raises now instead of when a request is built, where every judge
        call would fail alike. It is kept as given (a level string stays a string), except
        that an integer budget that is not an ``int`` (a numpy integer) is stored as ``int``.

        Raises:
            ValueError: If ``max_tokens`` is neither None nor an integer >= 1, ``top_p``
                is neither None nor a real number in [0, 1] (NaN is not), or ``thinking``
                is not a ``ThinkingParam`` form: an unknown level, a ``bool``, a budget
                below -1, or a value of any other type; or ``images`` is neither None nor
                a ``bool`` (``1``, ``0`` and numpy booleans are not).
        """
        name = type(self).__name__
        if self.images is not None and not isinstance(self.images, bool):
            raise ValueError(f"{name}.images must be None, True or False; got {self.images!r}")
        if self.max_tokens is not None:
            max_tokens = _as_int(self.max_tokens)
            if max_tokens is None or max_tokens < 1:
                raise ValueError(
                    f"{name}.max_tokens must be None or an integer >= 1; got {self.max_tokens!r}"
                )
            self.max_tokens = max_tokens
        top_p = self.top_p
        if top_p is not None:
            is_real = isinstance(top_p, numbers.Real) and not isinstance(top_p, bool)
            if not (is_real and math.isfinite(top_p) and 0 <= top_p <= 1):
                raise ValueError(f"{name}.top_p must be None or a number in [0, 1]; got {top_p!r}")
            if type(top_p) not in (int, float):
                self.top_p = float(top_p)
        self.get_thinking_config()
        if not isinstance(self.thinking, (str, ThinkingConfig)) and self.thinking is not None:
            self.thinking = _as_int(self.thinking)

    def get_thinking_config(self) -> ThinkingConfig | None:
        """Get normalized thinking configuration.

        Raises:
            ValueError: If ``thinking`` is not a ``ThinkingParam`` form. ``__post_init__``
                refuses such a value, so only one assigned to ``thinking`` after
                construction can raise (fields assigned later are not checked again).
        """
        return _normalize_thinking_param(self.thinking, f"{type(self).__name__}.thinking")

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMConfig:
        """Load LLMConfig from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            LLMConfig instance with values from the YAML file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required fields are missing or invalid.

        Example YAML file (llm_config.yaml):
            model: openai/gpt-5.2
            # temperature omitted: the provider's default applies
            max_tokens: 1024
            cache_enabled: true
            cache_ttl: 3600

        ``thinking`` accepts a level string (``high``), an int token budget (``32000``),
        or a mapping of ``ThinkingConfig`` fields (``{level: high, budget_tokens: 9000}``),
        which is loaded as a ``ThinkingConfig``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LLM config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML config: expected dict, got {type(data).__name__}")

        if "model" not in data:
            raise ValueError("LLM config YAML must specify 'model' field")

        # Every dataclass field is a known key (derived, so new fields can't be missed);
        # anything else is a provider-specific param and goes to extra_params.
        known_fields = {f.name for f in fields(cls)}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        if extra:
            data.setdefault("extra_params", {}).update(extra)
            for k in extra:
                del data[k]

        # to_yaml writes a ThinkingConfig as a mapping of its fields; rebuild it. Level
        # strings and int budgets are already valid ThinkingParam values.
        thinking = data.get("thinking")
        if isinstance(thinking, dict):
            try:
                data["thinking"] = ThinkingConfig(**thinking)
            except TypeError as e:
                raise ValueError(f"Invalid 'thinking' mapping in LLM config YAML: {e}") from e

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save LLMConfig to a YAML file.

        The file holds only plain YAML (``yaml.safe_load`` reads it): enums are written as
        their values and paths as strings (so a ``Path`` ``cache_dir`` loads back as a
        ``str``). ``thinking`` is written so that ``from_yaml`` restores an equal value
        with the same ``get_thinking_config()``:

        - A level string or an int budget is written unchanged.
        - A ``ThinkingConfig`` is written as a mapping of its fields
          (``{level: high, budget_tokens: 9000}``) and loads back as a ``ThinkingConfig``.
        - A ``ThinkingLevel`` member is written as its string value (``high``) and loads
          back as that plain string, not the enum. Restoring the enum would need a custom
          YAML tag, which ``yaml.safe_load`` and hand-written files do not use, and it
          would add nothing: ``ThinkingLevel`` is a ``str`` enum, so the string compares
          equal to the member and normalizes to the same ``ThinkingConfig``.

        Args:
            path: Path to write YAML configuration file.
        """
        path = Path(path)
        data = asdict(self)

        # Remove None values and empty dicts for cleaner YAML
        data = {k: v for k, v in data.items() if v is not None and v != {}}

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, Dumper=_LLMConfigDumper, default_flow_style=False, sort_keys=False)


# LiteLLM's proxy route: it names no provider of its own, so any model-map entry fits it.
_ANY_PROVIDER = "litellm_proxy"


def _map_keys(model: str) -> tuple[str | None, list[str]]:
    """The provider ``model`` names, if any, and the keys LiteLLM's model map could hold it
    under, most specific first.

    A first path segment that LiteLLM lists as a provider (``litellm.provider_list``) is the
    provider, and a ``responses/`` segment after it (LiteLLM's Responses API route,
    ``openai/responses/gpt-5-mini``) is dropped, since the map keys the model, not the
    route. The keys are then the provider-prefixed name (``openai/gpt-4.1-mini``) and the
    bare one (``gpt-4.1-mini``), as LiteLLM keys its entries both ways. With no provider
    the model string is the one key.
    """
    head, sep, rest = model.partition("/")
    if not sep or head not in litellm.provider_list:
        return None, [model]
    if rest.startswith("responses/"):
        rest = rest[len("responses/") :]
    return head, [f"{head}/{rest}", rest]


def _same_provider(listed: object, provider: str | None) -> bool:
    """Whether a model-map entry whose ``litellm_provider`` is ``listed`` describes the model
    as ``provider`` serves it: an entry or a model naming no provider fits any, a proxy
    (``litellm_proxy``) fits every entry, and otherwise the names agree or one is a prefix
    of the other (``vertex_ai`` and ``vertex_ai-language-models``, ``azure_ai`` and
    ``azure``), much as LiteLLM's own lookup matches them."""
    if provider is None or listed is None or provider == _ANY_PROVIDER:
        return True
    if not isinstance(listed, str):
        return False
    return listed.startswith(provider) or provider.startswith(listed)


def _litellm_supports_vision(model: str) -> bool:
    """Whether LiteLLM's model map lists ``model`` as accepting images.

    It reads the model map LiteLLM loaded at import (``litellm.model_cost``) and nothing
    else: it sends no request and reads no file, so it never blocks, and it prints and logs
    nothing. (LiteLLM's own ``supports_vision`` is not used: for some providers, Ollama's
    among them, it asks the provider's server at run time.) The map is the one LiteLLM
    fetches from its repository at import unless ``LITELLM_LOCAL_MODEL_COST_MAP=True`` is
    set, so the answer follows the fetched map rather than the installed package's copy.

    The keys tried are ``_map_keys``'s, exactly and then ignoring case, as LiteLLM matches
    them; a ``responses/`` route is looked up by its model. The first entry found for the
    model's provider (``_same_provider``) that says whether the model accepts images
    decides: only ``supports_vision`` ``True`` confirms it. An entry for the provider that
    does not say leaves the next key to decide, whatever provider that key's entry names
    (``azure/gpt-4-turbo`` is silent, so ``gpt-4-turbo``, OpenAI's, decides), as in
    LiteLLM's own lookup. A model the map does not hold (an Ollama model, a
    model served through vLLM, a deployment name) is not confirmed, which is what
    ``LLMConfig(images=True)`` is for. Any error reading the map is not a confirmation
    either.
    """
    try:
        provider, keys = _map_keys(model)
        model_map = litellm.model_cost
        folded: dict[str, str] | None = None
        silent = False  # an entry for the model's provider was found that does not say
        for key in keys:
            if key not in model_map:
                if folded is None:
                    folded = {k.lower(): k for k in list(model_map)}
                key = folded.get(key.lower(), key)
            entry = model_map.get(key)
            if not isinstance(entry, dict):
                continue
            if not silent and not _same_provider(entry.get("litellm_provider"), provider):
                continue
            listed = entry.get("supports_vision")
            if isinstance(listed, bool):
                return listed
            silent = True
        return False
    except Exception:
        return False


def _open_response_cache(cache_dir: str | Path) -> diskcache.Cache:
    """Open the on-disk response cache at ``cache_dir``, creating the directory if needed.

    The one place the response store is opened: LLM responses (``LLMClient``) and
    decision-model responses (``autorubric.decision``) share it whenever their configs
    name the same ``cache_dir``. Keys never collide, because each client hashes its own
    request description with SHA-256.
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(directory=str(path))


class LLMClient:
    """Unified LLM client with retries, caching, and structured output support.

    Uses diskcache for efficient, thread-safe response caching.
    """

    def __init__(self, config: LLMConfig, *, cache_namespace: str | None = None):
        """Initialize LLM client.

        Args:
            config: LLMConfig instance. The model field is required.
            cache_namespace: Keeps this client's response-cache entries apart from those of
                other clients that send identical requests, such as several judges polling
                one model, whose answers are independent samples. It becomes part of the
                cache key; ``None`` (the default) leaves the key exactly as without it.

        Raises:
            ValueError: If config.model is not specified.
        """
        if not config.model:
            raise ValueError("LLMConfig.model is required and cannot be empty")

        self.config = config
        self._cache_namespace = cache_namespace
        self._cache: diskcache.Cache | None = None

        if self.config.cache_enabled:
            self._init_cache()

    def _init_cache(self) -> None:
        """Initialize diskcache instance."""
        self._cache = _open_response_cache(self.config.cache_dir)

    def _ensure_cache(self) -> diskcache.Cache:
        """Ensure cache is initialized, creating it if needed."""
        if self._cache is None:
            self._init_cache()
        return self._cache  # type: ignore[return-value]

    def _cache_key(
        self,
        model: str,
        system_prompt: str,
        user_prompt: Content,
        response_format: type | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> str:
        """Generate a unique cache key for the request.

        Includes sampling parameters so that different configurations
        (temperature, thinking, top_p, seed) produce distinct cache entries, and the
        client's ``cache_namespace`` when it has one.

        Args:
            model: Model identifier sent with the request.
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM. Text (a sequence of strings alone
                included, as their concatenation) enters the key as it is, the key it had
                before images existed; a document holding an image enters it as
                ``<content sha256=...>`` with its ``_content_digest``, so documents whose
                images hold the same bytes share a key wherever the images are read from.
            response_format: Optional Pydantic model class for structured output.
            overrides: The per-call overrides passed to ``generate``. A ``temperature``
                override replaces ``config.temperature`` in the key, so an override never
                reuses a response cached at a different temperature. None (provider
                default) is rendered as ``None``; explicit values keep the same key format
                as before, so existing cache entries still match. ``top_p`` and
                ``max_tokens`` are rendered as configured, except that a zero ``top_p``,
                which earlier releases keyed this way but did not send, gets a key of its
                own.
        """
        user = _normalize_content(user_prompt)
        user_field = user if isinstance(user, str) else f"<content sha256={_content_digest(user)}>"
        temperature = (overrides or {}).get("temperature", self.config.temperature)
        schema_name = response_format.__name__ if response_format else "str"
        thinking_config = self.config.get_thinking_config()
        thinking_str = str(thinking_config) if thinking_config else "none"
        # A zero top_p used to be left out of the request while keyed as configured, so the
        # responses cached under that key were sampled at the provider's default. It is
        # sent now and keyed apart from them; every other value keeps the key it had.
        top_p = self.config.top_p
        top_p_str = f"{top_p}:top_p_sent=True" if top_p is not None and top_p == 0 else top_p
        content = (
            f"{model}:{system_prompt}:{user_field}:{schema_name}"
            f":temp={temperature}"
            f":top_p={top_p_str}"
            f":max_tokens={self.config.max_tokens}"
            f":thinking={thinking_str}"
            f":seed={self.config.seed}"
        )
        if self._cache_namespace is not None:
            content += f":namespace={self._cache_namespace}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_retry_decorator(self) -> Any:
        """Build tenacity retry decorator from config."""
        # A configuration error fails every attempt alike (Anthropic's 403 comes back as an
        # APIConnectionError): retrying it only delays the failure.
        return retry(
            retry=retry_if_exception_type(
                (
                    litellm.RateLimitError,
                    litellm.ServiceUnavailableError,
                    litellm.APIConnectionError,
                    litellm.Timeout,
                )
            )
            & retry_if_exception(lambda exc: not _is_configuration_error(exc)),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            reraise=True,
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: Content,
        response_format: type[T] | None = None,
        use_cache: bool | None = None,
        return_thinking: bool = False,
        return_result: bool = False,
        **kwargs: Any,
    ) -> str | T | GenerateResult:
        """Generate LLM response with optional structured output.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM: text, or a document holding images,
                sent as content parts (see ``_user_message``). Its images are expected
                to hold their bytes already (``Image._loaded()``); one that does not is
                read from its file here.
            response_format: Optional Pydantic model class for structured output.
                When provided, LiteLLM uses the model's JSON schema to constrain
                the LLM output and returns a validated Pydantic instance.
            use_cache: Whether to use caching for this request.
                - None (default): Use config.cache_enabled setting
                - True: Force cache usage (initializes cache if needed)
                - False: Skip cache for this request
            return_thinking: If True and thinking is enabled, return a GenerateResult
                with both content and thinking. If False (default), only return content.
                Note: When response_format is provided, thinking is injected into the
                'reasoning' field if it exists, regardless of this setting.
            return_result: If True, always return a GenerateResult with full details
                including usage statistics and completion cost. This is useful when
                you need to track token usage. When True, takes precedence over the
                default return behavior.
            **kwargs: Override any LLMConfig parameters for this call. In particular,
                ``temperature`` accepts a float to send for this call, or None to omit it
                so the provider's default applies (regardless of ``config.temperature``).

        Returns:
            If return_result=True or return_thinking=True: GenerateResult with content,
                thinking, usage, cost, and parsed (if response_format was provided).
            If response_format is None: String response from the LLM.
            If response_format is provided: Validated Pydantic model instance.
                If thinking is enabled and the response_format has a 'reasoning'
                field, it will be populated with the model's thinking trace (a trace
                that is not a string is dropped).

        Raises:
            litellm.APIError: If all retries fail
            ValueError: If the response has no choices, or the reply has no content (e.g.
                a refusal) or content that is not a string (e.g. a list of content parts),
                with or without response_format; or if response_format is provided and the
                reply is not JSON (``json.JSONDecodeError``). Such a reply is not cached.
            pydantic.ValidationError: If response doesn't match schema, including a JSON
                reply that is not an object (with or without a thinking trace).
        """
        user = _normalize_content(user_prompt)

        # Determine caching behavior for this request
        should_cache = use_cache if use_cache is not None else self.config.cache_enabled

        # Effective temperature: a per-call override wins; None means "omit from the request".
        temperature: float | None = kwargs.get("temperature", self.config.temperature)

        # Check cache first
        cache_key: str | None = None
        if should_cache:
            cache = self._ensure_cache()
            cache_key = self._cache_key(
                self.config.model,
                system_prompt,
                user,
                response_format,
                kwargs,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key[:8]}...")
                return cached  # type: ignore[return-value]

        # Build request parameters
        model = kwargs.get("model", self.config.model)

        # Check if this is an Anthropic model that supports prompt caching
        # Anthropic requires cache_control on message content; other providers
        # handle caching automatically (OpenAI, Deepseek) or don't support it
        is_anthropic = model.startswith("anthropic/") or model.startswith("claude")
        use_prompt_caching = self.config.prompt_caching and is_anthropic
        if use_prompt_caching:
            # Anthropic requires cache_control on message content
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                _user_message(user),
            ]
        else:
            # Standard message format for other providers
            # OpenAI/Deepseek: Caching is automatic for prompts ≥1024 tokens
            messages = [
                {"role": "system", "content": system_prompt},
                _user_message(user),
            ]

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            # Only send temperature when set, so the provider default applies otherwise
            # (0.0 is a real value and is sent).
            **({"temperature": temperature} if temperature is not None else {}),
            "timeout": kwargs.get("timeout", self.config.timeout),
            **self.config.extra_params,
        }

        # Sent whenever set: LLMConfig refuses values no provider can mean, and a top_p of
        # 0 is a real setting.
        if self.config.max_tokens is not None:
            params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
        if self.config.top_p is not None:
            params["top_p"] = kwargs.get("top_p", self.config.top_p)
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.api_base:
            params["api_base"] = self.config.api_base

        # Thinking/Reasoning configuration (unified across providers)
        thinking_config = self.config.get_thinking_config()
        if thinking_config is not None:
            # Determine whether to use reasoning_effort or explicit thinking dict
            # Use explicit budget_tokens when specified for fine-grained control
            # Otherwise use reasoning_effort for better cross-provider compatibility
            if thinking_config.budget_tokens is not None:
                # Explicit token budget - use thinking dict (Anthropic/Gemini style)
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_config.budget_tokens,
                }
            else:
                # Level-based - use reasoning_effort for cross-provider support
                # LiteLLM translates this to provider-specific parameters
                params["reasoning_effort"] = thinking_config.get_reasoning_effort()

        # Extra headers configuration
        extra_headers = dict(self.config.extra_headers)
        if extra_headers:
            params["extra_headers"] = extra_headers

        if self.config.seed is not None:
            params["seed"] = self.config.seed

        # Enable structured output if Pydantic model provided. The schema sent to the
        # provider drops the post-hoc ``reasoning`` slot so strict-mode backends don't force
        # non-OpenAI models (e.g. Llama/Qwen on Groq) to emit it; see
        # _provider_response_format. Parsing below still uses the full Pydantic model.
        if response_format is not None:
            params["response_format"] = _provider_response_format(response_format)

        # Print debug prompt if debug mode is enabled
        import autorubric

        if autorubric.debug:
            _print_debug_prompt(system_prompt, user, model)

        # Make request with retries
        retry_decorator = self._get_retry_decorator()
        thinking_content: str | None = None
        refusal: Any = None
        raw_response: Any = None

        @retry_decorator
        async def _call() -> str:
            nonlocal thinking_content, refusal, raw_response
            response = await litellm.acompletion(**params)
            raw_response = response

            # A response with no choices (empty, ``None`` or missing) carries no reply at all;
            # like a reply with nothing to parse, it raises ``ValueError`` (``parse``), not the
            # ``IndexError`` of ``choices[0]``. ``ValueError`` is not retried.
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError("LLM reply has no choices")
            message = choices[0].message

            # Extract thinking/reasoning content (standardized across providers)
            # LiteLLM provides unified `reasoning_content` field
            thinking_content = _extract_thinking_content(message)
            # OpenAI-style structured-output refusals come with no content.
            refusal = getattr(message, "refusal", None)

            return message.content  # type: ignore[return-value]

        # Apply rate limiting if configured
        semaphore = await RateLimitPool.get_instance().get_semaphore(
            model, self.config.max_parallel_requests
        )
        if semaphore is not None:
            async with semaphore:
                response_content = await _call()
        else:
            response_content = await _call()

        # Extract usage and cost from the raw response
        usage = _extract_usage_from_response(raw_response)
        cost = _calculate_completion_cost(raw_response)

        # A reply with no content (e.g. a refusal), or whose content is not a string, is no
        # answer, structured or not: there is no JSON text to parse, and no text for a
        # plain-text call to return (it returns a ``str``, which callers treat as text). Like
        # a reply that is not JSON, it raises ``ValueError`` (``parse``), before anything is
        # cached. LiteLLM types the content as a string; anything else (e.g. a list of
        # content parts a provider transform left in place) would make ``json.loads`` raise
        # ``TypeError`` (``unknown``), or a caller's string operations fail.
        if response_content is None:
            detail = f" (refusal: {refusal})" if isinstance(refusal, str) and refusal else ""
            raise ValueError(f"LLM reply has no content{detail}")
        if not isinstance(response_content, str):
            raise ValueError(
                f"LLM reply content is not a string (got {type(response_content).__name__})"
            )

        # Parse structured output if requested
        parsed_response: T | None = None
        if response_format is not None:
            # LiteLLM returns JSON string when response_format is set
            # Parse it into the Pydantic model.
            data = json.loads(response_content)

            # Inject thinking content into the reasoning field if available. Only an object
            # can take it: any other JSON value is left as it is, to fail validation (#37).
            if (
                thinking_content
                and "reasoning" in response_format.model_fields
                and isinstance(data, dict)
            ):
                data["reasoning"] = thinking_content

            parsed_response = response_format.model_validate(data)

        # Determine what to return
        result: str | T | GenerateResult
        if return_result or return_thinking:
            # Return full GenerateResult with all details
            result = GenerateResult(
                content=response_content,
                thinking=thinking_content,
                raw_response=raw_response,
                usage=usage,
                cost=cost,
                parsed=parsed_response,
            )
        elif response_format is not None:
            # Return just the parsed Pydantic model
            result = parsed_response  # type: ignore[assignment]
        else:
            # Return just the string content
            result = response_content

        # Cache the response (cache the parsed object for structured outputs)
        if should_cache and cache_key:
            cache = self._ensure_cache()
            cache.set(
                cache_key,
                result,
                expire=self.config.cache_ttl,
            )
            logger.debug(f"Cached response for {cache_key[:8]}...")

        return result

    def clear_cache(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of entries cleared.
        """
        if self._cache is None:
            return 0
        count = len(self._cache)
        self._cache.clear()
        return count

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with 'size', 'count', and 'directory' keys.
        """
        if self._cache is None:
            return {"size": 0, "count": 0, "directory": None}
        return {
            "size": self._cache.volume(),
            "count": len(self._cache),
            "directory": str(self._cache.directory),
        }

    def close(self) -> None:
        """Close the on-disk cache, releasing its file handles.

        diskcache keeps the underlying SQLite database open for the lifetime of
        the cache object. Call this when done with the client so the cache
        directory can be removed (notably on Windows, which refuses to delete
        files that are still open). Safe to call when no cache was initialized.
        """
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# Convenience function for simple usage
async def generate(
    system_prompt: str,
    user_prompt: Content,
    model: str,
    response_format: type[T] | None = None,
    **kwargs: Any,
) -> str | T:
    """Simple one-shot generation function.

    For repeated calls, prefer creating an LLMClient instance.

    Args:
        system_prompt: System message for the LLM.
        user_prompt: User message for the LLM: text, or a document holding images.
        model: Model identifier (REQUIRED).
        response_format: Optional Pydantic model for structured output.
        **kwargs: Additional LLMConfig parameters.

    Example:
        # Simple string response
        response = await generate(
            "You are a helpful assistant.",
            "What is 2+2?",
            model="openai/gpt-5.2-mini"
        )

        # Structured output
        from pydantic import BaseModel

        class MathAnswer(BaseModel):
            result: int
            explanation: str

        answer = await generate(
            "You are a math tutor.",
            "What is 2+2?",
            model="openai/gpt-5.2-mini",
            response_format=MathAnswer
        )
        print(answer.result)  # 4
    """
    config = LLMConfig(model=model, **kwargs)
    client = LLMClient(config)
    return await client.generate(system_prompt, user_prompt, response_format=response_format)
