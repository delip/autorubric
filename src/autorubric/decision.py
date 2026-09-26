"""Decision-model judges: configuration, client, framing and answer mapping.

A *decision model* is a judge that does not generate text. It receives a ``state`` (the
material to judge) and a set of typed questions, and returns, for each question, a
probability distribution over that question's answer space. TypeSafe's Jev is the reference
model; any served endpoint that speaks the same System One protocol
(``POST {api_base}/v1/systemone``) works, for example a self-hosted model.

The transport is the official TypeSafe SDK (``typesafe-sdk``), an optional dependency
installed with ``pip install 'autorubric[typesafe]'``. It is imported only when a client is
built or a question object is created, so ``import autorubric`` and ``DecisionModelConfig``
never need it.

A decision model grades an item with one request for the whole rubric. The helpers below
are pure and synchronous: ``build_state`` builds the shared state (the rubric's guidelines
included), ``build_questions`` poses each criterion of the effective rubric as one question
(id ``question_id(criterion_idx)``), and ``answer_to_report`` turns each answer into a
``CriterionReport`` carrying the answer's ``probabilities`` and the ``selection_confidence``
of the selected outcome.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import weakref
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, get_args
from urllib.parse import urlsplit

import diskcache
from pydantic import BaseModel

from autorubric.llm import _open_response_cache
from autorubric.prompts import (
    CANNOT_ASSESS_DEFINITION,
    DECISION_MODEL_CRITERION_PREFIX,
    DECISION_MODEL_GUIDELINES_INSTRUCTION,
    DECISION_MODEL_REFERENCE_INSTRUCTION,
    DECISION_MODEL_TASK_INSTRUCTION,
    DECISION_MODEL_THINKING_OUTPUT_TASK_INSTRUCTION,
    MET_DEFINITION,
    NEGATIVE_MET_DEFINITION,
    NEGATIVE_UNMET_DEFINITION,
    UNMET_DEFINITION,
)
from autorubric.rate_limit import RateLimitPool
from autorubric.types import (
    Criterion,
    CriterionReport,
    CriterionVerdict,
    MultiChoiceVerdict,
    TokenUsage,
    _binary_worst_verdict,
)
from autorubric.utils import (
    _has_thinking_output_sections,
    _normalize_guidelines,
    parse_thinking_output,
)

if TYPE_CHECKING:
    import httpx2
    from typesafe_sdk import (
        Answer,
        AsyncTypeSafeClient,
        Choice,
        JSONContent,
        Noul,
        Question,
        Score,
        SystemOneResponse,
    )

logger = logging.getLogger(__name__)

BinaryFraming = Literal["noul", "noul_framed", "choice"]
"""How a binary criterion is posed to a decision model (see ``DecisionModelConfig``)."""

OrdinalFraming = Literal["choice", "score"]
"""How an ordinal multi-choice criterion is posed to a decision model."""

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


_SYSTEM_ONE_PATH = "/v1/systemone"
"""The System One endpoint's path, which the SDK appends to the base URL string."""

_DEFAULT_CONNECTIONS = 100
"""Requests a client keeps in flight per event loop when ``max_parallel_requests`` is
``None``: the connections of httpx's default pool, the one the SDK's own HTTP client has."""


def _shown_url(url: str) -> str:
    """``url`` as an error message may show it: never a URL that may hold credentials."""
    if "@" in url:
        return "a URL not repeated here, because it has an '@' and so may hold credentials"
    return repr(url)


def _check_base_url(url: str, source: str) -> None:
    """Raise ``ValueError`` unless every request can use ``url`` as its base URL.

    The SDK appends the endpoint path to the URL string, and its HTTP stack, ``httpx2``,
    parses the result on every request. Parsing it here once, with ``httpx2``, turns what
    that parse refuses (control characters, a host that is not a valid internationalized
    domain name or IPv4 address, a malformed port) into a construction error. The rest are
    URLs ``httpx2`` parses but no request could use:

    - Whitespace, such as a trailing newline or space, which ``httpx2`` refuses (line breaks,
      tabs) or percent-encodes, even in the host (spaces).
    - A URL that is not absolute http(s), which ``httpx2`` refuses only when it sends.
    - Credentials (``user:password@``), which the HTTP layer sends as Basic authentication
      in place of the bearer API key.
    - A query or fragment, even an empty one: the appended path would land in the query, or
      be dropped with the fragment.

    A URL with an ``@`` may hold credentials, so no message repeats it; ``httpx2``'s own
    messages never show credentials.
    """
    import httpx2

    shown = _shown_url(url)
    if any(char.isspace() for char in url):
        raise ValueError(f"{source} cannot contain whitespace; got {shown}")
    try:
        request_url = httpx2.URL(url + _SYSTEM_ONE_PATH)
    except httpx2.InvalidURL as exc:
        raise ValueError(f"{source} is not a URL HTTP requests can use: {exc}") from exc
    if request_url.scheme not in _DEFAULT_PORTS or not request_url.host:
        raise ValueError(
            f"{source} must be an absolute http(s) URL such as 'https://api.typesafe.ai'; "
            f"got {shown}"
        )
    if request_url.userinfo:
        raise ValueError(
            f"{source} cannot contain credentials (user:password@host): the HTTP layer would "
            "send them as Basic authentication in place of the API key. Authenticate with "
            "api_key instead (DecisionModelConfig.api_key or the TYPESAFE_API_KEY "
            "environment variable)."
        )
    if "?" in url or "#" in url:
        raise ValueError(
            f"{source} cannot have a query or fragment ('?' or '#'): requests go to "
            f"{{base URL}}{_SYSTEM_ONE_PATH}, and a path appended after a query or fragment "
            f"never reaches the endpoint; got {shown}"
        )


_SDK_OWNED_HEADERS = frozenset(
    {
        "accept",
        "authorization",
        "content-type",
        "user-agent",
        "x-typesafe-retry-count",
        "x-typesafe-runtime",
        "x-typesafe-sdk",
    }
)
"""The headers (lowercased) the TypeSafe SDK sets on every System One request, replacing
any value given for them: ``Authorization`` carries the bearer API key, and
``X-TypeSafe-Retry-Count`` is set on retries only. Header names are case-insensitive."""


def _check_extra_headers(headers: Mapping[str, str], source: str) -> None:
    """Raise ``ValueError`` for a header no request can carry as given.

    The SDK's HTTP stack checks a header only when it sends it: ``h11``, which writes each
    HTTP/1.1 request, refuses a name that is not a token and a value that is not an ASCII
    string or has a line break or surrounding whitespace (typically a secret read from a
    file with its trailing newline). Refused on every request, after every retry, the error
    would repeat the value into every failed report. So each header is checked here, once,
    with ``h11``. A header the SDK sets itself (``_SDK_OWNED_HEADERS``) is refused too: the
    SDK replaces the value given, so it would never be sent. Header values may be secrets,
    so messages name the header, never its value.
    """
    import h11

    for name, value in headers.items():
        try:
            h11.Request(
                method="POST", target=_SYSTEM_ONE_PATH, headers=[("Host", "h"), (name, value)]
            )
        except (h11.LocalProtocolError, TypeError, UnicodeError):
            raise ValueError(
                f"{source}[{name!r}] is not a header HTTP can send: its name must be a token "
                "(letters, digits or !#$%&'*+-.^_`|~) and its value an ASCII string with no "
                "line breaks or leading or trailing whitespace. The value is not shown."
            ) from None
        if name.lower() in _SDK_OWNED_HEADERS:
            hint = (
                " The API key is sent as the bearer token of the Authorization header: set "
                "it with api_key or the TYPESAFE_API_KEY environment variable."
                if name.lower() == "authorization"
                else ""
            )
            raise ValueError(
                f"{source} cannot set {name!r}: the TypeSafe SDK sets that header on every "
                f"request, replacing any value given.{hint}"
            )


def _url_host(url: str) -> str:
    """The host of ``url`` as a URL writes it: lowercased, with any non-default port.

    Credentials in the URL are never part of it.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:  # an IPv6 literal keeps its brackets
        host = f"[{host}]"
    port = parts.port
    if port is not None and port != _DEFAULT_PORTS.get(parts.scheme):
        host = f"{host}:{port}"
    return host


@dataclass
class DecisionModelConfig:
    """Configuration of a decision-model judge.

    A decision model grades an item with **one request for the whole rubric**: the
    submission (with the prompt, a reference submission and rubric guidelines when present)
    is sent once as shared ``state``, and each criterion becomes one typed question. It
    returns probabilities, not text, so its votes carry ``probabilities`` and ``confidence``
    and have no explanation (``reason`` is ``None``).

    Use it wherever a judge config is accepted, e.g.
    ``CriterionGrader(judge_model_config=DecisionModelConfig(model="jev-latest"))`` or
    ``JudgeSpec(DecisionModelConfig(...), "jev")`` in an ensemble. Building the grader
    builds the client, which needs the TypeSafe SDK (``pip install
    'autorubric[typesafe]'``); constructing this config does not. The client's HTTP
    requests honour the environment's TLS and proxy settings (``SSL_CERT_FILE``,
    ``HTTPS_PROXY`` and the like), and settings no HTTP client can be built with (e.g. a CA
    file that does not exist) fail the grader's construction with ``ValueError``. Field names match
    ``LLMConfig`` wherever the concept is the same. Every field after ``model`` is
    keyword-only.

    Few-shot examples (``FewShotConfig``, ``training_data``) apply to LLM judges only and
    are never sent to a decision model; a grader whose only judges are decision models
    rejects them. A decision model's calibration channels are rubric guidelines and, in a
    cascade, the escalation threshold.

    Decision-model answers are not bit-deterministic: repeating an identical request can
    return slightly different probabilities, so a verdict whose probability lies near
    ``decision_threshold`` can change between runs. Enable ``cache_enabled`` to make
    re-runs reproducible: a cached request returns the stored answers, usage included.

    Attributes:
        model: Model name or alias sent with each request, e.g. ``"jev-latest"`` or the
            model id a self-hosted endpoint serves. Required.
        api_key: API key, sent as a bearer token. ``None`` (default) reads the
            ``TYPESAFE_API_KEY`` environment variable, which a ``.env`` file may set, as for
            LLM provider keys. Surrounding whitespace is stripped, as the SDK strips it, so
            a key read with its trailing newline works. Building a client without either,
            or with a key the SDK rejects (anything but printable ASCII without
            whitespace), raises ``ValueError``, so a grader fails at construction rather
            than abstaining on every item. No message repeats the key, and it is never
            recorded.
        api_base: Base URL of a System One-compatible endpoint; requests go to
            ``{api_base}/v1/systemone``. ``None`` (default) reads ``TYPESAFE_BASE_URL``
            (surrounding whitespace stripped), then falls back to TypeSafe's API,
            ``https://api.typesafe.ai``. The URL is resolved and checked once, when a client
            (and so the grader) is built: the request URL is parsed with the SDK's HTTP
            stack, which refuses, e.g., a host that is not a valid internationalized domain
            name, and the URL must be absolute http(s), with no whitespace (a trailing
            newline included), query or fragment (the path is appended to the URL) and no
            credentials (they would replace the bearer ``api_key``). Anything else raises
            ``ValueError``, never repeating a URL that holds an ``@``, so a base URL no
            request could use fails when the grader is built. The resolved URL is part of
            the response cache key, and its host (with any non-default port) names the
            rate-limit bucket. The URL is not a secret: a failed
            request's error, recorded in the reports it fails, shows it with its path, as
            the SDK's logs do. Only its host goes into an experiment's manifest.
        timeout: Per-request timeout in seconds, applied to each HTTP operation, as in
            ``LLMConfig``. It also sets the retry budget (see ``max_retries``). A request's
            wait for its turn (see ``max_parallel_requests``) is not timed.
        max_retries: Total attempts per request, including the first, as in ``LLMConfig``
            (which stops after that many attempts). Rate limits (429), request timeouts
            (408), server errors (5xx), connection failures and timeouts are retried with
            exponential backoff, honouring the endpoint's ``Retry-After``. Retries stay
            within a budget, counted from the first attempt, of ``max_retries * (timeout +
            b)`` seconds, where ``b`` is the SDK's longest backoff wait (5 s): room for
            every attempt's full timeout and a backoff wait before each, so backoff
            retries are never cut short. A retry whose wait would end past the budget is
            not made, and the request fails with its last error (its criteria abstain), so
            a ``Retry-After`` beyond the budget fails the request at once instead of
            stalling the run, much as ``LLMConfig`` caps each wait at ``retry_max_wait``.
        max_parallel_requests: Maximum concurrent requests to the endpoint, as in
            ``LLMConfig``. The limit is shared by every decision-model config whose resolved
            base URL has the same host, whatever its model name; the strictest limit wins.
            The limit applies within one event loop; each ``asyncio.run`` call has its own.
            ``None`` (default) sets no shared limit: the judge then keeps at most 100
            requests in flight per event loop, one per connection of its HTTP client (the
            size of the SDK's default pool). A limit sizes the pool instead, so one above
            100 is honoured. A request waits for its turn before it is sent, so the wait,
            however long, counts against neither ``timeout`` nor the retry budget: those
            measure the endpoint.
        cache_enabled: Cache responses on disk, as in ``LLMConfig``. There is one entry per
            request, keyed by the model, the resolved base URL, the state and the questions,
            plus the ``judge_id`` of any judge other than a lone ``judge_model_config``
            judge, so that judges of one decision model keep their own answers. A hit
            returns the stored response, usage included, without a request. Failed requests
            are never cached.
        cache_dir: Directory of the response cache. It is the same store that LLM judges
            use when their ``cache_dir`` matches.
        cache_ttl: Lifetime of a cache entry in seconds; ``None`` (default) never expires.
        extra_headers: Additional HTTP headers sent with every request, header names to
            values, both strings. Building a client checks each header with the HTTP/1.1
            layer's own rule (``h11``): a name that is not a token (letters, digits and
            ``!#$%&'*+-.^_`|~``), or a value that is not an ASCII string or has a line
            break or surrounding whitespace, such as a secret read from a file with its
            trailing newline, raises ``ValueError`` naming the header but never its value:
            the HTTP layer would refuse it on every request. Unlike ``LLMConfig``, a few
            headers cannot be set: the TypeSafe SDK sets ``Authorization`` (the bearer
            ``api_key``), ``Accept``, ``Content-Type``, ``User-Agent``, ``X-TypeSafe-SDK``,
            ``X-TypeSafe-Runtime`` and ``X-TypeSafe-Retry-Count`` itself on every request,
            replacing any value given, so naming one of them, in any letter case, raises
            ``ValueError`` too.
        binary_framing: How a binary criterion is posed. The criterion's ``requirement`` is
            always sent verbatim; only the structure around it differs.

            - ``"noul_framed"`` (default): a yes/no question. The requirement is wrapped in
              a fixed sentence asking whether the submission satisfies it, and the yes and
              no outcomes carry the same MET/UNMET definitions as the LLM judge's system
              prompt (the negative-criterion definitions for negative weights). P(MET) is
              compared with ``decision_threshold``. The fixed text adapts to the state: a
              structured submission is judged by its output, with the thinking as context
              only, and a reference submission comes with the LLM judge's rule for using
              it (calibrate expectations, judge the submission on its own merits).
            - ``"noul"``: a yes/no question whose instructions are the bare requirement,
              with no wrapper and no definitions. Thresholded like ``"noul_framed"``. It
              sees the input, reference and thinking only through the state.
            - ``"choice"``: a choice among ``MET``, ``UNMET`` and ``CANNOT_ASSESS`` with the
              same definitions and fixed text as ``"noul_framed"``; the selected option is
              the verdict. This is the only binary framing that can abstain.
        ordinal_framing: How an ordinal multi-choice criterion is posed. The requirement is
            the bare instructions and option labels are always sent verbatim, in the
            rubric's order. Nominal criteria are always posed as ``"choice"``. Option
            shuffling (``CriterionGrader(shuffle_options=...)``) does not apply to decision
            models.

            - ``"choice"`` (default): a choice over the option labels, the NA option
              included; the selected option is the verdict. It needs unique labels (they
              are the choice keys) and at most 255 options.
            - ``"score"``: an ordered scale over the non-NA options, which needs 2 to 10 of
              them. The expected level ``s`` is snapped to level ``round(s)``, clamped to
              the scale; ``round`` is Python's, half to even, so a level exactly halfway
              between two snaps to the even one (0.5 to 0, 1.5 to 2), whatever the
              criterion's weight sign. The NA option is not among the levels offered, so a
              Score-framed criterion cannot abstain.

            A criterion its framing cannot express is left out of the item's request and
            fails alone, as a ``parse`` error; the other criteria are still asked.
        decision_threshold: Cut on P(MET) for the Noul framings, in [0, 1]. Above it the
            verdict is MET, below it UNMET; at exactly the threshold the verdict is the
            worst case for the criterion's weight sign, the same tie rule as ensemble
            aggregation. Choice and Score answers ignore it. It is not part of the cache
            key: changing it re-reads cached answers instead of asking again.
        input_cost_per_token: Price in USD per input token, e.g. ``0.042e-6``. A request's
            cost is its input tokens times this price. There is no built-in price table,
            because vendor prices change: set it whenever cost matters. ``None`` (default)
            means the cost is unknown, and the decision model then contributes nothing to a
            report's ``completion_cost``.

    Examples:
        >>> jev = DecisionModelConfig(model="jev-latest", cache_enabled=True)
        >>> self_hosted = DecisionModelConfig(
        ...     model="my-org/rubric-dm-7b",
        ...     api_base="https://xyz.endpoints.huggingface.cloud",
        ...     api_key="hf_...",
        ... )
    """

    model: str
    _: KW_ONLY
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    max_parallel_requests: int | None = None
    cache_enabled: bool = False
    cache_dir: str | Path = ".autorubric_cache"
    cache_ttl: int | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    binary_framing: BinaryFraming = "noul_framed"
    ordinal_framing: OrdinalFraming = "choice"
    decision_threshold: float = 0.5
    input_cost_per_token: float | None = None

    def __post_init__(self) -> None:
        """Validate the configuration.

        ``api_base`` and ``extra_headers`` are checked when a client is built, with the SDK's
        HTTP stack, which this config does not need.

        Raises:
            ValueError: If a field is out of range or not one of its allowed values.
        """
        name = type(self).__name__
        if not isinstance(self.model, str) or not self.model:
            raise ValueError(f"{name}.model is required and cannot be empty")
        if self.binary_framing not in get_args(BinaryFraming):
            raise ValueError(
                f"{name}.binary_framing must be one of {get_args(BinaryFraming)}; "
                f"got {self.binary_framing!r}"
            )
        if self.ordinal_framing not in get_args(OrdinalFraming):
            raise ValueError(
                f"{name}.ordinal_framing must be one of {get_args(OrdinalFraming)}; "
                f"got {self.ordinal_framing!r}"
            )
        if not (_is_real_number(self.decision_threshold) and 0 <= self.decision_threshold <= 1):
            raise ValueError(
                f"{name}.decision_threshold must be a number in [0, 1]; "
                f"got {self.decision_threshold!r}"
            )
        if not (_is_real_number(self.timeout) and math.isfinite(self.timeout) and self.timeout > 0):
            raise ValueError(
                f"{name}.timeout must be a positive, finite number of seconds; got {self.timeout!r}"
            )
        if not _is_positive_int(self.max_retries):
            raise ValueError(
                f"{name}.max_retries is the total number of attempts and must be an "
                f"integer >= 1; got {self.max_retries!r}"
            )
        if self.max_parallel_requests is not None and not _is_positive_int(
            self.max_parallel_requests
        ):
            raise ValueError(
                f"{name}.max_parallel_requests must be None or an integer >= 1; "
                f"got {self.max_parallel_requests!r}"
            )
        if self.cache_ttl is not None and not (
            _is_real_number(self.cache_ttl) and self.cache_ttl > 0
        ):
            raise ValueError(
                f"{name}.cache_ttl must be None or a positive number of seconds; "
                f"got {self.cache_ttl!r}"
            )
        price = self.input_cost_per_token
        if price is not None and not (
            _is_real_number(price) and math.isfinite(price) and price >= 0
        ):
            raise ValueError(
                f"{name}.input_cost_per_token must be None or a finite number >= 0; got {price!r}"
            )


def _is_valid_api_key(key: str) -> bool:
    """Whether a stripped API key is one the SDK accepts: printable ASCII, no whitespace.

    This is the SDK's own rule, which it applies only when it builds a client, that is on
    a request. Applying it when the grader is built makes an unusable key fail there
    instead of abstaining on every item, and keeps a key with a line break or other
    control character from ever reaching the HTTP layer, whose error would repeat it
    (into every failed report's ``error``) under an SDK that does not redact it.
    """
    return key.isascii() and key.isprintable() and " " not in key


def _import_typesafe_sdk() -> ModuleType:
    """Import the optional TypeSafe SDK, or explain how to install it."""
    try:
        import typesafe_sdk
    except ImportError as exc:
        raise ImportError(
            "Decision-model judges need the TypeSafe SDK (typesafe-sdk), which is not "
            "installed. Install it with: pip install 'autorubric[typesafe]'",
            name="typesafe_sdk",
        ) from exc
    return typesafe_sdk


def _json_fallback(value: object) -> object:
    """Encode the non-JSON types a request may hold: question objects and abstract containers."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: object) -> str:
    """One deterministic JSON text for a request component.

    Question objects are written in their wire form and any mapping or sequence type as a
    JSON object or array, so equivalent representations of one request encode alike. Key
    order is kept, not sorted: the endpoint receives it, and a Choice question's option
    order is part of what the model sees.
    """
    return json.dumps(value, separators=(",", ":"), default=_json_fallback)


class DecisionModelClient:
    """Client for a System One-compatible decision model; the counterpart of ``LLMClient``.

    Wraps the SDK's ``AsyncTypeSafeClient`` and adds what AutoRubric gives every judge: a
    response cache in the shared on-disk store, rate limiting through ``RateLimitPool``,
    and usage and cost accounting. Retries and timeouts are delegated to the SDK.

    Resolution happens once, at construction: the SDK is imported, and the API key and base
    URL are resolved the way the SDK resolves them (explicit value, then ``TYPESAFE_API_KEY``
    / ``TYPESAFE_BASE_URL``, then, for the URL, the SDK's default) and checked, the key by
    the SDK's own rule, the request URL by the SDK's HTTP stack (``_check_base_url``) and
    each extra header by the HTTP/1.1 layer's rule (``_check_extra_headers``), so that a
    misconfiguration fails when the grader is built, never on every item.

    Each SDK client gets a new HTTP client, and ``httpx2`` reads the environment's TLS and
    proxy settings whenever it builds one, outside the SDK's error handling. So one is also
    built at construction, and an environment in which none can be built fails there. If the
    environment changes later so that one cannot be built, the request is not sent and
    raises ``TypeSafeAPIConnectionError``, which ``classify_grading_error`` routes as
    ``infrastructure`` (an abstention), as it does an endpoint that cannot be reached.

    SDK clients hold connections bound to the event loop that opened them. One SDK client is
    therefore shared by all requests in flight together on a loop, and it is closed, on
    that loop, when the last of them finishes. A later request, or a request on another
    loop (e.g. successive ``asyncio.run`` calls), gets a new one. No connection outlives
    its loop.

    A request is handed to the SDK only when its SDK client has a free connection: the
    client's HTTP client holds one connection per request it may have in flight
    (``max_parallel_requests``, else 100), and the request first waits for one of that many
    slots. Waiting inside the SDK instead would count against the request's timeout and its
    retry budget, both counted from the first attempt, although the endpoint is not
    involved; a long queue would then fail requests the endpoint never saw.
    """

    def __init__(self, config: DecisionModelConfig, *, cache_namespace: str | None = None) -> None:
        """Build a client for ``config``.

        Args:
            config: The decision-model configuration.
            cache_namespace: Keeps this client's response-cache entries apart from those of
                other clients that send identical requests, such as several judges of one
                decision model, each of which gets its own answer. It becomes part of the
                cache key; ``None`` (the default) leaves the key exactly as without it.

        Raises:
            ImportError: If the TypeSafe SDK is not installed.
            ValueError: If no API key is given or set in ``TYPESAFE_API_KEY``, or the key is
                not printable ASCII without whitespace (the SDK's rule); if the resolved
                base URL (``api_base`` or ``TYPESAFE_BASE_URL``) is one no request could use
                (``_check_base_url``); if an extra header is one no request could carry
                (``_check_extra_headers``); or if ``httpx2`` cannot build an HTTP client in
                this environment (e.g. an ``SSL_CERT_FILE`` that names a missing file, or a
                SOCKS proxy without the ``socksio`` package), chained to the error it raised.
        """
        self.config = config
        self._cache_namespace = cache_namespace
        self._typesafe = _import_typesafe_sdk()
        constants = self._typesafe.constants

        if config.api_key is not None:
            api_key, key_source = config.api_key, "DecisionModelConfig.api_key"
        else:
            api_key = os.environ.get(constants.API_KEY_ENV, "")
            key_source = f"the {constants.API_KEY_ENV} environment variable"
        self._api_key = api_key.strip()
        if not self._api_key:
            raise ValueError(
                f"No API key for decision model {config.model!r}: pass "
                f"DecisionModelConfig(api_key=...) or set the {constants.API_KEY_ENV} "
                "environment variable."
            )
        if not _is_valid_api_key(self._api_key):
            # The message never repeats the key.
            raise ValueError(
                f"The API key for decision model {config.model!r} ({key_source}) must "
                "contain only printable ASCII characters without whitespace."
            )

        if config.api_base is not None:
            base_url, source = config.api_base, "DecisionModelConfig.api_base"
        else:
            base_url = os.environ.get(constants.BASE_URL_ENV, "").strip()
            base_url, source = base_url or constants.DEFAULT_BASE_URL, constants.BASE_URL_ENV
        self._base_url: str = base_url.rstrip("/")
        _check_base_url(self._base_url, source)
        self._host = _url_host(self._base_url)
        _check_extra_headers(config.extra_headers, "DecisionModelConfig.extra_headers")

        limit = config.max_parallel_requests
        self._connections = _DEFAULT_CONNECTIONS if limit is None else limit
        # httpx2 reads TLS and proxy settings from the environment whenever it builds an HTTP
        # client, and each loop's SDK client gets a new one. Building one here makes an
        # environment in which none can be built fail construction, like a missing key,
        # instead of every request. It opens no connection, so it is simply dropped.
        try:
            self._new_http_client()
        except Exception as exc:
            raise ValueError(
                f"Decision model {config.model!r} cannot build an HTTP client "
                f"({type(exc).__name__}: {exc}): check the TLS and proxy settings httpx2 "
                "reads from the environment (SSL_CERT_FILE, SSL_CERT_DIR, HTTP_PROXY, "
                "HTTPS_PROXY, ALL_PROXY and NO_PROXY, in either letter case) and the "
                "system's proxy settings."
            ) from exc

        # max_retries counts attempts; the SDK counts retries after the first. The SDK honours
        # the endpoint's Retry-After with no cap of its own; the only bound on its waits is
        # the policy's total budget, counted from the first attempt. The budget leaves room
        # for every attempt's full timeout plus the longest backoff wait before each, so it
        # never cuts backoff retries short, while a Retry-After that would end past it fails
        # the request at once instead of stalling the run.
        longest_backoff = self._typesafe.RetryPolicy().backoff_max
        self._retry_policy = self._typesafe.RetryPolicy(
            max_retries=config.max_retries - 1,
            timeout=config.max_retries * (config.timeout + longest_backoff),
        )
        # Per loop: the SDK client, its connection slots, and the requests leasing them.
        self._sdk_clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, tuple[AsyncTypeSafeClient, asyncio.Semaphore, int]
        ] = weakref.WeakKeyDictionary()

        self._cache: diskcache.Cache | None = None
        if config.cache_enabled:
            self._cache = _open_response_cache(config.cache_dir)

    @property
    def base_url(self) -> str:
        """The resolved base URL, without a trailing slash."""
        return self._base_url

    @property
    def host(self) -> str:
        """Host of the resolved base URL, with any non-default port; never credentials."""
        return self._host

    @property
    def rate_limit_key(self) -> str:
        """The ``RateLimitPool`` bucket: ``decision-model@<host>``, shared by the host's models."""
        return f"decision-model@{self._host}"

    async def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        use_cache: bool | None = None,
    ) -> SystemOneResponse:
        """Send one System One request: ``questions`` about ``state``.

        Args:
            state: The material every question sees: text, a JSON object or an array.
            questions: Question objects (``Noul``, ``Choice``, ``Score``) or their dict
                forms, keyed by the ids the answers come back under.
            use_cache: Whether to use the response cache for this request. ``None``
                (default) follows ``config.cache_enabled``.

        Returns:
            The SDK's ``SystemOneResponse``: answers keyed by question id, the model that
            answered, and token usage. A cache hit returns the stored response; transport
            metadata (the raw HTTP response and request id) is not stored, because it
            carries the request's headers.

        Raises:
            typesafe_sdk.TypeSafeError: The request failed after the configured attempts,
                or the endpoint's answer did not validate. ``classify_grading_error`` routes
                every SDK exception.
        """
        should_cache = self.config.cache_enabled if use_cache is None else use_cache
        cache_key: str | None = None
        if should_cache:
            cache_key = self._cache_key(state, questions)
            cached = self._cached_response(cache_key)
            if cached is not None:
                return cached

        async with self._lease_sdk_client() as (sdk_client, connection_slots):
            semaphore = await RateLimitPool.get_instance().get_semaphore(
                self.rate_limit_key, self.config.max_parallel_requests
            )
            shared_limit = semaphore if semaphore is not None else contextlib.nullcontext()
            async with shared_limit, connection_slots:
                response = await sdk_client.system_one(state=state, questions=questions)

        if cache_key is not None:
            self._ensure_cache().set(
                cache_key, response.model_dump_json(), expire=self.config.cache_ttl
            )
            logger.debug("Cached decision-model response for %s...", cache_key[:8])
        return response

    @staticmethod
    def token_usage(response: SystemOneResponse) -> TokenUsage:
        """Token usage of a response: input tokens are prompt tokens, output are completion.

        A count the endpoint does not report counts as 0, as for LLM responses.
        """
        prompt = response.usage.input_tokens or 0
        completion = response.usage.output_tokens or 0
        return TokenUsage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )

    def completion_cost(self, response: SystemOneResponse) -> float | None:
        """Cost in USD: input tokens times ``config.input_cost_per_token``.

        A cached response carries its original usage, so a re-run served from the cache
        reports the same cost as the run that made the request, at the configured price.

        Returns:
            The cost, or ``None`` when it is unknown: no price is configured, or the
            endpoint did not report input tokens.
        """
        price = self.config.input_cost_per_token
        input_tokens = response.usage.input_tokens
        if price is None or input_tokens is None:
            return None
        return input_tokens * price

    def _cache_key(self, state: JSONContent, questions: Mapping[str, Question]) -> str:
        """SHA-256 over the model, the resolved base URL, the state and the questions.

        Everything that shapes the request is in the key, and so is the client's
        ``cache_namespace`` when it has one; nothing else is. Framing is already expressed in
        the questions, and ``decision_threshold`` only post-processes answers, so changing it
        re-reads cached answers instead of asking again.
        """
        parts: list[object] = [self.config.model, self._base_url, state, questions]
        if self._cache_namespace is not None:
            parts.append({"namespace": self._cache_namespace})
        content = _canonical_json(parts)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _cached_response(self, cache_key: str) -> SystemOneResponse | None:
        """The cached response for ``cache_key``; an unreadable entry counts as a miss."""
        cached = self._ensure_cache().get(cache_key)
        if cached is None:
            return None
        try:
            response = self._typesafe.SystemOneResponse.model_validate_json(cached)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Ignoring an unreadable decision-model cache entry %s... (asking the endpoint "
                "again and replacing it): %s",
                cache_key[:8],
                exc,
            )
            return None
        logger.debug("Cache hit for %s...", cache_key[:8])
        return response

    def _new_http_client(self) -> httpx2.AsyncClient:
        """A new HTTP client for an SDK client, holding ``self._connections`` connections.

        Its timeout times each HTTP operation but not a wait for a connection
        (``pool=None``), which is local, not the endpoint's. Building it reads the
        environment's TLS and proxy settings, as building any ``httpx2`` client does.
        """
        import httpx2

        return httpx2.AsyncClient(
            limits=httpx2.Limits(max_connections=self._connections),
            timeout=httpx2.Timeout(self.config.timeout, pool=None),
        )

    @contextlib.asynccontextmanager
    async def _lease_sdk_client(
        self,
    ) -> AsyncIterator[tuple[AsyncTypeSafeClient, asyncio.Semaphore]]:
        """Lease the running loop's SDK client and its connection slots.

        The first request in flight on a loop creates them: an SDK client whose HTTP client
        (``_new_http_client``) holds ``max_parallel_requests`` connections
        (``_DEFAULT_CONNECTIONS`` without a limit), and a semaphore with one slot per
        connection. A request holding a slot thus always finds a free connection. If the
        environment no longer lets an HTTP client be built, the request is not sent and
        raises ``TypeSafeAPIConnectionError`` (``infrastructure``).

        The last request to finish on a loop closes that loop's client there, the only place
        its connections can be closed; the SDK client closes the HTTP client it was given. A
        failure to close is logged, not raised: the request itself has already succeeded or
        failed on its own terms.
        """
        loop = asyncio.get_running_loop()
        lease = self._sdk_clients.get(loop)
        if lease is None:
            try:
                http_client = self._new_http_client()
            except Exception as exc:
                # One was built at construction, so the environment has changed since.
                raise self._typesafe.TypeSafeAPIConnectionError(
                    "Request not sent: no HTTP client can be built with the environment's TLS "
                    f"and proxy settings ({type(exc).__name__}: {exc})"
                ) from exc
            sdk_client = self._typesafe.AsyncTypeSafeClient(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self.config.model,
                headers=self.config.extra_headers,
                retry=self._retry_policy,
                http_client=http_client,  # the SDK takes its timeout from it
            )
            lease = (sdk_client, asyncio.Semaphore(self._connections), 0)
        sdk_client, connection_slots, in_flight = lease
        self._sdk_clients[loop] = (sdk_client, connection_slots, in_flight + 1)
        try:
            yield sdk_client, connection_slots
        finally:
            sdk_client, connection_slots, in_flight = self._sdk_clients[loop]
            if in_flight > 1:
                self._sdk_clients[loop] = (sdk_client, connection_slots, in_flight - 1)
            else:
                del self._sdk_clients[loop]
                try:
                    await sdk_client.aclose()
                except Exception:
                    logger.warning("Closing a decision-model SDK client failed", exc_info=True)

    def _ensure_cache(self) -> diskcache.Cache:
        """The response cache, opened on first use."""
        if self._cache is None:
            self._cache = _open_response_cache(self.config.cache_dir)
        return self._cache

    def close(self) -> None:
        """Close the on-disk cache, releasing its file handles.

        Mirrors ``LLMClient.close``: call it when done so the cache directory can be
        removed (Windows refuses to delete open files). Safe without a cache. SDK clients
        need no closing here; each is closed when its last request finishes.
        """
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self) -> DecisionModelClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# =============================================================================
# Framing: the shared state and one question per criterion
# =============================================================================

_MAX_CHOICE_OPTIONS = 255
"""Most options one Choice question may offer (a System One API limit)."""

_MIN_SCORE_LEVELS, _MAX_SCORE_LEVELS = 2, 10
"""Fewest and most levels one Score question may offer (a System One API limit)."""

_BINARY_VERDICTS = (CriterionVerdict.MET, CriterionVerdict.UNMET, CriterionVerdict.CANNOT_ASSESS)
"""The verdicts a binary Choice question offers, in the order it offers them."""

_QuestionKind = Literal["noul", "choice", "score"]


def question_id(criterion_idx: int) -> str:
    """Id of the question posing criterion ``criterion_idx``: ``"c{criterion_idx}"``.

    ``criterion_idx`` is the criterion's index in the effective rubric (the rubric after NA
    options are guaranteed), the same index that keys option shuffling and few-shot
    selection, so ids are unique and stable even for unnamed or duplicate-named criteria.
    """
    return f"c{criterion_idx}"


def build_state(
    to_grade: str,
    *,
    query: str | None = None,
    reference_submission: str | None = None,
    guidelines: str | None = None,
) -> dict[str, str]:
    """The state a decision model judges: every question of the item's request sees it.

    Field names mirror the XML tags of the LLM judge's user prompt. Optional fields appear
    only when present (non-empty, the rule the LLM user prompt applies; for guidelines, not
    blank, the rule of ``Rubric.guidelines``), in this order:

    - ``"guidelines"``: the rubric's guidelines, verbatim. Sent once per request whatever
      the number of criteria, and, being part of the state, part of the response cache key.
    - ``"input"``: the query that prompted the submission.
    - ``"reference_submission"``: an exemplar response, for calibration.
    - ``"submission"``: the text to grade, sent unchanged. When ``to_grade`` holds a
      ``<thinking>`` or ``<output>`` section (``Grader.grade`` flattens a
      ``{"thinking", "output"}`` submission into such sections), the parts recovered by
      ``parse_thinking_output`` are sent as ``"thinking"`` and ``"output"`` instead, both
      always, ``"thinking"`` possibly empty. Only an opening marker followed by its
      closing marker is a section; other text, stray markers included, is a plain
      submission.

    Args:
        to_grade: The submission, as ``CriterionGrader.judge`` receives it.
        query: The input that prompted the submission.
        reference_submission: An exemplar response for grading context.
        guidelines: The rubric's guidelines.

    Returns:
        The state, a JSON object of strings.

    Raises:
        TypeError: If ``guidelines`` is neither a ``str`` nor ``None``.
    """
    state: dict[str, str] = {}
    guidelines = _normalize_guidelines(guidelines)
    if guidelines is not None:
        state["guidelines"] = guidelines
    if query:
        state["input"] = query
    if reference_submission:
        state["reference_submission"] = reference_submission
    if _has_thinking_output_sections(to_grade):
        sections = parse_thinking_output(to_grade)
        state["thinking"] = sections.get("thinking", "")
        state["output"] = sections.get("output", "")
    else:
        state["submission"] = to_grade
    return state


def _question_kind(criterion: Criterion, config: DecisionModelConfig) -> _QuestionKind:
    """The question type posing ``criterion`` under ``config``'s framings."""
    if criterion.options is None:
        return "choice" if config.binary_framing == "choice" else "noul"
    if criterion.scale_type == "ordinal" and config.ordinal_framing == "score":
        return "score"
    return "choice"


def _framed_instructions(requirement: str, state: Mapping[str, object]) -> str:
    """Framed instructions for a binary criterion: task, context sentences, criterion.

    The task sentence names the judged state field (``submission``, or ``output`` when the
    state splits a structured submission); each optional state field the judgment depends
    on adds its usage sentence, in state order (``guidelines``, then
    ``reference_submission``); the requirement comes last, verbatim, after the
    ``"Criterion: "`` label. Every sentence is a ``prompts.py`` constant.
    """
    structured = "output" in state
    sentences = [
        DECISION_MODEL_THINKING_OUTPUT_TASK_INSTRUCTION
        if structured
        else DECISION_MODEL_TASK_INSTRUCTION
    ]
    if "guidelines" in state:
        sentences.append(DECISION_MODEL_GUIDELINES_INSTRUCTION)
    if "reference_submission" in state:
        judged = "output" if structured else "submission"
        sentences.append(DECISION_MODEL_REFERENCE_INSTRUCTION.format(judged=judged))
    sentences.append(DECISION_MODEL_CRITERION_PREFIX + requirement)
    return " ".join(sentences)


def _choice_labels(criterion: Criterion) -> list[str]:
    """A multi-choice criterion's option labels, as the keys of its Choice question.

    Raises:
        ValueError: If a Choice question cannot express the options: a label repeats
            (Choice options are keyed by label) or there are more options than one Choice
            question may offer.
    """
    labels = [option.label for option in criterion.options or []]
    repeated = [label for label, count in Counter(labels).items() if count > 1]
    if repeated:
        raise ValueError(
            "a Choice question is keyed by option label, and these option labels are not "
            f"unique: {repeated!r}"
        )
    if len(labels) > _MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"the criterion has {len(labels)} options, and a Choice question offers at most "
            f"{_MAX_CHOICE_OPTIONS}"
        )
    return labels


def _score_levels(criterion: Criterion) -> list[int]:
    """Original option indices of a criterion's Score levels: its non-NA options, in order.

    Raises:
        ValueError: If the number of non-NA options is outside what one Score question may
            offer.
    """
    levels = [i for i, option in enumerate(criterion.options or []) if not option.na]
    if not _MIN_SCORE_LEVELS <= len(levels) <= _MAX_SCORE_LEVELS:
        raise ValueError(
            f"the criterion has {len(levels)} non-NA levels, and a Score question takes "
            f"{_MIN_SCORE_LEVELS} to {_MAX_SCORE_LEVELS}"
        )
    return levels


def build_question(
    criterion: Criterion,
    config: DecisionModelConfig,
    state: Mapping[str, object],
) -> Noul | Choice | Score:
    """Pose one criterion of the effective rubric as a decision-model question.

    The requirement and every option label are sent verbatim; only the structure around
    them depends on the framing:

    - Binary, ``binary_framing="noul"``: ``Noul(instructions=requirement)``.
    - Binary, ``"noul_framed"``: ``Noul`` whose instructions are the framed instructions
      and whose yes/no outcomes carry the MET/UNMET definitions of the LLM judge's system
      prompt (its negative-criterion definitions for a negative weight).
    - Binary, ``"choice"``: ``Choice`` over ``MET``, ``UNMET`` and ``CANNOT_ASSESS`` with the
      same definitions and the framed instructions.
    - Multi-choice, nominal, or ordinal with ``ordinal_framing="choice"``: ``Choice`` whose
      keys are the option labels, NA option included, with no descriptions.
    - Multi-choice, ordinal with ``ordinal_framing="score"``: ``Score`` whose levels are the
      labels of the non-NA options, in order.

    Framed instructions (``prompts.py`` constants) ask whether the criterion is satisfied
    by the ``submission``, or by the ``output`` with the ``thinking`` as context only when
    ``state`` splits a structured submission; ``guidelines`` in ``state`` add their
    precedence rule ("Apply the `guidelines`; the criterion text governs."), then a
    ``reference_submission`` its usage rule. The requirement ends them, after
    ``"Criterion: "``. Bare Noul and multi-choice questions get no such text: they see
    those fields only through the state.

    Args:
        criterion: The criterion, as evaluated (after its NA option is guaranteed).
        config: The decision model's configuration, which selects the framings.
        state: The item's state, from ``build_state``. Only which fields it holds matters.

    Returns:
        The SDK question object.

    Raises:
        ValueError: If the question cannot express the criterion: repeated option labels
            or more than 255 options for a Choice, or fewer than 2 or more than 10 non-NA
            options for a Score. ``classify_grading_error`` routes it as ``parse``.
        ImportError: If the TypeSafe SDK is not installed.
    """
    sdk = _import_typesafe_sdk()
    kind = _question_kind(criterion, config)
    requirement = criterion.requirement
    if criterion.options is not None:
        if kind == "score":
            labels = [criterion.options[i].label for i in _score_levels(criterion)]
            return sdk.Score(instructions=requirement, criteria=labels)
        return sdk.Choice(
            instructions=requirement,
            criteria=dict.fromkeys(_choice_labels(criterion)),
        )
    if config.binary_framing == "noul":
        return sdk.Noul(instructions=requirement)
    negative = criterion.weight < 0
    met = NEGATIVE_MET_DEFINITION if negative else MET_DEFINITION
    unmet = NEGATIVE_UNMET_DEFINITION if negative else UNMET_DEFINITION
    instructions = _framed_instructions(requirement, state)
    if kind == "noul":
        return sdk.Noul(instructions=instructions, criteria={"true": met, "false": unmet})
    definitions = (met, unmet, CANNOT_ASSESS_DEFINITION)
    return sdk.Choice(
        instructions=instructions,
        criteria={
            verdict.value: definition
            for verdict, definition in zip(_BINARY_VERDICTS, definitions, strict=True)
        },
    )


def _describe(criterion: Criterion, criterion_idx: int) -> str:
    """How messages name a criterion: its question id, and its name when it has one."""
    qid = question_id(criterion_idx)
    return f"criterion {qid} ({criterion.name!r})" if criterion.name else f"criterion {qid}"


def build_questions(
    rubric: Sequence[Criterion],
    config: DecisionModelConfig,
    state: Mapping[str, object],
) -> tuple[dict[str, Question], dict[int, ValueError]]:
    """Pose every criterion of the effective rubric, for one request.

    Questions are built before anything is sent. A criterion whose question cannot be built
    is left out of the request, not failed with it: it gets its own error, and every other
    criterion still goes into the single request under its original id.

    Args:
        rubric: The effective rubric (NA options already guaranteed), in order.
        config: The decision model's configuration.
        state: The item's state, from ``build_state``.

    Returns:
        ``(questions, errors)``: the questions keyed by ``question_id(criterion_idx)`` in
        rubric order, and a ``ValueError`` per criterion index that could not be posed
        (``classify_grading_error`` routes it as ``parse``). When ``questions`` is empty
        there is nothing to send.
    """
    questions: dict[str, Question] = {}
    errors: dict[int, ValueError] = {}
    for criterion_idx, criterion in enumerate(rubric):
        try:
            questions[question_id(criterion_idx)] = build_question(criterion, config, state)
        except ValueError as exc:
            error = ValueError(
                f"Cannot pose {_describe(criterion, criterion_idx)} to the decision model: {exc}"
            )
            error.__cause__ = exc
            errors[criterion_idx] = error
    return questions, errors


# =============================================================================
# Answers: one report per criterion, and the confidence of what was selected
# =============================================================================


def selection_confidence(p_selected: float, n_outcomes: int) -> float:
    """A judge's support for the outcome it selected, in [0, 1].

    ``clamp((K * p - 1) / (K - 1), 0, 1)`` for ``K = n_outcomes`` offered and the selected
    outcome's probability ``p``: ``1.0`` when ``p`` is 1, ``0.0`` at chance (``p = 1/K``)
    or below. For the most probable of a Choice's options this is TypeSafe's reported
    confidence; for a Noul verdict at ``decision_threshold=0.5`` it is ``2 * |p - 0.5|``.
    An outcome can be selected without being the most probable (a Score level snapped
    from the expected score, a Noul verdict under a threshold other than 0.5); when its
    probability is at or below chance, the confidence is ``0.0``: no support for the
    outcome issued.

    Args:
        p_selected: Probability of the selected outcome, in [0, 1].
        n_outcomes: Number of outcomes offered, at least 2.

    Returns:
        The confidence.

    Raises:
        ValueError: If ``n_outcomes`` is not an integer >= 2 or ``p_selected`` is not a
            number in [0, 1].
    """
    if not (_is_positive_int(n_outcomes) and n_outcomes >= 2):
        raise ValueError(f"n_outcomes must be an integer >= 2; got {n_outcomes!r}")
    if not (_is_real_number(p_selected) and 0 <= p_selected <= 1):
        raise ValueError(f"p_selected must be a number in [0, 1]; got {p_selected!r}")
    return min(1.0, max(0.0, (n_outcomes * p_selected - 1) / (n_outcomes - 1)))


def _checked_probability(value: object, what: str) -> float:
    """``value`` as a probability, or ``ValueError`` naming ``what`` it was meant to be."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return float(value)  # NaN fails both comparisons and is rejected
    raise ValueError(f"{what}, {value!r}, is not a number in [0, 1]")


_Outcome = TypeVar("_Outcome", str, int)


def _checked_probabilities(
    probabilities: object, offered: Sequence[_Outcome]
) -> dict[_Outcome, float]:
    """An answer's probabilities, checked to cover exactly the ``offered`` outcomes.

    The values are the endpoint's own, unrenormalized: they sum to 1 only approximately
    when the endpoint rounds them.

    Raises:
        ValueError: If an offered outcome has no probability, a probability is for an
            outcome not offered, or a probability is not a number in [0, 1].
    """
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(offered):
        given = list(probabilities) if isinstance(probabilities, Mapping) else probabilities
        raise ValueError(
            f"its probabilities must cover exactly the outcomes offered, {list(offered)!r}; "
            f"got {given!r}"
        )
    by_outcome = cast(Mapping[_Outcome, object], probabilities)
    return {
        outcome: _checked_probability(by_outcome[outcome], f"the probability of {outcome!r}")
        for outcome in offered
    }


def _snapped_level(expected: float, n_levels: int) -> int:
    """The level an expected Score snaps to: ``round(expected)`` clamped to the scale.

    ``round`` is Python's, half to even (0.5 snaps to 0, 1.5 to 2), with no tolerance band.
    Halves go down as often as up, so snapping adds no systematic bias toward either end
    of the scale, and the snap does not depend on the criterion's weight sign.
    """
    return min(max(round(expected), 0), n_levels - 1)


def _report(
    criterion: Criterion,
    *,
    probabilities: dict[str, float],
    confidence: float,
    verdict: CriterionVerdict | None = None,
    multi_choice_verdict: MultiChoiceVerdict | None = None,
) -> CriterionReport:
    """The criterion's report: its own fields, the outcome, and no explanation."""
    return CriterionReport(
        **{name: getattr(criterion, name) for name in Criterion.model_fields},
        verdict=verdict,
        multi_choice_verdict=multi_choice_verdict,
        reason=None,
        reasoning=None,
        probabilities=probabilities,
        confidence=confidence,
    )


def _binary_report(
    criterion: Criterion, answer: Any, config: DecisionModelConfig
) -> CriterionReport:
    """Report for a binary criterion's Noul or Choice answer."""
    if answer.type == "noul":
        p = _checked_probability(answer.noul, "its probability of yes")
        threshold = config.decision_threshold
        if p > threshold:
            verdict = CriterionVerdict.MET
        elif p < threshold:
            verdict = CriterionVerdict.UNMET
        else:  # exactly at the threshold: the ensemble tie rule
            verdict = _binary_worst_verdict(criterion.weight)
        probabilities = {CriterionVerdict.MET.value: p, CriterionVerdict.UNMET.value: 1.0 - p}
    else:
        offered = [verdict.value for verdict in _BINARY_VERDICTS]
        probabilities = _checked_probabilities(answer.probabilities, offered)
        if answer.choice not in offered:
            raise ValueError(
                f"its choice {answer.choice!r} is not one of the verdicts offered, {offered!r}"
            )
        verdict = CriterionVerdict(answer.choice)
    return _report(
        criterion,
        verdict=verdict,
        probabilities=probabilities,
        confidence=selection_confidence(probabilities[verdict.value], len(probabilities)),
    )


def _multi_choice_report(criterion: Criterion, answer: Any) -> CriterionReport:
    """Report for a multi-choice criterion's Choice or Score answer."""
    options = criterion.options or []
    if answer.type == "score":
        levels = _score_levels(criterion)
        by_level = _checked_probabilities(answer.probabilities, list(range(len(levels))))
        expected = answer.score
        if not (_is_real_number(expected) and math.isfinite(expected)):
            raise ValueError(f"its expected score, {expected!r}, is not a finite number")
        # NA is not a level, so a Score answer never abstains.
        selected = levels[_snapped_level(expected, len(levels))]
        probabilities = {str(levels[level]): p for level, p in by_level.items()}
    else:
        labels = _choice_labels(criterion)
        by_label = _checked_probabilities(answer.probabilities, labels)
        if answer.choice not in labels:
            raise ValueError(f"its choice {answer.choice!r} is not one of the option labels")
        selected = labels.index(answer.choice)
        probabilities = {str(index): by_label[label] for index, label in enumerate(labels)}
    option = options[selected]
    return _report(
        criterion,
        multi_choice_verdict=MultiChoiceVerdict(
            selected_index=selected,
            selected_label=option.label,
            value=option.value,
            na=option.na,
        ),
        probabilities=probabilities,
        confidence=selection_confidence(probabilities[str(selected)], len(probabilities)),
    )


def answer_to_report(
    criterion: Criterion,
    criterion_idx: int,
    answers: Mapping[str, Answer],
    config: DecisionModelConfig,
) -> CriterionReport:
    """Map the decision model's answer for one criterion to its ``CriterionReport``.

    The answer is the one under ``question_id(criterion_idx)``; it must be of the type the
    criterion was posed as (``build_question``).

    - Noul: P(yes) is P(MET). Above ``decision_threshold`` the verdict is MET, below it
      UNMET; exactly at it, the worst case for the weight sign (``_binary_worst_verdict``,
      the ensemble tie rule). ``probabilities`` is ``{"MET": p, "UNMET": 1 - p}``.
    - Binary Choice: the selected option is the verdict; ``CANNOT_ASSESS`` is an
      abstention. ``probabilities`` has the three verdict values as keys.
    - Multi-choice Choice: the option whose label equals the choice; the NA option sets
      ``na=True``. ``probabilities`` has ``str(i)`` for every option ``i``, NA included.
    - Score: level ``round(score)`` (Python's half-to-even ``round``) clamped to the scale,
      mapped to its original option index. ``probabilities`` has ``str(i)`` for the non-NA
      options only: NA was not offered, so it is absent, not ``0.0``. A Score-framed
      criterion cannot abstain.

    Probability keys follow the rubric's original option order. A multi-choice report's
    ``value`` is the selected option's ``value``, never a fractional expected score.
    ``confidence`` is ``selection_confidence`` of the selected outcome, over the outcomes
    offered. A decision model gives no explanation: ``reason`` and ``reasoning`` are
    ``None``.

    Args:
        criterion: The criterion, as evaluated (after its NA option is guaranteed).
        criterion_idx: Its index in the effective rubric.
        answers: The response's answers, keyed by question id.
        config: The decision model's configuration (framings and ``decision_threshold``).

    Returns:
        The criterion's report.

    Raises:
        ValueError: If there is no answer for the criterion, the answer has the wrong type,
            selects an outcome that was not offered, or has malformed probabilities, or if
            the criterion cannot be posed at all. ``classify_grading_error`` routes it as
            ``parse``.
    """
    describe = _describe(criterion, criterion_idx)
    answer = answers.get(question_id(criterion_idx))
    if answer is None:
        raise ValueError(f"The decision model returned no answer for {describe}")
    try:
        expected_type = _question_kind(criterion, config)
        answer_type = getattr(answer, "type", None)
        if answer_type != expected_type:
            raise ValueError(f"expected a {expected_type} answer, got {answer_type!r}")
        if criterion.options is None:
            return _binary_report(criterion, answer, config)
        return _multi_choice_report(criterion, answer)
    except ValueError as exc:
        raise ValueError(f"Unusable decision-model answer for {describe}: {exc}") from exc


__all__ = ["BinaryFraming", "DecisionModelConfig", "OrdinalFraming"]
