"""Tests for ``DecisionModelClient``: transport wiring, event loops, cache, rate limits, cost.

Every SDK client is either a recording fake (``fake_sdk``) or the real SDK client on an
in-memory transport (``http_endpoint``); see ``conftest.py``. No test reaches the network.
"""

import asyncio
import json
import logging
import sys
import types

import h11
import httpx2
import pytest
import typesafe_sdk
from typesafe_sdk import Choice, Noul, RetryPolicy, Score
from typesafe_sdk.constants import DEFAULT_BASE_URL

from autorubric import DecisionModelConfig, TokenUsage, classify_grading_error
from autorubric.decision import _SYSTEM_ONE_PATH, DecisionModelClient
from autorubric.llm import LLMClient, LLMConfig
from autorubric.rate_limit import RateLimitPool

STATE = {"input": "Explain photosynthesis.", "submission": "Plants turn light into sugar."}
QUESTIONS = {
    "c0": Noul(instructions="Mentions light"),
    "c1": Choice(instructions="Tone", criteria={"formal": None, "casual": None}),
}


def dm_config(**overrides) -> DecisionModelConfig:
    return DecisionModelConfig(**{"model": "jev-latest", "api_key": "test-key", **overrides})


def _http_layer_sends(name: str, value: str) -> bool:
    """Whether the SDK's HTTP stack would send the header: httpx2 encodes it, h11 checks it."""
    try:
        raw = httpx2.Headers({name: value}).raw
        h11.Request(method="POST", target="/v1/systemone", headers=[(b"host", b"h"), *raw])
    except (UnicodeEncodeError, h11.LocalProtocolError):
        return False
    return True


@pytest.fixture
def make_client(tmp_path):
    """Build clients whose on-disk caches are closed at teardown (Windows keeps them open)."""
    clients: list[DecisionModelClient] = []

    def build(**overrides) -> DecisionModelClient:
        overrides.setdefault("cache_dir", tmp_path / "cache")
        client = DecisionModelClient(dm_config(**overrides))
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.close()


# =============================================================================
# Construction: optional SDK, API key, base URL
# =============================================================================


class TestConstruction:
    def test_missing_sdk_raises_an_actionable_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
        with pytest.raises(ImportError, match=r"pip install 'autorubric\[typesafe\]'"):
            DecisionModelClient(dm_config())

    def test_missing_api_key_raises_value_error(self):
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            DecisionModelClient(DecisionModelConfig(model="jev-latest"))

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_api_key_counts_as_missing(self, monkeypatch, value):
        monkeypatch.setenv("TYPESAFE_API_KEY", value)
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            DecisionModelClient(DecisionModelConfig(model="jev-latest"))
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            DecisionModelClient(DecisionModelConfig(model="jev-latest", api_key=value))

    @pytest.mark.parametrize(
        "key",
        [
            "SECRET abc def",
            "SECRETKEY123\nX",
            "SECRET\tabc",
            "SECRET\u00a0abc",
            "SECRET-\u00e9",
            "SECRET\x7fabc",
        ],
    )
    def test_malformed_api_key_raises_value_error_without_echoing_it(self, key):
        """Internal whitespace, control and non-ASCII characters: the SDK rejects such a key
        on the first request, and a key with a line break would otherwise reach the HTTP
        layer, whose error message (recorded on every failed report) repeats it."""
        with pytest.raises(ValueError, match="printable ASCII") as excinfo:
            DecisionModelClient(dm_config(api_key=key))
        message = str(excinfo.value)
        assert "DecisionModelConfig.api_key" in message
        assert "SECRET" not in message

    def test_malformed_environment_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "SECRET abc def")
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY") as excinfo:
            DecisionModelClient(DecisionModelConfig(model="jev-latest"))
        assert "printable ASCII" in str(excinfo.value)
        assert "SECRET" not in str(excinfo.value)

    @pytest.mark.parametrize(
        "key",
        [
            "sk-abc_123.XYZ~+/=",
            "  padded-key  ",
            "!#$%&'()*,:;<>?@[]^`{|}",
            "sk abc",
            "sk\tabc",
            "sk\nabc",
            "sk\rabc",
            "sk\u00a0abc",
            "sk-\u00e9",
            "sk\x00abc",
            "sk\x7fabc",
            "sk\u200babc",
        ],
    )
    def test_the_key_rule_is_the_sdks(self, key):
        """A key is accepted exactly when the installed SDK would accept it."""
        try:
            typesafe_sdk.TypeSafeClient(api_key=key, base_url="https://dm.example.com").close()
        except typesafe_sdk.TypeSafeError:
            sdk_accepts = False
        else:
            sdk_accepts = True
        try:
            DecisionModelClient(dm_config(api_key=key))
        except ValueError:
            accepts = False
        else:
            accepts = True
        assert accepts == sdk_accepts

    async def _sent_api_key(self, client: DecisionModelClient, fake_sdk) -> str:
        await client.system_one(STATE, QUESTIONS)
        return fake_sdk.clients[-1].kwargs["api_key"]

    @pytest.mark.asyncio
    async def test_api_key_falls_back_to_the_environment(self, monkeypatch, fake_sdk):
        monkeypatch.setenv("TYPESAFE_API_KEY", " env-key ")
        client = DecisionModelClient(DecisionModelConfig(model="jev-latest"))
        assert await self._sent_api_key(client, fake_sdk) == "env-key"

    @pytest.mark.asyncio
    async def test_explicit_api_key_wins_over_the_environment(self, monkeypatch, fake_sdk):
        monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
        client = DecisionModelClient(dm_config(api_key="explicit-key"))
        assert await self._sent_api_key(client, fake_sdk) == "explicit-key"

    def test_base_url_defaults_to_the_sdk_default(self):
        client = DecisionModelClient(dm_config())
        assert client.base_url == DEFAULT_BASE_URL == "https://api.typesafe.ai"

    def test_base_url_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_BASE_URL", " https://dm.example.com/ ")
        assert DecisionModelClient(dm_config()).base_url == "https://dm.example.com"

    def test_blank_environment_base_url_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_BASE_URL", "  ")
        assert DecisionModelClient(dm_config()).base_url == DEFAULT_BASE_URL

    def test_explicit_base_url_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_BASE_URL", "https://env.example.com")
        client = DecisionModelClient(dm_config(api_base="https://xyz.endpoints.hf.cloud/v2/"))
        assert client.base_url == "https://xyz.endpoints.hf.cloud/v2"

    def test_malformed_environment_base_url_raises(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_BASE_URL", "dm.example.com")
        with pytest.raises(ValueError, match="TYPESAFE_BASE_URL"):
            DecisionModelClient(dm_config())

    @pytest.mark.parametrize(
        "url", ["https://dm.exam\nple.com", "https://dm.example.com/v1\r\nX", "https://dm .com"]
    )
    def test_environment_base_url_with_inner_whitespace_raises(self, monkeypatch, url):
        """Only surrounding whitespace is stripped from ``TYPESAFE_BASE_URL``; whitespace
        inside the URL fails construction, not every request in the HTTP layer."""
        monkeypatch.setenv("TYPESAFE_BASE_URL", url)
        with pytest.raises(ValueError, match="TYPESAFE_BASE_URL"):
            DecisionModelClient(dm_config())

    @pytest.mark.parametrize("url", ["https://dm.example.com\n", "\thttps://dm.example.com"])
    def test_a_rejected_base_url_is_one_the_http_layer_refuses(self, url):
        """The whitespace rule reports at construction what every request would fail on."""
        with pytest.raises(httpx2.InvalidURL):
            httpx2.URL(url)
        with pytest.raises(ValueError, match="api_base"):
            dm_config(api_base=url)

    @pytest.mark.parametrize(
        "url",
        ["https://\uff41\uff50\uff49.typesafe.ai", "https://\u2603\u2603.com", "https://1.2.3.999"],
        ids=["fullwidth-host", "symbol-host", "impossible-ipv4"],
    )
    def test_base_url_the_http_layer_refuses_fails_at_construction(self, monkeypatch, url):
        """A host the HTTP layer cannot encode, such as a name that is not a valid IDNA name
        (fullwidth letters pasted from a document) or an impossible IPv4 address, fails every
        request inside the HTTP stack, as an error no category covers: the conservative
        worst case on every criterion of every item. The client checks the request URL
        with the SDK's HTTP stack when it is built, for both sources of the URL."""
        with pytest.raises(httpx2.InvalidURL):
            httpx2.URL(url + "/v1/systemone")
        with pytest.raises(ValueError, match="api_base"):
            DecisionModelClient(dm_config(api_base=url))
        monkeypatch.setenv("TYPESAFE_BASE_URL", url)
        with pytest.raises(ValueError, match="TYPESAFE_BASE_URL"):
            DecisionModelClient(dm_config())

    @pytest.mark.parametrize(
        "url, base_url",
        [
            ("https://api.typesafe.ai", "https://api.typesafe.ai"),
            ("https://xyz.endpoints.huggingface.cloud/", "https://xyz.endpoints.huggingface.cloud"),
            ("http://localhost:8080", "http://localhost:8080"),
            ("http://[::1]:9000/prefix", "http://[::1]:9000/prefix"),
            ("https://B\u00dcCHER.example/v2", "https://B\u00dcCHER.example/v2"),
            ("https://dm.example.com/a%20b/@org/", "https://dm.example.com/a%20b/@org"),
        ],
    )
    def test_base_urls_the_http_layer_accepts_build_a_client(self, monkeypatch, url, base_url):
        assert DecisionModelClient(dm_config(api_base=url)).base_url == base_url
        monkeypatch.setenv("TYPESAFE_BASE_URL", url)
        assert DecisionModelClient(dm_config()).base_url == base_url

    @pytest.mark.parametrize(
        "url",
        [
            "https://dm.example.com/api?tenant=a",
            "https://dm.example.com/api#x",
            "https://svc:SECRET@dm.example.com/api",
        ],
        ids=["query", "fragment", "credentials"],
    )
    def test_environment_base_url_with_a_query_fragment_or_credentials_raises(
        self, monkeypatch, url
    ):
        monkeypatch.setenv("TYPESAFE_BASE_URL", url)
        with pytest.raises(ValueError, match="TYPESAFE_BASE_URL") as excinfo:
            DecisionModelClient(dm_config())
        assert "SECRET" not in str(excinfo.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "https://dm.example.com/api?tenant=a",
            "https://dm.example.com/api?",
            "https://dm.example.com/api#x",
            "https://svc:pw@dm.example.com/api",
        ],
        ids=["query", "empty-query", "fragment", "credentials"],
    )
    async def test_a_base_url_rejected_for_its_structure_misdirects_every_request(self, url):
        """The rule reports at construction what every request would get wrong. The SDK
        appends ``/v1/systemone`` to the base URL string, so after a query or fragment the
        request misses the endpoint's path; and the HTTP layer turns URL credentials into
        Basic authentication, replacing the bearer API key."""
        sent: list[httpx2.Request] = []

        def handle(request: httpx2.Request) -> httpx2.Response:
            sent.append(request)
            return httpx2.Response(200, json=_OK_BODY)

        sdk_client = typesafe_sdk.AsyncTypeSafeClient(
            api_key="sk-test", base_url=url, transport=httpx2.MockTransport(handle)
        )
        async with sdk_client:
            await sdk_client.system_one(state=STATE, questions=QUESTIONS)

        (request,) = sent
        assert (request.url.path, request.headers["authorization"]) != (
            "/api/v1/systemone",
            "Bearer sk-test",
        )
        with pytest.raises(ValueError, match="api_base"):
            dm_config(api_base=url)

    @pytest.mark.parametrize(
        "name, value",
        [
            ("X-Goog-Api-Key", "AIzaSECRET\n"),
            ("Ocp-Apim-Subscription-Key", "abc123SECRET\r"),
            ("X-Org", "v\r\nX-Injected: 1"),
            ("X-Org", " SECRET"),
            ("X-Org", "caf\u00e9"),
            ("X Org", "v"),
            ("X-Org:", "v"),
        ],
    )
    def test_headers_the_http_layer_refuses_fail_at_construction(self, name, value):
        """Headers the HTTP layer would refuse on every request (after every retry, with an
        error repeating the value) are rejected when the config is built instead."""
        assert not _http_layer_sends(name, value)
        with pytest.raises(ValueError, match="extra_headers") as excinfo:
            dm_config(extra_headers={name: value})
        assert "SECRET" not in str(excinfo.value)

    @pytest.mark.parametrize(
        "name, value",
        [
            ("X-Team", "evals"),
            ("X-Empty", ""),
            ("X-Spaced", "a b\tc"),
            ("!#$%&'*+-.^_`|~0-9aZ", "~!@#$%^&*()_+`-={}[]|\\:;\"'<>,.?/"),
        ],
    )
    def test_accepted_headers_are_ones_the_http_layer_sends(self, name, value):
        assert dm_config(extra_headers={name: value}).extra_headers == {name: value}
        assert _http_layer_sends(name, value)

    def test_base_url_is_resolved_once_at_construction(self, monkeypatch):
        client = DecisionModelClient(dm_config())
        monkeypatch.setenv("TYPESAFE_BASE_URL", "https://later.example.com")
        assert client.base_url == DEFAULT_BASE_URL

    def test_cache_is_opened_only_when_enabled(self, make_client):
        assert make_client()._cache is None
        assert make_client(cache_enabled=True)._cache is not None


# =============================================================================
# SDK wiring: what the SDK client is built with and what one request carries
# =============================================================================


class TestSdkWiring:
    @pytest.mark.asyncio
    async def test_one_request_carries_exactly_the_given_state_and_questions(self, fake_sdk):
        client = DecisionModelClient(dm_config())
        response = await client.system_one(STATE, QUESTIONS)

        assert response is fake_sdk.response
        assert len(fake_sdk.calls) == 1
        state, questions, extra = fake_sdk.calls[0]
        assert state is STATE
        assert questions is QUESTIONS
        assert extra == {}

    @pytest.mark.asyncio
    async def test_sdk_client_is_built_from_the_config(self, fake_sdk):
        client = DecisionModelClient(
            dm_config(
                model="my-org/rubric-dm-7b",
                api_base="https://xyz.endpoints.huggingface.cloud",
                api_key="hf-token",
                timeout=12.5,
                max_retries=4,
                extra_headers={"X-Team": "evals"},
            )
        )
        await client.system_one(STATE, QUESTIONS)

        kwargs = fake_sdk.clients[0].kwargs
        assert kwargs == {
            "api_key": "hf-token",
            "base_url": "https://xyz.endpoints.huggingface.cloud",
            "model": "my-org/rubric-dm-7b",
            "headers": {"X-Team": "evals"},
            "retry": kwargs["retry"],
            "timeout": 12.5,
        }
        retry = kwargs["retry"]
        assert isinstance(retry, RetryPolicy)
        # max_retries counts attempts; the SDK counts retries after the first attempt.
        assert retry.max_retries == 3
        # The retry budget: every attempt's full timeout plus the longest backoff wait each.
        budget = 4 * (12.5 + RetryPolicy().backoff_max)
        assert retry.timeout == budget
        assert retry == RetryPolicy(max_retries=3, timeout=budget)

    @pytest.mark.asyncio
    async def test_single_attempt_disables_sdk_retries(self, fake_sdk):
        await DecisionModelClient(dm_config(max_retries=1)).system_one(STATE, QUESTIONS)
        assert fake_sdk.clients[0].kwargs["retry"].max_retries == 0

    @pytest.mark.asyncio
    async def test_errors_propagate_unchanged(self, fake_sdk):
        fake_sdk.error = typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused")
        with pytest.raises(typesafe_sdk.TypeSafeAPIConnectionError):
            await DecisionModelClient(dm_config()).system_one(STATE, QUESTIONS)


# =============================================================================
# Event loops and SDK-client lifetime
# =============================================================================


class TestEventLoops:
    def test_each_event_loop_gets_its_own_sdk_client_closed_on_that_loop(self, fake_sdk):
        """Successive ``asyncio.run`` calls: a new SDK client per loop, closed before it ends."""
        client = DecisionModelClient(dm_config())
        loops: list[asyncio.AbstractEventLoop] = []

        async def grade_once() -> None:
            loops.append(asyncio.get_running_loop())
            await client.system_one(STATE, QUESTIONS)

        asyncio.run(grade_once())
        asyncio.run(grade_once())

        assert loops[0] is not loops[1]
        assert len(fake_sdk.clients) == 2
        for sdk_client, loop in zip(fake_sdk.clients, loops, strict=True):
            assert sdk_client.loop is loop
            assert sdk_client.closed_in is loop
            assert len(sdk_client.calls) == 1

    @pytest.mark.asyncio
    async def test_overlapping_requests_share_one_sdk_client(self, fake_sdk):
        fake_sdk.delay = 0.01
        client = DecisionModelClient(dm_config())

        await asyncio.gather(
            *(client.system_one({"submission": str(i)}, QUESTIONS) for i in range(5))
        )

        assert len(fake_sdk.clients) == 1
        shared = fake_sdk.clients[0]
        assert len(shared.calls) == 5
        assert shared.closed_in is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_a_request_after_all_others_finished_gets_a_new_sdk_client(self, fake_sdk):
        client = DecisionModelClient(dm_config())
        await client.system_one(STATE, QUESTIONS)
        await client.system_one(STATE, QUESTIONS)

        assert len(fake_sdk.clients) == 2
        assert all(c.closed_in is asyncio.get_running_loop() for c in fake_sdk.clients)

    @pytest.mark.asyncio
    async def test_sdk_client_is_closed_after_a_failed_request(self, fake_sdk):
        fake_sdk.error = typesafe_sdk.TypeSafeInternalServerError(503, None, _headers())
        client = DecisionModelClient(dm_config())
        with pytest.raises(typesafe_sdk.TypeSafeInternalServerError):
            await client.system_one(STATE, QUESTIONS)
        assert fake_sdk.clients[0].closed_in is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_a_failure_to_close_is_logged_not_raised(self, fake_sdk, caplog):
        """The request has already succeeded (or failed) on its own terms: a failure to close
        the SDK client must not replace its outcome, which would turn a good answer into a
        failed request (an ``unknown`` error, the worst case on every criterion)."""
        fake_sdk.close_error = RuntimeError("connection pool already torn down")
        client = DecisionModelClient(dm_config())

        with caplog.at_level(logging.WARNING, logger="autorubric.decision"):
            response = await client.system_one(STATE, QUESTIONS)

        assert response is fake_sdk.response
        assert fake_sdk.clients[0].closed_in is asyncio.get_running_loop()
        assert "Closing a decision-model SDK client failed" in caplog.text
        assert "connection pool already torn down" in caplog.text

    @pytest.mark.asyncio
    async def test_a_failure_to_close_does_not_mask_the_request_error(self, fake_sdk, caplog):
        fake_sdk.error = typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused")
        fake_sdk.close_error = RuntimeError("connection pool already torn down")
        client = DecisionModelClient(dm_config())

        with caplog.at_level(logging.WARNING, logger="autorubric.decision"):
            with pytest.raises(typesafe_sdk.TypeSafeAPIConnectionError):
                await client.system_one(STATE, QUESTIONS)

        assert "Closing a decision-model SDK client failed" in caplog.text

    @pytest.mark.asyncio
    async def test_sdk_client_is_closed_after_a_cancelled_request(self, fake_sdk):
        fake_sdk.gate = asyncio.Event()  # never set: the request hangs until cancelled
        client = DecisionModelClient(dm_config())
        task = asyncio.create_task(client.system_one(STATE, QUESTIONS))
        while not fake_sdk.clients or fake_sdk.in_flight == 0:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fake_sdk.clients[0].closed_in is asyncio.get_running_loop()


# =============================================================================
# Response cache
# =============================================================================


class TestCache:
    @pytest.mark.asyncio
    async def test_hit_makes_no_request_and_returns_the_original_response(
        self, fake_sdk, make_client, sdk_response
    ):
        fake_sdk.response = sdk_response(
            {
                "c0": {"type": "noul", "noul": 0.30000000000000004},
                "c1": {
                    "type": "choice",
                    "choice": "formal",
                    "confidence": 0.7,
                    "probabilities": {"formal": 0.85, "casual": 0.15},
                },
            },
            input_tokens=5021,
            output_tokens=2,
        )
        client = make_client(cache_enabled=True, input_cost_per_token=0.042e-6)
        first = await client.system_one(STATE, QUESTIONS)
        assert len(fake_sdk.calls) == 1

        second = await client.system_one(STATE, QUESTIONS)

        assert len(fake_sdk.calls) == 1  # served from the cache
        assert second == first
        assert second.model == first.model
        assert second.answers == first.answers
        assert second.usage == first.usage
        assert client.token_usage(second) == client.token_usage(first)
        assert client.completion_cost(second) == client.completion_cost(first)

    @pytest.mark.asyncio
    async def test_hit_is_served_to_a_new_client_on_the_same_store(
        self, fake_sdk, make_client, sdk_response
    ):
        fake_sdk.response = sdk_response(input_tokens=77)
        await make_client(cache_enabled=True).system_one(STATE, QUESTIONS)

        replayed = await make_client(cache_enabled=True).system_one(STATE, QUESTIONS)

        assert len(fake_sdk.calls) == 1
        assert replayed == fake_sdk.response
        assert replayed.usage.input_tokens == 77

    def test_hit_survives_event_loop_changes(self, fake_sdk, make_client):
        client = make_client(cache_enabled=True)
        first = asyncio.run(client.system_one(STATE, QUESTIONS))
        second = asyncio.run(client.system_one(STATE, QUESTIONS))
        assert second == first
        assert len(fake_sdk.calls) == 1

    @pytest.mark.asyncio
    async def test_failures_are_not_cached(self, fake_sdk, make_client):
        client = make_client(cache_enabled=True)
        fake_sdk.error = typesafe_sdk.TypeSafeRateLimitError(429, None, _headers())
        with pytest.raises(typesafe_sdk.TypeSafeRateLimitError):
            await client.system_one(STATE, QUESTIONS)
        assert len(client._ensure_cache()) == 0

        fake_sdk.error = None
        await client.system_one(STATE, QUESTIONS)
        assert len(fake_sdk.calls) == 2  # the failure was retried against the endpoint

    @pytest.mark.asyncio
    async def test_disabled_cache_always_asks(self, fake_sdk, make_client):
        client = make_client()
        await client.system_one(STATE, QUESTIONS)
        await client.system_one(STATE, QUESTIONS)
        assert len(fake_sdk.calls) == 2
        assert client._cache is None

    @pytest.mark.asyncio
    async def test_per_call_override(self, fake_sdk, make_client):
        enabled = make_client(cache_enabled=True)
        await enabled.system_one(STATE, QUESTIONS)
        await enabled.system_one(STATE, QUESTIONS, use_cache=False)
        assert len(fake_sdk.calls) == 2

        disabled = make_client(cache_dir=enabled.config.cache_dir)
        await disabled.system_one(STATE, QUESTIONS, use_cache=True)
        assert len(fake_sdk.calls) == 2  # forced cache use hit the shared store

    @pytest.mark.asyncio
    async def test_ttl_is_applied(self, fake_sdk, make_client):
        client = make_client(cache_enabled=True, cache_ttl=3600)
        await client.system_one(STATE, QUESTIONS)
        key = client._cache_key(STATE, QUESTIONS)
        _, expire_time = client._ensure_cache().get(key, expire_time=True)
        assert expire_time is not None

    @pytest.mark.asyncio
    async def test_shares_the_llm_response_store(self, fake_sdk, make_client, tmp_path):
        client = make_client(cache_enabled=True)
        await client.system_one(STATE, QUESTIONS)

        llm = LLMClient(LLMConfig(model="openai/gpt-5.2", cache_dir=tmp_path / "cache"))
        try:
            store = llm._ensure_cache()
            assert client._cache_key(STATE, QUESTIONS) in store
        finally:
            llm.close()

    @pytest.mark.asyncio
    async def test_unreadable_entry_is_a_miss_and_is_replaced(self, fake_sdk, make_client, caplog):
        client = make_client(cache_enabled=True)
        key = client._cache_key(STATE, QUESTIONS)
        client._ensure_cache().set(key, '{"not": "a System One response"}')

        with caplog.at_level(logging.WARNING, logger="autorubric.decision"):
            response = await client.system_one(STATE, QUESTIONS)

        assert response == fake_sdk.response
        assert len(fake_sdk.calls) == 1
        assert "cache" in caplog.text
        assert await client.system_one(STATE, QUESTIONS) == fake_sdk.response
        assert len(fake_sdk.calls) == 1  # the replacement entry is served


class TestCacheKey:
    def test_is_a_sha256_hex_digest(self, make_client):
        key = make_client()._cache_key(STATE, QUESTIONS)
        assert len(key) == 64
        int(key, 16)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"decision_threshold": 0.9},
            {"binary_framing": "choice", "ordinal_framing": "score"},
            {"input_cost_per_token": 1e-6},
            {"timeout": 5.0, "max_retries": 1, "max_parallel_requests": 2},
            {"extra_headers": {"X-Team": "evals"}},
            {"api_key": "another-key"},
            {"cache_ttl": 60},
        ],
        ids=["threshold", "framing", "price", "transport", "headers", "api_key", "ttl"],
    )
    def test_ignores_settings_that_do_not_change_the_request(self, make_client, overrides):
        """``decision_threshold`` only post-processes answers, so cached answers are reused."""
        baseline = make_client()._cache_key(STATE, QUESTIONS)
        assert make_client(**overrides)._cache_key(STATE, QUESTIONS) == baseline

    def test_changes_with_the_model(self, make_client):
        assert make_client(model="jev-a")._cache_key(STATE, QUESTIONS) != make_client(
            model="jev-b"
        )._cache_key(STATE, QUESTIONS)

    def test_changes_with_the_resolved_base_url(self, make_client, monkeypatch):
        default = make_client()._cache_key(STATE, QUESTIONS)
        explicit = make_client(api_base="https://dm.example.com")._cache_key(STATE, QUESTIONS)
        monkeypatch.setenv("TYPESAFE_BASE_URL", "https://env.example.com")
        from_env = make_client()._cache_key(STATE, QUESTIONS)
        assert len({default, explicit, from_env}) == 3

    def test_changes_with_the_state(self, make_client):
        client = make_client()
        other = {**STATE, "submission": "Plants eat sunlight."}
        assert client._cache_key(STATE, QUESTIONS) != client._cache_key(other, QUESTIONS)
        assert client._cache_key("text", QUESTIONS) != client._cache_key(["text"], QUESTIONS)

    def test_changes_with_the_questions(self, make_client):
        client = make_client()
        base = client._cache_key(STATE, QUESTIONS)
        variants = [
            {**QUESTIONS, "c0": Noul(instructions="Mentions sunlight")},
            {"c0": QUESTIONS["c0"]},
            {**QUESTIONS, "c2": Score(instructions="Clarity", criteria=["low", "high"])},
            {**QUESTIONS, "c0": Noul(instructions="Mentions light", criteria={"true": "yes"})},
        ]
        keys = {client._cache_key(STATE, v) for v in variants}
        assert base not in keys
        assert len(keys) == len(variants)

    def test_choice_option_order_is_part_of_the_request(self, make_client):
        """The endpoint sees option order, so reordered options are a different request."""
        client = make_client()
        a = {"c0": Choice(instructions="Tone", criteria={"formal": None, "casual": None})}
        b = {"c0": Choice(instructions="Tone", criteria={"casual": None, "formal": None})}
        assert client._cache_key(STATE, a) != client._cache_key(STATE, b)

    def test_equivalent_representations_share_a_key(self, make_client):
        """A question object and its wire-form dict, or any mapping type, are one request."""
        client = make_client()
        as_objects = {"c0": Noul(instructions="Mentions light")}
        as_dicts = {"c0": {"type": "noul", "instructions": "Mentions light"}}
        assert client._cache_key(STATE, as_objects) == client._cache_key(STATE, as_dicts)
        assert client._cache_key(types.MappingProxyType(STATE), as_objects) == (
            client._cache_key(STATE, as_objects)
        )


# =============================================================================
# Rate limiting
# =============================================================================


class TestRateLimit:
    @pytest.mark.parametrize(
        "api_base, expected",
        [
            (None, "decision-model@api.typesafe.ai"),
            (
                "https://XYZ.Endpoints.HuggingFace.cloud/v1",
                "decision-model@xyz.endpoints.huggingface.cloud",
            ),
            ("http://localhost:8080", "decision-model@localhost:8080"),
            ("https://dm.example.com:443", "decision-model@dm.example.com"),
            ("http://dm.example.com:80/", "decision-model@dm.example.com"),
            ("https://dm.example.com/tenant-a/v2", "decision-model@dm.example.com"),
            ("http://[::1]:9000", "decision-model@[::1]:9000"),
        ],
    )
    def test_key_is_the_host_of_the_resolved_base_url(self, api_base, expected):
        assert DecisionModelClient(dm_config(api_base=api_base)).rate_limit_key == expected

    def test_models_on_one_host_share_a_key_despite_provider_like_prefixes(self):
        a = DecisionModelClient(dm_config(model="jev-latest"))
        b = DecisionModelClient(dm_config(model="my-org/rubric-dm-7b"))
        assert a.rate_limit_key == b.rate_limit_key

    @pytest.mark.asyncio
    async def test_concurrency_is_limited_through_the_shared_pool(self, fake_sdk):
        fake_sdk.delay = 0.01
        client = DecisionModelClient(dm_config(max_parallel_requests=2))

        await asyncio.gather(
            *(client.system_one({"submission": str(i)}, QUESTIONS) for i in range(6))
        )

        assert fake_sdk.max_in_flight == 2
        assert RateLimitPool.get_instance().get_current_limit(client.rate_limit_key) == 2

    @pytest.mark.asyncio
    async def test_requests_queued_on_the_limit_share_one_sdk_client(self, fake_sdk):
        """The SDK client is leased before waiting on the limit, so requests queued behind it
        keep the client (and its connections) alive for one another instead of each
        building, and tearing down, a client of its own."""
        fake_sdk.delay = 0.01
        client = DecisionModelClient(dm_config(max_parallel_requests=1))

        await asyncio.gather(
            *(client.system_one({"submission": str(i)}, QUESTIONS) for i in range(4))
        )

        assert fake_sdk.max_in_flight == 1
        assert len(fake_sdk.clients) == 1
        assert len(fake_sdk.clients[0].calls) == 4
        assert fake_sdk.clients[0].closed_in is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_same_host_shares_the_strictest_limit(self, fake_sdk):
        fake_sdk.delay = 0.01
        loose = DecisionModelClient(dm_config(model="jev-a", max_parallel_requests=3))
        strict = DecisionModelClient(dm_config(model="my-org/b", max_parallel_requests=1))
        await strict.system_one(STATE, QUESTIONS)

        await asyncio.gather(
            *(loose.system_one({"submission": str(i)}, QUESTIONS) for i in range(4))
        )

        assert fake_sdk.max_in_flight == 1

    @pytest.mark.asyncio
    async def test_unlimited_by_default(self, fake_sdk):
        fake_sdk.delay = 0.01
        client = DecisionModelClient(dm_config())
        await asyncio.gather(
            *(client.system_one({"submission": str(i)}, QUESTIONS) for i in range(4))
        )
        assert fake_sdk.max_in_flight == 4
        assert RateLimitPool.get_instance().get_current_limit(client.rate_limit_key) is None


# =============================================================================
# Usage and cost
# =============================================================================


class TestUsageAndCost:
    def test_usage_maps_to_token_usage(self, sdk_response):
        usage = DecisionModelClient.token_usage(sdk_response(input_tokens=5021, output_tokens=3))
        assert usage == TokenUsage(prompt_tokens=5021, completion_tokens=3, total_tokens=5024)

    def test_unreported_token_counts_count_as_zero_like_llm_usage(self, sdk_response):
        response = sdk_response(input_tokens=None, output_tokens=None)
        assert DecisionModelClient.token_usage(response) == TokenUsage()

    def test_cost_is_none_without_a_price(self, sdk_response):
        client = DecisionModelClient(dm_config())
        assert client.completion_cost(sdk_response(input_tokens=5021)) is None

    def test_cost_is_input_tokens_times_price(self, sdk_response):
        client = DecisionModelClient(dm_config(input_cost_per_token=0.042e-6))
        cost = client.completion_cost(sdk_response(input_tokens=5021, output_tokens=900))
        assert cost == pytest.approx(5021 * 0.042e-6)

    def test_zero_price_is_a_known_zero_cost(self, sdk_response):
        client = DecisionModelClient(dm_config(input_cost_per_token=0.0))
        assert client.completion_cost(sdk_response(input_tokens=5021)) == 0.0

    def test_cost_is_none_when_input_tokens_are_unreported(self, sdk_response):
        client = DecisionModelClient(dm_config(input_cost_per_token=0.042e-6))
        assert client.completion_cost(sdk_response(input_tokens=None)) is None


# =============================================================================
# The real SDK client on an in-memory endpoint
# =============================================================================

_OK_BODY = {
    "model": "jev-2026-09-15",
    "usage": {"input_tokens": 321, "output_tokens": 2},
    "answers": {
        "c0": {"type": "noul", "noul": 0.91},
        "c1": {
            "type": "choice",
            "choice": "casual",
            "confidence": 0.4,
            "probabilities": {"formal": 0.4, "casual": 0.6},
        },
    },
}
_NO_WAIT = {"retry-after-ms": "0"}  # retried immediately, so the tests do not sleep


def _headers() -> httpx2.Headers:
    return httpx2.Headers()


class TestRealSdkClient:
    @pytest.mark.asyncio
    async def test_wire_request(self, http_endpoint):
        http_endpoint.reply(200, _OK_BODY)
        client = DecisionModelClient(
            dm_config(
                api_base="https://dm.example.com/",
                api_key="sk-test",
                timeout=12.5,
                extra_headers={"X-Team": "evals"},
            )
        )

        response = await client.system_one(STATE, QUESTIONS)

        (request,) = http_endpoint.requests
        assert request.method == "POST"
        assert str(request.url) == "https://dm.example.com/v1/systemone"
        # The request URL the client checks at construction is the one the SDK requests.
        assert str(request.url) == client.base_url + _SYSTEM_ONE_PATH
        assert json.loads(request.content) == {
            "state": STATE,
            "model": "jev-latest",
            "questions": {
                "c0": {"type": "noul", "instructions": "Mentions light"},
                "c1": {
                    "type": "choice",
                    "instructions": "Tone",
                    "criteria": {"formal": None, "casual": None},
                },
            },
        }
        assert request.headers["authorization"] == "Bearer sk-test"
        assert request.headers["x-team"] == "evals"
        assert request.extensions["timeout"] == {
            "connect": 12.5,
            "read": 12.5,
            "write": 12.5,
            "pool": 12.5,
        }
        assert response.answers["c0"].noul == 0.91
        assert response.usage.input_tokens == 321

    @pytest.mark.asyncio
    async def test_transient_failures_are_retried_up_to_max_retries_attempts(self, http_endpoint):
        http_endpoint.reply(503, {"error": "busy"}, _NO_WAIT)
        http_endpoint.reply(429, {"error": "slow down"}, _NO_WAIT)
        http_endpoint.reply(200, _OK_BODY)
        client = DecisionModelClient(dm_config(max_retries=3))

        response = await client.system_one(STATE, QUESTIONS)

        assert response.answers["c0"].noul == 0.91
        assert len(http_endpoint.requests) == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries_attempts(self, http_endpoint):
        for _ in range(3):
            http_endpoint.reply(503, {"error": "busy"}, _NO_WAIT)
        client = DecisionModelClient(dm_config(max_retries=2))

        with pytest.raises(typesafe_sdk.TypeSafeInternalServerError) as excinfo:
            await client.system_one(STATE, QUESTIONS)

        assert len(http_endpoint.requests) == 2
        assert classify_grading_error(excinfo.value) == "infrastructure"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("retry_after", [{"retry-after": "3600"}, {"retry-after-ms": "9e9"}])
    async def test_a_retry_after_beyond_the_budget_fails_at_once(self, http_endpoint, retry_after):
        """The endpoint's ``Retry-After`` is honoured only within the retry budget
        (``max_retries * (timeout + backoff_max)``, 45 s here): a server asking for an hour
        is not waited for; the request fails with the 429 and its criteria abstain as
        ``infrastructure``."""
        http_endpoint.reply(429, {"error": "quota exhausted"}, retry_after)
        http_endpoint.reply(200, _OK_BODY)
        client = DecisionModelClient(dm_config(max_retries=3, timeout=10.0))

        with pytest.raises(typesafe_sdk.TypeSafeRateLimitError) as excinfo:
            await asyncio.wait_for(client.system_one(STATE, QUESTIONS), timeout=5)

        assert len(http_endpoint.requests) == 1
        assert classify_grading_error(excinfo.value) == "infrastructure"

    @pytest.mark.asyncio
    async def test_a_retry_after_within_the_budget_is_honoured(self, http_endpoint, monkeypatch):
        """The retry waits exactly the endpoint's ``Retry-After``. The wait the retry loop asks
        ``asyncio.sleep`` for is recorded (and still slept), rather than timed with the
        clock, whose resolution and early timer firing on some platforms (Windows before
        Python 3.13 ticks every 15.6 ms) would make a wall-clock bound flaky."""
        waits: list[float] = []
        real_sleep = asyncio.sleep

        async def recording_sleep(delay: float, *args, **kwargs):
            waits.append(delay)
            return await real_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(asyncio, "sleep", recording_sleep)
        http_endpoint.reply(429, {"error": "slow down"}, {"retry-after-ms": "50"})
        http_endpoint.reply(200, _OK_BODY)
        client = DecisionModelClient(dm_config(max_retries=2, timeout=1.0))

        response = await client.system_one(STATE, QUESTIONS)

        assert [wait for wait in waits if wait > 0] == [pytest.approx(0.05)]
        assert response.answers["c0"].noul == 0.91
        assert len(http_endpoint.requests) == 2

    @pytest.mark.asyncio
    async def test_the_budget_never_cuts_backoff_retries_short(self, http_endpoint):
        """Without ``Retry-After`` the SDK backs off exponentially (0.5 s, then 1 s). Even with
        a timeout shorter than those waits, every configured attempt is made: the budget has
        room for each attempt's timeout plus the longest backoff wait."""
        for _ in range(3):
            http_endpoint.reply(503, {"error": "busy"})
        client = DecisionModelClient(dm_config(max_retries=3, timeout=0.2))

        with pytest.raises(typesafe_sdk.TypeSafeInternalServerError):
            await client.system_one(STATE, QUESTIONS)

        assert len(http_endpoint.requests) == 3

    @pytest.mark.parametrize(
        "status, error_type, category",
        [
            (400, typesafe_sdk.TypeSafeBadRequestError, "parse"),
            (422, typesafe_sdk.TypeSafeUnprocessableEntityError, "parse"),
            (401, typesafe_sdk.TypeSafeAuthenticationError, "infrastructure"),
            (404, typesafe_sdk.TypeSafeNotFoundError, "infrastructure"),
        ],
    )
    @pytest.mark.asyncio
    async def test_rejections_are_not_retried(self, http_endpoint, status, error_type, category):
        http_endpoint.reply(status, {"detail": "rejected"})
        client = DecisionModelClient(dm_config())

        with pytest.raises(error_type) as excinfo:
            await client.system_one(STATE, QUESTIONS)

        assert len(http_endpoint.requests) == 1
        assert classify_grading_error(excinfo.value) == category

    @pytest.mark.asyncio
    async def test_invalid_response_is_a_parse_failure_and_is_not_cached(
        self, http_endpoint, make_client
    ):
        http_endpoint.reply(200, {"model": "jev", "answers": {"c0": {"type": "noul"}}})
        http_endpoint.reply(200, _OK_BODY)
        client = make_client(cache_enabled=True)

        with pytest.raises(typesafe_sdk.TypeSafeAPIResponseValidationError) as excinfo:
            await client.system_one(STATE, QUESTIONS)
        assert classify_grading_error(excinfo.value) == "parse"

        response = await client.system_one(STATE, QUESTIONS)
        assert response.usage.input_tokens == 321
        assert len(http_endpoint.requests) == 2

    @pytest.mark.asyncio
    async def test_cached_response_round_trips_losslessly(self, http_endpoint, make_client):
        http_endpoint.reply(200, _OK_BODY)
        live = await make_client(cache_enabled=True).system_one(STATE, QUESTIONS)

        cached = await make_client(cache_enabled=True).system_one(STATE, QUESTIONS)

        assert len(http_endpoint.requests) == 1
        assert cached == live
        assert cached.model_dump() == live.model_dump()
        assert cached.choices["c1"].probabilities == {"formal": 0.4, "casual": 0.6}


# =============================================================================
# Error classification of every TypeSafe exception
# =============================================================================


def _typesafe_errors() -> list[tuple[BaseException, str]]:
    h = _headers()
    return [
        (typesafe_sdk.TypeSafeAPIResponseValidationError(200, {}, h, "answers.c0"), "parse"),
        (typesafe_sdk.TypeSafeBadRequestError(400, {"detail": "bad"}, h), "parse"),
        (typesafe_sdk.TypeSafeUnprocessableEntityError(422, {"detail": "long"}, h), "parse"),
        (typesafe_sdk.TypeSafeAuthenticationError(401, None, h), "infrastructure"),
        (typesafe_sdk.TypeSafePermissionDeniedError(403, None, h), "infrastructure"),
        (typesafe_sdk.TypeSafeNotFoundError(404, None, h), "infrastructure"),
        (typesafe_sdk.TypeSafeRateLimitError(429, None, h), "infrastructure"),
        (typesafe_sdk.TypeSafeInternalServerError(503, None, h), "infrastructure"),
        (typesafe_sdk.TypeSafeAPIError(418, None, h), "infrastructure"),
        (typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused"), "infrastructure"),
        (typesafe_sdk.TypeSafeAPITimeoutError(60.0), "infrastructure"),
        (typesafe_sdk.TypeSafeError("No API key was provided."), "infrastructure"),
    ]


class TestClassifyTypeSafeErrors:
    @pytest.mark.parametrize(
        "exc, expected",
        _typesafe_errors(),
        ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else v,
    )
    def test_category(self, exc, expected):
        assert classify_grading_error(exc) == expected

    def test_every_public_sdk_exception_is_covered(self):
        covered = {type(exc) for exc, _ in _typesafe_errors()}
        public = {
            getattr(typesafe_sdk, name)
            for name in typesafe_sdk.__all__
            if isinstance(getattr(typesafe_sdk, name), type)
            and issubclass(getattr(typesafe_sdk, name), BaseException)
        }
        assert public <= covered

    def test_the_sdk_branch_takes_precedence_over_value_error(self):
        """A TypeSafe error is routed by the TypeSafe rules even if it is also a ValueError."""

        class ValueFlavouredError(typesafe_sdk.TypeSafeError, ValueError):
            pass

        class RejectedRequest(typesafe_sdk.TypeSafeBadRequestError):
            pass

        assert classify_grading_error(ValueFlavouredError("x")) == "infrastructure"
        assert classify_grading_error(RejectedRequest(400, None, _headers())) == "parse"

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (TimeoutError("t"), "unknown"),
            (ConnectionError("c"), "unknown"),
            (ValueError("v"), "parse"),
            (RuntimeError("r"), "unknown"),
        ],
    )
    def test_non_sdk_errors_are_unchanged(self, exc, expected):
        assert classify_grading_error(exc) == expected


@pytest.mark.asyncio
async def test_the_network_guard_blocks_real_requests():
    """Safety net for this suite: an unmocked SDK client fails instead of reaching TypeSafe."""
    client = DecisionModelClient(dm_config())
    with pytest.raises(AssertionError, match="tried to reach the network"):
        await client.system_one(STATE, QUESTIONS)
