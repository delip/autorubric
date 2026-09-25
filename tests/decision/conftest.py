"""Fixtures for the decision-model tests.

Nothing here reaches the network. Two independent guards make sure of it:

- ``_no_network`` makes every real ``httpx2`` transport refuse to send, so even a request
  built by the real SDK fails loudly instead of reaching TypeSafe (a real call costs money).
- ``_isolate_typesafe_env`` removes the ``TYPESAFE_*`` variables, which autorubric's
  ``.env`` loading may have set from a developer's real credentials.

Tests either replace ``typesafe_sdk.AsyncTypeSafeClient`` with a recording fake
(``fake_sdk``) or keep the real SDK client. The real client either sends through an
in-memory handler in place of the HTTP transport (``http_endpoint``), which exercises the
SDK's own request building, retries and errors, or runs on the whole HTTP stack, connection
pool included, over an in-memory network (``http_network``).
"""

from __future__ import annotations

import asyncio
import http
import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpcore2
import httpx2
import pytest
import typesafe_sdk
from typesafe_sdk import SystemOneResponse

from autorubric.rate_limit import RateLimitPool

TYPESAFE_ENV_VARS = ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL")

# Saved before ``_no_network`` replaces it, so ``http_network`` can put the real transport
# back once every connection it opens is an in-memory one.
_REAL_ASYNC_HANDLE_REQUEST = httpx2.AsyncHTTPTransport.handle_async_request


@pytest.fixture(autouse=True)
def _isolate_typesafe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in TYPESAFE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse_async(self: Any, request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"a test tried to reach the network: {request.url}")

    def refuse_sync(self: Any, request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"a test tried to reach the network: {request.url}")

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", refuse_async)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", refuse_sync)


@pytest.fixture(autouse=True)
def _fresh_rate_limit_pool() -> Any:
    RateLimitPool.reset_instance()
    yield
    RateLimitPool.reset_instance()


HTTP_ENVIRONMENT_FAILURES = (
    "missing-ca-file",
    "socks-proxy-without-socksio",
    "unknown-proxy-scheme",
)
"""Ways to break the environment ``httpx2`` builds HTTP clients in (see ``HttpEnvironment``)."""


@dataclass
class HttpEnvironment:
    """The environment ``httpx2`` reads whenever it builds an HTTP client, and one way to
    break it.

    That environment is the TLS trust (``SSL_CERT_FILE``, ``SSL_CERT_DIR``) and the proxies
    (every ``*_proxy`` variable, in any letter case). Changes are undone at teardown.
    """

    monkeypatch: pytest.MonkeyPatch
    tmp_path: Path
    how: str
    """One of ``HTTP_ENVIRONMENT_FAILURES``: ``SSL_CERT_FILE`` naming a missing file
    (``FileNotFoundError``), a SOCKS proxy without the ``socksio`` package (``ImportError``),
    or a proxy URL whose scheme no proxy has (``ValueError``)."""

    def clear(self) -> None:
        """Unset every TLS and proxy variable, leaving an environment httpx2 can build in."""
        for name in list(os.environ):
            if name.upper() in ("SSL_CERT_FILE", "SSL_CERT_DIR") or name.lower().endswith("_proxy"):
                self.monkeypatch.delenv(name)

    def break_(self) -> None:
        """Make the environment one in which httpx2 cannot build an HTTP client."""
        self.clear()
        if self.how == "missing-ca-file":
            self.monkeypatch.setenv("SSL_CERT_FILE", str(self.tmp_path / "missing-ca.pem"))
        elif self.how == "socks-proxy-without-socksio":
            self.monkeypatch.setitem(sys.modules, "socksio", None)  # its import fails
            self.monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
        elif self.how == "unknown-proxy-scheme":
            self.monkeypatch.setenv("HTTPS_PROXY", "ftp://127.0.0.1:2121")
        else:
            raise ValueError(f"unknown way to break the environment: {self.how!r}")


@pytest.fixture(params=HTTP_ENVIRONMENT_FAILURES)
def http_environment(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> HttpEnvironment:
    """A cleared HTTP environment, which the test breaks (``break_``) when it chooses.

    Parametrized: a test using it runs once for each of ``HTTP_ENVIRONMENT_FAILURES``.
    """
    environment = HttpEnvironment(monkeypatch, tmp_path, request.param)
    environment.clear()
    return environment


def make_response(
    answers: dict[str, Any] | None = None,
    *,
    input_tokens: int | None = 1234,
    output_tokens: int | None = 7,
    model: str = "jev-2026-09-15",
) -> SystemOneResponse:
    """Build a ``SystemOneResponse`` the way the SDK does, from its JSON wire form."""
    if answers is None:
        answers = {"c0": {"type": "noul", "noul": 0.8312}}
    usage: dict[str, int] = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    payload = {"model": model, "usage": usage, "answers": answers}
    return SystemOneResponse.model_validate_json(json.dumps(payload))


@pytest.fixture
def sdk_response() -> Callable[..., SystemOneResponse]:
    """The ``make_response`` builder, as a fixture so test modules need not import conftest."""
    return make_response


class FakeSDKClient:
    """Stand-in for ``typesafe_sdk.AsyncTypeSafeClient`` that records its lifecycle."""

    def __init__(self, sdk: FakeSDK, **kwargs: Any) -> None:
        self.sdk = sdk
        self.kwargs = kwargs
        self.loop = asyncio.get_running_loop()
        self.closed_in: asyncio.AbstractEventLoop | None = None
        self.calls: list[tuple[Any, Any, dict[str, Any]]] = []
        sdk.clients.append(self)

    async def system_one(self, state: Any, questions: Any, **kwargs: Any) -> SystemOneResponse:
        self.calls.append((state, questions, kwargs))
        return await self.sdk.respond()

    async def aclose(self) -> None:
        assert self.closed_in is None, "closed twice"
        self.closed_in = asyncio.get_running_loop()
        # Like the SDK client, close the HTTP client it was given.
        http_client = self.kwargs.get("http_client")
        if http_client is not None:
            await http_client.aclose()
        if self.sdk.close_error is not None:
            raise self.sdk.close_error


@dataclass
class FakeSDK:
    """Records every fake SDK client built and controls how requests are answered."""

    clients: list[FakeSDKClient] = field(default_factory=list)
    response: SystemOneResponse = field(default_factory=make_response)
    error: BaseException | None = None
    close_error: BaseException | None = None
    delay: float = 0.0
    gate: asyncio.Event | None = None
    in_flight: int = 0
    max_in_flight: int = 0

    @property
    def calls(self) -> list[tuple[Any, Any, dict[str, Any]]]:
        return [call for client in self.clients for call in client.calls]

    async def respond(self) -> SystemOneResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    sdk = FakeSDK()
    monkeypatch.setattr(
        typesafe_sdk, "AsyncTypeSafeClient", lambda **kwargs: FakeSDKClient(sdk, **kwargs)
    )
    return sdk


@dataclass
class MockEndpoint:
    """An in-memory System One endpoint behind the real SDK client."""

    replies: list[tuple[int, Any, dict[str, str]]] = field(default_factory=list)
    requests: list[httpx2.Request] = field(default_factory=list)
    client_kwargs: list[dict[str, Any]] = field(default_factory=list)

    def reply(self, status: int, body: Any, headers: dict[str, str] | None = None) -> None:
        self.replies.append((status, body, headers or {}))

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        status, body, headers = self.replies.pop(0)
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx2.Response(
            status,
            content=content,
            headers={"content-type": "application/json", **headers},
            request=request,
        )


@pytest.fixture
def http_endpoint(monkeypatch: pytest.MonkeyPatch) -> MockEndpoint:
    """The real SDK client, with every request its HTTP client sends answered by ``handle``.

    The SDK client keeps the HTTP client autorubric gives it (timeouts included); only the
    transport's sending is replaced, the way ``httpx2.MockTransport`` answers a request.
    """
    endpoint = MockEndpoint()
    real_client: Callable[..., Any] = typesafe_sdk.AsyncTypeSafeClient

    def build(**kwargs: Any) -> Any:
        endpoint.client_kwargs.append(kwargs)
        return real_client(**kwargs)

    async def handle_async_request(self: Any, request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        return endpoint.handle(request)

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", build)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", handle_async_request)
    return endpoint


# A valid System One answer, served by ``http_network`` for status 200.
_NETWORK_OK_BODY = {
    "model": "jev-2026-09-15",
    "usage": {"input_tokens": 1, "output_tokens": 0},
    "answers": {"c0": {"type": "noul", "noul": 0.9}},
}


@dataclass
class InMemoryNetwork:
    """A System One endpoint on an in-memory network, under the real HTTP stack.

    Every connection the HTTP stack opens is an in-memory stream, so requests go through
    the real connection pool (limits, waiting for a connection, the pool's timeout) and
    reach ``answer``, which returns the status to reply with and may wait first. A 200
    carries a valid System One answer; any other status tells the SDK to retry at once
    (``retry-after-ms: 0``). The streams ignore their read and write timeouts: the endpoint
    is never slow unless ``answer`` makes it wait, and even then only the pool's own
    timeout, waiting for a connection, can fire.
    """

    answer: Callable[[bytes], Awaitable[int]]
    connections: int = 0
    requests: list[bytes] = field(default_factory=list)
    in_flight: int = 0
    max_in_flight: int = 0
    client_kwargs: list[dict[str, Any]] = field(default_factory=list)

    async def respond(self, request: bytes) -> bytes:
        """The raw HTTP response to ``request`` (the bytes written since the last one)."""
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            status = await self.answer(request)
        finally:
            self.in_flight -= 1
        body = json.dumps(_NETWORK_OK_BODY if status == 200 else {"error": "busy"}).encode()
        head = (
            f"HTTP/1.1 {status} {http.HTTPStatus(status).phrase}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            "retry-after-ms: 0\r\n\r\n"
        )
        return head.encode() + body

    async def wait_for_in_flight(self, count: int) -> None:
        """Wait until ``count`` requests are at the endpoint at once (at most 5 s)."""

        async def poll() -> None:
            while self.in_flight < count:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(poll(), timeout=5)


class _InMemoryStream(httpcore2.AsyncNetworkStream):
    """One connection to an ``InMemoryNetwork``: each read answers what was written."""

    def __init__(self, network: InMemoryNetwork) -> None:
        self._network = network
        self._written = b""

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._written += buffer

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        request, self._written = self._written, b""
        return await self._network.respond(request)

    async def aclose(self) -> None:
        pass

    async def start_tls(
        self, ssl_context: Any, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore2.AsyncNetworkStream:
        return self

    def get_extra_info(self, info: str) -> Any:
        return None


class _InMemoryBackend(httpcore2.AsyncNetworkBackend):
    def __init__(self, network: InMemoryNetwork) -> None:
        self._network = network

    async def connect_tcp(self, *args: Any, **kwargs: Any) -> httpcore2.AsyncNetworkStream:
        self._network.connections += 1
        return _InMemoryStream(self._network)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@pytest.fixture
def http_network(monkeypatch: pytest.MonkeyPatch) -> InMemoryNetwork:
    """The real SDK client on the whole HTTP stack over an in-memory network.

    Every connection pool is built on the in-memory network backend, so the real transport
    that ``_no_network`` disabled can be put back; the default backend's TCP connections
    are refused in its place, so a pool that somehow missed the in-memory backend still
    cannot reach the network. The test sets ``answer`` before any request is made.
    """

    async def unset(request: bytes) -> int:
        raise AssertionError("the test did not set InMemoryNetwork.answer")

    network = InMemoryNetwork(answer=unset)
    real_client: Callable[..., Any] = typesafe_sdk.AsyncTypeSafeClient
    real_pool_init = httpcore2.AsyncConnectionPool.__init__

    def build(**kwargs: Any) -> Any:
        network.client_kwargs.append(kwargs)
        return real_client(**kwargs)

    def pool_init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_pool_init(self, *args, **{**kwargs, "network_backend": _InMemoryBackend(network)})

    async def refuse_tcp(self: Any, host: str, port: int, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a test tried to reach the network: {host}:{port}")

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", build)
    monkeypatch.setattr(httpcore2.AsyncConnectionPool, "__init__", pool_init)
    monkeypatch.setattr(httpcore2.AnyIOBackend, "connect_tcp", refuse_tcp)
    monkeypatch.setattr(
        httpx2.AsyncHTTPTransport, "handle_async_request", _REAL_ASYNC_HANDLE_REQUEST
    )
    return network
