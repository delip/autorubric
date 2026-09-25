"""Fixtures for the decision-model tests.

Nothing here reaches the network. Two independent guards make sure of it:

- ``_no_network`` makes every real ``httpx2`` transport refuse to send, so even a request
  built by the real SDK fails loudly instead of reaching TypeSafe (a real call costs money).
- ``_isolate_typesafe_env`` removes the ``TYPESAFE_*`` variables, which autorubric's
  ``.env`` loading may have set from a developer's real credentials.

Tests either replace ``typesafe_sdk.AsyncTypeSafeClient`` with a recording fake
(``fake_sdk``) or keep the real SDK client and give it an in-memory ``httpx2.MockTransport``
(``http_endpoint``), which exercises the SDK's own request building, retries and errors.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest
import typesafe_sdk
from typesafe_sdk import SystemOneResponse

from autorubric.rate_limit import RateLimitPool

TYPESAFE_ENV_VARS = ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL")


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
    endpoint = MockEndpoint()
    real_client: Callable[..., Any] = typesafe_sdk.AsyncTypeSafeClient

    def build(**kwargs: Any) -> Any:
        endpoint.client_kwargs.append(kwargs)
        return real_client(transport=httpx2.MockTransport(endpoint.handle), **kwargs)

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", build)
    return endpoint
