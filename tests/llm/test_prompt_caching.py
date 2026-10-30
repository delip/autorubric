"""How ``LLMClient`` arranges a request for prompt caches, and reads pre-1.7.0 replies."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import diskcache
import pytest

from autorubric.llm import (
    LLMClient,
    LLMConfig,
    _caches_prompts,
    _is_anthropic_model,
    _prompt_cache_min_tokens,
)

SHARED = "<submission>\nS\n</submission>"
PROMPT = f"{SHARED}\n\n<criterion>\nC\n</criterion>"
TAIL = "<criterion>\nC\n</criterion>"


def _response() -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = "ok"
    response.choices[0].message.reasoning_content = None
    response.choices[0].message.thinking_blocks = None
    response.usage = None
    return response


async def _messages(model: str, *, prompt_caching: bool = True, **generate: Any) -> list[dict]:
    client = LLMClient(LLMConfig(model=model, prompt_caching=prompt_caching))
    with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as completion:
        completion.return_value = _response()
        await client.generate("SYS", PROMPT, use_cache=False, **generate)
    return completion.call_args.kwargs["messages"]


class TestMessages:
    @pytest.mark.asyncio
    async def test_split_for_a_caching_model(self) -> None:
        assert await _messages("gpt-5.6-luna", shared_prefix=SHARED) == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": SHARED},
            {"role": "user", "content": TAIL},
        ]

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4-5",
            "claude-sonnet-4-5",
            "openrouter/anthropic/claude-sonnet-4.5",
        ],
    )
    @pytest.mark.asyncio
    async def test_breakpoints_on_claude(self, model: str) -> None:
        with patch("autorubric.llm._caches_prompts", return_value=True):
            messages = await _messages(model, shared_prefix=SHARED)
        mark = {"type": "ephemeral"}
        assert messages[0]["content"] == [{"type": "text", "text": "SYS", "cache_control": mark}]
        assert messages[1]["content"] == [{"type": "text", "text": SHARED, "cache_control": mark}]
        assert messages[2] == {"role": "user", "content": TAIL}

    @pytest.mark.asyncio
    async def test_one_message_for_a_model_not_known_to_cache(self) -> None:
        assert await _messages("openai/local-model", shared_prefix=SHARED) == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": PROMPT},
        ]

    @pytest.mark.asyncio
    async def test_one_message_without_prompt_caching(self) -> None:
        messages = await _messages(
            "anthropic/claude-sonnet-4-5", prompt_caching=False, shared_prefix=SHARED
        )
        assert messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": PROMPT},
        ]

    @pytest.mark.parametrize(
        "model",
        ["bedrock/anthropic.claude-sonnet-4-5-v1:0", "vertex_ai/claude-sonnet-4-5"],
    )
    @pytest.mark.asyncio
    async def test_bedrock_and_vertex_claude_are_unmarked_and_joined(self, model: str) -> None:
        assert await _messages(model, shared_prefix=SHARED) == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": PROMPT},
        ]

    @pytest.mark.asyncio
    async def test_callers_without_a_shared_prefix_are_unchanged(self) -> None:
        assert await _messages("gpt-5.6-luna") == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": PROMPT},
        ]

    @pytest.mark.asyncio
    async def test_a_prefix_that_is_not_one_stays_joined(self) -> None:
        messages = await _messages("gpt-5.6-luna", shared_prefix="something else")
        assert messages[1:] == [{"role": "user", "content": PROMPT}]


class TestModelLookups:
    def test_routed_claude_is_recognized(self) -> None:
        assert _is_anthropic_model("openrouter/anthropic/claude-sonnet-4.5")
        assert not _is_anthropic_model("gpt-5.6-luna")

    def test_lookup_failure_means_not_caching(self) -> None:
        with patch("autorubric.llm.litellm.utils.supports_prompt_caching", side_effect=KeyError):
            assert _caches_prompts("gpt-lookup-failure-test") is False

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.6-luna",
            "gemini/gemini-3.8-flash",
            "anthropic/claude-sonnet-4-5",
            "openrouter/anthropic/claude-sonnet-4.5",
            "openrouter/deepseek/deepseek-v4.1-flash",
        ],
    )
    def test_the_real_lookup_confirms_these_models(self, model: str) -> None:
        assert _caches_prompts(model) is True

    def test_unknown_model_does_not_cache(self) -> None:
        assert _caches_prompts("no-such-provider/no-such-model") is False

    def test_min_tokens_gemini_and_fallback(self) -> None:
        assert _prompt_cache_min_tokens("gemini/gemini-3.8-flash") == 4096
        assert _prompt_cache_min_tokens("gpt-5.6-luna") == 1024

    def test_recorded_minimum_wins(self) -> None:
        assert _prompt_cache_min_tokens("claude-opus-5-5") == 512


def _key_written_by_1_6(model: str, system: str, user: str) -> str:
    """The key format of 1.6.1, spelled out so a change to ``_cache_key`` fails here."""
    content = (
        f"{model}:{system}:{user}:str:temp=None:top_p=None:max_tokens=None:thinking=none:seed=None"
    )
    return hashlib.sha256(content.encode()).hexdigest()


class TestResponseCacheFallback:
    OLD = "<criterion>\nC\n</criterion>\n\n<submission>\nS\n</submission>"

    def _client(self, tmp_path: Path, **config: Any) -> LLMClient:
        return LLMClient(
            LLMConfig(model="gpt-5.6-luna", cache_enabled=True, cache_dir=tmp_path, **config)
        )

    def _seed(self, tmp_path: Path, key: str, value: str) -> None:
        with diskcache.Cache(directory=str(tmp_path)) as cache:
            cache.set(key, value)

    async def _generate(self, client: LLMClient, **kwargs: Any) -> tuple[Any, AsyncMock]:
        with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as completion:
            completion.return_value = _response()
            reply = await client.generate(
                "SYS", PROMPT, shared_prefix=SHARED, pre_1_7_user_prompt=self.OLD, **kwargs
            )
        return reply, completion

    @pytest.mark.asyncio
    async def test_old_cached_reply_hits_with_no_call(self, tmp_path: Path) -> None:
        self._seed(tmp_path, _key_written_by_1_6("gpt-5.6-luna", "SYS", self.OLD), "old reply")
        reply, completion = await self._generate(self._client(tmp_path))
        assert reply == "old reply"
        completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_entry_wins_over_old(self, tmp_path: Path) -> None:
        self._seed(tmp_path, _key_written_by_1_6("gpt-5.6-luna", "SYS", self.OLD), "old reply")
        self._seed(tmp_path, _key_written_by_1_6("gpt-5.6-luna", "SYS", PROMPT), "new reply")
        reply, _ = await self._generate(self._client(tmp_path))
        assert reply == "new reply"

    @pytest.mark.asyncio
    async def test_fallback_writes_nothing(self, tmp_path: Path) -> None:
        self._seed(tmp_path, _key_written_by_1_6("gpt-5.6-luna", "SYS", self.OLD), "old reply")
        await self._generate(self._client(tmp_path))
        with diskcache.Cache(directory=str(tmp_path)) as cache:
            assert list(cache.iterkeys()) == [_key_written_by_1_6("gpt-5.6-luna", "SYS", self.OLD)]

    @pytest.mark.asyncio
    async def test_a_new_reply_is_stored_under_the_new_key_only(self, tmp_path: Path) -> None:
        reply, completion = await self._generate(self._client(tmp_path))
        assert reply == "ok"
        completion.assert_called_once()
        with diskcache.Cache(directory=str(tmp_path)) as cache:
            assert list(cache.iterkeys()) == [_key_written_by_1_6("gpt-5.6-luna", "SYS", PROMPT)]

    @pytest.mark.asyncio
    async def test_changed_temperature_misses(self, tmp_path: Path) -> None:
        self._seed(tmp_path, _key_written_by_1_6("gpt-5.6-luna", "SYS", self.OLD), "old reply")
        _, completion = await self._generate(self._client(tmp_path, temperature=0.5))
        completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_namespaced_client_misses_a_plain_entry(self, tmp_path: Path) -> None:
        self._seed(tmp_path, _key_written_by_1_6("gpt-5.6-luna", "SYS", self.OLD), "old reply")
        client = LLMClient(
            LLMConfig(model="gpt-5.6-luna", cache_enabled=True, cache_dir=tmp_path),
            cache_namespace="judge_a",
        )
        _, completion = await self._generate(client)
        completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_use_cache_false_skips_the_fallback(self, tmp_path: Path) -> None:
        self._seed(tmp_path, _key_written_by_1_6("gpt-5.6-luna", "SYS", self.OLD), "old reply")
        reply, completion = await self._generate(self._client(tmp_path), use_cache=False)
        assert reply == "ok"
        completion.assert_called_once()
