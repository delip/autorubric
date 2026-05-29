"""Tests for LLMClient class."""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from autorubric.llm import LLMClient, LLMConfig


class MockResponse(BaseModel):
    """Mock Pydantic model for structured output tests."""

    result: str
    value: int


class TestLLMClientInitialization:
    """Tests for LLMClient initialization."""

    def test_raises_value_error_for_empty_model(self):
        """LLMClient raises ValueError when model is empty."""
        config = LLMConfig.__new__(LLMConfig)
        # Manually set the model to empty string
        object.__setattr__(config, "model", "")
        object.__setattr__(config, "temperature", 0.0)
        object.__setattr__(config, "max_tokens", None)
        object.__setattr__(config, "top_p", None)
        object.__setattr__(config, "timeout", 60.0)
        object.__setattr__(config, "max_retries", 3)
        object.__setattr__(config, "retry_min_wait", 1.0)
        object.__setattr__(config, "retry_max_wait", 60.0)
        object.__setattr__(config, "cache_enabled", False)
        object.__setattr__(config, "cache_dir", ".autorubric_cache")
        object.__setattr__(config, "cache_ttl", None)
        object.__setattr__(config, "api_key", None)
        object.__setattr__(config, "api_base", None)
        object.__setattr__(config, "thinking", None)
        object.__setattr__(config, "prompt_caching", False)
        object.__setattr__(config, "seed", None)
        object.__setattr__(config, "extra_headers", {})
        object.__setattr__(config, "extra_params", {})

        with pytest.raises(ValueError, match="model is required and cannot be empty"):
            LLMClient(config)

    def test_initializes_with_valid_config(self):
        """LLMClient initializes successfully with valid config."""
        config = LLMConfig(model="openai/gpt-5.2")
        client = LLMClient(config)
        assert client.config == config
        assert client._cache is None

    def test_initializes_cache_when_enabled(self):
        """LLMClient initializes cache when cache_enabled is True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LLMConfig(
                model="openai/gpt-5.2",
                cache_enabled=True,
                cache_dir=temp_dir,
            )
            client = LLMClient(config)
            assert client._cache is not None
            client.close()


class TestLLMClientCacheKey:
    """Tests for LLMClient cache key generation."""

    def test_cache_key_generation(self):
        """Cache key is a consistent hash based on inputs."""
        config = LLMConfig(model="openai/gpt-5.2")
        client = LLMClient(config)

        key1 = client._cache_key(
            model="openai/gpt-5.2",
            system_prompt="You are helpful.",
            user_prompt="Hello",
            response_format=None,
        )

        key2 = client._cache_key(
            model="openai/gpt-5.2",
            system_prompt="You are helpful.",
            user_prompt="Hello",
            response_format=None,
        )

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex digest length

    @pytest.mark.parametrize(
        "varied_args",
        [
            pytest.param(("openai/gpt-5.2", "System B", "User prompt", None), id="system"),
            pytest.param(("openai/gpt-5.2", "System A", "Different prompt", None), id="user"),
            pytest.param(("gpt-3.5", "System A", "User prompt", None), id="model"),
            pytest.param(
                ("openai/gpt-5.2", "System A", "User prompt", MockResponse),
                id="response_format",
            ),
        ],
    )
    def test_cache_key_differs_for_different_inputs(self, varied_args):
        """Cache keys differ when any input field differs (model/system/user/response_format)."""
        config = LLMConfig(model="openai/gpt-5.2")
        client = LLMClient(config)

        baseline = client._cache_key("openai/gpt-5.2", "System A", "User prompt", None)
        varied = client._cache_key(*varied_args)

        assert baseline != varied


class TestLLMClientCacheStats:
    """Tests for LLMClient cache_stats method."""

    def test_cache_stats_when_no_cache(self):
        """cache_stats returns zeros when cache is not initialized."""
        config = LLMConfig(model="openai/gpt-5.2", cache_enabled=False)
        client = LLMClient(config)

        stats = client.cache_stats()

        assert stats == {"size": 0, "count": 0, "directory": None}

    def test_cache_stats_with_initialized_cache(self):
        """cache_stats returns proper stats when cache is initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LLMConfig(
                model="openai/gpt-5.2",
                cache_enabled=True,
                cache_dir=temp_dir,
            )
            client = LLMClient(config)

            stats = client.cache_stats()

            assert stats["count"] == 0
            assert stats["directory"] == temp_dir
            assert "size" in stats
            client.close()


class TestLLMClientClearCache:
    """Tests for LLMClient clear_cache method."""

    def test_clear_cache_when_no_cache(self):
        """clear_cache returns 0 when cache is not initialized."""
        config = LLMConfig(model="openai/gpt-5.2", cache_enabled=False)
        client = LLMClient(config)

        count = client.clear_cache()

        assert count == 0

    def test_clear_cache_with_initialized_cache(self):
        """clear_cache clears entries and returns count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LLMConfig(
                model="openai/gpt-5.2",
                cache_enabled=True,
                cache_dir=temp_dir,
            )
            client = LLMClient(config)

            # Add some entries to cache
            client._cache.set("key1", "value1")
            client._cache.set("key2", "value2")
            assert len(client._cache) == 2

            count = client.clear_cache()

            assert count == 2
            assert len(client._cache) == 0
            client.close()


class TestLLMClientEnsureCache:
    """Tests for LLMClient _ensure_cache method."""

    def test_ensure_cache_initializes_cache_when_needed(self):
        """_ensure_cache initializes cache if not already initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LLMConfig(
                model="openai/gpt-5.2",
                cache_enabled=False,  # Start with cache disabled
                cache_dir=temp_dir,
            )
            client = LLMClient(config)

            assert client._cache is None

            cache = client._ensure_cache()

            assert cache is not None
            assert client._cache is not None
            client.close()

    def test_ensure_cache_returns_existing_cache(self):
        """_ensure_cache returns existing cache without reinitializing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LLMConfig(
                model="openai/gpt-5.2",
                cache_enabled=True,
                cache_dir=temp_dir,
            )
            client = LLMClient(config)

            original_cache = client._cache
            returned_cache = client._ensure_cache()

            assert returned_cache is original_cache
            client.close()


class TestLLMClientGenerate:
    """Tests for LLMClient generate method using mocks."""

    @pytest.mark.asyncio
    async def test_generate_calls_litellm(self):
        """generate makes a call to litellm.acompletion."""
        config = LLMConfig(model="openai/gpt-5.2")
        client = LLMClient(config)

        mock_message = MagicMock()
        mock_message.content = "Hello, world!"
        mock_message.thinking = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            result = await client.generate(
                system_prompt="You are helpful.",
                user_prompt="Say hello",
            )

            assert result == "Hello, world!"
            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args.kwargs
            assert call_kwargs["model"] == "openai/gpt-5.2"
            assert len(call_kwargs["messages"]) == 2

    @pytest.mark.asyncio
    async def test_generate_with_cache_hit(self):
        """generate returns cached response on cache hit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LLMConfig(
                model="openai/gpt-5.2",
                cache_enabled=True,
                cache_dir=temp_dir,
            )
            client = LLMClient(config)

            # Pre-populate cache
            cache_key = client._cache_key("openai/gpt-5.2", "System", "User", None)
            client._cache.set(cache_key, "cached response")

            with patch(
                "autorubric.llm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_completion:
                result = await client.generate(
                    system_prompt="System",
                    user_prompt="User",
                )

                assert result == "cached response"
                mock_completion.assert_not_called()

            client.close()

    @pytest.mark.asyncio
    async def test_generate_with_structured_output(self):
        """generate parses structured output into Pydantic model."""
        config = LLMConfig(model="openai/gpt-5.2")
        client = LLMClient(config)

        mock_message = MagicMock()
        mock_message.content = '{"result": "success", "value": 42}'
        mock_message.thinking = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            result = await client.generate(
                system_prompt="You are helpful.",
                user_prompt="Give me a result",
                response_format=MockResponse,
            )

            assert isinstance(result, MockResponse)
            assert result.result == "success"
            assert result.value == 42

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model", "thinking", "assert_thinking_params"),
        [
            pytest.param(
                "anthropic/claude-sonnet-4-5-20250929",
                10000,  # Direct token budget
                lambda kw: (
                    "thinking" in kw
                    and kw["thinking"]["type"] == "enabled"
                    and kw["thinking"]["budget_tokens"] == 10000
                ),
                id="budget_tokens",
            ),
            pytest.param(
                "openai/responses/gpt-5-mini",
                "high",  # Level-based thinking
                lambda kw: "reasoning_effort" in kw and kw["reasoning_effort"] == "high",
                id="level",
            ),
        ],
    )
    async def test_generate_with_thinking(self, model, thinking, assert_thinking_params):
        """generate routes thinking config to the correct provider param.

        budget_tokens -> params['thinking'] dict; level -> params['reasoning_effort'].
        """
        config = LLMConfig(model=model, thinking=thinking)
        client = LLMClient(config)

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.reasoning_content = "I thought about this..."

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            await client.generate(
                system_prompt="You are helpful.",
                user_prompt="Think carefully",
            )

            call_kwargs = mock_completion.call_args.kwargs
            assert assert_thinking_params(call_kwargs)

    @pytest.mark.asyncio
    async def test_generate_use_cache_override(self):
        """generate respects use_cache parameter override."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LLMConfig(
                model="openai/gpt-5.2",
                cache_enabled=False,  # Cache disabled by default
                cache_dir=temp_dir,
            )
            client = LLMClient(config)

            mock_message = MagicMock()
            mock_message.content = "Response"
            mock_message.thinking = None

            mock_choice = MagicMock()
            mock_choice.message = mock_message

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            with patch(
                "autorubric.llm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_completion:
                mock_completion.return_value = mock_response

                # Force cache usage
                await client.generate(
                    system_prompt="System",
                    user_prompt="User",
                    use_cache=True,
                )

                # Cache should be initialized now
                assert client._cache is not None
                # Response should be cached
                cache_key = client._cache_key("openai/gpt-5.2", "System", "User", None)
                assert client._cache.get(cache_key) == "Response"

            client.close()
