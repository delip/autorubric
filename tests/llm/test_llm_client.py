"""Tests for LLMClient class."""

import tempfile
from pathlib import Path
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

    def test_cache_key_differs_for_different_inputs(self):
        """Cache keys differ when inputs differ."""
        config = LLMConfig(model="openai/gpt-5.2")
        client = LLMClient(config)

        key1 = client._cache_key("openai/gpt-5.2", "System A", "User prompt", None)
        key2 = client._cache_key("openai/gpt-5.2", "System B", "User prompt", None)
        key3 = client._cache_key("openai/gpt-5.2", "System A", "Different prompt", None)
        key4 = client._cache_key("gpt-3.5", "System A", "User prompt", None)

        assert key1 != key2
        assert key1 != key3
        assert key1 != key4

    def test_cache_key_includes_response_format(self):
        """Cache key includes the response format class name."""
        config = LLMConfig(model="openai/gpt-5.2")
        client = LLMClient(config)

        key_no_format = client._cache_key("openai/gpt-5.2", "System", "User", None)
        key_with_format = client._cache_key("openai/gpt-5.2", "System", "User", MockResponse)

        assert key_no_format != key_with_format


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

            with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
                result = await client.generate(
                    system_prompt="System",
                    user_prompt="User",
                )

                assert result == "cached response"
                mock_completion.assert_not_called()

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
    async def test_generate_with_thinking_budget_tokens(self):
        """generate includes thinking parameter when budget_tokens specified."""
        config = LLMConfig(
            model="anthropic/claude-sonnet-4-5-20250929",
            thinking=10000,  # Direct token budget
        )
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
            assert "thinking" in call_kwargs
            assert call_kwargs["thinking"]["type"] == "enabled"
            assert call_kwargs["thinking"]["budget_tokens"] == 10000

    @pytest.mark.asyncio
    async def test_generate_with_thinking_level(self):
        """generate includes reasoning_effort parameter when level specified."""
        config = LLMConfig(
            model="openai/responses/gpt-5-mini",
            thinking="high",  # Level-based thinking
        )
        client = LLMClient(config)

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.reasoning_content = "I reasoned about this..."

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
            assert "reasoning_effort" in call_kwargs
            assert call_kwargs["reasoning_effort"] == "high"

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

            with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
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
