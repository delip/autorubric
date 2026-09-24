"""Tests for LLMClient class."""

import hashlib
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from autorubric.llm import LLMClient, LLMConfig, _provider_response_format


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


def _text_response(content: str = "Response") -> MagicMock:
    """Build a minimal litellm-style completion response carrying ``content``."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.reasoning_content = None
    mock_message.thinking_blocks = None
    mock_message.thinking = None
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestLLMClientTemperature:
    """Temperature is sent only when explicitly set; None means the provider default."""

    @staticmethod
    async def _sent_kwargs(config: LLMConfig, **generate_kwargs) -> dict:
        client = LLMClient(config)
        with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = _text_response()
            await client.generate(system_prompt="System", user_prompt="User", **generate_kwargs)
        mock_completion.assert_called_once()
        return dict(mock_completion.call_args.kwargs)

    @pytest.mark.asyncio
    async def test_default_config_omits_temperature(self):
        """A default config leaves temperature to the provider (key absent)."""
        sent = await self._sent_kwargs(LLMConfig(model="openai/gpt-5.2"))
        assert "temperature" not in sent

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temperature", [0.0, 1.0], ids=["zero", "one"])
    async def test_explicit_temperature_is_sent(self, temperature):
        """An explicit temperature is sent unchanged; 0.0 is not dropped as falsy."""
        sent = await self._sent_kwargs(LLMConfig(model="openai/gpt-5.2", temperature=temperature))
        assert "temperature" in sent
        assert sent["temperature"] == temperature

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("config_temperature", "call_temperature"),
        [
            pytest.param(None, 0.7, id="none-to-0.7"),
            pytest.param(0.0, 1.0, id="0.0-to-1.0"),
            pytest.param(0.5, 0.0, id="0.5-to-0.0"),
        ],
    )
    async def test_per_call_override_is_sent(self, config_temperature, call_temperature):
        """A per-call temperature overrides the config value."""
        config = LLMConfig(model="openai/gpt-5.2", temperature=config_temperature)
        sent = await self._sent_kwargs(config, temperature=call_temperature)
        assert sent["temperature"] == call_temperature

    @pytest.mark.asyncio
    async def test_per_call_none_omits_temperature(self):
        """A per-call temperature=None omits the key even when the config sets one."""
        config = LLMConfig(model="openai/gpt-5.2", temperature=0.5)
        sent = await self._sent_kwargs(config, temperature=None)
        assert "temperature" not in sent

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config_temperature", [None, 0.0], ids=["none", "zero"])
    async def test_extra_params_temperature_still_wins(self, config_temperature):
        """extra_params keeps its precedence over the temperature field."""
        config = LLMConfig(
            model="openai/gpt-5.2",
            temperature=config_temperature,
            extra_params={"temperature": 0.3},
        )
        sent = await self._sent_kwargs(config)
        assert sent["temperature"] == 0.3

    def test_cache_key_differs_between_none_and_zero(self):
        """Provider-default and explicit 0.0 temperature never share a cache entry."""
        args = ("openai/gpt-5.2", "System", "User", None)
        key_none = LLMClient(LLMConfig(model="openai/gpt-5.2"))._cache_key(*args)
        key_zero = LLMClient(LLMConfig(model="openai/gpt-5.2", temperature=0.0))._cache_key(*args)
        assert key_none != key_zero

    @pytest.mark.parametrize("temperature", [0.0, 0.7], ids=["zero", "point-seven"])
    def test_cache_key_for_explicit_temperature_matches_legacy_format(self, temperature):
        """Explicit temperatures keep byte-identical cache keys, so existing caches still hit."""
        client = LLMClient(LLMConfig(model="openai/gpt-5.2", temperature=temperature))
        legacy_content = (
            f"openai/gpt-5.2:System:User:str:temp={temperature}"
            ":top_p=None:max_tokens=None:thinking=none:seed=None"
        )
        expected = hashlib.sha256(legacy_content.encode()).hexdigest()
        assert client._cache_key("openai/gpt-5.2", "System", "User", None) == expected

    def test_cache_key_for_default_temperature_is_stable(self):
        """The provider-default key renders temperature as None and is stable across clients."""
        args = ("openai/gpt-5.2", "System", "User", None)
        key1 = LLMClient(LLMConfig(model="openai/gpt-5.2"))._cache_key(*args)
        key2 = LLMClient(LLMConfig(model="openai/gpt-5.2"))._cache_key(*args)
        content = (
            "openai/gpt-5.2:System:User:str:temp=None"
            ":top_p=None:max_tokens=None:thinking=none:seed=None"
        )
        assert key1 == key2 == hashlib.sha256(content.encode()).hexdigest()

    @pytest.mark.asyncio
    async def test_per_call_override_does_not_reuse_config_temperature_cache(self, tmp_path):
        """A per-call temperature gets its own cache entry instead of the config one."""
        config = LLMConfig(
            model="openai/gpt-5.2",
            temperature=0.0,
            cache_enabled=True,
            cache_dir=tmp_path,
        )
        client = LLMClient(config)
        try:
            with patch(
                "autorubric.llm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_completion:
                mock_completion.side_effect = [
                    _text_response("at config temperature"),
                    _text_response("at override temperature"),
                ]

                first = await client.generate(system_prompt="System", user_prompt="User")
                assert first == "at config temperature"
                assert mock_completion.call_count == 1

                overridden = await client.generate(
                    system_prompt="System", user_prompt="User", temperature=1.0
                )
                assert overridden == "at override temperature"
                assert mock_completion.call_count == 2
                assert mock_completion.call_args.kwargs["temperature"] == 1.0

                repeat = await client.generate(system_prompt="System", user_prompt="User")
                assert repeat == "at config temperature"
                repeat_overridden = await client.generate(
                    system_prompt="System", user_prompt="User", temperature=1.0
                )
                assert repeat_overridden == "at override temperature"
                assert mock_completion.call_count == 2
        finally:
            client.close()

    @pytest.mark.asyncio
    async def test_cached_call_accepts_per_call_model_override(self, tmp_path):
        """Per-call overrides reach the cache key without clashing with its own arguments."""
        config = LLMConfig(model="openai/gpt-5.2", cache_enabled=True, cache_dir=tmp_path)
        client = LLMClient(config)
        try:
            with patch(
                "autorubric.llm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_completion:
                mock_completion.return_value = _text_response("ok")
                result = await client.generate(
                    system_prompt="System", user_prompt="User", model="openai/gpt-4.1-mini"
                )
                assert result == "ok"
                assert mock_completion.call_args.kwargs["model"] == "openai/gpt-4.1-mini"
        finally:
            client.close()


class TestProviderResponseFormat:
    """Tests for _provider_response_format.

    It strips the unused ``reasoning`` slot from the schema sent to the provider so that
    strict-mode backends (OpenAI/Groq) do not force non-OpenAI models to emit it. Parsing
    still uses the full Pydantic model, so reasoning injection (when thinking is on) is
    unaffected.
    """

    def test_strips_reasoning_from_provider_schema(self):
        from autorubric.types import CriterionJudgment

        param = _provider_response_format(CriterionJudgment)
        # A model carrying `reasoning` becomes a json_schema dict with that field removed.
        assert isinstance(param, dict)
        schema = param["json_schema"]["schema"]
        assert "reasoning" not in schema.get("properties", {})
        assert "reasoning" not in schema.get("required", [])
        # The fields the prompt actually asks for survive.
        assert "criterion_status" in schema["properties"]
        assert "explanation" in schema["properties"]

    def test_passthrough_when_no_reasoning_field(self):
        # A response format without a `reasoning` field is returned unchanged.
        assert _provider_response_format(MockResponse) is MockResponse

    @pytest.mark.asyncio
    async def test_generate_sends_schema_without_reasoning(self):
        from autorubric.types import CriterionJudgment

        mock_message = MagicMock()
        mock_message.content = '{"criterion_status": "MET", "explanation": "ok"}'
        mock_message.reasoning_content = None
        mock_message.thinking_blocks = None
        mock_message.thinking = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        client = LLMClient(LLMConfig(model="groq/llama-3.3-70b-versatile"))
        with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response
            result = await client.generate(
                system_prompt="judge this",
                user_prompt="submission",
                response_format=CriterionJudgment,
            )

        sent = mock_completion.call_args.kwargs["response_format"]
        assert isinstance(sent, dict)
        assert "reasoning" not in sent["json_schema"]["schema"].get("required", [])
        # Parsing still uses the full model; reasoning defaults to None when not emitted.
        assert isinstance(result, CriterionJudgment)
        assert result.criterion_status.value == "MET"
        assert result.reasoning is None
