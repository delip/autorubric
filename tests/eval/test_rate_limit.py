"""Tests for RateLimitPool rate limiting infrastructure."""

import asyncio

import pytest

from autorubric.rate_limit import RateLimitPool


class TestRateLimitPoolSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        RateLimitPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RateLimitPool.reset_instance()

    def test_get_instance_returns_same_instance(self):
        """Test that get_instance always returns the same instance."""
        instance1 = RateLimitPool.get_instance()
        instance2 = RateLimitPool.get_instance()
        assert instance1 is instance2

    def test_reset_instance_creates_new_instance(self):
        """Test that reset_instance creates a fresh instance."""
        instance1 = RateLimitPool.get_instance()
        RateLimitPool.reset_instance()
        instance2 = RateLimitPool.get_instance()
        assert instance1 is not instance2


class TestRateLimitPoolSemaphores:
    """Tests for semaphore creation and management."""

    def setup_method(self):
        """Reset singleton before each test."""
        RateLimitPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RateLimitPool.reset_instance()

    @pytest.mark.asyncio
    async def test_get_semaphore_returns_none_for_unlimited(self):
        """Test that None max_parallel returns None semaphore."""
        pool = RateLimitPool.get_instance()
        semaphore = await pool.get_semaphore("openai/gpt-4", None)
        assert semaphore is None

    @pytest.mark.asyncio
    async def test_get_semaphore_creates_semaphore_for_limit(self):
        """Test that a limit creates a semaphore."""
        pool = RateLimitPool.get_instance()
        semaphore = await pool.get_semaphore("openai/gpt-4", 10)
        assert semaphore is not None
        assert isinstance(semaphore, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_same_provider_returns_same_semaphore(self):
        """Test that same provider returns same semaphore instance."""
        pool = RateLimitPool.get_instance()
        sem1 = await pool.get_semaphore("openai/gpt-4", 10)
        sem2 = await pool.get_semaphore("openai/gpt-4", 10)
        assert sem1 is sem2

    @pytest.mark.asyncio
    async def test_different_providers_get_different_semaphores(self):
        """Test that different providers get different semaphores."""
        pool = RateLimitPool.get_instance()
        sem_openai = await pool.get_semaphore("openai/gpt-4", 10)
        sem_anthropic = await pool.get_semaphore("anthropic/claude-sonnet", 10)
        assert sem_openai is not sem_anthropic


class TestRateLimitPoolProviderNormalization:
    """Tests for provider key normalization."""

    def setup_method(self):
        """Reset singleton before each test."""
        RateLimitPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RateLimitPool.reset_instance()

    @pytest.mark.asyncio
    async def test_same_provider_different_models_share_semaphore(self):
        """Test that different models from same provider share semaphore."""
        pool = RateLimitPool.get_instance()
        sem_gpt4 = await pool.get_semaphore("openai/gpt-4", 10)
        sem_gpt4_turbo = await pool.get_semaphore("openai/gpt-4-turbo", 10)
        assert sem_gpt4 is sem_gpt4_turbo

    @pytest.mark.asyncio
    async def test_model_without_slash_uses_full_name_as_provider(self):
        """Test that model without slash uses full name as provider key."""
        pool = RateLimitPool.get_instance()
        sem1 = await pool.get_semaphore("gpt-4", 10)
        sem2 = await pool.get_semaphore("gpt-4", 10)
        assert sem1 is sem2

        # Different model without slash should be different provider
        sem3 = await pool.get_semaphore("claude-3", 10)
        assert sem1 is not sem3

    def test_normalize_to_provider_extracts_provider(self):
        """Test provider extraction from model strings."""
        pool = RateLimitPool.get_instance()

        assert pool._normalize_to_provider("openai/gpt-4") == "openai"
        assert pool._normalize_to_provider("anthropic/claude-sonnet-4-5-20250929") == "anthropic"
        assert pool._normalize_to_provider("gemini/gemini-2.5-pro") == "gemini"
        assert pool._normalize_to_provider("gpt-4") == "gpt-4"  # No slash


class TestRateLimitPoolMinimumLimit:
    """Tests for minimum limit enforcement."""

    def setup_method(self):
        """Reset singleton before each test."""
        RateLimitPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RateLimitPool.reset_instance()

    @pytest.mark.asyncio
    async def test_uses_minimum_limit_when_same_provider_different_limits(self):
        """Test that the minimum limit is used when same provider has different limits."""
        pool = RateLimitPool.get_instance()

        # First call with limit of 10
        await pool.get_semaphore("openai/gpt-4", 10)
        assert pool.get_current_limit("openai/gpt-4") == 10

        # Second call with stricter limit of 5 - should update
        await pool.get_semaphore("openai/gpt-4-turbo", 5)
        assert pool.get_current_limit("openai/gpt-4") == 5

    @pytest.mark.asyncio
    async def test_does_not_increase_limit_once_set(self):
        """Test that limit cannot be increased once set."""
        pool = RateLimitPool.get_instance()

        # First call with limit of 5
        await pool.get_semaphore("openai/gpt-4", 5)
        assert pool.get_current_limit("openai/gpt-4") == 5

        # Second call with higher limit of 10 - should NOT update
        await pool.get_semaphore("openai/gpt-4", 10)
        assert pool.get_current_limit("openai/gpt-4") == 5


class TestRateLimitPoolGetCurrentLimit:
    """Tests for get_current_limit method."""

    def setup_method(self):
        """Reset singleton before each test."""
        RateLimitPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RateLimitPool.reset_instance()

    @pytest.mark.asyncio
    async def test_get_current_limit_returns_none_for_unknown_provider(self):
        """Test that get_current_limit returns None for unknown providers."""
        pool = RateLimitPool.get_instance()
        assert pool.get_current_limit("unknown/model") is None

    @pytest.mark.asyncio
    async def test_get_current_limit_returns_limit_for_known_provider(self):
        """Test that get_current_limit returns correct limit."""
        pool = RateLimitPool.get_instance()
        await pool.get_semaphore("openai/gpt-4", 15)
        assert pool.get_current_limit("openai/gpt-4") == 15
        # Same provider via different model
        assert pool.get_current_limit("openai/gpt-3.5-turbo") == 15


class TestRateLimitPoolReset:
    """Tests for reset methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        RateLimitPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RateLimitPool.reset_instance()

    @pytest.mark.asyncio
    async def test_reset_clears_semaphores_and_limits(self):
        """Test that reset clears all semaphores and limits."""
        pool = RateLimitPool.get_instance()
        await pool.get_semaphore("openai/gpt-4", 10)
        assert pool.get_current_limit("openai/gpt-4") == 10

        RateLimitPool.reset()

        # Limit should be cleared
        assert pool.get_current_limit("openai/gpt-4") is None

    @pytest.mark.asyncio
    async def test_reset_allows_new_limits_to_be_set(self):
        """Test that reset allows new limits after clearing."""
        pool = RateLimitPool.get_instance()
        await pool.get_semaphore("openai/gpt-4", 5)
        assert pool.get_current_limit("openai/gpt-4") == 5

        RateLimitPool.reset()

        # Now we can set a new limit
        await pool.get_semaphore("openai/gpt-4", 20)
        assert pool.get_current_limit("openai/gpt-4") == 20


class TestRateLimitPoolConcurrency:
    """Tests for concurrent access patterns."""

    def setup_method(self):
        """Reset singleton before each test."""
        RateLimitPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RateLimitPool.reset_instance()

    @pytest.mark.asyncio
    async def test_semaphore_actually_limits_concurrency(self):
        """Test that semaphore actually limits concurrent access."""
        pool = RateLimitPool.get_instance()
        semaphore = await pool.get_semaphore("openai/gpt-4", 2)

        concurrent_count = 0
        max_concurrent = 0

        async def task():
            nonlocal concurrent_count, max_concurrent
            async with semaphore:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.01)  # Small delay to allow overlap
                concurrent_count -= 1

        # Run 5 tasks with limit of 2
        tasks = [task() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Max concurrent should not exceed 2
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_concurrent_get_semaphore_calls_are_safe(self):
        """Test that concurrent calls to get_semaphore are thread-safe."""
        pool = RateLimitPool.get_instance()

        async def get_sem(model: str, limit: int):
            return await pool.get_semaphore(model, limit)

        # Make multiple concurrent calls
        results = await asyncio.gather(
            get_sem("openai/gpt-4", 10),
            get_sem("openai/gpt-4-turbo", 10),
            get_sem("openai/gpt-3.5", 10),
        )

        # All should return the same semaphore (same provider)
        assert results[0] is results[1]
        assert results[1] is results[2]
