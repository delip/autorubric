"""Tests for RateLimitPool rate limiting infrastructure."""

import asyncio
import gc
import threading
import weakref

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
    @pytest.mark.parametrize(
        ("max_parallel", "expects_semaphore"),
        [(None, False), (10, True)],
    )
    async def test_get_semaphore_return_type_matches_limit(self, max_parallel, expects_semaphore):
        """Test that None max_parallel returns None and a limit returns a Semaphore."""
        pool = RateLimitPool.get_instance()
        semaphore = await pool.get_semaphore("openai/gpt-4", max_parallel)
        if expects_semaphore:
            assert semaphore is not None
            assert isinstance(semaphore, asyncio.Semaphore)
        else:
            assert semaphore is None

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


class TestRateLimitPoolEventLoops:
    """An asyncio semaphore belongs to one event loop, so the pool keeps one per loop.

    A semaphore binds to the loop on which it first makes a task wait, and waiting on it
    from another loop raises ``RuntimeError``. Successive ``asyncio.run`` calls (one per
    dataset, one per notebook cell) are separate loops, so each loop gets its own
    semaphore for a key, while the key's limit, the strictest one requested, is shared.
    """

    def setup_method(self):
        """Reset singleton before each test."""
        RateLimitPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RateLimitPool.reset_instance()

    @staticmethod
    async def _peak_concurrency(model: str, limit: int, n_tasks: int = 3) -> int:
        """Run ``n_tasks`` tasks that each hold the key's semaphore briefly; the peak held."""
        pool = RateLimitPool.get_instance()
        active = peak = 0

        async def task() -> None:
            nonlocal active, peak
            semaphore = await pool.get_semaphore(model, limit)
            assert semaphore is not None
            async with semaphore:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(task() for _ in range(n_tasks)))
        return peak

    def test_successive_event_loops_can_each_wait_on_the_limit(self):
        """Tasks that wait for the limit on a later loop are limited, not failed."""
        assert asyncio.run(self._peak_concurrency("openai/gpt-4", 1)) == 1
        assert asyncio.run(self._peak_concurrency("openai/gpt-4", 1)) == 1
        assert asyncio.run(self._peak_concurrency("openai/gpt-4-turbo", 2, n_tasks=5)) == 1

    def test_a_loop_left_open_keeps_its_semaphore_and_a_later_loop_gets_its_own(self):
        """A loop that is not closed after its run (``run_until_complete`` on a loop the
        caller keeps) still owns its semaphore, bound to it by a task that waited. A later
        ``asyncio.run`` gets a semaphore of its own, and the open loop keeps using its own."""
        loop = asyncio.new_event_loop()
        try:
            assert loop.run_until_complete(self._peak_concurrency("openai/gpt-4", 1)) == 1
            assert asyncio.run(self._peak_concurrency("openai/gpt-4", 1)) == 1
            assert loop.run_until_complete(self._peak_concurrency("openai/gpt-4", 1)) == 1
        finally:
            loop.close()

    def test_loops_running_at_once_in_different_threads_each_get_their_own(self):
        """The pool's maps are guarded by a thread lock, so loops running at the same time in
        different threads share a key's limit, each waiting on its own semaphore."""
        contended = threading.Event()
        release = threading.Event()
        outcome: dict[str, object] = {}

        async def stay_running() -> None:
            outcome["first"] = await self._peak_concurrency("openai/gpt-4", 1)
            contended.set()
            # This loop keeps running, its semaphore bound to it, while another contends.
            await asyncio.to_thread(release.wait, 10)
            outcome["again"] = await self._peak_concurrency("openai/gpt-4", 1)

        def run_in_thread() -> None:
            try:
                asyncio.run(stay_running())
            except BaseException as exc:  # reported by the assertion below
                outcome["error"] = exc

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        try:
            assert contended.wait(10)
            assert asyncio.run(self._peak_concurrency("openai/gpt-4", 1)) == 1
        finally:
            release.set()
            thread.join(10)
        assert not thread.is_alive()
        assert outcome == {"first": 1, "again": 1}

    def test_one_semaphore_per_key_within_a_loop_and_a_new_one_per_loop(self):
        async def get_two() -> tuple[asyncio.Semaphore | None, asyncio.Semaphore | None]:
            pool = RateLimitPool.get_instance()
            return (
                await pool.get_semaphore("openai/gpt-4", 2),
                await pool.get_semaphore("openai/gpt-4-turbo", 2),
            )

        first_a, first_b = asyncio.run(get_two())
        second_a, second_b = asyncio.run(get_two())

        assert first_a is first_b
        assert second_a is second_b
        assert first_a is not second_a

    def test_a_stricter_limit_requested_on_another_loop_applies_to_every_loop(self):
        loop = asyncio.new_event_loop()
        try:
            loose = loop.run_until_complete(
                RateLimitPool.get_instance().get_semaphore("openai/gpt-4", 3)
            )
            assert loose is not None
            asyncio.run(RateLimitPool.get_instance().get_semaphore("openai/gpt-4", 1))

            assert RateLimitPool.get_instance().get_current_limit("openai/gpt-4") == 1
            # The earlier loop's later requests get the strictest limit too.
            assert loop.run_until_complete(self._peak_concurrency("openai/gpt-4", 3)) == 1
        finally:
            loop.close()

    def test_the_pool_never_keeps_a_finished_loop_alive(self):
        """A semaphore that made a task wait references its loop; the pool drops a closed
        loop's semaphore instead of holding the loop for the life of the process."""
        loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

        async def contend() -> int:
            loops.append(weakref.ref(asyncio.get_running_loop()))
            return await self._peak_concurrency("openai/gpt-4", 1)

        for _ in range(3):
            assert asyncio.run(contend()) == 1
        gc.collect()

        assert [ref() is None for ref in loops] == [True, True, False]
        asyncio.run(self._peak_concurrency("openai/gpt-4", 1))
        gc.collect()
        assert loops[-1]() is None
