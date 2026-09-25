"""Global rate limiting for LLM requests across all clients.

This module provides a singleton RateLimitPool that manages per-provider
semaphores for rate limiting concurrent API requests.
"""

from __future__ import annotations

import asyncio
import weakref
from threading import Lock
from typing import ClassVar


class RateLimitPool:
    """Global singleton managing per-provider semaphores.

    Thread-safe management of asyncio semaphores that are shared across
    all LLMClient instances using the same provider. This ensures that
    rate limits are respected even when multiple graders use the same model.

    The pool normalizes model names to provider level (e.g., "openai/gpt-4"
    and "openai/gpt-4-turbo" both use the "openai" semaphore), since API
    rate limits are typically at the provider/account level.

    Usage:
        # Get or create semaphore for a model
        pool = RateLimitPool.get_instance()
        semaphore = await pool.get_semaphore("openai/gpt-4", max_parallel=10)

        if semaphore:
            async with semaphore:
                # Make LLM request
                pass

    Note:
        If the same provider is used with different max_parallel values,
        the pool uses the MINIMUM value to ensure the strictest limit is
        respected across all usages.

        An asyncio semaphore belongs to one event loop: it binds to the loop on which it
        first makes a task wait, and waiting on it from another loop raises
        ``RuntimeError``. The pool therefore keeps one semaphore per provider **per event
        loop**, so successive ``asyncio.run`` calls (one per dataset, one per notebook
        cell) each get their own, while the provider's limit is shared. The limit applies
        within a loop; requests on different loops never wait for each other. The pool
        never keeps a loop alive: a closed loop's semaphore is dropped at the provider's
        next request, and a semaphore that never made a task wait goes with its loop.
    """

    _instance: ClassVar[RateLimitPool | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self) -> None:
        """Initialize the rate limit pool.

        This should not be called directly - use get_instance() instead.
        """
        # provider -> event loop -> that loop's semaphore for the provider.
        self._semaphores: dict[
            str, weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]
        ] = {}
        self._limits: dict[str, int] = {}
        # Guards the two maps. The critical section never awaits, so a thread lock (which
        # also covers loops running in different threads) never blocks a loop for long.
        self._maps_lock = Lock()

    @classmethod
    def get_instance(cls) -> RateLimitPool:
        """Get or create the singleton instance.

        Thread-safe singleton pattern using double-checked locking.

        Returns:
            The global RateLimitPool instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def get_semaphore(
        self,
        model: str,
        max_parallel: int | None,
    ) -> asyncio.Semaphore | None:
        """Get or create the running loop's semaphore for the given model's provider.

        Must be awaited on the event loop whose tasks will acquire the semaphore; each loop
        gets its own semaphore for a provider (see the class note).

        Args:
            model: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-sonnet-4-5-20250929").
            max_parallel: Maximum parallel requests. None means unlimited.

        Returns:
            Semaphore if rate limited, None if unlimited.

        Note:
            If called multiple times with different max_parallel values for
            the same provider, the semaphore uses the MINIMUM value to ensure
            the strictest limit is respected. A stricter limit replaces the provider's
            semaphores on every loop; requests already holding a replaced semaphore
            finish under it.
        """
        if max_parallel is None:
            return None

        loop = asyncio.get_running_loop()
        # Normalize to provider level for shared rate limits
        provider_key = self._normalize_to_provider(model)
        with self._maps_lock:
            current_limit = self._limits.get(provider_key)
            if current_limit is None or max_parallel < current_limit:
                # A new provider, or a stricter limit: every loop's semaphore is replaced.
                self._limits[provider_key] = max_parallel
                self._semaphores[provider_key] = weakref.WeakKeyDictionary()
            per_loop = self._semaphores[provider_key]
            # Loops are weak keys, but a semaphore that made a task wait references its
            # loop, which would keep the entry alive: drop closed loops explicitly.
            for closed in [other for other in per_loop if other.is_closed()]:
                del per_loop[closed]
            semaphore = per_loop.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._limits[provider_key])
                per_loop[loop] = semaphore
            return semaphore

    def _normalize_to_provider(self, model: str) -> str:
        """Normalize model name to provider for rate limiting.

        Groups models by provider since API rate limits are typically
        at the provider/account level:
        - "openai/gpt-4" and "openai/gpt-4-turbo" -> "openai"
        - "anthropic/claude-sonnet-4-5-20250929" -> "anthropic"
        - "gemini/gemini-2.5-pro" -> "gemini"

        Args:
            model: Full model identifier.

        Returns:
            Provider name for rate limit grouping.
        """
        if "/" in model:
            return model.split("/")[0]
        return model

    def get_current_limit(self, model: str) -> int | None:
        """Get the current limit for a provider (if any).

        Args:
            model: Model identifier.

        Returns:
            Current limit or None if unlimited.
        """
        provider_key = self._normalize_to_provider(model)
        return self._limits.get(provider_key)

    @classmethod
    def reset(cls) -> None:
        """Reset all semaphores and limits.

        Useful for testing to ensure clean state between tests.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._semaphores.clear()
                cls._instance._limits.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Completely reset the singleton instance.

        Use this for testing when you need a fresh instance.
        """
        with cls._lock:
            cls._instance = None
