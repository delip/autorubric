"""Real provider calls: does an item's first call warm the cache for its other calls?

Skipped unless ``AUTORUBRIC_LIVE_PROMPT_CACHE=1``, and never run in CI: each model costs under
a cent or so. Every run grades a fresh nonce, so nothing starts out cached.
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import patch

import pytest

from autorubric import LLMConfig
from autorubric.graders import CriterionGrader
from autorubric.llm import LLMClient
from autorubric.prompts import _shared_user_content, build_user_prompt
from autorubric.types import Criterion

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTORUBRIC_LIVE_PROMPT_CACHE") != "1",
    reason="real provider calls: set AUTORUBRIC_LIVE_PROMPT_CACHE=1",
)

MODELS = [
    "gpt-5.6-luna",
    "anthropic/claude-haiku-5-5",
    "openrouter/deepseek/deepseek-v4.1-flash",
]
RUBRIC = [
    Criterion(name="a", requirement="Mentions a river", weight=1.0),
    Criterion(name="b", requirement="Mentions a mountain", weight=1.0),
    Criterion(name="c", requirement="Mentions a city", weight=1.0),
]


def _submission() -> str:
    """About 8,000 tokens, past every provider's minimum, and new to every cache."""
    sentence = "The river passes the old mill and the mountain road before the city gate. "
    return f"Report {uuid.uuid4()}. " + sentence * 500


def _cached(raw_response: Any) -> int:
    """The prompt tokens a response reports read from the provider's cache."""
    usage = raw_response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return (
        getattr(details, "cached_tokens", None)
        or getattr(usage, "cache_read_input_tokens", None)
        or 0
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.asyncio
async def test_a_later_criterion_reuses_the_submission(model: str) -> None:
    """The second call differs from the first only in its criterion, after the shared block."""
    client = LLMClient(LLMConfig(model=model, max_tokens=2000, cache_enabled=False))
    submission = _submission()
    shared = _shared_user_content(submission)
    results = []
    for criterion in RUBRIC:
        results.append(
            await client.generate(
                "You are a careful grader. " * 300,
                build_user_prompt(criterion, submission),
                return_result=True,
                shared_prefix=shared,
            )
        )
    cached = [_cached(r.raw_response) for r in results]
    prompts = [r.raw_response.usage.prompt_tokens for r in results]
    print(f"\n{model}: prompt tokens {prompts}, cached {cached}")
    assert max(cached[1:]) > 0.8 * prompts[0], (model, cached, prompts)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.asyncio
async def test_a_graded_item_hits_the_cache(model: str) -> None:
    """Through the grader: warm-first leaves the calls after the first reading the cache."""
    grader = CriterionGrader(
        judge_model_config=LLMConfig(model=model, max_tokens=2000, cache_enabled=False)
    )
    seen: list[Any] = []
    client = grader._clients[grader._judges[0].judge_id]
    generate = client.generate

    async def recording(*args: Any, **kwargs: Any) -> Any:
        result = await generate(*args, **kwargs)
        seen.append(result)
        return result

    with patch.object(client, "generate", recording):
        report = await grader.judge(_submission(), RUBRIC)
    cached = [_cached(r.raw_response) for r in seen]
    print(f"\n{model}: cached per call in finish order {cached}")
    assert len(seen) == len(RUBRIC), report
    average = sum(r.raw_response.usage.prompt_tokens for r in seen) / len(seen)
    assert max(cached) > 0.8 * average, (model, cached)
