"""How ``LLMClient`` arranges a per-criterion request holding images for prompt caches.

A per-criterion prompt without examples starts with what an item's calls share
(``_shared_user_content``), and an image in the submission, input or reference submission
sits inside that start. Sent split, the shared message carries those images as content parts
with the cache marker on its last text part (the ``"\\n</submission>"`` that ends it), and
the criterion follows as a message of its own. A few-shot prompt starts with what a
criterion's calls share across items (``_few_shot_prefix``): the examples, their images
included, and the criterion, the marker on the criterion's closing text; the item, with the
submission's images, follows. The response-cache key hashes the whole prompt, image bytes
included, as it does without a split.
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import PIL.Image as pil_image
import pytest

from autorubric import Criterion, CriterionVerdict, FewShotExample, Image
from autorubric.content import _concat, _content_removeprefix
from autorubric.llm import LLMClient, LLMConfig, _prefix_reaches_cache_minimum
from autorubric.prompts import (
    _binary_criterion_block,
    _few_shot_prefix,
    _pre_1_7_few_shot_user_prompt,
    _pre_1_7_user_prompt,
    _shared_user_content,
    build_few_shot_user_prompt,
    build_user_prompt,
)


def encoded_png(color: str) -> bytes:
    buffer = io.BytesIO()
    pil_image.new("RGB", (2, 2), color).save(buffer, "PNG")
    return buffer.getvalue()


RED, BLUE, GRAY = (Image(encoded_png(c)) for c in ("red", "blue", "gray"))
CRITERION = Criterion(weight=1.0, requirement="The figure shows a rising trend")
SUBMISSION = [RED, "Accuracy rises with model size."]
QUERY = [BLUE, "Plot accuracy."]


def _response() -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = "ok"
    response.choices[0].message.reasoning_content = None
    response.choices[0].message.thinking_blocks = None
    response.usage = None
    return response


EXAMPLE = FewShotExample([GRAY, "Flat."], CriterionVerdict.UNMET, None)


def _prompt(*, examples: bool = False) -> tuple[Any, Any]:
    """A per-criterion prompt and the shared prefix ``CriterionGrader`` passes with it."""
    if examples:
        prompt = build_few_shot_user_prompt(CRITERION, SUBMISSION, [EXAMPLE], QUERY)
        return prompt, _few_shot_prefix(CRITERION, [EXAMPLE])
    return build_user_prompt(CRITERION, SUBMISSION, QUERY), _shared_user_content(SUBMISSION, QUERY)


async def _messages(model: str, prompt: Any, shared: Any, **config: Any) -> list[dict]:
    client = LLMClient(LLMConfig(model=model, **config))
    with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as completion:
        completion.return_value = _response()
        await client.generate("SYS", prompt, use_cache=False, shared_prefix=shared)
    return completion.call_args.kwargs["messages"]


def _url(image: Image) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image._data_url}}


def _expected_shared_parts(marker_key: str | None, marker: Any) -> list[dict[str, Any]]:
    shared = _shared_user_content(SUBMISSION, QUERY)
    assert isinstance(shared, tuple)
    parts: list[dict[str, Any]] = [
        _url(part) if isinstance(part, Image) else {"type": "text", "text": part} for part in shared
    ]
    assert parts[-1] == {"type": "text", "text": "Accuracy rises with model size.\n</submission>"}
    if marker_key is not None:
        parts[-1][marker_key] = marker
    return parts


class TestTheSharedStartHoldsTheImages:
    def test_the_shared_start_is_a_document_ending_in_text(self) -> None:
        shared = _shared_user_content(SUBMISSION, QUERY)
        assert isinstance(shared, tuple)
        assert shared[-1].endswith("\n</submission>")
        assert [p for p in shared if isinstance(p, Image)] == [BLUE, RED]

    def test_the_prompt_starts_with_it(self) -> None:
        prompt, shared = _prompt()
        rest = _content_removeprefix(prompt, shared)
        assert isinstance(rest, str)
        assert rest == f"\n\n{_binary_criterion_block(CRITERION)}"
        assert _concat(shared, rest) == prompt

    def test_a_few_shot_prompt_starts_with_its_examples_and_criterion(self) -> None:
        prompt, prefix = _prompt(examples=True)
        assert isinstance(prefix, tuple)
        assert [p for p in prefix if isinstance(p, Image)] == [GRAY]
        last = prefix[-1]
        assert isinstance(last, str) and last.endswith(_binary_criterion_block(CRITERION))
        rest = _content_removeprefix(prompt, prefix)
        assert isinstance(rest, tuple)
        head = rest[0]
        assert isinstance(head, str) and head == "\n\n<input>"
        assert [p for p in rest if isinstance(p, Image)] == [BLUE, RED]
        assert _concat(prefix, rest) == prompt

    def test_no_pre_1_7_key_for_a_document(self) -> None:
        prompt, shared = _prompt()
        assert _pre_1_7_user_prompt(prompt, shared, _binary_criterion_block(CRITERION)) is None

    @pytest.mark.parametrize(
        "examples, submission",
        [("<examples>E</examples>", SUBMISSION), ((GRAY, "<examples>E</examples>"), "S")],
        ids=["submission_image", "example_image"],
    )
    def test_no_pre_1_7_few_shot_key_for_a_document(self, examples: Any, submission: Any) -> None:
        block = _binary_criterion_block(CRITERION)
        assert _pre_1_7_few_shot_user_prompt(block, examples, submission) is None
        assert isinstance(_pre_1_7_few_shot_user_prompt(block, "<examples>E</examples>", "S"), str)


class TestSplitMessages:
    @pytest.mark.asyncio
    async def test_anthropic_marks_the_last_text_part_of_the_shared_message(self) -> None:
        prompt, shared = _prompt()
        system, shared_message, tail = await _messages(
            "anthropic/claude-sonnet-4-5", prompt, shared
        )
        mark = {"type": "ephemeral"}
        assert system["content"] == [{"type": "text", "text": "SYS", "cache_control": mark}]
        assert shared_message == {
            "role": "user",
            "content": _expected_shared_parts("cache_control", mark),
        }
        assert tail == {"role": "user", "content": _binary_criterion_block(CRITERION)}

    @pytest.mark.asyncio
    async def test_openai_explicit_breakpoint_on_the_last_text_part(self) -> None:
        prompt, shared = _prompt()
        _, shared_message, tail = await _messages("openai/gpt-5.6-luna", prompt, shared)
        assert shared_message["content"] == _expected_shared_parts(
            "prompt_cache_breakpoint", {"mode": "explicit"}
        )
        assert tail["content"] == _binary_criterion_block(CRITERION)

    @pytest.mark.asyncio
    async def test_automatic_caching_sends_the_shared_parts_unmarked(self) -> None:
        prompt, shared = _prompt()
        system, shared_message, _ = await _messages("deepseek/deepseek-chat", prompt, shared)
        assert system == {"role": "system", "content": "SYS"}
        assert shared_message["content"] == _expected_shared_parts(None, None)

    @pytest.mark.asyncio
    async def test_gemini_context_cache_holds_the_images(self) -> None:
        prompt, shared = _prompt()
        with patch("autorubric.llm._prefix_reaches_cache_minimum", return_value=True):
            _, shared_message, _ = await _messages("gemini/gemini-3.8-flash", prompt, shared)
        assert shared_message["content"] == _expected_shared_parts(
            "cache_control", {"type": "ephemeral", "ttl": "300s"}
        )

    @pytest.mark.asyncio
    async def test_example_images_are_shared_and_the_submission_follows_in_the_tail(
        self,
    ) -> None:
        prompt, prefix = _prompt(examples=True)
        _, shared_message, tail = await _messages("deepseek/deepseek-chat", prompt, prefix)
        shared_parts = shared_message["content"]
        assert isinstance(shared_parts, list)
        assert [p for p in shared_parts if p["type"] == "image_url"] == [_url(GRAY)]
        assert shared_parts[0] == {
            "type": "text",
            "text": "<examples>\n<example_1>\n<example_submission>",
        }
        assert shared_parts[-1]["text"].endswith(_binary_criterion_block(CRITERION))
        assert all("cache_control" not in p for p in shared_parts)
        parts = tail["content"]
        assert isinstance(parts, list)
        assert [p for p in parts if p["type"] == "image_url"] == [_url(BLUE), _url(RED)]
        assert parts[0] == {"type": "text", "text": "<input>"}
        assert parts[-1] == {
            "type": "text",
            "text": "Accuracy rises with model size.\n</submission>",
        }

    @pytest.mark.asyncio
    async def test_anthropic_caches_the_example_images_with_the_criterion(self) -> None:
        prompt, prefix = _prompt(examples=True)
        _, shared_message, tail = await _messages("anthropic/claude-sonnet-4-5", prompt, prefix)
        mark = {"type": "ephemeral"}
        parts = shared_message["content"]
        marked = [p for p in parts if "cache_control" in p]
        assert marked == [parts[-1]]
        assert parts[-1]["type"] == "text" and parts[-1]["cache_control"] == mark
        assert parts[-1]["text"].endswith(_binary_criterion_block(CRITERION))
        assert _url(GRAY) in parts
        # The item is not cached: no marker in the tail, which holds the submission's images.
        assert all("cache_control" not in p for p in tail["content"])
        assert _url(RED) in tail["content"]

    @pytest.mark.asyncio
    async def test_a_text_prefix_does_not_start_a_document(self) -> None:
        prompt, _ = _prompt()
        messages = await _messages("deepseek/deepseek-chat", prompt, "<submission>")
        assert len(messages) == 2
        assert messages[1]["content"] == [
            _url(p) if isinstance(p, Image) else {"type": "text", "text": p} for p in prompt
        ]

    @pytest.mark.asyncio
    async def test_a_failed_gemini_context_cache_resends_the_images_in_one_message(self) -> None:
        prompt, shared = _prompt()
        client = LLMClient(LLMConfig(model="gemini/gemini-3.8-flash", max_retries=1))
        refusal = litellm.BadRequestError("refused", "gemini-3.8-flash", "gemini")
        with (
            patch("autorubric.llm._prefix_reaches_cache_minimum", return_value=True),
            patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as completion,
        ):
            completion.side_effect = [refusal, _response()]
            reply = await client.generate("SYS", prompt, use_cache=False, shared_prefix=shared)
        assert reply == "ok"
        system, user = completion.call_args.kwargs["messages"]
        assert system == {"role": "system", "content": "SYS"}
        assert user["content"] == [
            _url(p) if isinstance(p, Image) else {"type": "text", "text": p} for p in prompt
        ]


class TestCacheMinimum:
    def test_a_document_is_counted_as_messages_with_its_images(self) -> None:
        _, shared = _prompt()
        with patch("autorubric.llm.litellm.token_counter", return_value=5000) as counter:
            assert _prefix_reaches_cache_minimum("gemini/gemini-3.8-flash", "SYS", shared)
        (call,) = counter.call_args_list
        messages = call.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "SYS"}
        assert _url(RED) in messages[1]["content"]
        assert "text" not in call.kwargs

    def test_text_is_counted_as_text(self) -> None:
        with patch("autorubric.llm.litellm.token_counter", return_value=10) as counter:
            assert not _prefix_reaches_cache_minimum("gemini/gemini-3.8-flash", "SYS", "<s>")
        assert counter.call_args.kwargs == {"model": "gemini/gemini-3.8-flash", "text": "SYS<s>"}

    def test_litellm_counts_a_data_url_image_offline(self) -> None:
        _, shared = _prompt()
        # LiteLLM sizes the image from its data: URL, so the count succeeds with no request
        # (a failed count would read as too short).
        with (
            patch("httpx.Client.get", side_effect=AssertionError("no request")),
            patch("autorubric.llm._prompt_cache_min_tokens", return_value=1),
        ):
            assert _prefix_reaches_cache_minimum("openai/gpt-5.6-luna", "SYS", shared)


class TestResponseCacheKey:
    @pytest.mark.asyncio
    async def test_a_split_request_is_keyed_by_the_whole_prompt(self, tmp_path) -> None:
        client = LLMClient(
            LLMConfig(model="openai/gpt-5.6-luna", cache_enabled=True, cache_dir=tmp_path)
        )
        prompt, shared = _prompt()
        with patch("autorubric.llm.litellm.acompletion", new_callable=AsyncMock) as completion:
            completion.return_value = _response()
            await client.generate("SYS", prompt, shared_prefix=shared)
            assert len(completion.call_args.kwargs["messages"]) == 3
            await client.generate("SYS", prompt)
        completion.assert_called_once()
        key = client._cache_key("openai/gpt-5.6-luna", "SYS", prompt, None, {})
        assert list(client._ensure_cache().iterkeys()) == [key]

    def test_image_bytes_change_the_key(self, tmp_path) -> None:
        client = LLMClient(LLMConfig(model="openai/gpt-5.6-luna", cache_dir=tmp_path))
        prompt, _ = _prompt()
        key = client._cache_key("m", "SYS", prompt)
        other = build_user_prompt(CRITERION, [Image(encoded_png("green")), SUBMISSION[1]], QUERY)
        assert client._cache_key("m", "SYS", other) != key
        assert client._cache_key("m", "SYS", build_user_prompt(CRITERION, SUBMISSION, QUERY)) == key
