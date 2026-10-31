"""An item's first criterion calls warm the provider's prompt cache for the rest."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from unittest.mock import patch

import litellm
import pytest

from autorubric.dataset import RubricDataset
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig, _caches_prompts, _prompt_cache_min_tokens
from autorubric.rubric import Rubric
from autorubric.types import (
    Criterion,
    CriterionJudgment,
    CriterionOption,
    CriterionVerdict,
    FewShotConfig,
)

LONG = "word " * 5000


@pytest.fixture(autouse=True)
def _clear_model_lookups() -> None:
    _caches_prompts.cache_clear()
    _prompt_cache_min_tokens.cache_clear()


def _binary(name: str) -> Criterion:
    return Criterion(name=name, requirement=f"Requirement {name}", weight=1.0)


def _multi(name: str) -> Criterion:
    return Criterion(
        name=name,
        requirement=f"Question {name}",
        weight=1.0,
        options=[CriterionOption(label="a", value=0.0), CriterionOption(label="b", value=1.0)],
    )


class _Recorder:
    """A fake client recording when each call starts and ends."""

    def __init__(self, fail_first: BaseException | None = None) -> None:
        self.events: list[tuple[str, str]] = []
        self.fail_first = fail_first

    async def generate(self, **kwargs: Any) -> GenerateResult:
        name = re.findall(r"(?:Requirement|Question) (\w+)", kwargs["user_prompt"])[-1]
        self.events.append(("start", name))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if self.fail_first is not None:
            error, self.fail_first = self.fail_first, None
            self.events.append(("end", name))
            raise error
        self.events.append(("end", name))
        fmt = kwargs["response_format"]
        if fmt is CriterionJudgment:
            parsed = fmt(criterion_status=CriterionVerdict.MET, explanation="x")
        else:
            parsed = fmt(selected_option=1, explanation="x")
        return GenerateResult(content="", parsed=parsed)


def _grader(model: str = "gpt-5.6-luna", **config: Any) -> tuple[CriterionGrader, _Recorder]:
    grader = CriterionGrader(judge_model_config=LLMConfig(model=model, **config))
    recorder = _Recorder()
    grader._clients[grader._judges[0].judge_id] = recorder  # type: ignore[assignment]
    return grader, recorder


def _started_before_first_end(recorder: _Recorder) -> list[str]:
    first_end = next(i for i, (kind, _) in enumerate(recorder.events) if kind == "end")
    return [name for kind, name in recorder.events[:first_end] if kind == "start"]


async def _judge(grader: CriterionGrader, rubric: list[Criterion], submission: str = LONG) -> Any:
    return await grader._judge_all_criteria_for_judge(grader._judges[0], rubric, submission)


class TestWarming:
    @pytest.mark.asyncio
    async def test_warming_call_finishes_before_the_rest(self) -> None:
        grader, recorder = _grader()
        await _judge(grader, [_binary("a"), _binary("b"), _binary("c")])
        assert _started_before_first_end(recorder) == ["a"]

    @pytest.mark.asyncio
    async def test_mixed_rubric_warms_each_kind_in_parallel(self) -> None:
        grader, recorder = _grader()
        rubric = [_binary("a"), _multi("m"), _binary("b"), _multi("n")]
        await _judge(grader, rubric)
        assert sorted(_started_before_first_end(recorder)) == ["a", "m"]

    @pytest.mark.asyncio
    async def test_a_lone_criterion_of_a_kind_joins_the_gather(self) -> None:
        grader, recorder = _grader()
        await _judge(grader, [_binary("a"), _binary("b"), _multi("m")])
        assert _started_before_first_end(recorder) == ["a"]

    @pytest.mark.parametrize(
        ("model", "config", "submission", "rubric_size"),
        [
            ("gpt-5.6-luna", {}, LONG, 1),
            ("gpt-5.6-luna", {}, "short", 3),
            ("openai/local-model", {}, LONG, 3),
            ("gpt-5.6-luna", {"prompt_caching": False}, LONG, 3),
        ],
    )
    @pytest.mark.asyncio
    async def test_no_warming(
        self, model: str, config: dict, submission: str, rubric_size: int
    ) -> None:
        grader, recorder = _grader(model, **config)
        rubric = [_binary(f"c{i}") for i in range(rubric_size)]
        # The system prompt alone passes 1,024 tokens, so raise the bar for the short case.
        with patch("autorubric.llm._prompt_cache_min_tokens", return_value=5000):
            await _judge(grader, rubric, submission)
        assert len(_started_before_first_end(recorder)) == rubric_size

    @pytest.mark.parametrize(
        ("model", "submission", "warmed"),
        [
            ("gemini/gemini-3.8-flash", LONG, True),
            ("gemini/gemini-3.8-flash", "short", False),
            ("gemini/gemini-4-argon", LONG, False),
        ],
    )
    @pytest.mark.asyncio
    async def test_gemini_warms_only_a_context_cache(
        self, model: str, submission: str, warmed: bool
    ) -> None:
        grader, recorder = _grader(model)
        await _judge(grader, [_binary("a"), _binary("b")], submission)
        assert len(_started_before_first_end(recorder)) == (1 if warmed else 2)

    @pytest.mark.asyncio
    async def test_a_tokenizer_failure_means_no_warming(self) -> None:
        grader, recorder = _grader()
        with patch("autorubric.llm.litellm.token_counter", side_effect=ValueError("x")):
            await _judge(grader, [_binary("a"), _binary("b")])
        assert len(_started_before_first_end(recorder)) == 2

    @pytest.mark.asyncio
    async def test_failed_warming_call_leaves_the_rest_judged(self) -> None:
        grader, recorder = _grader()
        recorder.fail_first = TimeoutError("slow")
        results = await _judge(grader, [_binary("a"), _binary("b"), _binary("c")])
        reports = [r.report for r in results.criterion_results]
        assert "slow" in reports[0].error
        assert [r.verdict for r in reports[1:]] == [CriterionVerdict.MET] * 2

    @pytest.mark.asyncio
    async def test_a_configuration_error_in_the_warming_call_raises_and_sends_no_more(self) -> None:
        grader, recorder = _grader()
        recorder.fail_first = litellm.AuthenticationError("bad key", "openai", "gpt-5.6-luna")
        with pytest.raises(ValueError):
            await _judge(grader, [_binary("a"), _binary("b"), _binary("c")])
        assert [name for kind, name in recorder.events if kind == "start"] == ["a"]

    @pytest.mark.asyncio
    async def test_escalated_calls_warm_too(self) -> None:
        grader, recorder = _grader()
        rubric = [_binary("a"), _binary("b"), _binary("c")]
        await grader._judge_escalated_criteria(grader._judges[0], rubric, [1, 2], LONG)
        assert _started_before_first_end(recorder) == ["b"]

    @pytest.mark.asyncio
    async def test_results_stay_in_rubric_order(self) -> None:
        grader, _ = _grader()
        rubric = [_multi("m"), _binary("a"), _multi("n"), _binary("b")]
        results = await _judge(grader, rubric)
        names = [r.report.name for r in results.criterion_results]
        assert names == ["m", "a", "n", "b"]

    @pytest.mark.asyncio
    async def test_a_call_with_nothing_to_share_with_writes_no_cache(self) -> None:
        grader, recorder = _grader()
        seen: list[dict] = []
        original = recorder.generate

        async def spy(**kwargs: Any) -> GenerateResult:
            seen.append(kwargs)
            return await original(**kwargs)

        recorder.generate = spy  # type: ignore[method-assign]
        await _judge(grader, [_binary("a"), _binary("b"), _multi("m")], "S")
        by_kind = {k["response_format"] is CriterionJudgment: k for k in seen}
        assert by_kind[True]["shared_prefix"] is not None
        assert by_kind[False]["shared_prefix"] is None

    @pytest.mark.asyncio
    async def test_generate_receives_the_shared_prefix(self) -> None:
        grader, recorder = _grader()
        seen: list[dict] = []
        original = recorder.generate

        async def spy(**kwargs: Any) -> GenerateResult:
            seen.append(kwargs)
            return await original(**kwargs)

        recorder.generate = spy  # type: ignore[method-assign]
        await _judge(grader, [_binary("a"), _binary("b"), _multi("m"), _multi("n")], "S")
        assert {k["shared_prefix"] for k in seen} == {"<submission>\nS\n</submission>"}
        for k in seen:
            assert k["user_prompt"].startswith(k["shared_prefix"] + "\n\n")
            assert k["pre_1_7_user_prompt"].endswith("<submission>\nS\n</submission>")


class TestFewShotPrefixes:
    """Few-shot prefixes are shared across items, so the first item's call warms them."""

    def _grader(self, recorder: _Recorder, rubric: list[Criterion]) -> CriterionGrader:
        labels = [
            [CriterionVerdict.MET if c.options is None else "a" for c in rubric],
            [CriterionVerdict.UNMET if c.options is None else "b" for c in rubric],
        ]
        data = RubricDataset(prompt="P", rubric=Rubric(rubric))
        for i, ground_truth in enumerate(labels):
            data.add_item(f"Training {i}. " + LONG, f"Training {i}", ground_truth=ground_truth)
        grader = CriterionGrader(
            judge_model_config=LLMConfig(model="gpt-5.6-luna"),
            training_data=data,
            few_shot_config=FewShotConfig(n_examples=2),
        )
        grader._clients[grader._judges[0].judge_id] = recorder  # type: ignore[assignment]
        return grader

    @pytest.mark.asyncio
    async def test_the_first_item_warms_a_criterions_prefix_for_concurrent_items(self) -> None:
        recorder = _Recorder()
        rubric = [_binary("a")]
        grader = self._grader(recorder, rubric)
        await asyncio.gather(*(_judge(grader, rubric, f"Item {i}") for i in range(3)))
        assert _started_before_first_end(recorder) == ["a"]
        assert len([e for e in recorder.events if e[0] == "start"]) == 3

    @pytest.mark.asyncio
    async def test_criteria_of_one_item_do_not_wait_for_each_other(self) -> None:
        recorder = _Recorder()
        rubric = [_binary("a"), _binary("b")]
        grader = self._grader(recorder, rubric)
        await _judge(grader, rubric, "Item")
        assert sorted(_started_before_first_end(recorder)) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_a_failed_warming_call_lets_the_next_call_warm(self) -> None:
        recorder = _Recorder(fail_first=TimeoutError("slow"))
        rubric = [_binary("a")]
        grader = self._grader(recorder, rubric)
        await _judge(grader, rubric, "Item 0")
        assert grader._warmed_prefixes == {}
        await _judge(grader, rubric, "Item 1")
        assert len(grader._warmed_prefixes) == 1

    @pytest.mark.asyncio
    async def test_the_prefix_reaches_generate_with_the_old_prompts(self) -> None:
        recorder = _Recorder()
        seen: list[dict] = []
        original = recorder.generate

        async def spy(**kwargs: Any) -> GenerateResult:
            seen.append(kwargs)
            return await original(**kwargs)

        recorder.generate = spy  # type: ignore[method-assign]
        grader = self._grader(recorder, [_multi("m")])
        await _judge(grader, [grader._effective_criterion(_multi("m"))], "S")
        (request,) = seen
        assert request["shared_prefix"].endswith("</question>")
        assert "<selected_option>" in request["pre_1_7_user_prompt"]
        assert "- <selected_option>" in request["pre_1_7_system_prompt"]
        assert "- <selected_option>" not in request["system_prompt"]
