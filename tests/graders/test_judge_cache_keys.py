"""Judges of an ensemble or cascade keep their own response-cache entries.

The response cache is keyed by the request. Several judges polling one model send
identical requests for a binary criterion, so without a per-judge key every copy would
read the same cached answer on a rerun (or a copy that runs later would read another's).
A lone judge (``judge_id="default"``) keeps the plain key, so its existing caches stay
valid.
"""

import itertools
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from litellm import ModelResponse

from autorubric import EnsembleEvaluationReport, LLMConfig, Rubric
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import LLMClient


def _key(client: LLMClient) -> str:
    return client._cache_key("openai/gpt-4.1-mini", "System", "User", None)


def _cached_config(tmp_path: Path) -> LLMConfig:
    return LLMConfig(
        model="openai/gpt-4.1-mini", cache_enabled=True, cache_dir=str(tmp_path / "cache")
    )


@pytest.mark.asyncio
async def test_rerun_of_repeated_judges_reads_each_judges_own_answer(tmp_path: Path) -> None:
    verdicts = itertools.cycle(["MET", "UNMET", "UNMET"])
    calls: list[dict] = []

    async def fake_acompletion(**params):
        calls.append(params)
        content = json.dumps({"criterion_status": next(verdicts), "explanation": "x"})
        return ModelResponse(
            model="gpt-4.1-mini",
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    rubric = Rubric.from_dict([{"name": "c", "weight": 1.0, "requirement": "Is correct"}])
    config = _cached_config(tmp_path)
    runs = []
    with patch("litellm.acompletion", side_effect=fake_acompletion):
        for _ in range(2):
            calls.clear()
            grader = CriterionGrader(judges=[JudgeSpec(config, f"gpt-{i}") for i in range(3)])
            report = await rubric.grade(to_grade="text", grader=grader)
            assert isinstance(report, EnsembleEvaluationReport) and report.report is not None
            votes = {vote.judge_id: vote.verdict for vote in report.report[0].votes}
            runs.append((votes, len(calls)))

    (first_votes, first_calls), (rerun_votes, rerun_calls) = runs
    assert first_calls == 3
    assert sorted(v.value for v in first_votes.values()) == ["MET", "UNMET", "UNMET"]
    assert rerun_calls == 0
    assert rerun_votes == first_votes


def test_a_lone_judge_keeps_the_plain_cache_key(tmp_path: Path) -> None:
    config = _cached_config(tmp_path)
    lone = CriterionGrader(judge_model_config=config)._clients["default"]

    assert _key(lone) == _key(LLMClient(config))


def test_ensemble_judges_of_one_model_have_their_own_cache_keys(tmp_path: Path) -> None:
    config = _cached_config(tmp_path)
    grader = CriterionGrader(judges=[JudgeSpec(config, "a"), JudgeSpec(config, "b")])
    keys = {judge_id: _key(client) for judge_id, client in grader._clients.items()}

    assert keys["a"] != keys["b"]
    assert _key(LLMClient(config)) not in keys.values()
