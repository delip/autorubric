"""Decision-model judges through ``evaluate()``: requests, checkpoints and the manifest.

``evaluate()`` grades every item through ``CriterionGrader``, so a decision-model judge
makes exactly one request per item. Its reports persist through the ordinary checkpoint
path (``ItemResult.to_dict``/``from_dict``, ``EvalResult.from_experiment``) with their
``probabilities``, ``confidence`` and ``reason=None`` intact, and the experiment manifest
records the judge's kind and framing without ever writing its API key.

The real ``DecisionModelClient`` runs on the recording fake SDK client of ``conftest.py``
(``fake_sdk``); LLM judges are patched in for ``LLMClient``. Nothing reaches the network.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import httpx2
import openai
import pytest
import typesafe_sdk
from typesafe_sdk import SystemOneResponse

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DecisionModelConfig,
    EvalResult,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
    evaluate,
)
from autorubric.eval import ItemResult, _serialize_grader_config
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.types import CANONICAL_NA_OPTION, CriterionJudgment, MultiChoiceJudgment

API_KEY = "ts-test-key-never-persisted-2b8e"
# A token some gateways take as a path segment: the manifest records a URL's host only.
URL_SECRET = "url-path-token-never-persisted-91d0"
NA_LABEL = CANONICAL_NA_OPTION.label

RUBRIC = Rubric(
    [
        Criterion(name="light", weight=5.0, requirement="Mentions light"),
        Criterion(name="myth", weight=-3.0, requirement="Claims plants eat soil"),
        Criterion(
            name="clarity",
            weight=2.0,
            requirement="How clear is the explanation?",
            scale_type="ordinal",
            options=[
                CriterionOption(label="Unclear", value=0.0),
                CriterionOption(label="Mostly clear", value=0.6),
                CriterionOption(label="Very clear", value=1.0),
            ],
        ),
    ]
)

ANSWERS = {
    "c0": {"type": "noul", "noul": 0.9},
    "c1": {"type": "noul", "noul": 0.2},
    "c2": {
        "type": "choice",
        "choice": "Mostly clear",
        "confidence": 0.6,
        "probabilities": {"Unclear": 0.1, "Mostly clear": 0.7, "Very clear": 0.15, NA_LABEL: 0.05},
    },
}


def response() -> SystemOneResponse:
    payload = {
        "model": "jev-test",
        "usage": {"input_tokens": 900, "output_tokens": 3},
        "answers": ANSWERS,
    }
    return SystemOneResponse.model_validate_json(json.dumps(payload))


GROUND_TRUTH = [
    [CriterionVerdict.MET, CriterionVerdict.UNMET, "Mostly clear"],
    [CriterionVerdict.UNMET, CriterionVerdict.UNMET, "Mostly clear"],
    [CriterionVerdict.MET, CriterionVerdict.MET, "Unclear"],
]


def dataset(n_items: int = 3) -> RubricDataset:
    data = RubricDataset(prompt="Explain photosynthesis.", rubric=RUBRIC, name="photosynthesis")
    for i in range(n_items):
        data.add_item(
            submission=f"Answer {i}: plants use light to make sugar.",
            description=f"item {i}",
            ground_truth=GROUND_TRUTH[i % len(GROUND_TRUTH)],
        )
    return data


def dm(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(**{"model": "jev-latest", "api_key": API_KEY, **overrides})


class _LLM:
    async def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> Any:
        parsed: Any
        if "<options>" in user_prompt:
            parsed = MultiChoiceJudgment(selected_option=2, explanation="fairly clear")
        else:
            parsed = CriterionJudgment(criterion_status=CriterionVerdict.MET, explanation="yes")
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=None,
            parsed=parsed,
        )


@pytest.fixture
def graders():
    built: list[CriterionGrader] = []
    yield built
    for grader in built:
        for client in grader._decision_clients.values():
            client.close()


async def run(grader: CriterionGrader, tmp_path: Path, name: str, n_items: int = 3) -> EvalResult:
    return await evaluate(
        dataset(n_items),
        grader,
        show_progress=False,
        experiment_name=name,
        experiments_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_evaluate_sends_one_request_per_item(fake_sdk, graders, tmp_path):
    fake_sdk.response = response()
    grader = CriterionGrader(judge_model_config=dm())
    graders.append(grader)
    result = await run(grader, tmp_path, "one-per-item", n_items=5)

    assert result.successful_items == 5
    assert len(fake_sdk.calls) == 5
    assert all(list(questions) == ["c0", "c1", "c2"] for _, questions, _ in fake_sdk.calls)
    submissions = sorted(state["submission"] for state, _, _ in fake_sdk.calls)
    assert submissions == sorted(item.submission for item in dataset(5))


@pytest.mark.asyncio
async def test_decision_model_reports_round_trip_through_checkpoints(fake_sdk, graders, tmp_path):
    fake_sdk.response = response()
    grader = CriterionGrader(judge_model_config=dm(input_cost_per_token=1e-6))
    graders.append(grader)
    result = await run(grader, tmp_path, "round-trip")

    first = result.item_results[0]
    light = first.report.report[0]
    assert light.votes[0].probabilities == {"MET": 0.9, "UNMET": pytest.approx(0.1)}
    assert light.votes[0].confidence == pytest.approx(0.8)
    assert light.votes[0].reason is None and light.final_reason is None
    assert first.report.completion_cost == pytest.approx(900e-6)

    # In memory: ItemResult.to_dict / from_dict.
    restored = ItemResult.from_dict(json.loads(json.dumps(first.to_dict())), first.item)
    assert restored.report.model_dump() == first.report.model_dump()

    # On disk: the saved experiment.
    loaded = EvalResult.from_experiment(tmp_path / "round-trip")
    by_idx = {r.item_idx: r for r in loaded.item_results}
    for original in result.item_results:
        assert by_idx[original.item_idx].report.model_dump() == original.report.model_dump()
    clarity_vote = by_idx[0].report.report[2].multi_choice_votes[0]
    assert clarity_vote.probabilities == {"0": 0.1, "1": 0.7, "2": 0.15, "3": 0.05}
    assert clarity_vote.reason is None
    assert loaded.total_completion_cost == pytest.approx(3 * 900e-6)

    # Metrics read the restored decision-model verdicts like any others: every item gets
    # the same answers, which match the ground truth on two items of three per criterion.
    restored_metrics = loaded.compute_metrics(dataset())
    assert restored_metrics.criterion_accuracy == pytest.approx(2 / 3)
    assert (
        restored_metrics.criterion_accuracy == result.compute_metrics(dataset()).criterion_accuracy
    )


@pytest.mark.asyncio
async def test_manifest_records_judge_kinds_and_never_the_key(fake_sdk, graders, tmp_path):
    fake_sdk.response = response()
    llm_config = LLMConfig(model="test-model", temperature=0.3, max_parallel_requests=4)
    with patch("autorubric.graders.criterion_grader.LLMClient", return_value=_LLM()):
        grader = CriterionGrader(
            judges=[
                JudgeSpec(
                    dm(
                        api_base=f"https://DM.Example.com:8443/{URL_SECRET}/v2/",
                        binary_framing="choice",
                        ordinal_framing="score",
                        decision_threshold=0.6,
                        max_parallel_requests=2,
                    ),
                    "jev",
                    weight=2.0,
                ),
                JudgeSpec(llm_config, "llm"),
            ],
            seed=11,
        )
    graders.append(grader)
    await run(grader, tmp_path, "manifest")

    manifest = json.loads((tmp_path / "manifest" / "manifest.json").read_text(encoding="utf-8"))
    jev, llm = manifest["grader_config"]["judges"]
    assert jev == {
        "judge_id": "jev",
        "judge_kind": "decision_model",
        "model": "jev-latest",
        "weight": 2.0,
        "max_parallel_requests": 2,
        "binary_framing": "choice",
        "ordinal_framing": "score",
        "decision_threshold": 0.6,
        "api_base_host": "dm.example.com:8443",
    }
    # An LLM judge keeps every key it had, with the same values, plus its kind.
    assert llm == {
        "judge_id": "llm",
        "judge_kind": "llm",
        "model": "test-model",
        "temperature": 0.3,
        "weight": 1.0,
        "max_parallel_requests": 4,
    }

    written = b"".join(p.read_bytes() for p in (tmp_path / "manifest").rglob("*") if p.is_file())
    assert written  # the manifest and the checkpoints
    assert API_KEY.encode() not in written
    assert URL_SECRET.encode() not in written


@pytest.mark.asyncio
async def test_manifest_records_the_resolved_default_host(fake_sdk, graders, tmp_path):
    fake_sdk.response = response()
    grader = CriterionGrader(judge_model_config=dm())
    graders.append(grader)
    await run(grader, tmp_path, "default-host", n_items=1)

    manifest = json.loads((tmp_path / "default-host" / "manifest.json").read_text(encoding="utf-8"))
    (jev,) = manifest["grader_config"]["judges"]
    assert jev["api_base_host"] == "api.typesafe.ai"
    assert (jev["binary_framing"], jev["ordinal_framing"]) == ("noul_framed", "choice")
    assert jev["decision_threshold"] == 0.5
    assert "temperature" not in jev  # an LLM-only setting


class _LLMFailingItem2:
    """An LLM stand-in giving the decision model's verdicts, whose calls fail on item 2."""

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> Any:
        if "Answer 2:" in user_prompt:
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://llm.test"))
        parsed: Any
        if "<options>" in user_prompt:
            parsed = MultiChoiceJudgment(selected_option=2, explanation="mostly clear")
        else:
            verdict = "MET" if "Mentions light" in user_prompt else "UNMET"
            parsed = CriterionJudgment(criterion_status=verdict, explanation="as asked")
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=None,
            parsed=parsed,
        )


@pytest.mark.asyncio
async def test_a_failed_request_scores_like_an_llm_grader_whose_calls_all_failed(
    fake_sdk, graders, tmp_path
):
    """One request per item: a request that fails after its retries fails every criterion of
    the item, each with an infrastructure abstention. The report is the one an LLM grader
    builds when every call of the item fails (the scoring core's 0.0 over no scored
    criterion, an ``error`` on every criterion report), and ``compute_metrics`` treats it
    exactly as it has always treated that LLM report: the item is scored, not counted as
    errored, so the two runs have identical metrics."""
    fake_sdk.response = response()
    grader = CriterionGrader(judge_model_config=dm())
    graders.append(grader)
    client = grader._decision_clients["default"]
    ask = client.system_one

    async def fail_item_2(state: Any, questions: Any, **kwargs: Any) -> SystemOneResponse:
        if state["submission"].startswith("Answer 2:"):
            raise typesafe_sdk.TypeSafeInternalServerError(
                503, {"error": "overloaded"}, httpx2.Headers({})
            )
        return await ask(state, questions, **kwargs)

    client.system_one = fail_item_2
    data = RubricDataset(prompt="Explain photosynthesis.", rubric=RUBRIC, name="one-fails")
    for i in range(4):
        data.add_item(
            submission=f"Answer {i}: plants use light to make sugar.",
            description=f"item {i}",
            ground_truth=GROUND_TRUTH[0],  # what every answered item gets right
        )
    result = await evaluate(
        data, grader, show_progress=False, experiment_name="dm-fails", experiments_dir=tmp_path
    )
    with patch("autorubric.graders.criterion_grader.LLMClient", return_value=_LLMFailingItem2()):
        llm_grader = CriterionGrader(
            judge_model_config=LLMConfig(model="test-model"), shuffle_options=False
        )
    llm_result = await evaluate(
        data, llm_grader, show_progress=False, experiment_name="llm-fails", experiments_dir=tmp_path
    )

    failed = next(r for r in result.item_results if r.item_idx == 2)
    llm_failed = next(r for r in llm_result.item_results if r.item_idx == 2)
    assert failed.error is None and failed.report.error is None
    assert failed.report.score == llm_failed.report.score == 0.0
    assert all(cr.error.startswith("infrastructure: ") for cr in failed.report.report)
    assert all(cr.error.startswith("infrastructure: ") for cr in llm_failed.report.report)

    metrics = result.compute_metrics(data)
    assert metrics.model_dump_json() == llm_result.compute_metrics(data).model_dump_json()
    assert metrics.n_items == 4
    assert metrics.coverage_stats is not None and metrics.coverage_stats.n_errored == 0
    assert not any("grading errored" in warning for warning in metrics.warnings)
    # Item 2's 0.0 is paired with its ground-truth score, as for any scored item.
    true_score = data.compute_weighted_score(GROUND_TRUTH[0])
    assert metrics.score_rmse == pytest.approx(math.sqrt(true_score**2 / 4))
    assert metrics.criterion_accuracy == pytest.approx(1.0)  # its abstentions are excluded


@pytest.mark.asyncio
async def test_reference_submissions_reach_every_request_per_dataset_and_per_item(
    fake_sdk, graders, tmp_path
):
    """The dataset's reference goes to every item without its own; an item's own wins."""
    fake_sdk.response = response()
    grader = CriterionGrader(judge_model_config=dm())
    graders.append(grader)
    data = RubricDataset(
        prompt="Explain photosynthesis.",
        rubric=RUBRIC,
        name="references",
        reference_submission="Dataset reference: light makes sugar.",
    )
    for i in range(3):
        data.add_item(
            submission=f"Answer {i}: plants use light to make sugar.",
            description=f"item {i}",
            ground_truth=GROUND_TRUTH[i],
            reference_submission="Item reference for answer 1." if i == 1 else None,
        )
    await evaluate(
        data, grader, show_progress=False, experiment_name="references", experiments_dir=tmp_path
    )

    references = {
        state["submission"][:8]: state.get("reference_submission") for state, _, _ in fake_sdk.calls
    }
    assert references == {
        "Answer 0": "Dataset reference: light makes sugar.",
        "Answer 1": "Item reference for answer 1.",
        "Answer 2": "Dataset reference: light makes sugar.",
    }
    for state, questions, _ in fake_sdk.calls:
        assert list(state) == ["input", "reference_submission", "submission"]
        assert "`reference_submission`" in questions["c0"].instructions


class _GraderWithoutDecisionClients:
    """A grader exposing ``_judges`` but no decision-model clients (a mock or another grader)."""

    def __init__(self, judges: list[JudgeSpec]) -> None:
        self._judges = judges


@pytest.mark.parametrize(
    "api_base, host",
    [
        (f"https://DM.Example.com:8443/{URL_SECRET}/v2/", "dm.example.com:8443"),
        ("https://dm.example.com:443/", "dm.example.com"),
        ("http://[::1]:9000", "[::1]:9000"),
        (None, None),
    ],
)
def test_manifest_host_falls_back_to_an_explicit_api_base(api_base, host):
    """Without a built client, ``api_base_host`` is the host of an explicit ``api_base``
    (lowercased, with a non-default port, never its path), else ``None``:
    the environment and the SDK default are resolved only when a client is built."""
    grader: Any = _GraderWithoutDecisionClients([JudgeSpec(dm(api_base=api_base), "jev")])

    (jev,) = _serialize_grader_config(grader)["judges"]

    assert jev["judge_kind"] == "decision_model"
    assert jev["api_base_host"] == host
    assert URL_SECRET not in json.dumps(jev) and API_KEY not in json.dumps(jev)
