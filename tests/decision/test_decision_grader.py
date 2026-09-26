"""Decision-model judges inside ``CriterionGrader``: one request per item.

A decision-model judge (``DecisionModelConfig``) grades an item with **one** request for the
whole rubric: the submission goes once as shared state and every expressible criterion of
the effective rubric becomes one question. Its answers map back to one per-criterion result
each, and everything downstream (aggregation, ``score_reports``, ``judge_scores``, usage and
cost) is the path LLM judges use.

The real ``DecisionModelClient`` runs (cache, rate limit, usage, cost) on top of the
recording fake SDK client of ``conftest.py`` (``fake_sdk``), so ``fake_sdk.calls`` counts
exactly the requests that would have reached the endpoint. LLM judges are recording fakes
patched in for ``LLMClient``. Nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import sys
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import httpx2
import pytest
import typesafe_sdk
from typesafe_sdk import Noul, SystemOneResponse

from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    Criterion,
    CriterionOption,
    CriterionReport,
    CriterionVerdict,
    DecisionModelConfig,
    FewShotConfig,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
)
from autorubric.decision import DecisionModelClient
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.meta import ImprovementConfig
from autorubric.prompts import (
    DECISION_MODEL_REFERENCE_INSTRUCTION,
    DECISION_MODEL_THINKING_OUTPUT_TASK_INSTRUCTION,
    MET_DEFINITION,
    UNMET_DEFINITION,
)
from autorubric.scoring import score_reports
from autorubric.types import CANONICAL_NA_OPTION, CriterionJudgment, MultiChoiceJudgment

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CANNOT_ASSESS = CriterionVerdict.CANNOT_ASSESS
NA_LABEL = CANONICAL_NA_OPTION.label

API_KEY = "ts-test-key-never-persisted-7f3c"
QUERY = "Explain photosynthesis."
SUBMISSION = "Plants use light to turn water and carbon dioxide into sugar."
REFERENCE = "Photosynthesis uses light energy to make glucose from CO2 and water."

LIGHT = Criterion(name="light", weight=5.0, requirement="Mentions light")
MYTH = Criterion(name="myth", weight=-3.0, requirement="Claims plants eat soil")
CLARITY = Criterion(
    name="clarity",
    weight=2.0,
    requirement="How clear is the explanation?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Unclear", value=0.0),
        CriterionOption(label="Mostly clear", value=0.6),
        CriterionOption(label="Very clear", value=1.0),
    ],
)
RUBRIC = [LIGHT, MYTH, CLARITY]
TONE_DUPLICATE_LABELS = Criterion(
    name="tone",
    weight=1.0,
    requirement="Which tone?",
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Formal", value=0.0),
    ],
)


def dm(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(**{"model": "jev-latest", "api_key": API_KEY, **overrides})


def noul(p: float) -> dict[str, Any]:
    return {"type": "noul", "noul": p}


def choice(selected: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": selected,
        "confidence": 0.5,
        "probabilities": probabilities,
    }


CLARITY_MOSTLY = choice(
    "Mostly clear", {"Unclear": 0.1, "Mostly clear": 0.7, "Very clear": 0.15, NA_LABEL: 0.05}
)
ANSWERS = {"c0": noul(0.9), "c1": noul(0.2), "c2": CLARITY_MOSTLY}


def response(
    answers: dict[str, Any] | None = None,
    *,
    input_tokens: int | None = 1234,
    output_tokens: int | None = 7,
) -> SystemOneResponse:
    usage: dict[str, int] = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    payload = {
        "model": "jev-test",
        "usage": usage,
        "answers": ANSWERS if answers is None else answers,
    }
    return SystemOneResponse.model_validate_json(json.dumps(payload))


def wire(questions: Any) -> dict[str, Any]:
    """Questions as the JSON the endpoint receives."""
    return {qid: question.model_dump(mode="json") for qid, question in questions.items()}


def request_json(call: tuple[Any, Any, dict[str, Any]]) -> str:
    """One recorded request (state and questions) as JSON text; key order counts."""
    state, questions, _ = call
    return json.dumps([state, wire(questions)])


@pytest.fixture
def graders() -> Iterator[list[CriterionGrader]]:
    """Graders whose decision-model caches are closed at teardown (Windows keeps them open)."""
    built: list[CriterionGrader] = []
    yield built
    for grader in built:
        for client in grader._decision_clients.values():
            client.close()


@pytest.fixture
def make_grader(graders: list[CriterionGrader]):
    def build(**kwargs: Any) -> CriterionGrader:
        grader = CriterionGrader(**kwargs)
        graders.append(grader)
        return grader

    return build


class RecordingLLMClient:
    """Stand-in for ``LLMClient``: records prompts, MET / first option with an explanation."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.prompts: list[tuple[str, str]] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult:
        self.prompts.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        parsed: Any
        if "<options>" in user_prompt:
            parsed = MultiChoiceJudgment(selected_option=3, explanation="llm: very clear")
        else:
            parsed = CriterionJudgment(criterion_status=MET, explanation="llm: present")
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=0.001,
            parsed=parsed,
        )


def patch_llm_client(client: RecordingLLMClient):
    return patch("autorubric.graders.criterion_grader.LLMClient", return_value=client)


# =============================================================================
# Construction: type widening, client map, the few-shot rule
# =============================================================================


class TestConstruction:
    def test_judge_model_config_accepts_a_decision_model(self, make_grader):
        grader = make_grader(judge_model_config=dm())
        assert [j.judge_id for j in grader._judges] == ["default"]
        assert isinstance(grader._judges[0].judge_model_config, DecisionModelConfig)
        assert isinstance(grader._decision_clients["default"], DecisionModelClient)
        assert "default" not in grader._clients

    def test_deprecated_llm_config_alias_accepts_a_decision_model(self, make_grader):
        with pytest.warns(DeprecationWarning, match="llm_config is deprecated"):
            grader = make_grader(llm_config=dm())
        assert isinstance(grader._decision_clients["default"], DecisionModelClient)

    def test_judge_spec_accepts_a_decision_model_in_every_form(self):
        config = dm()
        positional = JudgeSpec(config, "jev", 2.0)
        keyword = JudgeSpec(judge_model_config=config, judge_id="jev", weight=2.0)
        stored = JudgeSpec(llm_config=config, judge_id="jev", weight=2.0)
        assert positional == keyword == stored
        assert keyword.judge_model_config is config

    def test_mixed_ensemble_builds_one_client_of_each_kind(self, make_grader):
        grader = make_grader(
            judges=[JudgeSpec(dm(), "jev"), JudgeSpec(LLMConfig(model="test-model"), "llm")]
        )
        assert list(grader._decision_clients) == ["jev"]
        assert list(grader._clients) == ["llm"]

    def test_missing_sdk_fails_at_construction(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
        with pytest.raises(ImportError, match=r"pip install 'autorubric\[typesafe\]'"):
            CriterionGrader(judge_model_config=dm())

    def test_missing_api_key_fails_at_construction(self):
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            CriterionGrader(judge_model_config=DecisionModelConfig(model="jev-latest"))

    def test_malformed_api_key_fails_at_construction_not_on_every_item(self):
        """A key the SDK would reject is as unusable as a missing one: the grader fails at
        construction instead of abstaining on every criterion of every item."""
        with pytest.raises(ValueError, match="printable ASCII"):
            CriterionGrader(judge_model_config=dm(api_key="sk abc def"))

    def test_base_url_with_a_line_break_fails_at_construction_not_on_every_item(self, monkeypatch):
        """The HTTP layer refuses such a URL on every request, as an error no category covers,
        which would give every criterion of every item its conservative worst case."""
        with pytest.raises(ValueError, match="api_base"):
            CriterionGrader(judge_model_config=dm(api_base="https://dm.example.com\n"))
        monkeypatch.setenv("TYPESAFE_BASE_URL", "https://dm.exam\r\nple.com")
        with pytest.raises(ValueError, match="TYPESAFE_BASE_URL"):
            CriterionGrader(judge_model_config=dm())

    def test_base_url_with_an_invalid_host_fails_at_construction_not_on_every_item(
        self, monkeypatch
    ):
        """A host the HTTP layer cannot encode (here fullwidth letters, not a valid IDNA
        name) raises inside every request, as an error no category covers, which would give
        every criterion of every item its conservative worst case."""
        fullwidth = "https://\uff41\uff50\uff49.typesafe.ai"
        with pytest.raises(ValueError, match="api_base"):
            CriterionGrader(judge_model_config=dm(api_base=fullwidth))
        monkeypatch.setenv("TYPESAFE_BASE_URL", fullwidth)
        with pytest.raises(ValueError, match="TYPESAFE_BASE_URL"):
            CriterionGrader(judge_model_config=dm())

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://dm.example.com/api?tenant=a",
            "https://dm.example.com/api#x",
            "https://svc:pw@dm.example.com/api",
        ],
        ids=["query", "fragment", "credentials"],
    )
    def test_base_url_that_misdirects_every_request_fails_at_construction(self, api_base):
        """After a query or fragment the appended endpoint path is lost (every request gets
        a 404), and URL credentials replace the bearer key (every request gets a 401): an
        abstention on every criterion of every item."""
        with pytest.raises(ValueError, match="api_base"):
            CriterionGrader(judge_model_config=dm(api_base=api_base))

    def test_malformed_extra_header_fails_at_construction_not_on_every_item(self):
        """The HTTP layer refuses such a header on every request after every retry, and its
        error, which repeats the value, would be recorded in every failed report."""
        with pytest.raises(ValueError, match="extra_headers") as excinfo:
            CriterionGrader(
                judges=[
                    JudgeSpec(dm(extra_headers={"X-Goog-Api-Key": "AIzaSECRET\n"}), "jev"),
                    JudgeSpec(LLMConfig(model="test-model"), "llm"),
                ]
            )
        assert "SECRET" not in str(excinfo.value)

    def test_an_environment_no_http_client_can_be_built_in_fails_at_construction(
        self, http_environment
    ):
        """httpx2 reads TLS and proxy settings from the environment (here a CA file that does
        not exist, a SOCKS proxy without ``socksio``, or a proxy URL with a scheme no proxy
        has), outside the SDK's error handling. Raised inside every request, its error would
        give every criterion of every item the conservative worst case."""
        http_environment.break_()
        with pytest.raises(ValueError, match="SSL_CERT_FILE") as excinfo:
            CriterionGrader(
                judges=[JudgeSpec(dm(), "jev"), JudgeSpec(LLMConfig(model="test-model"), "llm")]
            )
        assert "'jev-latest'" in str(excinfo.value)


def training_data() -> RubricDataset:
    dataset = RubricDataset(prompt=QUERY, rubric=Rubric(RUBRIC), name="train")
    dataset.add_item("Light drives it.", "good", ground_truth=[MET, UNMET, "Very clear"])
    dataset.add_item("Plants eat soil.", "bad", ground_truth=[UNMET, MET, "Unclear"])
    dataset.add_item("Sun helps.", "ok", ground_truth=[MET, UNMET, "Mostly clear"])
    return dataset


class TestFewShotRule:
    @pytest.mark.parametrize(
        "few_shot",
        [{"few_shot_config": FewShotConfig(n_examples=2)}, {"training_data": training_data()}],
        ids=["few_shot_config", "training_data"],
    )
    def test_few_shot_with_only_decision_models_is_rejected(self, few_shot):
        with pytest.raises(ValueError, match="few-shot examples apply to LLM judges only"):
            CriterionGrader(judge_model_config=dm(), **few_shot)
        with pytest.raises(ValueError, match="few-shot examples apply to LLM judges only"):
            CriterionGrader(judges=[JudgeSpec(dm(), "a"), JudgeSpec(dm(), "b")], **few_shot)

    @pytest.mark.asyncio
    async def test_few_shot_reaches_llm_judges_only(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        judges = [JudgeSpec(dm(), "jev"), JudgeSpec(LLMConfig(model="test-model"), "llm")]
        with patch_llm_client(RecordingLLMClient()):
            plain = make_grader(judges=judges, seed=3)
        llm = RecordingLLMClient()
        with patch_llm_client(llm):
            few_shot = make_grader(
                judges=judges,
                seed=3,
                training_data=training_data(),
                few_shot_config=FewShotConfig(n_examples=2),
            )

        llm_alone = RecordingLLMClient()
        with patch_llm_client(llm_alone):
            without_decision_model = make_grader(
                judges=[judges[1]],
                seed=3,
                training_data=training_data(),
                few_shot_config=FewShotConfig(n_examples=2),
            )

        # Examples are selected for the LLM judge alone, exactly as without the decision model.
        selected = {*few_shot._criterion_examples, *few_shot._multi_choice_examples}
        assert selected and {judge_id for _, judge_id in selected} == {"llm"}
        assert few_shot._criterion_examples == without_decision_model._criterion_examples
        assert few_shot._multi_choice_examples == without_decision_model._multi_choice_examples

        await Rubric(RUBRIC).grade(SUBMISSION, grader=plain, query=QUERY)
        await Rubric(RUBRIC).grade(SUBMISSION, grader=few_shot, query=QUERY)
        await Rubric(RUBRIC).grade(SUBMISSION, grader=without_decision_model, query=QUERY)

        # The decision-model request is the same with and without few-shot.
        assert len(fake_sdk.calls) == 2
        assert request_json(fake_sdk.calls[0]) == request_json(fake_sdk.calls[1])
        # The LLM judge's prompts carry the examples, as they would with no decision model.
        assert len(llm.prompts) == len(RUBRIC)
        assert all("<examples>" in user for _, user in llm.prompts)
        assert sorted(llm.prompts) == sorted(llm_alone.prompts)


# =============================================================================
# One request per item
# =============================================================================


class TestOneRequestPerItem:
    @pytest.mark.asyncio
    async def test_standalone_sends_one_request_with_every_criterion(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        assert len(fake_sdk.calls) == 1
        state, questions, _ = fake_sdk.calls[0]
        assert state == {"input": QUERY, "submission": SUBMISSION}
        assert list(questions) == ["c0", "c1", "c2"]
        assert [q.type for q in questions.values()] == ["noul", "noul", "choice"]
        assert questions["c0"].criteria == {"true": MET_DEFINITION, "false": UNMET_DEFINITION}
        # The effective rubric: the ordinal criterion carries its auto-injected NA option.
        assert list(questions["c2"].criteria) == ["Unclear", "Mostly clear", "Very clear", NA_LABEL]

        light, myth, clarity = report.report
        assert light.final_verdict == MET and myth.final_verdict == UNMET
        assert clarity.final_multi_choice_verdict.selected_index == 1
        assert clarity.final_multi_choice_verdict.value == 0.6

        vote = light.votes[0]
        assert vote.judge_id == "default"
        assert vote.probabilities == {"MET": 0.9, "UNMET": pytest.approx(0.1)}
        assert vote.confidence == pytest.approx(0.8)
        assert vote.reason is None and vote.reasoning is None and vote.error is None
        mc_vote = clarity.multi_choice_votes[0]
        assert mc_vote.probabilities == {"0": 0.1, "1": 0.7, "2": 0.15, "3": 0.05}
        assert mc_vote.confidence == pytest.approx((4 * 0.7 - 1) / 3)
        assert mc_vote.shuffle_order is None
        assert all(cr.final_reason is None for cr in report.report)

    @pytest.mark.asyncio
    async def test_a_reference_submission_goes_into_the_state_with_its_usage_rule(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        await Rubric(RUBRIC).grade(
            SUBMISSION, grader=grader, query=QUERY, reference_submission=REFERENCE
        )

        (call,) = fake_sdk.calls
        state, questions, _ = call
        assert list(state.items()) == [
            ("input", QUERY),
            ("reference_submission", REFERENCE),
            ("submission", SUBMISSION),
        ]
        rule = DECISION_MODEL_REFERENCE_INSTRUCTION.format(judged="submission")
        # The framed binary questions carry the rule; the multi-choice one sees the state only.
        assert rule in questions["c0"].instructions and rule in questions["c1"].instructions
        assert questions["c2"].instructions == CLARITY.requirement

    @pytest.mark.asyncio
    async def test_no_reference_submission_means_no_state_field_and_no_rule(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        state, questions, _ = fake_sdk.calls[0]
        assert "reference_submission" not in state
        assert "reference_submission" not in questions["c0"].instructions

    @pytest.mark.asyncio
    async def test_mixed_ensemble_one_request_per_decision_model_and_n_llm_calls(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response()
        llm = RecordingLLMClient()
        with patch_llm_client(llm):
            grader = make_grader(
                judges=[
                    JudgeSpec(dm(), "jev-a"),
                    JudgeSpec(LLMConfig(model="test-model"), "llm"),
                    JudgeSpec(dm(decision_threshold=0.95), "jev-b"),
                ]
            )
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        assert len(fake_sdk.calls) == 2  # one per decision-model judge
        assert len(llm.prompts) == len(RUBRIC)  # one per criterion for the LLM judge
        light = report.report[0]
        assert [(v.judge_id, v.verdict) for v in light.votes] == [
            ("jev-a", MET),
            ("llm", MET),
            ("jev-b", UNMET),  # P(MET) = 0.9 is below its threshold
        ]
        assert light.final_verdict == MET
        # Only the LLM vote explains itself.
        assert light.final_reason == "llm: llm: present"
        assert [v.confidence is None for v in light.votes] == [False, True, False]
        assert set(report.judge_scores) == {"jev-a", "llm", "jev-b"}

    @pytest.mark.asyncio
    async def test_inexpressible_criterion_is_left_out_of_the_single_request(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response({"c0": noul(0.9), "c2": noul(0.2), "c3": CLARITY_MOSTLY})
        grader = make_grader(judge_model_config=dm())
        rubric = Rubric([LIGHT, TONE_DUPLICATE_LABELS, MYTH, CLARITY])
        report = await rubric.grade(SUBMISSION, grader=grader, query=QUERY)

        assert len(fake_sdk.calls) == 1
        assert list(fake_sdk.calls[0][1]) == ["c0", "c2", "c3"]
        tone = report.report[1]
        vote = tone.multi_choice_votes[0]
        assert vote.error.startswith("parse: Cannot pose criterion c1 ('tone')")
        assert vote.na is True and tone.final_multi_choice_verdict.na is True
        assert [cr.is_error for cr in report.report] == [False, True, False, False]
        assert report.report[0].final_verdict == MET

    @pytest.mark.asyncio
    async def test_all_criteria_inexpressible_sends_nothing(self, fake_sdk, make_grader):
        grader = make_grader(judge_model_config=dm())
        report = await Rubric([TONE_DUPLICATE_LABELS]).grade(SUBMISSION, grader=grader)

        assert fake_sdk.calls == []
        assert report.report[0].error.startswith("parse: Cannot pose criterion c0")
        assert report.report[0].final_multi_choice_verdict.na is True
        assert report.token_usage is None and report.completion_cost is None

    @pytest.mark.asyncio
    async def test_missing_and_ill_typed_answers_fail_only_their_criterion(
        self, fake_sdk, make_grader
    ):
        # c0 is missing; c1 is a Choice answer to a Noul question; c2 is fine.
        fake_sdk.response = response(
            {
                "c1": choice("MET", {"MET": 0.8, "UNMET": 0.1, "CANNOT_ASSESS": 0.1}),
                "c2": CLARITY_MOSTLY,
            }
        )
        grader = make_grader(judge_model_config=dm())
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        assert len(fake_sdk.calls) == 1
        light, myth, clarity = report.report
        assert light.votes[0].error == (
            "parse: The decision model returned no answer for criterion c0 ('light')"
        )
        assert light.final_verdict == CANNOT_ASSESS
        assert myth.votes[0].error.startswith(
            "parse: Unusable decision-model answer for criterion c1 ('myth'): expected a noul"
        )
        assert myth.final_verdict == CANNOT_ASSESS
        assert not clarity.is_error
        # Only the answered criterion scores: 0.6 of its weight 2, over positive weight 2.
        assert report.score == pytest.approx(0.6)
        assert report.cannot_assess_count == 2
        # The request was made: its usage is reported, though c0 (first) has no answer.
        assert report.token_usage == TokenUsage(1234, 7, 1241)

    @pytest.mark.asyncio
    async def test_thinking_output_submission_goes_out_as_two_state_fields(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        to_grade = {"thinking": "Recall the light reactions.", "output": SUBMISSION}
        await Rubric(RUBRIC).grade(to_grade, grader=grader, query=QUERY)

        state, questions, _ = fake_sdk.calls[0]
        assert state == {
            "input": QUERY,
            "thinking": "Recall the light reactions.",
            "output": SUBMISSION,
        }
        assert questions["c0"].instructions.startswith(
            DECISION_MODEL_THINKING_OUTPUT_TASK_INSTRUCTION
        )

    @pytest.mark.asyncio
    async def test_plain_submission_is_sent_unchanged(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        padded = "  Plants use light.  \n"
        await Rubric(RUBRIC).grade(padded, grader=grader)
        assert fake_sdk.calls[0][0] == {"submission": padded}

    @pytest.mark.asyncio
    async def test_cache_hit_sends_nothing_and_reports_the_same(
        self, fake_sdk, make_grader, tmp_path
    ):
        fake_sdk.response = response()
        config = dm(cache_enabled=True, cache_dir=tmp_path / "cache", input_cost_per_token=1e-6)
        first = await Rubric(RUBRIC).grade(
            SUBMISSION, grader=make_grader(judge_model_config=config), query=QUERY
        )
        assert len(fake_sdk.calls) == 1

        again = await Rubric(RUBRIC).grade(
            SUBMISSION, grader=make_grader(judge_model_config=config), query=QUERY
        )
        assert len(fake_sdk.calls) == 1
        assert again.model_dump() == first.model_dump()
        assert again.token_usage == first.token_usage == TokenUsage(1234, 7, 1241)


# =============================================================================
# Response-cache keys: each judge of an ensemble or cascade has its own
# =============================================================================


def _key(client: DecisionModelClient) -> str:
    return client._cache_key(
        {"submission": SUBMISSION}, {"c0": Noul(instructions="Mentions light")}
    )


class TestJudgeCacheKeys:
    """Decision-model judges of an ensemble or cascade keep their own response-cache entries.

    As for LLM judges (``tests/graders/test_judge_cache_keys.py``): judges of one decision
    model send identical requests, so without a per-judge key a rerun would replay one
    judge's answers to all of them. A lone judge (``judge_id="default"``) keeps the plain
    key, so its existing caches stay valid.
    """

    @pytest.mark.asyncio
    async def test_rerun_of_repeated_judges_reads_each_judges_own_answer(
        self, fake_sdk, make_grader, monkeypatch, tmp_path
    ):
        replies = itertools.cycle([response({"c0": noul(p)}) for p in (0.9, 0.2, 0.3)])

        async def respond() -> SystemOneResponse:
            return next(replies)

        monkeypatch.setattr(fake_sdk, "respond", respond)
        config = dm(cache_enabled=True, cache_dir=tmp_path / "cache")
        runs = []
        for _ in range(2):
            sent_before = len(fake_sdk.calls)
            grader = make_grader(judges=[JudgeSpec(config, f"jev-{i}") for i in range(3)])
            report = await Rubric([LIGHT]).grade(SUBMISSION, grader=grader, query=QUERY)
            votes = {
                vote.judge_id: (vote.verdict, vote.confidence) for vote in report.report[0].votes
            }
            runs.append((votes, len(fake_sdk.calls) - sent_before))

        (first_votes, first_sent), (rerun_votes, rerun_sent) = runs
        assert first_sent == 3
        assert sorted(verdict.value for verdict, _ in first_votes.values()) == [
            "MET",
            "UNMET",
            "UNMET",
        ]
        assert len({confidence for _, confidence in first_votes.values()}) == 3
        assert rerun_sent == 0
        assert rerun_votes == first_votes

    def test_a_lone_judge_keeps_the_plain_cache_key(self, make_grader):
        config = dm()
        lone = make_grader(judge_model_config=config)._decision_clients["default"]

        assert _key(lone) == _key(DecisionModelClient(config))

    def test_ensemble_judges_of_one_model_have_their_own_cache_keys(self, make_grader):
        config = dm()
        grader = make_grader(judges=[JudgeSpec(config, "a"), JudgeSpec(config, "b")])
        keys = {judge_id: _key(client) for judge_id, client in grader._decision_clients.items()}

        assert keys["a"] != keys["b"]
        assert _key(DecisionModelClient(config)) not in keys.values()


# =============================================================================
# Whole-request failure: every posed criterion fails with it, never re-sent
# =============================================================================


REQUEST_FAILURES = {
    "infrastructure": lambda: typesafe_sdk.TypeSafeAPIConnectionError("Connection error: refused"),
    "parse": lambda: typesafe_sdk.TypeSafeBadRequestError(
        400, {"detail": "bad"}, httpx2.Headers({})
    ),
    "unknown": lambda: RuntimeError("boom"),
}


class TestWholeRequestFailure:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("category", list(REQUEST_FAILURES))
    async def test_every_criterion_gets_the_llm_failure_report(
        self, fake_sdk, make_grader, category
    ):
        error = REQUEST_FAILURES[category]()
        fake_sdk.error = error
        grader = make_grader(judge_model_config=dm())
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        assert len(fake_sdk.calls) == 1  # no per-criterion retry
        for cr in report.report:
            vote = (cr.votes or cr.multi_choice_votes)[0]
            assert vote.error == f"{category}: {error}"
            assert vote.reason == f"Judge call failed ({category}): {error}"
            assert vote.probabilities is None and vote.confidence is None

        light, myth, clarity = report.report
        if category == "unknown":
            # The conservative worst case, as for LLM judges.
            assert light.final_verdict == UNMET and myth.final_verdict == MET
            assert clarity.final_multi_choice_verdict.selected_index == 0
            assert clarity.final_multi_choice_verdict.na is False
        else:
            assert light.final_verdict == myth.final_verdict == CANNOT_ASSESS
            assert clarity.final_multi_choice_verdict.na is True
            assert clarity.final_multi_choice_verdict.selected_label == NA_LABEL
        assert report.token_usage is None and report.completion_cost is None

        # Built by the same failure-report helper as an LLM judge's failed call: an LLM
        # grader whose every call raises the same exception produces the identical report.
        with patch_llm_client(RecordingLLMClient(error=error)):
            llm_grader = CriterionGrader(
                judge_model_config=LLMConfig(model="test-model"), shuffle_options=False
            )
        llm_report = await Rubric(RUBRIC).grade(SUBMISSION, grader=llm_grader, query=QUERY)
        assert report.model_dump() == llm_report.model_dump()

    @pytest.mark.asyncio
    async def test_an_environment_broken_after_construction_abstains_never_the_worst_case(
        self, fake_sdk, make_grader, http_environment
    ):
        """An environment that changes during a run so that no HTTP client can be built
        (TLS or proxy settings) says nothing about the submission: every criterion abstains
        (``infrastructure``), as when the endpoint is unreachable, instead of getting the
        conservative worst case (UNMET for a positive weight, MET for a negative one)."""
        grader = make_grader(judge_model_config=dm())
        http_environment.break_()

        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        assert fake_sdk.calls == []
        light, myth, clarity = report.report
        assert light.final_verdict == myth.final_verdict == CANNOT_ASSESS
        assert clarity.final_multi_choice_verdict.na is True
        for cr in report.report:
            vote = (cr.votes or cr.multi_choice_votes)[0]
            assert vote.error.startswith("infrastructure: ")

    @pytest.mark.asyncio
    async def test_an_inexpressible_criterion_keeps_its_own_error(self, fake_sdk, make_grader):
        fake_sdk.error = RuntimeError("boom")
        grader = make_grader(judge_model_config=dm())
        report = await Rubric([LIGHT, TONE_DUPLICATE_LABELS]).grade(SUBMISSION, grader=grader)

        assert list(fake_sdk.calls[0][1]) == ["c0"]
        assert report.report[0].votes[0].error == "unknown: boom"
        assert report.report[1].multi_choice_votes[0].error.startswith("parse: Cannot pose")


# =============================================================================
# Successive event loops under a concurrency limit
# =============================================================================


class TestSuccessiveEventLoops:
    def test_rate_limited_requests_on_a_later_loop_are_answered_like_the_first(
        self, fake_sdk, make_grader
    ):
        """One grader, two ``asyncio.run`` batches, ``max_parallel_requests=1``: the requests
        that wait for the limit in the second batch are answered exactly like the first
        batch's, never failed by a semaphore that belongs to the first batch's loop."""
        fake_sdk.response = response()
        fake_sdk.delay = 0.01
        grader = make_grader(judge_model_config=dm(max_parallel_requests=1))

        async def batch() -> list[Any]:
            return await asyncio.gather(
                *(
                    Rubric(RUBRIC).grade(f"{SUBMISSION} ({i})", grader=grader, query=QUERY)
                    for i in range(3)
                )
            )

        first, second = asyncio.run(batch()), asyncio.run(batch())

        assert len(fake_sdk.calls) == 6
        assert fake_sdk.max_in_flight == 1
        for report in first + second:
            assert not any(cr.is_error for cr in report.report)
        assert [r.model_dump() for r in second] == [r.model_dump() for r in first]


# =============================================================================
# Scoring, usage and cost
# =============================================================================


class TestScoringUsageCost:
    @pytest.mark.asyncio
    async def test_score_goes_through_score_reports(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        # light MET (+5), myth UNMET (0), clarity "Mostly clear" (0.6 * 2), over weight 7.
        assert report.score == pytest.approx(6.2 / 7)
        assert report.raw_score == pytest.approx(6.2)
        per_judge = [
            CriterionReport(
                **{name: getattr(cr.criterion, name) for name in Criterion.model_fields},
                verdict=cr.final_verdict,
                multi_choice_verdict=cr.final_multi_choice_verdict,
                reason=cr.final_reason,
            )
            for cr in report.report
        ]
        assert report.score == score_reports(per_judge, CannotAssessConfig())
        assert report.judge_scores == {"default": report.score}

    @pytest.mark.asyncio
    async def test_abstentions_are_skipped_by_default(self, fake_sdk, make_grader):
        fake_sdk.response = response(
            {
                "c0": choice("CANNOT_ASSESS", {"MET": 0.2, "UNMET": 0.2, "CANNOT_ASSESS": 0.6}),
                "c1": choice("UNMET", {"MET": 0.1, "UNMET": 0.8, "CANNOT_ASSESS": 0.1}),
                "c2": CLARITY_MOSTLY,
            }
        )
        grader = make_grader(judge_model_config=dm(binary_framing="choice"))
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)

        assert report.report[0].final_verdict == CANNOT_ASSESS
        assert not report.report[0].is_error  # a genuine abstention, not an error
        assert report.cannot_assess_count == 1
        assert report.score == pytest.approx(1.2 / 2)  # light leaves the denominator

        zero = make_grader(
            judge_model_config=dm(binary_framing="choice"),
            cannot_assess_config=CannotAssessConfig(strategy=CannotAssessStrategy.ZERO),
        )
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=zero, query=QUERY)
        assert report.score == pytest.approx(1.2 / 7)

    @pytest.mark.asyncio
    async def test_usage_rides_on_the_first_result_and_unknown_cost_is_none(
        self, fake_sdk, make_grader
    ):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm())
        (judge_results,) = await grader.judge(SUBMISSION, RUBRIC, QUERY)
        usages = [r.usage for r in judge_results.criterion_results]
        assert usages == [TokenUsage(1234, 7, 1241), None, None]
        assert judge_results.total_usage == TokenUsage(1234, 7, 1241)
        assert judge_results.total_cost is None

        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        assert report.token_usage == TokenUsage(1234, 7, 1241)
        # No price configured: the cost is unknown, and the report says so with None.
        assert report.completion_cost is None

    @pytest.mark.asyncio
    async def test_usage_and_cost_ride_on_the_first_result_even_when_it_failed(
        self, fake_sdk, make_grader
    ):
        """The first result carries the request's usage and cost whatever its outcome:
        a first criterion that cannot be posed still bills the request its neighbours were
        asked in, so the item's usage and cost are never lost."""
        fake_sdk.response = response({"c1": noul(0.9)})
        grader = make_grader(judge_model_config=dm(input_cost_per_token=1e-6))
        rubric = [TONE_DUPLICATE_LABELS, LIGHT]

        (judge_results,) = await grader.judge(SUBMISSION, rubric, QUERY)
        first, second = judge_results.criterion_results
        assert first.report.error is not None and first.report.error.startswith("parse:")
        assert second.report.error is None
        assert (first.usage, first.cost) == (TokenUsage(1234, 7, 1241), 1234 * 1e-6)
        assert (second.usage, second.cost) == (None, None)

        report = await Rubric(rubric).grade(SUBMISSION, grader=grader, query=QUERY)
        assert [list(call[1]) for call in fake_sdk.calls] == [["c1"], ["c1"]]
        assert report.report[0].is_error and not report.report[1].is_error
        assert report.token_usage == TokenUsage(1234, 7, 1241)
        assert report.completion_cost == 1234 * 1e-6

    @pytest.mark.asyncio
    async def test_cost_with_a_price_is_exact(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        grader = make_grader(judge_model_config=dm(input_cost_per_token=0.042e-6))
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        assert report.completion_cost == 1234 * 0.042e-6

    @pytest.mark.asyncio
    async def test_mixed_ensemble_sums_usage_and_cost(self, fake_sdk, make_grader):
        fake_sdk.response = response()
        with patch_llm_client(RecordingLLMClient()):
            grader = make_grader(
                judges=[
                    JudgeSpec(dm(input_cost_per_token=1e-6), "jev"),
                    JudgeSpec(LLMConfig(model="test-model"), "llm"),
                ]
            )
        report = await Rubric(RUBRIC).grade(SUBMISSION, grader=grader, query=QUERY)
        assert report.token_usage == TokenUsage(1234 + 3 * 10, 7 + 3 * 5, 1241 + 3 * 15)
        assert report.completion_cost == pytest.approx(1234 * 1e-6 + 3 * 0.001)


# =============================================================================
# Roles decision models cannot fill
# =============================================================================


class TestImprovementLoopRejectsDecisionModels:
    def test_decision_model_eval_judge_is_rejected(self):
        llm = LLMConfig(model="test-model")
        with pytest.raises(ValueError, match="eval_llm.*decision model"):
            ImprovementConfig(
                eval_llm=[JudgeSpec(llm, "llm"), JudgeSpec(dm(), "jev")], revision_llm=llm
            )
        with pytest.raises(ValueError, match="eval_llm.*decision model"):
            ImprovementConfig(eval_llm=dm(), revision_llm=llm)  # the type says LLMConfig

    def test_llm_eval_judges_are_accepted(self):
        llm = LLMConfig(model="test-model")
        config = ImprovementConfig(
            eval_llm=[JudgeSpec(llm, "a"), JudgeSpec(llm, "b")], revision_llm=llm
        )
        assert dataclasses.replace(config, eval_llm=llm).eval_llm is llm
