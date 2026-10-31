"""A decision model's, or a cascade's escalation judge's, configuration error raises (#40).

A 401, 403 or 404 from TypeSafe (a revoked key, a key without access, a base URL that
names no System One endpoint) fails every request the judge makes, like an LLM judge's
configuration error: the grade raises a ``ValueError`` naming the judge, chained from the
SDK's exception, and never abstains, nor escalates in a cascade. Every other request
failure is routed as before. A cascade's escalation judges are LLM judges, and their
configuration errors raise as any LLM judge's do. A parameter LiteLLM refuses raises on the
judge's first call.

The decision model runs on the recording fake SDK client of ``conftest.py``; LLM judges are
local stubs, or a real ``LLMClient`` whose call LiteLLM refuses before any request. Nothing
reaches the network.
"""

from __future__ import annotations

from typing import Any

import httpx2
import litellm
import pytest
import typesafe_sdk

from autorubric import (
    Criterion,
    CriterionVerdict,
    DecisionModelConfig,
    EscalationConfig,
    LLMConfig,
    Rubric,
    classify_grading_error,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, _is_configuration_error
from autorubric.types import CriterionJudgment

API_KEY = "ts-test-key-never-persisted-40c1"
RUBRIC = Rubric(
    [
        Criterion(name="light", weight=5.0, requirement="Mentions light"),
        Criterion(name="myth", weight=-3.0, requirement="Claims plants eat soil"),
    ]
)
SUBMISSION = "Plants use light to make sugar."


def dm() -> DecisionModelConfig:
    return DecisionModelConfig(model="jev-latest", api_key=API_KEY)


def _headers() -> httpx2.Headers:
    return httpx2.Headers({})


CONFIGURATION_ERRORS = {
    "TypeSafeAuthenticationError": lambda: typesafe_sdk.TypeSafeAuthenticationError(
        401, {"detail": "invalid key"}, _headers()
    ),
    "TypeSafePermissionDeniedError": lambda: typesafe_sdk.TypeSafePermissionDeniedError(
        403, {"detail": "no access"}, _headers()
    ),
    "TypeSafeNotFoundError": lambda: typesafe_sdk.TypeSafeNotFoundError(
        404, {"detail": "not found"}, _headers()
    ),
}
ROUTED_ERRORS = {
    "TypeSafeAPIConnectionError": lambda: typesafe_sdk.TypeSafeAPIConnectionError("refused"),
    "TypeSafeRateLimitError": lambda: typesafe_sdk.TypeSafeRateLimitError(429, None, _headers()),
    "TypeSafeBadRequestError": lambda: typesafe_sdk.TypeSafeBadRequestError(
        400, {"detail": "bad"}, _headers()
    ),
    "TypeSafeUnprocessableEntityError": lambda: typesafe_sdk.TypeSafeUnprocessableEntityError(
        422, {"detail": "too long"}, _headers()
    ),
}


class StubLLM:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls = 0

    async def generate(self, **kwargs: Any) -> GenerateResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return GenerateResult(
            content="{}",
            parsed=CriterionJudgment(criterion_status=CriterionVerdict.MET, explanation="ok"),
        )


@pytest.fixture
def make_grader():
    built: list[CriterionGrader] = []

    def build(**kwargs: Any) -> CriterionGrader:
        grader = CriterionGrader(**kwargs)
        built.append(grader)
        return grader

    yield build
    for grader in built:
        for client in grader._decision_clients.values():
            client.close()


class TestTheClassification:
    @pytest.mark.parametrize("name", list(CONFIGURATION_ERRORS))
    def test_configuration_errors(self, name):
        error = CONFIGURATION_ERRORS[name]()
        assert _is_configuration_error(error)
        assert classify_grading_error(error) == "infrastructure"  # the public taxonomy

    @pytest.mark.parametrize("name", list(ROUTED_ERRORS))
    def test_other_request_failures_are_not(self, name):
        assert not _is_configuration_error(ROUTED_ERRORS[name]())


class TestTheDecisionModel:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(CONFIGURATION_ERRORS))
    async def test_the_grade_raises(self, fake_sdk, make_grader, name):
        error = CONFIGURATION_ERRORS[name]()
        fake_sdk.error = error
        grader = make_grader(judge_model_config=dm())

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        assert f"Judge 'default' is misconfigured ({name})" in str(excinfo.value)
        assert excinfo.value.__cause__ is error
        assert len(fake_sdk.calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(ROUTED_ERRORS))
    async def test_other_failures_are_routed(self, fake_sdk, make_grader, name):
        fake_sdk.error = ROUTED_ERRORS[name]()
        grader = make_grader(judge_model_config=dm())

        report = await RUBRIC.grade(SUBMISSION, grader=grader)

        assert report.score is None
        assert report.error is not None and report.error.startswith(
            "Every criterion's judgment failed"
        )

    @pytest.mark.asyncio
    async def test_a_cascade_never_escalates_it(self, fake_sdk, make_grader):
        """A misconfigured decision model raises: escalating every criterion to the LLM
        judges on every item would hide it behind a run made of LLM verdicts alone."""
        fake_sdk.error = CONFIGURATION_ERRORS["TypeSafeAuthenticationError"]()
        grader = make_grader(
            judge_model_config=dm(),
            escalation=EscalationConfig(
                judges=[JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "llm")], threshold=0.5
            ),
        )
        llm = StubLLM()
        grader._clients["llm"] = llm

        with pytest.raises(ValueError, match="Judge 'default' is misconfigured"):
            await RUBRIC.grade(SUBMISSION, grader=grader)
        assert llm.calls == 0


class TestTheEscalationJudges:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    async def test_a_call_time_configuration_error_raises(self, fake_sdk, make_grader, llm_calls):
        # The request fails transiently, so every criterion escalates.
        fake_sdk.error = ROUTED_ERRORS["TypeSafeAPIConnectionError"]()
        error = litellm.AuthenticationError(
            message="Incorrect API key provided", llm_provider="openai", model="m"
        )
        grader = make_grader(
            judge_model_config=dm(),
            escalation=EscalationConfig(
                judges=[JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "llm")], threshold=0.5
            ),
            llm_calls=llm_calls,
        )
        grader._clients["llm"] = StubLLM(error)

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        assert "Judge 'llm' is misconfigured (AuthenticationError)" in str(excinfo.value)
        assert excinfo.value.__cause__ is error

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    async def test_a_refused_parameter_raises_on_the_first_call(
        self, fake_sdk, make_grader, monkeypatch, refuse_network, llm_calls
    ):
        """LiteLLM refuses ``reasoning_effort`` for this model on the client, before any
        request, so the escalation judge's first call raises. The real ``LLMClient`` runs,
        with a dummy key and the network refused, so a mistake cannot send a request."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-never-valid-40")
        refuse_network("a refused parameter reached the network")
        # The request fails transiently, so every criterion escalates.
        fake_sdk.error = ROUTED_ERRORS["TypeSafeAPIConnectionError"]()
        grader = make_grader(
            judge_model_config=dm(),
            escalation=EscalationConfig(
                judges=[JudgeSpec(LLMConfig(model="openai/gpt-6-luna", thinking="low"), "luna")],
                threshold=0.5,
            ),
            llm_calls=llm_calls,
        )

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        assert "Judge 'luna' is misconfigured (UnsupportedParamsError)" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, litellm.UnsupportedParamsError)
