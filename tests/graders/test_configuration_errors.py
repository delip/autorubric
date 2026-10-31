"""A misconfigured LLM judge fails loudly, never as an abstention (#40).

For every LLM judge (a single judge, a panel member, a cascade's escalation judge; the
cascade is covered in ``tests/decision/test_configuration_errors.py``), a configuration
error at call time (``UnsupportedParamsError``, a 401, 403 or 404, the OpenAI SDK's refusal
to build its client) is raised by the grade as a ``ValueError`` naming the judge, chained
from the provider's exception, instead of being routed as an ``infrastructure`` abstention.
Item-specific 400s keep abstaining. A parameter LiteLLM refuses fails the judge's first
call: LiteLLM raises ``UnsupportedParamsError`` on the client, before any request.

Nothing reaches the network: judge clients are local stubs, real LiteLLM calls are answered
by a provider on the loopback interface or fail before any request (with a dummy key and
the network refused).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast

import httpx
import litellm
import openai
import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
    classify_grading_error,
    evaluate,
    fill_ground_truth,
)
from autorubric.eval import EvalConfig, EvalRunner
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import (
    GenerateResult,
    LLMClient,
    _is_configuration_error,
)
from autorubric.types import (
    CriterionJudgment,
    MultiChoiceJudgment,
    RubricCriterionJudgment,
    RubricJudgment,
)

RUBRIC = Rubric(
    [
        Criterion(name="cites", weight=1.0, requirement="Cites a source"),
        Criterion(name="wrong", weight=-1.0, requirement="States a factual error"),
        Criterion(
            name="clarity",
            weight=1.0,
            requirement="How clear is it?",
            scale_type="ordinal",
            options=[
                CriterionOption(label="Unclear", value=0.0),
                CriterionOption(label="Clear", value=1.0),
            ],
        ),
    ]
)
SUBMISSION = "See Smith (2020)."

# LiteLLM 1.95.0 knows OpenAI reasoning models by the gpt-5 and o-series names, so it
# refuses reasoning_effort (sent by thinking="low") for this model before any request.
REFUSED = LLMConfig(model="openai/gpt-6-luna", thinking="low")
WORKING = LLMConfig(model="openai/gpt-4.1-mini")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://llm.test/v1"))


CONFIGURATION_ERRORS = {
    "UnsupportedParamsError": lambda: litellm.UnsupportedParamsError(
        message="openai does not support parameters: ['reasoning_effort']"
    ),
    "AuthenticationError": lambda: litellm.AuthenticationError(
        message="Incorrect API key provided", llm_provider="openai", model="m"
    ),
    "PermissionDeniedError": lambda: litellm.PermissionDeniedError(
        message="forbidden", llm_provider="openai", model="m", response=_response(403)
    ),
    "NotFoundError": lambda: litellm.NotFoundError(
        message="The model does not exist", model="m", llm_provider="openai"
    ),
}

# 400s that can be specific to one item, a plain 400, and transient failures: all still
# abstain as infrastructure failures.
ABSTAINING_ERRORS = {
    "ContextWindowExceededError": lambda: litellm.ContextWindowExceededError(
        message="too long", model="m", llm_provider="openai"
    ),
    "ContentPolicyViolationError": lambda: litellm.ContentPolicyViolationError(
        message="flagged", model="m", llm_provider="openai"
    ),
    "BadRequestError": lambda: litellm.BadRequestError(
        message="bad request", model="m", llm_provider="openai"
    ),
    "RateLimitError": lambda: litellm.RateLimitError(
        message="slow down", llm_provider="openai", model="m"
    ),
    "Timeout": lambda: litellm.Timeout(message="timed out", model="m", llm_provider="openai"),
}


class StubJudge:
    """A local stand-in for an ``LLMClient``: raises ``error``, or answers every criterion.

    ``delay`` holds every answer back, and ``cancelled`` counts the calls cancelled while
    waiting.
    """

    def __init__(self, error: BaseException | None = None, delay: float = 0.0) -> None:
        self.error = error
        self.delay = delay
        self.calls = 0
        self.cancelled = 0

    async def generate(self, **kwargs: Any) -> GenerateResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        response_format = kwargs["response_format"]
        parsed: Any
        if response_format is RubricJudgment:
            parsed = RubricJudgment(
                judgments=[
                    RubricCriterionJudgment(
                        criterion_id="c0", criterion_status=CriterionVerdict.MET, explanation="a"
                    ),
                    RubricCriterionJudgment(
                        criterion_id="c1", criterion_status=CriterionVerdict.UNMET, explanation="b"
                    ),
                    RubricCriterionJudgment(criterion_id="c2", selected_option=1, explanation="c"),
                ]
            )
        elif response_format is MultiChoiceJudgment:
            parsed = MultiChoiceJudgment(selected_option=1, explanation="ok")
        else:
            verdict = (
                CriterionVerdict.MET
                if "Cites a source" in kwargs["user_prompt"]
                else CriterionVerdict.UNMET
            )
            parsed = CriterionJudgment(criterion_status=verdict, explanation="ok")
        return GenerateResult(
            content="{}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=0.0,
            parsed=parsed,
        )


def _grader(**kwargs: Any) -> CriterionGrader:
    return CriterionGrader(shuffle_options=False, **kwargs)


def _install(grader: CriterionGrader, **stubs: StubJudge) -> None:
    """Make each judge named in ``stubs`` call its stub instead of LiteLLM."""
    for judge_id, stub in stubs.items():
        grader._clients[judge_id] = cast(LLMClient, stub)


def _assert_names_the_judge(excinfo: pytest.ExceptionInfo[ValueError], judge_id: str, exc: Any):
    assert f"Judge {judge_id!r} is misconfigured ({type(exc).__name__})" in str(excinfo.value)
    assert str(exc) in str(excinfo.value)
    assert excinfo.value.__cause__ is exc


# =============================================================================
# The predicate and the unchanged classification
# =============================================================================


class TestWhatIsAConfigurationError:
    @pytest.mark.parametrize("name", list(CONFIGURATION_ERRORS))
    def test_configuration_errors(self, name):
        assert _is_configuration_error(CONFIGURATION_ERRORS[name]())

    @pytest.mark.parametrize("name", list(ABSTAINING_ERRORS))
    def test_item_specific_and_transient_errors_are_not(self, name):
        assert not _is_configuration_error(ABSTAINING_ERRORS[name]())

    def test_unsupported_params_is_the_one_bad_request_that_is(self):
        """UnsupportedParamsError subclasses BadRequestError, like the item-specific 400s."""
        assert issubclass(litellm.UnsupportedParamsError, litellm.BadRequestError)
        assert issubclass(litellm.ContextWindowExceededError, litellm.BadRequestError)

    @pytest.mark.parametrize("name", list(CONFIGURATION_ERRORS))
    def test_classify_grading_error_is_unchanged(self, name):
        """The public taxonomy keeps its values: the grader raises these before routing."""
        assert classify_grading_error(CONFIGURATION_ERRORS[name]()) == "infrastructure"


# =============================================================================
# Call time: the grade raises
# =============================================================================


class TestTheGradeRaises:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(CONFIGURATION_ERRORS))
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    async def test_a_single_judge(self, name, llm_calls):
        error = CONFIGURATION_ERRORS[name]()
        grader = _grader(judge_model_config=WORKING, llm_calls=llm_calls)
        _install(grader, default=StubJudge(error))

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        _assert_names_the_judge(excinfo, "default", error)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(CONFIGURATION_ERRORS))
    async def test_a_panel_member(self, name):
        """A panel never scores an item from its other judges while one is misconfigured."""
        error = CONFIGURATION_ERRORS[name]()
        grader = _grader(judges=[JudgeSpec(WORKING, "broken"), JudgeSpec(WORKING, "ok")])
        _install(grader, broken=StubJudge(error))
        _install(grader, ok=StubJudge())

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        _assert_names_the_judge(excinfo, "broken", error)

    @pytest.mark.asyncio
    async def test_the_other_calls_of_the_item_are_cancelled(self):
        """Once the grade has failed, the working judge's calls still waiting are cancelled
        rather than left to run, and bill, for a grade that already raised."""
        working = StubJudge(delay=10.0)
        grader = _grader(judges=[JudgeSpec(WORKING, "broken"), JudgeSpec(WORKING, "ok")])
        _install(grader, broken=StubJudge(CONFIGURATION_ERRORS["AuthenticationError"]()))
        _install(grader, ok=working)

        with pytest.raises(ValueError):
            await asyncio.wait_for(RUBRIC.grade(SUBMISSION, grader=grader), timeout=5.0)
        await asyncio.sleep(0)

        # Only the warming call is out when the failure lands: the rest were never sent.
        assert working.calls >= 1
        assert working.cancelled == working.calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(ABSTAINING_ERRORS))
    async def test_other_failures_still_abstain(self, name):
        grader = _grader(judge_model_config=WORKING)
        _install(grader, default=StubJudge(ABSTAINING_ERRORS[name]()))

        report = await RUBRIC.grade(SUBMISSION, grader=grader)

        assert report.score is None
        assert report.error is not None and "infrastructure: " in report.error
        assert report.report is not None
        cites, wrong, clarity = report.report
        assert cites.final_verdict == wrong.final_verdict == CriterionVerdict.CANNOT_ASSESS
        assert clarity.final_multi_choice_verdict is not None
        assert clarity.final_multi_choice_verdict.na is True

    @pytest.mark.asyncio
    async def test_a_panel_member_with_an_item_specific_error_still_abstains(self):
        too_long = ABSTAINING_ERRORS["ContextWindowExceededError"]()
        grader = _grader(judges=[JudgeSpec(WORKING, "short"), JudgeSpec(WORKING, "ok")])
        _install(grader, short=StubJudge(too_long))
        _install(grader, ok=StubJudge())

        report = await RUBRIC.grade(SUBMISSION, grader=grader)

        assert report.error is None
        assert report.judge_scores["short"] is None and report.judge_scores["ok"] is not None


class TestRunsStopOnTheFirstItem:
    @staticmethod
    def _dataset() -> RubricDataset:
        data = RubricDataset(prompt="Q", rubric=RUBRIC)
        for i in range(3):
            data.add_item(submission=f"answer {i}", description=f"item {i}")
        return data

    @pytest.mark.asyncio
    async def test_evaluate_fails_the_item_with_the_judge_named(self, tmp_path):
        grader = _grader(judge_model_config=WORKING)
        _install(grader, default=StubJudge(CONFIGURATION_ERRORS["NotFoundError"]()))

        result = await evaluate(
            self._dataset(), grader, show_progress=False, experiments_dir=tmp_path
        )

        assert result.failed_items == 3
        assert all(
            r.error is not None and r.error.startswith("Judge 'default' is misconfigured")
            for r in result.item_results
        )

    @pytest.mark.asyncio
    async def test_fail_fast_stops_at_the_first_item(self, tmp_path):
        grader = _grader(judge_model_config=WORKING)
        _install(grader, default=StubJudge(CONFIGURATION_ERRORS["AuthenticationError"]()))

        with pytest.raises(RuntimeError, match="Judge 'default' is misconfigured"):
            await evaluate(
                self._dataset(),
                grader,
                fail_fast=True,
                show_progress=False,
                experiments_dir=tmp_path,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("show_progress", [False, True])
    async def test_fill_ground_truth_raises_instead_of_leaving_every_item_out(self, show_progress):
        error = CONFIGURATION_ERRORS["AuthenticationError"]()
        grader = _grader(judge_model_config=WORKING)
        _install(grader, default=StubJudge(error))

        with pytest.raises(ValueError) as excinfo:
            await fill_ground_truth(self._dataset(), grader, show_progress=show_progress)

        _assert_names_the_judge(excinfo, "default", error)

    @pytest.mark.asyncio
    async def test_fill_ground_truth_still_leaves_out_an_item_that_failed(self):
        grader = _grader(judge_model_config=WORKING)
        _install(grader, default=StubJudge(ABSTAINING_ERRORS["Timeout"]()))

        labeled = await fill_ground_truth(self._dataset(), grader, show_progress=False)

        assert len(labeled) == 0


# =============================================================================
# A parameter LiteLLM refuses: the judge's first call raises
# =============================================================================


@pytest.fixture
def no_request(monkeypatch):
    """A dummy key and a refused network, so that a mistake cannot send a paid request.
    LiteLLM refuses a parameter on the client, before it needs either."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-never-valid-40")

    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a refused parameter reached the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)


class TestARefusedParameter:
    """LiteLLM decides on the client whether a model takes a call's parameters, and raises
    ``UnsupportedParamsError`` before sending any request, so the judge's first call fails
    with it and the grade raises. The real ``LLMClient`` and ``litellm.acompletion`` run."""

    def test_the_grader_still_constructs(self):
        CriterionGrader(judge_model_config=REFUSED)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    async def test_a_single_judge(self, no_request, llm_calls):
        grader = _grader(judge_model_config=REFUSED, llm_calls=llm_calls)

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        message = str(excinfo.value)
        assert "Judge 'default' is misconfigured (UnsupportedParamsError)" in message
        assert "'reasoning_effort'" in message
        assert isinstance(excinfo.value.__cause__, litellm.UnsupportedParamsError)

    @pytest.mark.asyncio
    async def test_the_message_names_the_workarounds(self, no_request):
        """LiteLLM's message, which the ``ValueError`` carries, says how to send the
        parameter anyway (``allowed_openai_params``) or drop it (``drop_params``)."""
        grader = _grader(judge_model_config=REFUSED)

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        assert "allowed_openai_params=['reasoning_effort']" in str(excinfo.value)
        assert "drop_params" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_panel_member(self, no_request):
        grader = _grader(judges=[JudgeSpec(WORKING, "mini"), JudgeSpec(REFUSED, "luna")])
        _install(grader, mini=StubJudge())

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        assert "Judge 'luna' is misconfigured (UnsupportedParamsError)" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, litellm.UnsupportedParamsError)


# =============================================================================
# A missing OpenAI key: the shape LiteLLM really raises
# =============================================================================


class _Provider(BaseHTTPRequestHandler):
    """Answers every request with the status and JSON body its server is set to."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        server = cast(Any, self.server)
        server.requests += 1
        body = json.dumps(server.body).encode()
        self.send_response(server.status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def provider():
    """A provider on the loopback interface, so LiteLLM's real HTTP stack and exception
    mapping turn its answer into the exception a call raises. Nothing leaves the machine."""
    server = HTTPServer(("127.0.0.1", 0), _Provider)
    server_any = cast(Any, server)
    server_any.status, server_any.body, server_any.requests = 403, {}, 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server_any
    finally:
        server.shutdown()
        server.server_close()


# Real-shaped answers. The OpenAI-compatible body serves OpenAI, Groq and DeepSeek.
OPENAI_403 = {
    "error": {
        "message": "Country, region, or territory not supported",
        "type": "request_forbidden",
        "param": None,
        "code": "unsupported_country_region_territory",
    }
}
ANTHROPIC_403 = {
    "type": "error",
    "error": {
        "type": "permission_error",
        "message": "Your API key does not have permission to use the specified resource.",
    },
}
GEMINI_403 = {
    "error": {
        "code": 403,
        "message": "Generative Language API has not been used in project 1 before",
        "status": "PERMISSION_DENIED",
    }
}
OPENROUTER_MODERATION_403 = {
    "error": {
        "code": 403,
        "message": "meta-llama/llama-guard-3-8b requires moderation: input flagged",
        "metadata": {"reasons": ["S1"], "flagged_input": "..."},
    }
}
OPENAI_400 = {"error": {"message": "Invalid value for 'n'", "type": "invalid_request_error"}}
OPENAI_401 = {
    "error": {
        "message": "Incorrect API key provided: test-key.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
}
OPENAI_404 = {
    "error": {
        "message": "The model `gpt-4.1-mini` does not exist or you do not have access to it.",
        "type": "invalid_request_error",
        "param": None,
        "code": "model_not_found",
    }
}
OPENAI_429 = {"error": {"message": "Rate limit reached", "type": "requests", "code": None}}


def _base(provider: Any, model: str) -> str:
    base = f"http://127.0.0.1:{provider.server_port}"
    return base if model.startswith(("anthropic/", "gemini/")) else f"{base}/v1"


async def _litellm_error(provider: Any, model: str) -> BaseException:
    with pytest.raises(Exception) as excinfo:
        await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            api_base=_base(provider, model),
            num_retries=0,
            max_retries=0,
        )
    return excinfo.value


class TestAForbiddenAnswer:
    """A 403 says the key or project may not use the model, and fails every call the judge
    makes. LiteLLM 1.95.0 reports few of them as ``PermissionDeniedError``: an
    OpenAI-compatible provider's comes back as a plain ``APIError``, Anthropic's as an
    ``APIConnectionError`` (which ``LLMClient`` would retry), Gemini's as a
    ``BadRequestError``. Each keeps the provider's HTTP response in its chain, which is what
    says it was a 403. OpenRouter's 403 flags an input by moderation, specific to the item,
    so it keeps abstaining."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model", "body", "litellm_type"),
        [
            ("openai/gpt-4.1-mini", OPENAI_403, litellm.APIError),
            ("groq/llama-3.1-8b-instant", OPENAI_403, litellm.APIError),
            ("deepseek/deepseek-chat", OPENAI_403, litellm.APIError),
            ("anthropic/claude-sonnet-4-5", ANTHROPIC_403, litellm.APIConnectionError),
            ("gemini/gemini-2.5-flash", GEMINI_403, litellm.BadRequestError),
        ],
    )
    async def test_is_a_configuration_error(self, provider, model, body, litellm_type):
        provider.status, provider.body = 403, body

        error = await _litellm_error(provider, model)

        assert type(error) is litellm_type
        assert not isinstance(error, openai.PermissionDeniedError)
        assert _is_configuration_error(error)

    @pytest.mark.asyncio
    async def test_openrouter_moderation_is_not(self, provider):
        provider.status, provider.body = 403, OPENROUTER_MODERATION_403

        error = await _litellm_error(provider, "openrouter/openai/gpt-4.1-mini")

        assert isinstance(error, litellm.APIError)
        assert not _is_configuration_error(error)
        assert classify_grading_error(error) == "infrastructure"

    def test_openrouter_reached_through_another_prefix_is_not(self):
        """``openai/`` with OpenRouter's base URL: the response's host says it is OpenRouter."""
        answer = httpx.Response(
            403, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        )
        try:
            raise httpx.HTTPStatusError("403", request=answer.request, response=answer)
        except httpx.HTTPStatusError:
            try:
                raise litellm.APIError(
                    status_code=403, message="flagged", llm_provider="openai", model="m"
                )
            except litellm.APIError as error:
                assert not _is_configuration_error(error)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "body", "configuration"),
        [
            (401, OPENAI_401, True),
            (404, OPENAI_404, True),
            (400, OPENAI_400, False),
            (429, OPENAI_429, False),
        ],
    )
    async def test_other_statuses(self, provider, status, body, configuration):
        provider.status, provider.body = status, body

        error = await _litellm_error(provider, "openai/gpt-4.1-mini")

        assert _is_configuration_error(error) is configuration

    @pytest.mark.asyncio
    async def test_is_not_retried(self, provider):
        """Anthropic's 403 arrives as an ``APIConnectionError``, the kind ``LLMClient``
        retries; retrying a configuration error only delays the failure."""
        provider.status, provider.body = 403, ANTHROPIC_403
        model = "anthropic/claude-sonnet-4-5"
        client = LLMClient(
            LLMConfig(
                model=model,
                api_key="test-key",
                api_base=_base(provider, model),
                max_retries=3,
                retry_min_wait=0.0,
                retry_max_wait=0.0,
                cache_enabled=False,
            )
        )

        with pytest.raises(litellm.APIConnectionError):
            await client.generate(system_prompt="s", user_prompt="u")

        assert provider.requests == 1

    @pytest.mark.asyncio
    async def test_a_transient_connection_error_is_still_retried(self, provider):
        provider.status, provider.body = 503, {"type": "error", "error": {"type": "x"}}
        model = "anthropic/claude-sonnet-4-5"
        client = LLMClient(
            LLMConfig(
                model=model,
                api_key="test-key",
                api_base=_base(provider, model),
                max_retries=2,
                retry_min_wait=0.0,
                retry_max_wait=0.0,
                cache_enabled=False,
            )
        )

        with pytest.raises(Exception) as excinfo:
            await client.generate(system_prompt="s", user_prompt="u")

        assert not _is_configuration_error(excinfo.value)
        assert provider.requests >= 2

    @pytest.mark.asyncio
    async def test_the_grade_raises(self, provider):
        provider.status, provider.body = 403, OPENAI_403
        model = "openai/gpt-4.1-mini"
        grader = _grader(
            judge_model_config=LLMConfig(
                model=model, api_key="test-key", api_base=_base(provider, model)
            )
        )

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        assert "Judge 'default' is misconfigured (APIError)" in str(excinfo.value)
        assert "Country, region, or territory not supported" in str(excinfo.value)


class TestAMissingOpenAIKey:
    """With no OpenAI key, the OpenAI SDK refuses to build its client, before any request,
    and LiteLLM wraps that as an ``InternalServerError`` (not ``AuthenticationError``),
    with the SDK's bare ``openai.OpenAIError`` in its ``__context__`` chain."""

    @pytest.fixture
    def no_openai_key(self, monkeypatch):
        for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "openai_key", None)

        def no_network(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a missing key reached the network")

        monkeypatch.setattr(socket.socket, "connect", no_network)
        monkeypatch.setattr(socket, "getaddrinfo", no_network)

    @pytest.mark.asyncio
    async def test_the_real_litellm_error_is_a_configuration_error(self, no_openai_key):
        with pytest.raises(Exception) as excinfo:
            await litellm.acompletion(
                model="openai/gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}]
            )

        assert isinstance(excinfo.value, litellm.InternalServerError)
        assert _is_configuration_error(excinfo.value)

    @pytest.mark.asyncio
    async def test_the_grade_raises(self, no_openai_key):
        grader = _grader(judge_model_config=WORKING)

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        assert "Judge 'default' is misconfigured" in str(excinfo.value)
        assert "Missing credentials" in str(excinfo.value)

    def test_a_bare_sdk_error_raised_elsewhere_is_not(self):
        """The SDK also raises a bare ``OpenAIError`` after a request, e.g. when a workload
        identity's token exchange fails with a 503: that one can be transient."""
        try:
            raise openai.OpenAIError("Token exchange failed with status 503")
        except openai.OpenAIError as sdk_error:
            try:
                raise litellm.InternalServerError(
                    message=str(sdk_error), llm_provider="openai", model="m"
                )
            except litellm.InternalServerError as wrapped:
                assert not _is_configuration_error(wrapped)


class TestAMissingAzureKey:
    """With no Azure key, the OpenAI SDK's ``AsyncAzureOpenAI`` (in ``openai.lib.azure``)
    refuses to build its client, before any request, and LiteLLM wraps that as an
    ``APIError``, with the SDK's bare ``openai.OpenAIError`` in its chain."""

    @pytest.fixture
    def no_azure_key(self, monkeypatch):
        for name in (
            "AZURE_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_AD_TOKEN",
            "AZURE_AD_TOKEN",
            "AZURE_CLIENT_ID",
            "AZURE_CLIENT_SECRET",
            "AZURE_TENANT_ID",
            "AZURE_USERNAME",
            "AZURE_PASSWORD",
            "AZURE_SCOPE",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AZURE_API_BASE", "https://example-resource.openai.azure.com")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-02-01")
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "azure_key", None)

        def no_network(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a missing key reached the network")

        monkeypatch.setattr(socket.socket, "connect", no_network)
        monkeypatch.setattr(socket, "getaddrinfo", no_network)

    @pytest.mark.asyncio
    async def test_the_real_litellm_error_is_a_configuration_error(self, no_azure_key):
        with pytest.raises(Exception) as excinfo:
            await litellm.acompletion(
                model="azure/my-deployment", messages=[{"role": "user", "content": "hi"}]
            )

        assert isinstance(excinfo.value, litellm.APIError)
        assert "Missing credentials" in str(excinfo.value)
        assert _is_configuration_error(excinfo.value)

    @pytest.mark.asyncio
    async def test_the_grade_raises(self, no_azure_key):
        grader = _grader(judge_model_config=LLMConfig(model="azure/my-deployment"))

        with pytest.raises(ValueError) as excinfo:
            await RUBRIC.grade(SUBMISSION, grader=grader)

        assert "Judge 'default' is misconfigured" in str(excinfo.value)
        assert "Missing credentials" in str(excinfo.value)


# =============================================================================
# A fresh start whose manifest cannot be built
# =============================================================================


class TestAFreshStartWhoseManifestCannotBeBuilt:
    """``EvalRunner._start_fresh_checkpoint`` builds the manifest before it removes the
    previous run's ``items.jsonl``, so a grader config that cannot be serialized raises with
    the previous checkpoint still whole."""

    def test_a_fresh_start_writes_the_manifest_before_it_replaces_the_items(
        self, tmp_path, monkeypatch
    ):
        """A manifest that cannot be built leaves the previous run's items in place, never
        a stale manifest without its items."""
        runner = EvalRunner(
            TestRunsStopOnTheFirstItem._dataset(),
            _grader(judge_model_config=WORKING),
            EvalConfig(experiment_name="run", experiments_dir=tmp_path, show_progress=False),
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "items.jsonl").write_text("previous run\n", encoding="utf-8")
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

        def unserializable(grader: Any) -> Any:
            raise ValueError("cannot serialize")

        monkeypatch.setattr("autorubric.eval._serialize_grader_config", unserializable)
        with pytest.raises(ValueError, match="cannot serialize"):
            runner._start_fresh_checkpoint(datetime.now())

        assert (run_dir / "items.jsonl").read_text(encoding="utf-8") == "previous run\n"
        assert (run_dir / "manifest.json").read_text(encoding="utf-8") == "{}"
