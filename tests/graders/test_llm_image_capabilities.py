"""Which LLM judges an item holding images may reach.

An image never reaches an LLM judge not known to accept images. A judge accepts them when
its ``LLMConfig.images`` says so, or, with ``images`` unset, when LiteLLM's model map lists
its model as accepting images (``litellm.supports_vision``), asked once per grader per
model, the first time the grader sees content holding images. Every judge that could be
sent the images (a single judge, every panel member, a cascade's escalation judges, and
every decision model) is checked before any judge is called, so a refusal sends no request
at all; training images, which go out with every call, are checked when the grader is
built. Text-only grading never asks.

Nothing here reaches the network: ``litellm.acompletion`` is patched, and decision-model
requests go to the recording fake SDK client of the decision-model tests.
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import litellm
import PIL.Image as pil_image
import pytest

# The decision-model tests' guards and fake SDK, registered for this module by importing
# them: no ``httpx2`` transport may send, the ``TYPESAFE_*`` variables are cleared, and the
# rate-limit pool starts fresh for each test.
from decision.conftest import (  # noqa: F401
    FakeSDK,
    _fresh_rate_limit_pool,
    _isolate_typesafe_env,
    _no_network,
    fake_sdk,
)
from litellm import ModelResponse

from autorubric import (
    Criterion,
    CriterionVerdict,
    DataItem,
    DecisionModelConfig,
    EscalationConfig,
    FewShotConfig,
    Image,
    LLMConfig,
    Rubric,
    RubricDataset,
)
from autorubric.decision import _CONFIGURATION_ERROR_NOTE
from autorubric.eval import _serialize_grader_config
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.graders.criterion_grader import _is_judge_configuration_error
from autorubric.llm import _litellm_supports_vision
from autorubric.utils import fill_ground_truth

SEED = 3
API_KEY = "ts-test-key-never-persisted-77c2"
CAPTION = "Figure 1: accuracy by model size."
LEGIBLE = Criterion(name="legible", weight=1.0, requirement="The figure is legible")

# Models the patched LiteLLM lookup answers for (``vision``): a model it lists as
# accepting images, and one it does not.
SEEING = "acme/seeing-1"
BLIND = "acme/blind-1"


def png(color: str) -> bytes:
    buffer = io.BytesIO()
    pil_image.new("RGB", (2, 2), color).save(buffer, "PNG")
    return buffer.getvalue()


FIGURE = Image(png("red"))
DOCUMENT: list[str | Image] = [FIGURE, CAPTION]


def unconfirmed(judge_id: str, model: str) -> str:
    return (
        f"Judge {judge_id!r} uses LLM {model!r}, which LiteLLM does not list as accepting "
        "images; if it does, pass LLMConfig(images=True)"
    )


def declared_false(judge_id: str, model: str) -> str:
    return (
        f"Judge {judge_id!r} uses LLM {model!r}, which does not accept images; "
        "its LLMConfig(images=False) says so"
    )


def met_reply(**_: Any) -> ModelResponse:
    return ModelResponse(
        model="m",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"criterion_status": "MET", "explanation": "Fine."}),
                },
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def image_parts(call: Any) -> list[dict[str, Any]]:
    (message,) = [m for m in call.kwargs["messages"] if m["role"] == "user"]
    content = message["content"]
    if isinstance(content, str):
        return []
    return [part for part in content if part["type"] == "image_url"]


class Lookup:
    """``litellm.supports_vision``'s stand-in: answers from ``answers`` (anything else is
    ``False``, as LiteLLM answers for a model it does not map) and records each model."""

    def __init__(self) -> None:
        self.answers: dict[str, Any] = {SEEING: True, BLIND: False}
        self.models: list[str] = []

    def __call__(self, model: str, custom_llm_provider: str | None = None) -> Any:
        self.models.append(model)
        answer = self.answers.get(model, False)
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def vision() -> Iterator[Lookup]:
    lookup = Lookup()
    with patch("litellm.supports_vision", side_effect=lookup):
        yield lookup


@pytest.fixture
def provider() -> Iterator[AsyncMock]:
    reply = AsyncMock(side_effect=met_reply)
    with patch("litellm.acompletion", reply):
        yield reply


def grader_for(*configs: LLMConfig | DecisionModelConfig, **kwargs: Any) -> CriterionGrader:
    """A grader whose judges are ``configs``: the single judge for one, else a panel whose
    judges are named ``j0``, ``j1``, ..."""
    if len(configs) == 1:
        return CriterionGrader(judge_model_config=configs[0], seed=SEED, **kwargs)
    return CriterionGrader(
        judges=[JudgeSpec(config, judge_id=f"j{i}") for i, config in enumerate(configs)],
        seed=SEED,
        **kwargs,
    )


def jev() -> DecisionModelConfig:
    return DecisionModelConfig(model="jev-latest", api_key=API_KEY)


def training_set(*submissions: Any) -> RubricDataset:
    dataset = RubricDataset(prompt="Rate the figure.", rubric=Rubric([LEGIBLE]))
    labels = [CriterionVerdict.MET, CriterionVerdict.UNMET]
    for i, submission in enumerate(submissions):
        dataset.items.append(
            DataItem(submission=submission, description=str(i), ground_truth=[labels[i % 2]])
        )
    return dataset


# =============================================================================
# LLMConfig.images
# =============================================================================


class TestTheConfigField:
    def test_it_is_unset_by_default(self):
        assert LLMConfig(model=BLIND).images is None

    @pytest.mark.parametrize("value", [True, False, None])
    def test_a_bool_or_none_is_kept(self, value):
        assert LLMConfig(model=BLIND, images=value).images is value

    @pytest.mark.parametrize("value", [1, 0, "yes", "true", 1.0, [True]])
    def test_anything_else_is_refused(self, value):
        with pytest.raises(ValueError, match=re.escape("LLMConfig.images must be None, True or")):
            LLMConfig(model=BLIND, images=value)

    def test_a_numpy_bool_is_refused(self):
        np = pytest.importorskip("numpy")
        with pytest.raises(ValueError, match="LLMConfig.images"):
            LLMConfig(model=BLIND, images=np.bool_(True))

    @pytest.mark.parametrize("value", [True, False])
    def test_it_round_trips_through_yaml(self, tmp_path, value):
        path = tmp_path / "llm.yaml"
        LLMConfig(model=BLIND, images=value).to_yaml(path)
        assert LLMConfig.from_yaml(path).images is value

    def test_an_unset_value_is_not_written_to_yaml(self, tmp_path):
        path = tmp_path / "llm.yaml"
        LLMConfig(model=BLIND).to_yaml(path)
        assert "images" not in path.read_text()


# =============================================================================
# The LiteLLM lookup
# =============================================================================


class TestTheLiteLLMLookup:
    def test_the_real_map_lists_a_vision_model(self):
        assert _litellm_supports_vision("openai/gpt-4.1-mini") is True

    @pytest.mark.parametrize(
        "model", ["openai/gpt-3.5-turbo", "acme/unmapped-1", "no-provider-at-all"]
    )
    def test_the_real_map_confirms_nothing_else(self, model):
        assert _litellm_supports_vision(model) is False

    @pytest.mark.parametrize(
        "answer", [False, None, "True", 1, RuntimeError("map unreadable")], ids=repr
    )
    def test_any_answer_but_true_is_not_a_confirmation(self, vision, answer):
        vision.answers["acme/odd-1"] = answer
        assert _litellm_supports_vision("acme/odd-1") is False

    @pytest.mark.parametrize("model", ["acme/unmapped-1", "no-provider-at-all"])
    def test_it_prints_and_logs_nothing(self, capsys, caplog, model):
        """LiteLLM prints a provider-list banner and logs a debug record for a model it
        cannot place; the lookup lets neither through, and leaves LiteLLM's settings as
        they were."""
        before = litellm.suppress_debug_info
        with caplog.at_level(logging.DEBUG):
            _litellm_supports_vision(model)
        assert capsys.readouterr() == ("", "")
        assert [r for r in caplog.records if r.name.startswith("LiteLLM")] == []
        assert litellm.suppress_debug_info is before


# =============================================================================
# Grading an item holding images
# =============================================================================


class TestWhoIsSentImages:
    @pytest.mark.asyncio
    async def test_a_model_litellm_confirms_is_sent_them(self, vision, provider):
        report = await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader_for(LLMConfig(SEEING)))
        assert report.error is None
        (call,) = provider.call_args_list
        assert len(image_parts(call)) == 1
        assert vision.models == [SEEING]

    @pytest.mark.asyncio
    async def test_a_model_litellm_does_not_confirm_is_refused(self, vision, provider):
        with pytest.raises(ValueError) as excinfo:
            await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader_for(LLMConfig(BLIND)))
        assert str(excinfo.value) == unconfirmed("default", BLIND)
        assert excinfo.value.__notes__ == [_CONFIGURATION_ERROR_NOTE]
        assert _is_judge_configuration_error(excinfo.value)
        provider.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("where", ["query", "reference_submission"])
    async def test_images_in_the_query_or_reference_are_checked_too(self, vision, provider, where):
        with pytest.raises(ValueError, match=re.escape(unconfirmed("default", BLIND))):
            await Rubric([LEGIBLE]).grade(
                "Plain text.", grader=grader_for(LLMConfig(BLIND)), **{where: DOCUMENT}
            )
        provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_declaration_that_it_accepts_images_wins(self, vision, provider):
        grader = grader_for(LLMConfig(BLIND, images=True))
        report = await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader)
        assert report.error is None
        (call,) = provider.call_args_list
        assert len(image_parts(call)) == 1
        assert vision.models == []  # a declaration is not looked up

    @pytest.mark.asyncio
    async def test_a_declaration_that_it_does_not_wins(self, vision, provider):
        with pytest.raises(ValueError) as excinfo:
            await Rubric([LEGIBLE]).grade(
                DOCUMENT, grader=grader_for(LLMConfig(SEEING, images=False))
            )
        assert str(excinfo.value) == declared_false("default", SEEING)
        assert excinfo.value.__notes__ == [_CONFIGURATION_ERROR_NOTE]
        provider.assert_not_called()
        assert vision.models == []

    @pytest.mark.asyncio
    async def test_a_panel_member_refused_stops_every_member(self, vision, provider):
        grader = grader_for(LLMConfig(SEEING), LLMConfig(BLIND))
        with pytest.raises(ValueError, match=re.escape(unconfirmed("j1", BLIND))):
            await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader)
        provider.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_first", [True, False], ids=["llm-first", "dm-first"])
    async def test_a_mixed_panel_sends_no_request_by_either_judge(
        self,
        vision,
        provider,
        fake_sdk: FakeSDK,  # noqa: F811
        llm_first,
    ):
        configs = [LLMConfig(BLIND), jev()]
        if not llm_first:
            configs.reverse()
        grader = grader_for(*configs)
        llm_id = "j0" if llm_first else "j1"
        with pytest.raises(ValueError, match=re.escape(unconfirmed(llm_id, BLIND))):
            await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader)
        provider.assert_not_called()
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_a_cascade_refuses_before_its_decision_models_request(
        self,
        vision,
        provider,
        fake_sdk: FakeSDK,  # noqa: F811
    ):
        grader = CriterionGrader(
            judge_model_config=jev(),
            escalation=EscalationConfig(
                judges=[JudgeSpec(LLMConfig(BLIND), judge_id="fallback")], threshold=0.5
            ),
            seed=SEED,
        )
        with pytest.raises(ValueError, match=re.escape(unconfirmed("fallback", BLIND))):
            await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader)
        assert fake_sdk.calls == []
        provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_direct_judge_call_is_checked_too(self, vision, provider):
        grader = grader_for(LLMConfig(BLIND))
        with pytest.raises(ValueError, match=re.escape(unconfirmed("default", BLIND))):
            await grader.judge(DOCUMENT, [LEGIBLE])
        provider.assert_not_called()


# =============================================================================
# Asked once, and never for text
# =============================================================================


class TestWhenLiteLLMIsAsked:
    @pytest.mark.asyncio
    async def test_text_only_grading_never_asks(self, vision, provider):
        grader = grader_for(LLMConfig(BLIND), LLMConfig(SEEING))
        report = await Rubric([LEGIBLE]).grade(
            "Plain text.", grader=grader, query="Describe it.", reference_submission="A ref."
        )
        assert report.error is None
        assert provider.call_count == 2
        assert vision.models == []

    @pytest.mark.asyncio
    async def test_text_only_prompts_do_not_depend_on_the_field(self, vision, provider):
        """A text item's request is the same whatever ``images`` says."""
        for images in (None, True, False):
            await Rubric([LEGIBLE]).grade(
                "Plain text.", grader=grader_for(LLMConfig(BLIND, images=images))
            )
        first, *others = provider.call_args_list
        assert all(call.kwargs == first.kwargs for call in others)
        assert vision.models == []

    @pytest.mark.asyncio
    async def test_each_model_is_asked_once_across_items(self, vision, provider):
        grader = grader_for(LLMConfig(SEEING), LLMConfig("acme/seeing-2"))
        vision.answers["acme/seeing-2"] = True
        for _ in range(3):
            await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader)
            await Rubric([LEGIBLE]).grade("Plain text.", grader=grader)
        assert sorted(vision.models) == sorted([SEEING, "acme/seeing-2"])
        assert provider.call_count == 2 * 6

    @pytest.mark.asyncio
    async def test_a_refusal_is_not_asked_again(self, vision, provider):
        grader = grader_for(LLMConfig(BLIND))
        for _ in range(3):
            with pytest.raises(ValueError, match="LiteLLM does not list"):
                await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader)
        assert vision.models == [BLIND]

    @pytest.mark.asyncio
    async def test_each_grader_asks_for_itself(self, vision, provider):
        for _ in range(2):
            await Rubric([LEGIBLE]).grade(DOCUMENT, grader=grader_for(LLMConfig(SEEING)))
        assert vision.models == [SEEING, SEEING]


# =============================================================================
# Training images, sent with every call, are checked when the grader is built
# =============================================================================


class TestFewShotImages:
    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    def test_an_unconfirmed_judge_shown_training_images_fails_construction(self, vision, llm_calls):
        data = training_set([Image(png("green")), "A."], [Image(png("blue")), "B."])
        with pytest.raises(ValueError) as excinfo:
            grader_for(
                LLMConfig(BLIND),
                training_data=data,
                few_shot_config=FewShotConfig(n_examples=2),
                llm_calls=llm_calls,
            )
        assert str(excinfo.value) == unconfirmed("default", BLIND)
        assert excinfo.value.__notes__ == [_CONFIGURATION_ERROR_NOTE]

    def test_an_escalation_judge_shown_training_images_is_checked(self, vision, fake_sdk):  # noqa: F811
        data = training_set([Image(png("green")), "A."], [Image(png("blue")), "B."])
        with pytest.raises(ValueError, match=re.escape(unconfirmed("fallback", BLIND))):
            CriterionGrader(
                judge_model_config=jev(),
                escalation=EscalationConfig(
                    judges=[JudgeSpec(LLMConfig(BLIND), judge_id="fallback")], threshold=0.5
                ),
                training_data=data,
                few_shot_config=FewShotConfig(n_examples=2),
                seed=SEED,
            )

    def test_a_declared_or_confirmed_judge_is_built(self, vision):
        data = training_set([Image(png("green")), "A."], [Image(png("blue")), "B."])
        for config in (LLMConfig(BLIND, images=True), LLMConfig(SEEING)):
            grader_for(config, training_data=data, few_shot_config=FewShotConfig(n_examples=2))
        assert vision.models == [SEEING]

    @pytest.mark.asyncio
    async def test_text_training_data_never_asks(self, vision, provider):
        data = training_set("Text A.", "Text B.")
        grader = grader_for(
            LLMConfig(BLIND), training_data=data, few_shot_config=FewShotConfig(n_examples=2)
        )
        await Rubric([LEGIBLE]).grade("Plain text.", grader=grader)
        assert vision.models == []
        provider.assert_called_once()


# =============================================================================
# fill_ground_truth and the manifest
# =============================================================================


class TestFillGroundTruth:
    @pytest.mark.asyncio
    async def test_the_refusal_is_raised_not_recorded_per_item(self, vision, provider):
        dataset = RubricDataset(prompt="Describe the figure.", rubric=Rubric([LEGIBLE]))
        dataset.add_item(submission="Plain text.", description="text")
        # Datasets are annotated as text; grading takes any content.
        dataset.add_item(submission=cast(str, DOCUMENT), description="a figure")
        with pytest.raises(ValueError, match=re.escape(unconfirmed("default", BLIND))):
            await fill_ground_truth(dataset, grader_for(LLMConfig(BLIND)), show_progress=False)


class TestManifest:
    @pytest.mark.parametrize("images", [True, False])
    def test_a_declaration_is_recorded(self, images):
        (entry,) = _serialize_grader_config(grader_for(LLMConfig(BLIND, images=images)))["judges"]
        assert entry["images"] is images
        assert list(entry)[-1] == "images"

    def test_an_unset_field_is_not_recorded(self):
        (entry,) = _serialize_grader_config(grader_for(LLMConfig(BLIND)))["judges"]
        assert "images" not in entry
        assert list(entry) == [
            "judge_id",
            "judge_kind",
            "model",
            "temperature",
            "weight",
            "max_parallel_requests",
        ]

    def test_an_escalation_judges_declaration_is_recorded(self, fake_sdk):  # noqa: F811
        grader = CriterionGrader(
            judge_model_config=jev(),
            escalation=EscalationConfig(
                judges=[JudgeSpec(LLMConfig(BLIND, images=True), judge_id="fallback")],
                threshold=0.5,
            ),
            seed=SEED,
        )
        (entry,) = _serialize_grader_config(grader)["escalation"]["judges"]
        assert entry["images"] is True
