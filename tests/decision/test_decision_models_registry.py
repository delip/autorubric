"""The decision-model registry: which models accept images, and the images they accept.

autorubric ships ``data/decision_models.json``, keyed by the model id as sent to the API.
``decision_model_capabilities`` reads it (once, checked) and looks a model up, with or
without a ``typesafe/`` prefix. ``CriterionGrader.judge`` consults it only for content
holding images, before any judge is called: a model the registry does not list, or lists
as not accepting images, raises ``ValueError`` with no request sent by any judge, and text
never reads it. A model that accepts images has them fitted into its entry, and the fitted
images of one request must fit its ``max_request_bytes`` together.

Decision models answer through the recording fake SDK (``fake_sdk``, see ``conftest.py``);
LLM judges' completions are patched. No test reaches the network.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import PIL.Image as pil_image
import pytest
import typesafe_sdk
from litellm import ModelResponse
from typesafe_sdk import Noul

from autorubric import Criterion, DecisionModelConfig, Image, LLMConfig, Rubric, RubricDataset
from autorubric import decision as decision_module
from autorubric.decision import (
    _CONFIGURATION_ERROR_NOTE,
    DecisionModelClient,
    _api_model_id,
    _decision_model_registry,
    decision_model_capabilities,
)
from autorubric.eval import _serialize_grader_config
from autorubric.graders import CriterionGrader, EscalationConfig, JudgeSpec
from autorubric.graders import criterion_grader as grader_module
from autorubric.graders.criterion_grader import _is_judge_configuration_error
from autorubric.utils import fill_ground_truth

API_KEY = "ts-test-key-never-persisted-5a9d"
SHIPPED = decision_module._REGISTRY_RESOURCE
"""The shipped registry file (captured before any test points the registry elsewhere)."""
LEGIBLE = Criterion(name="legible", weight=1.0, requirement="The figure is legible")
CAPTION = "Figure 1: accuracy by model size."
LIMIT_KEYS = ("image_formats", "min_side_px", "max_side_px", "max_image_bytes", "max_request_bytes")
JEV_1_14 = {
    "provider": "typesafe",
    "images": True,
    "image_formats": ["png", "jpeg"],
    "min_side_px": 280,
    "max_side_px": 1568,
    "max_image_bytes": 16777216,
    "max_request_bytes": 33554432,
}


def png(color: str, size: tuple[int, int] = (400, 300)) -> bytes:
    buffer = io.BytesIO()
    pil_image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


FIGURE = Image(png("red"))
OTHER = Image(png("blue"))


def met_reply() -> ModelResponse:
    """An LLM judge's completion judging the criterion met."""
    content = json.dumps({"criterion_status": "MET", "explanation": "Legible."})
    return ModelResponse(
        model="gpt-4.1-mini",
        choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    )


def dm(model: str) -> DecisionModelConfig:
    return DecisionModelConfig(model=model, api_key=API_KEY)


@pytest.fixture
def make_grader() -> Iterator[Callable[..., CriterionGrader]]:
    """Build graders whose decision-model clients are closed at teardown."""
    graders: list[CriterionGrader] = []

    def build(*judges: JudgeSpec) -> CriterionGrader:
        grader = CriterionGrader(judges=list(judges))
        graders.append(grader)
        return grader

    yield build
    for grader in graders:
        for client in grader._decision_clients.values():
            client.close()


@pytest.fixture
def registry_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Path]]:
    """Point the registry at a file of the test's own, read afresh; restored at teardown."""

    def write(content: object) -> Path:
        path = tmp_path / "decision_models.json"
        text = content if isinstance(content, str) else json.dumps(content)
        path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(decision_module, "_REGISTRY_RESOURCE", path)
        _decision_model_registry.cache_clear()
        return path

    yield write
    _decision_model_registry.cache_clear()


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every registry lookup, through either module's name for the function."""
    seen: list[str] = []

    def spy(model: str) -> Any:
        seen.append(model)
        return decision_model_capabilities(model)

    monkeypatch.setattr(decision_module, "decision_model_capabilities", spy)
    monkeypatch.setattr(grader_module, "decision_model_capabilities", spy)
    return seen


# =============================================================================
# The shipped file
# =============================================================================


class TestShippedRegistry:
    def test_it_lists_jev_releases_exactly(self):
        assert json.loads(SHIPPED.read_text(encoding="utf-8")) == {
            "jev-1.13.0": {"provider": "typesafe", "images": False},
            "jev-1.14.0": JEV_1_14,
            "jev-latest": {
                **JEV_1_14,
                "notes": (
                    "Tracks the newest Jev release; re-check its capabilities on every Jev release."
                ),
            },
        }

    def test_it_loads_and_validates(self):
        registry = _decision_model_registry()
        assert set(registry) == {"jev-1.13.0", "jev-1.14.0", "jev-latest"}
        for entry in registry.values():
            assert isinstance(entry["provider"], str) and entry["provider"]
            assert isinstance(entry["images"], bool)
            assert isinstance(entry.get("notes", ""), str)
            if entry["images"]:
                assert all(key in entry for key in LIMIT_KEYS)
                assert set(entry["image_formats"]) <= {"png", "jpeg"}
                assert entry["min_side_px"] <= entry["max_side_px"]
                assert entry["max_image_bytes"] <= entry["max_request_bytes"]
            else:
                assert not any(key in entry for key in LIMIT_KEYS)

    def test_entries_are_read_only(self):
        entry = decision_model_capabilities("jev-1.14.0")
        assert entry is not None
        with pytest.raises(TypeError):
            cast(Any, entry)["images"] = False
        assert entry["image_formats"] == ("png", "jpeg")


class TestLookup:
    def test_the_typesafe_prefix_is_optional(self):
        bare = decision_model_capabilities("jev-1.14.0")
        assert bare is not None and dict(bare) == {**JEV_1_14, "image_formats": ("png", "jpeg")}
        assert decision_model_capabilities("typesafe/jev-1.14.0") == bare

    @pytest.mark.parametrize("model", ["nope", "typesafe/nope", "openai/jev-1.14.0"])
    def test_an_unlisted_model_is_none(self, model):
        assert decision_model_capabilities(model) is None

    def test_the_prefix_matches_only_a_typesafe_entry(self, registry_file):
        registry_file({"jev-1.14.0": JEV_1_14, "rubric-dm": {"provider": "acme", "images": False}})
        assert decision_model_capabilities("rubric-dm") is not None
        assert decision_model_capabilities("typesafe/rubric-dm") is None

    @pytest.mark.parametrize(
        "model, sent",
        [
            ("jev-1.14.0", "jev-1.14.0"),
            ("typesafe/jev-1.14.0", "jev-1.14.0"),
            ("typesafe/typesafe/x", "typesafe/x"),
            ("my-org/rubric-dm-7b", "my-org/rubric-dm-7b"),
        ],
    )
    def test_the_api_model_id(self, model, sent):
        assert _api_model_id(model) == sent

    def test_both_names_share_cache_entries(self, tmp_path):
        state, questions = {"submission": "Text."}, {"c0": Noul(instructions="Legible")}
        keys = set()
        for model in ("jev-1.14.0", "typesafe/jev-1.14.0"):
            config = DecisionModelConfig(model=model, api_key=API_KEY, cache_dir=tmp_path)
            with DecisionModelClient(config) as client:
                keys.add(client._cache_key(state, questions))
        assert len(keys) == 1

    def test_a_config_naming_only_the_prefix_is_refused(self):
        with pytest.raises(ValueError, match="names no model after its provider prefix"):
            DecisionModelConfig(model="typesafe/")


class TestMalformedRegistry:
    @pytest.mark.parametrize(
        "content, problem",
        [
            ({"jev-x": {"images": False}}, r'entry \'jev-x\': "provider" must be'),
            ({"jev-x": {"provider": "typesafe", "images": "yes"}}, '"images" must be true'),
            ({"jev-x": {"provider": "typesafe"}}, '"images" must be true'),
            ({"jev-x": {"provider": "typesafe", "images": False, "notes": 3}}, '"notes" must'),
            ({"jev-x": ["typesafe"]}, "it must be a JSON object"),
            ({"typesafe/jev-x": {"provider": "typesafe", "images": False}}, "without a"),
            (
                {"jev-x": {"provider": "typesafe", "images": False, "max_side_px": 1568}},
                r"unknown keys \['max_side_px'\]",
            ),
            (
                {"jev-x": {**JEV_1_14, "max_sidepx": 1}},
                r"unknown keys \['max_sidepx'\]",
            ),
            (
                {"jev-x": {k: v for k, v in JEV_1_14.items() if k != "max_request_bytes"}},
                r"it accepts images, so it needs \['max_request_bytes'\]",
            ),
            ({"jev-x": {**JEV_1_14, "image_formats": ["webp"]}}, '"image_formats" must be'),
            ({"jev-x": {**JEV_1_14, "image_formats": []}}, '"image_formats" must be'),
            ({"jev-x": {**JEV_1_14, "image_formats": ["png", "png"]}}, '"image_formats"'),
            ({"jev-x": {**JEV_1_14, "min_side_px": 0}}, '"min_side_px" must be an integer'),
            ({"jev-x": {**JEV_1_14, "max_side_px": 1568.0}}, '"max_side_px" must be an'),
            ({"jev-x": {**JEV_1_14, "max_image_bytes": True}}, '"max_image_bytes" must be'),
            ({"jev-x": {**JEV_1_14, "min_side_px": 2000}}, '"min_side_px" is over'),
            ({"jev-x": {**JEV_1_14, "max_request_bytes": 1}}, '"max_image_bytes" is over'),
        ],
    )
    def test_a_malformed_entry_is_named(self, registry_file, content, problem):
        path = registry_file(content)
        with pytest.raises(ValueError, match=problem) as excinfo:
            decision_model_capabilities("jev-1.14.0")
        (key,) = content
        assert str(excinfo.value).startswith(
            f"The decision-model registry {path} has a malformed entry {key!r}: "
        )

    @pytest.mark.parametrize(
        "content, problem",
        [
            ("[]", "must be a JSON object of entries"),
            ("{", "cannot be read"),
            (
                '{"jev-x": {"provider": "typesafe", "images": false},'
                ' "jev-x": {"provider": "typesafe", "images": true}}',
                r"cannot be read: repeated keys \['jev-x'\]",
            ),
        ],
    )
    def test_a_malformed_file_is_named(self, registry_file, content, problem):
        path = registry_file(content)
        with pytest.raises(ValueError, match=problem) as excinfo:
            decision_model_capabilities("jev-1.14.0")
        assert str(path) in str(excinfo.value)


# =============================================================================
# Enforcement in the grader: only for images, before any judge is called
# =============================================================================


class TestImagesAreRefused:
    @pytest.mark.asyncio
    async def test_a_model_without_images(self, fake_sdk, make_grader):
        grader = make_grader(JudgeSpec(dm("jev-1.13.0"), "jev"))
        with pytest.raises(ValueError) as excinfo:
            await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        assert str(excinfo.value) == (
            "Judge 'jev' uses 'jev-1.13.0', which does not accept images; use jev-1.14.0 or later"
        )
        assert _is_judge_configuration_error(excinfo.value)
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_a_model_not_in_the_registry(self, fake_sdk, make_grader):
        grader = make_grader(JudgeSpec(dm("my-org/rubric-dm-7b"), "mine"))
        with pytest.raises(ValueError) as excinfo:
            await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        assert str(excinfo.value) == (
            "Judge 'mine' uses decision model 'my-org/rubric-dm-7b', which is not in "
            "autorubric's decision-model registry (src/autorubric/data/decision_models.json); "
            "add an entry saying whether it accepts images"
        )
        assert _is_judge_configuration_error(excinfo.value)
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("where", ["query", "reference_submission"])
    async def test_images_in_the_context_are_refused_too(self, fake_sdk, make_grader, where):
        grader = make_grader(JudgeSpec(dm("jev-1.13.0"), "jev"))
        with pytest.raises(ValueError, match="does not accept images"):
            await Rubric([LEGIBLE]).grade("A caption.", grader=grader, **{where: [FIGURE]})
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_a_mixed_panel_sends_no_request_at_all(self, fake_sdk, make_grader):
        """The decision model is the panel's second judge, so a check inside its own
        coroutine would come after the LLM judge's call had started."""
        grader = make_grader(
            JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "llm"),
            JudgeSpec(dm("jev-1.13.0"), "jev"),
        )
        completion = AsyncMock(side_effect=AssertionError("an LLM judge was called"))
        with patch("litellm.acompletion", completion):
            with pytest.raises(ValueError, match="Judge 'jev' uses 'jev-1.13.0'"):
                await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        completion.assert_not_called()
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_a_cascade_sends_no_request_at_all(self, fake_sdk):
        grader = CriterionGrader(
            judge_model_config=dm("jev-1.13.0"),
            escalation=EscalationConfig(
                judges=[JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "llm")], threshold=0.9
            ),
        )
        completion = AsyncMock(side_effect=AssertionError("an LLM judge was called"))
        try:
            with patch("litellm.acompletion", completion):
                with pytest.raises(ValueError, match="Judge 'default' uses 'jev-1.13.0'"):
                    await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        finally:
            for client in grader._decision_clients.values():
                client.close()
        completion.assert_not_called()
        assert fake_sdk.calls == []


class TestImagesAreSent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["jev-1.14.0", "typesafe/jev-1.14.0", "jev-latest"])
    async def test_a_model_with_images(self, fake_sdk, make_grader, model):
        grader = make_grader(JudgeSpec(dm(model), "jev"))
        report = await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)

        assert report.score == 1.0
        ((_, _, kwargs),) = fake_sdk.calls
        assert list(kwargs["assets"].values())[0].data == FIGURE.read_bytes()
        assert kwargs["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
        # The prefix is never sent; the manifest records the model as given.
        (sdk_client,) = fake_sdk.clients
        assert sdk_client.kwargs["model"] == model.removeprefix("typesafe/")
        (entry,) = _serialize_grader_config(grader)["judges"]
        assert entry["model"] == model

    @pytest.mark.asyncio
    async def test_the_images_are_fitted_into_the_models_entry(
        self, fake_sdk, make_grader, registry_file
    ):
        registry_file({"jev-small": {**JEV_1_14, "min_side_px": 100, "max_side_px": 200}})
        grader = make_grader(JudgeSpec(dm("jev-small"), "jev"))
        await Rubric([LEGIBLE]).grade([FIGURE], grader=grader)
        ((_, _, kwargs),) = fake_sdk.calls
        (asset,) = kwargs["assets"].values()
        assert pil_image.open(io.BytesIO(asset.data)).size == (200, 150)

    @pytest.mark.asyncio
    async def test_images_over_the_request_limit_together_are_refused(
        self, fake_sdk, make_grader, registry_file
    ):
        limit = max(len(FIGURE.read_bytes()), len(OTHER.read_bytes())) + 1
        registry_file(
            {"jev-1.14.0": {**JEV_1_14, "max_image_bytes": limit, "max_request_bytes": limit}}
        )
        grader = make_grader(JudgeSpec(dm("jev-1.14.0"), "jev"))
        total = len(FIGURE.read_bytes()) + len(OTHER.read_bytes())
        with pytest.raises(ValueError) as excinfo:
            await Rubric([LEGIBLE]).grade([FIGURE, OTHER, CAPTION], grader=grader)
        assert str(excinfo.value) == (
            f"Judge 'jev' cannot send this item's images to 'jev-1.14.0': fitted, they total "
            f"{total} bytes, over the {limit}-byte limit on a request's images "
            "(max_request_bytes)"
        )
        assert not _is_judge_configuration_error(excinfo.value)  # the item's images
        assert fake_sdk.calls == []

        # Either image alone fits.
        await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        assert len(fake_sdk.calls) == 1


class TestTextNeverReadsTheRegistry:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["jev-1.13.0", "my-org/rubric-dm-7b"])
    async def test_text_is_graded_as_before(self, fake_sdk, make_grader, lookups, model):
        grader = make_grader(JudgeSpec(dm(model), "jev"))
        report = await Rubric([LEGIBLE]).grade(
            "Plain text.", grader=grader, query="Describe it.", reference_submission="Text."
        )
        assert report.score == 1.0
        ((_, _, kwargs),) = fake_sdk.calls
        assert kwargs == {}
        assert lookups == []

    @pytest.mark.asyncio
    async def test_a_malformed_registry_does_not_matter_for_text(
        self, fake_sdk, make_grader, registry_file
    ):
        registry_file("not json")
        grader = make_grader(JudgeSpec(dm("jev-1.14.0"), "jev"))
        await Rubric([LEGIBLE]).grade("Plain text.", grader=grader)
        assert len(fake_sdk.calls) == 1

    @pytest.mark.asyncio
    async def test_images_with_an_llm_only_grader_never_read_it(self, lookups):
        grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"))
        reply = AsyncMock(return_value=met_reply())
        with patch("litellm.acompletion", reply):
            report = await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        assert report.score == 1.0
        reply.assert_called_once()
        assert lookups == []


# =============================================================================
# fill_ground_truth: configuration errors are raised, not recorded per item
# =============================================================================


def image_dataset(*images: Image) -> RubricDataset:
    """An item holding ``images`` (``FIGURE`` by default) and a caption, then a text item."""
    dataset = RubricDataset(prompt="Describe the figure.", rubric=Rubric([LEGIBLE]))
    # Datasets are annotated as text; grading takes any content.
    document = [*(images or (FIGURE,)), CAPTION]
    dataset.add_item(submission=cast(str, document), description="a figure")
    dataset.add_item(submission="Plain text.", description="text")
    return dataset


class TestFillGroundTruth:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["jev-1.13.0", "my-org/rubric-dm-7b"])
    async def test_a_registry_refusal_is_raised(self, fake_sdk, make_grader, model):
        grader = make_grader(JudgeSpec(dm(model), "jev"))
        with pytest.raises(ValueError, match="jev") as excinfo:
            await fill_ground_truth(image_dataset(), grader, show_progress=False)
        assert _CONFIGURATION_ERROR_NOTE in excinfo.value.__notes__

    @pytest.mark.asyncio
    async def test_an_sdk_without_image_is_raised(self, fake_sdk, make_grader, monkeypatch):
        grader = make_grader(JudgeSpec(dm("jev-1.14.0"), "jev"))
        monkeypatch.delattr(typesafe_sdk, "Image")
        with pytest.raises(ImportError, match=r"typesafe-sdk 0\.7\.5 or later"):
            await fill_ground_truth(image_dataset(), grader, show_progress=False)

    @pytest.mark.asyncio
    async def test_an_item_whose_images_cannot_be_sent_is_left_out(
        self, fake_sdk, make_grader, registry_file
    ):
        """Images over the request limit are the item's own: the item is left out and the
        others are labelled, as for any item that fails to grade."""
        limit = max(len(FIGURE.read_bytes()), len(OTHER.read_bytes())) + 1
        registry_file(
            {"jev-1.14.0": {**JEV_1_14, "max_image_bytes": limit, "max_request_bytes": limit}}
        )
        grader = make_grader(JudgeSpec(dm("jev-1.14.0"), "jev"))
        labeled = await fill_ground_truth(image_dataset(FIGURE, OTHER), grader, show_progress=False)
        assert [item.description for item in labeled.items] == ["text"]
        assert len(fake_sdk.calls) == 1  # the text item's request
