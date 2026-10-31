"""A decision model's capabilities declared on its config: ``DecisionModelConfig.capabilities``.

An installed user cannot edit the packaged registry (``data/decision_models.json``), so a
judge's config may declare its model's entry itself. When set, ``capabilities`` replaces the
registry entry for that judge, for listed and unlisted models alike. It is checked when the
config is built, by the same rules as a packaged entry (``_checked_capabilities``), stored as
a plain dict and handed out read-only by the one lookup (``_judge_capabilities``). It is read
only for a request that would carry images, enters the cache key of such a request as the
envelope its images were fitted into, and is recorded in the manifest only when set.

Decision models answer through the recording fake SDK (``fake_sdk``, see ``conftest.py``);
LLM judges' completions are patched. No test reaches the network.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import pickle
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import PIL.Image as pil_image
import pytest
from typesafe_sdk import Choice, Noul

from autorubric import Criterion, DecisionModelConfig, Image, LLMConfig, Rubric
from autorubric import decision as decision_module
from autorubric.decision import (
    DecisionModelClient,
    _canonical_json,
    _decision_model_registry,
    _judge_capabilities,
    build_state,
    decision_model_capabilities,
    state_images,
)
from autorubric.eval import _serialize_grader_config
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.graders import criterion_grader as grader_module
from autorubric.graders.criterion_grader import _is_judge_configuration_error

API_KEY = "ts-test-key-never-persisted-31f8"
LEGIBLE = Criterion(name="legible", weight=1.0, requirement="The figure is legible")
CAPTION = "Figure 1: accuracy by model size."
SMALL = {
    "provider": "acme",
    "images": True,
    "image_formats": ["jpeg"],
    "min_side_px": 100,
    "max_side_px": 200,
    "max_image_bytes": 1_000_000,
    "max_request_bytes": 2_000_000,
}
"""An unlisted model's entry: JPEG only, sides 100 to 200 pixels."""
NO_IMAGES = {"provider": "typesafe", "images": False}
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


def dm(model: str = "my-org/rubric-dm-7b", **overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(model=model, api_key=API_KEY, **overrides)


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
def make_client(tmp_path: Path) -> Iterator[Callable[..., DecisionModelClient]]:
    """Build clients whose on-disk caches are closed at teardown."""
    clients: list[DecisionModelClient] = []

    def build(config: DecisionModelConfig) -> DecisionModelClient:
        client = DecisionModelClient(config)
        clients.append(client)
        return client

    yield build
    for client in clients:
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


# =============================================================================
# Grading with declared capabilities
# =============================================================================


class TestAnUnlistedModelWithCapabilities:
    @pytest.mark.asyncio
    async def test_its_images_are_sent_fitted_into_the_given_limits(self, fake_sdk, make_grader):
        grader = make_grader(JudgeSpec(dm(capabilities=SMALL), "mine"))
        report = await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)

        assert report.score == 1.0
        ((_, _, kwargs),) = fake_sdk.calls
        assert kwargs["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
        (asset,) = kwargs["assets"].values()
        fitted = pil_image.open(io.BytesIO(asset.data))
        assert (fitted.format, fitted.size) == ("JPEG", (200, 150))

    @pytest.mark.asyncio
    async def test_the_request_limit_is_the_given_one(self, fake_sdk, make_grader):
        limits = {**SMALL, "image_formats": ["png"], "min_side_px": 280, "max_side_px": 1568}
        limit = len(FIGURE.read_bytes()) + 1
        limits.update(max_image_bytes=limit, max_request_bytes=limit)
        grader = make_grader(JudgeSpec(dm(capabilities=limits), "mine"))
        with pytest.raises(ValueError, match=f"over the {limit}-byte limit"):
            await Rubric([LEGIBLE]).grade([FIGURE, Image(png("blue"))], grader=grader)
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_without_capabilities_the_refusal_names_the_field(self, fake_sdk, make_grader):
        grader = make_grader(JudgeSpec(dm(), "mine"))
        with pytest.raises(ValueError) as excinfo:
            await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        assert str(excinfo.value) == (
            "Judge 'mine' uses decision model 'my-org/rubric-dm-7b', which is not in "
            "autorubric's decision-model registry (autorubric/data/decision_models.json, "
            "inside the installed package); add an entry saying whether it accepts images, "
            "or pass DecisionModelConfig(capabilities=...)"
        )
        assert _is_judge_configuration_error(excinfo.value)
        assert fake_sdk.calls == []


class TestCapabilitiesReplaceTheRegistryEntry:
    @pytest.mark.asyncio
    async def test_a_listed_model_follows_the_given_limits(self, fake_sdk, make_grader):
        grader = make_grader(JudgeSpec(dm("jev-1.14.0", capabilities=SMALL), "jev"))
        await Rubric([LEGIBLE]).grade([FIGURE], grader=grader)
        ((_, _, kwargs),) = fake_sdk.calls
        (asset,) = kwargs["assets"].values()
        assert pil_image.open(io.BytesIO(asset.data)).size == (200, 150)

    @pytest.mark.asyncio
    async def test_images_false_refuses_a_listed_image_model(self, fake_sdk, make_grader):
        grader = make_grader(JudgeSpec(dm("jev-1.14.0", capabilities=NO_IMAGES), "jev"))
        with pytest.raises(ValueError) as excinfo:
            await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        assert str(excinfo.value) == (
            "Judge 'jev' uses 'jev-1.14.0', which does not accept images; "
            "its DecisionModelConfig(capabilities=...) says images: false"
        )
        assert _is_judge_configuration_error(excinfo.value)
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_images_false_in_a_mixed_panel_sends_no_request_at_all(
        self, fake_sdk, make_grader
    ):
        grader = make_grader(
            JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini"), "llm"),
            JudgeSpec(dm("jev-1.14.0", capabilities=NO_IMAGES), "jev"),
        )
        completion = AsyncMock(side_effect=AssertionError("an LLM judge was called"))
        with patch("litellm.acompletion", completion):
            with pytest.raises(ValueError, match="Judge 'jev' uses 'jev-1.14.0'"):
                await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        completion.assert_not_called()
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_a_malformed_registry_does_not_matter_with_capabilities(
        self, fake_sdk, make_grader, registry_file
    ):
        registry_file("not json")
        grader = make_grader(JudgeSpec(dm("jev-1.14.0", capabilities=SMALL), "jev"))
        await Rubric([LEGIBLE]).grade([FIGURE], grader=grader)
        assert len(fake_sdk.calls) == 1

    def test_the_typesafe_prefix_does_not_restrict_capabilities(self):
        """The registry matches ``typesafe/x`` only to a TypeSafe entry; a declared entry
        replaces that lookup, whatever its provider."""
        entry = _judge_capabilities(dm("typesafe/rubric-dm", capabilities=SMALL))
        assert entry is not None and entry["provider"] == "acme"


class TestTheLookup:
    def test_it_hands_out_a_read_only_view(self):
        entry = _judge_capabilities(dm(capabilities=SMALL))
        assert isinstance(entry, MappingProxyType)
        assert dict(entry) == {**SMALL, "image_formats": ("jpeg",)}
        with pytest.raises(TypeError):
            cast(Any, entry)["images"] = False

    def test_without_capabilities_it_is_the_registry_lookup(self):
        assert _judge_capabilities(dm("jev-1.14.0")) == decision_model_capabilities("jev-1.14.0")
        assert _judge_capabilities(dm()) is None

    def test_a_dict_changed_after_construction_is_checked_again(self):
        config = dm(capabilities=SMALL)
        assert config.capabilities is not None
        cast(dict[str, Any], config.capabilities)["max_side_px"] = "200"
        with pytest.raises(ValueError, match='"max_side_px" must be an integer') as excinfo:
            _judge_capabilities(config)
        assert _is_judge_configuration_error(excinfo.value)


# =============================================================================
# Validation: the packaged entry's rules, the same messages
# =============================================================================

MALFORMED = [
    ({"images": False}, '"provider" must be'),
    ({"provider": "", "images": False}, '"provider" must be'),
    ({"provider": "typesafe", "images": "yes"}, '"images" must be true'),
    ({"provider": "typesafe"}, '"images" must be true'),
    ({"provider": "typesafe", "images": False, "notes": 3}, '"notes" must'),
    (["typesafe"], "it must be a JSON object"),
    ({"provider": "typesafe", "images": False, "max_side_px": 1568}, r"unknown keys \['max_side"),
    ({**JEV_1_14, "max_sidepx": 1}, r"unknown keys \['max_sidepx'\]"),
    (
        {k: v for k, v in JEV_1_14.items() if k != "max_request_bytes"},
        r"it accepts images, so it needs \['max_request_bytes'\]",
    ),
    ({**JEV_1_14, "image_formats": ["webp"]}, '"image_formats" must be'),
    ({**JEV_1_14, "image_formats": []}, '"image_formats" must be'),
    ({**JEV_1_14, "image_formats": ["png", "png"]}, '"image_formats"'),
    ({**JEV_1_14, "image_formats": "png"}, '"image_formats" must be'),
    ({**JEV_1_14, "min_side_px": 0}, '"min_side_px" must be an integer'),
    ({**JEV_1_14, "max_side_px": 1568.0}, '"max_side_px" must be an'),
    ({**JEV_1_14, "max_image_bytes": True}, '"max_image_bytes" must be'),
    ({**JEV_1_14, "min_side_px": 2000}, '"min_side_px" is over'),
    ({**JEV_1_14, "max_request_bytes": 1}, '"max_image_bytes" is over'),
]


class TestValidation:
    @pytest.mark.parametrize("capabilities, problem", MALFORMED)
    def test_a_malformed_shape_is_refused_naming_the_field(self, capabilities, problem):
        with pytest.raises(ValueError, match=problem) as excinfo:
            dm(capabilities=capabilities)
        assert str(excinfo.value).startswith("DecisionModelConfig.capabilities is malformed: ")

    @pytest.mark.parametrize("capabilities, problem", MALFORMED)
    def test_the_message_is_the_registrys(self, registry_file, capabilities, problem):
        path = registry_file({"jev-x": capabilities})
        with pytest.raises(ValueError) as from_registry:
            decision_model_capabilities("jev-x")
        with pytest.raises(ValueError) as from_config:
            dm(capabilities=capabilities)
        registry_prefix = f"The decision-model registry {path} has a malformed entry 'jev-x': "
        config_prefix = "DecisionModelConfig.capabilities is malformed: "
        assert str(from_registry.value).startswith(registry_prefix)
        assert str(from_registry.value).removeprefix(registry_prefix) == str(
            from_config.value
        ).removeprefix(config_prefix)

    def test_a_registry_entry_can_be_built_on(self):
        """An entry ``decision_model_capabilities`` hands out (a read-only view whose
        ``image_formats`` is a tuple) is accepted as a starting point."""
        listed = decision_model_capabilities("jev-latest")
        assert listed is not None
        config = dm(capabilities={**listed, "max_side_px": 1024})
        assert config.capabilities == {
            **JEV_1_14,
            "max_side_px": 1024,
            "notes": listed["notes"],
        }
        assert dm(capabilities=listed).capabilities == {**JEV_1_14, "notes": listed["notes"]}


# =============================================================================
# The stored value: a plain dict copy that pickles and compares
# =============================================================================


class TestTheStoredValue:
    def test_it_is_a_plain_dict_copy(self):
        given = copy.deepcopy(SMALL)
        config = dm(capabilities=given)
        assert type(config.capabilities) is dict
        assert config.capabilities == SMALL
        given["max_side_px"] = 9999
        cast(list[str], given["image_formats"]).append("png")
        assert config.capabilities == SMALL

    def test_tuple_formats_are_stored_as_a_list(self):
        config = dm(capabilities={**SMALL, "image_formats": ("jpeg",)})
        assert config.capabilities == SMALL
        assert config == dm(capabilities=SMALL)

    def test_it_pickles_and_compares(self):
        config = dm(capabilities=SMALL)
        restored = pickle.loads(pickle.dumps(config))
        assert restored == config
        assert restored.capabilities == SMALL
        assert config != dm(capabilities={**SMALL, "max_side_px": 199})
        assert config != dm()

    def test_replace_keeps_it(self):
        config = dm(capabilities=SMALL)
        assert dataclasses.replace(config, timeout=5.0).capabilities == SMALL

    def test_the_default_is_none(self):
        assert dm().capabilities is None
        assert "capabilities" in {f.name for f in dataclasses.fields(DecisionModelConfig)}


# =============================================================================
# The manifest: recorded only when set
# =============================================================================


class TestManifest:
    def test_it_is_recorded_when_set(self, make_grader):
        grader = make_grader(JudgeSpec(dm(capabilities=SMALL), "mine"))
        (entry,) = _serialize_grader_config(grader)["judges"]
        assert entry["capabilities"] == SMALL
        assert json.loads(json.dumps(entry))["capabilities"] == SMALL

    def test_without_it_the_entry_is_unchanged(self, make_grader):
        grader = make_grader(JudgeSpec(dm(), "mine"))
        (entry,) = _serialize_grader_config(grader)["judges"]
        assert list(entry) == [
            "judge_id",
            "judge_kind",
            "model",
            "weight",
            "max_parallel_requests",
            "binary_framing",
            "ordinal_framing",
            "decision_threshold",
            "api_base_host",
        ]


# =============================================================================
# The cache key: an image request's names the envelope its images were fitted into
# =============================================================================

QUESTIONS = {"c0": Noul(instructions="Legible")}
TEXT_STATE = {"input": "Explain photosynthesis.", "submission": "Plants turn light into sugar."}
TEXT_QUESTIONS = {
    "c0": Noul(instructions="Mentions light"),
    "c1": Choice(instructions="Tone", criteria={"formal": None, "casual": None}),
}


class TestCacheKeys:
    def test_an_image_key_adds_the_envelopes_image_limits(self, make_client, tmp_path):
        client = make_client(dm(cache_dir=tmp_path))
        state = build_state([FIGURE, CAPTION])
        assets = state_images([FIGURE])
        envelope = MappingProxyType({**SMALL, "notes": "ignored", "image_formats": ("jpeg",)})
        parts = [
            "my-org/rubric-dm-7b",
            "https://api.typesafe.ai",
            state,
            QUESTIONS,
            {
                "assets": sorted(assets),
                "envelope": {
                    "image_formats": ["jpeg"],
                    "min_side_px": 100,
                    "max_side_px": 200,
                    "max_image_bytes": 1_000_000,
                    "max_request_bytes": 2_000_000,
                },
            },
        ]
        expected = hashlib.sha256(_canonical_json(parts).encode("utf-8")).hexdigest()
        assert client._cache_key(state, QUESTIONS, assets, envelope) == expected

    def test_different_envelopes_have_different_keys(self, make_client, tmp_path):
        client = make_client(dm(cache_dir=tmp_path))
        state = build_state([FIGURE, CAPTION])
        assets = state_images([FIGURE])
        keys = {
            client._cache_key(state, QUESTIONS, assets, envelope)
            for envelope in (
                SMALL,
                {**SMALL, "max_side_px": 199},
                {**SMALL, "image_formats": ["png", "jpeg"]},
                {**SMALL, "max_request_bytes": 2_000_001},
            )
        }
        assert len(keys) == 4

    def test_provider_and_notes_do_not_change_the_key(self, make_client, tmp_path):
        """Only the limits shape the fitted images."""
        client = make_client(dm(cache_dir=tmp_path))
        state = build_state([FIGURE, CAPTION])
        assets = state_images([FIGURE])
        assert client._cache_key(state, QUESTIONS, assets, SMALL) == client._cache_key(
            state, QUESTIONS, assets, {**SMALL, "provider": "other", "notes": "n"}
        )

    def test_an_image_key_needs_an_envelope(self, make_client, tmp_path):
        client = make_client(dm(cache_dir=tmp_path))
        with pytest.raises(ValueError, match="envelope"):
            client._cache_key(build_state([FIGURE]), QUESTIONS, state_images([FIGURE]))

    @pytest.mark.asyncio
    async def test_a_request_with_assets_needs_an_envelope(self, fake_sdk, make_client, tmp_path):
        client = make_client(dm(cache_dir=tmp_path))
        assets = client.prepare_assets(state_images([FIGURE]), envelope=SMALL)
        with pytest.raises(ValueError, match="envelope"):
            await client.system_one(build_state([FIGURE]), QUESTIONS, assets=assets)
        assert fake_sdk.calls == []

    def test_text_keys_are_unchanged(self, make_client, tmp_path):
        """The literal key ``tests/decision/test_decision_images.py`` pins, captured before
        image support; declaring capabilities does not change a text request's key."""
        client = make_client(dm("jev-latest", cache_dir=tmp_path, capabilities=SMALL))
        literal = "956ba06dc3503fc62fec13591b0b43377b5ddf4ce5d0b1002dba9e32c657589b"
        assert client._cache_key(TEXT_STATE, TEXT_QUESTIONS) == literal
        assert client._cache_key(TEXT_STATE, TEXT_QUESTIONS, {}) == literal
        assert client._cache_key(TEXT_STATE, TEXT_QUESTIONS, {}, SMALL) == literal

    @pytest.mark.asyncio
    async def test_changed_limits_ask_again(self, fake_sdk, make_grader, tmp_path):
        """Answers for images fitted into one envelope are never reused for another."""
        for limits in (SMALL, SMALL, {**SMALL, "max_side_px": 150}):
            config = dm(capabilities=limits, cache_enabled=True, cache_dir=tmp_path)
            grader = make_grader(JudgeSpec(config, "mine"))
            await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        sizes = [
            pil_image.open(io.BytesIO(next(iter(kwargs["assets"].values())).data)).size
            for _, _, kwargs in fake_sdk.calls
        ]
        assert sizes == [(200, 150), (150, 112)]  # the second grade was a cache hit

    @pytest.mark.asyncio
    async def test_changed_registry_limits_ask_again(
        self, fake_sdk, make_grader, registry_file, tmp_path
    ):
        for limits in (SMALL, {**SMALL, "max_side_px": 150}):
            registry_file({"rubric-dm": limits})
            config = dm("rubric-dm", cache_enabled=True, cache_dir=tmp_path)
            grader = make_grader(JudgeSpec(config, "mine"))
            await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        assert len(fake_sdk.calls) == 2


# =============================================================================
# Text never reads the capabilities or the registry
# =============================================================================


class TestTextNeverReadsCapabilities:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("capabilities", [SMALL, NO_IMAGES, None])
    async def test_text_is_graded_without_a_lookup(
        self, fake_sdk, make_grader, registry_file, monkeypatch, capabilities
    ):
        registry_file("not json")

        def refuse(*_: Any) -> Any:
            raise AssertionError("capabilities were looked up for text")

        monkeypatch.setattr(decision_module, "_judge_capabilities", refuse)
        monkeypatch.setattr(grader_module, "_judge_capabilities", refuse)
        monkeypatch.setattr(decision_module, "decision_model_capabilities", refuse)
        grader = make_grader(JudgeSpec(dm("jev-1.13.0", capabilities=capabilities), "jev"))
        report = await Rubric([LEGIBLE]).grade(
            "Plain text.", grader=grader, query="Describe it.", reference_submission="Text."
        )
        assert report.score == 1.0
        ((_, _, kwargs),) = fake_sdk.calls
        assert kwargs == {}
