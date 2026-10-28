"""Images sent to decision models: state references, SDK assets, fitting, header, cache key.

A document's images are written into the decision model's state as deterministic
references, ``<image:{sha256}>``, and sent as the request's System One assets under those
references, each fitted into the envelope Jev accepts (System One never resizes an image).
A request carrying images has the ``TypeSafe-Danger-Zone-Images: true`` header; a text-only
request, its state and its cache key are exactly what they were before image support.

Requests go to the recording fake SDK client (``fake_sdk``) or, for the wire test, to the
real SDK client answered in place of its transport (``http_endpoint``); see
``conftest.py``. No test reaches the network.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import io
import json
import sys
import threading
from typing import Any

import PIL.Image as pil_image
import pytest
import typesafe_sdk
from PIL import ExifTags, PngImagePlugin
from typesafe_sdk import Choice, Noul

from autorubric import Criterion, DecisionModelConfig, Image, Rubric
from autorubric import content as content_module
from autorubric.decision import (
    _IMAGES_HEADER,
    _JEV_IMAGE_ENVELOPE,
    DecisionModelClient,
    _canonical_json,
    _fit_for_system_one,
    _system_one_size,
    build_state,
    state_images,
)
from autorubric.graders import CriterionGrader, JudgeSpec

API_KEY = "ts-test-key-never-persisted-7c2e"
CAPTION = "Figure 1: accuracy by model size."
LEGIBLE = Criterion(name="legible", weight=1.0, requirement="The figure is legible")
QUESTIONS = {
    "c0": Noul(instructions="Mentions light"),
    "c1": Choice(instructions="Tone", criteria={"formal": None, "casual": None}),
}
WHITE = (255, 255, 255)
RED = (255, 0, 0)


def encoded(
    size: tuple[int, int], image_format: str, *, mode: str = "RGB", color: Any = RED, **save: Any
) -> bytes:
    """An image of one ``color``, encoded by Pillow as ``image_format``."""
    buffer = io.BytesIO()
    pil_image.new(mode, size, color).save(buffer, image_format, **save)
    return buffer.getvalue()


def two_frames(image_format: str, **save: Any) -> bytes:
    """A 300 x 300 image of two frames, red then blue (animated, or an MPO's two pictures)."""
    buffer = io.BytesIO()
    first, second = pil_image.new("RGB", (300, 300), RED), pil_image.new("RGB", (300, 300), "blue")
    first.save(buffer, image_format, save_all=True, append_images=[second], **save)
    return buffer.getvalue()


def decoded(data: bytes) -> pil_image.Image:
    opened = pil_image.open(io.BytesIO(data))
    opened.load()
    return opened


def reference(image: Image) -> str:
    return f"<image:{hashlib.sha256(image.read_bytes()).hexdigest()}>"


def dm(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(model="jev-latest", api_key=API_KEY, **overrides)


@pytest.fixture
def make_client(tmp_path):
    """Build clients whose on-disk caches are closed at teardown (Windows keeps them open)."""
    clients: list[DecisionModelClient] = []

    def build(**overrides: Any) -> DecisionModelClient:
        overrides.setdefault("cache_dir", tmp_path / "cache")
        client = DecisionModelClient(dm(**overrides))
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.close()


@pytest.fixture
def make_grader():
    """Build single decision-model graders whose clients are closed at teardown."""
    graders: list[CriterionGrader] = []

    def build(**overrides: Any) -> CriterionGrader:
        grader = CriterionGrader(judges=[JudgeSpec(dm(**overrides), "jev")])
        graders.append(grader)
        return grader

    yield build
    for grader in graders:
        for client in grader._decision_clients.values():
            client.close()


FIGURE = Image(encoded((400, 300), "PNG"))
OTHER = Image(encoded((400, 300), "PNG", color="blue"))


# =============================================================================
# The state: each image's reference where the image is
# =============================================================================


class TestState:
    def test_a_documents_images_become_references_in_place(self):
        state = build_state(["Before ", FIGURE, " between ", OTHER, " after"])
        assert state == {
            "submission": f"Before {reference(FIGURE)} between {reference(OTHER)} after"
        }

    def test_the_reference_is_the_images_full_sha256(self):
        assert reference(FIGURE) == f"<image:{FIGURE.digest}>"
        assert len(FIGURE.digest) == 64

    def test_an_image_that_occurs_twice_has_one_reference(self):
        state = build_state([FIGURE, CAPTION, FIGURE])
        assert state["submission"] == reference(FIGURE) + CAPTION + reference(FIGURE)

    def test_query_and_reference_documents_carry_references_too(self):
        state = build_state(
            "Plain text.", query=[FIGURE, "Describe this."], reference_submission=[OTHER]
        )
        assert state == {
            "input": reference(FIGURE) + "Describe this.",
            "reference_submission": reference(OTHER),
            "submission": "Plain text.",
        }

    def test_a_documents_text_is_never_split_into_thinking_and_output(self):
        text = "<thinking>T</thinking><output>O</output>"
        assert build_state([FIGURE, text]) == {"submission": reference(FIGURE) + text}
        # Text alone is split, as before images existed.
        assert build_state(text) == {"thinking": "T", "output": "O"}

    def test_text_is_sent_as_the_same_string(self):
        text = "Plants turn light into sugar."
        state = build_state(text, query="Explain photosynthesis.")
        assert state == {"input": "Explain photosynthesis.", "submission": text}
        assert state["submission"] is text

    def test_a_sequence_of_strings_is_their_text(self):
        assert build_state(["a", "b"]) == {"submission": "ab"}


class TestStateImages:
    def test_images_in_order_of_first_occurrence_each_once(self):
        images = state_images([FIGURE, CAPTION], [OTHER, FIGURE], None)
        assert images == {reference(FIGURE): FIGURE, reference(OTHER): OTHER}
        assert list(images) == [reference(FIGURE), reference(OTHER)]

    def test_the_same_bytes_from_another_source_are_one_image(self, tmp_path):
        path = tmp_path / "figure.png"
        path.write_bytes(FIGURE.read_bytes())
        images = state_images([FIGURE], [Image(path)])
        assert list(images) == [reference(FIGURE)]

    def test_text_has_no_images(self):
        assert state_images("text", None, ["a", "b"]) == {}

    def test_every_reference_in_the_state_is_an_asset(self):
        submission, query = [FIGURE, CAPTION, OTHER], [OTHER]
        state = build_state(submission, query=query)
        images = state_images(submission, query)
        for text in state.values():
            for alias in images:
                text = text.replace(alias, "")
            assert "<image:" not in text


# =============================================================================
# Fitting each image into Jev's envelope
# =============================================================================


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((300, 300), (300, 300, 300, 300)),  # in the envelope
        ((280, 1568), (280, 1568, 280, 1568)),  # at both bounds, inclusive
        ((4000, 3000), (1568, 1176, 1568, 1176)),  # longest side down to 1568
        ((3000, 4000), (1176, 1568, 1176, 1568)),
        ((100, 100), (280, 280, 280, 280)),  # shortest side up to 280
        ((200, 1000), (280, 1400, 280, 1400)),  # up while the longest stays within 1568
        ((953, 89), (1568, 146, 1568, 280)),  # longest up to 1568, short side padded
        ((89, 953), (146, 1568, 280, 1568)),
        ((2000, 100), (1568, 78, 1568, 280)),  # longest down, short side padded
        ((1, 1), (280, 280, 280, 280)),
        ((20000, 1), (1568, 1, 1568, 280)),  # a side never rounds below 1 pixel
        ((1, 20000), (1, 1568, 280, 1568)),
        ((1600, 285), (1568, 280, 1568, 280)),  # 279.3 rounded: set to 280, not padded by 1
        ((285, 1600), (280, 1568, 280, 1568)),
        ((1500, 267), (1568, 280, 1568, 280)),  # up to 1568 x 279.1: set to 280
        ((1567, 279), (1567, 279, 1567, 280)),  # up by 1.0006: kept and padded, not resampled
        ((279, 279), (279, 279, 280, 280)),
    ],
)
def test_system_one_size(size, expected):
    assert _system_one_size(*size, min_side=280, max_side=1568) == expected


def fit(data: bytes, **envelope: Any) -> tuple[bytes, str]:
    return _fit_for_system_one(Image(data), envelope={**_JEV_IMAGE_ENVELOPE, **envelope})


class TestPassThrough:
    @pytest.mark.parametrize(
        ("data", "image_format"),
        [
            (encoded((400, 300), "PNG"), "png"),
            (encoded((280, 1568), "PNG"), "png"),
            (encoded((400, 300), "PNG", mode="RGBA", color=(255, 0, 0, 128)), "png"),
            (encoded((400, 300), "PNG", mode="P", transparency=0), "png"),
            (encoded((400, 300), "JPEG"), "jpeg"),
            (encoded((400, 300), "JPEG", mode="L", color=128), "jpeg"),
        ],
        ids=["png", "png-at-bounds", "png-rgba", "png-palette", "jpeg", "jpeg-grayscale"],
    )
    def test_an_image_system_one_takes_is_sent_as_its_original_bytes(self, data, image_format):
        assert fit(data) == (data, image_format)

    def test_a_jpeg_with_upright_exif_orientation_passes_through(self):
        exif = pil_image.Exif()
        exif[ExifTags.Base.Orientation] = 1
        data = encoded((400, 300), "JPEG", exif=exif)
        assert fit(data) == (data, "jpeg")

    def test_an_image_over_the_byte_limit_is_re_encoded(self):
        """A PNG within the size bounds but heavy with metadata is re-encoded without it."""
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", "x" * 50_000)
        data = encoded((400, 300), "PNG", pnginfo=info)
        fitted, image_format = fit(data, max_image_bytes=20_000)
        assert image_format == "png"
        assert len(fitted) < 20_000
        assert decoded(fitted).size == (400, 300)


class TestResizing:
    def test_an_oversize_image_is_scaled_down_to_the_longest_side(self):
        fitted, image_format = fit(encoded((4000, 3000), "JPEG"))
        assert image_format == "jpeg"
        out = decoded(fitted)
        assert (out.format, out.size, out.mode) == ("JPEG", (1568, 1176), "RGB")

    def test_a_tiny_image_is_scaled_up_to_the_shortest_side(self):
        fitted, image_format = fit(encoded((100, 100), "PNG"))
        out = decoded(fitted)
        assert (image_format, out.format, out.size) == ("png", "PNG", (280, 280))
        assert out.getpixel((140, 140)) == RED

    def test_an_elongated_image_is_scaled_then_padded_centered_with_white(self):
        fitted, _ = fit(encoded((953, 89), "PNG"))
        out = decoded(fitted)
        assert out.size == (1568, 280)
        # 146 rows of the image, centered: 67 white rows above, 67 below.
        assert [out.getpixel((784, y)) for y in (0, 66, 67, 212, 213, 279)] == [
            WHITE,
            WHITE,
            RED,
            RED,
            WHITE,
            WHITE,
        ]

    @pytest.mark.parametrize(
        ("size", "fitted_size"), [((1, 1), (280, 280)), ((20000, 1), (1568, 280))]
    )
    def test_extreme_sizes_are_fitted(self, size, fitted_size):
        fitted, _ = fit(encoded(size, "PNG"))
        assert decoded(fitted).size == fitted_size

    def test_transparency_is_kept_and_padding_is_transparent(self):
        fitted, image_format = fit(encoded((953, 89), "PNG", mode="RGBA", color=(255, 0, 0, 200)))
        out = decoded(fitted)
        assert (image_format, out.mode, out.size) == ("png", "RGBA", (1568, 280))
        assert out.getpixel((0, 0)) == (255, 255, 255, 0)
        assert out.getpixel((784, 140)) == (255, 0, 0, 200)

    def test_fitting_is_deterministic(self):
        data = encoded((4000, 3000), "JPEG")
        assert fit(data) == fit(data)


class TestReEncoding:
    @pytest.mark.parametrize("image_format", ["GIF", "WEBP"])
    def test_gif_and_webp_become_png(self, image_format):
        lossless: dict[str, Any] = {"lossless": True} if image_format == "WEBP" else {}
        fitted, sent_format = fit(encoded((400, 300), image_format, **lossless))
        out = decoded(fitted)
        assert (sent_format, out.format, out.size) == ("png", "PNG", (400, 300))
        assert out.convert("RGB").getpixel((200, 150)) == RED

    @pytest.mark.parametrize("image_format", ["GIF", "WEBP", "PNG"])
    def test_an_animated_image_sends_its_first_frame(self, image_format):
        lossless = {"lossless": True} if image_format == "WEBP" else {}
        fitted, sent_format = fit(two_frames(image_format, **lossless))
        out = decoded(fitted)
        assert (sent_format, out.format) == ("png", "PNG")
        assert not getattr(out, "is_animated", False)
        assert out.convert("RGB").getpixel((150, 150)) == RED

    def test_a_progressive_jpeg_becomes_baseline(self):
        fitted, image_format = fit(encoded((400, 300), "JPEG", progressive=True))
        out = decoded(fitted)
        assert (image_format, out.format, out.mode, out.size) == ("jpeg", "JPEG", "RGB", (400, 300))
        assert not out.info.get("progressive")

    def test_a_cmyk_jpeg_becomes_rgb(self):
        fitted, image_format = fit(encoded((400, 300), "JPEG", mode="CMYK", color=(0, 0, 0, 0)))
        out = decoded(fitted)
        assert (image_format, out.format, out.mode) == ("jpeg", "JPEG", "RGB")

    def test_a_multi_picture_jpeg_sends_its_first_picture(self):
        data = two_frames("MPO")
        assert decoded(data).format == "MPO"
        fitted, image_format = fit(data)
        out = decoded(fitted)
        assert (image_format, out.format) == ("jpeg", "JPEG")
        pixel = out.getpixel((150, 150))
        assert isinstance(pixel, tuple)
        red, _, blue = pixel
        assert red > 200 and blue < 50  # the red picture (JPEG is lossy)

    def test_the_exif_orientation_is_applied(self):
        exif = pil_image.Exif()
        exif[ExifTags.Base.Orientation] = 6  # rotated 90 degrees
        fitted, image_format = fit(encoded((400, 300), "JPEG", exif=exif))
        out = decoded(fitted)
        assert (image_format, out.size) == ("jpeg", (300, 400))
        assert out.getexif().get(ExifTags.Base.Orientation, 1) == 1

    def test_an_envelope_without_png_gets_a_jpeg_flattened_onto_white(self):
        fitted, image_format = fit(
            encoded((400, 300), "PNG", mode="RGBA", color=(0, 0, 0, 0)), image_formats=["jpeg"]
        )
        out = decoded(fitted)
        assert (image_format, out.format, out.mode) == ("jpeg", "JPEG", "RGB")
        pixel = out.getpixel((200, 150))
        assert isinstance(pixel, tuple)
        assert all(channel > 250 for channel in pixel)


class TestDeepSamplesAndOtherSources:
    """Samples deeper than 8 bits are scaled, not clipped, and a palette or 1-bit image is
    sent as a PNG whatever its source. ``Image`` holds PNG, JPEG, GIF or WebP only, so the
    TIFF and BMP sources are admitted here by widening its format table: the fitting rule
    goes by mode, not by what ``Image`` accepts today."""

    @pytest.fixture
    def tiff_and_bmp(self, monkeypatch):
        monkeypatch.setitem(content_module._MEDIA_TYPES, "TIFF", "image/tiff")
        monkeypatch.setitem(content_module._MEDIA_TYPES, "BMP", "image/bmp")

    def test_a_16_bit_image_is_scaled_to_8_bits(self):
        fitted, image_format = fit(encoded((100, 100), "PNG", mode="I;16", color=32768))
        out = decoded(fitted)
        assert (image_format, out.mode, out.size) == ("png", "L", (280, 280))
        pixel = out.getpixel((140, 140))
        assert isinstance(pixel, int)
        assert abs(pixel - 128) <= 2

    @pytest.mark.usefixtures("tiff_and_bmp")
    @pytest.mark.parametrize(("value", "expected"), [(128.0, 128), (300.0, 255), (-5.0, 0)])
    def test_a_floating_point_image_is_clamped_to_8_bits(self, value, expected):
        fitted, image_format = fit(encoded((300, 300), "TIFF", mode="F", color=value))
        out = decoded(fitted)
        assert (image_format, out.mode) == ("jpeg", "L")
        pixel = out.getpixel((150, 150))
        assert isinstance(pixel, int)
        assert abs(pixel - expected) <= 2

    @pytest.mark.usefixtures("tiff_and_bmp")
    @pytest.mark.parametrize(("mode", "color"), [("P", 3), ("1", 1)])
    def test_a_palette_or_1_bit_image_is_sent_as_a_png(self, mode, color):
        fitted, image_format = fit(encoded((300, 300), "BMP", mode=mode, color=color))
        assert (image_format, decoded(fitted).format) == ("png", "PNG")

    def test_an_image_that_scaling_would_barely_change_is_padded_without_resampling(self):
        width, height = 1567, 279
        source = pil_image.frombytes(
            "L", (width, height), bytes((x + y) % 256 for y in range(height) for x in range(width))
        )
        buffer = io.BytesIO()
        source.save(buffer, "PNG")
        fitted, _ = fit(buffer.getvalue())
        out = decoded(fitted)
        assert (out.mode, out.size) == ("L", (1567, 280))
        assert out.crop((0, 0, width, height)).tobytes() == source.tobytes()  # rows 0-278
        assert set(out.crop((0, height, width, height + 1)).tobytes()) == {255}  # white


class TestFittingFailures:
    def test_an_image_that_cannot_be_decoded_is_refused_by_name(self):
        data = encoded((400, 300), "PNG")
        truncated = Image(data[: len(data) // 2])
        with pytest.raises(ValueError, match=r"Image\(<\d+ bytes, image/png>\) cannot be sent"):
            _fit_for_system_one(truncated)

    def test_a_file_backed_image_is_named_by_its_path(self, tmp_path):
        data = encoded((400, 300), "PNG")
        path = tmp_path / "broken.png"
        path.write_bytes(data[:100])
        with pytest.raises(ValueError, match="broken.png"):
            _fit_for_system_one(Image(path))

    def test_an_image_still_over_the_byte_limit_is_refused(self):
        with pytest.raises(ValueError, match=r"still \d+ bytes, over the 100-byte limit"):
            fit(encoded((4000, 3000), "JPEG"), max_image_bytes=100)


# =============================================================================
# The client: assets, header, cache key
# =============================================================================


class TestPrepareAssets:
    def test_assets_are_sdk_images_under_the_references(self, make_client):
        tiny = Image(encoded((100, 100), "PNG"))
        images = state_images([FIGURE, tiny])
        assets = make_client().prepare_assets(images)
        assert list(assets) == list(images)
        assert all(isinstance(asset, typesafe_sdk.Image) for asset in assets.values())
        assert assets[reference(FIGURE)].data == FIGURE.read_bytes()  # passed through
        assert assets[reference(tiny)].data == fit(tiny.read_bytes())[0]  # fitted

    def test_no_images_need_no_sdk_image(self, make_client, monkeypatch):
        client = make_client()
        monkeypatch.delattr(typesafe_sdk, "Image")
        assert client.prepare_assets({}) == {}

    def test_an_sdk_without_image_raises_an_actionable_import_error(self, make_client, monkeypatch):
        client = make_client()
        monkeypatch.delattr(typesafe_sdk, "Image")
        with pytest.raises(
            ImportError, match=r"typesafe-sdk 0\.7\.5 or later.*autorubric\[typesafe\]"
        ):
            client.prepare_assets(state_images([FIGURE]))


TEXT_STATE = {"input": "Explain photosynthesis.", "submission": "Plants turn light into sugar."}
TEXT_CACHE_KEY = "956ba06dc3503fc62fec13591b0b43377b5ddf4ce5d0b1002dba9e32c657589b"
"""The cache key of ``TEXT_STATE`` and ``QUESTIONS`` for ``jev-latest`` at TypeSafe's API,
captured from the client before image support."""


class TestRequests:
    @pytest.mark.asyncio
    async def test_a_text_only_request_is_unchanged(self, fake_sdk, make_client):
        client = make_client()
        await client.system_one(TEXT_STATE, QUESTIONS)
        await client.system_one(TEXT_STATE, QUESTIONS, assets={})
        assert [kwargs for _, _, kwargs in fake_sdk.calls] == [{}, {}]
        assert client._cache_key(TEXT_STATE, QUESTIONS) == TEXT_CACHE_KEY
        assert client._cache_key(TEXT_STATE, QUESTIONS, {}) == TEXT_CACHE_KEY

    @pytest.mark.asyncio
    async def test_a_request_with_images_carries_its_assets_and_the_header(
        self, fake_sdk, make_client
    ):
        client = make_client(extra_headers={"X-Team": "evals"})
        state = build_state([FIGURE, CAPTION])
        assets = client.prepare_assets(state_images([FIGURE, CAPTION]))
        await client.system_one(state, QUESTIONS, assets=assets)

        ((sent_state, sent_questions, kwargs),) = fake_sdk.calls
        assert (sent_state, sent_questions) == (state, QUESTIONS)
        assert kwargs == {
            "assets": assets,
            "extra_headers": {"TypeSafe-Danger-Zone-Images": "true"},
        }
        assert _IMAGES_HEADER == "TypeSafe-Danger-Zone-Images"
        # The configured headers still go on every request, as the SDK client's own.
        (sdk_client,) = fake_sdk.clients
        assert sdk_client.kwargs["headers"] == {"X-Team": "evals"}

    def test_the_cache_key_of_a_request_with_images_adds_its_sorted_references(self, make_client):
        client = make_client()
        state = build_state([OTHER, FIGURE])
        assets = {reference(OTHER): object(), reference(FIGURE): object()}
        parts = [
            "jev-latest",
            "https://api.typesafe.ai",
            state,
            QUESTIONS,
            {"assets": sorted([reference(OTHER), reference(FIGURE)])},
        ]
        expected = hashlib.sha256(_canonical_json(parts).encode("utf-8")).hexdigest()
        assert client._cache_key(state, QUESTIONS, assets) == expected

    def test_documents_with_different_image_bytes_have_different_keys(self, make_client):
        client = make_client()
        keys = set()
        for image in (FIGURE, OTHER):
            document = [image, CAPTION]
            keys.add(client._cache_key(build_state(document), QUESTIONS, state_images(document)))
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_a_cached_request_with_images_is_not_sent_again(self, fake_sdk, make_client):
        client = make_client(cache_enabled=True)
        state = build_state([FIGURE, CAPTION])
        assets = client.prepare_assets(state_images([FIGURE]))
        first = await client.system_one(state, QUESTIONS, assets=assets)
        again = await client.system_one(state, QUESTIONS, assets=assets)
        assert len(fake_sdk.calls) == 1
        assert again.model_dump() == first.model_dump()

    @pytest.mark.parametrize("name", ["TypeSafe-Danger-Zone-Images", "typesafe-danger-zone-images"])
    def test_the_images_header_cannot_be_configured(self, name):
        with pytest.raises(ValueError, match="autorubric sets that header itself"):
            DecisionModelClient(dm(extra_headers={name: "true"}))


# =============================================================================
# The grader: assets fitted before the request, raising rather than routing
# =============================================================================


class TestGrading:
    @pytest.mark.asyncio
    async def test_a_document_is_graded_with_one_request_carrying_its_assets(
        self, fake_sdk, make_grader, monkeypatch
    ):
        tiny = Image(encoded((100, 100), "PNG"))
        grader = make_grader()
        threads: list[threading.Thread] = []
        real = DecisionModelClient.prepare_assets

        def recording(self: DecisionModelClient, images: Any) -> Any:
            threads.append(threading.current_thread())
            return real(self, images)

        monkeypatch.setattr(DecisionModelClient, "prepare_assets", recording)
        report = await Rubric([LEGIBLE]).grade([tiny, CAPTION], grader=grader)

        assert report.score == 1.0
        ((state, _, kwargs),) = fake_sdk.calls
        assert state == {"submission": reference(tiny) + CAPTION}
        assert list(kwargs["assets"]) == [reference(tiny)]
        assert kwargs["assets"][reference(tiny)].data == fit(tiny.read_bytes())[0]
        assert kwargs["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
        (thread,) = threads
        assert thread is not threading.main_thread()  # fitted off the event loop

    @pytest.mark.asyncio
    async def test_an_image_that_cannot_be_fitted_raises_from_the_grade(
        self, fake_sdk, make_grader
    ):
        data = encoded((400, 300), "PNG")
        broken = Image(data[: len(data) // 2])
        with pytest.raises(ValueError, match="cannot be sent to a decision model"):
            await Rubric([LEGIBLE]).grade([broken, CAPTION], grader=make_grader())
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_an_sdk_without_image_raises_from_the_grade(
        self, fake_sdk, make_grader, monkeypatch
    ):
        grader = make_grader()
        monkeypatch.delattr(typesafe_sdk, "Image")
        with pytest.raises(ImportError, match=r"typesafe-sdk 0\.7\.5 or later"):
            await Rubric([LEGIBLE]).grade([FIGURE, CAPTION], grader=grader)
        assert fake_sdk.calls == []

    @pytest.mark.asyncio
    async def test_text_grading_needs_no_sdk_image(self, fake_sdk, make_grader, monkeypatch):
        grader = make_grader()
        monkeypatch.delattr(typesafe_sdk, "Image")
        await Rubric([LEGIBLE]).grade("Plain text.", grader=grader)
        ((_, _, kwargs),) = fake_sdk.calls
        assert kwargs == {}


# =============================================================================
# The wire: the real SDK client's multipart request
# =============================================================================


def multipart_parts(content_type: str, body: bytes) -> list[tuple[dict[str, str], bytes]]:
    """The parts of a ``multipart/form-data`` body: each part's headers and content."""
    boundary = content_type.split("boundary=")[1].encode()
    parts = []
    for chunk in body.split(b"--" + boundary)[1:-1]:
        head, _, content = chunk[2:-2].partition(b"\r\n\r\n")
        headers = dict(line.split(": ", 1) for line in head.decode().split("\r\n"))
        parts.append(({k.lower(): v for k, v in headers.items()}, content))
    return parts


def request_body(request: Any) -> bytes:
    """The request's body, decompressed when the SDK compressed it (zstd, when available)."""
    if request.headers.get("content-encoding") == "zstd":
        # The SDK compresses with the zstd it can import: Python 3.14's, else the backport.
        zstd = importlib.import_module(
            "compression.zstd" if sys.version_info >= (3, 14) else "backports.zstd"
        )
        return zstd.decompress(request.content)
    return request.content


class TestWire:
    @pytest.mark.asyncio
    async def test_the_sdk_sends_the_state_with_its_assets_as_parts(self, http_endpoint):
        http_endpoint.reply(
            200,
            {
                "model": "jev",
                "usage": {"input_tokens": 1, "output_tokens": 0},
                "answers": {"c0": {"type": "noul", "noul": 0.9}},
            },
        )
        client = DecisionModelClient(dm(extra_headers={"X-Team": "evals"}))
        document = [FIGURE, CAPTION]
        state = build_state(document)
        assets = await asyncio.to_thread(client.prepare_assets, state_images(document))
        await client.system_one(state, {"c0": Noul(instructions="Legible")}, assets=assets)

        (request,) = http_endpoint.requests
        assert request.headers["typesafe-danger-zone-images"] == "true"
        assert request.headers["x-team"] == "evals"
        (request_part, request_json), (asset_part, asset_bytes) = multipart_parts(
            request.headers["content-type"], request_body(request)
        )
        assert request_part["content-disposition"] == 'form-data; name="request"'
        assert json.loads(request_json) == {
            "state": {"submission": reference(FIGURE) + CAPTION},
            "model": "jev-latest",
            "questions": {"c0": {"type": "noul", "instructions": "Legible"}},
            "assets": {reference(FIGURE): {"type": "image", "part": "asset-0"}},
        }
        assert asset_part["content-disposition"] == 'form-data; name="asset-0"; filename="asset-0"'
        assert asset_part["content-type"] == "image/png"
        assert asset_bytes == FIGURE.read_bytes()

    @pytest.mark.asyncio
    async def test_a_text_only_request_is_json_without_the_header(self, http_endpoint):
        http_endpoint.reply(
            200, {"model": "jev", "usage": {}, "answers": {"c0": {"type": "noul", "noul": 0.9}}}
        )
        client = DecisionModelClient(dm())
        await client.system_one(TEXT_STATE, {"c0": Noul(instructions="Legible")})
        (request,) = http_endpoint.requests
        assert "typesafe-danger-zone-images" not in request.headers
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request_body(request))["state"] == TEXT_STATE
