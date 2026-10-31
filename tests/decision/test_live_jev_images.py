"""Live tests: image requests to the real TypeSafe System One API (Jev).

These tests send real requests and spend the account's credit, about eight requests per
run. They run only when both hold:

- pytest is given ``--live`` (registered in ``tests/conftest.py``, which skips every test
  marked ``live`` without it, however the test was selected, ``-m live`` included);
- ``TYPESAFE_API_KEY`` is set, in the environment or in the repository's ``.env``, which
  autorubric loads at import.

Otherwise every test here is skipped, with a reason saying which condition is missing, so
``uv run pytest`` stays offline and free. CI (``.github/workflows/ci.yaml``) runs plain
``uv run pytest`` and never passes ``--live``. To run them, with the account owner's
go-ahead for each run::

    uv run pytest --live tests/decision/test_live_jev_images.py

A live test uses the developer's own TypeSafe configuration (``TYPESAFE_API_KEY`` and,
when set, ``TYPESAFE_BASE_URL``): the decision-model conftest's network and environment
guards stand aside for it. The response cache is off, so every run asks the API again.
Images are synthetic charts drawn with Pillow when a test runs. Assertions are on the
shape of what comes back (judged, no error, a vote per criterion with probabilities, a
verdict or option of the rubric), never on probabilities, which the model chooses.

What each test proves:

1. ``test_a_text_only_item_is_judged``: a text-only item graded by
   ``typesafe/jev-1.14.0`` is judged, and the request carries no assets and no images
   header, so text requests are unchanged by image support.
2. ``test_an_image_document_is_judged``: ``[Image, caption]`` on a binary and a 5-level
   ordinal criterion is judged, with one vote per criterion, and the request with the
   image is billed more input tokens than the same caption alone.
3. ``test_a_strip_outside_the_envelope_is_fitted_and_accepted``: a 953 x 89 strip, whose
   short side is under the model's minimum, is fitted by autorubric and accepted (no 400
   or 413).
4. ``test_two_images_in_one_state_are_accepted`` and
   ``test_an_image_used_twice_is_sent_once_and_accepted``: two images in one state are
   sent as two assets and accepted; one image used twice is one reference, one asset, and
   accepted.
5. ``test_a_model_without_images_refuses_before_any_request``: ``typesafe/jev-1.13.0``
   with an image raises the registry's ``ValueError`` before any request, with the real
   client built (it spends nothing). The offline suite checks the same refusal through
   the fake SDK (``TestImagesAreRefused`` in ``test_decision_models_registry.py``).
6. ``test_an_image_outside_the_envelope_is_rejected_by_the_server``: an image outside the
   envelope, sent unfitted (``prepare_assets`` bypassed), is rejected by the server with a
   400, 413 or 422 (not an authentication, permission, not-found or rate-limit error), which
   ``_classify_typesafe_error`` routes as ``parse`` or ``infrastructure``. It
   documents the server's own limits; if the server accepts the image, its limits have
   changed, and the test is skipped saying so.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Iterator
from typing import Any

import PIL.Image as pil_image
import pytest
import typesafe_sdk
from PIL import ImageDraw, ImageFont
from typesafe_sdk import Noul

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DecisionModelConfig,
    Image,
    Rubric,
)
from autorubric.decision import (
    _IMAGES_HEADER,
    DecisionModelClient,
    _judge_capabilities,
    build_state,
    state_images,
)
from autorubric.graders import CriterionGrader
from autorubric.llm import _classify_typesafe_error
from autorubric.types import CANONICAL_NA_OPTION, EnsembleEvaluationReport

pytestmark = [
    pytest.mark.live,
    # tests/conftest.py skips every live test without --live; this repeats the check here so
    # that a run which does not load that conftest (``--noconftest``) still sends nothing.
    pytest.mark.skipif(
        "not config.getoption('--live', default=False)",
        reason="live test: it calls a real paid API; run it with --live",
    ),
    pytest.mark.skipif(
        not os.environ.get("TYPESAFE_API_KEY", "").strip(),
        reason="TYPESAFE_API_KEY is not set (in the environment or the repository's .env)",
    ),
]

MODEL = "typesafe/jev-1.14.0"
CAPTION = "Figure 1: a labelled chart."
AXES = Criterion(
    name="labelled_axes",
    weight=1.0,
    requirement="The figure shows a plot with both axes labelled",
)
CLARITY = Criterion(
    name="clarity",
    weight=1.0,
    requirement="How clearly the figure presents its data",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Unreadable", value=0.0),
        CriterionOption(label="Hard to read", value=0.25),
        CriterionOption(label="Readable with effort", value=0.5),
        CriterionOption(label="Clear", value=0.75),
        CriterionOption(label="Very clear", value=1.0),
    ],
)
RUBRIC = Rubric([AXES, CLARITY])
CLARITY_LABELS = {option.label for option in CLARITY.options or []} | {CANONICAL_NA_OPTION.label}
"""The options a clarity verdict can name: the rubric's, and the NA option the grader adds."""


# =============================================================================
# Synthetic images
# =============================================================================


def _png(canvas: pil_image.Image) -> bytes:
    buffer = io.BytesIO()
    canvas.save(buffer, "PNG")
    return buffer.getvalue()


def chart_png(size: tuple[int, int], *, kind: str = "line") -> bytes:
    """A chart of accuracy by model size, ``size`` pixels, with titled, labelled axes.

    ``kind`` is ``"line"`` or ``"bar"``; the two draw different images of one layout.
    """
    width, height = size
    canvas = pil_image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(8, min(width, height) // 30))
    left, right = int(width * 0.16), int(width * 0.95)
    top, bottom = int(height * 0.12), int(height * 0.80)
    line = max(1, min(width, height) // 200)
    draw.text((width // 2, height // 40), f"Accuracy by model size ({kind})", "black", font, "ma")
    draw.line([(left, top), (left, bottom), (right, bottom)], "black", line * 2)
    # One slot per model size, its tick at the slot's centre.
    xs = [left + (right - left) * (2 * slot + 1) // 10 for slot in range(5)]
    for tick, x in enumerate(xs):
        y = bottom - (bottom - top) * tick // 4
        draw.line([(x, bottom), (x, bottom + 4 * line)], "black", line)
        draw.text((x, bottom + 6 * line), f"{2**tick}B", "black", font, "ma")
        draw.line([(left - 4 * line, y), (left, y)], "black", line)
        draw.text((left - 6 * line, y), f"{25 * tick}%", "black", font, "rm")
    draw.text(
        ((left + right) // 2, height - height // 40), "Model size (parameters)", "black", font, "md"
    )
    draw.text((width // 80, top - height // 30), "Accuracy (%)", "black", font, "ld")
    accuracies = [0.30, 0.45, 0.62, 0.71, 0.78]
    points = [(x, bottom - int((bottom - top) * a)) for x, a in zip(xs, accuracies, strict=True)]
    if kind == "line":
        draw.line(points, (31, 119, 180), line * 3)
        for x, y in points:
            draw.ellipse([x - 4 * line, y - 4 * line, x + 4 * line, y + 4 * line], (31, 119, 180))
    else:
        half = (right - left) // 30
        for x, y in points:
            draw.rectangle([x - half, y, x + half, bottom - line], (255, 127, 14))
    return _png(canvas)


def strip_png(size: tuple[int, int] = (953, 89)) -> bytes:
    """A chart legend cut out as a wide strip (953 x 89 by default): three labelled series."""
    width, height = size
    canvas = pil_image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(8, height // 4))
    series = [
        ("baseline", (127, 127, 127)),
        ("ours", (31, 119, 180)),
        ("ours + images", (255, 127, 14)),
    ]
    draw.text((width // 60, height // 2), "Legend:", "black", font, "lm")
    for i, (label, color) in enumerate(series):
        x = width * (i + 1) // 4
        draw.line([(x, height // 2), (x + width // 16, height // 2)], color, max(2, height // 15))
        draw.text((x + width // 16 + width // 80, height // 2), label, "black", font, "lm")
    return _png(canvas)


@pytest.fixture
def figure() -> Image:
    """An 800 x 600 line chart, inside Jev's envelope: sent as it is."""
    return Image(chart_png((800, 600)))


@pytest.fixture
def other_figure() -> Image:
    """An 800 x 600 bar chart: a second image, distinct from ``figure``."""
    return Image(chart_png((800, 600), kind="bar"))


@pytest.fixture
def strip() -> Image:
    """A 953 x 89 strip, whose short side is under Jev's minimum: fitted before sending."""
    return Image(strip_png())


# =============================================================================
# Clients, and what they send
# =============================================================================


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Each request handed to the SDK, in order: its asset references and extra headers.

    The request then goes on to the API unchanged. Recording happens before it is sent, so
    a request the server rejects is recorded too, and a test can tell that none was sent.
    """
    real = typesafe_sdk.AsyncTypeSafeClient.system_one
    requests: list[dict[str, Any]] = []

    async def recording(self: Any, *args: Any, **kwargs: Any) -> Any:
        requests.append(
            {
                "assets": sorted(kwargs.get("assets") or {}),
                "extra_headers": dict(kwargs.get("extra_headers") or {}),
            }
        )
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(typesafe_sdk.AsyncTypeSafeClient, "system_one", recording)
    return requests


def live_config(model: str = MODEL) -> DecisionModelConfig:
    """A real decision-model config: key and base URL from the environment, no cache."""
    return DecisionModelConfig(model=model, cache_enabled=False)


@pytest.fixture
def make_grader() -> Iterator[Callable[..., CriterionGrader]]:
    """Build single decision-model graders whose clients are closed at teardown."""
    graders: list[CriterionGrader] = []

    def build(model: str = MODEL) -> CriterionGrader:
        grader = CriterionGrader(judge_model_config=live_config(model))
        graders.append(grader)
        return grader

    yield build
    for grader in graders:
        for client in grader._decision_clients.values():
            client.close()


@pytest.fixture
def live_client() -> Iterator[DecisionModelClient]:
    client = DecisionModelClient(live_config())
    yield client
    client.close()


def assert_judged(report: EnsembleEvaluationReport) -> None:
    """The report judges both criteria: one answered vote each, a verdict of the rubric."""
    assert report.error is None
    assert report.score is not None
    assert report.token_usage is not None and report.token_usage.prompt_tokens > 0
    assert report.report is not None
    axes, clarity = report.report
    assert axes.error is None
    assert axes.final_verdict in (CriterionVerdict.MET, CriterionVerdict.UNMET)
    assert len(axes.votes) == 1
    assert axes.votes[0].error is None
    assert axes.votes[0].probabilities
    assert axes.votes[0].confidence is not None
    assert clarity.error is None
    assert clarity.final_multi_choice_verdict is not None
    assert clarity.final_multi_choice_verdict.selected_label in CLARITY_LABELS
    assert len(clarity.multi_choice_votes) == 1
    assert clarity.multi_choice_votes[0].error is None
    assert clarity.multi_choice_votes[0].probabilities
    assert clarity.multi_choice_votes[0].confidence is not None


NO_IMAGES = {"assets": [], "extra_headers": {}}
WITH_IMAGES = {_IMAGES_HEADER: "true"}


# =============================================================================
# Cases
# =============================================================================


@pytest.mark.asyncio
async def test_a_text_only_item_is_judged(make_grader, sent):
    report = await RUBRIC.grade(CAPTION, grader=make_grader())
    assert_judged(report)
    assert sent == [NO_IMAGES]


@pytest.mark.asyncio
async def test_an_image_document_is_judged(make_grader, sent, figure):
    grader = make_grader()
    text_report = await RUBRIC.grade(CAPTION, grader=grader)
    image_report = await RUBRIC.grade([figure, CAPTION], grader=grader)
    assert_judged(text_report)
    assert_judged(image_report)
    assert text_report.token_usage is not None and image_report.token_usage is not None
    assert image_report.token_usage.prompt_tokens > text_report.token_usage.prompt_tokens
    assert sent == [
        NO_IMAGES,
        {"assets": [f"<image:{figure.digest}>"], "extra_headers": WITH_IMAGES},
    ]


@pytest.mark.asyncio
async def test_a_strip_outside_the_envelope_is_fitted_and_accepted(make_grader, sent, strip):
    report = await RUBRIC.grade([strip, CAPTION], grader=make_grader())
    assert_judged(report)
    assert sent == [{"assets": [f"<image:{strip.digest}>"], "extra_headers": WITH_IMAGES}]


@pytest.mark.asyncio
async def test_two_images_in_one_state_are_accepted(make_grader, sent, figure, other_figure):
    report = await RUBRIC.grade([figure, "and", other_figure], grader=make_grader())
    assert_judged(report)
    expected = sorted([f"<image:{figure.digest}>", f"<image:{other_figure.digest}>"])
    assert sent == [{"assets": expected, "extra_headers": WITH_IMAGES}]


@pytest.mark.asyncio
async def test_an_image_used_twice_is_sent_once_and_accepted(make_grader, sent, figure):
    document = [figure, "twice", figure]
    reference = f"<image:{figure.digest}>"
    assert list(state_images(document)) == [reference]
    assert build_state(document)["submission"] == f"{reference}twice{reference}"
    report = await RUBRIC.grade(document, grader=make_grader())
    assert_judged(report)
    assert sent == [{"assets": [reference], "extra_headers": WITH_IMAGES}]


@pytest.mark.asyncio
async def test_a_model_without_images_refuses_before_any_request(make_grader, sent, figure):
    grader = make_grader("typesafe/jev-1.13.0")
    with pytest.raises(ValueError, match="does not accept images; use jev-1.14.0 or later"):
        await RUBRIC.grade([figure, CAPTION], grader=grader)
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size", [(3000, 3000), (100, 100)], ids=["over-the-largest-side", "under-the-smallest-side"]
)
async def test_an_image_outside_the_envelope_is_rejected_by_the_server(live_client, sent, size):
    document = [Image(chart_png(size)), CAPTION]
    # The images as given, not fitted: ``prepare_assets`` would bring them into the envelope.
    assets = {
        reference: typesafe_sdk.Image(image.read_bytes())
        for reference, image in state_images(document).items()
    }
    question = {"c0": Noul(instructions=AXES.requirement)}
    try:
        await live_client.system_one(
            build_state(document),
            question,
            assets=assets,
            envelope=_judge_capabilities(live_client.config),
        )
    except typesafe_sdk.TypeSafeAPIError as error:
        # A rejection of the image, not of the key, the model or the request rate.
        assert not isinstance(
            error,
            (
                typesafe_sdk.TypeSafeAuthenticationError,
                typesafe_sdk.TypeSafePermissionDeniedError,
                typesafe_sdk.TypeSafeNotFoundError,
                typesafe_sdk.TypeSafeRateLimitError,
            ),
        ), f"not a rejection of the image: {error}"
        assert error.status in (400, 413, 422), f"not a rejection of the image: {error}"
        assert _classify_typesafe_error(error) in ("parse", "infrastructure")
    else:
        pytest.skip(
            f"System One accepted an unfitted {size[0]} x {size[1]} image: its limits have "
            "changed, so check jev-1.14.0's entry in src/autorubric/data/decision_models.json"
        )
    assert sent == [{"assets": sorted(assets), "extra_headers": WITH_IMAGES}]
