"""Images inside LLM user prompts.

Every user-prompt builder in ``autorubric.prompts`` (the four per-criterion builders and
``build_rubric_user_prompt``, the prompt of ``llm_calls="per_item"``) and every few-shot
example formatter assembles its prompt with ``_concat``. An image given in a submission, a
query, a reference submission or a few-shot example's submission therefore sits in the prompt
exactly where its text would: put a sentinel string in place of each image part, join the
parts, and the result is the prompt built from text with the sentinel written at that place.
The tags around it are the ones a text prompt has, and the image parts are the very ``Image``
objects that were given.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import PIL.Image as pil_image
import pytest

from autorubric import Criterion, CriterionOption, CriterionVerdict, FewShotExample, Image
from autorubric.content import Content, _content_text, _normalize_content
from autorubric.prompts import (
    _format_rubric_examples,
    build_few_shot_user_prompt,
    build_multi_choice_few_shot_user_prompt,
    build_multi_choice_user_prompt,
    build_rubric_user_prompt,
    build_user_prompt,
)


def encoded_png(color: str) -> bytes:
    """A tiny PNG of one ``color``, encoded by Pillow."""
    buffer = io.BytesIO()
    pil_image.new("RGB", (2, 2), color).save(buffer, "PNG")
    return buffer.getvalue()


RED, BLUE, GREEN, GRAY = (Image(encoded_png(c)) for c in ("red", "blue", "green", "gray"))
SENTINEL = "<<IMG>>"

BINARY = Criterion(name="figure", weight=2.0, requirement="The figure shows a rising trend")
ORDINAL = Criterion(
    name="legibility",
    weight=1.0,
    requirement="How legible is the chart?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Illegible", value=0.0),
        CriterionOption(label="Legible", value=1.0),
        CriterionOption(label="Cannot assess", value=0.0, na=True),
    ],
)
GUIDELINES = "Judge the chart, not the caption."
SUBMISSION = "Accuracy rises with model size."
QUERY = "Plot accuracy against model size."
REFERENCE = "A line chart of accuracy rising from 60% to 90%."
EXAMPLE = "Accuracy is flat."

FIELDS = ("submission", "query", "reference", "example")


def _build_user(fields: dict[str, Any]) -> Content:
    return build_user_prompt(
        BINARY, fields["submission"], fields["query"], fields["reference"], guidelines=GUIDELINES
    )


def _build_few_shot(fields: dict[str, Any]) -> Content:
    return build_few_shot_user_prompt(
        criterion=BINARY,
        to_grade=fields["submission"],
        examples=[
            FewShotExample(fields["example"], CriterionVerdict.UNMET, "No trend."),
            FewShotExample("Accuracy rises.", CriterionVerdict.MET, None),
        ],
        query=fields["query"],
        include_reason=True,
        reference_submission=fields["reference"],
        guidelines=GUIDELINES,
    )


def _build_multi_choice(fields: dict[str, Any]) -> Content:
    return build_multi_choice_user_prompt(
        ORDINAL, fields["submission"], fields["query"], fields["reference"], guidelines=GUIDELINES
    )


def _build_multi_choice_few_shot(fields: dict[str, Any]) -> Content:
    return build_multi_choice_few_shot_user_prompt(
        criterion=ORDINAL,
        to_grade=fields["submission"],
        examples=[(fields["example"], 0, "Unreadable axes."), ("A clean chart.", 1, None)],
        query=fields["query"],
        include_reason=True,
        reference_submission=fields["reference"],
        guidelines=GUIDELINES,
    )


def _build_rubric(fields: dict[str, Any]) -> Content:
    return build_rubric_user_prompt(
        [("c0", BINARY), ("c1", ORDINAL)],
        fields["submission"],
        fields["query"],
        fields["reference"],
        guidelines=GUIDELINES,
    )


def _build_rubric_with_examples(fields: dict[str, Any]) -> Content:
    examples_text = _format_rubric_examples(
        [
            (
                fields["example"],
                [
                    ("c0", BINARY, CriterionVerdict.UNMET, "No trend."),
                    ("c1", ORDINAL, 0, None),
                ],
            ),
            ("A clean chart.", [("c0", BINARY, CriterionVerdict.MET, None)]),
        ],
        include_reason=True,
    )
    return build_rubric_user_prompt(
        [("c0", BINARY), ("c1", ORDINAL)],
        fields["submission"],
        fields["query"],
        fields["reference"],
        examples_text=examples_text,
        guidelines=GUIDELINES,
    )


# Each builder and the fields it puts in the prompt.
BUILDERS: dict[str, tuple[Callable[[dict[str, Any]], Content], tuple[str, ...]]] = {
    "user": (_build_user, FIELDS[:3]),
    "few_shot": (_build_few_shot, FIELDS),
    "multi_choice": (_build_multi_choice, FIELDS[:3]),
    "multi_choice_few_shot": (_build_multi_choice_few_shot, FIELDS),
    "rubric": (_build_rubric, FIELDS[:3]),
    "rubric_with_examples": (_build_rubric_with_examples, FIELDS),
}
TEXT_FIELDS: dict[str, Any] = {
    "submission": SUBMISSION,
    "query": QUERY,
    "reference": REFERENCE,
    "example": EXAMPLE,
}

# Content holding images, the text it reads as with the sentinel for each image, and its
# images in order.
FORMS: dict[str, tuple[Content, str, tuple[Image, ...]]] = {
    "inline": (["Before ", RED, " after."], f"Before {SENTINEL} after.", (RED,)),
    "adjacent": ([RED, BLUE], SENTINEL * 2, (RED, BLUE)),
    "whole": (RED, SENTINEL, (RED,)),
}


def _with_sentinels(prompt: Content) -> str:
    """``prompt``'s parts joined, with ``SENTINEL`` in place of each image."""
    assert isinstance(prompt, tuple)
    return _content_text(prompt, image_marker=SENTINEL)


def _image_parts(prompt: Content) -> list[Any]:
    assert isinstance(prompt, tuple)
    return [part for part in prompt if isinstance(part, Image)]


CASES = [
    (builder, field, form)
    for builder, (_, fields) in BUILDERS.items()
    for field in fields
    for form in FORMS
]


@pytest.mark.parametrize(("builder", "field", "form"), CASES)
def test_an_image_sits_where_its_text_would(builder: str, field: str, form: str) -> None:
    """With images in one field, the prompt is a tuple of parts that reads as the text
    prompt with a sentinel where each image is, and holds the given images themselves."""
    build, _ = BUILDERS[builder]
    content, as_text, images = FORMS[form]
    prompt = build({**TEXT_FIELDS, field: content})
    assert isinstance(prompt, tuple)
    # Canonical: no empty strings and no two strings in a row.
    assert _normalize_content(prompt) == prompt
    assert _with_sentinels(prompt) == build({**TEXT_FIELDS, field: as_text})
    parts = _image_parts(prompt)
    assert len(parts) == len(images)
    assert all(part is image for part, image in zip(parts, images, strict=True))


@pytest.mark.parametrize("builder", sorted(BUILDERS))
def test_text_alone_builds_a_string(builder: str) -> None:
    """Without images every builder returns a ``str``."""
    build, _ = BUILDERS[builder]
    assert isinstance(build(TEXT_FIELDS), str)


# The builders of a per-criterion prompt without examples, which puts what an item's calls
# share first (the guidelines, input, reference submission and submission) and the criterion
# after it. A few-shot per-criterion prompt puts what a criterion's calls share across items
# first (the guidelines, examples and criterion) and, like ``build_rubric_user_prompt``,
# keeps the examples ahead of the input and ends with the submission.
SHARED_ITEM_FIRST = {"user", "multi_choice"}


@pytest.mark.parametrize("builder", sorted(BUILDERS))
def test_a_submission_that_is_one_image_is_the_whole_submission_block(builder: str) -> None:
    """A bare image as the submission is the only thing inside ``<submission>``."""
    build, _ = BUILDERS[builder]
    prompt = build({**TEXT_FIELDS, "submission": RED})
    assert isinstance(prompt, tuple)
    assert len(prompt) == 3
    head, image, tail = prompt
    assert isinstance(head, str)
    assert head.endswith("<submission>\n")
    assert image is RED
    if builder in SHARED_ITEM_FIRST:
        # The submission block closes the shared start; what follows it is the text
        # prompt's own ending.
        text_prompt = build({**TEXT_FIELDS, "submission": SENTINEL})
        assert isinstance(text_prompt, str)
        assert isinstance(tail, str)
        assert tail == text_prompt.split(SENTINEL, 1)[1]
        assert tail.startswith("\n</submission>\n\n")
    else:
        assert tail == "\n</submission>"


@pytest.mark.parametrize("builder", sorted(BUILDERS))
def test_images_in_every_field_keep_the_prompt_order(builder: str) -> None:
    """Images in several fields appear in the order the prompt reads: in a per-criterion
    prompt without examples the input, then the reference submission, then the submission;
    in a few-shot or whole-rubric prompt the examples first, then the other three."""
    build, fields = BUILDERS[builder]
    given = {"example": GRAY, "query": GREEN, "reference": BLUE, "submission": RED}
    prompt = build({**TEXT_FIELDS, **{field: given[field] for field in fields}})
    order = (
        ("query", "reference", "submission")
        if builder in SHARED_ITEM_FIRST
        else ("example", "query", "reference", "submission")
    )
    expected = [given[field] for field in order if field in fields]
    parts = _image_parts(prompt)
    assert len(parts) == len(expected)
    assert all(part is image for part, image in zip(parts, expected, strict=True))
    assert _with_sentinels(prompt) == build(
        {**TEXT_FIELDS, **{field: SENTINEL for field in fields}}
    )


@pytest.mark.parametrize("empty", [[], [""], ("", "")], ids=["empty_list", "one_empty", "two"])
@pytest.mark.parametrize("builder", sorted(BUILDERS))
def test_an_empty_query_or_reference_is_none(builder: str, empty: Content) -> None:
    """A query or reference submission holding no text and no image adds no block, as
    ``""`` and ``None`` add none."""
    build, _ = BUILDERS[builder]
    without = build({**TEXT_FIELDS, "query": None, "reference": None})
    assert build({**TEXT_FIELDS, "query": empty, "reference": empty}) == without
    assert build({**TEXT_FIELDS, "query": "", "reference": ""}) == without
