"""Few-shot examples identify a training item's submission by its content.

Drawing examples per criterion (``_balanced_pick``), each label's slots are drawn first and
no remaining slot is filled with an item whose submission is already shown; drawing whole
items (``_select_item_examples``, tested with the greedy coverage in
``test_per_item_few_shot.py``), an item whose submission was picked is skipped. A submission's
identity is ``_example_identity``: text is its own identity, never hashed; a document is
identified by its ``_content_digest``, so documents whose images hold the same bytes are one
submission wherever the images are read from. The draws themselves (the RNG and its order)
do not depend on the content, so a pool of documents gives the examples a pool of texts
with the same duplicates gives.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import PIL.Image as pil_image
import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    FewShotConfig,
    Image,
    LLMConfig,
    Rubric,
    RubricDataset,
)
from autorubric.content import _content_digest
from autorubric.graders import CriterionGrader
from autorubric.graders.criterion_grader import _example_identity

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET

BINARY = Criterion(weight=1.0, requirement="Mentions light")
CLARITY = Criterion(
    weight=1.0,
    requirement="How clear is the explanation?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Unclear", value=0.0),
        CriterionOption(label="Mostly clear", value=0.6),
        CriterionOption(label="Very clear", value=1.0),
    ],
)

# (name, text, label): a alone has its label, so once each label has a slot the remaining
# slots are filled from the others; b repeats a's text, so it never fills one, and e repeats
# c's, so it fills none once c is shown.
BINARY_ROWS = [
    ("a", "A", MET),
    ("b", "A", UNMET),
    ("c", "C", UNMET),
    ("d", "D", UNMET),
    ("e", "C", UNMET),
]
MULTI_CHOICE_ROWS = [
    ("a", "A", "Very clear"),
    ("b", "A", "Unclear"),
    ("c", "C", "Unclear"),
    ("d", "D", "Unclear"),
    ("e", "C", "Unclear"),
]
COLORS = {"A": "red", "C": "green", "D": "blue"}


def encoded_png(color: str) -> bytes:
    """A tiny PNG of one ``color``, encoded by Pillow."""
    buffer = io.BytesIO()
    pil_image.new("RGB", (2, 2), color).save(buffer, "PNG")
    return buffer.getvalue()


def unique_texts(rows: list[tuple[str, str, Any]]) -> list[tuple[str, str, Any]]:
    """``rows`` with every text its own: the name appended to it."""
    return [(name, text + name, label) for name, text, label in rows]


def documents(
    rows: list[tuple[str, str, Any]], tmp_path: Path, *, same_images: bool
) -> list[tuple[str, Any, Any]]:
    """``rows`` with each text ``t`` made the document ``[image, t]``.

    With ``same_images`` the image holds the bytes of its text's color, read from a file
    for every other row, so rows sharing a text share a document's content but not its
    sources; otherwise every row's image is its own.
    """
    result = []
    for position, (name, text, label) in enumerate(rows):
        if same_images:
            data = encoded_png(COLORS[text])
            path = tmp_path / f"{name}.png"
            path.write_bytes(data)
            image = Image(path) if position % 2 else Image(data)
        else:
            image = Image(encoded_png(["red", "green", "blue", "gray", "teal"][position]))
        result.append((name, [image, text], label))
    return result


def pool(rubric: Criterion, rows: list[tuple[str, Any, Any]]) -> RubricDataset:
    """A training pool, each item's name its ground-truth reason (which never affects the
    draw), so that an example names its item."""
    dataset = RubricDataset(prompt="How do plants make food?", rubric=Rubric([rubric]))
    for name, submission, label in rows:
        dataset.add_item(submission, name, ground_truth=[label], ground_truth_reasons=[name])
    return dataset


def drawn(rubric: Criterion, rows: list[tuple[str, Any, Any]], n: int, seed: int) -> list[str]:
    """The names of the items a single judge's examples for criterion 0 show, in order."""
    grader = CriterionGrader(
        judge_model_config=LLMConfig(model="test-model"),
        training_data=pool(rubric, rows),
        few_shot_config=FewShotConfig(n_examples=n, balance_verdicts=True, seed=seed),
    )
    if rubric.is_multi_choice:
        return [
            cast(str, reason) for _s, _i, reason in grader._multi_choice_examples[(0, "default")]
        ]
    return [cast(str, example.reason) for example in grader._criterion_examples[(0, "default")]]


SEEDS = range(40)


class TestTheIdentity:
    def test_text_is_its_own_identity(self):
        # A lone surrogate cannot be encoded, so it could not be hashed.
        unencodable = "Plants use light \ud83d to make sugar."
        assert _example_identity(unencodable) is unencodable
        assert _example_identity(["Li", "ght."]) == "Light."

    def test_a_document_is_identified_by_its_digest(self, tmp_path):
        red = encoded_png("red")
        red_file = tmp_path / "red.png"
        red_file.write_bytes(red)
        document = [Image(red), "x"]
        assert _example_identity(document) == ("content_sha256", _content_digest(document))
        assert _example_identity([Image(red_file), "x"]) == _example_identity(document)
        assert _example_identity([Image(encoded_png("blue")), "x"]) != _example_identity(document)
        assert _example_identity(["x", Image(red)]) != _example_identity(document)

    @pytest.mark.parametrize("llm_calls", ["per_criterion", "per_item"])
    def test_training_text_that_cannot_be_encoded_is_drawn(self, llm_calls):
        """Text is never hashed to identify it, so training submissions holding a lone
        surrogate are drawn as the same texts without it are, in either mode."""

        def names_drawn(rows: list[tuple[str, Any, Any]]) -> list[str | None]:
            grader = CriterionGrader(
                judge_model_config=LLMConfig(model="test-model"),
                training_data=pool(BINARY, rows),
                few_shot_config=FewShotConfig(n_examples=4, balance_verdicts=True, seed=3),
                llm_calls=llm_calls,
            )
            if llm_calls == "per_item":
                return [reasons[0] for _s, _l, reasons in grader._item_examples["default"]]
            return [example.reason for example in grader._criterion_examples[(0, "default")]]

        unencodable = [(name, f"{text} \ud83d", label) for name, text, label in BINARY_ROWS]
        assert names_drawn(unencodable) == names_drawn(BINARY_ROWS)

    def test_no_text_is_a_documents_identity(self):
        document = [Image(encoded_png("red")), "x"]
        assert _example_identity(_content_digest(document)) != _example_identity(document)


@pytest.mark.parametrize(
    "rubric, rows",
    [(BINARY, BINARY_ROWS), (CLARITY, MULTI_CHOICE_ROWS)],
    ids=["binary", "multi_choice"],
)
class TestDrawsPerCriterion:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_documents_with_the_same_images_draw_as_their_texts(self, tmp_path, rubric, rows, n):
        same = documents(rows, tmp_path, same_images=True)
        for seed in SEEDS:
            assert drawn(rubric, same, n, seed) == drawn(rubric, rows, n, seed), seed

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_documents_with_their_own_images_draw_as_unique_texts(self, tmp_path, rubric, rows, n):
        own = documents(rows, tmp_path, same_images=False)
        for seed in SEEDS:
            assert drawn(rubric, own, n, seed) == drawn(rubric, unique_texts(rows), n, seed), seed

    def test_a_repeated_submission_is_not_shown_again(self, rubric, rows):
        """The pools differ only in whether b repeats a's submission and e c's: with the
        repeats, a remaining slot is never filled with one, so some draws differ."""
        differing = [
            seed
            for seed in SEEDS
            if drawn(rubric, rows, 4, seed) != drawn(rubric, unique_texts(rows), 4, seed)
        ]
        assert differing
