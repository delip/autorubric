"""Rubric guidelines through ``RubricDataset``: serialization, per-item rubrics, splits.

A rubric with guidelines is written in the dict form ``{"guidelines": ..., "criteria": [...]}``;
a rubric without them keeps the list form, byte-identical to before guidelines existed
(``test_legacy_reserialization.py`` pins that against the unmodified library).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DataItem,
    Rubric,
    RubricDataset,
)
from autorubric.eval import _compute_dataset_hash
from autorubric.types import CriterionReport, EvaluationReport
from autorubric.utils import fill_ground_truth

GLOBAL_GUIDELINES = "Judge against a grade 8-12 level.\nAny attribution counts as a citation."
ITEM_GUIDELINES = "This item's criteria concern the lab report only."

MET, UNMET = CriterionVerdict.MET, CriterionVerdict.UNMET


def _criteria() -> list[Criterion]:
    return [
        Criterion(name="thesis", weight=3.0, requirement="States a clear, arguable thesis"),
        Criterion(
            name="tone",
            weight=2.0,
            requirement="Which tone best describes the essay?",
            scale_type="nominal",
            options=[
                CriterionOption(label="Formal", value=1.0),
                CriterionOption(label="Casual", value=0.0),
                CriterionOption(label="N/A", value=0.0, na=True),
            ],
            aggregation="mode",
        ),
    ]


def _item_criteria() -> list[Criterion]:
    return [Criterion(name="method", weight=1.0, requirement="Describes the method")]


def _dataset(*, global_guidelines: str | None = GLOBAL_GUIDELINES) -> RubricDataset:
    """A global rubric plus one item with its own rubric (with its own guidelines)."""
    dataset = RubricDataset(
        prompt="Write an essay.",
        rubric=Rubric(_criteria(), guidelines=global_guidelines),
        name="guidelines-demo",
    )
    labels = [(MET, "Formal"), (UNMET, "Casual"), (MET, "Casual"), (UNMET, "Formal")]
    for i, gt in enumerate(labels):
        dataset.add_item(submission=f"Essay {i}", description=f"item {i}", ground_truth=list(gt))
    dataset.add_item(
        submission="Lab report",
        description="per-item rubric",
        ground_truth=[MET],
        rubric=Rubric(_item_criteria(), guidelines=ITEM_GUIDELINES),
    )
    return dataset


def _criteria_payload(criteria: list[Criterion]) -> list[dict[str, Any]]:
    """The list form ``_serialize_rubric`` has always written for these criteria."""
    return json.loads(RubricDataset(rubric=Rubric(criteria)).to_json())["rubric"]


class TestSerializeRubric:
    def test_list_form_without_guidelines(self) -> None:
        dataset = RubricDataset()
        payload = dataset._serialize_rubric(Rubric(_criteria()))
        assert isinstance(payload, list)
        assert payload == _criteria_payload(_criteria())

    def test_dict_form_with_guidelines(self) -> None:
        dataset = RubricDataset()
        payload = dataset._serialize_rubric(Rubric(_criteria(), guidelines=GLOBAL_GUIDELINES))
        assert payload == {
            "guidelines": GLOBAL_GUIDELINES,
            "criteria": _criteria_payload(_criteria()),
        }
        assert list(payload) == ["guidelines", "criteria"]

    def test_to_json_writes_dict_forms_only_where_guidelines_exist(self) -> None:
        dataset = _dataset()
        dataset.add_item(
            submission="Second lab report",
            description="per-item rubric without guidelines",
            rubric=Rubric(_item_criteria()),
        )
        data = json.loads(dataset.to_json())
        assert data["rubric"] == {
            "guidelines": GLOBAL_GUIDELINES,
            "criteria": _criteria_payload(_criteria()),
        }
        item_rubrics = [item.get("rubric") for item in data["items"]]
        assert item_rubrics[:4] == [None] * 4
        assert item_rubrics[4] == {
            "guidelines": ITEM_GUIDELINES,
            "criteria": _criteria_payload(_item_criteria()),
        }
        assert item_rubrics[5] == _criteria_payload(_item_criteria())

    def test_without_guidelines_rubrics_keep_the_list_form(self) -> None:
        """No guidelines (none or blank) means the list form, as before guidelines existed."""
        plain = _dataset(global_guidelines=None)
        plain.items[4].rubric = Rubric(_item_criteria())
        blank = _dataset(global_guidelines="  ")
        blank.items[4].rubric = Rubric(_item_criteria(), guidelines="")
        assert blank.to_json() == plain.to_json()
        data = json.loads(plain.to_json())
        assert data["rubric"] == _criteria_payload(_criteria())
        assert data["items"][4]["rubric"] == _criteria_payload(_item_criteria())


class TestRoundTrip:
    def test_json_round_trip_keeps_global_and_per_item_guidelines(self) -> None:
        dataset = _dataset()
        loaded = RubricDataset.from_json(dataset.to_json())
        assert loaded.rubric is not None
        assert loaded.rubric.guidelines == GLOBAL_GUIDELINES
        assert loaded.items[4].rubric is not None
        assert loaded.items[4].rubric.guidelines == ITEM_GUIDELINES
        assert [i.ground_truth for i in loaded.items] == [i.ground_truth for i in dataset.items]
        # Re-serialization is idempotent.
        assert loaded.to_json() == dataset.to_json()

    def test_file_round_trip(self, tmp_path) -> None:
        path = tmp_path / "dataset.json"
        dataset = _dataset()
        dataset.to_file(path)
        loaded = RubricDataset.from_file(path)
        assert loaded.rubric is not None and loaded.rubric.guidelines == GLOBAL_GUIDELINES
        assert loaded.to_json() == dataset.to_json()

    def test_dataset_file_accepts_every_rubric_form(self) -> None:
        criteria = _criteria_payload(_item_criteria())
        text = json.dumps(
            {
                "prompt": "p",
                "rubric": {"guidelines": GLOBAL_GUIDELINES, "sections": [{"criteria": criteria}]},
                "items": [
                    {"submission": "a", "description": "global rubric"},
                    {
                        "submission": "b",
                        "description": "own rubric, rubric wrapper",
                        "rubric": {"rubric": {"guidelines": ITEM_GUIDELINES, "criteria": criteria}},
                    },
                    {"submission": "c", "description": "own list rubric", "rubric": criteria},
                ],
            }
        )
        loaded = RubricDataset.from_json(text)
        assert [loaded.get_item_rubric(i).guidelines for i in range(3)] == [
            GLOBAL_GUIDELINES,
            ITEM_GUIDELINES,
            None,
        ]

    def test_invalid_guidelines_in_dataset_file_raises(self) -> None:
        text = json.dumps(
            {
                "prompt": "p",
                "rubric": {"guidelines": 5, "criteria": _criteria_payload(_item_criteria())},
                "items": [],
            }
        )
        with pytest.raises(ValueError, match="Expected 'guidelines' to be a string, got int"):
            RubricDataset.from_json(text)


class TestPerItemRubrics:
    def test_item_rubric_carries_its_own_guidelines(self) -> None:
        dataset = _dataset()
        assert dataset.get_item_rubric(0).guidelines == GLOBAL_GUIDELINES
        assert dataset.get_item_rubric(4).guidelines == ITEM_GUIDELINES

    def test_item_rubric_without_guidelines_does_not_inherit_global_ones(self) -> None:
        """Precedence is unchanged: an item's own rubric replaces the global one entirely."""
        dataset = _dataset()
        dataset.add_item(
            submission="Other report",
            description="own rubric without guidelines",
            rubric=Rubric(_item_criteria()),
        )
        assert dataset.get_item_rubric(5).guidelines is None


class TestSplitsAndCopies:
    @pytest.mark.parametrize("stratify", [True, False])
    def test_split_train_test_keeps_guidelines(self, stratify: bool) -> None:
        dataset = _dataset()
        train, test = dataset.split_train_test(n_train=2, stratify=stratify, seed=0)
        for part in (train, test):
            assert part.rubric is not None
            assert part.rubric.guidelines == GLOBAL_GUIDELINES
        own = [
            item.rubric.guidelines
            for part in (train, test)
            for item in part.items
            if item.rubric is not None
        ]
        assert own == [ITEM_GUIDELINES]

    def test_dataset_hash_ignores_guidelines(self) -> None:
        """The resume guard fingerprints rubric lengths only, so guidelines never block it."""
        assert _compute_dataset_hash(_dataset()) == _compute_dataset_hash(
            _dataset(global_guidelines=None)
        )

    @pytest.mark.asyncio
    async def test_fill_ground_truth_keeps_guidelines(self) -> None:
        dataset = RubricDataset(
            prompt="Write an essay.",
            rubric=Rubric(_item_criteria(), guidelines=GLOBAL_GUIDELINES),
            items=[
                DataItem(submission="a", description="global rubric"),
                DataItem(
                    submission="b",
                    description="own rubric",
                    rubric=Rubric(_item_criteria(), guidelines=ITEM_GUIDELINES),
                ),
            ],
        )
        report = EvaluationReport(
            score=1.0,
            raw_score=1.0,
            report=[
                CriterionReport(
                    requirement="Describes the method",
                    name="method",
                    weight=1.0,
                    verdict=MET,
                    reason="ok",
                )
            ],
        )
        grader = AsyncMock()
        grader.grade = AsyncMock(return_value=report)
        filled = await fill_ground_truth(dataset, grader, show_progress=False)
        assert filled.rubric is not None and filled.rubric.guidelines == GLOBAL_GUIDELINES
        assert filled.items[1].rubric is not None
        assert filled.items[1].rubric.guidelines == ITEM_GUIDELINES
        assert [item.ground_truth for item in filled.items] == [[MET], [MET]]
