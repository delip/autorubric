"""Cascade replay and calibration on submissions that read no image bytes.

A ``{"thinking", "output"}`` dict submission is graded as text, so a cascade's runs on such
items are replayed and calibrated as before images existed: their submissions are paired by
equality and enter the calibration fingerprint as they are (``json.dumps`` of the dict). A
document whose image file cannot be read (moved since the runs, or missing for an item
whose grading raised) does not stop a calibration: it enters the fingerprint by its parts'
sources. Runs are the scripted ones of ``cascade_runs``; nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from autorubric import (
    Criterion,
    EvalResult,
    Image,
    RubricDataset,
    calibrate_escalation,
    replay_escalation,
)
from autorubric.content import _content_digest
from autorubric.dataset import DataItem
from autorubric.escalation import _calibration_fingerprint, _fingerprint_key

LIGHT = Criterion(name="light", weight=5.0, requirement="Mentions light")
MYTH = Criterion(name="myth", weight=-3.0, requirement="Claims plants eat soil")


def base_fingerprint(items: list[tuple[DataItem, list[str]]]) -> str:
    """The calibration fingerprint as the library computed it before images existed: every
    field embedded as it is."""
    content = [
        [item.submission, item.prompt, item.reference_submission, requirements]
        for item, requirements in items
    ]
    return hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()


def with_submissions(result: EvalResult, submission_of) -> EvalResult:
    """``result`` with each item's submission replaced by ``submission_of(text)``."""
    return dataclasses.replace(
        result,
        item_results=[
            dataclasses.replace(
                r, item=dataclasses.replace(r.item, submission=submission_of(r.item.submission))
            )
            for r in result.item_results
        ],
    )


def data_with_submissions(data: RubricDataset, submission_of) -> RubricDataset:
    return dataclasses.replace(
        data,
        items=[
            dataclasses.replace(item, submission=submission_of(item.submission))
            for item in data.items
        ],
    )


def as_dict(text: str) -> dict[str, str]:
    return {"thinking": "Recall the light reactions.", "output": text}


def scores(result: EvalResult) -> list[float | None]:
    return [r.report.score for r in result.item_results]


class TestDictSubmissions:
    def test_the_fingerprint_is_the_one_computed_before_images(self):
        # DataItem is annotated as Content; a dataset file keeps a dict as it is.
        item = DataItem(submission=cast(Any, {"thinking": "t", "output": "o"}), description="d")
        assert _calibration_fingerprint([(item, [LIGHT])]) == base_fingerprint(
            [(item, [LIGHT.requirement])]
        )

    def test_text_fingerprints_are_unchanged(self):
        item = DataItem(submission="Plants use light.", description="d", prompt="Explain.")
        assert _calibration_fingerprint([(item, [LIGHT, MYTH])]) == base_fingerprint(
            [(item, [LIGHT.requirement, MYTH.requirement])]
        )

    @pytest.mark.asyncio
    async def test_a_replay_pairs_them_as_before(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None)
        dm_run, llm_run = (with_submissions(r, as_dict) for r in (runs.dm, runs.llm))

        replayed = replay_escalation(dm_run, llm_run, 0.5)
        assert scores(replayed) == scores(replay_escalation(runs.dm, runs.llm, 0.5))
        assert [r.item.submission for r in replayed.item_results] == [
            r.item.submission for r in dm_run.item_results
        ]

    @pytest.mark.asyncio
    async def test_a_replay_still_refuses_different_submissions(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None)
        dm_run = with_submissions(runs.dm, as_dict)
        llm_run = with_submissions(runs.llm, lambda text: {"output": text})
        with pytest.raises(ValueError, match="its submissions differ"):
            replay_escalation(dm_run, llm_run, 0.5)

    @pytest.mark.asyncio
    async def test_a_calibration_runs_with_the_base_fingerprint(self, cascade_runs):
        runs = await cascade_runs(per_criterion=None)
        data = data_with_submissions(runs.data, as_dict)
        dm_run, llm_run = (with_submissions(r, as_dict) for r in (runs.dm, runs.llm))

        curve = calibrate_escalation(data, dm_run, llm_run, thresholds=[0.0, 0.5, 1.0])
        text_curve = calibrate_escalation(runs.data, runs.dm, runs.llm, thresholds=[0.0, 0.5, 1.0])
        expected = base_fingerprint(
            [
                (item, [c.requirement for c in data.get_item_rubric(i).rubric])
                for i, item in enumerate(data.items)
            ]
        )
        for point, text_point in zip(curve.points, text_curve.points, strict=True):
            assert point.calibration_fingerprint == expected
            assert point.model_copy(
                update={"calibration_fingerprint": None}
            ) == text_point.model_copy(update={"calibration_fingerprint": None})


class TestImagesThatCannotBeRead:
    def test_a_missing_file_is_fingerprinted_by_its_path(self, tmp_path: Path):
        missing = Image(tmp_path / "moved" / "fig.png")
        held = Image(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
            b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
            b"\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        assert _fingerprint_key([missing, "Figure 1.", held]) == {
            "content_sources": [
                ["path", str(tmp_path / "moved" / "fig.png")],
                ["text", "Figure 1."],
                ["sha256", held.digest],
            ]
        }
        # A readable document is fingerprinted by its bytes, as before.
        assert _fingerprint_key([held, "Figure 1."]) == {
            "content_sha256": _content_digest([held, "Figure 1."])
        }

    @pytest.mark.asyncio
    async def test_a_calibration_does_not_stop(self, cascade_runs, tmp_path: Path):
        """The figures were moved after the runs: the calibration still sweeps, with a
        fingerprint taken from the images' paths."""
        runs = await cascade_runs(per_criterion=None)
        figures = {
            item.submission: Image(tmp_path / f"figure-{i}.png")
            for i, item in enumerate(runs.data.items)
        }

        def with_figure(text: str) -> list[str | Image]:
            return [figures[text], text]

        data = data_with_submissions(runs.data, with_figure)
        dm_run, llm_run = (with_submissions(r, with_figure) for r in (runs.dm, runs.llm))

        curve = calibrate_escalation(data, dm_run, llm_run, thresholds=[0.0, 0.5, 1.0])
        expected = _calibration_fingerprint(
            (item, data.get_item_rubric(i).rubric) for i, item in enumerate(data.items)
        )
        assert [p.calibration_fingerprint for p in curve.points] == [expected] * 3
        assert expected == base_fingerprint(
            [
                (
                    dataclasses.replace(
                        item,
                        submission=cast(
                            Any,
                            {
                                "content_sources": [
                                    ["path", str(figures[item.submission].path)],
                                    ["text", item.submission],
                                ]
                            },
                        ),
                    ),
                    [c.requirement for c in runs.data.get_item_rubric(i).rubric],
                )
                for i, item in enumerate(runs.data.items)
            ]
        )
        # A replay of the calibrated point recomputes the same fingerprint from its runs, so
        # it recognizes the calibration items.
        with pytest.warns(UserWarning, match="calibrated on the items it replays"):
            replay_escalation(dm_run, llm_run, curve.points[1])
