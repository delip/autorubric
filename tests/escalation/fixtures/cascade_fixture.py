"""Recorded judgments for the cascade replay parity test, loaded into plain structures.

``cascade_parity.json`` next to this module was built by ``build_cascade_fixture.py``, which
documents where its records come from. Two judges graded the same labelled items: a
decision model (TypeSafe's Jev, asked the framed Noul question, ``decision_threshold``
0.5) and an LLM judge (DeepSeek V4.1 Flash through ``CriterionGrader``). The fixture holds
two datasets:

- ``ricechem-q3-test-every-2``: one rubric shared by every item (``RecordedDataset.rubric``);
- ``healthbench-200-every-20``: a rubric per item (``RecordedDataset.rubric`` is ``None``).

For every (item, criterion) pair it holds what building a decision-model ``EvalResult`` and
an LLM ``EvalResult`` offline needs: the criterion, the ground truth, the decision model's
recorded P(MET) with the verdict, probabilities and confidence the library derives from it
(``autorubric.decision.answer_to_report``), and the LLM judge's recorded verdict.
Neither judge abstained or failed on any pair, and no P(MET) is exactly 0.5 (there the
simulation below reads MET where the library reads the worst case).

Each dataset also holds ``ExpectedPoint``s: what an offline simulation of a
decision-model-first cascade reports on exactly its pairs at a margin ``tau``. The
simulation (``simulated_point``) keeps the decision model's verdict unless
``|P(MET) - 0.5| < tau``, when it takes the LLM judge's verdict instead, and its accuracy is
the share of the dataset's pairs whose verdict equals the ground truth (``compute_metrics``
pools binary pairs the same way for ``criterion_accuracy``). A library cascade
escalates a criterion when the decision model's confidence is below its threshold; for a
Noul answer at ``decision_threshold=0.5`` that confidence is ``2 * |P(MET) - 0.5|``, so the
library escalates the same pairs at threshold ``2 * tau`` (``ExpectedPoint.threshold``).
The taus are odd multiples of 1/16, off the two-decimal grid of the recorded
probabilities, so no pair's margin equals a tau. Where one does, the two sides round the
margin differently (the library computes ``2 * (1 - p) - 1`` for an UNMET verdict) and can
decide differently: on the simulation's own grid of taus (multiples of 0.02) they do for
P(MET) 0.04, 0.06, 0.30, 0.32 and 0.46, each at the tau equal to its margin.

To build the two ``EvalResult`` objects: an item's position in ``RecordedDataset.items`` is
its ``item_idx``, and a pair's position in ``RecordedItem.pairs`` is its criterion's index
in ``RecordedItem.rubric``. Submission texts and the LLM judge's reasons are not stored:
neither escalation nor accuracy reads them. ``RecordedItem.submission_sha256`` (the SHA-256
of the UTF-8 text) stands in for an item's submission, distinct per item and traceable to
its source.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autorubric import Criterion, CriterionVerdict

FIXTURE_PATH = Path(__file__).with_name("cascade_parity.json")


@dataclass(frozen=True)
class RecordedPair:
    """One (item, criterion) pair: its ground truth and both judges' recorded judgments.

    Attributes:
        ground_truth: The human label.
        p_met: The decision model's recorded P(MET), the probability of its Noul "yes".
        dm_verdict: The decision model's verdict, as the library reads ``p_met``.
        dm_probabilities: ``{"MET": p_met, "UNMET": 1 - p_met}``, as the library records it.
        dm_confidence: The library's confidence in ``dm_verdict``.
        llm_verdict: The LLM judge's recorded verdict.
    """

    ground_truth: CriterionVerdict
    p_met: float
    dm_verdict: CriterionVerdict
    dm_probabilities: dict[str, float]
    dm_confidence: float
    llm_verdict: CriterionVerdict


@dataclass(frozen=True)
class RecordedItem:
    """One graded item.

    Attributes:
        source_item_idx: The item's index in the source run, for tracing it back.
        description: The item's description in the source dataset.
        submission_sha256: SHA-256 hex digest of the submission's UTF-8 text.
        rubric: The criteria the item was graded against, in order.
        pairs: One pair per criterion of ``rubric``, in the same order.
    """

    source_item_idx: int
    description: str
    submission_sha256: str
    rubric: tuple[Criterion, ...]
    pairs: tuple[RecordedPair, ...]


@dataclass(frozen=True)
class ExpectedPoint:
    """The simulated cascade at one margin ``tau``, over all pairs of a dataset.

    Attributes:
        tau: The simulation's margin: it defers a pair when ``|P(MET) - 0.5| < tau``.
        threshold: ``2 * tau``, the library's escalation threshold for the same pairs.
        n_deferred: Pairs deferred to the LLM judge.
        n_correct: Pairs whose cascade verdict equals the ground truth.
        accuracy: ``n_correct`` over the number of pairs.
    """

    tau: float
    threshold: float
    n_deferred: int
    n_correct: int
    accuracy: float


@dataclass(frozen=True)
class RecordedDataset:
    """One labelled dataset's recorded pairs and the simulated cascade on them.

    Attributes:
        name: The fixture's name for the subset.
        source: Where the subset comes from and how it was selected.
        llm_judge_id: The ``judge_id`` of the LLM judge's recorded votes.
        rubric: The rubric every item shares, or ``None`` when each item has its own.
        items: The items, in ``item_idx`` order.
        expected: The simulated cascade at each tau, in increasing tau.
    """

    name: str
    source: str
    llm_judge_id: str
    rubric: tuple[Criterion, ...] | None
    items: tuple[RecordedItem, ...]
    expected: tuple[ExpectedPoint, ...]


def simulated_point(pairs: Sequence[RecordedPair], tau: float) -> ExpectedPoint:
    """The simulation's cascade on ``pairs`` at margin ``tau``.

    A pair is deferred when ``|P(MET) - 0.5| < tau`` and then takes the LLM judge's
    verdict; otherwise it keeps the decision model's. A verdict is correct when it is MET
    exactly when the ground truth is MET.
    """
    n_deferred = n_correct = 0
    for pair in pairs:
        deferred = abs(pair.p_met - 0.5) < tau
        verdict = pair.llm_verdict if deferred else pair.dm_verdict
        n_deferred += deferred
        n_correct += (verdict == CriterionVerdict.MET) == (
            pair.ground_truth == CriterionVerdict.MET
        )
    return ExpectedPoint(
        tau=tau,
        threshold=2 * tau,
        n_deferred=n_deferred,
        n_correct=n_correct,
        accuracy=n_correct / len(pairs),
    )


def _rubric(data: list[dict[str, Any]]) -> tuple[Criterion, ...]:
    return tuple(Criterion(**criterion) for criterion in data)


def _pair(data: dict[str, Any]) -> RecordedPair:
    return RecordedPair(
        ground_truth=CriterionVerdict(data["ground_truth"]),
        p_met=data["p_met"],
        dm_verdict=CriterionVerdict(data["dm_verdict"]),
        dm_probabilities=data["dm_probabilities"],
        dm_confidence=data["dm_confidence"],
        llm_verdict=CriterionVerdict(data["llm_verdict"]),
    )


def _dataset(data: dict[str, Any]) -> RecordedDataset:
    shared = None if data["rubric"] is None else _rubric(data["rubric"])
    items = tuple(
        RecordedItem(
            source_item_idx=item["source_item_idx"],
            description=item["description"],
            submission_sha256=item["submission_sha256"],
            rubric=shared if shared is not None else _rubric(item["rubric"]),
            pairs=tuple(_pair(pair) for pair in item["pairs"]),
        )
        for item in data["items"]
    )
    return RecordedDataset(
        name=data["name"],
        source=data["source"],
        llm_judge_id=data["llm_judge_id"],
        rubric=shared,
        items=items,
        expected=tuple(ExpectedPoint(**point) for point in data["expected"]),
    )


def load_cascade_fixture(path: Path = FIXTURE_PATH) -> tuple[RecordedDataset, ...]:
    """The fixture's datasets: the shared-rubric one, then the per-item-rubric one."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(_dataset(dataset) for dataset in data["datasets"])
