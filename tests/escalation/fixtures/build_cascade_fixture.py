"""Build ``cascade_parity.json``, the recorded fixture of the cascade replay parity test.

The records come from the decision-model experiments kept beside this repository
(``jev_experiments/``, not part of the package). They graded labelled data with two judges
and simulated a decision-model-first cascade offline on the recorded outputs
(``jev_experiments/cascade_analysis.py``). This script takes a small subset of those
records, with no new calls:

- the decision model: TypeSafe's Jev (``jev-latest``), asked the framed Noul question for
  each criterion; its P(yes) is P(MET), and the experiments read MET at P(MET) >= 0.5;
- the LLM judge: ``openrouter/deepseek/deepseek-v4.1-flash`` through ``CriterionGrader``,
  a single judge with ``judge_id`` ``"default"``.

The subset is fixed by rule, not chosen by outcome: the items of a dataset whose
``item_idx`` is a multiple of a stride, and on which neither judge abstained or failed.

- RiceChem question 3, the 10% test split exactly as the experiments built it (``seed``
  42), stride 2: 15 of its 29 student responses, 7 criteria each, 105 pairs. Ground
  truth: the dataset's labels.
- HealthBench-200, the HealthBench-100 validation and test sets concatenated (200
  physician-graded completions, 1 to 3 consensus criteria each), stride 20: 10 items, 20
  pairs. Ground truth: the physicians' majority.

Neither subset leaves an item out. "Neither judge abstained or failed" means: both judges
gave every criterion of the item a MET or UNMET verdict, no record of the item carries an
error, and no P(MET) is 0.5, where the experiments read MET and the library the worst
case. It is read from both runs' checkpoints, not from the simulation's pair records
alone: those show a failed LLM call only as the worst-case verdict recorded for it
(UNMET), as for HealthBench items 161 and 170, outside this subset.

For each pair the fixture stores the criterion, the ground truth, the decision model's
P(MET) with the verdict, probabilities and confidence the library derives from it
(``autorubric.decision.answer_to_report``), and the LLM judge's verdict (from its
checkpointed vote). Submission texts are stored as their SHA-256 digests, and the LLM
judge's reasons, which quote the submissions, not at all: neither escalation nor accuracy
reads them. For each
dataset it stores the simulated cascade (``cascade_fixture.simulated_point``) at the taus
1/16, 3/16, 5/16 and 7/16, with the library threshold ``2 * tau``. These taus lie off the
two-decimal grid of the recorded probabilities, so no pair's margin ``|P(MET) - 0.5|``
equals a tau. A tau on that grid can equal a margin exactly, and there the library's
``confidence < 2 * tau`` and the simulation's ``margin < tau`` can round to different
sides.

Before writing, the script checks each of the following, and stops at the first mismatch:

1. The simulation's own loaders and ``simulate`` reproduce its recorded summary
   (``results/cascade/summary.json``), so the records read here are its inputs.
2. Each pair's records agree across the simulation's pair records, the source datasets
   (criteria, ground truth, descriptions) and both judges' checkpoints: the decision
   model's raw answer is a Noul answer with the recorded P(MET), and the LLM judge cast
   one vote per criterion, with the recorded verdict.
3. ``answer_to_report`` reads each P(MET) as the verdict the experiments recorded.
4. The experiments' ``simulate``, restricted to the subset, gives the accuracy and the
   deferred share ``simulated_point`` gives, at the fixture's taus and at every tau of the
   simulation's own grid.
5. At each point, the live cascade's escalation rule (``_escalates`` at the threshold
   ``2 * tau``) escalates exactly the pairs the simulation defers.
6. The written file loads back (``load_cascade_fixture``) to the same records.

Run it from the repository root with the experiments' environment, which has their
dependencies and this checkout's library installed in editable mode::

    PYTHONDONTWRITEBYTECODE=1 jev_experiments/.venv/bin/python \\
        tests/escalation/fixtures/build_cascade_fixture.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Callable, Hashable
from pathlib import Path
from types import ModuleType
from typing import Any

from cascade_fixture import (
    FIXTURE_PATH,
    RecordedDataset,
    RecordedItem,
    RecordedPair,
    load_cascade_fixture,
    simulated_point,
)
from typesafe_sdk import SystemOneResponse

from autorubric import Criterion, CriterionVerdict, DecisionModelConfig
from autorubric.dataset import DataItem, RubricDataset
from autorubric.decision import answer_to_report, question_id
from autorubric.graders.criterion_grader import _escalates

REPO_DIR = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_DIR / "jev_experiments"

TAUS = (1 / 16, 3 / 16, 5 / 16, 7 / 16)
RICECHEM_QUESTION = 3
RICECHEM_STRIDE = 2
HEALTHBENCH_STRIDE = 20
LLM = "deepseek"
"""The LLM judge's key in the simulation's ``LLMS``."""
LLM_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
LLM_JUDGE_ID = "default"
DECISION_MODEL = DecisionModelConfig(model="jev-latest", binary_framing="noul_framed")
BINARY_VERDICTS = ("MET", "UNMET")

ABOUT = (
    "Recorded judgments of a decision model (TypeSafe's Jev, framed Noul question, "
    "decision_threshold 0.5) and an LLM judge (DeepSeek V4.1 Flash through CriterionGrader) "
    "on the same labelled (item, criterion) pairs, with the accuracy an offline simulation "
    "of a decision-model-first cascade reports on them: it defers a pair to the LLM judge "
    "when |P(MET) - 0.5| < tau, which a library cascade does at threshold 2 * tau. "
    "Neither judge abstained or failed on any pair, and no P(MET) is 0.5. Submissions are "
    "stored as the SHA-256 of their UTF-8 text. Built by build_cascade_fixture.py (see its "
    "docstring for provenance and checks); loaded by cascade_fixture.py."
)

Record = dict[str, Any]
Records = tuple[dict[Hashable, Record], dict[str, dict[Hashable, Record]], Any, Any]
"""What the simulation's ``rc_pairs``/``hb_pairs`` return: the decision model's pair
records, each LLM judge's, and per-judge cost and compute time."""
PairKey = Callable[[DataItem, int, Criterion], Hashable]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verdict(value: Any) -> CriterionVerdict:
    """A ground-truth or recorded verdict, stored either as the enum or its value."""
    return value if isinstance(value, CriterionVerdict) else CriterionVerdict(value)


def noul_answers(criterion_idx: int, p_met: float) -> dict[str, Any]:
    """A Noul answer for one question, decoded from its wire form like a response."""
    payload = {
        "model": DECISION_MODEL.model,
        "usage": {"input_tokens": 1},
        "answers": {question_id(criterion_idx): {"type": "noul", "noul": p_met}},
    }
    return SystemOneResponse.model_validate_json(json.dumps(payload)).answers


def checkpoint(path: Path) -> dict[int, Record]:
    """An ``evaluate()`` run's checkpointed item records, by ``item_idx``."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {record["item_idx"]: record for record in records}


def judged_cleanly(dm_record: Record, llm_record: Record, dm_rows: list[Record]) -> bool:
    """Whether both judges gave every criterion of an item a MET or UNMET verdict, no record
    of the item carries an error, and no P(MET) is at the decision threshold."""
    if any(record["error"] is not None for record in (dm_record, llm_record)):
        return False
    dm_report, llm_report = dm_record["report"], llm_record["report"]
    dm_criteria = dm_report["criterion_reports"]
    llm_criteria = llm_report["criterion_reports"]
    votes = [vote for report in llm_criteria for vote in report["votes"]]
    return (
        dm_report["error"] is None
        and llm_report["error"] is None
        and all(report["error"] is None for report in dm_criteria + llm_criteria + votes)
        and all(report["verdict"] in BINARY_VERDICTS for report in dm_criteria + votes)
        and all(report["final_verdict"] in BINARY_VERDICTS for report in llm_criteria)
        and all(row["p_met"] != DECISION_MODEL.decision_threshold for row in dm_rows)
    )


def recorded_item(
    source_item_idx: int,
    item: DataItem,
    rubric: tuple[Criterion, ...],
    rows: tuple[list[Record], list[Record]],
    records: tuple[Record, Record],
) -> RecordedItem:
    """One item's pairs, checked across every record that describes them.

    Args:
        source_item_idx: The item's index in the source dataset and in both runs.
        item: The source dataset's item.
        rubric: The criteria it was graded against.
        rows: The simulation's pair records of its criteria: the decision model's, then the
            LLM judge's.
        records: Its checkpointed records: the decision model's run, then the LLM judge's.
    """
    dm_rows, llm_rows = rows
    dm_record, llm_record = records
    assert judged_cleanly(dm_record, llm_record, dm_rows)
    assert item.ground_truth is not None
    pairs = []
    for criterion_idx, (criterion, dm_row, llm_row, dm_report, llm_report, truth) in enumerate(
        zip(
            rubric,
            dm_rows,
            llm_rows,
            dm_record["report"]["criterion_reports"],
            llm_record["report"]["criterion_reports"],
            item.ground_truth,
            strict=True,
        )
    ):
        for row in (dm_row, llm_row):
            assert row["item_idx"] == source_item_idx
            assert (row["criterion_name"], row["criterion"], row["weight"]) == (
                criterion.name,
                criterion.requirement,
                criterion.weight,
            )
            assert verdict(row["ground_truth"]) == verdict(truth)
        assert Criterion(**{name: dm_report[name] for name in Criterion.model_fields}) == criterion
        assert Criterion(**llm_report["criterion"]) == criterion
        p_met = dm_row["p_met"]
        assert isinstance(p_met, float)
        assert json.loads(dm_report["reasoning"]) == {"type": "noul", "noul": p_met}
        assert dm_report["verdict"] == dm_row["predicted"]
        (vote,) = llm_report["votes"]
        assert vote["judge_id"] == LLM_JUDGE_ID
        assert vote["verdict"] == llm_report["final_verdict"] == llm_row["predicted"]
        library_report = answer_to_report(
            criterion, criterion_idx, noul_answers(criterion_idx, p_met), DECISION_MODEL
        )
        assert library_report.verdict is not None
        assert library_report.verdict.value == dm_row["predicted"]
        assert library_report.probabilities is not None
        assert library_report.confidence is not None
        pairs.append(
            RecordedPair(
                ground_truth=verdict(truth),
                p_met=p_met,
                dm_verdict=library_report.verdict,
                dm_probabilities=library_report.probabilities,
                dm_confidence=library_report.confidence,
                llm_verdict=CriterionVerdict(vote["verdict"]),
            )
        )
    return RecordedItem(
        source_item_idx=source_item_idx,
        description=item.description,
        submission_sha256=sha256(item.submission),
        rubric=rubric,
        pairs=tuple(pairs),
    )


def subset(
    name: str,
    source: str,
    dataset: RubricDataset,
    stride: int,
    pair_key: PairKey,
    records: Records,
    checkpoints: tuple[Path, Path],
) -> tuple[RecordedDataset, set[Hashable]]:
    """Every ``stride``-th item of ``dataset`` on which neither judge abstained or failed.

    Args:
        name: The fixture's name for the subset.
        source: The source dataset, for the subset's description.
        dataset: The dataset both judges graded, as the experiments loaded it.
        stride: Keep the items whose ``item_idx`` is a multiple of it.
        pair_key: The simulation's key of an (item, criterion index, criterion) pair.
        records: The simulation's records of the dataset (``rc_pairs``/``hb_pairs``).
        checkpoints: Both runs' checkpoints: the decision model's, then the LLM judge's.

    Returns:
        The subset, and the simulation's keys of its pairs.
    """
    jev, llm, _, _ = records
    dm_records, llm_records = (checkpoint(path) for path in checkpoints)
    items, keys, left_out = [], set(), []
    for item_idx in range(0, len(dataset.items), stride):
        item = dataset.items[item_idx]
        rubric = tuple(dataset.get_item_rubric(item_idx).rubric)
        item_keys = [pair_key(item, i, criterion) for i, criterion in enumerate(rubric)]
        rows = ([jev[key] for key in item_keys], [llm[LLM][key] for key in item_keys])
        item_records = (dm_records[item_idx], llm_records[item_idx])
        if not judged_cleanly(*item_records, rows[0]):
            left_out.append(item_idx)
            continue
        keys.update(item_keys)
        items.append(recorded_item(item_idx, item, rubric, rows, item_records))
    pairs = [pair for item in items for pair in item.pairs]
    recorded = RecordedDataset(
        name=name,
        source=(
            f"{source}: the items with item_idx % {stride} == 0 on which neither judge "
            f"abstained or failed, {len(items)} items (left out: "
            f"{', '.join(map(str, left_out)) or 'none'})"
        ),
        llm_judge_id=LLM_JUDGE_ID,
        rubric=None if dataset.rubric is None else tuple(dataset.rubric.rubric),
        items=tuple(items),
        expected=tuple(simulated_point(pairs, tau) for tau in TAUS),
    )
    return recorded, keys


def ricechem(cascade: ModuleType, records: Records) -> tuple[RecordedDataset, set[Hashable]]:
    runner = importlib.import_module("run_ricechem")
    run = Path("test") / f"q{RICECHEM_QUESTION}" / "items.jsonl"
    full = RubricDataset.from_file(runner.DATA_DIR / f"q{RICECHEM_QUESTION}.json")
    return subset(
        name=f"ricechem-q{RICECHEM_QUESTION}-test-every-{RICECHEM_STRIDE}",
        source=f"RiceChem question {RICECHEM_QUESTION}, the 10% test split (seed 42)",
        dataset=runner.test_split(full),
        stride=RICECHEM_STRIDE,
        pair_key=lambda item, criterion_idx, _: (
            RICECHEM_QUESTION,
            item.description,
            criterion_idx,
        ),
        records=records,
        checkpoints=(
            runner.RESULTS_DIR / cascade.JEV / run,
            runner.RESULTS_DIR / cascade.LLMS[LLM][0] / run,
        ),
    )


def healthbench(cascade: ModuleType, records: Records) -> tuple[RecordedDataset, set[Hashable]]:
    runner = importlib.import_module("run_healthbench")
    dataset, _ = runner.load_items(strip_reference=True)
    # The LLM run's checkpoint directory mirrors its litellm model id.
    assert cascade.LLMS[LLM][0] == "llm_" + LLM_MODEL.replace("/", "_")

    def pair_key(item: DataItem, _: int, criterion: Criterion) -> Hashable:
        meta = runner.item_meta(item.description)
        return (meta["prompt_id"], meta["completion_id"], criterion.name)

    return subset(
        name=f"healthbench-200-every-{HEALTHBENCH_STRIDE}",
        source="HealthBench-200 (the HealthBench-100 validation and test sets)",
        dataset=dataset,
        stride=HEALTHBENCH_STRIDE,
        pair_key=pair_key,
        records=records,
        checkpoints=(
            runner.RESULTS_DIR / f"jev_{cascade.JEV}" / "all200" / "items.jsonl",
            runner.RESULTS_DIR / f"llm_{LLM_MODEL}" / "all200" / "items.jsonl",
        ),
    )


def check_recorded_summary(cascade: ModuleType) -> None:
    """The simulation, rerun on the records read here, gives its recorded summary."""
    recorded = json.loads((cascade.OUT / "summary.json").read_text(encoding="utf-8"))
    rerun = {
        "RiceChem (test split)": cascade.simulate(*cascade.rc_pairs()),
        "HealthBench-200": cascade.simulate(*cascade.hb_pairs(), physician=True),
    }
    assert json.loads(json.dumps(rerun)) == recorded


def check_simulation(
    cascade: ModuleType, records: Records, keys: set[Hashable], dataset: RecordedDataset
) -> None:
    """The experiments' ``simulate``, on the subset, agrees with ``simulated_point``."""
    jev, llm, cost, secs = records
    subset_records = {key: jev[key] for key in keys}
    pairs = [pair for item in dataset.items for pair in item.pairs]
    assert len(subset_records) == len(pairs)
    grid = cascade.TAUS
    for taus in (TAUS, tuple(float(tau) for tau in grid)):
        cascade.TAUS = taus  # ``simulate`` sweeps its module's grid
        try:
            curve = cascade.simulate(subset_records, llm, cost, secs)["curves"][LLM]["points"]
        finally:
            cascade.TAUS = grid
        for row, tau in zip(curve, taus, strict=True):
            point = simulated_point(pairs, tau)
            assert row["tau"] == tau
            assert row["accuracy"] == point.accuracy, (dataset.name, tau)
            assert row["deferred"] == point.n_deferred / len(pairs), (dataset.name, tau)


def check_library_escalation(dataset: RecordedDataset) -> None:
    """At ``2 * tau`` the live escalation rule escalates exactly the deferred pairs."""
    for item in dataset.items:
        for criterion_idx, (criterion, pair) in enumerate(zip(item.rubric, item.pairs)):
            report = answer_to_report(
                criterion, criterion_idx, noul_answers(criterion_idx, pair.p_met), DECISION_MODEL
            )
            for point in dataset.expected:
                deferred = abs(pair.p_met - 0.5) < point.tau
                assert _escalates(report, point.threshold) == deferred, (pair.p_met, point)


def criterion_json(criterion: Criterion) -> Record:
    record = {
        "name": criterion.name,
        "requirement": criterion.requirement,
        "weight": criterion.weight,
    }
    assert Criterion(**record) == criterion  # nothing else is set
    return record


def pair_json(pair: RecordedPair) -> Record:
    return {
        "ground_truth": pair.ground_truth.value,
        "p_met": pair.p_met,
        "dm_verdict": pair.dm_verdict.value,
        "dm_probabilities": pair.dm_probabilities,
        "dm_confidence": pair.dm_confidence,
        "llm_verdict": pair.llm_verdict.value,
    }


def dataset_json(dataset: RecordedDataset) -> Record:
    items = []
    for item in dataset.items:
        entry: Record = {
            "source_item_idx": item.source_item_idx,
            "description": item.description,
            "submission_sha256": item.submission_sha256,
        }
        if dataset.rubric is None:
            entry["rubric"] = [criterion_json(criterion) for criterion in item.rubric]
        entry["pairs"] = [pair_json(pair) for pair in item.pairs]
        items.append(entry)
    shared = dataset.rubric
    return {
        "name": dataset.name,
        "source": dataset.source,
        "llm_judge_id": dataset.llm_judge_id,
        "rubric": None if shared is None else [criterion_json(c) for c in shared],
        "items": items,
        "expected": [
            {
                "tau": point.tau,
                "threshold": point.threshold,
                "n_deferred": point.n_deferred,
                "n_correct": point.n_correct,
                "accuracy": point.accuracy,
            }
            for point in dataset.expected
        ],
    }


def main() -> None:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    cascade = importlib.import_module("cascade_analysis")

    check_recorded_summary(cascade)
    built = []
    for build, records in ((ricechem, cascade.rc_pairs()), (healthbench, cascade.hb_pairs())):
        dataset, keys = build(cascade, records)
        check_simulation(cascade, records, keys, dataset)
        check_library_escalation(dataset)
        built.append(dataset)

    fixture = {
        "about": ABOUT,
        "judges": {
            "decision_model": {
                "model": DECISION_MODEL.model,
                "binary_framing": DECISION_MODEL.binary_framing,
                "decision_threshold": DECISION_MODEL.decision_threshold,
            },
            "llm": {"model": LLM_MODEL, "judge_id": LLM_JUDGE_ID},
        },
        "datasets": [dataset_json(dataset) for dataset in built],
    }
    text = json.dumps(fixture, indent=1, ensure_ascii=False) + "\n"
    FIXTURE_PATH.write_text(text, encoding="utf-8", newline="\n")
    assert load_cascade_fixture() == tuple(built)

    print(f"wrote {FIXTURE_PATH.relative_to(REPO_DIR)} ({len(text.encode('utf-8'))} bytes)")
    for dataset in built:
        n_pairs = sum(len(item.pairs) for item in dataset.items)
        print(f"{dataset.name}: {n_pairs} pairs; {dataset.source}")
        for point in dataset.expected:
            print(
                f"  tau={point.tau} threshold={point.threshold} deferred={point.n_deferred} "
                f"correct={point.n_correct} accuracy={point.accuracy!r}"
            )


if __name__ == "__main__":
    main()
