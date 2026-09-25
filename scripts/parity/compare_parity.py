"""Compare parity runs and write a report.

Subcommands:

- ``replay``: test R for a fresh baseline run and a replay of its response cache (items
  1-7).
- ``synthetic``: test R for two synthetic runs. Item 1 becomes "the recorded request logs
  are identical"; items 2-7 are as for replay, with nothing excluded, because the synthetic
  provider is deterministic (failures included).
- ``live``: test L for two baseline and two post-change fresh runs.

The comparator only reads run directories; it never imports ``autorubric``. It refuses to
compare runs whose pinned package versions, cost-map snapshot or parity manifest differ.
Exit status: 0 when every gating check passes, 1 when one fails, 2 when the runs are not
comparable.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import parity_config as pc

KNOWN_NEW_KEYS: dict[str, Any] = {
    "probabilities": None,
    "confidence": None,
    "superseded": False,
    "escalated": False,
}
"""Keys the decision-model feature adds to serialized reports, with their defaults. For
LLM-judge runs they may appear only at these values."""

KNOWN_NEW_METRICS_KEYS: dict[str, Any] = {
    "coverage": "full",
    "n_pairs": None,
}
"""Keys the decision-model feature adds to saved ``compute_metrics`` fields (each per-judge
entry's ``coverage`` and ``n_pairs``), with their defaults. For LLM-judge runs they may
appear only at these values."""

MASK = "<masked: the baseline call failed and was not cached>"

VOTE_FIELDS_BINARY = ("verdict", "reason", "reasoning", "error", "weight")
VOTE_FIELDS_MULTI = (
    "selected_index",
    "selected_label",
    "value",
    "na",
    "shuffle_order",
    "reason",
    "reasoning",
    "error",
    "weight",
)
CRITERION_FIELDS = (
    "final_verdict",
    "final_multi_choice_verdict",
    "agreement",
    "final_reason",
    "error",
)
REPORT_FIELDS = (
    "score",
    "raw_score",
    "judge_scores",
    "cannot_assess_count",
    "token_usage",
    "completion_cost",
    "mean_agreement",
    "error",
)

N_BOOTSTRAP = 10_000
ALPHA = 0.05
MAX_DISAGREEMENT_EXCESS = 0.03
TEST_SEED = 0
MAX_DETAILS = 25

Status = Literal["pass", "fail", "n/a", "info"]
PairKey = tuple[str, str]
"""``(config_id, dataset_id)``."""


# =======================================================================================
# Loading runs
# =======================================================================================


@dataclass
class Run:
    """A run directory written by ``run_parity.py``."""

    run_id: str
    path: Path
    meta: dict[str, Any]
    _checkpoints: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)
    _requests: list[dict[str, Any]] | None = None

    @classmethod
    def load(cls, parity_dir: Path, run_id: str) -> Run:
        """Read ``<parity_dir>/<run_id>/run.json``."""
        path = parity_dir / run_id
        meta_path = path / pc.RUN_FILE
        if not meta_path.exists():
            raise SystemExit(f"no parity run at {path}")
        return cls(run_id=run_id, path=path, meta=pc.read_json(meta_path))

    @property
    def code(self) -> str:
        """``baseline`` or ``after``."""
        return self.meta["code"]

    @property
    def mode(self) -> str:
        """``fresh``, ``replay`` or ``synthetic``."""
        return self.meta["mode"]

    @property
    def primary(self) -> str:
        """The run's primary construction."""
        return self.meta["primary_construction"]

    def passes(self) -> list[dict[str, Any]]:
        """Grading passes that completed."""
        return [p for p in self.meta["passes"] if p.get("status") == "completed"]

    def labels(self) -> set[str]:
        """Labels of every pass the run attempted."""
        return {p["pass"] for p in self.meta["passes"]}

    def by_key(self) -> dict[PairKey, dict[str, str]]:
        """``(config, dataset) -> {construction: pass label}`` for completed passes."""
        table: dict[PairKey, dict[str, str]] = defaultdict(dict)
        for entry in self.passes():
            table[(entry["config_id"], entry["dataset_id"])][entry["construction"]] = entry["pass"]
        return dict(table)

    def primary_label(self, key: PairKey) -> str | None:
        """Pass label of the primary construction for a config and dataset."""
        return self.by_key().get(key, {}).get(self.primary)

    def expected_keys(self) -> set[PairKey]:
        """Every ``(config, dataset)`` the run planned, completed or not."""
        return {(p["config_id"], p["dataset_id"]) for p in self.meta["passes"]}

    def checkpoint(self, label: str) -> dict[int, dict[str, Any]]:
        """Checkpoint records of a pass, keyed by item index."""
        if label not in self._checkpoints:
            self._checkpoints[label] = pc.read_checkpoint(self.path / pc.EXPERIMENTS_SUBDIR / label)
        return self._checkpoints[label]

    def requests(self) -> list[dict[str, Any]]:
        """Every recorded LLM request of the run."""
        if self._requests is None:
            self._requests = pc.read_jsonl(self.path / pc.REQUESTS_FILE)
        return self._requests

    def requests_for(self, label: str) -> list[dict[str, Any]]:
        """Recorded requests of one pass."""
        return [r for r in self.requests() if r["pass"] == label]

    def grader_config(self, label: str) -> dict[str, Any]:
        """``grader_config`` from a pass's experiment manifest."""
        manifest = pc.read_json(self.path / pc.EXPERIMENTS_SUBDIR / label / "manifest.json")
        return manifest.get("grader_config") or {}

    def warnings(self) -> list[dict[str, Any]]:
        """Recorded warnings."""
        return pc.read_json(self.path / pc.WARNINGS_FILE)

    def describe(self) -> dict[str, Any]:
        """Identity summary for reports."""
        library = self.meta["library"]
        return {
            "run_id": self.run_id,
            "code": self.code,
            "mode": self.mode,
            "status": self.meta.get("status"),
            "git_commit": library.get("git_commit"),
            "git_dirty": library.get("git_dirty"),
            "source_sha256": library.get("source_sha256"),
            "packages": self.meta["packages"]["pinned"],
            "provider_models": self.meta.get("provider_models"),
            "replay_source": self.meta.get("replay_source"),
            "checkpoint_source": self.meta.get("checkpoint_source"),
        }


def comparability(runs: list[Run]) -> tuple[list[str], list[str]]:
    """Problems that make runs incomparable, and notes worth reporting."""
    problems: list[str] = []
    notes: list[str] = []
    reference = runs[0]
    for run in runs[1:]:
        for name in pc.PINNED_PACKAGES:
            ours = reference.meta["packages"]["pinned"].get(name)
            theirs = run.meta["packages"]["pinned"].get(name)
            if ours != theirs:
                problems.append(f"{name} differs: {reference.run_id}={ours}, {run.run_id}={theirs}")
        for key, label in (
            ("cost_map_sha256", "model cost map snapshot"),
            ("parity_manifest_sha256", "parity manifest"),
            ("judges_fingerprint", "judge definitions"),
        ):
            if reference.meta.get(key) != run.meta.get(key):
                problems.append(f"{label} differs between {reference.run_id} and {run.run_id}")
        if reference.meta.get("harness_sha256") != run.meta.get("harness_sha256"):
            notes.append(f"harness scripts differ between {reference.run_id} and {run.run_id}")
        if reference.meta["packages"].get("other") != run.meta["packages"].get("other"):
            notes.append(
                f"unpinned package versions differ between {reference.run_id} and {run.run_id}"
            )
    for run in runs:
        if run.meta.get("status") != "completed":
            notes.append(f"{run.run_id} finished with status {run.meta.get('status')}")
        if run.meta.get("library", {}).get("git_dirty"):
            notes.append(f"{run.run_id} ran on a library tree with uncommitted changes")
    return problems, notes


# =======================================================================================
# Checks and reports
# =======================================================================================


@dataclass
class Check:
    """One parity check.

    Attributes:
        item: Short id (``R1``..``R7``, ``L``, ``L:<config>``, ...).
        title: What is checked.
        status: ``pass``, ``fail``, ``n/a`` or ``info``.
        summary: One-line result.
        gating: Whether a failure fails the comparison.
        details: Markdown lines with specifics (mismatches, diffs, tables).
        data: Machine-readable results.
    """

    item: str
    title: str
    status: Status
    summary: str
    gating: bool = True
    details: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


def status_of(failures: int) -> Status:
    """``pass`` when nothing failed."""
    return "pass" if failures == 0 else "fail"


@dataclass
class Report:
    """A comparison's checks, rendered to Markdown and JSON."""

    kind: str
    runs: dict[str, Run]
    checks: list[Check]
    notes: list[str]
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Every gating check passed (and the runs were comparable)."""
        if self.problems:
            return False
        return all(c.status != "fail" for c in self.checks if c.gating)

    def to_json(self) -> dict[str, Any]:
        """Machine-readable report."""
        return {
            "kind": self.kind,
            "passed": self.passed,
            "comparable": not self.problems,
            "problems": self.problems,
            "notes": self.notes,
            "runs": {role: run.describe() for role, run in self.runs.items()},
            "checks": [c.__dict__ for c in self.checks],
        }

    def to_markdown(self) -> str:
        """Human-readable report."""
        verdict = "PASS" if self.passed else "FAIL"
        lines = [f"# Parity report: {self.kind} ({verdict})", "", "## Runs", ""]
        lines += [
            "| role | run | code | mode | commit | dirty | library sha256 |",
            "|---|---|---|---|---|---|---|",
        ]
        for role, run in self.runs.items():
            d = run.describe()
            commit = (d["git_commit"] or "?")[:10]
            source = (d["source_sha256"] or "?")[:12]
            lines.append(
                f"| {role} | {d['run_id']} | {d['code']} | {d['mode']} | {commit} | "
                f"{d['git_dirty']} | {source} |"
            )
        first = next(iter(self.runs.values()))
        packages = ", ".join(f"{k} {v}" for k, v in first.meta["packages"]["pinned"].items())
        lines += ["", f"Pinned packages: {packages}", ""]
        if self.problems:
            lines += ["## Not comparable", ""] + [f"- {p}" for p in self.problems] + [""]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        lines += ["## Checks", "", "| item | check | status | gating | result |"]
        lines += ["|---|---|---|---|---|"]
        for check in self.checks:
            summary = check.summary.replace("|", "\\|")
            lines.append(
                f"| {check.item} | {check.title} | {check.status.upper()} | "
                f"{'yes' if check.gating else 'no'} | {summary} |"
            )
        for check in self.checks:
            if not check.details:
                continue
            lines += ["", f"### {check.item}: {check.title}", ""] + check.details
        return "\n".join(lines) + "\n"

    def write(self, stem: Path) -> None:
        """Write ``<stem>.md`` and ``<stem>.json``."""
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".md").write_text(self.to_markdown(), encoding="utf-8", newline="\n")
        pc.write_json(stem.with_suffix(".json"), self.to_json())


# =======================================================================================
# Structural comparison helpers
# =======================================================================================


def same(a: Any, b: Any) -> bool:
    """Exact equality of JSON data: types must match; NaN equals NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def short(value: Any, limit: int = 160) -> str:
    """Truncated ``repr`` for report lines."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def diff_paths(a: Any, b: Any, path: str = "$") -> list[str]:
    """Paths at which two JSON values differ (exact comparison)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(a.keys() | b.keys(), key=str):
            if key not in a:
                out.append(f"{path}.{key}: only in after ({short(b[key])})")
            elif key not in b:
                out.append(f"{path}.{key}: only in baseline ({short(a[key])})")
            else:
                out += diff_paths(a[key], b[key], f"{path}.{key}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            out += diff_paths(x, y, f"{path}[{index}]")
        return out
    return [] if same(a, b) else [f"{path}: {short(a)} != {short(b)}"]


def additive_problems(
    base: Any,
    new: Any,
    path: str,
    *,
    new_keys: dict[str, Any] | None,
    seen_new: Counter[str],
) -> list[str]:
    """Differences between ``base`` and ``new`` other than allowed additive keys.

    Args:
        base: Baseline JSON value.
        new: Post-change JSON value.
        path: JSON path of the values (for messages).
        new_keys: Keys ``new`` may add, each with the only value it may hold; ``None`` allows
            any new key with any value.
        seen_new: Receives ``"<key>=<value>"`` for each allowed new key encountered.
    """
    if isinstance(base, dict) and isinstance(new, dict):
        problems: list[str] = []
        for key in base:
            if key not in new:
                problems.append(f"{path}.{key}: missing after the change")
            else:
                problems += additive_problems(
                    base[key], new[key], f"{path}.{key}", new_keys=new_keys, seen_new=seen_new
                )
        for key in new.keys() - base.keys():
            if new_keys is None:
                seen_new[f"{key}={short(new[key], 60)}"] += 1
            elif key in new_keys and same(new[key], new_keys[key]):
                seen_new[f"{key}={new_keys[key]!r}"] += 1
            else:
                problems.append(f"{path}.{key}: new key with value {short(new[key])}")
        return problems
    if isinstance(base, list) and isinstance(new, list):
        if len(base) != len(new):
            return [f"{path}: length {len(base)} != {len(new)}"]
        problems = []
        for index, (x, y) in enumerate(zip(base, new, strict=True)):
            problems += additive_problems(
                x, y, f"{path}[{index}]", new_keys=new_keys, seen_new=seen_new
            )
        return problems
    return [] if same(base, new) else [f"{path}: {short(base)} != {short(new)}"]


def masked_record(record: dict[str, Any], masked: set[tuple[int, str]]) -> dict[str, Any]:
    """A checkpoint record without timing, and with failure text of masked votes hidden.

    A masked vote is one whose baseline call failed and was never cached: its replay raises
    a stand-in exception of the same failure category, so everything but the failure text
    (``reason``, ``error``, and the criterion's combined ``final_reason``/``error``) is
    still compared.
    """
    result = copy.deepcopy(record)
    result.pop("duration_seconds", None)
    report = result.get("report") or {}
    for criterion_idx, ecr in enumerate(report.get("criterion_reports") or []):
        hit = False
        for vote in (ecr.get("votes") or []) + (ecr.get("multi_choice_votes") or []):
            if (criterion_idx, vote.get("judge_id")) in masked:
                vote["reason"] = MASK
                if vote.get("error") is not None:
                    vote["error"] = MASK
                hit = True
        if hit:
            ecr["final_reason"] = MASK
            if ecr.get("error") is not None:
                ecr["error"] = MASK
    return result


def render_request(request: dict[str, Any]) -> list[str]:
    """Readable lines of a recorded request (messages first, then parameters)."""
    lines: list[str] = []
    for message in request.get("messages") or []:
        lines.append(f"=== message: {message.get('role')}")
        content = message.get("content")
        lines += (content if isinstance(content, str) else json.dumps(content)).split("\n")
    rest = {k: v for k, v in request.items() if k != "messages"}
    lines.append("=== parameters")
    lines += json.dumps(rest, indent=1, sort_keys=True, ensure_ascii=False).split("\n")
    return lines


def request_diff(
    before: dict[str, Any] | None, after: dict[str, Any], *, limit: int = 60
) -> list[str]:
    """Unified diff of two recorded requests, fenced for Markdown."""
    old = render_request(before) if before is not None else ["<no baseline request>"]
    diff = list(
        difflib.unified_diff(old, render_request(after), "baseline", "after", lineterm="", n=2)
    )
    if len(diff) > limit:
        diff = diff[:limit] + [f"... ({len(diff) - limit} more diff lines)"]
    return ["```diff", *diff, "```"]


def error_triples(run: Run, label: str) -> dict[tuple[int, int, str], str]:
    """Errored votes of a pass: ``(item, criterion, judge) -> error category``."""
    return {
        vote.triple: vote.error_category or "unknown"
        for record in run.checkpoint(label).values()
        for vote in pc.vote_rows(record)
        if vote.error is not None
    }


# =======================================================================================
# Test R
# =======================================================================================


def pair_passes(baseline: Run, after: Run) -> tuple[dict[PairKey, tuple[str, str]], list[str]]:
    """Match each baseline pass with the post-change primary pass of the same config."""
    pairs: dict[PairKey, tuple[str, str]] = {}
    problems: list[str] = []
    for key in sorted(baseline.expected_keys() | after.expected_keys()):
        ours, theirs = baseline.primary_label(key), after.primary_label(key)
        if ours is None or theirs is None:
            problems.append(f"{key[0]}/{key[1]}: not completed in both runs")
            continue
        pairs[key] = (ours, theirs)
    return pairs, problems


def attempted_triples(run: Run, label: str) -> set[tuple[int, int, str]]:
    """Triples a (network-blocked) pass tried to send."""
    return {
        (r["item_idx"], r["criterion_idx"], r["judge_id"])
        for r in run.requests_for(label)
        if None not in (r["item_idx"], r["criterion_idx"], r["judge_id"])
    }


def check_request_logs(baseline: Run, after: Run, pairs: dict[PairKey, tuple[str, str]]) -> Check:
    """Synthetic item 1: identical recorded request logs."""
    details: list[str] = []
    differing = 0
    compared = 0
    for key, (ours, theirs) in pairs.items():
        left = _requests_by_triple(baseline.requests_for(ours))
        right = _requests_by_triple(after.requests_for(theirs))
        compared += sum(len(v) for v in left.values())
        for triple in sorted(left.keys() | right.keys(), key=_triple_sort_key):
            a, b = left.get(triple, []), right.get(triple, [])
            if [_projection(r) for r in a] == [_projection(r) for r in b]:
                continue
            differing += 1
            if differing <= 5:
                details.append(
                    f"- {key[0]}/{key[1]} item {triple[0]} criterion {triple[1]} judge "
                    f"{triple[2]}: {len(a)} baseline vs {len(b)} after attempts"
                )
                details += request_diff(a[0]["request"] if a else None, b[0]["request"])
                if a and b:
                    fields = diff_paths(_projection(a[0]), _projection(b[0]))
                    details += [f"  - {line}" for line in fields[:10]]
    sockets = _socket_attempts(baseline, after)
    details += sockets
    failures = differing + len(sockets)
    summary = (
        f"{compared} baseline requests compared; {differing} (item, criterion, judge) "
        f"triples differ; {len(sockets)} other network attempts"
    )
    return Check(
        "R1",
        "recorded request logs identical",
        status_of(failures),
        summary,
        details=details,
        data={"compared": compared, "differing_triples": differing},
    )


def _projection(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in ("pass", "construction")}


def _triple_sort_key(triple: tuple[Any, Any, Any]) -> tuple[int, int, str]:
    item, criterion, judge = triple
    return (-1 if item is None else item, -1 if criterion is None else criterion, str(judge))


def _requests_by_triple(records: list[dict[str, Any]]) -> dict[tuple[Any, Any, Any], list[Any]]:
    table: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        table[(record["item_idx"], record["criterion_idx"], record["judge_id"])].append(record)
    for rows in table.values():
        rows.sort(key=lambda r: (r["attempt"], r["request_sha256"]))
    return dict(table)


def _socket_attempts(*runs: Run) -> list[str]:
    lines = []
    for run in runs:
        attempts = run.meta.get("socket_attempts")
        if attempts:
            lines.append(f"- {run.run_id}: {len(attempts)} network attempts outside litellm")
            lines += [f"  - {a['kind']} {a['target']}" for a in attempts[:MAX_DETAILS]]
    return lines


def check_network(baseline: Run, after: Run, pairs: dict[PairKey, tuple[str, str]]) -> Check:
    """Replay item 1: network attempts are exactly the baseline's uncached failures."""
    details: list[str] = []
    unexpected = unattributed = attempted_total = 0
    reproduced_failures = 0
    for key, (ours, _) in pairs.items():
        failed = error_triples(baseline, ours)
        for construction, label in sorted(after.by_key().get(key, {}).items()):
            records = after.requests_for(label)
            attempted_total += len(records)
            for record in records:
                triple = (record["item_idx"], record["criterion_idx"], record["judge_id"])
                if None in triple:
                    unattributed += 1
                    details.append(f"- {label}: unattributed network attempt")
                    details += request_diff(None, record["request"])
                    continue
                if triple in failed:
                    continue
                unexpected += 1
                if unexpected <= 5:
                    base_records = _requests_by_triple(baseline.requests_for(ours)).get(triple)
                    details.append(
                        f"- {label} item {triple[0]} criterion {triple[1]} judge {triple[2]}: "
                        "attempted a call the baseline answered (prompt, schema or sampling "
                        "changed)"
                    )
                    details += request_diff(
                        base_records[0]["request"] if base_records else None, record["request"]
                    )
            if construction == after.primary:
                reproduced_failures += len(set(failed) - attempted_triples(after, label))
    sockets = _socket_attempts(after)
    details += sockets
    failures = unexpected + unattributed + len(sockets)
    summary = (
        f"{attempted_total} blocked attempts; {unexpected} not among the baseline's failed "
        f"calls; {unattributed} unattributed; {reproduced_failures} baseline failures came "
        f"from cached responses and replayed; {len(sockets)} other network attempts"
    )
    return Check(
        "R1",
        "no unexpected network calls",
        status_of(failures),
        summary,
        details=details,
        data={
            "attempts": attempted_total,
            "unexpected": unexpected,
            "unattributed": unattributed,
            "reproduced_failures": reproduced_failures,
        },
    )


def masks_for(after: Run, label: str, masking: bool) -> dict[int, set[tuple[int, str]]]:
    """Per item, the ``(criterion, judge)`` votes whose failure text is masked."""
    table: dict[int, set[tuple[int, str]]] = defaultdict(set)
    if masking:
        for item, criterion, judge in attempted_triples(after, label):
            table[item].add((criterion, judge))
    return table


def check_processing(
    baseline: Run, after: Run, pairs: dict[PairKey, tuple[str, str]], masking: bool
) -> Check:
    """Item 2: per-vote, per-criterion and report-level fields are equal."""
    mismatches: list[str] = []
    counts: Counter[str] = Counter()
    for key, (ours, theirs) in pairs.items():
        left, right = baseline.checkpoint(ours), after.checkpoint(theirs)
        masks = masks_for(after, theirs, masking)
        if left.keys() != right.keys():
            mismatches.append(f"{key}: item sets differ ({sorted(left.keys() ^ right.keys())})")
            continue
        for item_idx in sorted(left):
            a = masked_record(left[item_idx], masks.get(item_idx, set()))
            b = masked_record(right[item_idx], masks.get(item_idx, set()))
            counts["masked votes"] += len(masks.get(item_idx, set()))
            prefix = f"{key[0]}/{key[1]} item {item_idx}"
            mismatches += _field_mismatches(prefix, a, b, "error", ("error",))
            report_a, report_b = a.get("report") or {}, b.get("report") or {}
            mismatches += _field_mismatches(prefix, report_a, report_b, "report", REPORT_FIELDS)
            counts["items"] += 1
            ecrs_a = report_a.get("criterion_reports") or []
            ecrs_b = report_b.get("criterion_reports") or []
            if len(ecrs_a) != len(ecrs_b):
                mismatches.append(f"{prefix}: {len(ecrs_a)} != {len(ecrs_b)} criteria")
                continue
            for criterion_idx, (ea, eb) in enumerate(zip(ecrs_a, ecrs_b, strict=True)):
                where = f"{prefix} criterion {criterion_idx}"
                mismatches += _field_mismatches(where, ea, eb, "criterion", CRITERION_FIELDS)
                counts["criteria"] += 1
                for votes_key, fields in (
                    ("votes", VOTE_FIELDS_BINARY),
                    ("multi_choice_votes", VOTE_FIELDS_MULTI),
                ):
                    va = {v["judge_id"]: v for v in ea.get(votes_key) or []}
                    vb = {v["judge_id"]: v for v in eb.get(votes_key) or []}
                    if va.keys() != vb.keys():
                        mismatches.append(f"{where}: judges {sorted(va)} != {sorted(vb)}")
                        continue
                    for judge_id in sorted(va):
                        counts["votes"] += 1
                        mismatches += _field_mismatches(
                            f"{where} judge {judge_id}", va[judge_id], vb[judge_id], "vote", fields
                        )
    summary = (
        f"{counts['items']} items, {counts['criteria']} criteria, {counts['votes']} votes "
        f"compared ({counts['masked votes']} failed votes with masked text); "
        f"{len(mismatches)} mismatches"
    )
    return Check(
        "R2",
        "identical processing (verdicts, scores, usage, cost)",
        status_of(len(mismatches)),
        summary,
        details=[f"- {m}" for m in mismatches[:MAX_DETAILS]],
        data={"counts": dict(counts), "mismatches": len(mismatches)},
    )


def _field_mismatches(
    where: str, a: dict[str, Any], b: dict[str, Any], level: str, fields: tuple[str, ...]
) -> list[str]:
    out = []
    for name in fields:
        if name not in a and name not in b:
            continue
        if name not in a or name not in b or not same(a[name], b[name]):
            out.append(f"{where} {level}.{name}: {short(a.get(name))} != {short(b.get(name))}")
    return out


def check_serialization(
    baseline: Run, after: Run, pairs: dict[PairKey, tuple[str, str]], masking: bool
) -> Check:
    """Item 3: ``ItemResult.to_dict`` output changes only by new keys at their defaults."""
    problems: list[str] = []
    seen_new: Counter[str] = Counter()
    compared = 0
    for key, (ours, theirs) in pairs.items():
        left, right = baseline.checkpoint(ours), after.checkpoint(theirs)
        masks = masks_for(after, theirs, masking)
        for item_idx in sorted(left.keys() & right.keys()):
            compared += 1
            problems += additive_problems(
                masked_record(left[item_idx], masks.get(item_idx, set())),
                masked_record(right[item_idx], masks.get(item_idx, set())),
                f"{key[0]}/{key[1]}[{item_idx}]",
                new_keys=KNOWN_NEW_KEYS,
                seen_new=seen_new,
            )
    new_keys = ", ".join(f"{k} (x{n})" for k, n in sorted(seen_new.items())) or "none"
    return Check(
        "R3",
        "serialization is additive only",
        status_of(len(problems)),
        f"{compared} records compared; {len(problems)} differences; new keys at defaults: "
        f"{new_keys}",
        details=[f"- {p}" for p in problems[:MAX_DETAILS]],
        data={"differences": len(problems), "new_keys_at_default": dict(seen_new)},
    )


def compare_metrics_dirs(
    left: Path, right: Path, seen_new: Counter[str] | None = None
) -> list[str]:
    """Differences between two saved ``compute_metrics`` outputs.

    Args:
        left: Metrics directory of the baseline library (or of the reference pass).
        right: Metrics directory compared with ``left``.
        seen_new: Given when ``left`` comes from the baseline library and ``right`` from the
            changed code: ``right``'s fields may then add the keys of
            ``KNOWN_NEW_METRICS_KEYS`` at their defaults, each counted here as
            ``"<key>=<value>"``. Without it the fields must be equal. Summary texts and
            frames must be equal either way.
    """
    import pandas as pd

    problems: list[str] = []
    for required in (left, right):
        if not (required / "metrics.json").exists():
            return [f"missing metrics in {required}"]
    base, new = pc.read_json(left / "metrics.json"), pc.read_json(right / "metrics.json")
    if seen_new is None:
        paths = diff_paths(base, new)
    else:
        paths = additive_problems(
            base, new, "$", new_keys=KNOWN_NEW_METRICS_KEYS, seen_new=seen_new
        )
    problems += [f"metrics.json {p}" for p in paths]
    for name in ("summary.txt", "summary_verbose.txt"):
        a = (left / name).read_text(encoding="utf-8")
        b = (right / name).read_text(encoding="utf-8")
        if a != b:
            diff = difflib.unified_diff(a.split("\n"), b.split("\n"), lineterm="", n=0)
            changed = [line for line in diff if line[:1] in "+-" and line[:3] not in "+++---"]
            problems.append(f"{name} differs: {short(changed[:4], 300)}")
    frame_a = pd.read_pickle(left / "dataframe.pkl")
    frame_b = pd.read_pickle(right / "dataframe.pkl")
    if not frame_a.equals(frame_b):
        problems.append("to_dataframe() frames differ (DataFrame.equals is False)")
    return problems


def check_metrics(baseline: Run, after: Run, pairs: dict[PairKey, tuple[str, str]]) -> Check:
    """Item 4: metrics, summaries and frames are identical for every config, except for new
    metrics keys at their defaults."""
    problems: list[str] = []
    seen_new: Counter[str] = Counter()
    compared = 0
    for key, (ours, theirs) in pairs.items():
        for sub in (pc.METRICS_SUBDIR, pc.CHECKPOINT_METRICS_SUBDIR):
            compared += 1
            found = compare_metrics_dirs(
                baseline.path / sub / ours, after.path / sub / theirs, seen_new
            )
            problems += [f"{key[0]}/{key[1]} ({sub}): {p}" for p in found]
    new_keys = ", ".join(f"{k} (x{n})" for k, n in sorted(seen_new.items())) or "none"
    return Check(
        "R4",
        "metrics identical (fields, summary text, frames)",
        status_of(len(problems)),
        f"{compared} metric sets compared (in-memory and from checkpoint); "
        f"{len(problems)} differences; new keys at defaults: {new_keys}",
        details=[f"- {p}" for p in problems[:MAX_DETAILS]],
        data={
            "compared": compared,
            "differences": len(problems),
            "new_keys_at_default": dict(seen_new),
        },
    )


def check_manifest(baseline: Run, after: Run, pairs: dict[PairKey, tuple[str, str]]) -> Check:
    """Item 5: ``grader_config`` changes additively."""
    problems: list[str] = []
    seen_new: Counter[str] = Counter()
    for key, (ours, theirs) in pairs.items():
        problems += additive_problems(
            baseline.grader_config(ours),
            after.grader_config(theirs),
            f"{key[0]}/{key[1]}.grader_config",
            new_keys=None,
            seen_new=seen_new,
        )
    new_keys = sorted({entry.split("=", 1)[0] for entry in seen_new})
    notes = []
    if "normalize" in new_keys:
        notes.append(
            "known exception: `normalize` is now recorded (the pre-existing `_normalize` read "
            "was fixed); it is a new key, not a changed value"
        )
    return Check(
        "R5",
        "manifest grader_config is a superset with equal values",
        status_of(len(problems)),
        f"{len(pairs)} manifests compared; {len(problems)} differences; new keys: "
        f"{', '.join(new_keys) or 'none'}",
        details=[f"- {n}" for n in notes] + [f"- {p}" for p in problems[:MAX_DETAILS]],
        data={"differences": len(problems), "new_keys": new_keys, "new_values": dict(seen_new)},
    )


def check_constructions(baseline: Run, after: Run) -> Check:
    """Item 6: every construction path processes identically; only the deprecated one warns."""
    problems: list[str] = []
    compared = 0
    for key, labels in sorted(after.by_key().items()):
        primary = labels.get(after.primary)
        if primary is None:
            continue
        for construction, label in sorted(labels.items()):
            if label == primary:
                continue
            compared += 1
            where = f"{key[0]}/{key[1]} {construction} vs {after.primary}"
            left, right = after.checkpoint(primary), after.checkpoint(label)
            if left.keys() != right.keys():
                problems.append(f"{where}: item sets differ")
            for item_idx in sorted(left.keys() & right.keys()):
                a, b = masked_record(left[item_idx], set()), masked_record(right[item_idx], set())
                problems += [f"{where} item {item_idx} {p}" for p in diff_paths(a, b)]
            for sub in (pc.METRICS_SUBDIR, pc.CHECKPOINT_METRICS_SUBDIR):
                found = compare_metrics_dirs(after.path / sub / primary, after.path / sub / label)
                problems += [f"{where} ({sub}): {p}" for p in found]
            if after.mode == "synthetic":
                a_log = [_projection(r) for r in after.requests_for(primary)]
                b_log = [_projection(r) for r in after.requests_for(label)]
                if a_log != b_log:
                    problems.append(f"{where}: recorded request logs differ")
            else:
                if attempted_triples(after, primary) != attempted_triples(after, label):
                    problems.append(f"{where}: network attempts differ")

    warn_problems, warn_details = _warning_checks(baseline, after)
    problems += warn_problems
    constructions = sorted({c for labels in after.by_key().values() for c in labels})
    summary = (
        f"constructions {', '.join(constructions)}; {compared} alternative passes compared "
        f"with the primary; {len(problems)} problems"
    )
    return Check(
        "R6",
        "construction paths agree; only the llm_config= path warns",
        status_of(len(problems)),
        summary,
        details=warn_details + [f"- {p}" for p in problems[:MAX_DETAILS]],
        data={"alternative_passes": compared, "problems": len(problems)},
    )


def _warning_checks(baseline: Run, after: Run) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    details: list[str] = []
    rows = [w for w in after.warnings() if w["is_deprecation"]]
    preexisting = {
        (w["category"], w["message"], w["autorubric_relpath"])
        for w in baseline.warnings()
        if w["is_deprecation"] and w["in_autorubric"]
    }
    for w in rows:
        if not w["in_autorubric"]:
            continue
        signature = (w["category"], w["message"], w["autorubric_relpath"])
        if signature in preexisting:
            details.append(f"- pre-existing in the baseline too: {short(signature, 200)}")
            continue
        problems.append(
            f"{w['pass']} ({w['phase']}): {w['category']} attributed to autorubric/"
            f"{w['autorubric_relpath']}:{w['lineno']}: {short(w['message'])}"
        )
    llm_config_rows = [w for w in rows if "llm_config" in w["message"]]
    if after.code == "after":
        construction_of = {p["pass"]: p for p in after.meta["passes"]}
        warned = {w["pass"] for w in llm_config_rows if w["phase"] == "grade"}
        for label, entry in construction_of.items():
            config = pc.CONFIGS[entry["config_id"]]
            expected = entry["construction"] == pc.DEPRECATED_CONSTRUCTION and not (
                config.is_ensemble
            )
            if expected and label not in warned:
                problems.append(f"{label}: CriterionGrader(llm_config=...) did not warn")
            if not expected and label in warned:
                problems.append(f"{label}: unexpected llm_config DeprecationWarning")
        sources = sorted({f"{Path(w['filename']).name}:{w['lineno']}" for w in llm_config_rows})
        details.append(
            f"- llm_config DeprecationWarnings: {len(llm_config_rows)} distinct, attributed to "
            f"{', '.join(sources) or 'nothing'}"
        )
    elif llm_config_rows:
        problems.append(f"{len(llm_config_rows)} llm_config DeprecationWarnings on {after.code}")
    return problems, details


def check_old_checkpoints(baseline: Run, after: Run) -> Check:
    """Item 7: the baseline's checkpoints load, re-score identically and resume with no work."""
    result_path = after.path / pc.CHECKPOINT_COMPAT_SUBDIR / "result.json"
    if after.meta.get("checkpoint_source") != baseline.run_id or not result_path.exists():
        return Check(
            "R7",
            "old checkpoints load, re-score and resume with no grading",
            "fail",
            f"{after.run_id} has no checkpoint check against {baseline.run_id} "
            f"(run it with --checkpoint-source {baseline.run_id})",
        )
    result = pc.read_json(result_path)
    problems: list[str] = []
    seen_new: Counter[str] = Counter()
    by_pass = {row["pass"]: row for row in result["passes"]}
    for label in sorted(baseline.labels()):
        row = by_pass.get(label)
        if row is None or row.get("status") != "completed":
            problems.append(f"{label}: not loaded/resumed ({(row or {}).get('status')})")
            continue
        if row["items_graded"] != 0 or row["network_attempts"] != 0:
            problems.append(
                f"{label}: resume graded {row['items_graded']} items with "
                f"{row['network_attempts']} network attempts"
            )
        if row["resumed_items"] != row["total_items"]:
            problems.append(f"{label}: {row['resumed_items']} of {row['total_items']} items")
        found = compare_metrics_dirs(
            baseline.path / pc.CHECKPOINT_METRICS_SUBDIR / label,
            after.path / pc.CHECKPOINT_COMPAT_SUBDIR / pc.METRICS_SUBDIR / label,
            seen_new,
        )
        problems += [f"{label}: {p}" for p in found]
    new_keys = ", ".join(f"{k} (x{n})" for k, n in sorted(seen_new.items())) or "none"
    return Check(
        "R7",
        "old checkpoints load, re-score and resume with no grading",
        status_of(len(problems)),
        f"{len(by_pass)} baseline experiments loaded and resumed; {len(problems)} problems; "
        f"new metrics keys at defaults: {new_keys}",
        details=[f"- {p}" for p in problems[:MAX_DETAILS]],
        data={
            "experiments": len(by_pass),
            "problems": len(problems),
            "new_keys_at_default": dict(seen_new),
        },
    )


def test_r(baseline: Run, after: Run, kind: Literal["replay", "synthetic"]) -> list[Check]:
    """Test R items 1-7."""
    pairs, pairing_problems = pair_passes(baseline, after)
    checks: list[Check] = []
    if pairing_problems:
        checks.append(
            Check(
                "R0",
                "every pass completed in both runs",
                "fail",
                f"{len(pairing_problems)} passes missing",
                details=[f"- {p}" for p in pairing_problems],
            )
        )
    if kind == "synthetic":
        checks.append(check_request_logs(baseline, after, pairs))
    else:
        checks.append(check_network(baseline, after, pairs))
    masking = kind == "replay"
    checks.append(check_processing(baseline, after, pairs, masking))
    checks.append(check_serialization(baseline, after, pairs, masking))
    checks.append(check_metrics(baseline, after, pairs))
    checks.append(check_manifest(baseline, after, pairs))
    checks.append(check_constructions(baseline, after))
    checks.append(check_old_checkpoints(baseline, after))
    return checks


# =======================================================================================
# Test L
# =======================================================================================

LIVE_ROLES = ("B1", "B2", "A1", "A2")
WITHIN = (("B1", "B2"), ("A1", "A2"))
CROSS = (("B1", "A1"), ("B1", "A2"), ("B2", "A1"), ("B2", "A2"))
UnitKey = tuple[str, str, int, int, str]
"""``(config, dataset, item, criterion, judge)``."""


def live_votes(run: Run) -> dict[UnitKey, pc.VoteRow]:
    """Votes of a run's paid configs (derived configs reuse the same calls)."""
    table: dict[UnitKey, pc.VoteRow] = {}
    for config_id in pc.PAID_CONFIGS:
        for dataset_id in pc.CONFIGS[config_id].datasets:
            label = run.primary_label((config_id, dataset_id))
            if label is None:
                continue
            for record in run.checkpoint(label).values():
                for vote in pc.vote_rows(record):
                    key = (config_id, dataset_id, vote.item_idx, vote.criterion_idx, vote.judge_id)
                    table[key] = vote
    return table


def cluster_bootstrap_disagreement(
    units: list[tuple[tuple[str, int], dict[str, pc.VoteRow]]], rng: np.random.Generator
) -> dict[str, Any]:
    """Within/cross-phase disagreement with a one-sided cluster bootstrap over items."""
    clusters = sorted({cluster for cluster, _ in units})
    if not clusters:
        return {"n_units": 0, "n_items": 0, "passed": None}
    index = {cluster: i for i, cluster in enumerate(clusters)}
    pairs = WITHIN + CROSS
    n = np.zeros(len(clusters))
    k = np.zeros((len(pairs), len(clusters)))
    mad_sum = np.zeros(len(pairs))
    mad_n = np.zeros(len(pairs))
    for cluster, votes in units:
        c = index[cluster]
        n[c] += 1
        for p, (x, y) in enumerate(pairs):
            k[p, c] += votes[x].category != votes[y].category
            vx, vy = votes[x], votes[y]
            if (
                vx.scale_type == "ordinal"
                and "NA" not in (vx.category, vy.category)
                and vx.value is not None
                and vy.value is not None
            ):
                mad_sum[p] += abs(vx.value - vy.value)
                mad_n[p] += 1
    d = k.sum(axis=1) / n.sum()
    within, cross = d[: len(WITHIN)].mean(), d[len(WITHIN) :].mean()
    delta = float(cross - within)
    draws = rng.integers(0, len(clusters), size=(N_BOOTSTRAP, len(clusters)))
    totals = n[draws].sum(axis=1)
    d_star = np.stack([k[p][draws].sum(axis=1) / totals for p in range(len(pairs))])
    delta_star = d_star[len(WITHIN) :].mean(axis=0) - d_star[: len(WITHIN)].mean(axis=0)
    p_value = float(np.mean(delta_star <= 0))
    lower = float(np.quantile(delta_star, ALPHA))
    significant = p_value < ALPHA
    mad = np.divide(mad_sum, mad_n, out=np.full(len(pairs), np.nan), where=mad_n > 0)
    return {
        "n_units": int(n.sum()),
        "n_items": len(clusters),
        "disagreement": {f"{x}-{y}": float(d[p]) for p, (x, y) in enumerate(pairs)},
        "within": float(within),
        "cross": float(cross),
        "delta": delta,
        "p_value": p_value,
        "lower_bound": lower,
        "significant": significant,
        "passed": (not significant) and delta <= MAX_DISAGREEMENT_EXCESS,
        "ordinal_mad_within": _nanmean(mad[: len(WITHIN)]),
        "ordinal_mad_cross": _nanmean(mad[len(WITHIN) :]),
    }


def _nanmean(values: np.ndarray) -> float | None:
    finite = values[~np.isnan(values)]
    return float(finite.mean()) if finite.size else None


def error_rate_bootstrap(
    rows_by_run: dict[str, dict[UnitKey, pc.VoteRow]],
    after_role: str,
    base_role: str,
    category: str | None,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Error-rate difference (after minus baseline) and its one-sided p-value by items."""
    clusters: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for role, sign in ((after_role, 0), (base_role, 2)):
        for key, vote in rows_by_run[role].items():
            slot = clusters[(key[1], key[2])]
            slot[sign] += 1
            if vote.error is not None and category in (None, vote.error_category):
                slot[sign + 1] += 1
    if not clusters:
        return 0.0, 0.0, 1.0
    table = np.array(list(clusters.values()))
    rate_after = table[:, 1].sum() / max(table[:, 0].sum(), 1)
    rate_base = table[:, 3].sum() / max(table[:, 2].sum(), 1)
    draws = rng.integers(0, len(table), size=(N_BOOTSTRAP, len(table)))
    sample = table[draws]
    diff_star = sample[..., 1].sum(1) / np.maximum(sample[..., 0].sum(1), 1) - sample[..., 3].sum(
        1
    ) / np.maximum(sample[..., 2].sum(1), 1)
    return float(rate_after), float(rate_base), float(np.mean(diff_star <= 0))


def error_rate_ci(
    rows: dict[UnitKey, pc.VoteRow], rng: np.random.Generator
) -> tuple[float, float, float]:
    """Error rate of a run's votes with a 95% cluster-bootstrap interval."""
    clusters: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for key, vote in rows.items():
        clusters[(key[1], key[2])][0] += 1
        clusters[(key[1], key[2])][1] += vote.error is not None
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    table = np.array(list(clusters.values()))
    rate = table[:, 1].sum() / table[:, 0].sum()
    draws = rng.integers(0, len(table), size=(N_BOOTSTRAP, len(table)))
    star = table[draws][..., 1].sum(1) / table[draws][..., 0].sum(1)
    return float(rate), float(np.quantile(star, 0.025)), float(np.quantile(star, 0.975))


def test_l(runs: dict[str, Run]) -> list[Check]:
    """Test L: live parity modulo LLM variance, plus side-by-side aggregates and flags."""
    rng = np.random.default_rng(TEST_SEED)
    votes = {role: live_votes(runs[role]) for role in LIVE_ROLES}
    all_keys = set().union(*(v.keys() for v in votes.values()))
    included: list[tuple[UnitKey, dict[str, pc.VoteRow]]] = []
    excluded = Counter()
    for key in sorted(all_keys):
        per_run = {role: votes[role].get(key) for role in LIVE_ROLES}
        if any(v is None for v in per_run.values()):
            excluded["missing in a run"] += 1
            continue
        if any(v.error is not None for v in per_run.values() if v is not None):
            excluded["errored in a run"] += 1
            continue
        included.append((key, {r: v for r, v in per_run.items() if v is not None}))

    checks: list[Check] = []
    scopes: list[tuple[str, str | None]] = [("L", None)] + [
        (f"L:{cid}", cid) for cid in pc.PAID_CONFIGS
    ]
    for item, config_id in scopes:
        units = [
            ((key[1], key[2]), per_run)
            for key, per_run in included
            if config_id is None or key[0] == config_id
        ]
        result = cluster_bootstrap_disagreement(units, rng)
        gating = config_id is None
        if result["passed"] is None:
            checks.append(Check(item, "live parity", "n/a", "no comparable units", gating))
            continue
        summary = (
            f"w={result['within']:.4f} c={result['cross']:.4f} c-w={result['delta']:+.4f} "
            f"(p={result['p_value']:.4f}, 95% lower bound {result['lower_bound']:+.4f}); "
            f"{result['n_units']} units over {result['n_items']} items"
        )
        details = [
            "- pairwise disagreement: "
            + ", ".join(f"{k}={v:.4f}" for k, v in result["disagreement"].items()),
            f"- ordinal mean |value difference|: within={result['ordinal_mad_within']}, "
            f"cross={result['ordinal_mad_cross']}",
        ]
        if config_id is None:
            details.append(f"- excluded units: {dict(excluded) or 'none'}")
        title = "live parity overall" if gating else f"live parity for {config_id}"
        checks.append(
            Check(
                item,
                title,
                "pass" if result["passed"] else "fail",
                summary,
                gating=gating,
                details=details,
                data=result,
            )
        )
    checks.append(aggregate_side_by_side(runs, votes, rng))
    checks.append(error_flags(votes, rng))
    checks.append(provider_versions(runs))
    return checks


def aggregate_side_by_side(
    runs: dict[str, Run], votes: dict[str, dict[UnitKey, pc.VoteRow]], rng: np.random.Generator
) -> Check:
    """Accuracy, kappa, RMSE and error rate per config and run, with flags."""
    lines = [
        "| config | dataset | run | accuracy [95% CI] | mean kappa [95% CI] | "
        "score RMSE [95% CI] | error rate [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    flags: list[str] = []
    data: dict[str, Any] = {}
    for config_id, config in pc.CONFIGS.items():
        for dataset_id in config.datasets:
            stats: dict[str, dict[str, Any]] = {}
            for role in LIVE_ROLES:
                run = runs[role]
                label = run.primary_label((config_id, dataset_id))
                if label is None:
                    continue
                metrics = pc.read_json(run.path / pc.METRICS_SUBDIR / label / "metrics.json")
                boot = metrics.get("bootstrap") or {}
                rate = error_rate_ci(live_votes_for(run, label, config_id, dataset_id), rng)
                stats[role] = {
                    "accuracy": metrics.get("criterion_accuracy"),
                    "accuracy_ci": boot.get("accuracy_ci"),
                    "kappa": metrics.get("mean_kappa"),
                    "kappa_ci": boot.get("kappa_ci"),
                    "rmse": metrics.get("score_rmse"),
                    "rmse_ci": boot.get("rmse_ci"),
                    "error_rate": rate,
                }
                lines.append(
                    f"| {config_id} | {dataset_id} | {role} | "
                    f"{_fmt(stats[role]['accuracy'], stats[role]['accuracy_ci'])} | "
                    f"{_fmt(stats[role]['kappa'], stats[role]['kappa_ci'])} | "
                    f"{_fmt(stats[role]['rmse'], stats[role]['rmse_ci'])} | "
                    f"{_fmt(rate[0], rate[1:])} |"
                )
            data[f"{config_id}/{dataset_id}"] = stats
            for role in ("A1", "A2"):
                for metric in ("accuracy", "kappa"):
                    value = (stats.get(role) or {}).get(metric)
                    bounds = [
                        ((stats.get(b) or {}).get(f"{metric}_ci") or [None])[0]
                        for b in ("B1", "B2")
                    ]
                    if value is not None and all(x is not None and value < x for x in bounds):
                        flags.append(
                            f"{config_id}/{dataset_id} {role}: {metric} {value:.4f} is below the "
                            f"lower CI bound of both baseline runs ({bounds[0]:.4f}, "
                            f"{bounds[1]:.4f})"
                        )
    return Check(
        "L-aggregate",
        "aggregate side-by-side",
        "fail" if flags else "pass",
        f"{len(flags)} flags (an after run below both baseline lower CI bounds)",
        gating=False,
        details=[*lines, "", *[f"- FLAG: {f}" for f in flags]],
        data={"stats": data, "flags": flags},
    )


def live_votes_for(
    run: Run, label: str, config_id: str, dataset_id: str
) -> dict[UnitKey, pc.VoteRow]:
    """Votes of one pass keyed like ``live_votes``."""
    return {
        (config_id, dataset_id, v.item_idx, v.criterion_idx, v.judge_id): v
        for record in run.checkpoint(label).values()
        for v in pc.vote_rows(record)
    }


def _fmt(value: Any, interval: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "None"
    text = f"{value:.4f}"
    if interval and all(isinstance(x, (int, float)) for x in interval):
        text += f" [{interval[0]:.4f}, {interval[1]:.4f}]"
    return text


def error_flags(votes: dict[str, dict[UnitKey, pc.VoteRow]], rng: np.random.Generator) -> Check:
    """Flag an after run whose error rate is significantly above both baseline runs."""
    lines = [
        "| scope | category | after | rate | vs B1 (rate, p) | vs B2 (rate, p) | flag |",
        "|---|---|---|---|---|---|---|",
    ]
    flags: list[str] = []
    scopes: list[str | None] = [None, *pc.PAID_CONFIGS]
    for scope in scopes:
        scoped = {
            role: {k: v for k, v in rows.items() if scope is None or k[0] == scope}
            for role, rows in votes.items()
        }
        for category in (None, "infrastructure", "parse", "unknown"):
            for after_role in ("A1", "A2"):
                results = [
                    error_rate_bootstrap(scoped, after_role, base, category, rng)
                    for base in ("B1", "B2")
                ]
                flagged = all(p < ALPHA for _, _, p in results)
                name = category or "any"
                lines.append(
                    f"| {scope or 'all'} | {name} | {after_role} | {results[0][0]:.4f} | "
                    f"{results[0][1]:.4f}, {results[0][2]:.4f} | "
                    f"{results[1][1]:.4f}, {results[1][2]:.4f} | {'FLAG' if flagged else ''} |"
                )
                if flagged:
                    flags.append(f"{scope or 'all'}/{name}/{after_role}")
    return Check(
        "L-errors",
        "error rates versus both baseline runs",
        "fail" if flags else "pass",
        f"{len(flags)} flags (one-sided cluster bootstrap, alpha {ALPHA})",
        gating=False,
        details=lines + [f"- FLAG: {f}" for f in flags],
        data={"flags": flags},
    )


def provider_versions(runs: dict[str, Run]) -> Check:
    """Report provider-reported model versions, a known confounder for test L."""
    seen = {role: run.meta.get("provider_models") or {} for role, run in runs.items()}
    distinct = {json.dumps(v, sort_keys=True) for v in seen.values()}
    lines = [f"- {role}: {models}" for role, models in seen.items()]
    changed = len(distinct) > 1
    return Check(
        "L-models",
        "provider-reported model versions",
        "info",
        "changed between runs (a confounder for test L)" if changed else "unchanged",
        gating=False,
        details=lines,
    )


# =======================================================================================
# Command line
# =======================================================================================


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--parity-dir", type=Path, default=pc.PARITY_DIR)
    parser.add_argument("--report", type=Path, help="output stem (writes .md and .json)")
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ("replay", "synthetic"):
        command = sub.add_parser(kind)
        command.add_argument("--baseline", required=True)
        command.add_argument("--after", required=True)
    live = sub.add_parser("live")
    for role in LIVE_ROLES:
        live.add_argument(f"--{role.lower()}", required=True, dest=role)
    live.add_argument(
        "--allow-nonstandard-runs",
        action="store_true",
        help="accept runs that are not fresh baseline/after runs (to exercise the test)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    parity_dir: Path = args.parity_dir
    if args.kind == "live":
        roles = {role: Run.load(parity_dir, getattr(args, role)) for role in LIVE_ROLES}
    else:
        roles = {
            "baseline": Run.load(parity_dir, args.baseline),
            "after": Run.load(parity_dir, args.after),
        }
    problems, notes = comparability(list(roles.values()))
    checks: list[Check] = []
    if args.kind == "live":
        expected = {"B1": "baseline", "B2": "baseline", "A1": "after", "A2": "after"}
        for role, run in roles.items():
            if run.mode != "fresh" or run.code != expected[role]:
                message = f"{role}={run.run_id} is a {run.code} {run.mode} run"
                if args.allow_nonstandard_runs:
                    notes.append(f"{message} (accepted with --allow-nonstandard-runs)")
                else:
                    problems.append(f"{message}; test L needs fresh {expected[role]} runs")
    else:
        baseline, after = roles["baseline"], roles["after"]
        if args.kind == "replay":
            if baseline.mode != "fresh" or after.mode != "replay":
                problems.append("replay parity needs a fresh baseline and a replay run")
            elif after.meta.get("replay_source") != baseline.run_id:
                problems.append(f"{after.run_id} does not replay {baseline.run_id}")
        elif baseline.mode != "synthetic" or after.mode != "synthetic":
            problems.append("synthetic parity needs two synthetic runs")
    if not problems:
        if args.kind == "live":
            checks = test_l(roles)
        else:
            checks = test_r(roles["baseline"], roles["after"], args.kind)
    report = Report(args.kind, roles, checks, notes, problems)
    ids = "_".join(run.run_id for run in roles.values())
    stem = args.report or parity_dir / "reports" / f"{args.kind}__{ids}"
    report.write(stem)
    for check in checks:
        print(f"{check.item:12} {check.status.upper():5} {check.summary}")
    for problem in problems:
        print(f"NOT COMPARABLE: {problem}")
    verdict = "PASS" if report.passed else "FAIL"
    print(f"{verdict}; report written to {stem.with_suffix('.md')}")
    if problems:
        return 2
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
