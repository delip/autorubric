#!/usr/bin/env python3
"""Convert HealthBench raw JSONL into AutoRubric RubricDataset format.

Source data (downloaded to ``raw_data/`` by the original ``simple-evals``
release; see ``README.md``):
- ``healthbench_main.jsonl``      — 5,000 prompts (the canonical pool)
- ``healthbench_hard.jsonl``      — 1,000 prompt_ids (subset of main)
- ``healthbench_consensus.jsonl`` — 3,671 prompt_ids (subset; rubrics filtered
                                    to cluster-level criteria only)
- ``healthbench_meta.jsonl``      — 29,511 (prompt, completion, criterion,
                                    physician-labels) rows for grader
                                    meta-evaluation

Per the design plan (`.claude/plans/...md`), main is the single source of
truth: the hard / consensus / meta files are all references into the same
5,000-prompt pool. We emit:

RubricDataset JSONs (under ``autorubric_dataset/``):
- ``healthbench.json``                       — 5,000 items, full rubrics,
                                               empty submission (template).
- ``healthbench_physician_ideal.json``       — 4,206 items (Groups 1+2+3),
                                               submission = ideal_completion.
- ``healthbench_physician_references.json``  — ~11k items (Groups 2+3, 4×
                                               fan-out), submission = ref
                                               completion, reference_submission
                                               = physician ideal.
- ``healthbench_meta.json``                  — 14,592 items (one per unique
                                               completion), per-item rubric
                                               of applicable consensus criteria,
                                               ground_truth = majority vote
                                               (ties → CANNOT_ASSESS).
- ``test_rubric_dataset.json``               — first 2 main items, smoke test.

Sidecar files (non-RubricDataset):
- ``healthbench_tags.jsonl``        — per-criterion tags, keyed by
                                      (prompt_id, criterion_idx). Out-of-band
                                      to keep ``Criterion.name`` clean for
                                      rubric-improvement substring matching.
- ``hard_prompt_ids.json``          — 1,000 UUIDs to reconstruct the hard split.
- ``consensus_prompt_ids.json``     — 3,671 UUIDs to reconstruct the consensus
                                      prompt subset (filter rubric to
                                      level:cluster via the tags sidecar).
- ``consensus_criteria_catalog.json`` — 37 entries: (cluster_id, criterion_text,
                                      axis, points). Reference catalog only.
- ``meta_physician_labels.jsonl``   — long-form raw labels (one row per
                                      (prompt_id, completion_id, criterion_name,
                                      physician_id, label)) so the published
                                      Macro-F1 grader-evaluation metric stays
                                      reproducible.

Conversation rendering follows simple-evals' format byte-for-byte:
``"\\n\\n".join(f"{m['role']}: {m['content']}" for m in messages)``.

Usage:
    cd health_bench
    uv run python convert_to_rubric_dataset.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

GENERIC_PROMPT = (
    "You are an assistant helping with a health-related conversation. "
    "Respond helpfully to the user's last message in the conversation."
)

MAIN_FILE = "healthbench_main.jsonl"
HARD_FILE = "healthbench_hard.jsonl"
CONSENSUS_FILE = "healthbench_consensus.jsonl"
META_FILE = "healthbench_meta.jsonl"


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON record from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(out_path: Path, payload: Any) -> None:
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_jsonl(out_path: Path, rows: Iterator[dict[str, Any]]) -> int:
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Field builders
# --------------------------------------------------------------------------- #


def render_conversation(messages: list[dict[str, Any]]) -> str:
    """Match simple-evals/healthbench_eval.py:408-410 byte-for-byte."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def make_criterion(idx: int, rubric_item: dict[str, Any]) -> dict[str, Any]:
    """Build an AutoRubric Criterion dict from a HealthBench rubric item.

    Name is plain ``C{i+1}`` — tags go to the sidecar.
    """
    return {
        "name": f"C{idx + 1}",
        "weight": rubric_item["points"],
        "requirement": rubric_item["criterion"],
    }


def make_per_item_rubric(rubrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [make_criterion(i, r) for i, r in enumerate(rubrics)]


def make_description(row: dict[str, Any], *extras: str) -> str:
    """Pipe-separated description carrying prompt_id, example tags, extras."""
    parts = [f"prompt_id={row['prompt_id']}"]
    parts.extend(row.get("example_tags") or [])
    parts.extend(extras)
    return " | ".join(parts)


def majority_verdict(labels: list[bool]) -> str:
    """Strict-majority physician vote → CriterionVerdict string.

    Ties are returned as ``CANNOT_ASSESS`` to flag ambiguity (default
    autorubric ``SKIP`` strategy will exclude these from the denominator).
    """
    n = len(labels)
    t = sum(labels)
    if t * 2 > n:
        return "MET"
    if t * 2 < n:
        return "UNMET"
    return "CANNOT_ASSESS"


# --------------------------------------------------------------------------- #
# Dataset builders
# --------------------------------------------------------------------------- #


def build_main_template(main_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """5,000 items, rubric only, empty submission (template)."""
    items = []
    for row in main_rows:
        items.append(
            {
                "submission": "",
                "description": make_description(row),
                "prompt": render_conversation(row["prompt"]),
                "ground_truth": None,
                "rubric": make_per_item_rubric(row["rubrics"]),
            }
        )
    return {
        "name": "healthbench",
        "rubric": None,
        "items": items,
    }


def build_physician_ideal(main_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """4,206 items (rows where ideal_completions_data is present)."""
    items = []
    for row in main_rows:
        ic = row.get("ideal_completions_data")
        if not ic:
            continue
        group = ic["ideal_completions_group"]
        items.append(
            {
                "submission": ic["ideal_completion"],
                "description": make_description(row, f"ideal_group={group}"),
                "prompt": render_conversation(row["prompt"]),
                "ground_truth": None,
                "rubric": make_per_item_rubric(row["rubrics"]),
            }
        )
    return {
        "name": "healthbench_physician_ideal",
        "rubric": None,
        "items": items,
    }


def build_physician_references(main_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """~11k items: fan-out of the 4 reference completions per prompt (G2+G3).

    ``reference_submission`` = the physician ideal completion (calibration
    anchor). Whether the grader sees it is controlled by
    ``EvalConfig.use_reference_submission``.
    """
    items = []
    for row in main_rows:
        ic = row.get("ideal_completions_data")
        if not ic:
            continue
        refs = ic.get("ideal_completions_ref_completions") or []
        group = ic["ideal_completions_group"]
        ideal = ic["ideal_completion"]
        for i, ref in enumerate(refs):
            items.append(
                {
                    "submission": ref,
                    "description": make_description(row, f"ref_group={group}", f"ref_idx={i}"),
                    "prompt": render_conversation(row["prompt"]),
                    "ground_truth": None,
                    "reference_submission": ideal,
                    "rubric": make_per_item_rubric(row["rubrics"]),
                }
            )
    return {
        "name": "healthbench_physician_references",
        "rubric": None,
        "items": items,
    }


def build_meta(meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One DataItem per unique (prompt_id, completion_id).

    Per-item rubric contains all consensus criteria evaluated for that
    completion (1–3 in practice). ``ground_truth`` is the per-criterion
    physician majority vote, length-matched to the rubric. Ties become
    ``CANNOT_ASSESS``.
    """
    # Group rows by (prompt_id, completion_id); each group becomes one item.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in meta_rows:
        grouped[(row["prompt_id"], row["completion_id"])].append(row)

    items = []
    for (prompt_id, completion_id), rows in grouped.items():
        first = rows[0]
        # Each row contributes one (rubric_text, category, labels) tuple.
        # The completion + prompt should be identical across rows for the
        # same completion_id (verified during cross-reference checks).
        criteria = []
        ground_truth = []
        rubric_meta = []  # for description
        for i, r in enumerate(rows):
            criteria.append(
                {
                    "name": f"C{i + 1}",
                    "weight": 10,  # meta-eval has no points; uniform binary
                    "requirement": r["rubric"],
                }
            )
            ground_truth.append(majority_verdict(r["binary_labels"]))
            rubric_meta.append(f"C{i + 1}:{r['category']}:n_phys={len(r['binary_labels'])}")
        items.append(
            {
                "submission": first["completion"],
                "description": (
                    f"prompt_id={prompt_id} | "
                    f"completion_id={completion_id} | "
                    f"criteria=[{'; '.join(rubric_meta)}]"
                ),
                "prompt": render_conversation(first["prompt"]),
                "ground_truth": ground_truth,
                "rubric": criteria,
            }
        )
    return {
        "name": "healthbench_meta",
        "rubric": None,
        "items": items,
    }


# --------------------------------------------------------------------------- #
# Sidecar builders
# --------------------------------------------------------------------------- #


def emit_tags_sidecar(main_rows: list[dict[str, Any]], out_path: Path) -> int:
    """One row per (prompt_id, criterion_idx) → tags list.

    Covers all four RubricDataset outputs because they share the same
    prompt_id → rubric mapping.
    """

    def rows():
        for row in main_rows:
            for i, rb in enumerate(row["rubrics"]):
                tags = rb.get("tags") or []
                yield {
                    "prompt_id": row["prompt_id"],
                    "criterion_idx": i,
                    "tags": tags,
                }

    return write_jsonl(out_path, rows())


def emit_consensus_catalog(consensus_rows: list[dict[str, Any]], out_path: Path) -> int:
    """37 entries: (cluster_id, criterion_text, axis, points)."""
    catalog: dict[str, dict[str, Any]] = {}
    for row in consensus_rows:
        for rb in row["rubrics"]:
            tags = rb.get("tags") or []
            cluster = next((t for t in tags if t.startswith("cluster:")), None)
            axis = next((t for t in tags if t.startswith("axis:")), None)
            if cluster is None:
                continue
            if cluster not in catalog:
                catalog[cluster] = {
                    "cluster_id": cluster,
                    "criterion": rb["criterion"],
                    "axis": axis,
                    "points": rb["points"],
                }
    entries = sorted(catalog.values(), key=lambda e: e["cluster_id"])
    write_json(out_path, entries)
    return len(entries)


def emit_meta_physician_labels(meta_rows: list[dict[str, Any]], out_path: Path) -> int:
    """Long-form per-physician labels for Macro-F1 reproduction.

    Each row: (prompt_id, completion_id, criterion_name, physician_id, label).
    Criterion names match the names autorubric assigns inside the
    healthbench_meta.json per-item rubrics — i.e. ``C{i+1}`` where ``i`` is
    the index of this criterion within the (prompt_id, completion_id)
    group used by ``build_meta``.
    """
    # Re-derive the same grouping ``build_meta`` uses, so criterion_name
    # stays consistent between the RubricDataset and the sidecar.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in meta_rows:
        grouped[(row["prompt_id"], row["completion_id"])].append(row)

    def rows():
        for (prompt_id, completion_id), group in grouped.items():
            for i, r in enumerate(group):
                criterion_name = f"C{i + 1}"
                category = r["category"]
                labels = r["binary_labels"]
                phys_ids = r["anonymized_physician_ids"]
                for phys_id, label in zip(phys_ids, labels, strict=True):
                    yield {
                        "prompt_id": prompt_id,
                        "completion_id": completion_id,
                        "criterion_name": criterion_name,
                        "category": category,
                        "physician_id": phys_id,
                        "label": label,
                    }

    return write_jsonl(out_path, rows())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    base = Path(__file__).parent
    raw = base / "raw_data"
    out = base / "autorubric_dataset"
    out.mkdir(exist_ok=True)

    print("== load raw ==")
    main_rows = load_jsonl(raw / MAIN_FILE)
    hard_rows = load_jsonl(raw / HARD_FILE)
    cons_rows = load_jsonl(raw / CONSENSUS_FILE)
    meta_rows = load_jsonl(raw / META_FILE)
    print(
        f"  main={len(main_rows)} hard={len(hard_rows)} cons={len(cons_rows)} meta={len(meta_rows)}"
    )

    print("\n== build RubricDataset JSONs ==")

    template = build_main_template(main_rows)
    write_json(out / "healthbench.json", template)
    print(f"  healthbench.json: {len(template['items'])} items")

    pi = build_physician_ideal(main_rows)
    write_json(out / "healthbench_physician_ideal.json", pi)
    print(f"  healthbench_physician_ideal.json: {len(pi['items'])} items")

    pr = build_physician_references(main_rows)
    write_json(out / "healthbench_physician_references.json", pr)
    print(f"  healthbench_physician_references.json: {len(pr['items'])} items")

    meta_ds = build_meta(meta_rows)
    write_json(out / "healthbench_meta.json", meta_ds)
    print(f"  healthbench_meta.json: {len(meta_ds['items'])} items")

    # Tiny smoke-test dataset
    test_ds = {**template, "items": template["items"][:2], "name": "healthbench_test"}
    write_json(out / "test_rubric_dataset.json", test_ds)
    print(f"  test_rubric_dataset.json: {len(test_ds['items'])} items")

    print("\n== sidecars ==")

    n = emit_tags_sidecar(main_rows, out / "healthbench_tags.jsonl")
    print(f"  healthbench_tags.jsonl: {n} (prompt_id, criterion_idx) rows")

    hard_ids = sorted({r["prompt_id"] for r in hard_rows})
    write_json(out / "hard_prompt_ids.json", hard_ids)
    print(f"  hard_prompt_ids.json: {len(hard_ids)} ids")

    cons_ids = sorted({r["prompt_id"] for r in cons_rows})
    write_json(out / "consensus_prompt_ids.json", cons_ids)
    print(f"  consensus_prompt_ids.json: {len(cons_ids)} ids")

    n = emit_consensus_catalog(cons_rows, out / "consensus_criteria_catalog.json")
    print(f"  consensus_criteria_catalog.json: {n} entries")

    n = emit_meta_physician_labels(meta_rows, out / "meta_physician_labels.jsonl")
    print(f"  meta_physician_labels.jsonl: {n} (criterion × physician) rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
