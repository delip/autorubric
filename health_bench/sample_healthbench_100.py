#!/usr/bin/env python3
"""Sample two 100-item subsets from HealthBench meta for grader experimentation.

Produces ``healthbench-100-val`` and ``healthbench-100-test``: RubricDataset
JSONs derived from ``healthbench_meta.json`` with two extra guarantees over
the raw meta dataset:

1. **No CANNOT_ASSESS** in ground truth. CANNOT_ASSESS values come from 1-1
   physician ties during meta conversion; they contaminate accuracy metrics.
2. **Both submission and reference_submission populated**. ``submission`` is
   the model completion physicians graded; ``reference_submission`` is the
   physician-written ideal completion for the same prompt (joined on
   prompt_id against ``healthbench_physician_ideal.json``). Flip
   ``EvalConfig.use_reference_submission`` to toggle whether the judge sees
   the anchor.

Sampling is **theme x verdict_class** stratified with mixed/all-UNMET
oversampled so the subsets stress the grader rather than reflecting the raw
~74% all-MET dominance of the filtered pool. Deterministic at fixed seed.

Outputs (under ``autorubric_dataset/``):
- ``healthbench-100-val.json``       — 100 items
- ``healthbench-100-test.json``      — 100 items
- ``healthbench-100-manifest.json``  — seed, filters, cell counts, splits

Usage:
    cd health_bench
    uv run python sample_healthbench_100.py
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SEED = 20260517

DATA_DIR = Path(__file__).parent / "autorubric_dataset"
META_PATH = DATA_DIR / "healthbench_meta.json"
IDEAL_PATH = DATA_DIR / "healthbench_physician_ideal.json"
VAL_OUT = DATA_DIR / "healthbench-100-val.json"
TEST_OUT = DATA_DIR / "healthbench-100-test.json"
MANIFEST_OUT = DATA_DIR / "healthbench-100-manifest.json"

THEMES = [
    "communication",
    "complex_responses",
    "context_seeking",
    "emergency_referrals",
    "global_health",
    "health_data_tasks",
    "hedging",
]
VERDICT_CLASSES = ["all_met", "mixed", "all_unmet"]

# Per-class target across val+test combined. 66 + 68 + 66 = 200.
# Mixed gets the +2 to oversample the most-informative class.
CLASS_TARGETS = {"all_met": 66, "mixed": 68, "all_unmet": 66}
# Per-cell floor; capped by available pool size. Combined val+test value,
# so floor=4 yields 2 in val and 2 in test where the pool permits.
CELL_FLOOR = 4

PID_RE = re.compile(r"prompt_id=([0-9a-f-]+)")
THEME_RE = re.compile(r"theme:([a-z_]+)")


def parse_prompt_id(description: str) -> str:
    m = PID_RE.search(description)
    if not m:
        raise ValueError(f"no prompt_id in description: {description[:120]!r}")
    return m.group(1)


def parse_theme(description: str) -> str:
    m = THEME_RE.search(description)
    if not m:
        raise ValueError(f"no theme tag in description: {description[:120]!r}")
    return m.group(1)


def classify_verdicts(ground_truth: list[str]) -> str:
    has_met = "MET" in ground_truth
    has_unmet = "UNMET" in ground_truth
    if has_met and not has_unmet:
        return "all_met"
    if has_unmet and not has_met:
        return "all_unmet"
    return "mixed"


def hamilton_allocate(cell_sizes: dict[str, int], total: int, floor: int) -> dict[str, int]:
    """Allocate ``total`` slots across themes with a per-theme floor.

    Floors are applied first (capped at cell size; a cell with size < floor
    contributes all of its items). The remainder is distributed by Hamilton's
    largest-remainder method proportional to remaining headroom.
    """
    themes = list(cell_sizes.keys())
    sizes = dict(cell_sizes)
    alloc = {t: min(floor, sizes[t]) for t in themes}

    if sum(alloc.values()) >= total:
        # Floors already meet or exceed target — clamp uniformly by trimming
        # from the most generously-floored themes until we hit `total`.
        overshoot = sum(alloc.values()) - total
        for t in sorted(themes, key=lambda t: -alloc[t]):
            while overshoot > 0 and alloc[t] > 0:
                alloc[t] -= 1
                overshoot -= 1
            if overshoot == 0:
                break
        return alloc

    remaining = total - sum(alloc.values())
    headroom = {t: sizes[t] - alloc[t] for t in themes}
    total_headroom = sum(headroom.values())

    if total_headroom == 0:
        return alloc

    if remaining >= total_headroom:
        for t in themes:
            alloc[t] += headroom[t]
        return alloc

    quotas = {t: remaining * headroom[t] / total_headroom for t in themes}
    int_part = {t: int(quotas[t]) for t in themes}
    frac_part = {t: quotas[t] - int_part[t] for t in themes}
    for t in themes:
        alloc[t] += int_part[t]

    leftover = remaining - sum(int_part.values())
    # Tie-break on theme name keeps allocation deterministic across runs.
    sorted_t = sorted(themes, key=lambda t: (-frac_part[t], t))
    for t in sorted_t:
        if leftover == 0:
            break
        if alloc[t] < sizes[t]:
            alloc[t] += 1
            leftover -= 1

    return alloc


def split_cell(
    items: list[dict[str, Any]], take: int, rng: random.Random, val_deficit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Shuffle, take ``take`` items, split into (val, test).

    For odd ``take``, the side currently behind on count gets the extra item.
    ``val_deficit`` is (test_count_so_far - val_count_so_far); positive means
    val is behind and gets the +1.
    """
    if take == 0:
        return [], [], val_deficit
    pool = items[:]
    rng.shuffle(pool)
    chosen = pool[:take]
    half = take // 2
    if take % 2 == 0:
        val = chosen[:half]
        test = chosen[half:]
    else:
        if val_deficit >= 0:
            val = chosen[: half + 1]
            test = chosen[half + 1 :]
        else:
            val = chosen[:half]
            test = chosen[half:]
    new_deficit = val_deficit + (len(test) - len(val))
    return val, test, new_deficit


def build_item(meta_item: dict[str, Any], ideal_text: str) -> dict[str, Any]:
    """Clone a meta item and set reference_submission to the physician ideal."""
    item = {
        "submission": meta_item["submission"],
        "description": meta_item["description"],
        "prompt": meta_item["prompt"],
        "ground_truth": list(meta_item["ground_truth"]),
        "reference_submission": ideal_text,
        "rubric": [dict(c) for c in meta_item["rubric"]],
    }
    return item


def main() -> None:
    print(f"Loading {META_PATH.name}...")
    meta = json.loads(META_PATH.read_text())
    print(f"Loading {IDEAL_PATH.name}...")
    ideal = json.loads(IDEAL_PATH.read_text())

    # Build prompt_id -> (theme, ideal_text). The physician_ideal description
    # carries the theme tag, so this single dict serves both joins.
    ideal_lookup: dict[str, tuple[str, str]] = {}
    for it in ideal["items"]:
        pid = parse_prompt_id(it["description"])
        theme = parse_theme(it["description"])
        ideal_lookup[pid] = (theme, it["submission"])

    print(f"physician_ideal coverage: {len(ideal_lookup)} prompt_ids")

    # Filter + bucket
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    dropped_cannot_assess = 0
    dropped_no_ideal = 0
    kept = 0
    for it in meta["items"]:
        gt = it.get("ground_truth")
        if not gt:
            dropped_cannot_assess += 1
            continue
        if any(v == "CANNOT_ASSESS" or v is None for v in gt):
            dropped_cannot_assess += 1
            continue
        pid = parse_prompt_id(it["description"])
        if pid not in ideal_lookup:
            dropped_no_ideal += 1
            continue
        theme, _ = ideal_lookup[pid]
        vclass = classify_verdicts(gt)
        cells[(theme, vclass)].append(it)
        kept += 1

    pool_total = sum(len(v) for v in cells.values())
    assert kept == pool_total
    print(
        f"meta items: total={len(meta['items'])}  "
        f"dropped_cannot_assess={dropped_cannot_assess}  "
        f"dropped_no_ideal={dropped_no_ideal}  "
        f"kept={kept}"
    )

    # Sort each cell deterministically (by prompt_id, completion_id) before
    # the seeded shuffle, so re-runs of this script on the same input
    # produce byte-identical output even if dict iteration order changes.
    def _sort_key(it: dict[str, Any]) -> tuple[str, str]:
        d = it["description"]
        pid = parse_prompt_id(d)
        cid_match = re.search(r"completion_id=([0-9a-f-]+)", d)
        return (pid, cid_match.group(1) if cid_match else "")

    for k in cells:
        cells[k].sort(key=_sort_key)

    # Allocate per verdict class across themes
    allocations: dict[tuple[str, str], int] = {}
    for vclass in VERDICT_CLASSES:
        cell_sizes = {t: len(cells.get((t, vclass), [])) for t in THEMES}
        alloc = hamilton_allocate(cell_sizes, CLASS_TARGETS[vclass], CELL_FLOOR)
        for t, n in alloc.items():
            allocations[(t, vclass)] = n
        print(
            f"class={vclass:>9s}  target={CLASS_TARGETS[vclass]:>3d}  "
            f"sum_alloc={sum(alloc.values()):>3d}  "
            f"per_theme=" + str({t: alloc[t] for t in THEMES})
        )

    total_alloc = sum(allocations.values())
    print(f"total allocated: {total_alloc} (target 200)")

    # Split each cell. Track val_deficit so odd cells alternate fairly.
    rng = random.Random(SEED)
    val_items: list[dict[str, Any]] = []
    test_items: list[dict[str, Any]] = []
    cell_counts_val: dict[str, dict[str, int]] = {t: {} for t in THEMES}
    cell_counts_test: dict[str, dict[str, int]] = {t: {} for t in THEMES}
    val_deficit = 0

    # Iterate in a stable order so the deficit tracking is deterministic.
    for vclass in VERDICT_CLASSES:
        for theme in THEMES:
            take = allocations[(theme, vclass)]
            pool = cells.get((theme, vclass), [])
            v, t, val_deficit = split_cell(pool, take, rng, val_deficit)
            cell_counts_val[theme][vclass] = len(v)
            cell_counts_test[theme][vclass] = len(t)
            for it in v:
                ideal_text = ideal_lookup[parse_prompt_id(it["description"])][1]
                val_items.append(build_item(it, ideal_text))
            for it in t:
                ideal_text = ideal_lookup[parse_prompt_id(it["description"])][1]
                test_items.append(build_item(it, ideal_text))

    print(f"val items: {len(val_items)}  test items: {len(test_items)}")

    # Final balance enforcement: if 100/100 isn't exact, swap one item from
    # the over-full side to the under-full side. This should be unreachable
    # given the deficit-aware splitter but we guard anyway.
    if len(val_items) != 100 or len(test_items) != 100:
        raise RuntimeError(
            f"split is unbalanced: val={len(val_items)}, test={len(test_items)}. "
            f"Inspect allocations: {allocations}"
        )

    # Write RubricDataset JSONs
    val_ds = {"name": "healthbench-100-val", "rubric": None, "items": val_items}
    test_ds = {"name": "healthbench-100-test", "rubric": None, "items": test_items}
    VAL_OUT.write_text(json.dumps(val_ds, indent=2) + "\n")
    TEST_OUT.write_text(json.dumps(test_ds, indent=2) + "\n")
    print(f"wrote {VAL_OUT.name} ({VAL_OUT.stat().st_size:,} bytes)")
    print(f"wrote {TEST_OUT.name} ({TEST_OUT.stat().st_size:,} bytes)")

    # Manifest
    def _pair(it: dict[str, Any]) -> dict[str, str]:
        d = it["description"]
        return {
            "prompt_id": parse_prompt_id(d),
            "completion_id": re.search(r"completion_id=([0-9a-f-]+)", d).group(1),
        }

    manifest = {
        "seed": SEED,
        "source": "healthbench_meta.json",
        "reference_join_source": "healthbench_physician_ideal.json",
        "filters": {
            "drop_cannot_assess_ground_truth": True,
            "require_physician_ideal_match": True,
        },
        "pool_sizes": {
            "raw_meta_items": len(meta["items"]),
            "dropped_cannot_assess": dropped_cannot_assess,
            "dropped_no_physician_ideal": dropped_no_ideal,
            "after_filters": kept,
        },
        "stratification": {
            "axes": ["theme", "verdict_class"],
            "class_targets": CLASS_TARGETS,
            "cell_floor_combined": CELL_FLOOR,
            "themes": THEMES,
            "verdict_classes": VERDICT_CLASSES,
        },
        "cell_counts": {"val": cell_counts_val, "test": cell_counts_test},
        "splits": {
            "val": [_pair(it) for it in val_items],
            "test": [_pair(it) for it in test_items],
        },
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST_OUT.name} ({MANIFEST_OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
