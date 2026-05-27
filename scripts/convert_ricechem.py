#!/usr/bin/env python3
"""Convert RiceChem dataset to AutoRubric RubricDataset JSON files."""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autorubric.dataset import DataItem, RubricDataset
from autorubric.rubric import Rubric
from autorubric.types import Criterion, CriterionVerdict

RAW_DIR = Path(__file__).resolve().parent.parent / "ricechem-rawdata"
OUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "data" / "ricechem"

# Columns to exclude per question (negative/flag columns)
EXCLUDE_COLS = {
    1: {"incorrect", "Blank", "Core charge calculation error"},
    2: {"Incorrect statement included", "Incorrect", "Blank"},
    3: {"Correct response", "Incorrect/Blank response"},
    4: {"incorrect/misleading statement", "incorrect/missing answer"},
}

# Non-criterion columns present in every graded rubric
META_COLS = {"SID", "Score", "Adjustment", "Comments"}

# Short names for criteria (matching plan order)
SHORT_NAMES = {
    1: [
        "decreased_repulsion",
        "repulsion_potential_energy",
        "same_core_charge",
        "same_shell_radius",
        "higher_core_charge",
        "smaller_radius",
        "full_pe_ie_explanation",
        "partial_pe_ie_explanation",
    ],
    2: [
        "freq_proportional_energy",
        "energy_levels_quantized",
        "full_energy_freq",
        "partial_energy_freq",
        "min_energy_eject",
        "additional_kinetic",
    ],
    3: [
        "sentence1_vbt_half_filled",
        "sentence2_correct_number",
        "sentence2_correct_type_sp2",
        "sentence3_n_hybridized",
        "sentence3_correct_type_sp2",
        "sentence3_hybrid_bonds",
        "sentence3_unhybridized_bonds",
    ],
    4: [
        "fixed_mass",
        "mass_data_lomp",
        "combine_compounds",
        "integer_ratio",
        "whole_numbers_indivisible",
        "indivisible_atom",
    ],
}


def read_student_answers(q: int) -> dict[str, tuple[str, str]]:
    """Return {sis_id: (answer_text, prompt)} for question q."""
    path = RAW_DIR / f"Student Answers Q{q}.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        prompt = header[1]
        # Strip the leading ID number prefix (e.g. "416344: ")
        if ":" in prompt[:10]:
            prompt = prompt[prompt.index(":") + 1 :].strip()
        answers = {}
        for row in reader:
            sid = row[0].strip()
            text = row[1].strip() if len(row) > 1 else ""
            answers[sid] = (text, prompt)
    return answers


def read_graded_rubric(q: int) -> tuple[list[str], list[dict]]:
    """Return (criterion_col_names, rows_as_dicts) for question q."""
    path = RAW_DIR / f"Graded Rubric Q{q}.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        criteria_cols = [c for c in fieldnames if c not in META_COLS and c not in EXCLUDE_COLS[q]]
        rows = list(reader)
    return criteria_cols, rows


def infer_weights(criteria_cols: list[str], rows: list[dict]) -> tuple[np.ndarray, float]:
    """Infer criterion weights via least-squares regression on Score."""
    n = len(rows)
    m = len(criteria_cols)
    X = np.zeros((n, m))
    y = np.zeros(n)
    valid = []
    for i, row in enumerate(rows):
        score_str = row.get("Score", "").strip()
        if not score_str:
            continue
        try:
            y_val = float(score_str)
        except ValueError:
            continue
        x_row = np.zeros(m)
        for j, col in enumerate(criteria_cols):
            x_row[j] = 1.0 if row.get(col, "").strip().upper() == "TRUE" else 0.0
        X[i] = x_row
        y[i] = y_val
        valid.append(i)

    X = X[valid]
    y = y[valid]

    # Account for Adjustment column
    adj = np.zeros(len(valid))
    for idx, i in enumerate(valid):
        adj_str = rows[i].get("Adjustment", "").strip()
        if adj_str:
            try:
                adj[idx] = float(adj_str)
            except ValueError:
                pass
    y_adjusted = y - adj  # remove adjustment to get pure rubric score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights, residuals, rank, sv = np.linalg.lstsq(X, y_adjusted, rcond=None)

    # Compute R²
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = X @ weights
        ss_res = np.sum((y_adjusted - y_pred) ** 2)
        ss_tot = np.sum((y_adjusted - np.mean(y_adjusted)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return weights, r2


def convert_question(q: int) -> None:
    """Convert a single question to AutoRubric format."""
    answers = read_student_answers(q)
    criteria_cols, graded_rows = read_graded_rubric(q)
    short_names = SHORT_NAMES[q]

    assert len(criteria_cols) == len(short_names), (
        f"Q{q}: expected {len(short_names)} criteria, got {len(criteria_cols)}: {criteria_cols}"
    )

    weights, r2 = infer_weights(criteria_cols, graded_rows)

    # Get prompt from first answer entry
    prompt = next(iter(answers.values()))[1]

    # Build criteria
    rubric_criteria = []
    for i, col in enumerate(criteria_cols):
        rubric_criteria.append(
            Criterion(
                name=short_names[i],
                weight=round(float(weights[i]), 4),
                requirement=col,
            )
        )

    rubric = Rubric(rubric_criteria)

    # Build items by joining on SID
    items = []
    blank_count = 0
    for row in graded_rows:
        sid = row["SID"].strip()
        answer_entry = answers.get(sid)
        if not answer_entry:
            continue
        text = answer_entry[0]

        # Skip blanks
        if not text or row.get("Blank", "").strip().upper() == "TRUE":
            blank_count += 1
            continue
        # Q3 has "Incorrect/Blank response" instead of "Blank"
        if row.get("Incorrect/Blank response", "").strip().upper() == "TRUE" and not text:
            blank_count += 1
            continue

        ground_truth = []
        for col in criteria_cols:
            val = row.get(col, "").strip().upper()
            ground_truth.append(CriterionVerdict.MET if val == "TRUE" else CriterionVerdict.UNMET)

        items.append(
            DataItem(
                submission=text,
                description=f"Student {sid[:8]}",
                ground_truth=ground_truth,
            )
        )

    dataset = RubricDataset(
        name=f"ricechem-q{q}",
        prompt=prompt,
        rubric=rubric,
        items=items,
    )

    out_path = OUT_DIR / f"q{q}.json"
    dataset.to_file(out_path)

    print(
        f"Q{q}: {len(criteria_cols)} criteria, {len(items)} students ({blank_count} blank excluded), R²={r2:.4f}"
    )
    print(f"  Weights: {[f'{short_names[i]}={weights[i]:.2f}' for i in range(len(weights))]}")
    print(f"  Saved to {out_path}")

    # Verify round-trip
    loaded = RubricDataset.from_file(out_path)
    assert len(loaded.items) == len(items)
    assert loaded.rubric is not None
    assert len(loaded.rubric.rubric) == len(criteria_cols)
    print("  Round-trip verified ✓")
    print()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_criteria = 0
    for q in range(1, 5):
        convert_question(q)
        total_criteria += len(SHORT_NAMES[q])
    print(f"Total criteria: {total_criteria}")


if __name__ == "__main__":
    main()
