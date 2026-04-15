#!/usr/bin/env python3
"""Convert RiceChem dataset to RubricGenerationDataset for AutorubricLM training.

Reads the raw RiceChem CSV files (student answers + graded rubrics for 4
chemistry exam questions) and produces a single RubricGenerationDataset JSON
in ``data/autorubriclm/converted/ricechem.json``.

Each question becomes one RubricGenerationExample:
  - grading_problem_description: the exam question prompt
  - criteria: binary criteria with weights normalised to sum to 100
  - responses: a stratified sample of student submissions showing the full
    quality range the rubric must discriminate between
  - reference_response: None (no gold-standard answer in the raw data)

Source: Sonkar et al. (2024), "Automated Long Answer Grading with RiceChem
Dataset", AIED 2024. arXiv:2404.14316
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from autorubric.generation.dataset import (
    RubricGenerationDataset,
    RubricGenerationExample,
)
from autorubric.types import Criterion

RAW_DIR = ROOT / "ricechem-rawdata"
OUT_PATH = ROOT / "data" / "autorubriclm" / "converted" / "ricechem.json"

# ── Per-question configuration ──────────────────────────────────────────────

# Columns to exclude (negative/flag columns, not positive rubric criteria)
EXCLUDE_COLS: dict[int, set[str]] = {
    1: {"incorrect", "Blank", "Core charge calculation error"},
    2: {"Incorrect statement included", "Incorrect", "Blank"},
    3: {"Correct response", "Incorrect/Blank response"},
    4: {"incorrect/misleading statement", "incorrect/missing answer"},
}

META_COLS = {"SID", "Score", "Adjustment", "Comments"}

SHORT_NAMES: dict[int, list[str]] = {
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

# Number of student responses to include per question as examples of the
# quality range. We stratify by score tercile (low/mid/high) to show
# diversity without bloating the dataset.
RESPONSES_PER_QUESTION = 9  # 3 per tercile


# ── Raw data readers ────────────────────────────────────────────────────────


def read_student_answers(q: int) -> dict[str, tuple[str, str]]:
    """Return {sid: (answer_text, prompt)} for question *q*."""
    path = RAW_DIR / f"Student Answers Q{q}.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        prompt = header[1]
        if ":" in prompt[:10]:
            prompt = prompt[prompt.index(":") + 1 :].strip()
        answers = {}
        for row in reader:
            sid = row[0].strip()
            text = row[1].strip() if len(row) > 1 else ""
            answers[sid] = (text, prompt)
    return answers


def read_graded_rubric(q: int) -> tuple[list[str], list[dict[str, str]]]:
    """Return (criterion_col_names, rows_as_dicts) for question *q*."""
    path = RAW_DIR / f"Graded Rubric Q{q}.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        criteria_cols = [
            c for c in fieldnames if c not in META_COLS and c not in EXCLUDE_COLS[q]
        ]
        rows = list(reader)
    return criteria_cols, rows


# ── Weight inference ────────────────────────────────────────────────────────


def infer_weights(
    criteria_cols: list[str], rows: list[dict[str, str]]
) -> tuple[np.ndarray, float]:
    """Least-squares regression to recover per-criterion point values."""
    n = len(rows)
    m = len(criteria_cols)
    X = np.zeros((n, m))
    y = np.zeros(n)
    valid: list[int] = []

    for i, row in enumerate(rows):
        score_str = row.get("Score", "").strip()
        if not score_str:
            continue
        try:
            y_val = float(score_str)
        except ValueError:
            continue

        for j, col in enumerate(criteria_cols):
            X[i, j] = 1.0 if row.get(col, "").strip().upper() == "TRUE" else 0.0
        y[i] = y_val
        valid.append(i)

    X = X[valid]
    y = y[valid]

    adj = np.zeros(len(valid))
    for idx, i in enumerate(valid):
        adj_str = rows[i].get("Adjustment", "").strip()
        if adj_str:
            try:
                adj[idx] = float(adj_str)
            except ValueError:
                pass
    y_adjusted = y - adj

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights, _, _, _ = np.linalg.lstsq(X, y_adjusted, rcond=None)

    y_pred = X @ weights
    ss_res = float(np.sum((y_adjusted - y_pred) ** 2))
    ss_tot = float(np.sum((y_adjusted - np.mean(y_adjusted)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return weights, r2


# ── Response sampling ───────────────────────────────────────────────────────


def _is_blank(row: dict[str, str], text: str) -> bool:
    if not text:
        return True
    if row.get("Blank", "").strip().upper() == "TRUE":
        return True
    if row.get("Incorrect/Blank response", "").strip().upper() == "TRUE" and not text:
        return True
    return False


def stratified_sample(
    answers: dict[str, tuple[str, str]],
    criteria_cols: list[str],
    rows: list[dict[str, str]],
    weights: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Pick *n* student responses stratified across score terciles."""
    scored: list[tuple[str, float]] = []
    for row in rows:
        sid = row["SID"].strip()
        entry = answers.get(sid)
        if not entry or _is_blank(row, entry[0]):
            continue
        score = sum(
            weights[j] * (1.0 if row.get(col, "").strip().upper() == "TRUE" else 0.0)
            for j, col in enumerate(criteria_cols)
        )
        scored.append((entry[0], score))

    scored.sort(key=lambda x: x[1])
    per_tercile = max(n // 3, 1)
    tercile_size = len(scored) // 3

    sampled: list[str] = []
    for t in range(3):
        start = t * tercile_size
        end = len(scored) if t == 2 else (t + 1) * tercile_size
        pool = scored[start:end]
        k = min(per_tercile, len(pool))
        indices = rng.choice(len(pool), size=k, replace=False)
        sampled.extend(pool[i][0] for i in indices)

    return sampled


# ── Per-question conversion ─────────────────────────────────────────────────


def convert_question(q: int, rng: np.random.Generator) -> RubricGenerationExample:
    """Convert a single RiceChem question to a RubricGenerationExample."""
    answers = read_student_answers(q)
    criteria_cols, graded_rows = read_graded_rubric(q)
    short_names = SHORT_NAMES[q]

    assert len(criteria_cols) == len(
        short_names
    ), f"Q{q}: expected {len(short_names)} criteria, got {len(criteria_cols)}"

    raw_weights, r2 = infer_weights(criteria_cols, graded_rows)

    # Normalise weights to sum to 100
    total = float(np.sum(np.abs(raw_weights)))
    normed = (raw_weights / total) * 100.0

    criteria = [
        Criterion(
            name=short_names[i],
            weight=round(float(normed[i]), 2),
            requirement=col,
        )
        for i, col in enumerate(criteria_cols)
    ]

    raw_prompt = next(iter(answers.values()))[1]

    grading_problem = (
        "Grade student responses to the following college-level chemistry "
        f"exam question:\n\n{raw_prompt}"
    )

    responses = stratified_sample(
        answers, criteria_cols, graded_rows, raw_weights, RESPONSES_PER_QUESTION, rng
    )

    example = RubricGenerationExample(
        grading_problem_description=grading_problem,
        responses=responses,
        criteria=criteria,
        source_dataset="ricechem",
        is_human_authored=True,
        domain="education_grading",
        language="en",
    )
    example = example.model_copy(update={"criterion_mix": example.effective_criterion_mix})

    print(f"Q{q}: {len(criteria)} criteria, R²={r2:.4f}")
    if r2 < 0.9:
        print(f"  WARNING: R²={r2:.4f} is below 0.9 threshold — inferred weights may be unreliable")
    print(f"  Weight sum: {sum(c.weight for c in criteria):.1f}")
    print(f"  Responses sampled: {len(responses)}")

    return example


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)

    examples = [convert_question(q, rng) for q in range(1, 5)]

    dataset = RubricGenerationDataset(
        name="ricechem",
        version="0.1.0",
        examples=examples,
    )

    dataset.to_json(str(OUT_PATH))
    print(f"\nWrote {len(dataset)} examples to {OUT_PATH}")

    # Round-trip verification
    loaded = RubricGenerationDataset.from_json(str(OUT_PATH))
    assert len(loaded) == len(dataset)
    for orig, rt in zip(dataset.examples, loaded.examples):
        assert orig.id == rt.id
        assert orig.num_criteria == rt.num_criteria
        assert orig.grading_problem_description == rt.grading_problem_description
    print("Round-trip verification passed.")


if __name__ == "__main__":
    main()
