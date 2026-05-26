#!/usr/bin/env python3
"""Convert the ``facebook/research-plan-gen`` HuggingFace dataset to AutoRubric JSON.

Source: arXiv:2512.23707 "Training AI Co-Scientists Using Rubric Rewards"
        https://huggingface.co/datasets/facebook/research-plan-gen

The dataset has 3 configs (arxiv, ml, pubmed) x 2 splits (train, test) = 6 parquet
files. Each row is a research goal with a list of grading criteria (plain text, no
weights) and a reference solution. We emit one RubricDataset JSON per (config, split):

    Goal               -> per-item prompt
    Rubric[i]          -> Criterion(name=f"C{i+1}", requirement=..., weight=10.0 default)
    Reference solution -> per-item reference_submission
    (no submission)    -> submission="" (generation benchmark; graded later)
    metadata           -> description

Run inside the ``readertools`` conda env (has hf CLI, pyarrow, autorubric):

    python scripts/convert_research_plan_gen.py
"""

import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autorubric.dataset import DataItem, RubricDataset
from autorubric.rubric import Rubric
from autorubric.types import Criterion

REPO_ID = "facebook/research-plan-gen"
CONFIGS = ["arxiv", "ml", "pubmed"]
SPLITS = ["train", "test"]

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / ".cache" / "research-plan-gen"
OUT_DIR = ROOT / "examples" / "data" / "research-plan-gen"


def parquet_path(config: str, split: str) -> Path:
    return RAW_DIR / config / split / "data.parquet"


def download_raw() -> None:
    """Fetch the parquet files via the ``hf`` CLI (skips if already present)."""
    if all(parquet_path(c, s).exists() for c in CONFIGS for s in SPLITS):
        print(f"Raw parquet already present in {RAW_DIR}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} parquet via hf CLI -> {RAW_DIR}")
    subprocess.run(
        [
            "hf",
            "download",
            REPO_ID,
            "--repo-type",
            "dataset",
            "--include",
            "*/*/data.parquet",
            "--local-dir",
            str(RAW_DIR),
        ],
        check=True,
    )


def build_description(
    config: str, subdomain: str, category: str, identifier: str, article_id: str, q_id: str
) -> str:
    """Compose a compact, source-traceable description, skipping empty fields."""
    parts = [config]
    taxonomy = (category or "").strip() or (subdomain or "").strip()
    if taxonomy:
        parts.append(taxonomy)
    if (identifier or "").strip():
        parts.append(f"id={identifier.strip()}")
    if (article_id or "").strip():
        parts.append(f"article={article_id.strip()}")
    if (q_id or "").strip():
        parts.append(f"q={q_id.strip()}")
    return " | ".join(parts)


def convert(config: str, split: str) -> None:
    table = pq.read_table(parquet_path(config, split))
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n_rows = table.num_rows

    items: list[DataItem] = []
    skipped = 0
    total_criteria = 0
    for i in range(n_rows):
        goal = (cols["Goal"][i] or "").strip()
        raw_rubric = cols["Rubric"][i] or []
        requirements = [r.strip() for r in raw_rubric if r and r.strip()]
        if not goal or not requirements:
            skipped += 1
            continue

        criteria = [
            Criterion(name=f"C{j + 1}", requirement=req) for j, req in enumerate(requirements)
        ]
        reference = (cols["Reference solution"][i] or "").strip()
        items.append(
            DataItem(
                submission="",
                description=build_description(
                    config,
                    cols["Subdomain"][i],
                    cols["Category"][i],
                    cols["Identifier"][i],
                    cols["article_id"][i],
                    cols["q_id"][i],
                ),
                prompt=goal,
                rubric=Rubric(criteria),
                reference_submission=reference or None,
            )
        )
        total_criteria += len(criteria)

    dataset = RubricDataset(
        name=f"research-plan-gen-{config}-{split}",
        prompt=None,
        rubric=None,
        items=items,
    )
    out_path = OUT_DIR / f"{config}_{split}.json"
    dataset.to_file(out_path)

    avg = total_criteria / len(items) if items else 0.0
    print(
        f"{config}/{split}: {n_rows} rows -> {len(items)} items "
        f"({skipped} skipped), avg {avg:.1f} criteria -> {out_path.name}"
    )

    # Round-trip verification.
    loaded = RubricDataset.from_file(out_path)
    assert len(loaded.items) == len(items), "item count changed on reload"
    assert all(it.rubric is not None for it in loaded.items), "missing per-item rubric"
    assert all(it.prompt for it in loaded.items), "missing per-item prompt"


def main() -> None:
    download_raw()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        for split in SPLITS:
            convert(config, split)
    print(f"\nDone. 6 files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
