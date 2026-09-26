"""Legacy rubric and dataset files re-serialize byte-identically.

Rubric-level guidelines added a dict rubric form (``{"guidelines": ..., "criteria": [...]}``)
that ``RubricDataset`` writes only for a rubric with guidelines. A file without guidelines
must therefore load and re-serialize exactly as the library did before guidelines existed.

``tests/golden/datasets/reserialized_sha256.json`` holds, for every tracked dataset file in
``examples/data`` (global and per-item rubrics) and the library's own meta-rubric files, the
SHA-256 of the re-serialization produced by the unmodified library (``main`` at the merge of
PR #17): ``RubricDataset.from_file(path).to_json()`` for a dataset file, and
``RubricDataset(rubric=Rubric.from_file(path)).to_json()`` for a rubric file, which has no
serializer of its own. The digest of the source file (line endings normalized, so a CRLF
checkout matches) is stored too, so a deliberately edited example file fails with a clear
"regenerate" message instead of a spurious byte mismatch.

Regenerate from a checkout whose serialization is the intended reference (``PYTHONPATH``
selects which library checkout is imported; the script prints the path it captured from)::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout>/src \\
        uv run --frozen python tests/dataset/test_legacy_reserialization.py --write
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import autorubric
from autorubric import Rubric, RubricDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "datasets" / "reserialized_sha256.json"

# Repo-relative paths, by kind. Dataset files carry a global rubric (list form) or per-item
# rubrics (sharma_etal_2025); rubric files use the {"rubric": {"sections": [...]}} form.
DATASET_FILES = (
    "examples/data/charm100.json",
    "examples/data/essay_grading_dataset.json",
    "examples/data/hashemi_etal_2024_dataset.json",
    "examples/data/peer_review_skill_eval.json",
    "examples/data/ricechem/q1.json",
    "examples/data/ricechem/q2.json",
    "examples/data/ricechem/q3.json",
    "examples/data/ricechem/q4.json",
    "examples/data/sharma_etal_2025_research_rubrics.json",
)
RUBRIC_FILES = (
    "src/autorubric/meta/data/meta_rubric_in_context.json",
    "src/autorubric/meta/data/meta_rubric_standalone.json",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_digest(path: Path) -> str:
    return _sha256(path.read_bytes().replace(b"\r\n", b"\n"))


def _reserialize(relative: str) -> str:
    path = REPO_ROOT / relative
    if relative in DATASET_FILES:
        return RubricDataset.from_file(path).to_json()
    return RubricDataset(rubric=Rubric.from_file(str(path))).to_json()


def _capture() -> dict[str, dict[str, str]]:
    return {
        relative: {
            "source_sha256": _source_digest(REPO_ROOT / relative),
            "reserialized_sha256": _sha256(_reserialize(relative).encode("utf-8")),
        }
        for relative in DATASET_FILES + RUBRIC_FILES
    }


def _golden() -> dict[str, dict[str, str]]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_covers_exactly_the_listed_files() -> None:
    assert sorted(_golden()) == sorted(DATASET_FILES + RUBRIC_FILES)


@pytest.mark.parametrize("relative", DATASET_FILES + RUBRIC_FILES)
def test_legacy_file_reserializes_byte_identically(relative: str) -> None:
    expected = _golden()[relative]
    if _source_digest(REPO_ROOT / relative) != expected["source_sha256"]:
        pytest.fail(
            f"{relative} changed since its golden digest was captured; if the edit is "
            "deliberate, regenerate the goldens (see this module's docstring)"
        )
    actual = _reserialize(relative)
    assert _sha256(actual.encode("utf-8")) == expected["reserialized_sha256"]
    # A rubric without guidelines is written in the list form, never the dict form.
    data = json.loads(actual)
    rubrics = [data["rubric"]] + [item.get("rubric") for item in data["items"]]
    assert all(r is None or isinstance(r, list) for r in rubrics)


@pytest.mark.parametrize("relative", DATASET_FILES + RUBRIC_FILES)
def test_legacy_file_loads_without_guidelines(relative: str) -> None:
    path = REPO_ROOT / relative
    if relative in DATASET_FILES:
        dataset = RubricDataset.from_file(path)
        rubrics = [dataset.rubric] + [item.rubric for item in dataset.items]
    else:
        rubrics = [Rubric.from_file(str(path))]
    assert all(r is None or r.guidelines is None for r in rubrics)


def _write_golden() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps(_capture(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python tests/dataset/test_legacy_reserialization.py --write")
    print(f"Capturing re-serialization digests from {autorubric.__file__}")
    _write_golden()
