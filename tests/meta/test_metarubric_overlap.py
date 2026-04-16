"""Ensure no two metarubric criteria share byte-identical requirement text."""

import json
from pathlib import Path

import pytest

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "autorubric" / "meta" / "data"


@pytest.fixture(params=["meta_rubric_standalone.json", "meta_rubric_in_context.json"])
def meta_rubric_path(request):
    return _DATA_DIR / request.param


def test_no_duplicate_requirements(meta_rubric_path):
    """No two criteria in the same metarubric have byte-identical requirement text."""
    with open(meta_rubric_path, encoding="utf-8") as f:
        data = json.load(f)

    requirements = []
    for section in data["rubric"]["sections"]:
        for criterion in section["criteria"]:
            requirements.append(criterion["requirement"].strip())

    seen = {}
    duplicates = []
    for req in requirements:
        if req in seen:
            duplicates.append(req)
        seen[req] = True

    assert not duplicates, f"Duplicate requirement text found: {duplicates}"


def test_no_duplicate_names(meta_rubric_path):
    """No two criteria share the same name."""
    with open(meta_rubric_path, encoding="utf-8") as f:
        data = json.load(f)

    names = []
    for section in data["rubric"]["sections"]:
        for criterion in section["criteria"]:
            names.append(criterion["name"])

    assert len(names) == len(set(names)), (
        f"Duplicate criterion names: {[n for n in names if names.count(n) > 1]}"
    )
