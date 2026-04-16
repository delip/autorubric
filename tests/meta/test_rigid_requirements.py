"""Tests for rigidity and low-signal criteria in the metarubric."""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "autorubric" / "meta" / "data"


def _load_criteria(filename: str) -> list[dict]:
    """Load all criteria from a metarubric JSON file."""
    with open(_DATA_DIR / filename, encoding="utf-8") as f:
        data = json.load(f)
    criteria = []
    for section in data["rubric"]["sections"]:
        criteria.extend(section["criteria"])
    return criteria


def _find_criterion(criteria: list[dict], name: str) -> dict | None:
    """Find a criterion by name."""
    return next((c for c in criteria if c["name"] == name), None)


# ============================================================================
# Overly Strict Requirements
# ============================================================================


class TestOverlyStrictRequirements:
    """Verify overly_strict_requirements exists in in-context only."""

    def test_exists_in_context(self):
        criteria = _load_criteria("meta_rubric_in_context.json")
        c = _find_criterion(criteria, "overly_strict_requirements")
        assert c is not None, "overly_strict_requirements not found in in-context"
        assert c["weight"] == -6

    def test_not_in_standalone(self):
        criteria = _load_criteria("meta_rubric_standalone.json")
        c = _find_criterion(criteria, "overly_strict_requirements")
        assert c is None, "overly_strict_requirements should not be in standalone"

    def test_is_antipattern(self):
        criteria = _load_criteria("meta_rubric_in_context.json")
        c = _find_criterion(criteria, "overly_strict_requirements")
        assert c["weight"] < 0


# ============================================================================
# Distinguishes Quality (Sharpened)
# ============================================================================


class TestDistinguishesQualitySharpened:
    """Verify distinguishes_quality has been sharpened for low-signal detection."""

    def test_requirement_mentions_materially_different(self):
        criteria = _load_criteria("meta_rubric_in_context.json")
        c = _find_criterion(criteria, "distinguishes_quality")
        assert c is not None
        assert "materially different" in c["requirement"].lower()

    def test_requirement_mentions_score_band(self):
        criteria = _load_criteria("meta_rubric_in_context.json")
        c = _find_criterion(criteria, "distinguishes_quality")
        assert "score band" in c["requirement"].lower() or "narrow" in c["requirement"].lower()
