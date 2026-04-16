"""Tests for grounding-axis criteria in the metarubric."""

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
# Grounding Specified
# ============================================================================


class TestGroundingSpecified:
    """Verify grounding_specified criterion exists in both metarubrics."""

    def test_exists_in_standalone(self):
        criteria = _load_criteria("meta_rubric_standalone.json")
        c = _find_criterion(criteria, "grounding_specified")
        assert c is not None, "grounding_specified not found in standalone"
        assert c["weight"] == 8

    def test_exists_in_context(self):
        criteria = _load_criteria("meta_rubric_in_context.json")
        c = _find_criterion(criteria, "grounding_specified")
        assert c is not None, "grounding_specified not found in in-context"
        assert c["weight"] == 8

    def test_requirement_mentions_source(self):
        criteria = _load_criteria("meta_rubric_standalone.json")
        c = _find_criterion(criteria, "grounding_specified")
        assert "source" in c["requirement"].lower()


# ============================================================================
# Unverifiable Claim
# ============================================================================


class TestUnverifiableClaim:
    """Verify unverifiable_claim criterion exists in both metarubrics."""

    def test_exists_in_standalone(self):
        criteria = _load_criteria("meta_rubric_standalone.json")
        c = _find_criterion(criteria, "unverifiable_claim")
        assert c is not None, "unverifiable_claim not found in standalone"
        assert c["weight"] == -8

    def test_exists_in_context(self):
        criteria = _load_criteria("meta_rubric_in_context.json")
        c = _find_criterion(criteria, "unverifiable_claim")
        assert c is not None, "unverifiable_claim not found in in-context"
        assert c["weight"] == -8

    def test_is_antipattern(self):
        criteria = _load_criteria("meta_rubric_standalone.json")
        c = _find_criterion(criteria, "unverifiable_claim")
        assert c["weight"] < 0


# ============================================================================
# Deterministic Assessability Narrowing
# ============================================================================


class TestDeterministicAssessabilityNarrowed:
    """Verify deterministic_assessability no longer covers external knowledge."""

    def test_standalone_no_external_knowledge(self):
        criteria = _load_criteria("meta_rubric_standalone.json")
        c = _find_criterion(criteria, "deterministic_assessability")
        assert c is not None
        assert "external knowledge" not in c["requirement"].lower()

    def test_in_context_no_external_knowledge(self):
        criteria = _load_criteria("meta_rubric_in_context.json")
        c = _find_criterion(criteria, "deterministic_assessability")
        assert c is not None
        assert "external knowledge" not in c["requirement"].lower()
