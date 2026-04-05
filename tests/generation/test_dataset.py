"""Tests for autorubric.generation.dataset module."""

import json
import tempfile
from pathlib import Path

import pytest

from autorubric.generation import (
    CriterionMix,
    HumanAuthored,
    RubricGenerationDataset,
    RubricGenerationExample,
)
from autorubric.types import Criterion, CriterionOption


# =============================================================================
# Fixtures
# =============================================================================


def _binary_criteria() -> list[Criterion]:
    return [
        Criterion(name="thesis", weight=30.0, requirement="Has a clear thesis statement"),
        Criterion(name="evidence", weight=40.0, requirement="Supports claims with evidence"),
        Criterion(name="organization", weight=30.0, requirement="Logically organized argument"),
    ]


def _ordinal_criterion() -> Criterion:
    return Criterion(
        name="depth",
        weight=20.0,
        requirement="Depth of analysis",
        scale_type="ordinal",
        options=[
            CriterionOption(label="Superficial", value=0.0),
            CriterionOption(label="Adequate", value=0.5),
            CriterionOption(label="Deep", value=1.0),
        ],
    )


def _make_example(**overrides) -> RubricGenerationExample:
    defaults = {
        "grading_problem_description": "Evaluate a student essay on the causes of WWI.",
        "criteria": _binary_criteria(),
    }
    defaults.update(overrides)
    return RubricGenerationExample(**defaults)


# =============================================================================
# RubricGenerationExample Tests
# =============================================================================


class TestRubricGenerationExample:
    """Tests for RubricGenerationExample model."""

    def test_minimal_creation(self):
        """Example can be created with only required fields."""
        ex = _make_example()
        assert ex.grading_problem_description.startswith("Evaluate")
        assert len(ex.criteria) == 3
        assert ex.id  # auto-generated

    def test_auto_id_is_deterministic(self):
        """Same content produces the same ID."""
        ex1 = _make_example()
        ex2 = _make_example()
        assert ex1.id == ex2.id

    def test_auto_id_changes_with_content(self):
        """Different content produces different IDs."""
        ex1 = _make_example()
        ex2 = _make_example(grading_problem_description="A different task.")
        assert ex1.id != ex2.id

    def test_explicit_id_preserved(self):
        """Explicitly provided ID is not overwritten."""
        ex = _make_example(id="custom-id-123")
        assert ex.id == "custom-id-123"

    def test_defaults(self):
        """Default values are correct."""
        ex = _make_example()
        assert ex.reference_response is None
        assert ex.responses is None
        assert ex.target_criteria_count is None
        assert ex.criterion_mix is None
        assert ex.source_dataset is None
        assert ex.is_human_authored == "unknown"
        assert ex.domain is None
        assert ex.language == "en"

    def test_all_fields(self):
        """Example can be created with all fields populated."""
        ex = RubricGenerationExample(
            grading_problem_description="Grade this essay.",
            reference_response="An ideal response would discuss...",
            responses=["Student A wrote...", "Student B wrote..."],
            target_criteria_count=5,
            criterion_mix="binary",
            criteria=_binary_criteria(),
            id="abc123",
            source_dataset="manual",
            is_human_authored=True,
            domain="writing",
            language="es",
        )
        assert ex.reference_response is not None
        assert len(ex.responses) == 2
        assert ex.target_criteria_count == 5
        assert ex.criterion_mix == "binary"
        assert ex.source_dataset == "manual"
        assert ex.is_human_authored is True  # noqa: E712
        assert ex.domain == "writing"
        assert ex.language == "es"

    def test_target_criteria_count_validation(self):
        """target_criteria_count must be between 1 and 30."""
        with pytest.raises(ValueError):
            _make_example(target_criteria_count=0)
        with pytest.raises(ValueError):
            _make_example(target_criteria_count=31)
        # Valid boundaries
        assert _make_example(target_criteria_count=1).target_criteria_count == 1
        assert _make_example(target_criteria_count=30).target_criteria_count == 30

    # ── Computed properties ──────────────────────────────────────────

    def test_num_criteria(self):
        ex = _make_example()
        assert ex.num_criteria == 3

    def test_has_negative_criteria_false(self):
        ex = _make_example()
        assert ex.has_negative_criteria is False

    def test_has_negative_criteria_true(self):
        criteria = _binary_criteria() + [
            Criterion(name="penalty", weight=-5.0, requirement="Deduct for plagiarism")
        ]
        ex = _make_example(criteria=criteria)
        assert ex.has_negative_criteria is True

    def test_positive_weight_sum(self):
        ex = _make_example()
        assert ex.positive_weight_sum == pytest.approx(100.0)

    def test_positive_weight_sum_excludes_negatives(self):
        criteria = _binary_criteria() + [
            Criterion(name="penalty", weight=-10.0, requirement="Penalty")
        ]
        ex = _make_example(criteria=criteria)
        assert ex.positive_weight_sum == pytest.approx(100.0)

    def test_criterion_type_counts_all_binary(self):
        ex = _make_example()
        counts = ex.criterion_type_counts
        assert counts == {"binary": 3, "ordinal": 0, "nominal": 0}

    def test_criterion_type_counts_mixed(self):
        criteria = _binary_criteria() + [_ordinal_criterion()]
        ex = _make_example(criteria=criteria)
        counts = ex.criterion_type_counts
        assert counts["binary"] == 3
        assert counts["ordinal"] == 1

    def test_effective_criterion_mix_binary(self):
        ex = _make_example()
        assert ex.effective_criterion_mix == "binary"

    def test_effective_criterion_mix_heterogeneous(self):
        criteria = _binary_criteria() + [_ordinal_criterion()]
        ex = _make_example(criteria=criteria)
        assert ex.effective_criterion_mix == "heterogeneous"

    # ── Serialization ────────────────────────────────────────────────

    def test_roundtrip_json(self):
        """Example survives JSON serialization round-trip."""
        ex = _make_example(domain="science", source_dataset="test")
        json_str = ex.model_dump_json()
        restored = RubricGenerationExample.model_validate_json(json_str)
        assert restored.id == ex.id
        assert restored.grading_problem_description == ex.grading_problem_description
        assert len(restored.criteria) == len(ex.criteria)
        assert restored.domain == ex.domain

    def test_computed_fields_in_json(self):
        """Computed fields appear in serialized output."""
        ex = _make_example()
        data = json.loads(ex.model_dump_json())
        assert "num_criteria" in data
        assert "has_negative_criteria" in data
        assert "positive_weight_sum" in data
        assert "criterion_type_counts" in data
        assert "effective_criterion_mix" in data


# =============================================================================
# RubricGenerationDataset Tests
# =============================================================================


class TestRubricGenerationDataset:
    """Tests for RubricGenerationDataset model."""

    def test_empty_dataset(self):
        ds = RubricGenerationDataset(name="empty")
        assert len(ds) == 0
        assert list(ds) == []

    def test_dataset_with_examples(self):
        examples = [_make_example(), _make_example(grading_problem_description="Another task.")]
        ds = RubricGenerationDataset(name="test", examples=examples)
        assert len(ds) == 2
        assert ds[0].grading_problem_description.startswith("Evaluate")
        assert ds[1].grading_problem_description == "Another task."

    def test_iteration(self):
        examples = [_make_example(), _make_example(grading_problem_description="Task 2")]
        ds = RubricGenerationDataset(name="test", examples=examples)
        collected = list(ds)
        assert len(collected) == 2

    def test_indexing(self):
        examples = [_make_example(), _make_example(grading_problem_description="Task 2")]
        ds = RubricGenerationDataset(name="test", examples=examples)
        assert ds[0].grading_problem_description.startswith("Evaluate")
        assert ds[1].grading_problem_description == "Task 2"

    def test_defaults(self):
        ds = RubricGenerationDataset(name="test")
        assert ds.version == "0.1.0"
        assert ds.created_at is not None
        assert ds.split is None
        assert ds.examples == []

    # ── JSON round-trip ──────────────────────────────────────────────

    def test_json_roundtrip(self):
        """Dataset survives JSON write/read round-trip with full metadata."""
        examples = [
            _make_example(domain="writing", source_dataset="manual"),
            _make_example(
                grading_problem_description="Grade a lab report.",
                domain="science",
                is_human_authored=True,
            ),
        ]
        ds = RubricGenerationDataset(
            name="v0.1",
            version="1.2.0",
            split="train",
            examples=examples,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.json")
            ds.to_json(path)

            loaded = RubricGenerationDataset.from_json(path)
            assert loaded.name == "v0.1"
            assert loaded.version == "1.2.0"
            assert loaded.split == "train"
            assert loaded.created_at == ds.created_at
            assert len(loaded) == 2
            assert loaded[0].domain == "writing"
            assert loaded[1].domain == "science"
            assert loaded[1].is_human_authored is True

    def test_json_ids_preserved(self):
        """Auto-generated IDs survive the JSON round-trip."""
        ex = _make_example()
        original_id = ex.id
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.json")
            RubricGenerationDataset(name="t", examples=[ex]).to_json(path)
            loaded = RubricGenerationDataset.from_json(path)
            assert loaded[0].id == original_id



    # ── Merge ───────────────────────────────────────────────────────

    def test_merge_concatenates_examples(self):
        ex_a = _make_example(domain="writing")
        ex_b = _make_example(grading_problem_description="Grade a lab report.", domain="science")
        ds_a = RubricGenerationDataset(name="a", examples=[ex_a])
        ds_b = RubricGenerationDataset(name="b", examples=[ex_b])

        merged = ds_a.merge(ds_b)
        assert len(merged) == 2
        assert merged[0].domain == "writing"
        assert merged[1].domain == "science"

    def test_merge_defaults_to_self_metadata(self):
        ds_a = RubricGenerationDataset(name="a", version="1.0.0", split="train")
        ds_b = RubricGenerationDataset(name="b", version="2.0.0", split="test")

        merged = ds_a.merge(ds_b)
        assert merged.name == "a"
        assert merged.version == "1.0.0"
        assert merged.split == "train"

    def test_merge_override_metadata(self):
        ds_a = RubricGenerationDataset(name="a", version="1.0.0", split="train")
        ds_b = RubricGenerationDataset(name="b")

        merged = ds_a.merge(ds_b, name="combined", version="3.0.0", split="all")
        assert merged.name == "combined"
        assert merged.version == "3.0.0"
        assert merged.split == "all"

    def test_merge_returns_new_dataset(self):
        """Merge does not mutate either input dataset."""
        ex_a = _make_example(domain="writing")
        ex_b = _make_example(grading_problem_description="Task B.", domain="science")
        ds_a = RubricGenerationDataset(name="a", examples=[ex_a])
        ds_b = RubricGenerationDataset(name="b", examples=[ex_b])

        merged = ds_a.merge(ds_b)
        assert len(ds_a) == 1
        assert len(ds_b) == 1
        assert len(merged) == 2

    def test_merge_with_empty_dataset(self):
        ds_a = RubricGenerationDataset(name="a", examples=[_make_example()])
        ds_b = RubricGenerationDataset(name="b")

        merged = ds_a.merge(ds_b)
        assert len(merged) == 1


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Verify public exports are accessible."""

    def test_import_from_generation(self):
        from autorubric.generation import (  # noqa: F401
            CriterionMix,
            RubricGenerationDataset,
            RubricGenerationExample,
        )

    def test_import_from_top_level(self):
        from autorubric import (  # noqa: F401
            CriterionMix,
            RubricGenerationDataset,
            RubricGenerationExample,
        )
