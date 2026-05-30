"""Tests for autorubric.dataset module."""

import json
import tempfile
from pathlib import Path

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric
from autorubric.dataset import DataItem, RubricDataset

# =============================================================================
# DataItem Tests
# =============================================================================


class TestDataItem:
    """Tests for DataItem dataclass."""

    def test_ground_truth_accepts_mixed_verdict_and_string(self):
        """DataItem accepts mixed CriterionVerdict and strings (binary + multi-choice)."""
        item = DataItem(
            submission="Test",
            description="Test",
            ground_truth=[CriterionVerdict.MET, "Very satisfied"],
        )
        assert item.ground_truth[0] == CriterionVerdict.MET
        assert item.ground_truth[1] == "Very satisfied"

    def test_ground_truth_validation_rejects_invalid_types(self):
        """DataItem rejects non-CriterionVerdict/non-string ground truth values."""
        with pytest.raises(ValueError, match="must be CriterionVerdict or str"):
            DataItem(
                submission="Test",
                description="Test",
                ground_truth=[123, 456],  # Integers are not valid
            )


# =============================================================================
# RubricDataset Tests
# =============================================================================


@pytest.fixture
def sample_rubric() -> Rubric:
    """Create a sample rubric for testing."""
    return Rubric(
        [
            Criterion(name="Accuracy", weight=10.0, requirement="Must be accurate"),
            Criterion(name="Clarity", weight=5.0, requirement="Must be clear"),
            Criterion(name="Errors", weight=-3.0, requirement="Contains errors"),
        ]
    )


@pytest.fixture
def sample_dataset(sample_rubric: Rubric) -> RubricDataset:
    """Create a sample dataset for testing."""
    dataset = RubricDataset(
        prompt="Explain the topic",
        rubric=sample_rubric,
    )
    dataset.add_item(
        submission="Good response with accurate information.",
        description="High quality",
        ground_truth=[
            CriterionVerdict.MET,
            CriterionVerdict.MET,
            CriterionVerdict.UNMET,
        ],
    )
    dataset.add_item(
        submission="Poor response.",
        description="Low quality",
        ground_truth=[
            CriterionVerdict.UNMET,
            CriterionVerdict.UNMET,
            CriterionVerdict.MET,
        ],
    )
    return dataset


class TestRubricDatasetCreation:
    """Tests for RubricDataset creation and validation."""

    def test_validation_rejects_mismatched_ground_truth(self, sample_rubric: Rubric):
        """Dataset rejects items with wrong number of ground truth verdicts."""
        items = [
            DataItem(
                submission="Text",
                description="Desc",
                ground_truth=[CriterionVerdict.MET],  # Only 1, but rubric has 3
            ),
        ]
        with pytest.raises(ValueError, match="ground truth values"):
            RubricDataset(prompt="Test", rubric=sample_rubric, items=items)


class TestRubricDatasetProperties:
    """Tests for RubricDataset properties."""

    def test_criterion_names(self, sample_dataset: RubricDataset):
        """criterion_names returns criterion names."""
        assert sample_dataset.criterion_names == ["Accuracy", "Clarity", "Errors"]

    def test_criterion_names_fallback(self):
        """criterion_names falls back to C{index} for unnamed criteria."""
        rubric = Rubric(
            [
                Criterion(weight=1.0, requirement="R1"),  # No name
                Criterion(name="Named", weight=1.0, requirement="R2"),
            ]
        )
        dataset = RubricDataset(prompt="Test", rubric=rubric)
        assert dataset.criterion_names == ["C1", "Named"]

    def test_total_positive_weight(self, sample_dataset: RubricDataset):
        """total_positive_weight sums only positive weights."""
        # 10.0 + 5.0 = 15.0 (excludes -3.0)
        assert sample_dataset.total_positive_weight == 15.0


class TestRubricDatasetAddItem:
    """Tests for RubricDataset.add_item()."""

    def test_add_item_without_ground_truth(self, sample_rubric: Rubric):
        """Items without ground truth can be added."""
        dataset = RubricDataset(prompt="Test", rubric=sample_rubric)
        dataset.add_item(submission="Text", description="Desc")
        assert len(dataset) == 1
        assert dataset[0].ground_truth is None

    def test_add_item_with_ground_truth(self, sample_rubric: Rubric):
        """Items with ground truth can be added."""
        dataset = RubricDataset(prompt="Test", rubric=sample_rubric)
        verdicts = [
            CriterionVerdict.MET,
            CriterionVerdict.UNMET,
            CriterionVerdict.UNMET,
        ]
        dataset.add_item(submission="Text", description="Desc", ground_truth=verdicts)
        assert dataset[0].ground_truth == verdicts

    def test_add_item_rejects_mismatched_ground_truth(self, sample_rubric: Rubric):
        """add_item rejects wrong number of ground truth verdicts."""
        dataset = RubricDataset(prompt="Test", rubric=sample_rubric)
        with pytest.raises(ValueError, match="Ground truth has"):
            dataset.add_item(
                submission="Text",
                description="Desc",
                ground_truth=[CriterionVerdict.MET],  # Only 1
            )


class TestRubricDatasetComputeWeightedScore:
    """Tests for RubricDataset.compute_weighted_score()."""

    @pytest.mark.parametrize(
        ("verdicts", "normalize", "expected"),
        [
            # All MET (no errors): (10 + 5) / 15 = 1.0
            (
                [CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.UNMET],
                True,
                1.0,
            ),
            # MET error criterion reduces normalized score: (10 + 5 - 3) / 15 = 0.8
            (
                [CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.MET],
                True,
                pytest.approx(0.8),
            ),
            # Raw (unnormalized) score is the weighted sum: +10
            (
                [CriterionVerdict.MET, CriterionVerdict.UNMET, CriterionVerdict.UNMET],
                False,
                10.0,
            ),
            # Normalized score clamped to [0, 1]: (-3) / 15 = -0.2 -> 0.0
            (
                [CriterionVerdict.UNMET, CriterionVerdict.UNMET, CriterionVerdict.MET],
                True,
                0.0,
            ),
        ],
    )
    def test_compute_weighted_score(
        self, sample_dataset: RubricDataset, verdicts, normalize, expected
    ):
        """compute_weighted_score across positive-sum, negative-weight, raw, and clamp cases."""
        score = sample_dataset.compute_weighted_score(verdicts, normalize=normalize)
        assert score == expected


class TestRubricDatasetIteration:
    """Tests for RubricDataset iteration and indexing."""

    def test_iter(self, sample_dataset: RubricDataset):
        """Dataset is iterable."""
        items = list(sample_dataset)
        assert len(items) == 2
        assert all(isinstance(item, DataItem) for item in items)


class TestRubricDatasetSerialization:
    """Tests for RubricDataset serialization."""

    def test_to_json(self, sample_dataset: RubricDataset):
        """Dataset can be serialized to JSON."""
        json_str = sample_dataset.to_json()
        data = json.loads(json_str)

        assert data["prompt"] == "Explain the topic"
        assert len(data["rubric"]) == 3
        assert len(data["items"]) == 2
        assert data["items"][0]["ground_truth"] == ["MET", "MET", "UNMET"]

    def test_from_json(self, sample_dataset: RubricDataset):
        """Dataset can be deserialized from JSON."""
        json_str = sample_dataset.to_json()
        loaded = RubricDataset.from_json(json_str)

        assert loaded.prompt == sample_dataset.prompt
        assert len(loaded) == len(sample_dataset)
        assert loaded.num_criteria == sample_dataset.num_criteria
        assert loaded[0].ground_truth == sample_dataset[0].ground_truth

    def test_roundtrip_json(self, sample_dataset: RubricDataset):
        """Dataset survives JSON roundtrip."""
        json_str = sample_dataset.to_json()
        loaded = RubricDataset.from_json(json_str)
        json_str2 = loaded.to_json()

        assert json.loads(json_str) == json.loads(json_str2)

    def test_from_json_prompt_optional(self):
        """from_json allows missing prompt (per-item prompts may be used)."""
        rubric_json = json.dumps(
            {
                "rubric": [{"name": "C1", "weight": 1.0, "requirement": "test"}],
                "items": [
                    {
                        "submission": "test",
                        "description": "test",
                        "ground_truth": None,
                        "prompt": "per-item prompt",
                    }
                ],
            }
        )
        ds = RubricDataset.from_json(rubric_json)
        assert ds.prompt is None
        assert ds.get_item_prompt(0) == "per-item prompt"

    @pytest.mark.parametrize(
        ("payload", "expected_error"),
        [
            # Unparseable JSON
            ("not valid json", "Failed to parse JSON"),
            # Missing required 'rubric' key
            ('{"prompt": "Test"}', "Missing required field: 'rubric'"),
            # Invalid verdict string in ground_truth
            (
                json.dumps(
                    {
                        "prompt": "Test",
                        "rubric": [{"weight": 1.0, "requirement": "R1"}],
                        "items": [
                            {
                                "submission": "T",
                                "description": "D",
                                "ground_truth": ["INVALID"],
                            }
                        ],
                    }
                ),
                "invalid verdict",
            ),
            # Item with no rubric and no global rubric
            (
                json.dumps(
                    {
                        "prompt": "Test",
                        "rubric": None,
                        "items": [
                            {
                                "submission": "T",
                                "description": "D",
                                "ground_truth": None,
                                # No rubric field
                            }
                        ],
                    }
                ),
                "Item 0 has no rubric",
            ),
        ],
    )
    def test_from_json_failure_modes(self, payload, expected_error):
        """from_json raises ValueError across its distinct malformed-input branches."""
        with pytest.raises(ValueError, match=expected_error):
            RubricDataset.from_json(payload)


class TestRubricDatasetFileIO:
    """Tests for RubricDataset file I/O."""

    def test_to_file_and_from_file(self, sample_dataset: RubricDataset):
        """Dataset can be saved and loaded from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            sample_dataset.to_file(temp_path)
            loaded = RubricDataset.from_file(temp_path)

            assert loaded.prompt == sample_dataset.prompt
            assert len(loaded) == len(sample_dataset)
        finally:
            temp_path.unlink()

    def test_from_file_not_found(self):
        """from_file raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            RubricDataset.from_file("/nonexistent/path.json")

    def test_from_file_accepts_string_path(self, sample_dataset: RubricDataset):
        """from_file accepts string path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            sample_dataset.to_file(temp_path)
            loaded = RubricDataset.from_file(temp_path)  # String path
            assert loaded.prompt == sample_dataset.prompt
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Per-Item Rubric Tests
# =============================================================================


class TestDataItemWithRubric:
    """Tests for DataItem with per-item rubric."""

    def test_ground_truth_rejects_length_mismatch_with_item_rubric(self):
        """DataItem rejects ground_truth that doesn't match item rubric length."""
        rubric = Rubric(
            [
                Criterion(name="C1", weight=1.0, requirement="R1"),
                Criterion(name="C2", weight=1.0, requirement="R2"),
            ]
        )
        with pytest.raises(ValueError, match="item rubric has 2 criteria"):
            DataItem(
                submission="Test",
                description="Test",
                ground_truth=[CriterionVerdict.MET],  # Only 1, but rubric has 2
                rubric=rubric,
            )


class TestRubricDatasetWithPerItemRubrics:
    """Tests for RubricDataset with per-item rubrics."""

    def test_dataset_with_no_global_rubric(self):
        """Dataset can have no global rubric if all items have rubrics."""
        rubric1 = Rubric([Criterion(name="C1", weight=1.0, requirement="R1")])
        rubric2 = Rubric([Criterion(name="C2", weight=2.0, requirement="R2")])

        item1 = DataItem(submission="Text1", description="D1", rubric=rubric1)
        item2 = DataItem(submission="Text2", description="D2", rubric=rubric2)

        dataset = RubricDataset(prompt="Test", rubric=None, items=[item1, item2])
        assert dataset.rubric is None
        assert len(dataset) == 2

    def test_get_item_rubric_returns_item_rubric(self):
        """get_item_rubric returns per-item rubric when present."""
        global_rubric = Rubric([Criterion(name="Global", weight=1.0, requirement="G")])
        item_rubric = Rubric([Criterion(name="Item", weight=2.0, requirement="I")])

        item = DataItem(submission="Test", description="D", rubric=item_rubric)
        dataset = RubricDataset(prompt="Test", rubric=global_rubric, items=[item])

        effective = dataset.get_item_rubric(0)
        assert effective == item_rubric
        assert effective.rubric[0].name == "Item"

    def test_get_item_rubric_falls_back_to_global(self):
        """get_item_rubric falls back to global rubric when item has none."""
        global_rubric = Rubric([Criterion(name="Global", weight=1.0, requirement="G")])

        item = DataItem(submission="Test", description="D")  # No per-item rubric
        dataset = RubricDataset(prompt="Test", rubric=global_rubric, items=[item])

        effective = dataset.get_item_rubric(0)
        assert effective == global_rubric

    def test_get_item_rubric_raises_when_no_rubric(self):
        """get_item_rubric raises ValueError when no rubric available."""
        item = DataItem(submission="Test", description="D")  # No rubric
        # Create dataset by bypassing normal validation
        dataset = RubricDataset.__new__(RubricDataset)
        dataset.prompt = "Test"
        dataset.rubric = None
        dataset.items = [item]
        dataset.name = None

        with pytest.raises(ValueError, match="no rubric and dataset has no global"):
            dataset.get_item_rubric(0)

    def test_properties_raise_when_no_global_rubric(self):
        """Properties raise ValueError when no global rubric is set."""
        item_rubric = Rubric([Criterion(name="Item", weight=1.0, requirement="R")])
        item = DataItem(submission="Test", description="D", rubric=item_rubric)
        dataset = RubricDataset(prompt="Test", rubric=None, items=[item])

        with pytest.raises(ValueError, match="no global rubric set"):
            _ = dataset.criterion_names

        with pytest.raises(ValueError, match="no global rubric set"):
            _ = dataset.num_criteria

        with pytest.raises(ValueError, match="no global rubric set"):
            _ = dataset.total_positive_weight

    def test_add_item_with_rubric(self):
        """add_item can add item with per-item rubric."""
        global_rubric = Rubric([Criterion(name="G", weight=1.0, requirement="R")])
        item_rubric = Rubric([Criterion(name="I", weight=2.0, requirement="R")])

        dataset = RubricDataset(prompt="Test", rubric=global_rubric)
        dataset.add_item(submission="Text", description="D", rubric=item_rubric)

        assert dataset[0].rubric == item_rubric

    def test_add_item_without_rubric_uses_global(self):
        """add_item without rubric uses global for validation."""
        global_rubric = Rubric(
            [
                Criterion(name="C1", weight=1.0, requirement="R1"),
                Criterion(name="C2", weight=1.0, requirement="R2"),
            ]
        )
        dataset = RubricDataset(prompt="Test", rubric=global_rubric)
        dataset.add_item(
            submission="Text",
            description="D",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
        )
        assert len(dataset) == 1

    def test_add_item_rejects_when_no_rubric_available(self):
        """add_item raises when no rubric is available."""
        dataset = RubricDataset.__new__(RubricDataset)
        dataset.prompt = "Test"
        dataset.rubric = None
        dataset.items = []
        dataset.name = None

        with pytest.raises(ValueError, match="no per-item rubric provided"):
            dataset.add_item(submission="Text", description="D")


class TestRubricDatasetSerializationWithPerItemRubrics:
    """Tests for serialization with per-item rubrics."""

    def test_to_json_with_null_global_rubric(self):
        """to_json outputs null for global rubric when None."""
        item_rubric = Rubric([Criterion(name="Item", weight=1.0, requirement="R")])
        item = DataItem(submission="Test", description="D", rubric=item_rubric)
        dataset = RubricDataset(prompt="Test", rubric=None, items=[item])

        json_str = dataset.to_json()
        data = json.loads(json_str)

        assert data["rubric"] is None
        assert "rubric" in data["items"][0]
        assert data["items"][0]["rubric"][0]["name"] == "Item"

    def test_to_json_with_per_item_rubrics(self):
        """to_json includes per-item rubrics."""
        global_rubric = Rubric([Criterion(name="Global", weight=1.0, requirement="G")])
        item_rubric = Rubric([Criterion(name="Item", weight=2.0, requirement="I")])

        item1 = DataItem(submission="T1", description="D1", rubric=item_rubric)
        item2 = DataItem(submission="T2", description="D2")  # Uses global

        dataset = RubricDataset(prompt="Test", rubric=global_rubric, items=[item1, item2])

        json_str = dataset.to_json()
        data = json.loads(json_str)

        assert data["rubric"][0]["name"] == "Global"
        assert data["items"][0]["rubric"][0]["name"] == "Item"
        assert "rubric" not in data["items"][1]  # No per-item rubric

    def test_from_json_with_null_global_rubric(self):
        """from_json parses null global rubric."""
        json_str = json.dumps(
            {
                "prompt": "Test",
                "rubric": None,
                "items": [
                    {
                        "submission": "T",
                        "description": "D",
                        "rubric": [{"name": "Item", "weight": 1.0, "requirement": "R"}],
                        "ground_truth": None,
                    }
                ],
            }
        )

        dataset = RubricDataset.from_json(json_str)
        assert dataset.rubric is None
        assert dataset[0].rubric is not None
        assert dataset[0].rubric.rubric[0].name == "Item"

    def test_roundtrip_with_per_item_rubrics(self):
        """Dataset with per-item rubrics survives JSON roundtrip."""
        item_rubric = Rubric([Criterion(name="Item", weight=2.0, requirement="IR")])
        item = DataItem(
            submission="Test",
            description="D",
            ground_truth=[CriterionVerdict.MET],
            rubric=item_rubric,
        )
        dataset = RubricDataset(prompt="Test", rubric=None, items=[item])

        json_str = dataset.to_json()
        loaded = RubricDataset.from_json(json_str)

        assert loaded.rubric is None
        assert loaded[0].rubric is not None
        assert loaded[0].rubric.rubric[0].name == "Item"
        assert loaded[0].rubric.rubric[0].weight == 2.0
        assert loaded[0].ground_truth == [CriterionVerdict.MET]


# =============================================================================
# Reference Submission Tests
# =============================================================================


class TestRubricDatasetWithReferenceSubmission:
    """Tests for RubricDataset with reference_submission field."""

    @pytest.mark.parametrize(
        ("item_reference", "global_reference", "expected"),
        [
            # Item-level reference takes precedence over global
            ("Item-specific reference", "Global reference", "Item-specific reference"),
            # Falls back to global when item has none
            (None, "Global reference", "Global reference"),
            # None when neither item nor global is set
            (None, None, None),
        ],
    )
    def test_get_item_reference_submission(
        self, sample_rubric: Rubric, item_reference, global_reference, expected
    ):
        """get_item_reference_submission precedence, global fallback, and None."""
        item = DataItem(
            submission="Student answer",
            description="D",
            reference_submission=item_reference,
        )
        dataset = RubricDataset(
            prompt="Test",
            rubric=sample_rubric,
            items=[item],
            reference_submission=global_reference,
        )

        assert dataset.get_item_reference_submission(0) == expected

    def test_add_item_with_reference(self, sample_rubric: Rubric):
        """add_item can add item with reference_submission."""
        dataset = RubricDataset(prompt="Test", rubric=sample_rubric)
        dataset.add_item(
            submission="Student answer",
            description="D",
            reference_submission="Item reference",
        )

        assert dataset[0].reference_submission == "Item reference"


class TestRubricDatasetSerializationWithReferenceSubmission:
    """Tests for serialization with reference_submission."""

    def test_to_json_with_global_reference(self, sample_rubric: Rubric):
        """to_json includes global reference_submission."""
        dataset = RubricDataset(
            prompt="Test",
            rubric=sample_rubric,
            reference_submission="Global exemplar",
        )

        json_str = dataset.to_json()
        data = json.loads(json_str)

        assert data["reference_submission"] == "Global exemplar"

    def test_to_json_with_item_reference(self, sample_rubric: Rubric):
        """to_json includes per-item reference_submission."""
        item = DataItem(
            submission="Answer",
            description="D",
            reference_submission="Item exemplar",
        )
        dataset = RubricDataset(
            prompt="Test",
            rubric=sample_rubric,
            items=[item],
        )

        json_str = dataset.to_json()
        data = json.loads(json_str)

        assert data["items"][0]["reference_submission"] == "Item exemplar"

    def test_to_json_omits_none_reference(self, sample_rubric: Rubric):
        """to_json omits reference_submission when None."""
        item = DataItem(submission="Answer", description="D")
        dataset = RubricDataset(
            prompt="Test",
            rubric=sample_rubric,
            items=[item],
        )

        json_str = dataset.to_json()
        data = json.loads(json_str)

        assert "reference_submission" not in data
        assert "reference_submission" not in data["items"][0]

    def test_from_json_with_global_reference(self, sample_rubric: Rubric):
        """from_json parses global reference_submission."""
        json_str = json.dumps(
            {
                "prompt": "Test",
                "rubric": [{"name": "C1", "weight": 1.0, "requirement": "R1"}],
                "reference_submission": "Global exemplar",
                "items": [{"submission": "Answer", "description": "D", "ground_truth": None}],
            }
        )

        dataset = RubricDataset.from_json(json_str)

        assert dataset.reference_submission == "Global exemplar"

    def test_from_json_with_item_reference(self, sample_rubric: Rubric):
        """from_json parses per-item reference_submission."""
        json_str = json.dumps(
            {
                "prompt": "Test",
                "rubric": [{"name": "C1", "weight": 1.0, "requirement": "R1"}],
                "items": [
                    {
                        "submission": "Answer",
                        "description": "D",
                        "ground_truth": None,
                        "reference_submission": "Item exemplar",
                    }
                ],
            }
        )

        dataset = RubricDataset.from_json(json_str)

        assert dataset.items[0].reference_submission == "Item exemplar"

    def test_roundtrip_with_reference_submissions(self, sample_rubric: Rubric):
        """Dataset with reference_submissions survives JSON roundtrip."""
        item = DataItem(
            submission="Student answer",
            description="D",
            reference_submission="Item reference",
        )
        dataset = RubricDataset(
            prompt="Test",
            rubric=sample_rubric,
            items=[item],
            reference_submission="Global reference",
        )

        json_str = dataset.to_json()
        loaded = RubricDataset.from_json(json_str)

        assert loaded.reference_submission == "Global reference"
        assert loaded[0].reference_submission == "Item reference"
        assert loaded.get_item_reference_submission(0) == "Item reference"


class TestRubricDatasetSplitWithReferenceSubmission:
    """Tests for split_train_test preserving reference_submission."""

    def test_split_preserves_global_reference(self, sample_rubric: Rubric):
        """split_train_test preserves global reference_submission."""
        dataset = RubricDataset(
            prompt="Test",
            rubric=sample_rubric,
            reference_submission="Global reference",
        )
        # Add items with ground_truth for stratification
        dataset.add_item(
            submission="A",
            description="D1",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.UNMET],
        )
        dataset.add_item(
            submission="B",
            description="D2",
            ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET, CriterionVerdict.UNMET],
        )

        train, test = dataset.split_train_test(n_train=1, seed=42)

        assert train.reference_submission == "Global reference"
        assert test.reference_submission == "Global reference"


# =============================================================================
# Per-Item Prompt Tests
# =============================================================================


class TestPerItemPrompt:
    """Tests for per-item prompt support."""

    def test_get_item_prompt_falls_back_to_dataset_prompt(self, sample_rubric: Rubric):
        """get_item_prompt falls back to dataset prompt when item prompt is None."""
        item = DataItem(
            submission="Test",
            description="D",
            # No prompt set on item
        )
        dataset = RubricDataset(
            prompt="Global prompt",
            rubric=sample_rubric,
            items=[item],
        )

        assert dataset.get_item_prompt(0) == "Global prompt"

    def test_get_item_prompt_raises_when_neither_is_set(self, sample_rubric: Rubric):
        """get_item_prompt raises ValueError when neither item nor dataset has prompt."""
        item = DataItem(
            submission="Test",
            description="D",
            # No prompt on item
        )
        # Create dataset without global prompt
        dataset = RubricDataset.__new__(RubricDataset)
        dataset.prompt = None
        dataset.rubric = sample_rubric
        dataset.items = [item]
        dataset.name = None
        dataset.reference_submission = None

        with pytest.raises(ValueError, match="no prompt and dataset has no global prompt"):
            dataset.get_item_prompt(0)

    def test_serialization_roundtrip_preserves_per_item_prompts(self, sample_rubric: Rubric):
        """Serialization round-trip preserves per-item prompts."""
        item1 = DataItem(
            submission="T1",
            description="D1",
            prompt="Item prompt 1",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.UNMET],
        )
        item2 = DataItem(
            submission="T2",
            description="D2",
            # No per-item prompt, relies on global
            ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET, CriterionVerdict.UNMET],
        )
        dataset = RubricDataset(
            prompt="Global prompt",
            rubric=sample_rubric,
            items=[item1, item2],
        )

        json_str = dataset.to_json()
        loaded = RubricDataset.from_json(json_str)

        assert loaded.get_item_prompt(0) == "Item prompt 1"
        assert loaded.get_item_prompt(1) == "Global prompt"

    def test_add_item_with_per_item_prompt(self, sample_rubric: Rubric):
        """add_item with per-item prompt works."""
        dataset = RubricDataset(
            prompt="Global prompt",
            rubric=sample_rubric,
        )
        dataset.add_item(
            submission="Test",
            description="D",
            prompt="Custom item prompt",
        )

        assert dataset[0].prompt == "Custom item prompt"
        assert dataset.get_item_prompt(0) == "Custom item prompt"

    def test_add_item_without_prompt_raises_when_no_global_prompt(self, sample_rubric: Rubric):
        """add_item without prompt raises ValueError when no global prompt."""
        dataset = RubricDataset.__new__(RubricDataset)
        dataset.prompt = None
        dataset.rubric = sample_rubric
        dataset.items = []
        dataset.name = None
        dataset.reference_submission = None

        with pytest.raises(
            ValueError,
            match="no per-item prompt provided and no global prompt set",
        ):
            dataset.add_item(submission="Test", description="D")

    def test_post_init_validation_error_when_no_prompt_available(self, sample_rubric: Rubric):
        """__post_init__ raises ValueError when item has no prompt and dataset has no prompt."""
        item = DataItem(
            submission="Test",
            description="D",
            # No prompt
        )
        with pytest.raises(
            ValueError,
            match="has no prompt and dataset has no global prompt",
        ):
            RubricDataset(
                prompt=None,  # No global prompt
                rubric=sample_rubric,
                items=[item],
            )

    def test_mixed_dataset_some_items_with_prompts(self, sample_rubric: Rubric):
        """Mixed dataset: some items with prompts, some relying on global prompt."""
        item1 = DataItem(
            submission="T1",
            description="D1",
            prompt="Item-specific prompt for question A",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.UNMET],
        )
        item2 = DataItem(
            submission="T2",
            description="D2",
            # Uses global prompt
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET, CriterionVerdict.UNMET],
        )
        item3 = DataItem(
            submission="T3",
            description="D3",
            prompt="Different item-specific prompt",
            ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET, CriterionVerdict.MET],
        )
        dataset = RubricDataset(
            prompt="Default global prompt",
            rubric=sample_rubric,
            items=[item1, item2, item3],
        )

        assert dataset.get_item_prompt(0) == "Item-specific prompt for question A"
        assert dataset.get_item_prompt(1) == "Default global prompt"
        assert dataset.get_item_prompt(2) == "Different item-specific prompt"
        assert len(dataset) == 3

    def test_to_json_includes_per_item_prompts(self, sample_rubric: Rubric):
        """to_json includes per-item prompts."""
        item1 = DataItem(
            submission="T1",
            description="D1",
            prompt="Custom prompt",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET, CriterionVerdict.UNMET],
        )
        item2 = DataItem(
            submission="T2",
            description="D2",
            ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET, CriterionVerdict.UNMET],
        )
        dataset = RubricDataset(
            prompt="Global prompt",
            rubric=sample_rubric,
            items=[item1, item2],
        )

        json_str = dataset.to_json()
        data = json.loads(json_str)

        assert data["items"][0]["prompt"] == "Custom prompt"
        assert "prompt" not in data["items"][1]  # Not serialized if None

    def test_from_json_with_per_item_prompts(self, sample_rubric: Rubric):
        """from_json parses per-item prompts correctly."""
        json_str = json.dumps(
            {
                "prompt": "Global prompt",
                "rubric": [
                    {"name": "C1", "weight": 1.0, "requirement": "R1"},
                    {"name": "C2", "weight": 1.0, "requirement": "R2"},
                    {"name": "C3", "weight": -1.0, "requirement": "R3"},
                ],
                "items": [
                    {
                        "submission": "T1",
                        "description": "D1",
                        "ground_truth": ["MET", "MET", "UNMET"],
                        "prompt": "Item-specific prompt",
                    },
                    {
                        "submission": "T2",
                        "description": "D2",
                        "ground_truth": ["UNMET", "MET", "UNMET"],
                    },
                ],
            }
        )

        dataset = RubricDataset.from_json(json_str)

        assert dataset[0].prompt == "Item-specific prompt"
        assert dataset[1].prompt is None
        assert dataset.get_item_prompt(0) == "Item-specific prompt"
        assert dataset.get_item_prompt(1) == "Global prompt"

    def test_dataset_with_only_per_item_prompts_no_global(self, sample_rubric: Rubric):
        """Dataset can have no global prompt if all items have per-item prompts."""
        item1 = DataItem(
            submission="T1",
            description="D1",
            prompt="Prompt for question 1",
        )
        item2 = DataItem(
            submission="T2",
            description="D2",
            prompt="Prompt for question 2",
        )
        dataset = RubricDataset(
            prompt=None,  # No global prompt
            rubric=sample_rubric,
            items=[item1, item2],
        )

        assert len(dataset) == 2
        assert dataset.get_item_prompt(0) == "Prompt for question 1"
        assert dataset.get_item_prompt(1) == "Prompt for question 2"
