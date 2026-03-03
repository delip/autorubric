"""Tests for multi-choice criterion functionality."""

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    MultiChoiceJudgment,
    MultiChoiceVerdict,
    AggregatedMultiChoiceVerdict,
    Rubric,
)
from autorubric.dataset import DataItem, RubricDataset
from autorubric.types import MultiChoiceJudgeVote


# =============================================================================
# CriterionOption Tests
# =============================================================================


class TestCriterionOption:
    """Tests for CriterionOption dataclass."""

    def test_create_basic_option(self):
        """CriterionOption can be created with label and value."""
        opt = CriterionOption(label="Satisfied", value=0.75)
        assert opt.label == "Satisfied"
        assert opt.value == 0.75
        assert opt.na is False

    def test_create_na_option(self):
        """CriterionOption can be marked as NA."""
        opt = CriterionOption(label="Not Applicable", value=0.0, na=True)
        assert opt.na is True

    def test_value_range_validation(self):
        """CriterionOption validates value is in [0, 1] for non-NA options."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            CriterionOption(label="Bad", value=1.5)
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            CriterionOption(label="Bad", value=-0.1)

    def test_na_option_allows_any_value(self):
        """NA options don't validate value range."""
        opt = CriterionOption(label="NA", value=999.0, na=True)
        assert opt.value == 999.0


# =============================================================================
# Criterion Multi-Choice Tests
# =============================================================================


class TestCriterionMultiChoice:
    """Tests for Criterion with multi-choice options."""

    @pytest.fixture
    def ordinal_criterion(self):
        """Create an ordinal multi-choice criterion."""
        return Criterion(
            name="satisfaction",
            requirement="How satisfied are you?",
            weight=10.0,
            scale_type="ordinal",
            options=[
                CriterionOption(label="1", value=0.0),
                CriterionOption(label="2", value=0.33),
                CriterionOption(label="3", value=0.67),
                CriterionOption(label="4", value=1.0),
            ],
        )

    @pytest.fixture
    def nominal_criterion(self):
        """Create a nominal multi-choice criterion."""
        return Criterion(
            name="efficiency",
            requirement="Is the dialogue efficient?",
            weight=5.0,
            scale_type="nominal",
            options=[
                CriterionOption(label="Too few", value=0.0),
                CriterionOption(label="Too many", value=0.0),
                CriterionOption(label="Just right", value=1.0),
            ],
        )

    @pytest.fixture
    def criterion_with_na(self):
        """Create a criterion with NA option."""
        return Criterion(
            name="citations",
            requirement="Are there citations?",
            weight=8.0,
            scale_type="ordinal",
            options=[
                CriterionOption(label="None", value=0.0),
                CriterionOption(label="Some", value=0.5),
                CriterionOption(label="All", value=1.0),
                CriterionOption(label="NA", value=0.0, na=True),
            ],
        )

    def test_is_multi_choice(self, ordinal_criterion):
        """Criterion with options is multi-choice."""
        assert ordinal_criterion.is_multi_choice is True
        assert ordinal_criterion.is_binary is False

    def test_binary_criterion_is_not_multi_choice(self):
        """Criterion without options is binary."""
        binary = Criterion(requirement="Is this accurate?", weight=10.0)
        assert binary.is_binary is True
        assert binary.is_multi_choice is False

    def test_find_option_by_label(self, ordinal_criterion):
        """find_option_by_label returns correct index."""
        assert ordinal_criterion.find_option_by_label("1") == 0
        assert ordinal_criterion.find_option_by_label("4") == 3

    def test_find_option_by_label_case_insensitive(self, nominal_criterion):
        """find_option_by_label is case-insensitive."""
        assert nominal_criterion.find_option_by_label("too few") == 0
        assert nominal_criterion.find_option_by_label("JUST RIGHT") == 2

    def test_find_option_by_label_strips_whitespace(self, nominal_criterion):
        """find_option_by_label strips whitespace."""
        assert nominal_criterion.find_option_by_label("  Too few  ") == 0

    def test_find_option_by_label_not_found(self, ordinal_criterion):
        """find_option_by_label raises ValueError for unknown label."""
        with pytest.raises(ValueError, match="Label 'unknown' not found"):
            ordinal_criterion.find_option_by_label("unknown")

    def test_get_option_value(self, ordinal_criterion):
        """get_option_value returns correct value for index."""
        assert ordinal_criterion.get_option_value(0) == 0.0
        assert ordinal_criterion.get_option_value(2) == 0.67
        assert ordinal_criterion.get_option_value(3) == 1.0

    def test_get_option_value_out_of_range(self, ordinal_criterion):
        """get_option_value raises ValueError for out-of-range index."""
        with pytest.raises(ValueError, match="out of range"):
            ordinal_criterion.get_option_value(10)

    def test_validation_requires_min_options(self):
        """Criterion requires at least 2 options."""
        with pytest.raises(ValueError, match="at least 2 options"):
            Criterion(
                requirement="Question?",
                options=[CriterionOption(label="Only one", value=1.0)],
            )

    def test_validation_requires_non_na_options(self):
        """Criterion requires at least 2 non-NA options."""
        with pytest.raises(ValueError, match="at least 2 non-NA options"):
            Criterion(
                requirement="Question?",
                options=[
                    CriterionOption(label="Option", value=1.0),
                    CriterionOption(label="NA", value=0.0, na=True),
                ],
            )


# =============================================================================
# MultiChoiceVerdict Tests
# =============================================================================


class TestMultiChoiceVerdict:
    """Tests for MultiChoiceVerdict dataclass."""

    def test_create_verdict(self):
        """MultiChoiceVerdict stores index, label, value."""
        verdict = MultiChoiceVerdict(
            selected_index=2,
            selected_label="Satisfied",
            value=0.75,
            na=False,
        )
        assert verdict.selected_index == 2
        assert verdict.selected_label == "Satisfied"
        assert verdict.value == 0.75
        assert verdict.na is False

    def test_create_na_verdict(self):
        """MultiChoiceVerdict can represent NA selection."""
        verdict = MultiChoiceVerdict(
            selected_index=3,
            selected_label="NA",
            value=0.0,
            na=True,
        )
        assert verdict.na is True


class TestAggregatedMultiChoiceVerdict:
    """Tests for AggregatedMultiChoiceVerdict dataclass."""

    def test_aggregated_verdict_stores_continuous_value(self):
        """AggregatedMultiChoiceVerdict stores both discrete and continuous values."""
        verdict = AggregatedMultiChoiceVerdict(
            selected_index=2,
            selected_label="3",
            value=0.67,  # Discrete (snapped) value
            na=False,
            aggregated_value=0.55,  # Continuous value before snapping
        )
        assert verdict.value == 0.67
        assert verdict.aggregated_value == 0.55


# =============================================================================
# MultiChoiceJudgment Tests
# =============================================================================


class TestMultiChoiceJudgment:
    """Tests for MultiChoiceJudgment (LLM response format)."""

    def test_judgment_parsing(self):
        """MultiChoiceJudgment parses selected_option (1-indexed)."""
        judgment = MultiChoiceJudgment(
            selected_option=3,  # 1-indexed for LLM
            explanation="This option best matches the submission.",
        )
        assert judgment.selected_option == 3
        assert judgment.explanation == "This option best matches the submission."


# =============================================================================
# Rubric Parsing Tests
# =============================================================================


class TestRubricMultiChoiceParsing:
    """Tests for parsing multi-choice rubrics from YAML/JSON."""

    def test_from_yaml_with_multi_choice(self):
        """Rubric.from_yaml parses multi-choice criteria correctly."""
        yaml_str = """
        - name: satisfaction
          requirement: "How satisfied are you?"
          weight: 10.0
          scale_type: ordinal
          options:
            - label: "1"
              value: 0.0
            - label: "2"
              value: 0.33
            - label: "3"
              value: 0.67
            - label: "4"
              value: 1.0
        """
        rubric = Rubric.from_yaml(yaml_str)
        assert len(rubric.rubric) == 1
        criterion = rubric.rubric[0]
        assert criterion.is_multi_choice
        assert len(criterion.options) == 4
        assert criterion.scale_type == "ordinal"

    def test_from_yaml_mixed_binary_and_multi_choice(self):
        """Rubric.from_yaml handles mixed binary and multi-choice criteria."""
        yaml_str = """
        - name: accuracy
          requirement: "Is this accurate?"
          weight: 10.0
        - name: satisfaction
          requirement: "How satisfied?"
          weight: 5.0
          scale_type: ordinal
          options:
            - label: "Low"
              value: 0.0
            - label: "Medium"
              value: 0.5
            - label: "High"
              value: 1.0
        """
        rubric = Rubric.from_yaml(yaml_str)
        assert len(rubric.rubric) == 2
        assert rubric.rubric[0].is_binary
        assert rubric.rubric[1].is_multi_choice


# =============================================================================
# Dataset with Multi-Choice Tests
# =============================================================================


class TestDatasetMultiChoice:
    """Tests for RubricDataset with multi-choice criteria."""

    @pytest.fixture
    def mixed_rubric(self):
        """Create a rubric with binary and multi-choice criteria."""
        return Rubric([
            Criterion(name="accuracy", requirement="Is this accurate?", weight=10.0),
            Criterion(
                name="satisfaction",
                requirement="How satisfied?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="1", value=0.0),
                    CriterionOption(label="2", value=0.33),
                    CriterionOption(label="3", value=0.67),
                    CriterionOption(label="4", value=1.0),
                ],
            ),
        ])

    def test_compute_weighted_score_multi_choice(self, mixed_rubric):
        """compute_weighted_score handles multi-choice ground truth."""
        dataset = RubricDataset(prompt="Test", rubric=mixed_rubric)

        # MET for binary (10.0), "4" for multi-choice (1.0 * 5.0)
        score = dataset.compute_weighted_score(
            [CriterionVerdict.MET, "4"],
            normalize=True,
        )
        # Total positive weight = 15, score = 15/15 = 1.0
        assert score == 1.0

    def test_compute_weighted_score_partial_multi_choice(self, mixed_rubric):
        """compute_weighted_score computes partial credit for multi-choice."""
        dataset = RubricDataset(prompt="Test", rubric=mixed_rubric)

        # MET for binary (10.0), "2" for multi-choice (0.33 * 5.0 = 1.65)
        score = dataset.compute_weighted_score(
            [CriterionVerdict.MET, "2"],
            normalize=True,
        )
        # Total positive weight = 15, score = (10 + 1.65)/15 = 0.777
        assert 0.77 < score < 0.78

    def test_dataset_serialization_with_multi_choice(self, mixed_rubric):
        """RubricDataset serializes and deserializes multi-choice ground truth."""
        dataset = RubricDataset(prompt="Test", rubric=mixed_rubric, name="test")
        dataset.add_item(
            submission="Test response",
            description="Good response",
            ground_truth=[CriterionVerdict.MET, "3"],
        )

        # Serialize
        json_str = dataset.to_json()

        # Deserialize
        loaded = RubricDataset.from_json(json_str)

        assert len(loaded) == 1
        gt = loaded.items[0].ground_truth
        assert gt[0] == CriterionVerdict.MET
        assert gt[1] == "3"

    def test_dataset_validates_multi_choice_labels(self, mixed_rubric):
        """RubricDataset.from_json validates multi-choice labels against options."""
        json_str = """
        {
            "prompt": "Test",
            "rubric": [
                {"name": "accuracy", "weight": 10.0, "requirement": "Accurate?"},
                {
                    "name": "satisfaction",
                    "weight": 5.0,
                    "requirement": "How satisfied?",
                    "scale_type": "ordinal",
                    "options": [
                        {"label": "Low", "value": 0.0, "na": false},
                        {"label": "High", "value": 1.0, "na": false}
                    ]
                }
            ],
            "items": [
                {
                    "submission": "Test",
                    "description": "Test",
                    "ground_truth": ["MET", "Invalid Label"]
                }
            ]
        }
        """
        with pytest.raises(ValueError, match="Label 'Invalid Label' not found"):
            RubricDataset.from_json(json_str)


# =============================================================================
# Aggregation Tests (Unit Tests)
# =============================================================================


class TestMultiChoiceAggregation:
    """Tests for multi-choice aggregation functions."""

    @pytest.fixture
    def ordinal_options(self):
        """Options for ordinal aggregation tests."""
        return [
            CriterionOption(label="1", value=0.0),
            CriterionOption(label="2", value=0.33),
            CriterionOption(label="3", value=0.67),
            CriterionOption(label="4", value=1.0),
        ]

    @pytest.fixture
    def nominal_options(self):
        """Options for nominal aggregation tests."""
        return [
            CriterionOption(label="Too few", value=0.0),
            CriterionOption(label="Too many", value=0.0),
            CriterionOption(label="Just right", value=1.0),
        ]

    def test_ordinal_mean_aggregation(self, ordinal_options):
        """Mean aggregation snaps to nearest option value."""
        from autorubric.graders.criterion_grader import CriterionGrader, JudgeSpec
        from autorubric import LLMConfig

        # Create a minimal grader to access aggregation methods
        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
            ordinal_aggregation="mean",
        )

        # Create votes
        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1",
                selected_index=1,
                selected_label="2",
                value=0.33,
                reason="",
            ),
            MultiChoiceJudgeVote(
                judge_id="j2",
                selected_index=2,
                selected_label="3",
                value=0.67,
                reason="",
            ),
            MultiChoiceJudgeVote(
                judge_id="j3",
                selected_index=3,
                selected_label="4",
                value=1.0,
                reason="",
            ),
        ]

        result = grader._aggregate_ordinal_votes(votes, ordinal_options, "mean")

        # Mean of [0.33, 0.67, 1.0] = 0.667
        # Nearest option: "3" with value 0.67
        assert result.selected_index == 2
        assert result.selected_label == "3"
        assert result.value == 0.67
        assert abs(result.aggregated_value - 0.667) < 0.01

    def test_nominal_mode_aggregation(self, nominal_options):
        """Mode aggregation picks most common selection."""
        from autorubric.graders.criterion_grader import CriterionGrader
        from autorubric import LLMConfig

        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
            nominal_aggregation="mode",
        )

        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1",
                selected_index=2,
                selected_label="Just right",
                value=1.0,
                reason="",
            ),
            MultiChoiceJudgeVote(
                judge_id="j2",
                selected_index=0,
                selected_label="Too few",
                value=0.0,
                reason="",
            ),
            MultiChoiceJudgeVote(
                judge_id="j3",
                selected_index=2,
                selected_label="Just right",
                value=1.0,
                reason="",
            ),
        ]

        result = grader._aggregate_nominal_votes(votes, nominal_options, "mode")

        # Mode: "Just right" appears twice
        assert result.selected_index == 2
        assert result.selected_label == "Just right"
        assert result.value == 1.0


# =============================================================================
# Option Shuffling Tests (Position Bias Mitigation)
# =============================================================================


class TestOptionShuffling:
    """Tests for option shuffling to mitigate position bias."""

    def test_shuffle_options_enabled_by_default(self):
        """CriterionGrader has shuffle_options=True by default."""
        from autorubric.graders.criterion_grader import CriterionGrader
        from autorubric import LLMConfig

        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
        )
        assert grader._shuffle_options is True

    def test_shuffle_options_can_be_disabled(self):
        """CriterionGrader shuffle_options can be set to False."""
        from autorubric.graders.criterion_grader import CriterionGrader
        from autorubric import LLMConfig

        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
            shuffle_options=False,
        )
        assert grader._shuffle_options is False

    def test_shuffle_index_mapping_logic(self):
        """Test that shuffle index mapping correctly maps back to original."""
        # This tests the core mapping logic without making LLM calls

        # Simulate: original options [A, B, C, D] at indices [0, 1, 2, 3]
        # Shuffled to position order [C, A, D, B]
        # shuffled_indices = [2, 0, 3, 1] means:
        #   - position 0 in shuffled list has original index 2 (C)
        #   - position 1 in shuffled list has original index 0 (A)
        #   - position 2 in shuffled list has original index 3 (D)
        #   - position 3 in shuffled list has original index 1 (B)
        shuffled_indices = [2, 0, 3, 1]

        # If LLM selects option 1 (1-indexed) = shuffled index 0
        # That's original index shuffled_indices[0] = 2 (option C)
        llm_selected = 1  # 1-indexed
        shuffled_idx = llm_selected - 1  # 0-indexed in shuffled space
        original_idx = shuffled_indices[shuffled_idx]
        assert original_idx == 2

        # If LLM selects option 3 (1-indexed) = shuffled index 2
        # That's original index shuffled_indices[2] = 3 (option D)
        llm_selected = 3
        shuffled_idx = llm_selected - 1
        original_idx = shuffled_indices[shuffled_idx]
        assert original_idx == 3

        # If LLM selects option 4 (1-indexed) = shuffled index 3
        # That's original index shuffled_indices[3] = 1 (option B)
        llm_selected = 4
        shuffled_idx = llm_selected - 1
        original_idx = shuffled_indices[shuffled_idx]
        assert original_idx == 1

    def test_shuffle_few_shot_example_transformation(self):
        """Test that few-shot example indices are correctly transformed."""
        # Original examples reference original indices
        # When shuffled, we need to transform to shuffled positions

        # shuffled_indices[shuffled_pos] = original_pos
        shuffled_indices = [2, 0, 3, 1]  # Same as above

        # Create inverse mapping: original_to_shuffled[original_pos] = shuffled_pos
        original_to_shuffled = {orig: shuf for shuf, orig in enumerate(shuffled_indices)}
        # Should be: {2: 0, 0: 1, 3: 2, 1: 3}
        assert original_to_shuffled == {2: 0, 0: 1, 3: 2, 1: 3}

        # Example originally points to index 0 (option A)
        # After shuffling, A is at position 1 in shuffled list
        original_example_idx = 0
        transformed_idx = original_to_shuffled[original_example_idx]
        assert transformed_idx == 1

        # Example originally points to index 2 (option C)
        # After shuffling, C is at position 0 in shuffled list
        original_example_idx = 2
        transformed_idx = original_to_shuffled[original_example_idx]
        assert transformed_idx == 0

    def test_shuffle_preserves_all_options(self):
        """Verify shuffling produces a valid permutation of all indices."""
        import random

        original_indices = list(range(5))  # [0, 1, 2, 3, 4]
        shuffled_indices = original_indices.copy()

        # Shuffle multiple times and verify invariants
        for _ in range(10):
            random.shuffle(shuffled_indices)

            # All original indices should be present
            assert sorted(shuffled_indices) == original_indices

            # Mapping back should cover all indices
            for shuffled_pos in range(len(shuffled_indices)):
                original_idx = shuffled_indices[shuffled_pos]
                assert 0 <= original_idx < len(original_indices)
