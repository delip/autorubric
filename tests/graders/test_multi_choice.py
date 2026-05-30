"""Tests for multi-choice criterion functionality."""

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    Rubric,
)
from autorubric.dataset import RubricDataset
from autorubric.types import MultiChoiceJudgeVote


def _mcvote(judge_id, selected_index, selected_label, value, weight=1.0, na=False):
    """Concise MultiChoiceJudgeVote builder for aggregation tests."""
    return MultiChoiceJudgeVote(
        judge_id=judge_id,
        selected_index=selected_index,
        selected_label=selected_label,
        value=value,
        reason="",
        weight=weight,
        na=na,
    )


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

    @pytest.mark.parametrize(
        ("fixture_name", "input_label", "expected_index"),
        [
            # Exact match.
            ("ordinal_criterion", "1", 0),
            ("ordinal_criterion", "4", 3),
            # Case-insensitive.
            ("nominal_criterion", "too few", 0),
            ("nominal_criterion", "JUST RIGHT", 2),
            # Whitespace-stripping.
            ("nominal_criterion", "  Too few  ", 0),
        ],
    )
    def test_find_option_by_label(self, request, fixture_name, input_label, expected_index):
        """find_option_by_label normalizes input (exact / case-insensitive / whitespace)."""
        criterion = request.getfixturevalue(fixture_name)
        assert criterion.find_option_by_label(input_label) == expected_index

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
        return Rubric(
            [
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
            ]
        )

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
    def ordinal_criterion(self):
        """Ordinal criterion (positive weight) for aggregation tests."""
        return Criterion(
            requirement="How satisfied?",
            scale_type="ordinal",
            weight=10.0,
            options=[
                CriterionOption(label="1", value=0.0),
                CriterionOption(label="2", value=0.33),
                CriterionOption(label="3", value=0.67),
                CriterionOption(label="4", value=1.0),
            ],
        )

    @pytest.fixture
    def nominal_criterion(self):
        """Nominal criterion (positive weight) for aggregation tests."""
        return Criterion(
            requirement="Length?",
            scale_type="nominal",
            weight=5.0,
            options=[
                CriterionOption(label="Too few", value=0.0),
                CriterionOption(label="Too many", value=0.0),
                CriterionOption(label="Just right", value=1.0),
            ],
        )

    @pytest.fixture
    def nominal_criterion_with_na(self):
        """Nominal criterion with an explicit NA option (index 3)."""
        return Criterion(
            requirement="Length?",
            scale_type="nominal",
            weight=5.0,
            options=[
                CriterionOption(label="Too few", value=0.0),
                CriterionOption(label="Too many", value=0.0),
                CriterionOption(label="Just right", value=1.0),
                CriterionOption(label="NA - not applicable", value=0.0, na=True),
            ],
        )

    def test_ordinal_mean_aggregation(self, ordinal_criterion):
        """Mean aggregation snaps to nearest option value."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

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

        result = grader._aggregate_ordinal_votes(votes, ordinal_criterion, "mean")

        # Mean of [0.33, 0.67, 1.0] = 0.667
        # Nearest option: "3" with value 0.67
        assert result.selected_index == 2
        assert result.selected_label == "3"
        assert result.value == 0.67
        assert abs(result.aggregated_value - 0.667) < 0.01

    def test_nominal_mode_aggregation(self, nominal_criterion):
        """Mode aggregation picks most common selection."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

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

        result = grader._aggregate_nominal_votes(votes, nominal_criterion, "mode")

        # Mode: "Just right" appears twice
        assert result.selected_index == 2
        assert result.selected_label == "Just right"
        assert result.value == 1.0

    @pytest.mark.parametrize(
        ("strategy", "expected_index", "expected_label", "expected_value"),
        [
            # Conservative: lowest selected value (0.67), not the mean (would snap to 1.0).
            ("min", 2, "3", 0.67),
            # Permissive: highest selected value.
            ("max", 3, "4", 1.0),
        ],
    )
    def test_ordinal_min_max_picks_extreme_selected_option(
        self, ordinal_criterion, strategy, expected_index, expected_label, expected_value
    ):
        """Ordinal 'min'/'max' return the extreme-value option any judge selected."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
            ordinal_aggregation=strategy,
        )

        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1", selected_index=2, selected_label="3", value=0.67, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j2", selected_index=2, selected_label="3", value=0.67, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j3", selected_index=3, selected_label="4", value=1.0, reason=""
            ),
        ]

        result = grader._aggregate_ordinal_votes(votes, ordinal_criterion, strategy)

        assert result.selected_index == expected_index
        assert result.selected_label == expected_label
        assert result.value == expected_value
        assert result.aggregated_value == expected_value
        assert result.na is False

    def test_ordinal_min_max_value_tie_breaks_to_lowest_index(self):
        """Value ties in 'min'/'max' resolve to the lowest option index (deterministic)."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))

        # Two distinct options share the same value (0.5).
        criterion = Criterion(
            requirement="r",
            scale_type="ordinal",
            weight=10.0,
            options=[
                CriterionOption(label="a", value=0.0),
                CriterionOption(label="b", value=0.5),
                CriterionOption(label="c", value=0.5),
                CriterionOption(label="d", value=1.0),
            ],
        )
        # Votes select the two tied options (idx 1, 2) plus the top option (idx 3).
        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1", selected_index=2, selected_label="c", value=0.5, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j2", selected_index=1, selected_label="b", value=0.5, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j3", selected_index=3, selected_label="d", value=1.0, reason=""
            ),
        ]

        # min value is 0.5, shared by idx 1 and 2 -> lowest index (1) wins.
        min_result = grader._aggregate_ordinal_votes(votes, criterion, "min")
        assert min_result.selected_index == 1

        # For max, build a set whose max value is shared by idx 1 and 2.
        votes_max = [
            MultiChoiceJudgeVote(
                judge_id="j1", selected_index=2, selected_label="c", value=0.5, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j2", selected_index=1, selected_label="b", value=0.5, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j3", selected_index=0, selected_label="a", value=0.0, reason=""
            ),
        ]
        max_result = grader._aggregate_ordinal_votes(votes_max, criterion, "max")
        assert max_result.selected_index == 1

    # -------------------------------------------------------------------------
    # Deterministic, weight-sign-aware tie-breaking
    #
    # mode / weighted_mode count/weight ties and mean/median snap equidistant ties all
    # resolve to the score-minimizing option by weight sign (lowest value for weight >= 0,
    # highest for weight < 0; lowest index on a value tie) via Criterion.worst_option_among.
    # Each case orders votes so the OLD first-seen / lowest-index behavior would pick the
    # OTHER option, pinning both the new rule and order-independence.
    # -------------------------------------------------------------------------

    def _make_ordinal(self, weight):
        return Criterion(
            requirement="r",
            scale_type="ordinal",
            weight=weight,
            options=[
                CriterionOption(label="1", value=0.0),
                CriterionOption(label="2", value=0.33),
                CriterionOption(label="3", value=0.67),
                CriterionOption(label="4", value=1.0),
            ],
        )

    def _make_nominal(self, weight):
        # Non-trivial values so worst-by-value is unambiguous: A=1.0, B=0.0, C=0.5.
        return Criterion(
            requirement="r",
            scale_type="nominal",
            weight=weight,
            options=[
                CriterionOption(label="A", value=1.0),
                CriterionOption(label="B", value=0.0),
                CriterionOption(label="C", value=0.5),
            ],
        )

    @pytest.mark.parametrize(
        ("scale_type", "weight", "votes_spec", "expected_index"),
        [
            # Ordinal, positive weight → lowest-value tied option (idx 1, not first-seen idx 3).
            ("ordinal", 10.0, [(3, "4", 1.0), (3, "4", 1.0), (1, "2", 0.33), (1, "2", 0.33)], 1),
            # Ordinal, negative weight → highest-value tied option (idx 3).
            ("ordinal", -10.0, [(3, "4", 1.0), (3, "4", 1.0), (1, "2", 0.33), (1, "2", 0.33)], 3),
            # Nominal, positive weight → lowest-value tied option (idx 1, B=0.0; not first-seen 0).
            ("nominal", 5.0, [(0, "A", 1.0), (0, "A", 1.0), (1, "B", 0.0), (1, "B", 0.0)], 1),
            # Nominal, negative weight → highest-value tied option (idx 0, A=1.0).
            ("nominal", -5.0, [(0, "A", 1.0), (0, "A", 1.0), (1, "B", 0.0), (1, "B", 0.0)], 0),
        ],
    )
    def test_mode_count_tie_breaks_to_worst_by_weight_sign(
        self, scale_type, weight, votes_spec, expected_index
    ):
        """Mode count-tie → score-minimizing tied option by weight sign (both dispatch paths)."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))
        votes = [
            _mcvote(f"j{i}", idx, label, value) for i, (idx, label, value) in enumerate(votes_spec)
        ]
        if scale_type == "ordinal":
            criterion = self._make_ordinal(weight=weight)
            result = grader._aggregate_ordinal_votes(votes, criterion, "mode")
        else:
            criterion = self._make_nominal(weight=weight)
            result = grader._aggregate_nominal_votes(votes, criterion, "mode")
        assert result.selected_index == expected_index

    def test_ordinal_snap_equidistant_tie_breaks_to_worst_negative_weight(self):
        """Mean snap equidistant-tie, negative weight → higher-value option."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))
        criterion = Criterion(
            requirement="r",
            scale_type="ordinal",
            weight=-10.0,
            options=[
                CriterionOption(label="a", value=0.0),
                CriterionOption(label="b", value=0.4),
                CriterionOption(label="c", value=0.6),
                CriterionOption(label="d", value=1.0),
            ],
        )
        # mean([0.4, 0.6]) = 0.5, equidistant from idx 1 (0.4) and idx 2 (0.6).
        votes = [_mcvote("j1", 1, "b", 0.4), _mcvote("j2", 2, "c", 0.6)]
        result = grader._aggregate_ordinal_votes(votes, criterion, "mean")
        # Negative weight → worst is the higher-value option (idx 2); old snap picked idx 1.
        assert result.selected_index == 2

    def test_ordinal_snap_equidistant_tie_is_value_based_not_index_based(self):
        """Snap tie is resolved by value (worst), not by lowest index."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))
        # Non-monotonic values: lowest value (0.4) sits at the HIGHER index (1).
        criterion = Criterion(
            requirement="r",
            scale_type="ordinal",
            weight=10.0,
            options=[
                CriterionOption(label="a", value=0.6),
                CriterionOption(label="b", value=0.4),
                CriterionOption(label="c", value=1.0),
            ],
        )
        votes = [_mcvote("j1", 0, "a", 0.6), _mcvote("j2", 1, "b", 0.4)]  # mean 0.5
        result = grader._aggregate_ordinal_votes(votes, criterion, "mean")
        # Positive weight → worst is the lower-value option (idx 1); old snap picked idx 0.
        assert result.selected_index == 1

    def test_nominal_weighted_mode_weight_tie_breaks_to_worst(self):
        """Nominal weighted_mode weight-tie → lowest-value tied option (not first-seen)."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))
        criterion = self._make_nominal(weight=5.0)
        # Equal summed weight on idx 0 (A, 1.0) and idx 1 (B, 0.0); idx 0 seen first.
        votes = [
            _mcvote("j1", 0, "A", 1.0, weight=1.0),
            _mcvote("j2", 1, "B", 0.0, weight=1.0),
        ]
        result = grader._aggregate_nominal_votes(votes, criterion, "weighted_mode")
        assert result.selected_index == 1

    def test_nominal_unanimous_all_agree_selects_that_option(self, nominal_criterion):
        """Nominal 'unanimous' returns the agreed option when all judges concur."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
            nominal_aggregation="unanimous",
        )

        votes = [
            MultiChoiceJudgeVote(
                judge_id=f"j{i}",
                selected_index=2,
                selected_label="Just right",
                value=1.0,
                reason="",
            )
            for i in range(3)
        ]

        result = grader._aggregate_nominal_votes(votes, nominal_criterion, "unanimous")

        assert result.selected_index == 2
        assert result.selected_label == "Just right"
        assert result.na is False

    def test_nominal_unanimous_disagreement_with_na_abstains(self, nominal_criterion_with_na):
        """Nominal 'unanimous' abstains via the NA option when judges disagree."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
            nominal_aggregation="unanimous",
        )

        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1", selected_index=0, selected_label="Too few", value=0.0, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j2", selected_index=1, selected_label="Too many", value=0.0, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j3", selected_index=2, selected_label="Just right", value=1.0, reason=""
            ),
        ]

        result = grader._aggregate_nominal_votes(votes, nominal_criterion_with_na, "unanimous")

        # No consensus -> abstain via the NA option (index 3), not the mode.
        assert result.na is True
        assert result.selected_index == 3
        assert result.selected_label == "NA - not applicable"

    def test_nominal_unanimous_disagreement_without_na_falls_back_to_mode_and_warns(
        self, nominal_criterion, caplog
    ):
        """Without an NA option, disagreeing 'unanimous' falls back to mode and warns."""
        import logging

        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
            nominal_aggregation="unanimous",
        )

        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1", selected_index=2, selected_label="Just right", value=1.0, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j2", selected_index=0, selected_label="Too few", value=0.0, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j3", selected_index=2, selected_label="Just right", value=1.0, reason=""
            ),
        ]

        with caplog.at_level(logging.WARNING):
            result = grader._aggregate_nominal_votes(votes, nominal_criterion, "unanimous")

        # Falls back to mode ("Just right", twice) without abstaining.
        assert result.selected_index == 2
        assert result.na is False
        assert any("unanimous" in r.message and "NA option" in r.message for r in caplog.records)

    def test_nominal_unanimous_differs_from_mode_on_disagreement(self, nominal_criterion_with_na):
        """'unanimous' is no longer a no-op alias of 'mode' on disagreement."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))

        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1", selected_index=0, selected_label="Too few", value=0.0, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j2", selected_index=1, selected_label="Too many", value=0.0, reason=""
            ),
            MultiChoiceJudgeVote(
                judge_id="j3", selected_index=2, selected_label="Just right", value=1.0, reason=""
            ),
        ]

        unanimous_result = grader._aggregate_nominal_votes(
            votes, nominal_criterion_with_na, "unanimous"
        )
        mode_result = grader._aggregate_nominal_votes(votes, nominal_criterion_with_na, "mode")

        assert unanimous_result.na is True
        assert mode_result.na is False
        assert unanimous_result.selected_index != mode_result.selected_index

    def test_all_na_prefers_genuine_na_index_over_none(self, nominal_criterion_with_na):
        """All-NA aggregation: prefer a vote that abstained into a real NA option.

        When NA votes mix a clean None-abstain (error, no option) with a genuine NA-option
        abstain, the aggregate surfaces the real NA index, not None.
        """
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))

        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1",
                selected_index=None,
                selected_label=None,
                value=0.0,
                reason="",
                na=True,
                error="infrastructure: down",
            ),
            MultiChoiceJudgeVote(
                judge_id="j2",
                selected_index=3,
                selected_label="NA - not applicable",
                value=0.0,
                reason="",
                na=True,
            ),
        ]

        result, _ = grader._aggregate_multi_choice_votes(votes, nominal_criterion_with_na)
        assert result.na is True
        assert result.selected_index == 3
        assert result.selected_label == "NA - not applicable"

    def test_all_na_all_none_yields_clean_abstain(self, nominal_criterion):
        """All-NA aggregation where every NA vote is a None-abstain -> clean None aggregate."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))

        votes = [
            MultiChoiceJudgeVote(
                judge_id="j1",
                selected_index=None,
                selected_label=None,
                value=0.0,
                reason="",
                na=True,
                error="infrastructure: a",
            ),
            MultiChoiceJudgeVote(
                judge_id="j2",
                selected_index=None,
                selected_label=None,
                value=0.0,
                reason="",
                na=True,
                error="infrastructure: b",
            ),
        ]

        result, _ = grader._aggregate_multi_choice_votes(votes, nominal_criterion)
        assert result.na is True
        assert result.selected_index is None
        assert result.selected_label is None


# =============================================================================
# Option Shuffling Tests (Position Bias Mitigation)
# =============================================================================


class TestOptionShuffling:
    """Tests for option shuffling to mitigate position bias."""

    def test_shuffle_options_enabled_by_default(self):
        """CriterionGrader has shuffle_options=True by default."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
        )
        assert grader._shuffle_options is True


class TestSeedReproducibility:
    """Tests for master seed coordination and reproducibility."""

    def test_seed_auto_generated(self):
        """Grader without explicit seed gets a valid auto-generated seed."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"))
        assert isinstance(grader.seed, int)
        assert 0 <= grader.seed < 2**31

    def test_seed_explicit(self):
        """Grader uses the explicit seed when provided."""
        from autorubric import LLMConfig
        from autorubric.graders.criterion_grader import CriterionGrader

        grader = CriterionGrader(llm_config=LLMConfig(model="openai/gpt-4"), seed=42)
        assert grader.seed == 42

    def test_seed_produces_deterministic_shuffle(self):
        """Same seed + same inputs → same shuffle order."""

        from autorubric.graders.criterion_grader import _derive_shuffle_rng

        indices_a = list(range(5))
        rng_a = _derive_shuffle_rng(42, "abc123", 0, "default")
        rng_a.shuffle(indices_a)

        indices_b = list(range(5))
        rng_b = _derive_shuffle_rng(42, "abc123", 0, "default")
        rng_b.shuffle(indices_b)

        assert indices_a == indices_b

    @pytest.mark.parametrize(
        ("key_a", "key_b"),
        [
            # Vary the master_seed slot.
            ((42, "abc123", 0, "default"), (99, "abc123", 0, "default")),
            # Vary the item_key slot.
            ((42, "item_one", 0, "default"), (42, "item_two", 0, "default")),
            # Vary the judge_id slot.
            ((42, "abc123", 0, "judge_a"), (42, "abc123", 0, "judge_b")),
        ],
    )
    def test_distinct_key_slots_produce_different_shuffles(self, key_a, key_b):
        """Each of seed / item_key / judge_id participates in the SHA-256 shuffle key."""
        from autorubric.graders.criterion_grader import _derive_shuffle_rng

        indices_a = list(range(10))
        rng_a = _derive_shuffle_rng(*key_a)
        rng_a.shuffle(indices_a)

        indices_b = list(range(10))
        rng_b = _derive_shuffle_rng(*key_b)
        rng_b.shuffle(indices_b)

        assert indices_a != indices_b

    @pytest.mark.parametrize(
        ("few_shot_seed", "expected_seed"),
        [
            # Unset few-shot seed → coordinated from the master seed.
            (None, 42),
            # Explicit few-shot seed → preserved, not overridden by the master seed.
            (99, 99),
        ],
    )
    def test_few_shot_seed_coordination(self, few_shot_seed, expected_seed):
        """Master seed flows to FewShotConfig only when its seed is unset."""
        from autorubric import Criterion, FewShotConfig, LLMConfig, Rubric
        from autorubric.dataset import RubricDataset
        from autorubric.graders.criterion_grader import CriterionGrader

        rubric = Rubric(rubric=[Criterion(weight=1.0, requirement="test")])
        dataset = RubricDataset(
            name="test",
            prompt="test",
            items=[],
            rubric=rubric,
        )
        few_shot_config = (
            FewShotConfig(n_examples=2)
            if few_shot_seed is None
            else FewShotConfig(n_examples=2, seed=few_shot_seed)
        )
        grader = CriterionGrader(
            llm_config=LLMConfig(model="openai/gpt-4"),
            training_data=dataset,
            few_shot_config=few_shot_config,
            seed=42,
        )
        assert grader._few_shot_config.seed == expected_seed

    def test_shuffle_order_field_in_criterion_report(self):
        """CriterionReport supports the shuffle_order field."""
        from autorubric.types import CriterionReport

        report = CriterionReport(
            weight=1.0,
            requirement="test",
            verdict=None,
            reason="test",
            shuffle_order=[2, 0, 1],
        )
        assert report.shuffle_order == [2, 0, 1]

        # Serialization includes shuffle_order
        dumped = report.model_dump(mode="json")
        assert dumped["shuffle_order"] == [2, 0, 1]

    def test_shuffle_order_none_for_binary(self):
        """Binary criteria have shuffle_order=None."""
        from autorubric.types import CriterionReport, CriterionVerdict

        report = CriterionReport(
            weight=1.0,
            requirement="test",
            verdict=CriterionVerdict.MET,
            reason="test",
        )
        assert report.shuffle_order is None

    def test_shuffle_order_backward_compat(self):
        """Old serialized data without shuffle_order deserializes correctly."""
        from autorubric.types import CriterionReport

        old_data = {
            "weight": 1.0,
            "requirement": "test",
            "verdict": None,
            "reason": "test",
        }
        report = CriterionReport.model_validate(old_data)
        assert report.shuffle_order is None
