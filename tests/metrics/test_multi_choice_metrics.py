"""Tests for multi-choice (ordinal and nominal) metrics computation."""

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric
from autorubric.dataset import RubricDataset
from autorubric.eval import EvalResult, ItemResult
from autorubric.metrics import (
    classify_criteria,
    classify_criterion,
    compute_metrics,
    filter_na_multi_choice,
    get_option_value,
    is_na_option,
    resolve_ground_truth,
)
from autorubric.metrics._compute import (
    _compute_nominal_criterion_metrics,
    _compute_ordinal_criterion_metrics,
    _compute_per_option_metrics,
)
from autorubric.types import CriterionReport, EvaluationReport, MultiChoiceVerdict

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ordinal_criterion() -> Criterion:
    """Create an ordinal criterion (satisfaction scale)."""
    return Criterion(
        name="satisfaction",
        weight=10.0,
        requirement="How satisfied are you?",
        scale_type="ordinal",
        options=[
            {"label": "Very dissatisfied", "value": 0.0},
            {"label": "Dissatisfied", "value": 0.33},
            {"label": "Satisfied", "value": 0.67},
            {"label": "Very satisfied", "value": 1.0},
        ],
    )


@pytest.fixture
def nominal_criterion() -> Criterion:
    """Create a nominal criterion (response length)."""
    return Criterion(
        name="length",
        weight=5.0,
        requirement="Is the response length appropriate?",
        scale_type="nominal",
        options=[
            {"label": "Too brief", "value": 0.0},
            {"label": "Too verbose", "value": 0.0},
            {"label": "Just right", "value": 1.0},
        ],
    )


@pytest.fixture
def nominal_criterion_with_na() -> Criterion:
    """Create an ordinal criterion with an NA option.

    NOTE: This fixture is misnamed for historical reasons — ``scale_type`` is
    ``"ordinal"``. Existing tests rely on the name. For genuinely nominal NA
    fixtures, see :func:`true_nominal_criterion_with_na` below.
    """
    return Criterion(
        name="specificity",
        weight=6.0,
        requirement="How specific are the recommendations?",
        scale_type="ordinal",
        options=[
            {"label": "Vague", "value": 0.0},
            {"label": "Somewhat specific", "value": 0.5},
            {"label": "Very specific", "value": 1.0},
            {"label": "N/A", "value": 0.0, "na": True},
        ],
    )


@pytest.fixture
def true_nominal_criterion_with_na() -> Criterion:
    """Genuinely nominal criterion with an NA option (for ``as_category`` tests)."""
    return Criterion(
        name="category",
        weight=4.0,
        requirement="Which category?",
        scale_type="nominal",
        options=[
            {"label": "Alpha", "value": 1.0},
            {"label": "Beta", "value": 0.0},
            {"label": "Gamma", "value": 0.5},
            {"label": "N/A", "value": 0.0, "na": True},
        ],
    )


@pytest.fixture
def negative_weight_criterion_with_na() -> Criterion:
    """Ordinal criterion with NEGATIVE weight and an NA option.

    For negative weight, the score-minimizing scored option is the HIGHEST
    value (a high value on a negative-weight criterion subtracts more from
    the score). Used to pin weight-sign aware ``as_unmet`` remapping.
    """
    return Criterion(
        name="severity_penalty",
        weight=-8.0,
        requirement="How severe is the safety violation?",
        scale_type="ordinal",
        options=[
            {"label": "None", "value": 0.0},
            {"label": "Minor", "value": 0.5},
            {"label": "Severe", "value": 1.0},
            {"label": "N/A", "value": 0.0, "na": True},
        ],
    )


@pytest.fixture
def binary_criterion() -> Criterion:
    """Create a binary criterion."""
    return Criterion(
        name="accuracy",
        weight=10.0,
        requirement="Is the response factually accurate?",
    )


@pytest.fixture
def hybrid_rubric(binary_criterion, ordinal_criterion, nominal_criterion) -> Rubric:
    """Create a rubric with mixed criterion types."""
    return Rubric([binary_criterion, ordinal_criterion, nominal_criterion])


# =============================================================================
# Test classify_criterion and classify_criteria
# =============================================================================


class TestClassifyCriterion:
    """Tests for classify_criterion helper."""

    def test_binary_criterion(self, binary_criterion):
        """Binary criteria are classified as 'binary'."""
        assert classify_criterion(binary_criterion) == "binary"

    def test_ordinal_criterion(self, ordinal_criterion):
        """Ordinal criteria are classified as 'ordinal'."""
        assert classify_criterion(ordinal_criterion) == "ordinal"

    def test_nominal_criterion(self, nominal_criterion):
        """Nominal criteria are classified as 'nominal'."""
        assert classify_criterion(nominal_criterion) == "nominal"


class TestClassifyCriteria:
    """Tests for classify_criteria helper."""

    def test_all_binary(self):
        """Classify all binary criteria."""
        criteria = [
            Criterion(weight=1.0, requirement="R1"),
            Criterion(weight=1.0, requirement="R2"),
        ]
        assert classify_criteria(criteria) == ["binary", "binary"]

    def test_mixed_types(self, binary_criterion, ordinal_criterion, nominal_criterion):
        """Classify mixed criteria types."""
        criteria = [binary_criterion, ordinal_criterion, nominal_criterion]
        types = classify_criteria(criteria)
        assert types == ["binary", "ordinal", "nominal"]


# =============================================================================
# Test resolve_ground_truth
# =============================================================================


class TestResolveGroundTruth:
    """Tests for resolve_ground_truth helper."""

    def test_binary_passthrough(self, binary_criterion):
        """Binary criteria keep CriterionVerdict unchanged."""
        ground_truth = [CriterionVerdict.MET]
        resolved = resolve_ground_truth(ground_truth, [binary_criterion])
        assert resolved == [CriterionVerdict.MET]

    def test_binary_string_to_verdict(self, binary_criterion):
        """Binary criteria accept string verdicts."""
        ground_truth = ["MET"]
        resolved = resolve_ground_truth(ground_truth, [binary_criterion])
        assert resolved == [CriterionVerdict.MET]

    def test_multi_choice_label_to_index(self, ordinal_criterion):
        """Multi-choice criteria resolve labels to indices."""
        ground_truth = ["Very satisfied"]
        resolved = resolve_ground_truth(ground_truth, [ordinal_criterion])
        assert resolved == [3]  # Index of "Very satisfied"

    def test_multi_choice_index_passthrough(self, ordinal_criterion):
        """Multi-choice criteria pass through integer indices."""
        ground_truth = [2]
        resolved = resolve_ground_truth(ground_truth, [ordinal_criterion])
        assert resolved == [2]

    def test_mixed_hybrid(self, binary_criterion, ordinal_criterion):
        """Mixed binary and multi-choice criteria."""
        criteria = [binary_criterion, ordinal_criterion]
        ground_truth = [CriterionVerdict.MET, "Satisfied"]
        resolved = resolve_ground_truth(ground_truth, criteria)
        assert resolved[0] == CriterionVerdict.MET
        assert resolved[1] == 2  # Index of "Satisfied"

    def test_invalid_binary_verdict_raises(self, binary_criterion):
        """Invalid binary verdict raises ValueError."""
        with pytest.raises(ValueError, match="Invalid binary verdict"):
            resolve_ground_truth(["INVALID"], [binary_criterion])

    def test_mismatched_length_raises(self, binary_criterion):
        """Mismatched ground truth length raises ValueError."""
        with pytest.raises(ValueError, match="doesn't match"):
            resolve_ground_truth([CriterionVerdict.MET, CriterionVerdict.UNMET], [binary_criterion])


# =============================================================================
# Test filter_na_multi_choice
# =============================================================================


class TestFilterNaMultiChoice:
    """Tests for filter_na_multi_choice helper."""

    def test_no_na_options(self, ordinal_criterion):
        """No NA options returns data unchanged."""
        pred = [0, 1, 2, 3]
        true = [1, 2, 3, 0]
        filtered_pred, filtered_true, na_agree, na_fp, na_fn = filter_na_multi_choice(
            pred, true, ordinal_criterion
        )
        assert filtered_pred == pred
        assert filtered_true == true
        assert na_agree == 0
        assert na_fp == 0
        assert na_fn == 0

    def test_exclude_na(self, nominal_criterion_with_na):
        """Exclude mode removes NA pairs."""
        pred = [0, 3, 2, 3]  # 3 is NA index
        true = [1, 3, 2, 0]  # 3 is NA index
        filtered_pred, filtered_true, na_agree, na_fp, na_fn = filter_na_multi_choice(
            pred, true, nominal_criterion_with_na, mode="exclude"
        )
        # First pair: both non-NA, keep
        # Second pair: both NA, skip (NA agreement)
        # Third pair: both non-NA, keep
        # Fourth pair: pred NA, true non-NA, skip (NA FP)
        assert len(filtered_pred) == 2
        assert len(filtered_true) == 2
        assert na_agree == 1
        assert na_fp == 1
        assert na_fn == 0

    # ---- T1-C: the full mirror of cannot_assess ---------------------------

    def test_as_worst_literal_is_rejected(self, nominal_criterion_with_na):
        """The old 'as_worst' literal is gone (hard break).

        Anyone passing it should get a clean ValueError pointing at the new
        modes; we do not silently accept the misnomer.
        """
        with pytest.raises(ValueError, match="as_worst|na_mode|mode"):
            filter_na_multi_choice(
                [0, 1, 2],
                [0, 1, 2],
                nominal_criterion_with_na,
                mode="as_worst",  # type: ignore[arg-type]
            )

    def test_as_category_keeps_na_as_column_nominal(self, true_nominal_criterion_with_na):
        """Nominal + as_category: NA pairs pass through unchanged (NA is its own column)."""
        # NA option is index 3.
        pred = [0, 3, 1, 3, 2]
        true = [0, 3, 2, 1, 2]
        filtered_pred, filtered_true, na_agree, na_fp, na_fn = filter_na_multi_choice(
            pred, true, true_nominal_criterion_with_na, mode="as_category"
        )
        # All pairs survive, including NA-on-NA, NA-on-non-NA, non-NA-on-NA.
        assert filtered_pred == pred
        assert filtered_true == true
        # FP/FN diagnostics still populate (mode-independent).
        assert na_agree == 1  # second pair (3, 3)
        assert na_fp == 1  # fourth pair (3, 1)
        assert na_fn == 0

    def test_as_category_raises_on_ordinal_with_na(self, nominal_criterion_with_na):
        """Ordinal + as_category + criterion has NA option → ValueError.

        NA has no ordinal position; quadratic-weighted Cohen's kappa would
        assign NA a geometrically meaningless distance based on its index.
        """
        # nominal_criterion_with_na is actually ordinal (see fixture docstring)
        # and has an NA option.
        with pytest.raises(ValueError, match="ordinal.*NA|as_category.*ordinal"):
            filter_na_multi_choice(
                [0, 3, 2],
                [1, 3, 2],
                nominal_criterion_with_na,
                mode="as_category",
            )

    def test_as_category_allowed_on_ordinal_without_na(self, ordinal_criterion):
        """Ordinal + as_category + criterion has NO NA option → no-op pass-through.

        The ordinal-NA combination is what's incoherent; an ordinal criterion
        without an NA option has no NA cells, so the guard does not fire.
        """
        pred = [0, 1, 2, 3]
        true = [1, 2, 3, 0]
        filtered_pred, filtered_true, na_agree, na_fp, na_fn = filter_na_multi_choice(
            pred, true, ordinal_criterion, mode="as_category"
        )
        assert filtered_pred == pred
        assert filtered_true == true
        assert na_agree == 0
        assert na_fp == 0
        assert na_fn == 0

    def test_as_unmet_remaps_na_to_lowest_value_positive_weight(self, nominal_criterion_with_na):
        """as_unmet + positive weight: NA → lowest-value scored option (index 0)."""
        # Options: [Vague(0.0), Somewhat(0.5), Very(1.0), N/A(na=True)]. Weight +6.
        # Worst scored = Vague at index 0.
        pred = [3, 1, 3, 2]  # NA at positions 0, 2
        true = [3, 1, 0, 3]  # NA at positions 0, 3
        filtered_pred, filtered_true, na_agree, na_fp, na_fn = filter_na_multi_choice(
            pred, true, nominal_criterion_with_na, mode="as_unmet"
        )
        # NA → 0 (lowest-value scored)
        assert filtered_pred == [0, 1, 0, 2]
        assert filtered_true == [0, 1, 0, 0]
        # All 4 pairs preserved (no drop under as_unmet).
        assert len(filtered_pred) == 4
        # FP/FN counts populate from the unremapped pairs.
        assert na_agree == 1  # (3, 3) at index 0
        assert na_fp == 1  # (3, 0) at index 2
        assert na_fn == 1  # (2, 3) at index 3

    def test_as_unmet_remaps_na_to_highest_value_negative_weight(
        self, negative_weight_criterion_with_na
    ):
        """as_unmet + NEGATIVE weight: NA → HIGHEST-value scored option.

        Pins weight-sign awareness: for negative weight, a high value subtracts
        more from the score, so the worst case flips.
        """
        # Options: [None(0.0), Minor(0.5), Severe(1.0), N/A(na=True)]. Weight -8.
        # Worst scored = Severe at index 2.
        pred = [3, 0, 3]
        true = [1, 3, 0]
        filtered_pred, filtered_true, _na_agree, _na_fp, _na_fn = filter_na_multi_choice(
            pred, true, negative_weight_criterion_with_na, mode="as_unmet"
        )
        # NA index 3 → index 2 (Severe, highest value, worst for negative weight).
        assert filtered_pred == [2, 0, 2]
        assert filtered_true == [1, 2, 0]

    def test_as_unmet_matches_grader_worst_case(
        self, nominal_criterion_with_na, negative_weight_criterion_with_na
    ):
        """The metrics ``as_unmet`` remap and the grader unknown-error worst case
        resolve to the same option for the same criterion.

        Pins the cross-layer reuse contract: both paths share
        ``Criterion.worst_scored_option()``.
        """
        for criterion in (nominal_criterion_with_na, negative_weight_criterion_with_na):
            method_idx, _method_opt = criterion.worst_scored_option()
            na_idx = next(i for i, opt in enumerate(criterion.options) if opt.na)
            filtered_pred, _filtered_true, _agree, _fp, _fn = filter_na_multi_choice(
                [na_idx],
                [na_idx],
                criterion,
                mode="as_unmet",
            )
            assert filtered_pred == [method_idx]

    def test_fp_fn_counts_under_all_modes(self, true_nominal_criterion_with_na):
        """na_fp and na_fn invariants: identical across modes; only dropping differs."""
        # NA option index = 3.
        pred = [0, 3, 1, 3]  # NA at positions 1, 3
        true = [0, 3, 3, 1]  # NA at positions 1, 2
        # (0,0) both non-NA. (3,3) both NA → agreement. (1,3) true NA → FN. (3,1) pred NA → FP.
        expected_agree, expected_fp, expected_fn = 1, 1, 1

        for mode in ("exclude", "as_unmet", "as_category"):
            _fp, _ft, na_agree, na_fp, na_fn = filter_na_multi_choice(
                pred, true, true_nominal_criterion_with_na, mode=mode
            )
            assert (na_agree, na_fp, na_fn) == (expected_agree, expected_fp, expected_fn), (
                f"NA counts drifted under mode={mode!r}"
            )

    def test_no_na_option_is_passthrough_all_modes(self, ordinal_criterion):
        """No NA option in criterion → all three modes pass through unchanged."""
        pred = [0, 1, 2, 3]
        true = [1, 2, 3, 0]
        for mode in ("exclude", "as_unmet", "as_category"):
            filtered_pred, filtered_true, na_agree, na_fp, na_fn = filter_na_multi_choice(
                pred, true, ordinal_criterion, mode=mode
            )
            assert filtered_pred == pred, mode
            assert filtered_true == true, mode
            assert (na_agree, na_fp, na_fn) == (0, 0, 0), mode


class TestCriterionWorstScoredOption:
    """Tests for the shared ``Criterion.worst_scored_option`` helper.

    This is the extracted worst-case selection used by both the grader's
    ``unknown``-error path (``criterion_grader.py``) and the metrics layer's
    ``na_mode="as_unmet"`` remap. The two paths cannot drift because they
    share this method.
    """

    def test_positive_weight_picks_lowest_value(self, nominal_criterion_with_na):
        """Positive weight → lowest-value scored option."""
        idx, opt = nominal_criterion_with_na.worst_scored_option()
        assert idx == 0
        assert opt.value == 0.0
        assert opt.na is False

    def test_negative_weight_picks_highest_value(self, negative_weight_criterion_with_na):
        """Negative weight → highest-value scored option (worst case flips)."""
        idx, opt = negative_weight_criterion_with_na.worst_scored_option()
        assert idx == 2  # Severe, value=1.0
        assert opt.value == 1.0
        assert opt.na is False

    def test_skips_na_options(self):
        """NA options are excluded from the search even if they have the lowest value."""
        criterion = Criterion(
            name="t",
            weight=10.0,
            requirement="R",
            scale_type="nominal",
            options=[
                {"label": "A", "value": 0.5},
                {"label": "B", "value": 0.7},
                # NA has the lowest value but must be skipped.
                {"label": "N/A", "value": 0.0, "na": True},
            ],
        )
        idx, opt = criterion.worst_scored_option()
        assert idx == 0
        assert opt.value == 0.5

    def test_ties_pick_first(self):
        """Ties at the worst value resolve to the first declared option."""
        criterion = Criterion(
            name="t",
            weight=10.0,
            requirement="R",
            scale_type="nominal",
            options=[
                {"label": "A", "value": 0.3},
                {"label": "B", "value": 0.3},
                {"label": "C", "value": 1.0},
            ],
        )
        idx, opt = criterion.worst_scored_option()
        assert idx == 0
        assert opt.label == "A"

    def test_binary_criterion_raises(self, binary_criterion):
        """Calling on a binary criterion is a programmer error."""
        with pytest.raises(ValueError, match="Binary criterion"):
            binary_criterion.worst_scored_option()


class TestCriterionWorstOptionAmong:
    """Tests for ``Criterion.worst_option_among`` — the score-minimizing option among a
    candidate subset, weight-sign aware with a deterministic lowest-index tie-break.

    This is the canonical tie-break shared by ensemble vote aggregation
    (mode/weighted_mode/snap, ``criterion_grader.py``) and ``worst_scored_option`` itself,
    so scoring, the unknown-error path, and aggregation cannot drift.
    """

    def test_positive_weight_picks_lowest_value_among_candidates(self, ordinal_criterion):
        """Positive weight → the lowest-value candidate (not the global lowest)."""
        # ordinal_criterion values: [0.0, 0.33, 0.67, 1.0]. Among {1, 2, 3} the worst is 1.
        assert ordinal_criterion.worst_option_among([3, 2, 1]) == 1

    def test_negative_weight_picks_highest_value_among_candidates(
        self, negative_weight_criterion_with_na
    ):
        """Negative weight → the highest-value candidate (worst case flips)."""
        # values: None=0.0(0), Minor=0.5(1), Severe=1.0(2). Among {0, 1, 2} the worst is 2.
        assert negative_weight_criterion_with_na.worst_option_among([0, 1, 2]) == 2

    def test_value_tie_breaks_to_lowest_index_order_independent(self):
        """Positive weight, value tie → lowest index, regardless of candidate order."""
        criterion = Criterion(
            name="t",
            weight=10.0,
            requirement="R",
            scale_type="nominal",
            options=[
                {"label": "A", "value": 0.0},  # idx 0  ─┐ tie at the worst value
                {"label": "B", "value": 0.5},  # idx 1   │
                {"label": "C", "value": 0.0},  # idx 2  ─┘
            ],
        )
        assert criterion.worst_option_among([2, 0]) == 0
        assert criterion.worst_option_among([0, 2]) == 0

    def test_negative_weight_value_tie_breaks_to_lowest_index(self):
        """Negative weight, value tie at the highest value → lowest index."""
        criterion = Criterion(
            name="t",
            weight=-3.0,
            requirement="R",
            scale_type="nominal",
            options=[
                {"label": "A", "value": 1.0},  # idx 0  ─┐ tie at the worst (highest) value
                {"label": "B", "value": 0.5},  # idx 1   │
                {"label": "C", "value": 1.0},  # idx 2  ─┘
            ],
        )
        assert criterion.worst_option_among([2, 0]) == 0
        assert criterion.worst_option_among([0, 2]) == 0

    def test_single_candidate_returned_as_is(self, ordinal_criterion):
        assert ordinal_criterion.worst_option_among([2]) == 2

    def test_empty_candidates_raises(self, ordinal_criterion):
        with pytest.raises(ValueError):
            ordinal_criterion.worst_option_among([])

    def test_binary_criterion_raises(self, binary_criterion):
        with pytest.raises(ValueError, match="Binary criterion"):
            binary_criterion.worst_option_among([0])

    def test_worst_scored_option_delegates_consistently(
        self, ordinal_criterion, negative_weight_criterion_with_na
    ):
        """``worst_scored_option`` equals ``worst_option_among`` over all non-NA indices."""
        for criterion in (ordinal_criterion, negative_weight_criterion_with_na):
            non_na = [i for i, o in enumerate(criterion.options) if not o.na]
            idx, _ = criterion.worst_scored_option()
            assert idx == criterion.worst_option_among(non_na)


class TestCriterionNaOptionIndex:
    """Tests for the ``Criterion.na_option_index`` property (first NA index, else None)."""

    def test_returns_first_na_index(self, nominal_criterion_with_na):
        """Returns the index of the (first) NA option."""
        assert nominal_criterion_with_na.na_option_index == 3

    def test_none_when_no_na_option(self, ordinal_criterion):
        """Multi-choice criterion without an NA option → None."""
        assert ordinal_criterion.na_option_index is None

    def test_none_for_binary(self, binary_criterion):
        """Binary criterion (no options) → None (safe, no raise)."""
        assert binary_criterion.na_option_index is None

    def test_returns_first_when_multiple_na(self):
        """With multiple NA options, returns the lowest index."""
        criterion = Criterion(
            name="t",
            weight=10.0,
            requirement="R",
            scale_type="nominal",
            options=[
                {"label": "A", "value": 0.5},
                {"label": "B", "value": 1.0},
                {"label": "N/A one", "value": 0.0, "na": True},
                {"label": "N/A two", "value": 0.0, "na": True},
            ],
        )
        assert criterion.na_option_index == 2


class TestCriterionWithGuaranteedNAOption:
    """Tests for ``Criterion.with_guaranteed_na_option`` — the auto-inject helper (T2-A).

    Appends a single canonical NA option when the criterion lacks one (so the judge
    always has an abstain channel, the analog of binary CANNOT_ASSESS); returns self
    unchanged when an author NA option already exists.
    """

    def test_appends_canonical_na_when_absent(self, ordinal_criterion):
        """No NA option → a canonical NA option is appended at the END."""
        from autorubric.types import CANONICAL_NA_OPTION

        result = ordinal_criterion.with_guaranteed_na_option()

        # One more option, appended at the highest index.
        assert len(result.options) == len(ordinal_criterion.options) + 1
        injected = result.options[-1]
        assert injected.na is True
        assert injected.value == 0.0
        assert injected == CANONICAL_NA_OPTION

    def test_preserves_original_indices(self, ordinal_criterion):
        """Indices 0..N-1 (labels and values) are unchanged by the append."""
        result = ordinal_criterion.with_guaranteed_na_option()
        for i, opt in enumerate(ordinal_criterion.options):
            assert result.options[i].label == opt.label
            assert result.options[i].value == opt.value
            assert result.options[i].na == opt.na

    def test_does_not_mutate_original(self, ordinal_criterion):
        """The author criterion (frozen) is never mutated."""
        original_len = len(ordinal_criterion.options)
        ordinal_criterion.with_guaranteed_na_option()
        assert len(ordinal_criterion.options) == original_len
        assert ordinal_criterion.na_option_index is None

    def test_idempotent_when_author_na_present(self, nominal_criterion_with_na):
        """Author already supplied an NA option → returns self unchanged (no 2nd NA)."""
        result = nominal_criterion_with_na.with_guaranteed_na_option()
        assert result is nominal_criterion_with_na
        na_count = sum(1 for o in result.options if o.na)
        assert na_count == 1

    def test_idempotent_when_author_na_not_last(self):
        """Author NA in a non-final position is respected — no canonical NA appended."""
        criterion = Criterion(
            name="t",
            weight=10.0,
            requirement="R",
            scale_type="nominal",
            options=[
                {"label": "N/A", "value": 0.0, "na": True},  # NA first
                {"label": "A", "value": 0.5},
                {"label": "B", "value": 1.0},
            ],
        )
        result = criterion.with_guaranteed_na_option()
        assert result is criterion
        assert sum(1 for o in result.options if o.na) == 1

    def test_binary_raises(self, binary_criterion):
        """Calling on a binary criterion is a programmer error."""
        with pytest.raises(ValueError, match="Binary criterion"):
            binary_criterion.with_guaranteed_na_option()

    def test_result_passes_validator(self, ordinal_criterion):
        """Appending an NA option keeps ≥2 non-NA options (validator still passes)."""
        result = ordinal_criterion.with_guaranteed_na_option()
        non_na = [o for o in result.options if not o.na]
        assert len(non_na) == len(ordinal_criterion.options)  # non-NA count unchanged
        assert len(non_na) >= 2


# =============================================================================
# Test get_option_value and is_na_option
# =============================================================================


class TestOptionHelpers:
    """Tests for option helper functions."""

    def test_get_option_value(self, ordinal_criterion):
        """Get option value by index."""
        assert get_option_value(ordinal_criterion, 0) == 0.0
        assert get_option_value(ordinal_criterion, 3) == 1.0

    def test_get_option_value_binary_raises(self, binary_criterion):
        """get_option_value raises for binary criteria."""
        with pytest.raises(ValueError, match="Cannot get option value for binary"):
            get_option_value(binary_criterion, 0)

    def test_is_na_option(self, nominal_criterion_with_na):
        """Check if option is NA."""
        assert is_na_option(nominal_criterion_with_na, 3) is True
        assert is_na_option(nominal_criterion_with_na, 0) is False


# =============================================================================
# Test per-criterion metric functions
# =============================================================================


class TestComputePerOptionMetrics:
    """Tests for _compute_per_option_metrics."""

    def test_perfect_predictions(self, ordinal_criterion):
        """Perfect predictions give F1=1 for all options."""
        pred = [0, 1, 2, 3, 0, 1, 2, 3]
        true = [0, 1, 2, 3, 0, 1, 2, 3]
        metrics = _compute_per_option_metrics(pred, true, ordinal_criterion)

        assert len(metrics) == 4
        for m in metrics:
            assert m.f1 == 1.0
            assert m.precision == 1.0
            assert m.recall == 1.0


class TestComputeOrdinalCriterionMetrics:
    """Tests for _compute_ordinal_criterion_metrics."""

    def test_perfect_predictions(self, ordinal_criterion):
        """Perfect predictions give exact accuracy 1.0."""
        pred = [0, 1, 2, 3]
        true = [0, 1, 2, 3]
        metrics = _compute_ordinal_criterion_metrics(pred, true, ordinal_criterion, 0)

        assert metrics.exact_accuracy == 1.0
        assert metrics.adjacent_accuracy == 1.0
        assert metrics.weighted_kappa == 1.0
        assert metrics.rmse == 0.0

    def test_adjacent_accuracy(self, ordinal_criterion):
        """Adjacent accuracy for off-by-one predictions."""
        pred = [1, 2, 3, 2]  # Off by 1 from true
        true = [0, 1, 2, 3]
        metrics = _compute_ordinal_criterion_metrics(pred, true, ordinal_criterion, 0)

        assert metrics.exact_accuracy == 0.0
        assert metrics.adjacent_accuracy == 1.0  # All within ±1

    def test_correlation(self, ordinal_criterion):
        """Ordinal metrics include correlations."""
        pred = [0, 1, 2, 3, 0, 1, 2, 3]
        true = [0, 1, 2, 3, 0, 1, 2, 3]
        metrics = _compute_ordinal_criterion_metrics(pred, true, ordinal_criterion, 0)

        assert metrics.spearman.coefficient == 1.0
        assert metrics.kendall.coefficient == 1.0


class TestComputeNominalCriterionMetrics:
    """Tests for _compute_nominal_criterion_metrics."""

    def test_perfect_predictions(self, nominal_criterion):
        """Perfect predictions give accuracy 1.0."""
        pred = [0, 1, 2, 0, 1, 2]
        true = [0, 1, 2, 0, 1, 2]
        metrics = _compute_nominal_criterion_metrics(pred, true, nominal_criterion, 0)

        assert metrics.exact_accuracy == 1.0
        assert metrics.kappa == 1.0

    def test_confusion_matrix(self, nominal_criterion):
        """Nominal metrics include confusion matrix."""
        pred = [0, 1, 2]
        true = [0, 1, 2]
        metrics = _compute_nominal_criterion_metrics(pred, true, nominal_criterion, 0)

        assert len(metrics.confusion_matrix) == 3
        assert metrics.confusion_matrix[0][0] == 1  # True 0, Pred 0


# =============================================================================
# Test compute_metrics with multi-choice
# =============================================================================


@pytest.fixture
def ordinal_dataset() -> RubricDataset:
    """Create a dataset with ordinal criteria."""
    rubric = Rubric(
        [
            Criterion(
                name="satisfaction",
                weight=10.0,
                requirement="Satisfaction level",
                scale_type="ordinal",
                options=[
                    {"label": "1", "value": 0.0},
                    {"label": "2", "value": 0.33},
                    {"label": "3", "value": 0.67},
                    {"label": "4", "value": 1.0},
                ],
            ),
        ]
    )
    dataset = RubricDataset(prompt="Test", rubric=rubric)
    # Add items with ground truth as string labels
    dataset.add_item(submission="A", description="D1", ground_truth=["4"])
    dataset.add_item(submission="B", description="D2", ground_truth=["3"])
    dataset.add_item(submission="C", description="D3", ground_truth=["2"])
    return dataset


@pytest.fixture
def hybrid_dataset(hybrid_rubric) -> RubricDataset:
    """Create a dataset with mixed criterion types."""
    dataset = RubricDataset(prompt="Test", rubric=hybrid_rubric)
    # Binary: MET, Ordinal: "Very satisfied" (index 3), Nominal: "Just right" (index 2)
    dataset.add_item(
        submission="A",
        description="D1",
        ground_truth=[CriterionVerdict.MET, "Very satisfied", "Just right"],
    )
    dataset.add_item(
        submission="B",
        description="D2",
        ground_truth=[CriterionVerdict.UNMET, "Dissatisfied", "Too brief"],
    )
    return dataset


def _make_ordinal_report(selected_index: int) -> EvaluationReport:
    """Create a mock EvaluationReport for ordinal criterion."""
    from autorubric.types import CriterionReport

    return EvaluationReport(
        score=0.5,
        raw_score=5.0,
        report=[
            CriterionReport(
                weight=10.0,
                requirement="Satisfaction level",
                name="satisfaction",
                verdict=CriterionVerdict.MET if selected_index >= 2 else CriterionVerdict.UNMET,
                reason="Test",
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=selected_index,
                    selected_label=str(selected_index + 1),
                    value=selected_index * 0.33,
                ),
            ),
        ],
    )


def _make_hybrid_report(
    binary_verdict: CriterionVerdict,
    ordinal_index: int,
    nominal_index: int,
) -> EvaluationReport:
    """Create a mock EvaluationReport for hybrid rubric."""
    from autorubric.types import CriterionReport

    return EvaluationReport(
        score=0.5,
        raw_score=10.0,
        report=[
            # Binary criterion
            CriterionReport(
                weight=10.0,
                requirement="Is accurate?",
                name="accuracy",
                verdict=binary_verdict,
                reason="Test",
            ),
            # Ordinal criterion
            CriterionReport(
                weight=10.0,
                requirement="Satisfaction",
                name="satisfaction",
                verdict=CriterionVerdict.MET,
                reason="Test",
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=ordinal_index,
                    selected_label="Test",
                    value=ordinal_index * 0.33,
                ),
            ),
            # Nominal criterion
            CriterionReport(
                weight=5.0,
                requirement="Length",
                name="length",
                verdict=CriterionVerdict.MET,
                reason="Test",
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=nominal_index,
                    selected_label="Test",
                    value=0.5,
                ),
            ),
        ],
    )


class TestComputeMetricsOrdinal:
    """Tests for compute_metrics with ordinal criteria."""

    def test_perfect_ordinal_predictions(self, ordinal_dataset):
        """Perfect ordinal predictions give accuracy 1.0."""
        # Create perfect predictions matching ground truth
        item_results = [
            ItemResult(
                item_idx=0,
                item=ordinal_dataset.items[0],
                report=_make_ordinal_report(3),  # Ground truth is "4" = index 3
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=ordinal_dataset.items[1],
                report=_make_ordinal_report(2),  # Ground truth is "3" = index 2
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=2,
                item=ordinal_dataset.items[2],
                report=_make_ordinal_report(1),  # Ground truth is "2" = index 1
                duration_seconds=0.1,
            ),
        ]

        from datetime import datetime

        eval_result = EvalResult(
            item_results=item_results,
            total_items=3,
            successful_items=3,
            failed_items=0,
            total_token_usage=None,
            total_completion_cost=None,
            timing_stats=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=[],
            experiment_name=None,
            experiment_dir=None,
        )

        metrics = compute_metrics(eval_result, ordinal_dataset)

        assert metrics.n_ordinal_criteria == 1
        assert metrics.n_binary_criteria == 0
        assert metrics.criterion_accuracy == 1.0
        assert len(metrics.per_criterion) == 1
        assert metrics.per_criterion[0].criterion_type == "ordinal"


class TestComputeMetricsHybrid:
    """Tests for compute_metrics with hybrid (mixed) rubrics."""

    def test_hybrid_metrics_has_all_types(self, hybrid_dataset):
        """Hybrid metrics include all criterion types."""
        item_results = [
            ItemResult(
                item_idx=0,
                item=hybrid_dataset.items[0],
                report=_make_hybrid_report(CriterionVerdict.MET, 3, 2),
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=hybrid_dataset.items[1],
                report=_make_hybrid_report(CriterionVerdict.UNMET, 1, 0),
                duration_seconds=0.1,
            ),
        ]

        from datetime import datetime

        eval_result = EvalResult(
            item_results=item_results,
            total_items=2,
            successful_items=2,
            failed_items=0,
            total_token_usage=None,
            total_completion_cost=None,
            timing_stats=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=[],
            experiment_name=None,
            experiment_dir=None,
        )

        metrics = compute_metrics(eval_result, hybrid_dataset)

        assert metrics.n_binary_criteria == 1
        assert metrics.n_ordinal_criteria == 1
        assert metrics.n_nominal_criteria == 1
        assert len(metrics.per_criterion) == 3

        # Check criterion types
        types = [cm.criterion_type for cm in metrics.per_criterion]
        assert "binary" in types
        assert "ordinal" in types
        assert "nominal" in types


class TestMetricsSummaryMultiChoice:
    """Tests for summary() method with multi-choice criteria."""

    def test_summary_shows_type_breakdown(self, hybrid_dataset):
        """Summary shows criterion type breakdown."""
        item_results = [
            ItemResult(
                item_idx=0,
                item=hybrid_dataset.items[0],
                report=_make_hybrid_report(CriterionVerdict.MET, 3, 2),
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=hybrid_dataset.items[1],
                report=_make_hybrid_report(CriterionVerdict.UNMET, 1, 0),
                duration_seconds=0.1,
            ),
        ]

        from datetime import datetime

        eval_result = EvalResult(
            item_results=item_results,
            total_items=2,
            successful_items=2,
            failed_items=0,
            total_token_usage=None,
            total_completion_cost=None,
            timing_stats=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=[],
            experiment_name=None,
            experiment_dir=None,
        )

        metrics = compute_metrics(eval_result, hybrid_dataset)
        summary = metrics.summary()

        # Check that summary contains type info
        assert "binary" in summary.lower()
        assert "ordinal" in summary.lower()
        assert "nominal" in summary.lower()


class TestBackwardsCompatibility:
    """Tests that binary-only rubrics work unchanged."""

    def test_binary_only_unchanged(self):
        """Binary-only rubrics produce identical results to before."""
        rubric = Rubric(
            [
                Criterion(name="C1", weight=10.0, requirement="R1"),
                Criterion(name="C2", weight=5.0, requirement="R2"),
            ]
        )
        dataset = RubricDataset(prompt="Test", rubric=rubric)
        dataset.add_item(
            submission="A",
            description="D1",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
        )
        dataset.add_item(
            submission="B",
            description="D2",
            ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.MET],
        )

        # Create matching predictions
        item_results = [
            ItemResult(
                item_idx=0,
                item=dataset.items[0],
                report=EvaluationReport(
                    score=1.0,
                    raw_score=15.0,
                    report=[
                        CriterionReport(
                            weight=10.0,
                            requirement="R1",
                            name="C1",
                            verdict=CriterionVerdict.MET,
                            reason="Test",
                        ),
                        CriterionReport(
                            weight=5.0,
                            requirement="R2",
                            name="C2",
                            verdict=CriterionVerdict.MET,
                            reason="Test",
                        ),
                    ],
                ),
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=dataset.items[1],
                report=EvaluationReport(
                    score=0.33,
                    raw_score=5.0,
                    report=[
                        CriterionReport(
                            weight=10.0,
                            requirement="R1",
                            name="C1",
                            verdict=CriterionVerdict.UNMET,
                            reason="Test",
                        ),
                        CriterionReport(
                            weight=5.0,
                            requirement="R2",
                            name="C2",
                            verdict=CriterionVerdict.MET,
                            reason="Test",
                        ),
                    ],
                ),
                duration_seconds=0.1,
            ),
        ]

        from datetime import datetime

        eval_result = EvalResult(
            item_results=item_results,
            total_items=2,
            successful_items=2,
            failed_items=0,
            total_token_usage=None,
            total_completion_cost=None,
            timing_stats=None,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            errors=[],
            experiment_name=None,
            experiment_dir=None,
        )

        metrics = compute_metrics(eval_result, dataset)

        # Check backwards compatibility
        assert metrics.n_binary_criteria == 2
        assert metrics.n_ordinal_criteria == 0
        assert metrics.n_nominal_criteria == 0
        assert metrics.criterion_accuracy == 1.0
        assert metrics.criterion_precision > 0
        assert metrics.criterion_recall > 0
        assert metrics.criterion_f1 > 0

        # All per_criterion should be binary
        for cm in metrics.per_criterion:
            assert cm.criterion_type == "binary"


# =============================================================================
# NA Kappa tests (T1-E): NAStats.na_kappa = Cohen's kappa on {NA, not-NA}
# =============================================================================


def _make_na_dataset(nominal_criterion_with_na, ground_truth_labels: list[str]) -> RubricDataset:
    """Build a dataset of N items using the supplied criterion and per-item GT label."""
    rubric = Rubric([nominal_criterion_with_na])
    dataset = RubricDataset(prompt="Test", rubric=rubric)
    for i, label in enumerate(ground_truth_labels):
        dataset.add_item(
            submission=f"S{i}",
            description=f"D{i}",
            ground_truth=[label],
        )
    return dataset


def _make_na_report(selected_index: int, criterion: Criterion) -> EvaluationReport:
    """Build a single-criterion EvaluationReport whose pick is ``selected_index``."""
    option = criterion.options[selected_index]
    return EvaluationReport(
        score=float(option.value),
        raw_score=float(option.value) * criterion.weight,
        report=[
            CriterionReport(
                weight=criterion.weight,
                requirement=criterion.requirement,
                name=criterion.name,
                verdict=(CriterionVerdict.MET if option.value >= 0.5 else CriterionVerdict.UNMET),
                reason="Test",
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=selected_index,
                    selected_label=option.label,
                    value=float(option.value),
                ),
            ),
        ],
    )


def _wrap_eval_result(item_results: list[ItemResult]) -> EvalResult:
    """Wrap ItemResults into an EvalResult with the metadata compute_metrics expects."""
    from datetime import datetime

    return EvalResult(
        item_results=item_results,
        total_items=len(item_results),
        successful_items=len(item_results),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=None,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        errors=[],
        experiment_name=None,
        experiment_dir=None,
    )


class TestNaKappa:
    """Cohen's kappa on the dichotomized {NA, not-NA} decision (T1-E)."""

    def test_na_kappa_perfect_agreement_is_one(self, nominal_criterion_with_na):
        """4 items, perfect NA-vs-not-NA agreement => kappa=1.0, interpretation 'almost perfect'.

        items 1,2: both pred and GT pick N/A (index 3).
        items 3,4: both pred and GT pick "Very specific" (index 2).
        A=2, fp=0, fn=0, N=2. P_o=1.0, P_e=0.5, kappa=1.0.
        """
        labels = ["N/A", "N/A", "Very specific", "Very specific"]
        preds = [3, 3, 2, 2]
        dataset = _make_na_dataset(nominal_criterion_with_na, labels)
        item_results = [
            ItemResult(
                item_idx=i,
                item=dataset.items[i],
                report=_make_na_report(preds[i], nominal_criterion_with_na),
                duration_seconds=0.1,
            )
            for i in range(len(labels))
        ]
        eval_result = _wrap_eval_result(item_results)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.na_stats is not None
        assert metrics.na_stats.na_kappa == pytest.approx(1.0)
        assert metrics.na_stats.na_kappa_interpretation == "almost perfect"

    def test_na_kappa_no_na_observed_is_none(self, nominal_criterion_with_na):
        """3 items, no NA in either pred or GT => kappa is undefined (single class) => None."""
        labels = ["Very specific", "Very specific", "Very specific"]
        preds = [2, 2, 2]
        dataset = _make_na_dataset(nominal_criterion_with_na, labels)
        item_results = [
            ItemResult(
                item_idx=i,
                item=dataset.items[i],
                report=_make_na_report(preds[i], nominal_criterion_with_na),
                duration_seconds=0.1,
            )
            for i in range(len(labels))
        ]
        eval_result = _wrap_eval_result(item_results)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.na_stats is not None
        assert metrics.na_stats.na_kappa is None
        assert metrics.na_stats.na_kappa_interpretation is None

    def test_na_kappa_binary_only_rubric_leaves_na_stats_none(self):
        """Regression guard: binary-only rubrics produce na_stats=None."""
        rubric = Rubric([Criterion(name="C1", weight=10.0, requirement="R1")])
        dataset = RubricDataset(prompt="Test", rubric=rubric)
        dataset.add_item(submission="A", description="D1", ground_truth=[CriterionVerdict.MET])
        dataset.add_item(submission="B", description="D2", ground_truth=[CriterionVerdict.UNMET])

        item_results = [
            ItemResult(
                item_idx=0,
                item=dataset.items[0],
                report=EvaluationReport(
                    score=1.0,
                    raw_score=10.0,
                    report=[
                        CriterionReport(
                            weight=10.0,
                            requirement="R1",
                            name="C1",
                            verdict=CriterionVerdict.MET,
                            reason="Test",
                        ),
                    ],
                ),
                duration_seconds=0.1,
            ),
            ItemResult(
                item_idx=1,
                item=dataset.items[1],
                report=EvaluationReport(
                    score=0.0,
                    raw_score=0.0,
                    report=[
                        CriterionReport(
                            weight=10.0,
                            requirement="R1",
                            name="C1",
                            verdict=CriterionVerdict.UNMET,
                            reason="Test",
                        ),
                    ],
                ),
                duration_seconds=0.1,
            ),
        ]
        eval_result = _wrap_eval_result(item_results)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.na_stats is None

    def test_na_kappa_disagreement_below_perfect(self, nominal_criterion_with_na):
        """6 items, 2x2 mix gives na_kappa = 1/3.

        Hand-built layout (pred, true):
            (NA, NA), (NA, NA)         -> A=2
            (NA, not-NA)               -> fp=1
            (not-NA, NA)               -> fn=1
            (not-NA, not-NA), (not-NA, not-NA) -> N=2
        P_o=(2+2)/6=4/6; pred_NA=A+fp=3, true_NA=A+fn=3; P_e=(3/6)^2*2=0.5;
        kappa=(4/6 - 0.5)/(1 - 0.5) = (1/6)/(1/2) = 1/3.
        """
        # Layout below; index 3 is NA, index 2 is "Very specific" (not-NA).
        preds = [3, 3, 3, 2, 2, 2]
        true_labels = ["N/A", "N/A", "Very specific", "N/A", "Very specific", "Very specific"]
        dataset = _make_na_dataset(nominal_criterion_with_na, true_labels)
        item_results = [
            ItemResult(
                item_idx=i,
                item=dataset.items[i],
                report=_make_na_report(preds[i], nominal_criterion_with_na),
                duration_seconds=0.1,
            )
            for i in range(len(true_labels))
        ]
        eval_result = _wrap_eval_result(item_results)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.na_stats is not None
        assert metrics.na_stats.na_kappa == pytest.approx(1 / 3, abs=1e-9)

    def test_na_counts_preserved(self, nominal_criterion_with_na):
        """Regression guard: na_count_* and na_false_* are still populated correctly."""
        preds = [3, 3, 3, 2, 2, 2]
        true_labels = ["N/A", "N/A", "Very specific", "N/A", "Very specific", "Very specific"]
        dataset = _make_na_dataset(nominal_criterion_with_na, true_labels)
        item_results = [
            ItemResult(
                item_idx=i,
                item=dataset.items[i],
                report=_make_na_report(preds[i], nominal_criterion_with_na),
                duration_seconds=0.1,
            )
            for i in range(len(true_labels))
        ]
        eval_result = _wrap_eval_result(item_results)

        metrics = compute_metrics(eval_result, dataset)

        assert metrics.na_stats is not None
        assert metrics.na_stats.na_count_true == 3
        assert metrics.na_stats.na_count_pred == 3
        assert metrics.na_stats.na_false_positive == 1
        assert metrics.na_stats.na_false_negative == 1


def _make_effective_report(selected_index: int, effective_criterion: Criterion) -> EvaluationReport:
    """A single-criterion report whose criterion carries the EFFECTIVE options.

    Models what the grader produces with ``auto_na_option`` on: the report's criterion
    has the injected NA appended at the end, and the verdict may point at it (an index
    out of range for the author rubric).
    """
    option = effective_criterion.options[selected_index]
    return EvaluationReport(
        score=0.0 if option.na else float(option.value),
        raw_score=0.0,
        report=[
            CriterionReport(
                weight=effective_criterion.weight,
                requirement=effective_criterion.requirement,
                name=effective_criterion.name,
                options=effective_criterion.options,
                scale_type=effective_criterion.scale_type,
                verdict=None,
                reason="Test",
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=selected_index,
                    selected_label=option.label,
                    value=float(option.value),
                    na=option.na,
                ),
            ),
        ],
    )


class TestMetricsAutoInjectedNA:
    """compute_metrics must interpret predicted auto-injected NA indices (T2-A).

    The grader appends the NA option at index ``N = len(author.options)`` (out of range
    for the author rubric used by the metrics layer). Metrics must reconstruct the same
    effective criterion so the prediction is recognized as NA rather than crashing.
    """

    def test_handles_predicted_injected_na(self, ordinal_criterion):
        """A predicted index == N (injected NA) is recognized as NA, not a crash."""
        effective = ordinal_criterion.with_guaranteed_na_option()
        na_index = len(ordinal_criterion.options)  # injected NA appended at the end

        dataset = RubricDataset(prompt="Test", rubric=Rubric([ordinal_criterion]))
        # Two abstain (predicted NA); three scored pairs (with variation, so kappa is
        # well-defined and no degenerate-confusion warning is emitted).
        gt_labels = [
            "Very satisfied",
            "Dissatisfied",
            "Satisfied",
            "Very satisfied",
            "Very dissatisfied",
        ]
        preds = [na_index, na_index, 2, 3, 0]
        for i, label in enumerate(gt_labels):
            dataset.add_item(submission=f"S{i}", description=f"D{i}", ground_truth=[label])

        item_results = [
            ItemResult(
                item_idx=i,
                item=dataset.items[i],
                report=_make_effective_report(preds[i], effective),
                duration_seconds=0.1,
            )
            for i in range(len(gt_labels))
        ]
        eval_result = _wrap_eval_result(item_results)

        metrics = compute_metrics(eval_result, dataset)  # must not raise

        assert metrics.na_stats is not None
        # Two predictions abstained (NA); ground truth never NA.
        assert metrics.na_stats.na_count_pred == 2
        assert metrics.na_stats.na_count_true == 0
        assert metrics.na_stats.na_false_positive == 2

    def test_as_category_refused_for_autoinjected_ordinal(self, ordinal_criterion):
        """na_mode='as_category' is refused once an ordinal criterion gains an NA option."""
        effective = ordinal_criterion.with_guaranteed_na_option()
        na_index = len(ordinal_criterion.options)

        dataset = RubricDataset(prompt="Test", rubric=Rubric([ordinal_criterion]))
        gt_labels = ["Very satisfied", "Dissatisfied"]
        preds = [na_index, 2]
        for i, label in enumerate(gt_labels):
            dataset.add_item(submission=f"S{i}", description=f"D{i}", ground_truth=[label])

        item_results = [
            ItemResult(
                item_idx=i,
                item=dataset.items[i],
                report=_make_effective_report(preds[i], effective),
                duration_seconds=0.1,
            )
            for i in range(len(gt_labels))
        ]
        eval_result = _wrap_eval_result(item_results)

        with pytest.raises(ValueError, match="ordinal|as_category"):
            compute_metrics(eval_result, dataset, na_mode="as_category")
