"""Tests for fill_ground_truth utility function."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    Rubric,
)
from autorubric.dataset import DataItem, RubricDataset
from autorubric.types import (
    CriterionReport,
    EnsembleCriterionReport,
    EvaluationReport,
    EnsembleEvaluationReport,
    MultiChoiceVerdict,
)
from autorubric.utils import fill_ground_truth


@pytest.fixture
def binary_criteria() -> list[Criterion]:
    return [
        Criterion(weight=1.0, requirement="Is factually accurate", name="accuracy"),
        Criterion(weight=1.0, requirement="Is well written", name="writing"),
    ]


@pytest.fixture
def binary_rubric(binary_criteria) -> Rubric:
    return Rubric(binary_criteria)


@pytest.fixture
def mixed_criteria() -> list[Criterion]:
    return [
        Criterion(weight=1.0, requirement="Is factually accurate", name="accuracy"),
        Criterion(
            weight=1.0,
            requirement="Quality rating",
            name="quality",
            options=[
                CriterionOption(label="Poor", value=0.0),
                CriterionOption(label="Fair", value=0.5),
                CriterionOption(label="Good", value=1.0),
            ],
        ),
    ]


@pytest.fixture
def mixed_rubric(mixed_criteria) -> Rubric:
    return Rubric(mixed_criteria)


def create_mock_binary_report(verdicts: list[CriterionVerdict]) -> EvaluationReport:
    """Create a mock EvaluationReport with binary verdicts."""
    return EvaluationReport(
        score=0.5,
        raw_score=1.0,
        report=[
            CriterionReport(
                weight=1.0,
                requirement=f"Criterion {i}",
                verdict=v,
                reason="Test reason",
            )
            for i, v in enumerate(verdicts)
        ],
    )


def create_mock_ensemble_binary_report(
    verdicts: list[CriterionVerdict], criteria: list[Criterion]
) -> EnsembleEvaluationReport:
    """Create a mock EnsembleEvaluationReport with binary verdicts."""
    return EnsembleEvaluationReport(
        score=0.5,
        raw_score=1.0,
        report=[
            EnsembleCriterionReport(
                criterion=c,
                final_verdict=v,
                final_reason="Test reason",
                votes=[],
                agreement=1.0,
            )
            for c, v in zip(criteria, verdicts)
        ],
        judge_scores={"judge1": 0.5},
        mean_agreement=1.0,
    )


def create_mock_mixed_report(
    binary_verdict: CriterionVerdict, multi_choice_label: str, criteria: list[Criterion]
) -> EnsembleEvaluationReport:
    """Create a mock report with both binary and multi-choice criteria."""
    mc_criterion = criteria[1]
    mc_option_idx = next(
        i for i, opt in enumerate(mc_criterion.options) if opt.label == multi_choice_label
    )
    mc_value = mc_criterion.options[mc_option_idx].value

    return EnsembleEvaluationReport(
        score=0.5,
        raw_score=1.0,
        report=[
            EnsembleCriterionReport(
                criterion=criteria[0],
                final_verdict=binary_verdict,
                final_reason="Test reason",
                votes=[],
                agreement=1.0,
            ),
            EnsembleCriterionReport(
                criterion=criteria[1],
                final_verdict=None,
                final_reason="Test reason",
                votes=[],
                agreement=1.0,
                final_multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=mc_option_idx,
                    selected_label=multi_choice_label,
                    value=mc_value,
                ),
            ),
        ],
        judge_scores={"judge1": 0.5},
        mean_agreement=1.0,
    )


@pytest.mark.asyncio
async def test_fill_ground_truth_basic(binary_rubric, binary_criteria):
    """Test basic fill_ground_truth with items missing ground_truth."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(submission="Response 1", description="Item 1"),
            DataItem(submission="Response 2", description="Item 2"),
        ],
        name="test",
    )

    mock_grader = MagicMock()

    verdicts_per_item = [
        [CriterionVerdict.MET, CriterionVerdict.UNMET],
        [CriterionVerdict.UNMET, CriterionVerdict.MET],
    ]
    call_count = [0]

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        idx = call_count[0]
        call_count[0] += 1
        return create_mock_ensemble_binary_report(verdicts_per_item[idx], binary_criteria)

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    assert len(result) == 2
    assert result.items[0].ground_truth == [CriterionVerdict.MET, CriterionVerdict.UNMET]
    assert result.items[1].ground_truth == [CriterionVerdict.UNMET, CriterionVerdict.MET]


@pytest.mark.asyncio
async def test_fill_ground_truth_preserves_existing(binary_rubric, binary_criteria):
    """Test that items with existing ground_truth are preserved."""
    existing_gt = [CriterionVerdict.MET, CriterionVerdict.MET]
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(submission="Response 1", description="Item 1", ground_truth=existing_gt),
            DataItem(submission="Response 2", description="Item 2"),
        ],
        name="test",
    )

    mock_grader = MagicMock()
    new_gt = [CriterionVerdict.UNMET, CriterionVerdict.UNMET]

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        return create_mock_ensemble_binary_report(new_gt, binary_criteria)

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    assert len(result) == 2
    # First item should keep its original ground_truth
    assert result.items[0].ground_truth == existing_gt
    # Second item should have new ground_truth
    assert result.items[1].ground_truth == new_gt


@pytest.mark.asyncio
async def test_fill_ground_truth_force_mode(binary_rubric, binary_criteria):
    """Test that force=True re-grades all items."""
    existing_gt = [CriterionVerdict.MET, CriterionVerdict.MET]
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(submission="Response 1", description="Item 1", ground_truth=existing_gt),
        ],
        name="test",
    )

    mock_grader = MagicMock()
    new_gt = [CriterionVerdict.UNMET, CriterionVerdict.UNMET]

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        return create_mock_ensemble_binary_report(new_gt, binary_criteria)

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, force=True, show_progress=False)

    assert len(result) == 1
    # Should be overwritten with new ground_truth
    assert result.items[0].ground_truth == new_gt


@pytest.mark.asyncio
async def test_fill_ground_truth_excludes_failed_items(binary_rubric, binary_criteria):
    """Test that items that fail to grade are excluded from result."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(submission="Response 1", description="Item 1"),
            DataItem(submission="Response 2", description="Item 2"),
        ],
        name="test",
    )

    mock_grader = MagicMock()
    call_count = [0]

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            raise Exception("Grading failed")
        return create_mock_ensemble_binary_report(
            [CriterionVerdict.MET, CriterionVerdict.MET], binary_criteria
        )

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    # Only the successful item should be in the result
    assert len(result) == 1
    assert result.items[0].submission == "Response 2"
    assert result.items[0].ground_truth == [CriterionVerdict.MET, CriterionVerdict.MET]


@pytest.mark.asyncio
async def test_fill_ground_truth_empty_dataset(binary_rubric):
    """Test that empty dataset raises ValueError."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[],
        name="test",
    )

    mock_grader = MagicMock()

    with pytest.raises(ValueError, match="Dataset has no items"):
        await fill_ground_truth(dataset, mock_grader, show_progress=False)


@pytest.mark.asyncio
async def test_fill_ground_truth_mixed_criteria(mixed_rubric, mixed_criteria):
    """Test fill_ground_truth with both binary and multi-choice criteria."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=mixed_rubric,
        items=[
            DataItem(submission="Response 1", description="Item 1"),
        ],
        name="test",
    )

    mock_grader = MagicMock()

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        return create_mock_mixed_report(CriterionVerdict.MET, "Good", mixed_criteria)

    with patch.object(mixed_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    assert len(result) == 1
    # Binary criterion should have CriterionVerdict
    assert result.items[0].ground_truth[0] == CriterionVerdict.MET
    # Multi-choice criterion should have string label
    assert result.items[0].ground_truth[1] == "Good"


@pytest.mark.asyncio
async def test_fill_ground_truth_maintains_order(binary_rubric, binary_criteria):
    """Test that items maintain their original order."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(submission="Response A", description="First"),
            DataItem(
                submission="Response B",
                description="Second",
                ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
            ),
            DataItem(submission="Response C", description="Third"),
        ],
        name="test",
    )

    mock_grader = MagicMock()
    graded_texts = []

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        graded_texts.append(to_grade)
        return create_mock_ensemble_binary_report(
            [CriterionVerdict.UNMET, CriterionVerdict.UNMET], binary_criteria
        )

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    assert len(result) == 3
    # Order should be maintained
    assert result.items[0].submission == "Response A"
    assert result.items[1].submission == "Response B"
    assert result.items[2].submission == "Response C"
    # Only items 0 and 2 should have been graded
    assert set(graded_texts) == {"Response A", "Response C"}


@pytest.mark.asyncio
async def test_fill_ground_truth_with_concurrency_limit(binary_rubric, binary_criteria):
    """Test that max_concurrent_items limits concurrency."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(submission=f"Response {i}", description=f"Item {i}") for i in range(5)
        ],
        name="test",
    )

    mock_grader = MagicMock()

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        return create_mock_ensemble_binary_report(
            [CriterionVerdict.MET, CriterionVerdict.MET], binary_criteria
        )

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(
            dataset, mock_grader, max_concurrent_items=2, show_progress=False
        )

    # All items should be successfully graded
    assert len(result) == 5
    for item in result.items:
        assert item.ground_truth == [CriterionVerdict.MET, CriterionVerdict.MET]


@pytest.mark.asyncio
async def test_fill_ground_truth_returns_new_dataset(binary_rubric, binary_criteria):
    """Test that fill_ground_truth returns a new dataset, not modifying the original."""
    original_item = DataItem(submission="Response 1", description="Item 1")
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[original_item],
        name="test",
    )

    mock_grader = MagicMock()

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        return create_mock_ensemble_binary_report(
            [CriterionVerdict.MET, CriterionVerdict.MET], binary_criteria
        )

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    # Original item should be unchanged
    assert original_item.ground_truth is None
    # Result should have ground_truth
    assert result.items[0].ground_truth is not None
    # Should be different objects
    assert result is not dataset
    assert result.items[0] is not original_item
