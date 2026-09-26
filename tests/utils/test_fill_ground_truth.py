"""Tests for fill_ground_truth utility function."""

from unittest.mock import MagicMock, patch

import litellm
import pytest
from litellm import ModelResponse

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    LLMConfig,
    Rubric,
)
from autorubric.dataset import DataItem, RubricDataset
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    CriterionReport,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    EvaluationReport,
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
                final_multi_choice_verdict=AggregatedMultiChoiceVerdict(
                    selected_index=mc_option_idx,
                    selected_label=multi_choice_label,
                    value=mc_value,
                    aggregated_value=mc_value,
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
        items=[DataItem(submission=f"Response {i}", description=f"Item {i}") for i in range(5)],
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


@pytest.mark.asyncio
async def test_fill_ground_truth_queries_with_per_item_prompt(binary_rubric, binary_criteria):
    """Each item is graded against its own prompt, falling back to the dataset prompt."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(submission="Response 1", description="Item 1", prompt="Summarize the paper"),
            DataItem(submission="Response 2", description="Item 2"),
        ],
        name="test",
    )

    mock_grader = MagicMock()
    queries: dict[str, str] = {}

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        queries[to_grade] = query
        return create_mock_ensemble_binary_report(
            [CriterionVerdict.MET, CriterionVerdict.MET], binary_criteria
        )

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        await fill_ground_truth(dataset, mock_grader, show_progress=False)

    assert queries == {
        "Response 1": "Summarize the paper",
        "Response 2": "Evaluate the response",
    }


@pytest.mark.asyncio
async def test_fill_ground_truth_preserves_graded_item_fields(binary_criteria):
    """A graded item keeps every per-item field; only ground_truth is filled in."""
    item_rubric = Rubric(binary_criteria)
    dataset = RubricDataset(
        prompt="Evaluate the response",
        items=[
            DataItem(
                submission="Response 1",
                description="Item 1",
                rubric=item_rubric,
                reference_submission="Reference answer",
                prompt="Summarize the paper",
            ),
        ],
        name="test",
    )

    mock_grader = MagicMock()

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        return create_mock_ensemble_binary_report(
            [CriterionVerdict.MET, CriterionVerdict.UNMET], binary_criteria
        )

    with patch.object(item_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    assert result.items == [
        DataItem(
            submission="Response 1",
            description="Item 1",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
            rubric=item_rubric,
            reference_submission="Reference answer",
            prompt="Summarize the paper",
        )
    ]


@pytest.mark.asyncio
async def test_fill_ground_truth_dataset_with_only_per_item_prompts(binary_rubric, binary_criteria):
    """A dataset with no global prompt (every item has its own) can be labeled."""
    dataset = RubricDataset(
        rubric=binary_rubric,
        items=[
            DataItem(submission="Response 1", description="Item 1", prompt="Prompt A"),
            DataItem(submission="Response 2", description="Item 2", prompt="Prompt B"),
        ],
        name="test",
    )

    mock_grader = MagicMock()
    queries: dict[str, str] = {}

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        queries[to_grade] = query
        return create_mock_ensemble_binary_report(
            [CriterionVerdict.MET, CriterionVerdict.MET], binary_criteria
        )

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    assert queries == {"Response 1": "Prompt A", "Response 2": "Prompt B"}
    assert [item.prompt for item in result.items] == ["Prompt A", "Prompt B"]
    assert [item.ground_truth for item in result.items] == [
        [CriterionVerdict.MET, CriterionVerdict.MET],
        [CriterionVerdict.MET, CriterionVerdict.MET],
    ]


@pytest.mark.asyncio
async def test_fill_ground_truth_force_drops_reasons_of_regraded_items(
    binary_rubric, binary_criteria
):
    """force=True replaces the ground truth, so the reasons written for the old one go."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(
                submission="Response 1",
                description="Item 1",
                ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
                ground_truth_reasons=["States the right date.", "Reads clearly."],
            ),
        ],
        name="test",
    )

    mock_grader = MagicMock()
    new_gt = [CriterionVerdict.UNMET, CriterionVerdict.UNMET]

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        return create_mock_ensemble_binary_report(new_gt, binary_criteria)

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, force=True, show_progress=False)

    assert result.items[0].ground_truth == new_gt
    assert result.items[0].ground_truth_reasons is None


@pytest.mark.asyncio
async def test_fill_ground_truth_keeps_reasons_of_items_it_does_not_grade(
    binary_rubric, binary_criteria
):
    """Without force, an item that has ground truth is kept as it is, reasons included."""
    dataset = RubricDataset(
        prompt="Evaluate the response",
        rubric=binary_rubric,
        items=[
            DataItem(
                submission="Response 1",
                description="Item 1",
                ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
                ground_truth_reasons=["States the right date.", None],
            ),
            DataItem(submission="Response 2", description="Item 2"),
        ],
        name="test",
    )

    mock_grader = MagicMock()
    new_gt = [CriterionVerdict.UNMET, CriterionVerdict.MET]

    async def mock_grade(to_grade, grader, query, reference_submission=None):
        return create_mock_ensemble_binary_report(new_gt, binary_criteria)

    with patch.object(binary_rubric, "grade", side_effect=mock_grade):
        result = await fill_ground_truth(dataset, mock_grader, show_progress=False)

    assert result.items[0].ground_truth == [CriterionVerdict.MET, CriterionVerdict.UNMET]
    assert result.items[0].ground_truth_reasons == ["States the right date.", None]
    assert result.items[1].ground_truth == new_gt


# =============================================================================
# Failed judge calls are never saved as labels (#22)
# =============================================================================

MET_JSON = '{"criterion_status": "MET", "explanation": "ok"}'


def _llm_response(content: str) -> ModelResponse:
    return ModelResponse(
        model="gpt-4.1-mini",
        choices=[
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _failure(category: str, model: str) -> BaseException | None:
    """An exception a judge call fails with in ``category``; None for a parse failure, which
    is a response that does not parse."""
    if category == "infrastructure":
        return litellm.APIConnectionError(
            message="Connection error.", llm_provider="openai", model=model
        )
    if category == "unknown":
        return RuntimeError("boom")
    return None


def _one_item(rubric: Rubric) -> RubricDataset:
    return RubricDataset(
        prompt="Evaluate the response",
        rubric=rubric,
        items=[DataItem(submission="Response 1", description="Item 1")],
        name="test",
    )


def _llm_grader(**kwargs) -> CriterionGrader:
    if "judges" not in kwargs:
        kwargs["judge_model_config"] = LLMConfig(model="openai/gpt-4.1-mini", max_retries=1)
    return CriterionGrader(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["infrastructure", "parse", "unknown"])
async def test_verdicts_standing_in_for_failed_judge_calls_are_not_labels(mixed_rubric, category):
    """For a failed call the grader stands in a verdict (CANNOT_ASSESS or the NA option;
    the worst case for an unknown failure) so that scoring can go on. It is no judgment, so
    the item is left out, like any item that fails to label."""

    async def acompletion(**params):
        error = _failure(category, params["model"])
        if error is not None:
            raise error
        return _llm_response("not json")

    grader = _llm_grader()
    with patch("litellm.acompletion", side_effect=acompletion):
        report = await mixed_rubric.grade(
            to_grade="Response 1", grader=grader, query="Evaluate the response"
        )
        labeled = await fill_ground_truth(_one_item(mixed_rubric), grader, show_progress=False)

    assert isinstance(report, EnsembleEvaluationReport) and report.report is not None
    assert [(cr.error or "").split(":")[0] for cr in report.report] == [category, category]
    assert len(labeled) == 0


@pytest.mark.asyncio
async def test_an_item_with_one_failed_criterion_is_left_out(mixed_rubric):
    """A label covers every criterion, so one failed judge call leaves the whole item out."""

    async def acompletion(**params):
        if "<options>" in str(params["messages"]):  # the multi-choice criterion
            raise RuntimeError("boom")
        return _llm_response(MET_JSON)

    with patch("litellm.acompletion", side_effect=acompletion):
        labeled = await fill_ground_truth(
            _one_item(mixed_rubric), _llm_grader(), show_progress=False
        )

    assert len(labeled) == 0


@pytest.mark.asyncio
async def test_a_panel_with_a_genuine_vote_still_labels_the_item(binary_rubric):
    """A criterion's report records an error only when every vote failed: a judge that
    answered decides the label, as it decides the grade."""

    async def acompletion(**params):
        if params["model"].endswith("gpt-4.1"):  # the judge that is down
            raise litellm.APIConnectionError(
                message="Connection error.", llm_provider="openai", model=params["model"]
            )
        return _llm_response(MET_JSON)

    grader = _llm_grader(
        judges=[
            JudgeSpec(LLMConfig(model="openai/gpt-4.1", max_retries=1), "down"),
            JudgeSpec(LLMConfig(model="openai/gpt-4.1-mini", max_retries=1), "up"),
        ]
    )
    with patch("litellm.acompletion", side_effect=acompletion):
        labeled = await fill_ground_truth(_one_item(binary_rubric), grader, show_progress=False)

    assert [item.ground_truth for item in labeled] == [[CriterionVerdict.MET, CriterionVerdict.MET]]
