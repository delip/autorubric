"""Held-out validation of multi-choice criteria.

``validate_held_out`` reads each item's ground truth as ``resolve_ground_truth`` does: a
binary label is a verdict, a multi-choice label an option of the criterion as a judge is
shown it (the NA option the grader adds included). A multi-choice judgment agrees with its
label when the judge chose that option. NA, on either side, is the multi-choice analog of
``CANNOT_ASSESS`` and is handled by the same ``cannot_assess`` mode: dropped (``exclude``),
read as the worst-scored option (``as_unmet``), or kept as its own category
(``as_category``, refused upfront for an ordinal criterion on which NA can occur, which is
stricter than ``compute_metrics``: it refuses only once NA is observed).

The judge is mocked by patching ``Rubric.grade``, as in ``test_held_out_diagnostics.py``, so
no LLM is ever called.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sklearn.metrics import cohen_kappa_score

from autorubric import Criterion, CriterionOption, CriterionVerdict, Rubric, TokenUsage
from autorubric.dataset import DataItem, RubricDataset
from autorubric.graders import CriterionGrader, Grader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.meta._display import render_improvement_report_html
from autorubric.meta._improve import (
    ImprovementConfig,
    ImprovementRunner,
    IterationResult,
    format_held_out_for_prompt,
    revise_rubric_held_out,
    validate_held_out,
)
from autorubric.prompts import (
    HELD_OUT_REVISION_SCALES_BLOCK,
    HELD_OUT_REVISION_USER_PROMPT_TEMPLATE,
)
from autorubric.types import (
    CANONICAL_NA_OPTION,
    AggregatedMultiChoiceVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
)

NA = CANONICAL_NA_OPTION.label
MET, UNMET = CriterionVerdict.MET, CriterionVerdict.UNMET

INTRO = Criterion(name="intro", weight=1.0, requirement="Has an introduction")
TONE = Criterion(
    name="tone",
    weight=1.0,
    requirement="What is the essay's tone?",
    scale_type="nominal",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Casual", value=0.0),
    ],
)
CLARITY = Criterion(
    name="clarity",
    weight=1.0,
    requirement="How clear is the essay?",
    scale_type="ordinal",
    options=[
        CriterionOption(label="Unclear", value=0.0),
        CriterionOption(label="Somewhat clear", value=0.5),
        CriterionOption(label="Clear", value=1.0),
    ],
)


def _rubric(*criteria: Criterion) -> Rubric:
    return Rubric(list(criteria))


def _dataset(rubric: Rubric, ground_truths: list[list]) -> RubricDataset:
    items = [
        DataItem(submission=f"submission {i}", ground_truth=gt, description=f"item {i}")
        for i, gt in enumerate(ground_truths)
    ]
    return RubricDataset(rubric=rubric, items=items, prompt="Write an essay")


def _criterion_report(criterion: Criterion, verdict, error: str | None = None):
    """A criterion's report: a binary verdict, or the index of the option the judge chose
    among the options it was shown (``with_guaranteed_na_option``)."""
    if criterion.is_binary:
        return EnsembleCriterionReport(
            criterion=criterion,
            final_verdict=verdict,
            final_reason="reason",
            votes=[JudgeVote(judge_id="j", verdict=verdict, reason="reason", error=error)],
            error=error,
        )
    options = criterion.with_guaranteed_na_option().options
    assert options is not None
    option = options[verdict]
    return EnsembleCriterionReport(
        criterion=criterion,
        final_verdict=None,
        final_reason=f"chose {option.label}",
        votes=[],
        final_multi_choice_verdict=AggregatedMultiChoiceVerdict(
            selected_index=verdict,
            selected_label=option.label,
            value=option.value,
            aggregated_value=option.value,
            na=option.na,
        ),
        multi_choice_votes=[
            MultiChoiceJudgeVote(
                judge_id="j",
                selected_index=verdict,
                selected_label=option.label,
                value=option.value,
                reason=f"chose {option.label}",
                na=option.na,
                error=error,
            )
        ],
        error=error,
    )


def _report(rubric: Rubric, verdicts: list, errors: dict[int, str] | None = None):
    errors = errors or {}
    return EnsembleEvaluationReport(
        score=0.5,
        raw_score=0.5,
        report=[
            _criterion_report(c, v, errors.get(i))
            for i, (c, v) in enumerate(zip(rubric.rubric, verdicts, strict=True))
        ],
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        completion_cost=0.01,
    )


async def _validate(rubric, ground_truths, judged, grader=None, **kwargs):
    ds = _dataset(rubric, ground_truths)
    reports = [_report(rubric, v) for v in judged]
    with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
        return await validate_held_out(
            rubric, ds, grader if grader is not None else MagicMock(), task_prompt="t", **kwargs
        )


class TestOptionAgreement:
    """A multi-choice judgment agrees with its label when the judge chose that option."""

    GROUND_TRUTH = [
        [MET, "Formal", "Clear"],
        [UNMET, "Casual", "Somewhat clear"],
        [MET, "Formal", "Unclear"],
    ]
    # Option indices: tone Formal=0 Casual=1; clarity Unclear=0 Somewhat=1 Clear=2.
    JUDGED = [[MET, 0, 2], [UNMET, 0, 1], [MET, 1, 2]]

    @pytest.mark.asyncio
    async def test_labels_are_compared_by_option(self) -> None:
        rubric = _rubric(INTRO, TONE, CLARITY)
        result = await _validate(rubric, self.GROUND_TRUTH, self.JUDGED)

        intro, tone, clarity = result.per_criterion
        assert (intro.n_samples, intro.accuracy) == (3, 1.0)
        assert (tone.n_samples, tone.accuracy) == (3, pytest.approx(1 / 3))
        assert (clarity.n_samples, clarity.accuracy) == (3, pytest.approx(2 / 3))
        assert result.mean_accuracy == pytest.approx((1.0 + 1 / 3 + 2 / 3) / 3)
        for cr in (tone, clarity):
            assert (cr.coverage, cr.ca_rate) == (1.0, 0.0)

    @pytest.mark.asyncio
    async def test_kappa_is_the_scale_s_cohen_kappa(self) -> None:
        """Nominal: unweighted Cohen's kappa; ordinal: quadratic-weighted, as in
        ``compute_metrics``."""
        rubric = _rubric(INTRO, TONE, CLARITY)
        result = await _validate(rubric, self.GROUND_TRUTH, self.JUDGED)

        _, tone, clarity = result.per_criterion
        assert tone.kappa == pytest.approx(cohen_kappa_score([0, 1, 0], [0, 0, 1]))
        assert clarity.kappa == pytest.approx(
            cohen_kappa_score([2, 1, 0], [2, 1, 2], weights="quadratic")
        )

    @pytest.mark.asyncio
    async def test_confusion_matrix_is_over_the_options(self) -> None:
        """Rows are labels, columns judgments, over the criterion's options; there is no
        positive class, so the false-positive and false-negative rates are undefined."""
        rubric = _rubric(INTRO, TONE, CLARITY)
        result = await _validate(rubric, self.GROUND_TRUTH, self.JUDGED)

        _, tone, clarity = result.per_criterion
        assert tone.confusion_matrix is not None
        assert tone.confusion_matrix.labels == ["Formal", "Casual"]
        assert tone.confusion_matrix.matrix == [[1, 1], [1, 0]]
        assert clarity.confusion_matrix is not None
        assert clarity.confusion_matrix.labels == ["Unclear", "Somewhat clear", "Clear"]
        assert clarity.confusion_matrix.matrix == [[0, 0, 1], [0, 1, 0], [0, 0, 1]]
        for cr in (tone, clarity):
            assert cr.false_positive_rate is None
            assert cr.false_negative_rate is None

    @pytest.mark.asyncio
    async def test_exemplars_name_the_options(self) -> None:
        rubric = _rubric(INTRO, TONE, CLARITY)
        result = await _validate(rubric, self.GROUND_TRUTH, self.JUDGED)

        tone = result.per_criterion[1]
        assert [
            (e.item_index, e.llm_verdict, e.ground_truth_verdict, e.llm_reason)
            for e in tone.disagreement_exemplars
        ] == [(1, "Formal", "Casual", "chose Formal"), (2, "Casual", "Formal", "chose Casual")]
        assert [(e.llm_verdict, e.ground_truth_verdict) for e in tone.agreement_exemplars] == [
            ("Formal", "Formal")
        ]

    @pytest.mark.asyncio
    async def test_the_revision_prompt_shows_the_options(self) -> None:
        rubric = _rubric(INTRO, TONE, CLARITY)
        result = await _validate(rubric, self.GROUND_TRUTH, self.JUDGED)

        text = format_held_out_for_prompt(result)
        assert "Judge verdict: Formal, Ground truth: Casual" in text
        assert "Judge verdict: Clear (correct)" in text
        assert f"{'tone':<30} {'33%':>10} {'n/a':>10} {'n/a':>10}" in text
        assert "(Accuracy: 33%, FP: n/a, FN: n/a)" in text

    @pytest.mark.asyncio
    async def test_the_html_report_renders(self) -> None:
        rubric = _rubric(INTRO, TONE, CLARITY)
        result = await _validate(rubric, self.GROUND_TRUTH, self.JUDGED)
        iteration = IterationResult(
            iteration=0,
            rubric=rubric,
            quality_score=result.mean_accuracy,
            agreement=None,
            per_criterion_agreement=None,
            issues=[],
            issues_fixed=[],
            issues_introduced=[],
            accepted=True,
            rejection_reason=None,
            quality_report=None,
            token_usage=None,
            completion_cost=None,
            held_out_diagnostics=result,
        )

        html = render_improvement_report_html(
            [iteration],
            convergence_reason="max_iterations",
            total_cost=0.0,
            original_rubric=rubric,
            final_rubric=rubric,
        )
        row = html[html.index("<td class='crit-name'>tone</td>") :].split("</tr>")[0]
        # Precision, the FP/FN rates and the 2x2 cells are binary only.
        assert row.count("<td>n/a</td>") == 7

    @pytest.mark.asyncio
    async def test_option_labels_named_met_and_unmet_get_no_binary_cells(self) -> None:
        """Whether the 2x2 cells apply is read from the criterion's kind, not from its
        option labels."""
        lookalike = Criterion(
            name="lookalike",
            weight=1.0,
            requirement="Which mark?",
            scale_type="nominal",
            options=[
                CriterionOption(label="MET", value=1.0),
                CriterionOption(label="UNMET", value=0.0),
            ],
        )
        rubric = _rubric(lookalike)
        result = await _validate(rubric, [["MET"], ["UNMET"]], [[0], [0]])
        iteration = IterationResult(
            iteration=0,
            rubric=rubric,
            quality_score=result.mean_accuracy,
            agreement=None,
            per_criterion_agreement=None,
            issues=[],
            issues_fixed=[],
            issues_introduced=[],
            accepted=True,
            rejection_reason=None,
            quality_report=None,
            token_usage=None,
            completion_cost=None,
            held_out_diagnostics=result,
        )
        html = render_improvement_report_html(
            [iteration],
            convergence_reason="max_iterations",
            total_cost=0.0,
            original_rubric=rubric,
            final_rubric=rubric,
        )
        row = html[html.index("<td class='crit-name'>lookalike</td>") :].split("</tr>")[0]
        assert row.count("<td>n/a</td>") == 7


class TestNA:
    """NA, on either side, is handled as binary CANNOT_ASSESS is."""

    # tone: the added NA option is index 2 (Formal=0, Casual=1).
    GROUND_TRUTH = [[NA], ["Formal"], ["Casual"], [NA]]
    JUDGED = [[0], [2], [1], [2]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mode", "n_samples", "accuracy", "coverage"),
        [
            # Only the item where neither side abstained is compared.
            ("exclude", 1, 1.0, 0.25),
            # NA reads as the worst-scored option, Casual: item 0 Casual vs Formal,
            # item 1 Formal vs Casual, items 2 and 3 agree.
            ("as_unmet", 4, 0.5, 1.0),
            # NA is its own category, so item 3 (NA, NA) agrees.
            ("as_category", 4, 0.5, 1.0),
        ],
    )
    async def test_na_follows_the_cannot_assess_mode(
        self, mode, n_samples, accuracy, coverage
    ) -> None:
        rubric = _rubric(TONE)
        result = await _validate(rubric, self.GROUND_TRUTH, self.JUDGED, cannot_assess=mode)

        (tone,) = result.per_criterion
        assert (tone.n_samples, tone.accuracy, tone.coverage) == (n_samples, accuracy, coverage)
        # An abstention on either side, in any mode, as for CANNOT_ASSESS.
        assert tone.ca_rate == 0.75

    @pytest.mark.asyncio
    async def test_the_added_na_option_is_a_confusion_matrix_label(self) -> None:
        rubric = _rubric(TONE)
        result = await _validate(
            rubric, self.GROUND_TRUTH, self.JUDGED, cannot_assess="as_category"
        )

        (tone,) = result.per_criterion
        assert tone.confusion_matrix is not None
        assert tone.confusion_matrix.labels == ["Formal", "Casual", NA]
        assert tone.confusion_matrix.matrix == [[0, 0, 1], [0, 1, 0], [1, 0, 1]]
        assert [(e.llm_verdict, e.ground_truth_verdict) for e in tone.agreement_exemplars] == [
            ("Casual", "Casual"),
            (NA, NA),
        ]

    @pytest.mark.asyncio
    async def test_without_na_the_options_are_the_criterion_s(self) -> None:
        """As in ``compute_metrics``, the added option joins the matrix only when a judgment
        or a label names it."""
        rubric = _rubric(TONE)
        result = await _validate(rubric, [["Formal"], ["Casual"]], [[0], [0]])

        (tone,) = result.per_criterion
        assert tone.confusion_matrix is not None
        assert tone.confusion_matrix.labels == ["Formal", "Casual"]

    @pytest.mark.asyncio
    async def test_an_own_na_option_is_an_abstention(self) -> None:
        own_na = Criterion(
            name="tone",
            requirement="What is the essay's tone?",
            scale_type="nominal",
            options=[
                CriterionOption(label="Formal", value=1.0),
                CriterionOption(label="Casual", value=0.0),
                CriterionOption(label="Not an essay", value=0.0, na=True),
            ],
        )
        rubric = _rubric(own_na)
        result = await _validate(rubric, [["Not an essay"], ["Formal"]], [[0], [0]])

        (tone,) = result.per_criterion
        assert (tone.n_samples, tone.coverage, tone.ca_rate) == (1, 0.5, 0.5)


class TestOrdinalAsCategory:
    """NA has no position on an ordinal scale, so ``as_category`` is refused for an ordinal
    criterion on which NA can occur, before any item is graded."""

    @pytest.mark.asyncio
    async def test_refused_when_the_judge_is_shown_an_na_option(self) -> None:
        rubric = _rubric(INTRO, CLARITY)
        ds = _dataset(rubric, [[MET, "Clear"]])
        grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/test"))
        with patch.object(rubric, "grade", new_callable=AsyncMock) as grade:
            with pytest.raises(ValueError, match="clarity.*ordinal"):
                await validate_held_out(rubric, ds, grader, cannot_assess="as_category")
        grade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refused_when_a_label_names_the_na_option(self) -> None:
        rubric = _rubric(CLARITY)
        ds = _dataset(rubric, [[NA]])
        grader = CriterionGrader(
            judge_model_config=LLMConfig(model="openai/test"), auto_na_option=False
        )
        with patch.object(rubric, "grade", new_callable=AsyncMock) as grade:
            with pytest.raises(ValueError, match="ordinal"):
                await validate_held_out(rubric, ds, grader, cannot_assess="as_category")
        grade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refused_for_a_grader_of_another_kind(self) -> None:
        """A grader that is not a ``CriterionGrader`` may show an NA option: refused, as
        under the default ``auto_na_option``."""

        class OtherGrader(Grader[EnsembleEvaluationReport]):
            async def judge(self, *args, **kwargs):
                raise AssertionError("never called")

            async def aggregate(self, *args, **kwargs):
                raise AssertionError("never called")

        rubric = _rubric(INTRO, CLARITY)
        ds = _dataset(rubric, [[MET, "Clear"]])
        with patch.object(rubric, "grade", new_callable=AsyncMock) as grade:
            with pytest.raises(ValueError, match="clarity.*ordinal"):
                await validate_held_out(rubric, ds, OtherGrader(), cannot_assess="as_category")
        grade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_accepted_when_na_cannot_occur(self) -> None:
        """A forced-choice judge, and labels that name scored options: nothing abstains."""
        rubric = _rubric(CLARITY)
        grader = CriterionGrader(
            judge_model_config=LLMConfig(model="openai/test"), auto_na_option=False
        )
        result = await _validate(
            rubric, [["Clear"], ["Unclear"]], [[2], [1]], grader, cannot_assess="as_category"
        )
        assert result.per_criterion[0].accuracy == 0.5

    @pytest.mark.asyncio
    async def test_nominal_criteria_keep_na_as_a_category(self) -> None:
        rubric = _rubric(TONE)
        grader = CriterionGrader(judge_model_config=LLMConfig(model="openai/test"))
        result = await _validate(rubric, [[NA]], [[2]], grader, cannot_assess="as_category")
        assert result.per_criterion[0].accuracy == 1.0


class TestLabels:
    @pytest.mark.asyncio
    async def test_an_unknown_label_is_refused_before_any_grading(self) -> None:
        rubric = _rubric(INTRO, TONE)
        ds = _dataset(rubric, [[MET, "Formal"], [MET, "Sarcastic"]])
        with patch.object(rubric, "grade", new_callable=AsyncMock) as grade:
            with pytest.raises(ValueError, match="Sarcastic"):
                await validate_held_out(rubric, ds, MagicMock())
        grade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_item_without_ground_truth_is_refused_before_any_grading(self) -> None:
        rubric = _rubric(INTRO)
        ds = RubricDataset(
            rubric=rubric,
            items=[
                DataItem(submission="a", description="a", ground_truth=[MET]),
                DataItem(submission="b", description="b"),
            ],
            prompt="Write an essay",
        )
        with patch.object(rubric, "grade", new_callable=AsyncMock) as grade:
            with pytest.raises(ValueError, match="item 1 has no ground truth"):
                await validate_held_out(rubric, ds, MagicMock())
        grade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_option_labels_resolve_as_find_option_by_label_does(self) -> None:
        rubric = _rubric(TONE)
        result = await _validate(rubric, [["  formal "]], [[0]])
        assert result.per_criterion[0].accuracy == 1.0


class TestFailedCells:
    @pytest.mark.asyncio
    async def test_a_failed_multi_choice_judgment_counts_only_in_the_denominator(self) -> None:
        """Its stand-in (the NA option) is no abstention and no judgment, as for binary."""
        rubric = _rubric(TONE)
        ds = _dataset(rubric, [["Formal"], ["Formal"]])
        reports = [
            _report(rubric, [0]),
            _report(rubric, [2], errors={0: "infrastructure: 429"}),
        ]
        with patch.object(rubric, "grade", new_callable=AsyncMock, side_effect=reports):
            result = await validate_held_out(rubric, ds, MagicMock())

        (tone,) = result.per_criterion
        assert (tone.n_samples, tone.accuracy, tone.coverage, tone.ca_rate) == (1, 1.0, 0.5, 0.0)
        assert not tone.disagreement_exemplars


class TestRevisionKeepsTheOptions:
    """The revision LLM may change wording and weights only. The options are what the
    ground truth's labels name, so each revised criterion keeps its scale unchanged."""

    @staticmethod
    async def _revise(rubric: Rubric, revised: list[dict]) -> Rubric:
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"), revision_llm=LLMConfig(model="test")
        )
        generate = AsyncMock(return_value=GenerateResult(content=json.dumps(revised), cost=0.0))
        with patch("autorubric.meta._improve.LLMClient") as client:
            client.return_value.generate = generate
            result, _ = await revise_rubric_held_out(rubric, "task", "diag", "history", config)
        return result

    @staticmethod
    async def _prompt(rubric: Rubric) -> str:
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"), revision_llm=LLMConfig(model="test")
        )
        echoed = json.dumps(
            [{"weight": c.weight, "requirement": c.requirement} for c in rubric.rubric]
        )
        generate = AsyncMock(return_value=GenerateResult(content=echoed, cost=0.0))
        capture: dict = {}
        with patch("autorubric.meta._improve.LLMClient") as client:
            client.return_value.generate = generate
            await revise_rubric_held_out(
                rubric, "task", "diag", "history", config, _capture=capture
            )
        return capture["user_prompt"]

    @pytest.mark.asyncio
    async def test_the_prompt_shows_each_multi_choice_scale(self) -> None:
        """The LLM sees a multi-choice criterion's options as fixed, so the wording it
        writes stays a question those options answer."""
        prompt = await self._prompt(_rubric(INTRO, TONE))
        assert prompt.startswith(HELD_OUT_REVISION_SCALES_BLOCK)
        shown = json.loads(prompt.split("## Current Rubric\n", 1)[1].split("\n\n## Held-Out", 1)[0])
        assert shown == [
            {"weight": 1.0, "requirement": "Has an introduction", "name": "intro"},
            {
                "weight": 1.0,
                "requirement": "What is the essay's tone?",
                "name": "tone",
                "scale_type": "nominal",
                "options": [
                    {"label": "Formal", "value": 1.0, "na": False},
                    {"label": "Casual", "value": 0.0, "na": False},
                ],
            },
        ]

    @pytest.mark.asyncio
    async def test_a_binary_rubric_s_prompt_is_unchanged(self) -> None:
        rubric = _rubric(INTRO)
        expected = HELD_OUT_REVISION_USER_PROMPT_TEMPLATE.format(
            task_prompt="task",
            original_criteria=json.dumps(
                [{"weight": 1.0, "requirement": "Has an introduction", "name": "intro"}],
                indent=2,
            ),
            diagnostics_text="diag",
            history_text="history",
            num_criteria=1,
        )
        assert await self._prompt(rubric) == expected

    @pytest.mark.asyncio
    async def test_criteria_written_without_options_keep_theirs(self) -> None:
        rubric = _rubric(INTRO, TONE)
        revised = await self._revise(
            rubric,
            [
                {"name": "intro", "weight": 2, "requirement": "Opens with an introduction"},
                {"name": "tone", "weight": 3, "requirement": "Which tone does the essay take?"},
            ],
        )
        intro, tone = revised.rubric
        assert (intro.requirement, intro.weight, intro.options) == (
            "Opens with an introduction",
            2,
            None,
        )
        assert (tone.requirement, tone.weight) == ("Which tone does the essay take?", 3)
        assert (tone.options, tone.scale_type, tone.aggregation) == (
            TONE.options,
            TONE.scale_type,
            TONE.aggregation,
        )

    @pytest.mark.asyncio
    async def test_options_the_llm_writes_are_not_taken(self) -> None:
        rubric = _rubric(INTRO, TONE)
        revised = await self._revise(
            rubric,
            [
                {
                    "name": "intro",
                    "requirement": "Has an introduction",
                    "options": [{"label": "Yes", "value": 1.0}, {"label": "No", "value": 0.0}],
                },
                {
                    "name": "tone",
                    "requirement": "Tone?",
                    "options": [
                        {"label": "Polite", "value": 1.0},
                        {"label": "Rude", "value": 0.0},
                    ],
                },
            ],
        )
        intro, tone = revised.rubric
        assert intro.is_binary
        assert tone.options == TONE.options


class TestTheLoop:
    @pytest.mark.asyncio
    async def test_a_multi_choice_rubric_is_revised_and_regraded(self, tmp_path: Path) -> None:
        """Two iterations: the revised rubric keeps its options, so the labels resolve again,
        and the artifacts record the options."""
        rubric = _rubric(INTRO, TONE)
        ds = _dataset(rubric, [[MET, "Formal"], [UNMET, "Casual"]])
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),
            strategy="held_out",
            validation_data=ds,
            max_iterations=2,
            held_out_min_accuracy=1.0,
            save_artifacts=True,
            artifacts_dir=str(tmp_path / "artifacts"),
            show_progress=False,
            display=None,
        )
        reports = [
            _report(rubric, [MET, 0]),
            _report(rubric, [UNMET, 0]),
            _report(rubric, [MET, 0]),
            _report(rubric, [UNMET, 1]),
        ]
        revised_json = json.dumps(
            [
                {"name": "intro", "weight": 1, "requirement": "Has an introduction"},
                {"name": "tone", "weight": 1, "requirement": "Which tone does it take?"},
            ]
        )
        generate = AsyncMock(return_value=GenerateResult(content=revised_json, cost=0.0))
        with (
            patch("autorubric.meta._improve.CriterionGrader", return_value=MagicMock()),
            patch.object(Rubric, "grade", new_callable=AsyncMock, side_effect=reports),
            patch("autorubric.meta._improve.LLMClient") as client,
        ):
            client.return_value.generate = generate
            result = await ImprovementRunner(rubric, "Write an essay", config=config).run()

        assert [it.quality_score for it in result.iterations] == [0.75, 1.0]
        assert result.iterations[1].rubric.rubric[1].options == TONE.options

        saved = json.loads((tmp_path / "artifacts" / "rubric-iter-01.json").read_text())
        assert Rubric.from_dict(saved).rubric[1].options == TONE.options
        iteration = json.loads((tmp_path / "artifacts" / "iter-00.json").read_text())
        assert iteration["rubric_criteria"][1]["options"] == [
            {"label": "Formal", "value": 1.0, "na": False},
            {"label": "Casual", "value": 0.0, "na": False},
        ]
