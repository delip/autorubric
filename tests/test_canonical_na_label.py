"""The label of the NA option the grader adds is a ground-truth label (#38).

With ``auto_na_option=True`` (the default) a multi-choice criterion that has no NA option of
its own is shown to the judge with ``CANONICAL_NA_OPTION`` appended
(``Criterion.with_guaranteed_na_option``). When the judge chooses it, ``fill_ground_truth``
records that option's label. Every place that resolves a multi-choice ground-truth label
resolves it against the same effective criterion, so the label names the appended option,
index ``len(criterion.options)``:

- ``RubricDataset.from_json`` / ``from_file`` accept it;
- ``resolve_ground_truth`` resolves it (and that index) to the appended option;
- ``Rubric.compute_score`` / ``RubricDataset.compute_weighted_score`` score it as an NA
  abstention;
- ``compute_metrics`` (per criterion and pooled) reads it as an NA label, exactly as it reads
  an NA option of the criterion's own;
- per-criterion few-shot selection shows it as the appended option, and a grader with
  ``auto_na_option=False``, which shows no NA option, leaves such an item out;
- the cascade tools read it as an abstaining label.

The label is legal only where ``with_guaranteed_na_option`` would add the option: a criterion
with an NA option of its own gets nothing appended, so there the label must name one of its
options. Nothing here reaches the network: ``litellm.acompletion`` is patched.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest
from litellm import ModelResponse

from autorubric import (
    CannotAssessStrategy,
    Criterion,
    CriterionOption,
    CriterionVerdict,
    FewShotConfig,
    LLMConfig,
    Rubric,
    RubricDataset,
    fill_ground_truth,
)
from autorubric.escalation import _labels
from autorubric.eval import EvalResult, EvalTimingStats, ItemResult
from autorubric.graders import CriterionGrader
from autorubric.metrics import OrdinalCriterionMetrics, resolve_ground_truth
from autorubric.types import (
    CANONICAL_NA_OPTION,
    CriterionReport,
    EvaluationReport,
    MultiChoiceVerdict,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
NA_LABEL = CANONICAL_NA_OPTION.label

# No NA option of its own: the grader appends one, at index 2.
CLARITY_OPTIONS = [
    CriterionOption(label="Unclear", value=0.0),
    CriterionOption(label="Clear", value=1.0),
]
CLARITY = Criterion(
    name="clarity",
    weight=1.0,
    requirement="How clear is the answer?",
    options=CLARITY_OPTIONS,
)
# The same criterion with that NA option written out by the author.
CLARITY_WITH_NA = CLARITY.model_copy(update={"options": [*CLARITY_OPTIONS, CANONICAL_NA_OPTION]})
# An NA option of its own, under another label: nothing is appended.
CLARITY_OWN_NA = CLARITY.model_copy(
    update={"options": [*CLARITY_OPTIONS, CriterionOption(label="N/A", value=0.0, na=True)]}
)
LIGHT = Criterion(name="light", weight=1.0, requirement="Mentions light")


def _choice_response(selected_option: int) -> ModelResponse:
    content = json.dumps({"selected_option": selected_option, "explanation": "Missing PDF."})
    return ModelResponse(
        model="gpt-4.1-mini",
        choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    )


def _grader(**kwargs) -> CriterionGrader:
    return CriterionGrader(
        judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"), shuffle_options=False, **kwargs
    )


async def _labelled_by_the_na_option() -> RubricDataset:
    """The issue's reproduction: the judge picks option 3, the NA option the grader adds."""
    dataset = RubricDataset(prompt="Explain photosynthesis.", rubric=Rubric([CLARITY]))
    dataset.add_item(submission="[see attached PDF]", description="unanswerable")

    async def acompletion(**params):
        return _choice_response(3)

    with patch("litellm.acompletion", side_effect=acompletion):
        return await fill_ground_truth(dataset, _grader(), show_progress=False)


def _eval_result(reports: list[EvaluationReport], dataset: RubricDataset) -> EvalResult:
    return EvalResult(
        item_results=[
            ItemResult(item_idx=i, item=dataset.items[i], report=r, duration_seconds=0.1)
            for i, r in enumerate(reports)
        ],
        total_items=len(reports),
        successful_items=len(reports),
        failed_items=0,
        total_token_usage=None,
        total_completion_cost=None,
        timing_stats=EvalTimingStats(
            total_duration_seconds=1.0,
            mean_item_duration_seconds=0.1,
            min_item_duration_seconds=0.1,
            max_item_duration_seconds=0.1,
            p50_item_duration_seconds=0.1,
            p95_item_duration_seconds=0.1,
            items_per_second=1.0,
        ),
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
    )


def _report(criteria: list[Criterion], picks: list[int | CriterionVerdict]) -> EvaluationReport:
    """A judge's report choosing ``picks`` (an option index or a verdict per criterion)."""
    reports = []
    for criterion, pick in zip(criteria, picks, strict=True):
        if isinstance(pick, CriterionVerdict):
            reports.append(
                CriterionReport(
                    requirement=criterion.requirement,
                    name=criterion.name,
                    weight=criterion.weight,
                    verdict=pick,
                    reason="r",
                )
            )
            continue
        shown = criterion.with_guaranteed_na_option()
        option = shown.options[pick]  # ty: ignore[not-subscriptable]
        reports.append(
            CriterionReport(
                requirement=criterion.requirement,
                name=criterion.name,
                weight=criterion.weight,
                options=shown.options,
                scale_type=criterion.scale_type,
                multi_choice_verdict=MultiChoiceVerdict(
                    selected_index=pick,
                    selected_label=option.label,
                    value=option.value,
                    na=option.na,
                ),
                reason="r",
            )
        )
    rubric = Rubric(criteria)
    score = rubric.compute_score(
        [
            p.value
            if isinstance(p, CriterionVerdict)
            else c.with_guaranteed_na_option()
            .options[p]  # ty: ignore[not-subscriptable]
            .label
            for c, p in zip(criteria, picks, strict=True)
        ]
    )
    return EvaluationReport(score=score, raw_score=score, report=reports)


# =============================================================================
# The issue: the dataset fill_ground_truth labels round-trips
# =============================================================================


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_the_recorded_label_survives_to_json_and_from_json(self):
        labelled = await _labelled_by_the_na_option()
        assert labelled.items[0].ground_truth == [NA_LABEL]

        reloaded = RubricDataset.from_json(labelled.to_json())

        assert reloaded.items[0].ground_truth == [NA_LABEL]
        # The rubric is still the author's: no option was added to it.
        assert reloaded.rubric is not None
        assert reloaded.rubric.rubric[0].options == CLARITY.options

    @pytest.mark.asyncio
    async def test_and_from_file(self, tmp_path):
        labelled = await _labelled_by_the_na_option()
        path = tmp_path / "labelled.json"
        labelled.to_file(path)
        assert RubricDataset.from_file(path).items[0].ground_truth == [NA_LABEL]

    @pytest.mark.asyncio
    async def test_the_reloaded_label_is_an_na_label_in_the_metrics(self):
        labelled = await _labelled_by_the_na_option()
        dataset = RubricDataset.from_json(labelled.to_json())
        dataset.add_item(
            submission="Plants use light.", description="clear", ground_truth=["Clear"]
        )
        # The judge chose "Clear" for both: the first is a false negative on NA.
        result = _eval_result([_report([CLARITY], [1]), _report([CLARITY], [1])], dataset)

        metrics = result.compute_metrics(dataset)

        assert not [w for w in metrics.warnings if "not found" in w]
        assert metrics.na_stats is not None
        assert metrics.na_stats.na_count_true == 1
        assert metrics.na_stats.na_false_negative == 1
        # Under the default na_mode="exclude" the NA pair leaves the agreement sample.
        (clarity,) = metrics.per_criterion
        assert isinstance(clarity, OrdinalCriterionMetrics)
        assert clarity.n_samples == 1
        assert clarity.exact_accuracy == 1.0

    @pytest.mark.asyncio
    async def test_the_reloaded_item_is_a_few_shot_example(self):
        labelled = await _labelled_by_the_na_option()
        training = RubricDataset.from_json(labelled.to_json())
        grader = _grader(training_data=training, few_shot_config=FewShotConfig(n_examples=2))
        assert grader._multi_choice_examples[(0, "default")] == [("[see attached PDF]", 2, None)]


# =============================================================================
# Where the label is legal
# =============================================================================


def _dataset_json(criterion: Criterion, label: str) -> str:
    dataset = RubricDataset(prompt="p", rubric=Rubric([criterion]))
    dataset.add_item(submission="s", description="d", ground_truth=[label])
    return dataset.to_json()


class TestLoading:
    @pytest.mark.parametrize("label", [NA_LABEL, "  cannot ASSESS / not applicable "])
    def test_a_criterion_without_an_na_option_takes_the_label(self, label):
        assert RubricDataset.from_json(_dataset_json(CLARITY, label)).items[0].ground_truth == [
            label
        ]

    def test_a_criterion_with_an_na_option_of_its_own_does_not(self):
        with pytest.raises(ValueError, match="Label 'Cannot assess / not applicable' not found"):
            RubricDataset.from_json(_dataset_json(CLARITY_OWN_NA, NA_LABEL))

    def test_a_criterion_whose_own_na_option_has_that_label_does(self):
        loaded = RubricDataset.from_json(_dataset_json(CLARITY_WITH_NA, NA_LABEL))
        assert loaded.items[0].ground_truth == [NA_LABEL]

    def test_a_per_item_rubric_takes_the_label_too(self):
        dataset = RubricDataset(prompt="p")
        dataset.add_item(
            submission="s", description="d", rubric=Rubric([CLARITY]), ground_truth=[NA_LABEL]
        )
        assert RubricDataset.from_json(dataset.to_json()).items[0].ground_truth == [NA_LABEL]

    def test_a_binary_criterion_does_not(self):
        data = json.loads(_dataset_json(CLARITY, NA_LABEL))
        data["rubric"] = [{"name": "light", "weight": 1.0, "requirement": "Mentions light"}]
        with pytest.raises(ValueError, match="invalid verdict"):
            RubricDataset.from_json(json.dumps(data))

    def test_other_unknown_labels_are_still_refused(self):
        with pytest.raises(ValueError, match="Label 'Crystal clear' not found"):
            RubricDataset.from_json(_dataset_json(CLARITY, "Crystal clear"))


class TestResolveGroundTruth:
    def test_the_label_resolves_to_the_appended_option(self):
        assert resolve_ground_truth([NA_LABEL, MET], [CLARITY, LIGHT]) == [2, MET]

    def test_so_does_its_index(self):
        assert resolve_ground_truth([2], [CLARITY]) == [2]  # ty: ignore[invalid-argument-type]

    def test_one_past_the_appended_option_is_out_of_range(self):
        with pytest.raises(ValueError, match=r"Option index 3 out of range \[0, 3\)"):
            resolve_ground_truth([3], [CLARITY])  # ty: ignore[invalid-argument-type]

    def test_a_criterion_with_an_na_option_of_its_own_gets_none_appended(self):
        with pytest.raises(ValueError, match="not found"):
            resolve_ground_truth([NA_LABEL], [CLARITY_OWN_NA])
        with pytest.raises(ValueError, match=r"Option index 3 out of range \[0, 3\)"):
            resolve_ground_truth([3], [CLARITY_OWN_NA])  # ty: ignore[invalid-argument-type]

    def test_it_resolves_as_the_author_written_option(self):
        assert resolve_ground_truth([NA_LABEL], [CLARITY]) == resolve_ground_truth(
            [NA_LABEL], [CLARITY_WITH_NA]
        )


class TestScoring:
    def test_the_label_scores_as_an_na_abstention(self):
        rubric = Rubric([CLARITY, LIGHT])
        # SKIP (the default) leaves the abstaining criterion out of the score.
        assert rubric.compute_score([NA_LABEL, MET]) == 1.0
        assert rubric.compute_score(
            [NA_LABEL, MET], cannot_assess_strategy=CannotAssessStrategy.ZERO
        ) == pytest.approx(0.5)
        assert Rubric([CLARITY]).compute_score([NA_LABEL]) is None

    def test_as_the_author_written_option_scores(self):
        for strategy in CannotAssessStrategy:
            assert Rubric([CLARITY, LIGHT]).compute_score(
                [NA_LABEL, UNMET], cannot_assess_strategy=strategy
            ) == Rubric([CLARITY_WITH_NA, LIGHT]).compute_score(
                [NA_LABEL, UNMET], cannot_assess_strategy=strategy
            )

    def test_the_dataset_scores_it_too(self):
        dataset = RubricDataset(prompt="p", rubric=Rubric([CLARITY, LIGHT]))
        assert dataset.compute_weighted_score([NA_LABEL, MET]) == 1.0

    def test_a_criterion_with_an_na_option_of_its_own_refuses_it(self):
        with pytest.raises(ValueError, match="not found"):
            Rubric([CLARITY_OWN_NA]).compute_score([NA_LABEL])


# =============================================================================
# Metrics: the label is read as an NA option of the criterion's own is read
# =============================================================================


def _metrics_dump(criterion: Criterion, **kwargs) -> dict:
    """compute_metrics over three items, one labelled NA, with ``criterion`` as clarity."""
    criteria = [criterion, LIGHT]
    dataset = RubricDataset(prompt="p", rubric=Rubric(criteria))
    for submission, labels in [
        ("a", [NA_LABEL, MET]),
        ("b", ["Clear", MET]),
        ("c", ["Unclear", UNMET]),
        ("d", ["Clear", UNMET]),
    ]:
        dataset.add_item(submission=submission, description=submission, ground_truth=labels)
    reports = [
        _report(criteria, [1, MET]),
        _report(criteria, [1, MET]),
        _report(criteria, [0, UNMET]),
        _report(criteria, [2, MET]),
    ]
    metrics = _eval_result(reports, dataset).compute_metrics(dataset, **kwargs)
    return metrics.model_dump()


class TestMetrics:
    @pytest.mark.parametrize("na_mode", ["exclude", "as_unmet"])
    def test_per_criterion_metrics_equal_the_author_written_option(self, na_mode):
        assert _metrics_dump(CLARITY, na_mode=na_mode) == _metrics_dump(
            CLARITY_WITH_NA, na_mode=na_mode
        )

    def test_the_label_is_counted_as_an_na_label(self):
        dump = _metrics_dump(CLARITY)
        assert not [w for w in dump["warnings"] if "not found" in w]
        assert dump["na_stats"]["na_count_true"] == 1
        assert dump["na_stats"]["na_false_negative"] == 1
        assert dump["na_stats"]["na_false_positive"] == 1

    def test_the_true_score_is_that_of_the_na_label(self):
        """The first item's true score is 1.0 (light alone), never light with a stand-in
        option such as "Unclear"."""
        criteria = [CLARITY, LIGHT]
        dataset = RubricDataset(prompt="p", rubric=Rubric(criteria))
        dataset.add_item(submission="a", description="a", ground_truth=[NA_LABEL, MET])
        dataset.add_item(submission="b", description="b", ground_truth=["Unclear", UNMET])
        # The predicted scores are the true scores: 1.0 (light alone) and 0.0.
        reports = [_report(criteria, [2, MET]), _report(criteria, [0, UNMET])]
        assert [r.score for r in reports] == [1.0, 0.0]
        metrics = _eval_result(reports, dataset).compute_metrics(dataset)
        assert metrics.score_rmse == pytest.approx(0.0)

    def test_the_pooled_path_reads_it_as_the_author_written_option(self):
        def pooled(criterion: Criterion) -> dict:
            dataset = RubricDataset(prompt="p")
            dataset.add_item(
                submission="a",
                description="a",
                rubric=Rubric([criterion]),
                ground_truth=[NA_LABEL],
            )
            dataset.add_item(
                submission="b", description="b", rubric=Rubric([LIGHT]), ground_truth=[MET]
            )
            reports = [_report([criterion], [1]), _report([LIGHT], [MET])]
            metrics = _eval_result(reports, dataset).compute_metrics(dataset)
            assert metrics.pooled_by_scale is not None
            return metrics.model_dump()

        assert pooled(CLARITY) == pooled(CLARITY_WITH_NA)


# =============================================================================
# Few-shot examples drawn per criterion
# =============================================================================


def _training() -> RubricDataset:
    training = RubricDataset(prompt="Explain photosynthesis.", rubric=Rubric([CLARITY]))
    training.add_item(submission="[see attached PDF]", description="na", ground_truth=[NA_LABEL])
    training.add_item(submission="Plants use light.", description="clear", ground_truth=["Clear"])
    return training


def _content_text(content: Any) -> list[str]:
    """A message's content as text, whether a string or a list of parts."""
    return [content] if isinstance(content, str) else [part["text"] for part in content]


class TestPerCriterionFewShot:
    @pytest.mark.parametrize("balance", [True, False])
    def test_the_label_names_the_appended_option(self, balance):
        grader = _grader(
            training_data=_training(),
            few_shot_config=FewShotConfig(n_examples=5, balance_verdicts=balance),
        )
        examples = grader._multi_choice_examples[(0, "default")]
        assert sorted(examples) == [("Plants use light.", 1, None), ("[see attached PDF]", 2, None)]

    def test_without_auto_na_option_there_is_no_option_to_name(self):
        grader = _grader(
            training_data=_training(),
            few_shot_config=FewShotConfig(n_examples=5),
            auto_na_option=False,
        )
        assert grader._multi_choice_examples[(0, "default")] == [("Plants use light.", 1, None)]

    def test_an_option_index_is_resolved_among_the_options_shown(self):
        training = _training()
        training.items[1].ground_truth = [2]  # ty: ignore[invalid-assignment]
        with_na = _grader(training_data=training, few_shot_config=FewShotConfig(n_examples=5))
        assert sorted(with_na._multi_choice_examples[(0, "default")]) == [
            ("Plants use light.", 2, None),
            ("[see attached PDF]", 2, None),
        ]
        forced = _grader(
            training_data=training,
            few_shot_config=FewShotConfig(n_examples=5),
            auto_na_option=False,
        )
        assert forced._multi_choice_examples[(0, "default")] == []

    @pytest.mark.asyncio
    async def test_the_example_is_shown_as_the_na_option(self):
        grader = _grader(training_data=_training(), few_shot_config=FewShotConfig(n_examples=5))
        prompts: list[str] = []

        async def acompletion(**params):
            # The examples sit in the start of the prompt, which may be a message of its own.
            parts = [part for m in params["messages"][1:] for part in _content_text(m["content"])]
            prompts.append("\n\n".join(parts))
            return _choice_response(2)

        with patch("litellm.acompletion", side_effect=acompletion):
            await Rubric([CLARITY]).grade("Leaves catch light.", grader=grader, query="q")

        (prompt,) = prompts
        shown = re.search(
            r"<example_submission>\[see attached PDF\]</example_submission>\n(.*?)\n",
            prompt,
        )
        assert shown is not None
        assert shown.groups() == (f"<selected_label>{NA_LABEL}</selected_label>",)


# =============================================================================
# The cascade tools' labels
# =============================================================================


class TestCascadeLabels:
    def test_the_label_abstains_and_leaves_the_other_labels(self):
        dataset = RubricDataset(prompt="p", rubric=Rubric([CLARITY, LIGHT]))
        dataset.add_item(submission="s", description="d", ground_truth=[NA_LABEL, MET])
        assert _labels(dataset.items[0], [CLARITY, LIGHT]) == [None, MET]
