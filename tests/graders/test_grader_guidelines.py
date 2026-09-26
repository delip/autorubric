"""Rubric guidelines through the grading API: ``Rubric.grade`` -> ``Grader.grade`` -> ``judge``.

A rubric's guidelines reach its grader without touching the abstract ``Grader`` API:

- ``Rubric.grade`` passes ``guidelines=`` to ``grader.grade`` only when the rubric has
  guidelines, so a grader is called exactly as before for a rubric without them.
- ``Grader.grade`` takes a keyword-only ``guidelines`` and forwards it to ``judge`` only if
  ``judge`` accepts a ``guidelines`` keyword or ``**kwargs``. A ``judge`` written against
  the abstract signature (every grader subclass written before guidelines existed) is
  called exactly as before, and the grader warns once per instance that it ignores them.
- ``CriterionGrader.judge`` accepts them and puts them in front of every LLM user prompt.

The legacy subclasses below are modelled on real ones: the experiments' decision-model
grader (its own keyword ``__init__`` calling ``super().__init__``, a ``judge`` with the
four abstract parameters) and the stub grader of ``tests/eval/test_error_report_none_score.py``.
No test makes a real LLM call.
"""

from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionReport,
    CriterionVerdict,
    DataItem,
    EvaluationReport,
    LengthPenalty,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
    evaluate,
)
from autorubric.graders import CriterionGrader, Grader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.types import CriterionJudgment, MultiChoiceJudgment
from autorubric.utils import fill_ground_truth

GUIDELINES = "Writers are grade 8-12 English-language learners; judge against that level."
QUERY = "Write a persuasive essay."
SUBMISSION = "School should start later because sleep improves learning."
CRITERIA = [
    Criterion(name="thesis", weight=3.0, requirement="States a clear, arguable thesis"),
    Criterion(name="evidence", weight=2.0, requirement="Supports claims with cited evidence"),
]
BLOCK_START = "<guidelines>\n"
PRECEDENCE_RULE = (
    "These guidelines apply to every criterion. The criterion text governs; the guidelines "
    "clarify how to apply it."
)


def _met_report(rubric: list[Criterion]) -> EvaluationReport:
    reports = [
        CriterionReport(
            requirement=c.requirement,
            name=c.name,
            weight=c.weight,
            verdict=CriterionVerdict.MET,
            reason="present",
        )
        for c in rubric
    ]
    return EvaluationReport(score=1.0, raw_score=sum(c.weight for c in rubric), report=reports)


# =============================================================================
# Graders written against the abstract signature (no guidelines keyword)
# =============================================================================


class LegacyJevStyleGrader(Grader):
    """Modelled on the experiments' decision-model grader: keyword ``__init__`` calling
    ``super().__init__``, ``judge`` with exactly the four abstract parameters."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        length_penalty: LengthPenalty | None = None,
        normalize: bool = True,
    ):
        super().__init__(length_penalty=length_penalty, normalize=normalize)
        self.threshold = threshold
        self.judge_calls: list[tuple[Any, ...]] = []

    async def judge(
        self,
        to_grade: str,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> dict[str, Any]:
        self.judge_calls.append((to_grade, rubric, query, reference_submission))
        return {"rubric": rubric}

    async def aggregate(
        self, judge_results: dict[str, Any], *, normalize: bool = True
    ) -> EvaluationReport:
        return _met_report(judge_results["rubric"])


class LegacyStubGrader(Grader):
    """Modelled on the stub grader of ``tests/eval/test_error_report_none_score.py``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.judge_calls: list[tuple[Any, ...]] = []

    async def judge(self, to_grade, rubric, query=None, reference_submission=None):  # noqa: D102
        self.judge_calls.append((to_grade, rubric, query, reference_submission))
        return rubric

    async def aggregate(self, judge_results, *, normalize: bool = True):  # noqa: D102
        return _met_report(judge_results)


class LegacyNoSuperInitGrader(Grader):
    """A subclass that never calls ``Grader.__init__`` and sets what ``grade`` reads itself."""

    def __init__(self) -> None:
        self.length_penalty = None
        self.normalize = True
        self.judge_calls: list[tuple[Any, ...]] = []

    async def judge(self, to_grade, rubric, query=None, reference_submission=None):  # noqa: D102
        self.judge_calls.append((to_grade, rubric, query, reference_submission))
        return rubric

    async def aggregate(self, judge_results, *, normalize: bool = True):  # noqa: D102
        return _met_report(judge_results)


class PositionalOnlyGuidelinesGrader(Grader):
    """Its ``guidelines`` parameter cannot be passed by keyword, so it takes none."""

    def __init__(self) -> None:
        super().__init__()
        self.judge_calls: list[tuple[Any, ...]] = []

    async def judge(
        self, to_grade, rubric, query=None, reference_submission=None, guidelines=None, /
    ):  # noqa: D102
        self.judge_calls.append((to_grade, rubric, query, reference_submission))
        return rubric

    async def aggregate(self, judge_results, *, normalize: bool = True):  # noqa: D102
        return _met_report(judge_results)


class UnreadableSignatureGrader(Grader):
    """Its ``judge`` takes ``**kwargs``, but ``inspect.signature`` cannot read it (its
    ``__signature__`` is not a ``Signature``, so ``TypeError`` on Python 3.11-3.13); a
    signature that cannot be read counts as not accepting ``guidelines=``."""

    def __init__(self) -> None:
        super().__init__()
        self.judge_calls: list[tuple[Any, ...]] = []
        self.judge_kwargs: list[dict[str, Any]] = []

    async def judge(self, to_grade, rubric, query=None, reference_submission=None, **kwargs):  # noqa: D102
        self.judge_calls.append((to_grade, rubric, query, reference_submission))
        self.judge_kwargs.append(kwargs)
        return rubric

    judge.__signature__ = object()

    async def aggregate(self, judge_results, *, normalize: bool = True):  # noqa: D102
        return _met_report(judge_results)


LEGACY_GRADERS = [
    LegacyJevStyleGrader,
    LegacyStubGrader,
    LegacyNoSuperInitGrader,
    PositionalOnlyGuidelinesGrader,
    UnreadableSignatureGrader,
]


class SlottedDuckGrader:
    """Not a ``Grader``: an object with the old ``grade`` signature, which ``Rubric.grade``
    accepts like any grader. ``__slots__`` leaves it without an instance ``__dict__``, so
    nothing is cached or flagged on it."""

    __slots__ = ("grade_calls",)

    def __init__(self) -> None:
        self.grade_calls: list[tuple[Any, ...]] = []

    async def grade(self, to_grade, rubric, query=None, reference_submission=None):  # noqa: D102
        self.grade_calls.append((to_grade, rubric, query, reference_submission))
        return _met_report(rubric)


# =============================================================================
# Graders whose judge takes guidelines
# =============================================================================


class KeywordGuidelinesGrader(Grader):
    """A ``judge`` with a keyword-only ``guidelines`` parameter."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.received: list[str | None] = []

    async def judge(
        self, to_grade, rubric, query=None, reference_submission=None, *, guidelines=None
    ):  # noqa: D102
        self.received.append(guidelines)
        return rubric

    async def aggregate(self, judge_results, *, normalize: bool = True):  # noqa: D102
        return _met_report(judge_results)


class PositionalOrKeywordGuidelinesGrader(KeywordGuidelinesGrader):
    """A ``judge`` whose ``guidelines`` parameter is positional-or-keyword."""

    async def judge(self, to_grade, rubric, query=None, reference_submission=None, guidelines=None):  # noqa: D102
        self.received.append(guidelines)
        return rubric


class VarKeywordGrader(KeywordGuidelinesGrader):
    """A ``judge`` taking ``**kwargs``."""

    async def judge(self, to_grade, rubric, query=None, reference_submission=None, **kwargs):  # noqa: D102
        self.received.append(kwargs.get("guidelines"))
        return rubric


ACCEPTING_GRADERS = [
    KeywordGuidelinesGrader,
    PositionalOrKeywordGuidelinesGrader,
    VarKeywordGrader,
]


@pytest.fixture
def all_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Every warning issued in the test, none filtered or deduplicated."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _ignores_guidelines(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if "ignores rubric guidelines" in str(w.message)]


# =============================================================================
# The API surface
# =============================================================================


class TestSignatures:
    def test_grader_grade_takes_a_keyword_only_guidelines_defaulting_to_none(self):
        parameter = inspect.signature(Grader.grade).parameters["guidelines"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    def test_abstract_judge_and_aggregate_signatures_are_unchanged(self):
        assert list(inspect.signature(Grader.judge).parameters) == [
            "self",
            "to_grade",
            "rubric",
            "query",
            "reference_submission",
        ]
        assert list(inspect.signature(Grader.aggregate).parameters) == [
            "self",
            "judge_results",
            "normalize",
        ]

    def test_criterion_grader_judge_takes_a_keyword_only_guidelines(self):
        parameter = inspect.signature(CriterionGrader.judge).parameters["guidelines"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


# =============================================================================
# Rubric.grade -> grader.grade
# =============================================================================


class TestRubricGradeForwarding:
    @pytest.mark.asyncio
    async def test_without_guidelines_grade_is_called_exactly_as_before(self):
        grader = MagicMock()
        grader.grade = AsyncMock(return_value=_met_report(CRITERIA))
        await Rubric(CRITERIA).grade(SUBMISSION, grader, query=QUERY)
        grader.grade.assert_awaited_once_with(
            to_grade=SUBMISSION, rubric=CRITERIA, query=QUERY, reference_submission=None
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank", ["", "  \n"])
    async def test_blank_guidelines_are_not_passed(self, blank):
        grader = MagicMock()
        grader.grade = AsyncMock(return_value=_met_report(CRITERIA))
        await Rubric(CRITERIA, guidelines=blank).grade(SUBMISSION, grader, query=QUERY)
        assert "guidelines" not in grader.grade.await_args.kwargs

    @pytest.mark.asyncio
    async def test_guidelines_are_passed_when_the_rubric_has_them(self):
        grader = MagicMock()
        grader.grade = AsyncMock(return_value=_met_report(CRITERIA))
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(
            SUBMISSION, grader, query=QUERY, reference_submission="ref"
        )
        grader.grade.assert_awaited_once_with(
            to_grade=SUBMISSION,
            rubric=CRITERIA,
            query=QUERY,
            reference_submission="ref",
            guidelines=GUIDELINES,
        )

    @pytest.mark.asyncio
    async def test_guidelines_reach_a_judge_that_accepts_them(self):
        grader = KeywordGuidelinesGrader()
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(SUBMISSION, grader)
        await Rubric(CRITERIA).grade(SUBMISSION, grader)
        assert grader.received == [GUIDELINES, None]

    @pytest.mark.asyncio
    async def test_a_grade_override_without_guidelines_is_called_without_them_and_warns_once(
        self, all_warnings
    ):
        """A grader that overrides ``grade`` with the old signature is called exactly as
        before, as a legacy ``judge`` is: it ignores the guidelines, and says so once."""

        class LegacyGradeOverride(LegacyStubGrader):
            async def grade(self, to_grade, rubric, query=None, reference_submission=None):  # noqa: D102
                return await super().grade(to_grade, rubric, query, reference_submission)

        grader = LegacyGradeOverride()
        rubric = Rubric(CRITERIA, guidelines=GUIDELINES)
        for _ in range(3):
            report = await rubric.grade(SUBMISSION, grader, query=QUERY)
            assert report.score == 1.0
        assert grader.judge_calls == [(SUBMISSION, CRITERIA, QUERY, None)] * 3
        (warning,) = _ignores_guidelines(all_warnings)
        assert warning.category is UserWarning
        assert str(warning.message) == "LegacyGradeOverride ignores rubric guidelines"
        assert warning.filename == __file__


# =============================================================================
# Grader.grade -> judge
# =============================================================================


class TestGraderGradeForwarding:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("grader_cls", ACCEPTING_GRADERS)
    async def test_a_judge_accepting_guidelines_receives_them(self, grader_cls, all_warnings):
        grader = grader_cls()
        await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        await grader.grade(SUBMISSION, CRITERIA)
        assert grader.received == [GUIDELINES, None]
        assert _ignores_guidelines(all_warnings) == []

    @pytest.mark.asyncio
    async def test_without_guidelines_judge_gets_the_four_arguments_positionally(self):
        """No guidelines: ``judge`` is called exactly as before guidelines existed."""
        grader = KeywordGuidelinesGrader()
        judge = AsyncMock(return_value=CRITERIA)
        grader.judge = judge
        await grader.grade(SUBMISSION, CRITERIA, QUERY, "ref")
        judge.assert_awaited_once_with(SUBMISSION, CRITERIA, QUERY, "ref")

    @pytest.mark.asyncio
    async def test_with_guidelines_judge_gets_them_as_a_keyword(self):
        grader = KeywordGuidelinesGrader()
        judge = AsyncMock(return_value=CRITERIA)
        grader.judge = judge
        await grader.grade(SUBMISSION, CRITERIA, QUERY, "ref", guidelines=GUIDELINES)
        judge.assert_awaited_once_with(SUBMISSION, CRITERIA, QUERY, "ref", guidelines=GUIDELINES)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank", ["", " \n\t"])
    async def test_blank_guidelines_are_no_guidelines(self, blank, all_warnings):
        grader = KeywordGuidelinesGrader()
        judge = AsyncMock(return_value=CRITERIA)
        grader.judge = judge
        await grader.grade(SUBMISSION, CRITERIA, guidelines=blank)
        judge.assert_awaited_once_with(SUBMISSION, CRITERIA, None, None)
        legacy = LegacyStubGrader()
        await legacy.grade(SUBMISSION, CRITERIA, guidelines=blank)
        assert _ignores_guidelines(all_warnings) == []

    @pytest.mark.asyncio
    async def test_non_string_guidelines_raise_type_error(self):
        with pytest.raises(TypeError, match="guidelines"):
            await KeywordGuidelinesGrader().grade(SUBMISSION, CRITERIA, guidelines=["x"])

    @pytest.mark.asyncio
    async def test_structured_submission_is_flattened_as_before(self):
        grader = KeywordGuidelinesGrader()
        judge = AsyncMock(return_value=CRITERIA)
        grader.judge = judge
        await grader.grade({"thinking": "T", "output": "O"}, CRITERIA, guidelines=GUIDELINES)
        judge.assert_awaited_once_with(
            "<thinking>T</thinking>\n<output>O</output>",
            CRITERIA,
            None,
            None,
            guidelines=GUIDELINES,
        )


class TestLegacyGraders:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("grader_cls", LEGACY_GRADERS)
    async def test_grades_without_guidelines_and_warns_once_per_instance(
        self, grader_cls, all_warnings
    ):
        grader = grader_cls()
        rubric = Rubric(CRITERIA, guidelines=GUIDELINES)
        for _ in range(3):
            report = await rubric.grade(SUBMISSION, grader, query=QUERY)
            assert report.score == 1.0
        await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        assert grader.judge_calls == [(SUBMISSION, CRITERIA, QUERY, None)] * 3 + [
            (SUBMISSION, CRITERIA, None, None)
        ]
        (warning,) = _ignores_guidelines(all_warnings)
        assert warning.category is UserWarning
        assert str(warning.message) == f"{grader_cls.__name__} ignores rubric guidelines"

        # Every instance says it once.
        other = grader_cls()
        await rubric.grade(SUBMISSION, other)
        await rubric.grade(SUBMISSION, other)
        assert len(_ignores_guidelines(all_warnings)) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("grader_cls", LEGACY_GRADERS)
    async def test_warning_raises_on_every_call_under_an_error_filter(self, grader_cls):
        """Under warnings-as-errors the warning is an exception on every call, as Python's
        own warning registry makes it: an instance whose warning raised has not warned, so
        it never goes on to grade without the guidelines unannounced. Once a warning is
        issued, the instance stays silent."""
        grader = grader_cls()
        rubric = Rubric(CRITERIA, guidelines=GUIDELINES)
        message = f"{grader_cls.__name__} ignores rubric guidelines"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _ in range(3):
                with pytest.raises(UserWarning, match=message):
                    await rubric.grade(SUBMISSION, grader, query=QUERY)
                with pytest.raises(UserWarning, match=message):
                    await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        assert grader.judge_calls == []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                await rubric.grade(SUBMISSION, grader, query=QUERY)
        assert [str(w.message) for w in _ignores_guidelines(caught)] == [message]
        assert grader.judge_calls == [(SUBMISSION, CRITERIA, QUERY, None)] * 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("grader_cls", LEGACY_GRADERS)
    async def test_no_warning_for_a_rubric_without_guidelines(self, grader_cls, all_warnings):
        grader = grader_cls()
        await Rubric(CRITERIA).grade(SUBMISSION, grader, query=QUERY)
        await grader.grade(SUBMISSION, CRITERIA, guidelines=None)
        assert grader.judge_calls == [
            (SUBMISSION, CRITERIA, QUERY, None),
            (SUBMISSION, CRITERIA, None, None),
        ]
        assert all_warnings == []

    @pytest.mark.asyncio
    async def test_warning_names_the_callers_line(self, all_warnings):
        """Attributed to the first frame outside autorubric: here, this test, whether it
        calls ``Rubric.grade`` or ``Grader.grade``."""
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(SUBMISSION, LegacyStubGrader())
        await LegacyJevStyleGrader().grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        caught = _ignores_guidelines(all_warnings)
        assert [Path(w.filename) for w in caught] == [Path(__file__)] * 2

    def test_evaluate_names_the_line_that_starts_the_event_loop(self, tmp_path):
        """``evaluate`` grades items in tasks, whose stacks hold no caller frame above the
        event loop: the warning names the line that started it. Three items, one warning."""
        dataset = RubricDataset(
            prompt=QUERY, rubric=Rubric(CRITERIA, guidelines=GUIDELINES), name="legacy"
        )
        for n in range(3):
            dataset.add_item(f"{SUBMISSION} ({n})", f"item {n}")
        grader = LegacyStubGrader()
        run = evaluate(
            dataset, grader, show_progress=False, experiment_name="legacy", experiments_dir=tmp_path
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = inspect.currentframe()
            assert frame is not None
            line = frame.f_lineno + 1
            result = asyncio.run(run)
        assert result.successful_items == 3 and len(grader.judge_calls) == 3
        (warning,) = _ignores_guidelines(caught)
        assert (Path(warning.filename), warning.lineno) == (Path(__file__), line)

    @pytest.mark.asyncio
    async def test_judge_signature_is_inspected_once_per_instance(self, monkeypatch):
        real_signature = inspect.signature
        inspected: list[Any] = []

        def counting_signature(obj: Any, *args: Any, **kwargs: Any) -> inspect.Signature:
            if getattr(obj, "__func__", None) is LegacyStubGrader.judge:
                inspected.append(obj)
            return real_signature(obj, *args, **kwargs)

        monkeypatch.setattr(inspect, "signature", counting_signature)
        grader = LegacyStubGrader()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(4):
                await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
            assert len(inspected) == 1
            await LegacyStubGrader().grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        assert len(inspected) == 2

    @pytest.mark.asyncio
    async def test_a_judge_replaced_on_the_instance_is_inspected_again(self, all_warnings):
        grader = LegacyStubGrader()
        await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        judge = AsyncMock(return_value=CRITERIA)
        grader.judge = judge  # accepts **kwargs
        await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        judge.assert_awaited_once_with(SUBMISSION, CRITERIA, None, None, guidelines=GUIDELINES)
        assert len(_ignores_guidelines(all_warnings)) == 1


class TestSignatureProbeFallbacks:
    """The documented fallbacks of the check whether a method accepts ``guidelines=``."""

    @pytest.mark.asyncio
    async def test_a_judge_whose_signature_cannot_be_read_is_called_without_guidelines(
        self, all_warnings
    ):
        grader = UnreadableSignatureGrader()
        with pytest.raises(TypeError):
            inspect.signature(grader.judge)
        await Rubric(CRITERIA, guidelines=GUIDELINES).grade(SUBMISSION, grader, query=QUERY)
        await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        assert grader.judge_calls == [
            (SUBMISSION, CRITERIA, QUERY, None),
            (SUBMISSION, CRITERIA, None, None),
        ]
        assert grader.judge_kwargs == [{}, {}]
        (warning,) = _ignores_guidelines(all_warnings)
        assert str(warning.message) == "UnreadableSignatureGrader ignores rubric guidelines"

    @pytest.mark.asyncio
    async def test_an_object_without_an_instance_dict_is_inspected_and_warns_on_every_call(
        self, monkeypatch, all_warnings
    ):
        real_signature = inspect.signature
        inspected: list[Any] = []

        def counting_signature(obj: Any, *args: Any, **kwargs: Any) -> inspect.Signature:
            if getattr(obj, "__func__", None) is SlottedDuckGrader.grade:
                inspected.append(obj)
            return real_signature(obj, *args, **kwargs)

        monkeypatch.setattr(inspect, "signature", counting_signature)
        grader = SlottedDuckGrader()
        assert not hasattr(grader, "__dict__")
        rubric = Rubric(CRITERIA, guidelines=GUIDELINES)
        for _ in range(3):
            report = await rubric.grade(SUBMISSION, grader, query=QUERY)
            assert report.score == 1.0
        assert grader.grade_calls == [(SUBMISSION, CRITERIA, QUERY, None)] * 3
        assert len(inspected) == 3
        caught = _ignores_guidelines(all_warnings)
        assert [str(w.message) for w in caught] == [
            "SlottedDuckGrader ignores rubric guidelines"
        ] * 3
        assert [Path(w.filename) for w in caught] == [Path(__file__)] * 3


class TestOtherGradingPaths:
    @pytest.mark.asyncio
    async def test_fill_ground_truth_grades_each_item_with_its_own_rubric_guidelines(self):
        dataset = RubricDataset(
            prompt=QUERY, rubric=Rubric(CRITERIA, guidelines="Global guidelines."), name="fill"
        )
        dataset.add_item("Item zero.", "global rubric")
        dataset.add_item(
            "Item one.", "own guidelines", rubric=Rubric(CRITERIA, guidelines="Item one.")
        )
        dataset.add_item("Item two.", "own rubric, none", rubric=Rubric(CRITERIA))
        grader = KeywordGuidelinesGrader()
        received: dict[str, str | None] = {}
        real_judge = grader.judge

        async def judge(to_grade, rubric, query=None, reference_submission=None, **kwargs):
            received[to_grade] = kwargs.get("guidelines")
            return await real_judge(to_grade, rubric, query, reference_submission)

        grader.judge = judge
        filled = await fill_ground_truth(dataset, grader, show_progress=False)
        assert received == {
            "Item zero.": "Global guidelines.",
            "Item one.": "Item one.",
            "Item two.": None,
        }
        assert [item.ground_truth for item in filled.items] == [
            [CriterionVerdict.MET, CriterionVerdict.MET]
        ] * 3


class TestLengthPenaltyPath:
    """``Grader.grade`` rebuilds the report to apply a length penalty; guidelines change
    nothing about that report but the judge's input."""

    @pytest.mark.asyncio
    async def test_penalized_report_is_the_same_with_and_without_guidelines(self):
        penalty = LengthPenalty(free_budget=2, max_cap=10, penalty_at_cap=0.5)
        grader = KeywordGuidelinesGrader(length_penalty=penalty)
        without = await grader.grade(SUBMISSION, CRITERIA)
        with_guidelines = await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        assert grader.received == [None, GUIDELINES]
        assert without.score is not None and without.score < 1.0
        assert with_guidelines.model_dump() == without.model_dump()

    @pytest.mark.asyncio
    async def test_criterion_grader_ensemble_report_has_no_new_fields(self):
        penalty = LengthPenalty(free_budget=2, max_cap=10, penalty_at_cap=0.5)
        llm = RecordingLLM()
        with patch("autorubric.graders.criterion_grader.LLMClient", return_value=llm):
            grader = CriterionGrader(
                judge_model_config=LLMConfig(model="m"), length_penalty=penalty
            )
        without = await grader.grade(SUBMISSION, CRITERIA)
        with_guidelines = await grader.grade(SUBMISSION, CRITERIA, guidelines=GUIDELINES)
        assert with_guidelines.model_dump() == without.model_dump()
        first, second = llm.user_prompts[:2], llm.user_prompts[2:]
        assert all(not p.startswith(BLOCK_START) for p in first)
        assert all(p.startswith(BLOCK_START) for p in second)


# =============================================================================
# CriterionGrader with LLM judges
# =============================================================================


class RecordingLLM:
    """Stand-in for ``LLMClient``: records user prompts; MET / the first option."""

    def __init__(self) -> None:
        self.user_prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def generate(
        self, system_prompt: str, user_prompt: str, response_format: Any = None, **kwargs: Any
    ) -> GenerateResult:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        parsed: Any
        if "<options>" in user_prompt:
            parsed = MultiChoiceJudgment(selected_option=1, explanation="first")
        else:
            parsed = CriterionJudgment(criterion_status=CriterionVerdict.MET, explanation="yes")
        return GenerateResult(
            content="{}",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=None,
            parsed=parsed,
        )


def _block(guidelines: str) -> str:
    return f"<guidelines>\n{guidelines}\n\n{PRECEDENCE_RULE}\n</guidelines>\n\n"


MIXED_CRITERIA = [
    *CRITERIA,
    Criterion(
        name="clarity",
        weight=1.0,
        requirement="How clear is the essay?",
        scale_type="ordinal",
        options=[
            CriterionOption(label="Unclear", value=0.0),
            CriterionOption(label="Clear", value=1.0),
        ],
    ),
]


class TestCriterionGraderLLM:
    @pytest.mark.asyncio
    async def test_judge_puts_the_block_in_front_of_every_criterion_prompt(self):
        llm = RecordingLLM()
        with patch("autorubric.graders.criterion_grader.LLMClient", return_value=llm):
            grader = CriterionGrader(judge_model_config=LLMConfig(model="m"), seed=1)
        await grader.judge(SUBMISSION, MIXED_CRITERIA, QUERY, guidelines=GUIDELINES)
        assert len(llm.user_prompts) == len(MIXED_CRITERIA)
        assert all(p.startswith(_block(GUIDELINES) + "<") for p in llm.user_prompts)
        assert sum("<options>" in p for p in llm.user_prompts) == 1

        # The same prompts, block removed, are what the grader sends without guidelines.
        with_block = sorted(p.removeprefix(_block(GUIDELINES)) for p in llm.user_prompts)
        llm.user_prompts.clear()
        await grader.judge(SUBMISSION, MIXED_CRITERIA, QUERY)
        assert sorted(llm.user_prompts) == with_block

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [5, ["x"]], ids=["int", "list"])
    @pytest.mark.parametrize("criteria", [[], MIXED_CRITERIA], ids=["no-criteria", "criteria"])
    async def test_judge_rejects_non_string_guidelines_before_any_call(self, criteria, bad):
        """Guidelines are text or ``None``; anything else raises ``TypeError`` up front,
        even for a rubric with no criteria to judge, and no judge call is made."""
        llm = RecordingLLM()
        with patch("autorubric.graders.criterion_grader.LLMClient", return_value=llm):
            grader = CriterionGrader(judge_model_config=LLMConfig(model="m"), seed=1)
        with pytest.raises(TypeError, match="guidelines must be a str or None, got"):
            await grader.judge(SUBMISSION, criteria, QUERY, guidelines=bad)
        assert llm.user_prompts == []

    @pytest.mark.asyncio
    async def test_evaluate_uses_each_items_own_rubric_guidelines(self, tmp_path):
        """``EvalRunner`` grades each item with ``get_item_rubric``: an item's own rubric
        brings its own guidelines, or none, and never inherits the dataset rubric's."""
        dataset = RubricDataset(
            prompt=QUERY,
            rubric=Rubric(CRITERIA, guidelines="Global guidelines."),
            name="per-item-guidelines",
        )
        dataset.items = [
            DataItem(submission="Item zero.", description="global rubric"),
            DataItem(
                submission="Item one.",
                description="own rubric, own guidelines",
                rubric=Rubric(MIXED_CRITERIA, guidelines="Item one guidelines."),
            ),
            DataItem(
                submission="Item two.",
                description="own rubric, no guidelines",
                rubric=Rubric(CRITERIA),
            ),
        ]
        llm = RecordingLLM()
        with patch("autorubric.graders.criterion_grader.LLMClient", return_value=llm):
            grader = CriterionGrader(judges=[JudgeSpec(LLMConfig(model="m"), "llm")], seed=3)
        result = await evaluate(
            dataset,
            grader,
            show_progress=False,
            experiment_name="guidelines",
            experiments_dir=tmp_path,
        )
        assert result.successful_items == 3

        def prompts_for(submission: str) -> list[str]:
            return [p for p in llm.user_prompts if f"<submission>\n{submission}\n" in p]

        zero, one, two = (prompts_for(f"Item {n}.") for n in ("zero", "one", "two"))
        assert len(zero) == 2 and all(p.startswith(_block("Global guidelines.")) for p in zero)
        assert len(one) == 3 and all(p.startswith(_block("Item one guidelines.")) for p in one)
        assert len(two) == 2 and all(p.startswith("<criterion_type>") for p in two)
