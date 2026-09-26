"""Repeated ``judge_id``s in a ``CriterionGrader`` panel.

A ``judge_id`` is a judge's identity: it keys the grader's clients, its few-shot examples
and option shuffles, ``EnsembleEvaluationReport.judge_scores`` and the per-judge metrics,
so judges that share one are conflated. A panel (``judges=``) with a repeated ``judge_id``
still builds and grades exactly as before, and warns (``FutureWarning``, shown by default,
since the conflation silently changes results) naming the repeats; a repeated
``judge_id`` becomes a ``ValueError`` in the next major version. A cascade already refuses
repeats (``tests/decision/test_decision_cascade.py``); the mixed-kind case is in
``tests/decision/test_decision_ensemble.py``.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from autorubric import Criterion, CriterionVerdict, EnsembleEvaluationReport, Rubric
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment, TokenUsage

A = LLMConfig(model="model-a")
B = LLMConfig(model="model-b")


def test_a_repeated_judge_id_warns_naming_it():
    with pytest.warns(
        FutureWarning, match=r"judge_ids should be unique; repeated: 'x'\."
    ) as record:
        CriterionGrader(judges=[JudgeSpec(A, "x"), JudgeSpec(B, "x")])
    assert len([w for w in record if w.category is FutureWarning]) == 1


def test_the_warning_names_each_repeated_id_once_in_sorted_order():
    judges = [
        JudgeSpec(A, "b"),
        JudgeSpec(A, "a"),
        JudgeSpec(B, "b"),
        JudgeSpec(B, "a"),
        JudgeSpec(B, "b"),
        JudgeSpec(A, "c"),
    ]
    with pytest.warns(FutureWarning, match=r"repeated: 'a', 'b'\."):
        CriterionGrader(judges=judges)


def test_the_warning_says_a_repeat_will_be_an_error():
    with pytest.warns(FutureWarning, match=r"will be a ValueError in the next major version"):
        CriterionGrader(judges=[JudgeSpec(A, "x"), JudgeSpec(A, "x")])


def test_the_warning_names_the_line_that_built_the_grader():
    with pytest.warns(FutureWarning) as record:
        CriterionGrader(judges=[JudgeSpec(A, "x"), JudgeSpec(B, "x")])
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "kwargs",
    [
        {"judge_model_config": A},
        {"judges": [JudgeSpec(A, "x")]},
        {"judges": [JudgeSpec(A, "a"), JudgeSpec(A, "b")]},
    ],
    ids=["single", "one-spec", "unique-panel"],
)
def test_unique_judge_ids_never_warn(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CriterionGrader(**kwargs)


def test_a_repeated_judge_id_still_builds_the_grader():
    with pytest.warns(FutureWarning):
        grader = CriterionGrader(judges=[JudgeSpec(A, "x"), JudgeSpec(B, "x")])
    assert grader.judge_ids == ["x", "x"]
    assert grader.is_ensemble


@pytest.mark.asyncio
async def test_a_panel_with_a_repeated_judge_id_grades_as_the_warning_describes():
    """Grading is unchanged: every judge still votes, all through the last judge's
    client, and the judges share one ``judge_scores`` entry."""
    clients: dict[str, MagicMock] = {}

    def client_for(config: LLMConfig, *, cache_namespace: str | None = None) -> MagicMock:
        client = MagicMock()

        async def generate(*_args, **_kwargs) -> GenerateResult:
            return GenerateResult(
                content="{}",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                parsed=CriterionJudgment(
                    criterion_status=CriterionVerdict.MET, explanation=config.model
                ),
            )

        client.generate = MagicMock(side_effect=generate)
        clients[config.model] = client
        return client

    rubric = Rubric([Criterion(weight=1.0, requirement="Is it correct?")])
    with patch("autorubric.graders.criterion_grader.LLMClient", side_effect=client_for):
        with pytest.warns(FutureWarning):
            grader = CriterionGrader(judges=[JudgeSpec(A, "x"), JudgeSpec(B, "x")])
        report = await rubric.grade("submission", grader=grader)

    assert isinstance(report, EnsembleEvaluationReport) and report.report is not None
    (criterion_report,) = report.report
    assert [v.judge_id for v in criterion_report.votes] == ["x", "x"]
    assert [v.reason for v in criterion_report.votes] == ["model-b", "model-b"]
    assert clients["model-a"].generate.call_count == 0
    assert clients["model-b"].generate.call_count == 2
    assert list(report.judge_scores) == ["x"]
