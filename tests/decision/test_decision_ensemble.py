"""Mixed ensembles: decision-model judges next to LLM judges in one panel.

A decision-model judge answers every criterion of an item in one request; each LLM judge
still makes one call per criterion. Downstream, a decision-model vote is one more vote:
aggregation (``majority``/``weighted``/``unanimous``/``any`` and every ordinal and nominal
strategy), abstention handling, ``judge_scores``, ``agreement`` and the metrics layer
(Fleiss' kappa and Krippendorff's alpha over the full panel, per-judge metrics) treat it
exactly as an LLM vote with the same verdict.

That is checked against a *mirror* panel: the same judges, where the decision model is
replaced by an LLM judge that returns the same verdicts. Every outcome must be equal; only
what a decision model adds (``probabilities``/``confidence``) or lacks (an explanation)
differs.

Decision models are scripted SDK clients and LLM judges scripted ``LLMClient`` stand-ins,
both answering from ``VOTES`` by the judge's model name. Nothing reaches the network.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typesafe_sdk
from typesafe_sdk import SystemOneResponse

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DecisionModelConfig,
    EnsembleEvaluationReport,
    LLMConfig,
    Rubric,
    RubricDataset,
    TokenUsage,
    evaluate,
)
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult
from autorubric.types import CANONICAL_NA_OPTION, CriterionJudgment, MultiChoiceJudgment

API_KEY = "ts-test-key-never-persisted-c41e"
NA = CANONICAL_NA_OPTION.label
MET, UNMET, CA = "MET", "UNMET", "CANNOT_ASSESS"

RUBRIC = Rubric(
    [
        Criterion(name="light", weight=5.0, requirement="Mentions light"),
        Criterion(name="myth", weight=-3.0, requirement="Claims plants eat soil"),
        Criterion(
            name="clarity",
            weight=2.0,
            requirement="How clear is the explanation?",
            scale_type="ordinal",
            options=[
                CriterionOption(label="Unclear", value=0.0),
                CriterionOption(label="Mostly clear", value=0.6),
                CriterionOption(label="Very clear", value=1.0),
            ],
        ),
        Criterion(
            name="tone",
            weight=1.0,
            requirement="Which tone does the answer take?",
            scale_type="nominal",
            options=[
                CriterionOption(label="Formal", value=1.0),
                CriterionOption(label="Casual", value=0.5),
                CriterionOption(label="Mixed", value=0.0),
            ],
        ),
    ]
)
N_CRITERIA = len(RUBRIC.rubric)
SUBMISSIONS = [f"Answer {i}: plants turn light into sugar." for i in range(6)]

# Each judge's verdicts, per item, on (light, myth, clarity, tone), keyed by the judge's
# model name. Binary verdicts are verdict values and multi-choice verdicts option labels;
# CANNOT_ASSESS and NA are abstentions, which the decision model "jev" makes too.
VOTES: dict[str, list[tuple[str, str, str, str]]] = {
    "jev": [
        (CA, UNMET, "Very clear", NA),
        (MET, UNMET, "Mostly clear", "Formal"),
        (UNMET, MET, "Unclear", "Casual"),
        (MET, UNMET, "Very clear", "Mixed"),
        (UNMET, UNMET, NA, "Formal"),
        (MET, MET, "Mostly clear", "Casual"),
    ],
    "a": [
        (MET, MET, "Mostly clear", "Formal"),
        (MET, UNMET, "Very clear", "Formal"),
        (UNMET, MET, "Unclear", "Mixed"),
        (UNMET, UNMET, "Mostly clear", "Mixed"),
        (UNMET, MET, "Unclear", "Formal"),
        (MET, UNMET, "Very clear", "Casual"),
    ],
    "b": [
        (MET, UNMET, "Unclear", "Casual"),
        (MET, UNMET, "Mostly clear", "Formal"),
        (MET, MET, "Unclear", "Casual"),
        (MET, UNMET, "Very clear", "Mixed"),
        (UNMET, UNMET, "Unclear", "Formal"),
        (CA, MET, "Mostly clear", NA),
    ],
}
GROUND_TRUTH = [
    (MET, UNMET, "Mostly clear", "Formal"),
    (MET, UNMET, "Very clear", "Formal"),
    (UNMET, MET, "Unclear", "Casual"),
    (MET, UNMET, "Very clear", "Mixed"),
    (UNMET, UNMET, "Unclear", "Formal"),
    (MET, MET, "Mostly clear", "Casual"),
]


def ground_truth(item: int) -> list[Any]:
    light, myth, clarity, tone = GROUND_TRUTH[item]
    return [CriterionVerdict(light), CriterionVerdict(myth), clarity, tone]


def dataset() -> RubricDataset:
    data = RubricDataset(prompt="Explain photosynthesis.", rubric=RUBRIC, name="mixed-panel")
    for item, submission in enumerate(SUBMISSIONS):
        data.add_item(submission, f"item {item}", ground_truth=ground_truth(item))
    return data


def dm(model: str) -> DecisionModelConfig:
    # Binary Choice framing, so the decision model can abstain on binary criteria too.
    return DecisionModelConfig(model=model, api_key=API_KEY, binary_framing="choice")


# =============================================================================
# Scripted judges
# =============================================================================


@dataclass
class CallLog:
    """Every request a decision model received and every call an LLM judge received."""

    decision_model: list[tuple[str, dict[str, Any], list[str]]] = field(default_factory=list)
    llm: list[tuple[str, int, int]] = field(default_factory=list)

    def llm_calls(self, model: str) -> list[tuple[int, int]]:
        return [(item, idx) for m, item, idx in self.llm if m == model]

    def decision_model_calls(self, model: str) -> list[tuple[dict[str, Any], list[str]]]:
        return [(state, qids) for m, state, qids in self.decision_model if m == model]


def _choice(selected: str, offered: list[str]) -> dict[str, Any]:
    """A Choice answer selecting ``selected`` with probability 0.7."""
    rest = 0.3 / (len(offered) - 1)
    probabilities = {label: 0.7 if label == selected else rest for label in offered}
    return {"type": "choice", "choice": selected, "confidence": 0.5, "probabilities": probabilities}


class ScriptedSDKClient:
    """Stand-in for ``typesafe_sdk.AsyncTypeSafeClient`` answering from ``VOTES``."""

    def __init__(self, log: CallLog, **kwargs: Any) -> None:
        self.log = log
        self.model = kwargs["model"]

    async def system_one(self, state: Any, questions: Any, **kwargs: Any) -> SystemOneResponse:
        self.log.decision_model.append((self.model, dict(state), list(questions)))
        verdicts = VOTES[self.model][SUBMISSIONS.index(state["submission"])]
        answers = {}
        for qid, question in questions.items():
            answers[qid] = _choice(verdicts[int(qid[1:])], list(question.criteria))
        payload = {"model": self.model, "usage": {"input_tokens": 100}, "answers": answers}
        return SystemOneResponse.model_validate(payload)

    async def aclose(self) -> None:
        pass


_SECTION = r"<{0}>\n(.*?)\n</{0}>"


class ScriptedLLMClient:
    """Stand-in for ``LLMClient`` answering from ``VOTES`` by its model name.

    It reads the item from the submission and the criterion from the prompt, and for a
    multi-choice criterion answers the position of its label among the options as
    presented, so it answers correctly whatever the option shuffle.
    """

    def __init__(self, log: CallLog, config: LLMConfig) -> None:
        self.log = log
        self.model = config.model

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> Any:
        submission = re.search(_SECTION.format("submission"), user_prompt, re.S)
        assert submission is not None
        item = SUBMISSIONS.index(submission.group(1))
        multi_choice = "<options>" in user_prompt
        tag = "question" if multi_choice else "criterion"
        requirement = re.search(_SECTION.format(tag), user_prompt, re.S)
        assert requirement is not None
        idx = [c.requirement for c in RUBRIC.rubric].index(requirement.group(1))
        self.log.llm.append((self.model, item, idx))

        verdict = VOTES[self.model][item][idx]
        explanation = f"{self.model} says {verdict}"
        parsed: Any
        if multi_choice:
            options = re.search(_SECTION.format("options"), user_prompt, re.S)
            assert options is not None
            presented = re.findall(r"^\d+\. (.*)$", options.group(1), re.M)
            parsed = MultiChoiceJudgment(
                selected_option=presented.index(verdict) + 1, explanation=explanation
            )
        else:
            parsed = CriterionJudgment(
                criterion_status=CriterionVerdict(verdict), explanation=explanation
            )
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=0.001,
            parsed=parsed,
        )


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> CallLog:
    call_log = CallLog()
    monkeypatch.setattr(
        typesafe_sdk,
        "AsyncTypeSafeClient",
        lambda **kwargs: ScriptedSDKClient(call_log, **kwargs),
    )
    return call_log


@pytest.fixture
def make_grader(log: CallLog) -> Iterator[Any]:
    """Build graders whose LLM judges are scripted; close decision-model clients at teardown."""
    built: list[CriterionGrader] = []

    def build(**kwargs: Any) -> CriterionGrader:
        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            side_effect=lambda config: ScriptedLLMClient(log, config),
        ):
            grader = CriterionGrader(**kwargs)
        built.append(grader)
        return grader

    yield build
    for grader in built:
        for client in grader._decision_clients.values():
            client.close()


def panel(*, mirror: bool, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> list:
    """The panel ``[jev, a, b]``: ``jev`` a decision model, or, mirrored, an LLM judge."""
    jev = LLMConfig(model="jev") if mirror else dm("jev")
    return [
        JudgeSpec(jev, "jev", weights[0]),
        JudgeSpec(LLMConfig(model="a"), "a", weights[1]),
        JudgeSpec(LLMConfig(model="b"), "b", weights[2]),
    ]


def outcome(report: EnsembleEvaluationReport) -> dict[str, Any]:
    """Everything aggregation, scoring and agreement decide, vote by vote and overall.

    Leaves out only what depends on the judge's kind rather than on its verdicts: reasons,
    ``probabilities``/``confidence``, the option order an LLM was shown, usage and cost.
    """
    return {
        "score": report.score,
        "raw_score": report.raw_score,
        "judge_scores": report.judge_scores,
        "mean_agreement": report.mean_agreement,
        "cannot_assess_count": report.cannot_assess_count,
        "criteria": [
            {
                "final_verdict": cr.final_verdict,
                "final_multi_choice_verdict": cr.final_multi_choice_verdict,
                "agreement": cr.agreement,
                "error": cr.error,
                "votes": [(v.judge_id, v.verdict, v.weight, v.error) for v in cr.votes],
                "multi_choice_votes": [
                    (v.judge_id, v.selected_index, v.selected_label, v.value, v.na, v.weight)
                    for v in cr.multi_choice_votes
                ],
            }
            for cr in report.report or []
        ],
    }


# =============================================================================
# Calls: one request per item per decision model, one call per criterion per LLM
# =============================================================================


@pytest.mark.asyncio
async def test_evaluate_calls_per_judge_kind(make_grader, log, tmp_path: Path):
    grader = make_grader(
        judges=[
            JudgeSpec(dm("jev"), "jev"),
            JudgeSpec(LLMConfig(model="a"), "a"),
            JudgeSpec(dm("b"), "b"),
        ],
    )
    result = await evaluate(
        dataset(), grader, show_progress=False, experiment_name="calls", experiments_dir=tmp_path
    )
    assert result.successful_items == len(SUBMISSIONS)

    # Each decision model: exactly one request per item, carrying every criterion.
    for model in ("jev", "b"):
        requests = log.decision_model_calls(model)
        assert sorted(state["submission"] for state, _ in requests) == sorted(SUBMISSIONS)
        assert all(qids == ["c0", "c1", "c2", "c3"] for _, qids in requests)
    # The LLM judge: exactly one call per (item, criterion).
    calls = Counter(log.llm_calls("a"))
    assert set(calls) == {(i, c) for i in range(len(SUBMISSIONS)) for c in range(N_CRITERIA)}
    assert set(calls.values()) == {1}


# =============================================================================
# Aggregation: a decision-model vote aggregates as an LLM vote with its verdict
# =============================================================================

AGGREGATIONS = {
    "defaults": {},
    "weighted": {
        "aggregation": "weighted",
        "ordinal_aggregation": "weighted_mean",
        "nominal_aggregation": "weighted_mode",
        "weights": (3.0, 1.0, 1.0),
    },
    "weighted-light-decision-model": {
        "aggregation": "weighted",
        "ordinal_aggregation": "weighted_mean",
        "nominal_aggregation": "weighted_mode",
        "weights": (0.5, 1.0, 2.0),
    },
    "conservative": {
        "aggregation": "unanimous",
        "ordinal_aggregation": "min",
        "nominal_aggregation": "unanimous",
    },
    "permissive": {"aggregation": "any", "ordinal_aggregation": "max"},
    "median": {"ordinal_aggregation": "median"},
    "mode": {"ordinal_aggregation": "mode"},
}


def _grader_kwargs(setting: dict[str, Any], *, mirror: bool) -> dict[str, Any]:
    kwargs = dict(setting)
    weights = kwargs.pop("weights", (1.0, 1.0, 1.0))
    return {"judges": panel(mirror=mirror, weights=weights), "seed": 5, **kwargs}


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", list(AGGREGATIONS))
async def test_mixed_panel_aggregates_like_its_all_llm_mirror(make_grader, log, setting):
    mixed = make_grader(**_grader_kwargs(AGGREGATIONS[setting], mirror=False))
    mirror = make_grader(**_grader_kwargs(AGGREGATIONS[setting], mirror=True))

    for submission in SUBMISSIONS:
        mixed_report = await RUBRIC.grade(submission, grader=mixed)
        mirror_report = await RUBRIC.grade(submission, grader=mirror)
        assert outcome(mixed_report) == outcome(mirror_report)

        # What the decision model adds, and what it lacks: probabilities and confidence,
        # and no explanation, so only the LLM judges' reasons are joined.
        for cr, mirror_cr in zip(mixed_report.report, mirror_report.report, strict=True):
            jev_vote, *llm_votes = cr.votes or cr.multi_choice_votes
            assert jev_vote.probabilities is not None and jev_vote.confidence is not None
            assert jev_vote.reason is None
            assert all(v.probabilities is None and v.reason for v in llm_votes)
            assert cr.final_reason == " | ".join(f"{v.judge_id}: {v.reason}" for v in llm_votes)
            assert mirror_cr.final_reason.startswith("jev: jev says")

    # The mirror's LLM "jev" was asked once per criterion, the decision model once per item.
    assert len(log.decision_model_calls("jev")) == len(SUBMISSIONS)
    assert len(log.llm_calls("jev")) == len(SUBMISSIONS) * N_CRITERIA


@pytest.mark.asyncio
async def test_decision_model_abstentions_leave_the_vote(make_grader):
    """Item 0: the decision model abstains on ``light`` (CANNOT_ASSESS) and ``tone`` (NA)."""
    grader = make_grader(judges=panel(mirror=False), aggregation="unanimous", seed=5)
    report = await RUBRIC.grade(SUBMISSIONS[0], grader=grader)
    light, _, _, tone = report.report

    assert light.votes[0].verdict == CriterionVerdict.CANNOT_ASSESS
    assert not light.votes[0].is_error  # a genuine abstention, not a failure
    # Both LLM judges say MET; the abstention is not counted against unanimity.
    assert light.final_verdict == CriterionVerdict.MET
    # The NA vote is set aside; "Formal" and "Casual" tie, broken to the lower value.
    assert tone.multi_choice_votes[0].na is True
    assert tone.final_multi_choice_verdict.selected_label == "Casual"
    # judge_scores: every judge, the decision model scored over its own verdicts.
    assert set(report.judge_scores) == {"jev", "a", "b"}
    assert all(score is not None for score in report.judge_scores.values())


# =============================================================================
# Metrics: a mixed panel is an ordinary full panel
# =============================================================================


# Bootstrap resamples of six items are often single-label; scikit-learn says so each time.
@pytest.mark.filterwarnings("ignore:A single label was found:UserWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["defaults", "weighted"])
async def test_metrics_of_a_mixed_panel_equal_its_mirrors(
    make_grader, log, tmp_path: Path, setting
):
    data = dataset()
    results = {}
    for mirror in (False, True):
        grader = make_grader(**_grader_kwargs(AGGREGATIONS[setting], mirror=mirror))
        results[mirror] = await evaluate(
            data,
            grader,
            show_progress=False,
            experiment_name=f"metrics-{setting}-{mirror}",
            experiments_dir=tmp_path,
        )

    metrics = {"per_judge": True, "bootstrap": True, "n_bootstrap": 100, "seed": 3}
    mixed = results[False].compute_metrics(data, **metrics)
    mirror = results[True].compute_metrics(data, **metrics)
    assert mixed.model_dump() == mirror.model_dump()
    assert mixed.summary() == mirror.summary()

    # Inter-judge agreement is measured over the full three-judge panel.
    assert mixed.mean_krippendorff_alpha is not None
    assert any(m.fleiss_kappa is not None for m in mixed.per_criterion)
    assert mixed.per_judge is not None and list(mixed.per_judge) == ["jev", "a", "b"]
