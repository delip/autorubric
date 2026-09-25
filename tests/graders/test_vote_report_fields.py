"""Vote/report fields for probabilistic judges, and ``None`` reasons.

``CriterionReport``, ``JudgeVote`` and ``MultiChoiceJudgeVote`` carry ``probabilities`` and
``confidence`` (``None`` for LLM votes); votes carry ``superseded`` (recorded but not
aggregated); ``EnsembleCriterionReport`` carries ``escalated``. ``reason`` /
``final_reason`` may be ``None`` ("this judge gives no explanation"), and the ensemble
reason joins skip such votes by identity, so an empty LLM explanation still renders.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autorubric import Criterion, CriterionOption, CriterionVerdict, LLMConfig
from autorubric.graders import CriterionGrader
from autorubric.graders.criterion_grader import CriterionResult, JudgeCriterionResults
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    CriterionReport,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
    MultiChoiceJudgeVote,
    MultiChoiceVerdict,
)

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CA = CriterionVerdict.CANNOT_ASSESS

BINARY = Criterion(name="facts", weight=2.0, requirement="States the facts")
OPTIONS = [
    CriterionOption(label="Poor", value=0.0),
    CriterionOption(label="Good", value=1.0),
    CriterionOption(label="N/A", value=0.0, na=True),
]
ORDINAL = Criterion(
    name="quality", weight=1.0, requirement="How good?", scale_type="ordinal", options=OPTIONS
)


def _grader(**kwargs) -> CriterionGrader:
    return CriterionGrader(judge_model_config=LLMConfig(model="test-model"), **kwargs)


def _mc_vote(judge_id: str, idx: int, **kwargs) -> MultiChoiceJudgeVote:
    opt = OPTIONS[idx]
    fields = {"reason": f"{judge_id} reason", **kwargs}
    return MultiChoiceJudgeVote(
        judge_id=judge_id,
        selected_index=idx,
        selected_label=opt.label,
        value=opt.value,
        na=opt.na,
        **fields,
    )


def _final_mc(idx: int) -> AggregatedMultiChoiceVerdict:
    opt = OPTIONS[idx]
    return AggregatedMultiChoiceVerdict(
        selected_index=idx,
        selected_label=opt.label,
        value=opt.value,
        na=opt.na,
        aggregated_value=opt.value,
    )


class TestFieldDefaults:
    def test_votes_default_to_llm_shape(self):
        vote = JudgeVote(judge_id="j", verdict=MET, reason="r")
        assert vote.probabilities is None
        assert vote.confidence is None
        assert vote.superseded is False
        mc = _mc_vote("j", 1)
        assert mc.probabilities is None
        assert mc.confidence is None
        assert mc.superseded is False

    def test_reports_default_to_llm_shape(self):
        report = CriterionReport(weight=1.0, requirement="r", verdict=MET, reason="r")
        assert report.probabilities is None
        assert report.confidence is None
        ecr = EnsembleCriterionReport(criterion=BINARY, final_verdict=MET, final_reason="r")
        assert ecr.escalated is False

    def test_new_fields_accept_values(self):
        vote = JudgeVote(
            judge_id="jev",
            verdict=MET,
            reason=None,
            probabilities={"MET": 0.83, "UNMET": 0.17},
            confidence=0.66,
            superseded=True,
        )
        assert vote.probabilities == {"MET": 0.83, "UNMET": 0.17}
        assert vote.confidence == pytest.approx(0.66)
        assert vote.superseded is True
        mc = _mc_vote("jev", 1, reason=None, probabilities={"0": 0.2, "1": 0.8}, confidence=0.6)
        assert mc.probabilities == {"0": 0.2, "1": 0.8}
        report = CriterionReport(
            weight=1.0,
            requirement="r",
            verdict=MET,
            reason=None,
            probabilities={"MET": 0.9, "UNMET": 0.1},
            confidence=0.8,
        )
        assert report.reason is None and report.confidence == pytest.approx(0.8)

    def test_reason_accepts_none_but_stays_required(self):
        assert JudgeVote(judge_id="j", verdict=MET, reason=None).reason is None
        ecr = EnsembleCriterionReport(criterion=BINARY, final_verdict=MET, final_reason=None)
        assert ecr.final_reason is None
        with pytest.raises(ValidationError):
            JudgeVote(judge_id="j", verdict=MET)
        with pytest.raises(ValidationError):
            CriterionReport(weight=1.0, requirement="r", verdict=MET)
        with pytest.raises(ValidationError):
            EnsembleCriterionReport(criterion=BINARY, final_verdict=MET)

    def test_judge_scores_values_may_be_none(self):
        report = EnsembleEvaluationReport(
            score=0.5, raw_score=1.0, judge_scores={"jev": 0.5, "escalation": None}
        )
        assert report.judge_scores == {"jev": 0.5, "escalation": None}


class TestAgreementIgnoresSupersededVotes:
    def test_binary(self):
        ecr = EnsembleCriterionReport(
            criterion=BINARY,
            final_verdict=UNMET,
            final_reason="llm: no",
            votes=[
                JudgeVote(judge_id="jev", verdict=MET, reason=None, superseded=True),
                JudgeVote(judge_id="llm", verdict=UNMET, reason="no"),
            ],
            escalated=True,
        )
        # Only the non-superseded vote counts: 1/1, not 1/2.
        assert ecr.agreement == pytest.approx(1.0)

    def test_binary_partial_agreement_among_active_votes(self):
        ecr = EnsembleCriterionReport(
            criterion=BINARY,
            final_verdict=UNMET,
            final_reason="x",
            votes=[
                JudgeVote(judge_id="jev", verdict=UNMET, reason=None, superseded=True),
                JudgeVote(judge_id="a", verdict=UNMET, reason="no"),
                JudgeVote(judge_id="b", verdict=MET, reason="yes"),
            ],
            escalated=True,
        )
        assert ecr.agreement == pytest.approx(0.5)

    def test_multi_choice(self):
        ecr = EnsembleCriterionReport(
            criterion=ORDINAL,
            final_verdict=None,
            final_reason="llm: good",
            final_multi_choice_verdict=_final_mc(1),
            multi_choice_votes=[
                _mc_vote("jev", 0, reason=None, superseded=True),
                _mc_vote("llm", 1),
            ],
            escalated=True,
        )
        assert ecr.agreement == pytest.approx(1.0)

    def test_without_superseded_votes_unchanged(self):
        ecr = EnsembleCriterionReport(
            criterion=BINARY,
            final_verdict=MET,
            final_reason="x",
            votes=[
                JudgeVote(judge_id="a", verdict=MET, reason="yes"),
                JudgeVote(judge_id="b", verdict=UNMET, reason="no"),
            ],
        )
        assert ecr.agreement == pytest.approx(0.5)

    def test_supplied_nonzero_agreement_is_kept(self):
        ecr = EnsembleCriterionReport(
            criterion=BINARY,
            final_verdict=UNMET,
            final_reason="x",
            votes=[
                JudgeVote(judge_id="jev", verdict=MET, reason=None, superseded=True),
                JudgeVote(judge_id="llm", verdict=UNMET, reason="no"),
            ],
            agreement=0.25,
        )
        assert ecr.agreement == pytest.approx(0.25)

    @pytest.mark.parametrize("mode", ["python", "json"])
    def test_round_trip_is_idempotent(self, mode):
        binary = EnsembleCriterionReport(
            criterion=BINARY,
            final_verdict=UNMET,
            final_reason="llm: no",
            votes=[
                JudgeVote(
                    judge_id="jev",
                    verdict=MET,
                    reason=None,
                    probabilities={"MET": 0.55, "UNMET": 0.45},
                    confidence=0.1,
                    superseded=True,
                ),
                JudgeVote(judge_id="llm", verdict=UNMET, reason="no"),
            ],
            escalated=True,
        )
        mc = EnsembleCriterionReport(
            criterion=ORDINAL,
            final_verdict=None,
            final_reason=None,
            final_multi_choice_verdict=_final_mc(1),
            multi_choice_votes=[
                _mc_vote("jev", 0, reason=None, superseded=True, confidence=0.05),
                _mc_vote("a", 1, reason=None),
                _mc_vote("b", 0),
            ],
            escalated=True,
        )
        for ecr in (binary, mc):
            restored = EnsembleCriterionReport.model_validate(ecr.model_dump(mode=mode))
            assert restored == ecr
            again = EnsembleCriterionReport.model_validate(restored.model_dump(mode=mode))
            assert again == ecr
        assert mc.agreement == pytest.approx(0.5)


class TestReasonJoins:
    def test_binary_join_skips_none_by_identity(self):
        grader = _grader()
        votes = [
            JudgeVote(judge_id="jev", verdict=MET, reason=None),
            JudgeVote(judge_id="gemini", verdict=MET, reason=""),
            JudgeVote(judge_id="claude", verdict=MET, reason="looks right"),
        ]
        _, reason = grader._aggregate_votes(votes, weight=1.0)
        # An empty LLM explanation still renders as "gemini: ", exactly as before.
        assert reason == "gemini:  | claude: looks right"

    def test_binary_join_is_none_when_no_vote_has_a_reason(self):
        grader = _grader()
        votes = [JudgeVote(judge_id="jev", verdict=MET, reason=None)]
        verdict, reason = grader._aggregate_votes(votes, weight=1.0)
        assert verdict == MET
        assert reason is None

    def test_binary_join_unchanged_for_string_reasons(self):
        grader = _grader()
        votes = [
            JudgeVote(judge_id="a", verdict=MET, reason="yes"),
            JudgeVote(judge_id="b", verdict=UNMET, reason=""),
        ]
        _, reason = grader._aggregate_votes(votes, weight=1.0)
        assert reason == "a: yes | b: "

    def test_binary_synthetic_reasons_unchanged(self):
        grader = _grader()
        assert grader._aggregate_votes([], weight=1.0) == (CA, "No votes")
        votes = [JudgeVote(judge_id="jev", verdict=CA, reason=None)]
        assert grader._aggregate_votes(votes, weight=1.0) == (CA, "All judges could not assess")

    def test_multi_choice_join_skips_none(self):
        grader = _grader()
        report = CriterionReport(
            weight=1.0,
            requirement="How good?",
            options=OPTIONS,
            scale_type="ordinal",
            reason="",
        )
        votes = [_mc_vote("jev", 1, reason=None), _mc_vote("gemini", 1, reason="")]
        _, reason = grader._aggregate_multi_choice_votes(votes, report)
        assert reason == "gemini: "
        _, none_reason = grader._aggregate_multi_choice_votes(
            [_mc_vote("jev", 1, reason=None)], report
        )
        assert none_reason is None

    def test_multi_choice_all_na_join_skips_none(self):
        grader = _grader()
        report = CriterionReport(
            weight=1.0,
            requirement="How good?",
            options=OPTIONS,
            scale_type="ordinal",
            reason="",
        )
        votes = [_mc_vote("jev", 2, reason=None), _mc_vote("gemini", 2, reason="no data")]
        verdict, reason = grader._aggregate_multi_choice_votes(votes, report)
        assert verdict.na is True
        assert reason == "gemini: no data"
        verdict, reason = grader._aggregate_multi_choice_votes(
            [_mc_vote("jev", 2, reason=None)], report
        )
        assert verdict.na is True
        assert reason is None


def _binary_result(verdict, reason, **kwargs) -> CriterionResult:
    return CriterionResult(
        report=CriterionReport(
            weight=BINARY.weight,
            requirement=BINARY.requirement,
            name=BINARY.name,
            verdict=verdict,
            reason=reason,
            **kwargs,
        )
    )


def _mc_result(idx, reason, **kwargs) -> CriterionResult:
    opt = OPTIONS[idx]
    return CriterionResult(
        report=CriterionReport(
            weight=ORDINAL.weight,
            requirement=ORDINAL.requirement,
            name=ORDINAL.name,
            options=OPTIONS,
            scale_type="ordinal",
            multi_choice_verdict=MultiChoiceVerdict(
                selected_index=idx, selected_label=opt.label, value=opt.value, na=opt.na
            ),
            reason=reason,
            **kwargs,
        )
    )


class TestAggregateCarriesNewFields:
    @pytest.mark.asyncio
    async def test_probabilities_confidence_and_none_reason_reach_votes(self):
        grader = _grader()
        judge_results = [
            JudgeCriterionResults(
                judge_id="jev",
                weight=1.0,
                criterion_results=[
                    _binary_result(
                        MET, None, probabilities={"MET": 0.9, "UNMET": 0.1}, confidence=0.8
                    ),
                    _mc_result(
                        1, None, probabilities={"0": 0.3, "1": 0.7, "2": 0.0}, confidence=0.55
                    ),
                ],
            ),
        ]
        report = await grader.aggregate(judge_results)
        assert report.report is not None
        binary, mc = report.report
        assert binary.votes[0].probabilities == {"MET": 0.9, "UNMET": 0.1}
        assert binary.votes[0].confidence == pytest.approx(0.8)
        assert binary.votes[0].reason is None
        assert binary.votes[0].superseded is False
        assert binary.final_reason is None
        assert binary.escalated is False
        assert mc.multi_choice_votes[0].probabilities == {"0": 0.3, "1": 0.7, "2": 0.0}
        assert mc.multi_choice_votes[0].confidence == pytest.approx(0.55)
        assert mc.final_reason is None
        assert report.judge_scores == {"jev": pytest.approx(1.0)}
        assert report.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_mixed_reasons_join_only_existing_ones(self):
        grader = _grader()
        judge_results = [
            JudgeCriterionResults(
                judge_id="jev", weight=1.0, criterion_results=[_binary_result(MET, None)]
            ),
            JudgeCriterionResults(
                judge_id="gemini", weight=1.0, criterion_results=[_binary_result(MET, "")]
            ),
        ]
        report = await grader.aggregate(judge_results)
        assert report.report is not None
        assert report.report[0].final_reason == "gemini: "

    @pytest.mark.asyncio
    async def test_llm_votes_keep_llm_shape(self):
        grader = _grader()
        judge_results = [
            JudgeCriterionResults(
                judge_id="gemini", weight=1.0, criterion_results=[_binary_result(MET, "ok")]
            ),
        ]
        report = await grader.aggregate(judge_results)
        assert report.report is not None
        vote = report.report[0].votes[0]
        assert (vote.probabilities, vote.confidence, vote.superseded) == (None, None, False)
        assert report.report[0].final_reason == "gemini: ok"
