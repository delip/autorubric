"""Tests for the rubric improvement engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric, TokenUsage
from autorubric.dataset import DataItem, RubricDataset
from autorubric.graders import CriterionGrader
from autorubric.graders.criterion_grader import JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.meta._improve import (
    ImprovementConfig,
    IssueDetail,
    IterationResult,
    _ConvergenceState,
    _build_revision_history,
    _check_convergence,
    _diff_issues,
    _extract_issues,
    _format_agreement_for_prompt,
    _format_error_criteria,
    _format_issues_for_prompt,
    _match_issue_to_criteria,
    _pareto_accept,
    _select_diagnostic_items,
    _serialize_iteration,
    compute_expected_scores,
    format_ground_truth_for_prompt,
    improve_rubric,
    validate_agreement,
    validate_ground_truth,
)
from autorubric.types import (
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_criterion(
    name: str, weight: float = 1.0, requirement: str | None = None
) -> Criterion:
    return Criterion(
        name=name,
        weight=weight,
        requirement=requirement or f"Requirement for {name}",
    )


def _make_ensemble_criterion_report(
    name: str,
    weight: float,
    verdict: CriterionVerdict,
    reason: str = "test reason",
    agreement: float = 1.0,
) -> EnsembleCriterionReport:
    criterion = _make_criterion(name, weight)
    return EnsembleCriterionReport(
        criterion=criterion,
        final_verdict=verdict,
        final_reason=reason,
        votes=[
            JudgeVote(
                judge_id="judge_0",
                verdict=verdict,
                reason=reason,
            )
        ],
        agreement=agreement,
    )


def _make_ensemble_report(
    criterion_reports: list[EnsembleCriterionReport],
    score: float = 0.8,
    mean_agreement: float = 0.9,
) -> EnsembleEvaluationReport:
    return EnsembleEvaluationReport(
        score=score,
        raw_score=score,
        report=criterion_reports,
        mean_agreement=mean_agreement,
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        completion_cost=0.01,
    )


def _make_issue(
    name: str, is_antipattern: bool = False, weight: float = 1.0
) -> IssueDetail:
    return IssueDetail(
        criterion_name=name,
        requirement=f"Requirement for {name}",
        weight=-abs(weight) if is_antipattern else abs(weight),
        is_antipattern=is_antipattern,
        feedback=f"Feedback for {name}",
    )


def _make_iteration_result(
    iteration: int,
    quality_score: float = 0.8,
    agreement: float | None = 0.9,
    issues: list[IssueDetail] | None = None,
    issues_fixed: list[str] | None = None,
    issues_introduced: list[str] | None = None,
    accepted: bool = True,
    rejection_reason: str | None = None,
) -> IterationResult:
    rubric = Rubric([_make_criterion("test_criterion")])
    report = _make_ensemble_report(
        [
            _make_ensemble_criterion_report(
                "test_criterion", 1.0, CriterionVerdict.MET
            )
        ]
    )
    return IterationResult(
        iteration=iteration,
        rubric=rubric,
        quality_score=quality_score,
        agreement=agreement,
        per_criterion_agreement={"test_criterion": agreement} if agreement else None,
        issues=issues or [],
        issues_fixed=issues_fixed or [],
        issues_introduced=issues_introduced or [],
        accepted=accepted,
        rejection_reason=rejection_reason,
        quality_report=report,
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        completion_cost=0.01,
    )


# ============================================================================
# _extract_issues
# ============================================================================


class TestExtractIssues:
    def test_positive_unmet_is_issue(self):
        report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "clarity", 1.0, CriterionVerdict.UNMET, reason="Not clear"
                )
            ]
        )
        issues = _extract_issues(report)
        assert len(issues) == 1
        assert issues[0].criterion_name == "clarity"
        assert issues[0].is_antipattern is False
        assert issues[0].feedback == "Not clear"

    def test_negative_met_is_antipattern_issue(self):
        report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "overlap", -1.0, CriterionVerdict.MET, reason="Overlap detected"
                )
            ]
        )
        issues = _extract_issues(report)
        assert len(issues) == 1
        assert issues[0].criterion_name == "overlap"
        assert issues[0].is_antipattern is True

    def test_positive_met_not_an_issue(self):
        report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "clarity", 1.0, CriterionVerdict.MET
                )
            ]
        )
        issues = _extract_issues(report)
        assert len(issues) == 0

    def test_negative_unmet_not_an_issue(self):
        report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "overlap", -1.0, CriterionVerdict.UNMET
                )
            ]
        )
        issues = _extract_issues(report)
        assert len(issues) == 0

    def test_empty_report_returns_empty(self):
        report = EnsembleEvaluationReport(score=1.0, report=None)
        issues = _extract_issues(report)
        assert issues == []

    def test_mixed_criteria(self):
        report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report("good", 1.0, CriterionVerdict.MET),
                _make_ensemble_criterion_report("bad", 1.0, CriterionVerdict.UNMET),
                _make_ensemble_criterion_report("anti_ok", -1.0, CriterionVerdict.UNMET),
                _make_ensemble_criterion_report("anti_bad", -1.0, CriterionVerdict.MET),
            ]
        )
        issues = _extract_issues(report)
        assert len(issues) == 2
        names = {i.criterion_name for i in issues}
        assert names == {"bad", "anti_bad"}


# ============================================================================
# _diff_issues
# ============================================================================


class TestDiffIssues:
    def test_fixed_issues(self):
        prev = [_make_issue("a"), _make_issue("b")]
        curr = [_make_issue("b")]
        fixed, introduced = _diff_issues(prev, curr)
        assert fixed == ["a"]
        assert introduced == []

    def test_introduced_issues(self):
        prev = [_make_issue("a")]
        curr = [_make_issue("a"), _make_issue("c")]
        fixed, introduced = _diff_issues(prev, curr)
        assert fixed == []
        assert introduced == ["c"]

    def test_both_empty(self):
        fixed, introduced = _diff_issues([], [])
        assert fixed == []
        assert introduced == []

    def test_identical_sets(self):
        issues = [_make_issue("x"), _make_issue("y")]
        fixed, introduced = _diff_issues(issues, issues)
        assert fixed == []
        assert introduced == []

    def test_complete_turnover(self):
        prev = [_make_issue("a"), _make_issue("b")]
        curr = [_make_issue("c"), _make_issue("d")]
        fixed, introduced = _diff_issues(prev, curr)
        assert fixed == ["a", "b"]
        assert introduced == ["c", "d"]


# ============================================================================
# _match_issue_to_criteria
# ============================================================================


class TestMatchIssueToCriteria:
    def _rubric(self) -> Rubric:
        return Rubric([
            Criterion(name="clarity", weight=1.0, requirement="The rubric must be written clearly"),
            Criterion(name="coverage", weight=1.0, requirement="All aspects should be covered"),
            Criterion(name="overlap", weight=-1.0, requirement="Short req"),
        ])

    def test_affects_tag_parsed(self):
        """The primary path: [Affects: #1, #3] tag is parsed correctly."""
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="Some feedback text [Affects: #1, #3]",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert result == [1, 3]

    def test_affects_tag_all(self):
        """[Affects: all] returns all criterion indices."""
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="Rubric-wide issue [Affects: all]",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert result == [1, 2, 3]

    def test_affects_tag_single(self):
        """[Affects: #2] returns just that index."""
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="Problem with criterion [Affects: #2]",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert result == [2]

    def test_name_match(self):
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="The clarity criterion is poorly written",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert 1 in result

    def test_number_pattern_match(self):
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="Criterion #2 has vague wording",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert result == [2]

    def test_criterion_word_number_match(self):
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="criterion 1 and criterion 3 overlap",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert 1 in result
        assert 3 in result

    def test_rubric_wide_no_match(self):
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="The rubric as a whole is too long",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert result == []

    def test_multiple_matches(self):
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="Both clarity and coverage are vague",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert result == [1, 2]

    def test_requirement_substring_match(self):
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="'the rubric must be written clearly' is too prescriptive",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert 1 in result

    def test_short_requirement_not_matched_by_substring(self):
        """Requirements shorter than 20 chars are not matched by substring heuristic."""
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="Short req is mentioned but should not match by substring",
        )
        result = _match_issue_to_criteria(issue, rubric)
        # "overlap" has name not in feedback and "Short req" is <20 chars
        assert 3 not in result

    def test_out_of_range_number_ignored(self):
        rubric = self._rubric()
        issue = IssueDetail(
            criterion_name="meta_crit",
            requirement="req",
            weight=1.0,
            is_antipattern=False,
            feedback="Criterion #99 is problematic",
        )
        result = _match_issue_to_criteria(issue, rubric)
        assert result == []


# ============================================================================
# MetaCriterionJudgment structured output
# ============================================================================


class TestMetaCriterionJudgment:
    def test_extends_criterion_judgment(self):
        from autorubric.meta._evaluate import MetaCriterionJudgment

        judgment = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.UNMET,
            explanation="Criterion #1 is vague",
            affected_criteria=[1, 3],
        )
        assert judgment.criterion_status == CriterionVerdict.UNMET
        assert judgment.explanation == "Criterion #1 is vague"
        assert judgment.affected_criteria == [1, 3]

    def test_defaults_to_empty_list(self):
        from autorubric.meta._evaluate import MetaCriterionJudgment

        judgment = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="All good",
        )
        assert judgment.affected_criteria == []

    @pytest.mark.asyncio
    async def test_grader_injects_affects_tag(self):
        """When binary_response_format returns affected_criteria, the tag is injected."""
        from autorubric.meta._evaluate import MetaCriterionJudgment

        judgment = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.UNMET,
            explanation="Criterion is unclear",
            affected_criteria=[1, 3],
        )
        gen_result = GenerateResult(
            content="",
            cost=0.01,
            parsed=judgment,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        generate_mock = AsyncMock(return_value=gen_result)

        grader = CriterionGrader(
            llm_config=LLMConfig(model="test-model"),
            binary_response_format=MetaCriterionJudgment,
        )

        with patch.object(
            list(grader._clients.values())[0], "generate", generate_mock
        ):
            rubric = Rubric([
                Criterion(name="clarity", weight=1.0, requirement="Must be clear"),
            ])
            result = await rubric.grade(to_grade="test submission", grader=grader)

        assert result.report is not None
        reason = result.report[0].final_reason
        assert "[Affects: #1, #3]" in reason

    @pytest.mark.asyncio
    async def test_grader_no_tag_when_empty_affected(self):
        """When affected_criteria is empty, no tag is injected."""
        from autorubric.meta._evaluate import MetaCriterionJudgment

        judgment = MetaCriterionJudgment(
            criterion_status=CriterionVerdict.MET,
            explanation="All good",
            affected_criteria=[],
        )
        gen_result = GenerateResult(
            content="",
            cost=0.01,
            parsed=judgment,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        generate_mock = AsyncMock(return_value=gen_result)

        grader = CriterionGrader(
            llm_config=LLMConfig(model="test-model"),
            binary_response_format=MetaCriterionJudgment,
        )

        with patch.object(
            list(grader._clients.values())[0], "generate", generate_mock
        ):
            rubric = Rubric([
                Criterion(name="clarity", weight=1.0, requirement="Must be clear"),
            ])
            result = await rubric.grade(to_grade="test submission", grader=grader)

        assert result.report is not None
        reason = result.report[0].final_reason
        assert "[Affects:" not in reason
        assert "All good" in reason

    @pytest.mark.asyncio
    async def test_grader_default_format_has_no_affected(self):
        """Default CriterionJudgment has no affected_criteria, so no tag."""
        from autorubric.types import CriterionJudgment

        judgment = CriterionJudgment(
            criterion_status=CriterionVerdict.UNMET,
            explanation="Not good",
        )
        gen_result = GenerateResult(
            content="",
            cost=0.01,
            parsed=judgment,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        generate_mock = AsyncMock(return_value=gen_result)

        grader = CriterionGrader(llm_config=LLMConfig(model="test-model"))

        with patch.object(
            list(grader._clients.values())[0], "generate", generate_mock
        ):
            rubric = Rubric([
                Criterion(name="clarity", weight=1.0, requirement="Must be clear"),
            ])
            result = await rubric.grade(to_grade="test submission", grader=grader)

        assert result.report is not None
        reason = result.report[0].final_reason
        assert "[Affects:" not in reason
        assert "Not good" in reason


# ============================================================================
# _format_issues_for_prompt
# ============================================================================


class TestFormatIssuesForPrompt:
    def test_empty_returns_no_issues(self):
        assert _format_issues_for_prompt([]) == "No issues found."

    def test_mixed_issue_types(self):
        issues = [
            _make_issue("clarity", is_antipattern=False),
            _make_issue("overlap", is_antipattern=True),
        ]
        result = _format_issues_for_prompt(issues)
        assert "[QUALITY GAP] clarity" in result
        assert "[ANTI-PATTERN DETECTED] overlap" in result
        assert "Feedback for clarity" in result
        assert "Feedback for overlap" in result

    def test_numbering(self):
        issues = [_make_issue("a"), _make_issue("b"), _make_issue("c")]
        result = _format_issues_for_prompt(issues)
        assert "1. [QUALITY GAP] a" in result
        assert "2. [QUALITY GAP] b" in result
        assert "3. [QUALITY GAP] c" in result


# ============================================================================
# _format_agreement_for_prompt
# ============================================================================


class TestFormatAgreementForPrompt:
    def test_none_returns_empty(self):
        assert _format_agreement_for_prompt(None) == ""

    def test_empty_dict_returns_empty(self):
        assert _format_agreement_for_prompt({}) == ""

    def test_high_medium_low_annotations(self):
        data = {
            "criterion_high": 0.85,
            "criterion_medium": 0.65,
            "criterion_low": 0.40,
        }
        result = _format_agreement_for_prompt(data)
        assert result.startswith("## Inter-Judge Agreement")
        assert "(HIGH)" in result
        assert "(MEDIUM)" in result
        assert "(LOW)" in result
        assert "criterion_high" in result
        assert "criterion_medium" in result
        assert "criterion_low" in result

    def test_boundary_values(self):
        data = {"exactly_80": 0.8, "exactly_60": 0.6, "below_60": 0.59}
        result = _format_agreement_for_prompt(data)
        assert "exactly_80: 80% (HIGH)" in result
        assert "exactly_60: 60% (MEDIUM)" in result
        assert "below_60: 59% (LOW)" in result


# ============================================================================
# _build_revision_history
# ============================================================================


class TestBuildRevisionHistory:
    def test_empty_iterations(self):
        assert _build_revision_history([], window=3) == "No previous iterations."

    def test_formats_iterations(self):
        iterations = [
            _make_iteration_result(0, quality_score=0.6, agreement=0.7, issues=[_make_issue("a")]),
            _make_iteration_result(1, quality_score=0.8, agreement=0.85),
        ]
        result = _build_revision_history(iterations, window=5)
        assert "Iteration 0:" in result
        assert "Quality: 60.0%" in result
        assert "Agreement: 70%" in result
        assert "Iteration 1:" in result
        assert "Quality: 80.0%" in result

    def test_window_limit_respected(self):
        iterations = [
            _make_iteration_result(i, quality_score=0.5 + i * 0.1)
            for i in range(5)
        ]
        result = _build_revision_history(iterations, window=2)
        assert "Iteration 3:" in result
        assert "Iteration 4:" in result
        assert "Iteration 0:" not in result
        assert "Iteration 1:" not in result
        assert "Iteration 2:" not in result

    def test_shows_fixed_and_introduced(self):
        iteration = _make_iteration_result(
            0,
            issues_fixed=["fixed_a"],
            issues_introduced=["new_b"],
        )
        result = _build_revision_history([iteration], window=3)
        assert "Fixed: fixed_a" in result
        assert "Introduced: new_b" in result

    def test_shows_rejected(self):
        iteration = _make_iteration_result(
            0, accepted=False, rejection_reason="Agreement regressed"
        )
        result = _build_revision_history([iteration], window=3)
        assert "REJECTED: Agreement regressed" in result

    def test_none_agreement(self):
        iteration = _make_iteration_result(0, agreement=None)
        result = _build_revision_history([iteration], window=3)
        assert "Agreement: N/A" in result


# ============================================================================
# _pareto_accept
# ============================================================================


class TestParetoAccept:
    def test_accepts_when_agreement_improves(self):
        accepted, reason = _pareto_accept(0.9, 0.8, True, 0)
        assert accepted is True
        assert reason is None

    def test_rejects_when_agreement_regresses(self):
        accepted, reason = _pareto_accept(0.7, 0.8, True, 0)
        assert accepted is False
        assert "regressed" in reason.lower()

    def test_accepts_within_epsilon(self):
        accepted, reason = _pareto_accept(0.78, 0.8, True, 0, epsilon=0.03)
        assert accepted is True
        assert reason is None

    def test_rejects_beyond_epsilon(self):
        accepted, reason = _pareto_accept(0.76, 0.8, True, 0, epsilon=0.03)
        assert accepted is False

    def test_accepts_after_2_consecutive_rejections(self):
        accepted, reason = _pareto_accept(0.5, 0.9, True, 2)
        assert accepted is True
        assert reason is None

    def test_accepts_when_curr_agreement_none(self):
        accepted, reason = _pareto_accept(None, 0.8, True, 0)
        assert accepted is True

    def test_accepts_when_prev_agreement_none(self):
        accepted, reason = _pareto_accept(0.8, None, True, 0)
        assert accepted is True

    def test_accepts_when_reject_regression_false(self):
        accepted, reason = _pareto_accept(0.5, 0.9, False, 0)
        assert accepted is True
        assert reason is None


# ============================================================================
# _check_convergence
# ============================================================================


class TestCheckConvergence:
    def _default_config(self, **overrides) -> ImprovementConfig:
        cfg = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),
            max_iterations=10,
            min_quality_score=0.95,
            min_agreement=0.85,
            score_plateau_threshold=0.02,
            plateau_patience=2,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_no_issues_converges(self):
        config = self._default_config()
        state = _ConvergenceState()
        result = _check_convergence(0, [], 0.5, 0.5, config, state, 0.0)
        assert result == "no_issues"

    def test_thresholds_met(self):
        config = self._default_config()
        state = _ConvergenceState()
        issues = [_make_issue("x")]
        result = _check_convergence(0, issues, 0.96, 0.90, config, state, 0.0)
        assert result == "thresholds_met"

    def test_thresholds_met_without_validation_data(self):
        config = self._default_config()
        state = _ConvergenceState()
        issues = [_make_issue("x")]
        # When validation_data is None, agreement threshold is auto-satisfied
        result = _check_convergence(0, issues, 0.96, None, config, state, 0.0)
        assert result == "thresholds_met"

    def test_max_iterations(self):
        config = self._default_config(max_iterations=5)
        state = _ConvergenceState()
        issues = [_make_issue("x")]
        result = _check_convergence(4, issues, 0.5, 0.5, config, state, 0.0)
        assert result == "max_iterations"

    def test_score_plateau(self):
        config = self._default_config(plateau_patience=2, score_plateau_threshold=0.02)
        state = _ConvergenceState(prev_score=0.80)
        issues = [_make_issue("x")]

        # First plateau tick: improvement < threshold
        result = _check_convergence(0, issues, 0.81, 0.5, config, state, 0.0)
        assert result is None
        assert state.plateau_count == 1

        # Second plateau tick triggers convergence
        result = _check_convergence(1, issues, 0.82, 0.5, config, state, 0.0)
        assert result == "score_plateau"

    def test_plateau_resets_on_improvement(self):
        config = self._default_config(plateau_patience=2, score_plateau_threshold=0.02)
        state = _ConvergenceState(prev_score=0.80, plateau_count=1)
        issues = [_make_issue("x")]

        # Big improvement resets plateau counter
        result = _check_convergence(0, issues, 0.85, 0.5, config, state, 0.0)
        assert result is None
        assert state.plateau_count == 0

    def test_cost_limit(self):
        config = self._default_config(max_total_cost=1.0)
        state = _ConvergenceState()
        issues = [_make_issue("x")]
        result = _check_convergence(0, issues, 0.5, 0.5, config, state, 1.5)
        assert result == "cost_limit"

    def test_continues_when_no_stopping_condition(self):
        config = self._default_config()
        state = _ConvergenceState()
        issues = [_make_issue("x")]
        result = _check_convergence(0, issues, 0.5, 0.5, config, state, 0.0)
        assert result is None


# ============================================================================
# improve_rubric (integration test with mocks)
# ============================================================================


class TestImproveRubricIntegration:
    @pytest.mark.asyncio
    async def test_converges_on_no_issues(self):
        """When the initial evaluation finds no issues, the loop stops immediately."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
            Criterion(name="accuracy", weight=1.0, requirement="Is accurate"),
        ])

        quality_report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report("clarity", 1.0, CriterionVerdict.MET),
                _make_ensemble_criterion_report("accuracy", 1.0, CriterionVerdict.MET),
            ],
            score=1.0,
            mean_agreement=1.0,
        )

        eval_llm = LLMConfig(model="test-model")
        revision_llm = LLMConfig(model="test-model")

        with patch(
            "autorubric.meta._improve.evaluate_rubric_in_context",
            new_callable=AsyncMock,
            return_value=quality_report,
        ):
            result = await improve_rubric(
                rubric,
                "Write a summary",
                eval_llm=eval_llm,
                revision_llm=revision_llm,
                max_iterations=5,
            )

            assert result.convergence_reason == "no_issues"
            assert len(result.iterations) == 1
            assert result.best_rubric is rubric
            assert result.original_rubric is rubric

    @pytest.mark.asyncio
    async def test_revision_loop_runs_and_converges(self):
        """The loop revises the rubric and converges when thresholds are met."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        # First eval: has issues. Second eval: no issues.
        report_with_issues = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "clarity", 1.0, CriterionVerdict.UNMET, reason="Not clear"
                ),
            ],
            score=0.5,
            mean_agreement=0.9,
        )
        report_clean = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "clarity", 1.0, CriterionVerdict.MET, reason="Clear now"
                ),
            ],
            score=1.0,
            mean_agreement=1.0,
        )

        eval_mock = AsyncMock(side_effect=[report_with_issues, report_clean])

        revised_criteria_json = json.dumps([
            {"name": "clarity", "weight": 1.0, "requirement": "Is clear and concise"}
        ])

        generate_result = GenerateResult(
            content=revised_criteria_json, cost=0.005
        )
        generate_mock = AsyncMock(return_value=generate_result)

        eval_llm = LLMConfig(model="test-model")
        revision_llm = LLMConfig(model="test-model")

        config = ImprovementConfig(
            eval_llm=eval_llm,
            revision_llm=revision_llm,
            save_artifacts=False,
            max_iterations=5,
        )

        with (
            patch(
                "autorubric.meta._improve.evaluate_rubric_in_context",
                eval_mock,
            ),
            patch(
                "autorubric.meta._improve.LLMClient",
            ) as mock_llm_client_cls,
        ):
            mock_client_instance = MagicMock()
            mock_client_instance.generate = generate_mock
            mock_llm_client_cls.return_value = mock_client_instance

            result = await improve_rubric(
                rubric,
                "Write a summary",
                config=config,
            )

            assert result.convergence_reason == "no_issues"
            assert len(result.iterations) == 2
            assert result.iterations[0].quality_score == 0.5
            assert result.iterations[1].quality_score == 1.0

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        """The loop stops after max_iterations even if issues remain."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        report_with_issues = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "clarity", 1.0, CriterionVerdict.UNMET, reason="Not clear"
                ),
            ],
            score=0.5,
            mean_agreement=0.9,
        )

        eval_mock = AsyncMock(return_value=report_with_issues)

        revised_criteria_json = json.dumps([
            {"name": "clarity", "weight": 1.0, "requirement": "Is clear v2"}
        ])
        generate_result = GenerateResult(
            content=revised_criteria_json, cost=0.005
        )
        generate_mock = AsyncMock(return_value=generate_result)

        eval_llm = LLMConfig(model="test-model")
        revision_llm = LLMConfig(model="test-model")

        config = ImprovementConfig(
            eval_llm=eval_llm,
            revision_llm=revision_llm,
            save_artifacts=False,
            max_iterations=3,
            score_plateau_threshold=0.0,
        )

        with (
            patch(
                "autorubric.meta._improve.evaluate_rubric_in_context",
                eval_mock,
            ),
            patch(
                "autorubric.meta._improve.LLMClient",
            ) as mock_llm_client_cls,
        ):
            mock_client_instance = MagicMock()
            mock_client_instance.generate = generate_mock
            mock_llm_client_cls.return_value = mock_client_instance

            result = await improve_rubric(
                rubric,
                "Write a summary",
                config=config,
            )

            assert result.convergence_reason == "max_iterations"
            assert len(result.iterations) == 3

    @pytest.mark.asyncio
    async def test_requires_config_or_llm_params(self):
        """Raises ValueError when neither config nor llm params are provided."""
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        with pytest.raises(ValueError, match="Either config or both"):
            await improve_rubric(rubric, "prompt")

    @pytest.mark.asyncio
    async def test_requires_task_prompt_for_in_context(self):
        """Raises ValueError when task_prompt is None in in_context mode."""
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            mode="in_context",
        )
        with pytest.raises(ValueError, match="task_prompt is required"):
            await improve_rubric(rubric, task_prompt=None, config=config)


# ============================================================================
# Public function names and backward-compat aliases
# ============================================================================


class TestPublicFunctionNames:
    """Verify that building blocks are importable by their public names."""

    def test_public_names_importable(self):
        from autorubric.meta._improve import (
            build_revision_history,
            compute_expected_scores,
            diff_issues,
            extract_issues,
            format_agreement_for_prompt,
            format_ground_truth_for_prompt,
            format_issues_for_prompt,
            pareto_accept,
            revise_rubric,
            validate_agreement,
            validate_ground_truth,
        )
        for fn in [
            extract_issues, diff_issues, format_issues_for_prompt,
            format_agreement_for_prompt, build_revision_history,
            validate_agreement, pareto_accept, revise_rubric,
            compute_expected_scores, validate_ground_truth,
            format_ground_truth_for_prompt,
        ]:
            assert callable(fn)

    def test_private_aliases_match_public(self):
        from autorubric.meta._improve import (
            _build_revision_history,
            _diff_issues,
            _extract_issues,
            _format_agreement_for_prompt,
            _format_issues_for_prompt,
            _pareto_accept,
            _revise_rubric,
            _validate_agreement,
            build_revision_history,
            diff_issues,
            extract_issues,
            format_agreement_for_prompt,
            format_issues_for_prompt,
            pareto_accept,
            revise_rubric,
            validate_agreement,
        )
        assert _extract_issues is extract_issues
        assert _diff_issues is diff_issues
        assert _format_issues_for_prompt is format_issues_for_prompt
        assert _format_agreement_for_prompt is format_agreement_for_prompt
        assert _build_revision_history is build_revision_history
        assert _validate_agreement is validate_agreement
        assert _pareto_accept is pareto_accept
        assert _revise_rubric is revise_rubric

    def test_meta_package_exports(self):
        from autorubric.meta import (
            ConvergenceFn,
            ImprovementProgressDisplay,
            ImprovementRunner,
            build_revision_history,
            compute_expected_scores,
            diff_issues,
            extract_issues,
            format_agreement_for_prompt,
            format_ground_truth_for_prompt,
            format_issues_for_prompt,
            pareto_accept,
            revise_rubric,
            validate_agreement,
            validate_ground_truth,
        )
        assert ImprovementRunner is not None
        assert ImprovementProgressDisplay is not None
        assert ConvergenceFn is not None
        for fn in [
            extract_issues, diff_issues, format_issues_for_prompt,
            format_agreement_for_prompt, build_revision_history,
            validate_agreement, pareto_accept, revise_rubric,
            compute_expected_scores, validate_ground_truth,
            format_ground_truth_for_prompt,
        ]:
            assert callable(fn)


# ============================================================================
# ImprovementRunner
# ============================================================================


class TestImprovementRunner:
    def test_requires_config(self):
        from autorubric.meta._improve import ImprovementRunner
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        with pytest.raises(ValueError, match="config is required"):
            ImprovementRunner(rubric, "prompt")

    @pytest.mark.asyncio
    async def test_run_converges_on_no_issues(self):
        from autorubric.meta._improve import ImprovementRunner

        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        quality_report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report("clarity", 1.0, CriterionVerdict.MET),
            ],
            score=1.0,
            mean_agreement=1.0,
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),
            save_artifacts=False,
            show_progress=False,
            max_iterations=5,
        )

        with patch(
            "autorubric.meta._improve.evaluate_rubric_in_context",
            new_callable=AsyncMock,
            return_value=quality_report,
        ):
            runner = ImprovementRunner(rubric, "Write a summary", config=config)
            result = await runner.run()

            assert result.convergence_reason == "no_issues"
            assert len(result.iterations) == 1
            assert result.best_rubric is rubric

    @pytest.mark.asyncio
    async def test_requires_task_prompt_for_in_context(self):
        from autorubric.meta._improve import ImprovementRunner

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            mode="in_context",
        )
        runner = ImprovementRunner(rubric, task_prompt=None, config=config)
        with pytest.raises(ValueError, match="task_prompt is required"):
            await runner.run()


# ============================================================================
# _build_config
# ============================================================================


class TestBuildConfig:
    def test_creates_config_from_kwargs(self):
        from autorubric.meta._improve import _build_config

        eval_llm = LLMConfig(model="eval-model")
        revision_llm = LLMConfig(model="revision-model")
        config = _build_config(
            None,
            eval_llm=eval_llm,
            revision_llm=revision_llm,
            max_iterations=5,
        )
        assert config.eval_llm is eval_llm
        assert config.revision_llm is revision_llm
        assert config.max_iterations == 5

    def test_overrides_base_config(self):
        from autorubric.meta._improve import _build_config

        base = ImprovementConfig(
            eval_llm=LLMConfig(model="base-eval"),
            revision_llm=LLMConfig(model="base-revision"),
            max_iterations=10,
        )
        new_eval = LLMConfig(model="new-eval")
        config = _build_config(base, eval_llm=new_eval, max_iterations=3)
        assert config.eval_llm is new_eval
        assert config.revision_llm is base.revision_llm
        assert config.max_iterations == 3

    def test_returns_base_config_when_no_overrides(self):
        from autorubric.meta._improve import _build_config

        base = ImprovementConfig(
            eval_llm=LLMConfig(model="eval"),
            revision_llm=LLMConfig(model="revision"),
        )
        config = _build_config(base)
        assert config is base

    def test_error_when_no_config_and_no_llms(self):
        from autorubric.meta._improve import _build_config

        with pytest.raises(ValueError, match="Either config or both"):
            _build_config(None, max_iterations=5)

    def test_artifacts_dir_override(self):
        from autorubric.meta._improve import _build_config

        base = ImprovementConfig(
            eval_llm=LLMConfig(model="eval"),
            revision_llm=LLMConfig(model="revision"),
        )
        config = _build_config(base, artifacts_dir="/tmp/test")
        assert config.artifacts_dir == "/tmp/test"


# ============================================================================
# Custom convergence_fn
# ============================================================================


class TestCustomConvergenceFn:
    @pytest.mark.asyncio
    async def test_custom_convergence_stops_loop(self):
        """Custom convergence_fn is called and can stop the loop."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        report_with_issues = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "clarity", 1.0, CriterionVerdict.UNMET, reason="Not clear"
                ),
            ],
            score=0.5,
        )

        eval_mock = AsyncMock(return_value=report_with_issues)

        revised_criteria_json = json.dumps([
            {"name": "clarity", "weight": 1.0, "requirement": "Is clear v2"}
        ])
        generate_result = GenerateResult(
            content=revised_criteria_json, cost=0.005
        )
        generate_mock = AsyncMock(return_value=generate_result)

        def stop_after_one(current, history):
            return "custom_stop" if len(history) >= 1 else None

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),
            save_artifacts=False,
            show_progress=False,
            max_iterations=10,
            convergence_fn=stop_after_one,
        )

        with (
            patch(
                "autorubric.meta._improve.evaluate_rubric_in_context",
                eval_mock,
            ),
            patch(
                "autorubric.meta._improve.LLMClient",
            ) as mock_llm_client_cls,
        ):
            mock_client_instance = MagicMock()
            mock_client_instance.generate = generate_mock
            mock_llm_client_cls.return_value = mock_client_instance

            result = await improve_rubric(
                rubric,
                "Write a summary",
                config=config,
            )

            assert result.convergence_reason == "custom_stop"
            assert len(result.iterations) == 1

    @pytest.mark.asyncio
    async def test_none_convergence_fn_uses_builtin(self):
        """When convergence_fn is None, built-in logic is used."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        quality_report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report("clarity", 1.0, CriterionVerdict.MET),
            ],
            score=1.0,
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),
            save_artifacts=False,
            show_progress=False,
            convergence_fn=None,
        )

        with patch(
            "autorubric.meta._improve.evaluate_rubric_in_context",
            new_callable=AsyncMock,
            return_value=quality_report,
        ):
            result = await improve_rubric(
                rubric,
                "Write a summary",
                config=config,
            )

            assert result.convergence_reason == "no_issues"


# ============================================================================
# ImprovementProgressDisplay (new display methods)
# ============================================================================


class TestImprovementProgressDisplay:
    def test_log_issues_table_with_issues(self):
        from io import StringIO

        from rich.console import Console

        from autorubric.meta._improve import ImprovementProgressDisplay

        display = ImprovementProgressDisplay()
        display._console = Console(file=StringIO(), force_terminal=True)

        rubric = Rubric([
            Criterion(name="clarity_criterion", weight=1.0, requirement="Must be clear"),
            Criterion(name="overlap_criterion", weight=-1.0, requirement="No overlapping criteria"),
        ])
        issues = [
            IssueDetail(
                criterion_name="meta_clarity",
                requirement="Rubric should be clear",
                weight=1.0,
                is_antipattern=False,
                feedback="Criterion clarity_criterion is unclear",
            ),
            IssueDetail(
                criterion_name="meta_overlap",
                requirement="No overlap",
                weight=-1.0,
                is_antipattern=True,
                feedback="Overlap detected in criterion #2",
            ),
        ]
        display.log_issues_table(issues, rubric=rubric)
        output = display._console.file.getvalue()
        assert "meta_clarity" in output
        assert "meta_overlap" in output
        assert "2" in output  # count in title
        assert "Rubric #" in output
        assert "#1" in output  # clarity_criterion matched
        assert "#2" in output  # explicit #2 pattern matched

    def test_log_issues_table_empty(self):
        from io import StringIO

        from rich.console import Console

        from autorubric.meta._improve import ImprovementProgressDisplay

        display = ImprovementProgressDisplay()
        display._console = Console(file=StringIO(), force_terminal=True)

        display.log_issues_table([])
        output = display._console.file.getvalue()
        assert "No issues found" in output

    def test_log_rubric(self):
        from io import StringIO

        from rich.console import Console

        from autorubric.meta._improve import ImprovementProgressDisplay

        display = ImprovementProgressDisplay()
        display._console = Console(file=StringIO(), force_terminal=True)

        rubric = Rubric([
            Criterion(name="a", weight=1.0, requirement="First criterion"),
            Criterion(name="b", weight=-0.5, requirement="Anti-pattern check"),
        ])
        display.log_rubric(rubric, iteration=0)
        output = display._console.file.getvalue()
        assert "First criterion" in output
        assert "Anti-pattern check" in output
        assert "Iteration 0" in output

    def test_log_rubric_diff(self):
        from io import StringIO

        from rich.console import Console

        from autorubric.meta._improve import ImprovementProgressDisplay

        display = ImprovementProgressDisplay()
        # Use no_color to avoid ANSI escapes splitting inline-highlighted text
        display._console = Console(file=StringIO(), force_terminal=True, no_color=True)

        prev = Rubric([
            Criterion(name="a", weight=1.0, requirement="Old requirement"),
            Criterion(name="b", weight=2.0, requirement="Unchanged"),
        ])
        curr = Rubric([
            Criterion(name="a", weight=1.0, requirement="New requirement"),
            Criterion(name="b", weight=2.0, requirement="Unchanged"),
        ])
        display.log_rubric_diff(prev, curr, iteration=1)
        output = display._console.file.getvalue()
        assert "Iteration 1" in output
        assert "Old requirement" in output
        assert "New requirement" in output
        # Verify paired before/after format
        assert "- 1." in output
        assert "+ 1." in output
        # Unchanged line shown without +/-
        assert "Unchanged" in output

    def test_log_rubric_diff_no_changes(self):
        from io import StringIO

        from rich.console import Console

        from autorubric.meta._improve import ImprovementProgressDisplay

        display = ImprovementProgressDisplay()
        display._console = Console(file=StringIO(), force_terminal=True)

        rubric = Rubric([
            Criterion(name="a", weight=1.0, requirement="Same"),
        ])
        display.log_rubric_diff(rubric, rubric, iteration=1)
        output = display._console.file.getvalue()
        assert "No changes" in output

    def test_begin_advance_end_lifecycle(self):
        from io import StringIO

        from rich.console import Console

        from autorubric.meta._improve import ImprovementProgressDisplay

        display = ImprovementProgressDisplay()
        display._console = Console(file=StringIO(), force_terminal=True)

        display.begin_iteration(0, 5, total_steps=3)
        assert display._progress is not None
        assert display._task_id is not None

        display.advance(phase_name="Testing agreement")
        display.advance()
        display.advance()

        display.end_iteration()
        assert display._progress is None
        assert display._task_id is None

    def test_advance_without_begin_is_noop(self):
        from autorubric.meta._improve import ImprovementProgressDisplay

        display = ImprovementProgressDisplay()
        display.advance()
        display.advance(phase_name="test")

    def test_end_iteration_without_begin_is_noop(self):
        from autorubric.meta._improve import ImprovementProgressDisplay

        display = ImprovementProgressDisplay()
        display.end_iteration()


# ============================================================================
# validate_agreement callback and cost tracking
# ============================================================================


class TestValidateAgreementCallback:
    @pytest.mark.asyncio
    async def test_callback_called_per_sample(self):
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        samples = ["sample1", "sample2", "sample3"]

        mock_report = _make_ensemble_report(
            [_make_ensemble_criterion_report("x", 1.0, CriterionVerdict.MET)],
            score=1.0,
            mean_agreement=0.9,
        )

        callback = MagicMock()

        with patch.object(rubric, "grade", new_callable=AsyncMock, return_value=mock_report):
            agreement, per_crit, cost = await validate_agreement(
                rubric,
                samples,
                [JudgeSpec(llm_config=LLMConfig(model="test"), judge_id="j1")],
                on_sample_complete=callback,
            )

        assert callback.call_count == 3
        assert agreement > 0
        assert cost is not None

    @pytest.mark.asyncio
    async def test_none_callback_is_safe(self):
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        samples = ["sample1"]

        mock_report = _make_ensemble_report(
            [_make_ensemble_criterion_report("x", 1.0, CriterionVerdict.MET)],
            score=1.0,
            mean_agreement=0.9,
        )

        with patch.object(rubric, "grade", new_callable=AsyncMock, return_value=mock_report):
            agreement, per_crit, cost = await validate_agreement(
                rubric,
                samples,
                [JudgeSpec(llm_config=LLMConfig(model="test"), judge_id="j1")],
                on_sample_complete=None,
            )

        assert agreement > 0

    @pytest.mark.asyncio
    async def test_agreement_cost_accumulation(self):
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        samples = ["s1", "s2"]

        report1 = _make_ensemble_report(
            [_make_ensemble_criterion_report("x", 1.0, CriterionVerdict.MET)],
            score=1.0,
            mean_agreement=0.9,
        )
        report1.completion_cost = 0.01

        report2 = _make_ensemble_report(
            [_make_ensemble_criterion_report("x", 1.0, CriterionVerdict.MET)],
            score=1.0,
            mean_agreement=0.8,
        )
        report2.completion_cost = 0.02

        with patch.object(
            rubric, "grade", new_callable=AsyncMock, side_effect=[report1, report2]
        ):
            _, _, cost = await validate_agreement(
                rubric,
                samples,
                [JudgeSpec(llm_config=LLMConfig(model="test"), judge_id="j1")],
            )

        assert cost == pytest.approx(0.03)


# ============================================================================
# revise_rubric cost tracking
# ============================================================================


class TestReviseRubricCost:
    @pytest.mark.asyncio
    async def test_returns_cost(self):
        from autorubric.meta._improve import revise_rubric

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])

        revised_json = json.dumps([
            {"name": "x", "weight": 1.0, "requirement": "Improved test"}
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.05)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )

        with patch("autorubric.meta._improve.LLMClient") as mock_cls:
            mock_cls.return_value.generate = generate_mock

            revised, cost = await revise_rubric(
                rubric, "task", [], "no validation", "no history", config
            )

        assert len(revised.rubric) == 1
        assert revised.rubric[0].requirement == "Improved test"
        assert cost == 0.05


# ============================================================================
# HTML improvement report
# ============================================================================


class TestRenderImprovementReportHtml:
    def test_renders_html(self):
        from autorubric.meta._display import render_improvement_report_html

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        iterations = [
            _make_iteration_result(0, quality_score=0.6, issues=[_make_issue("a")]),
            _make_iteration_result(1, quality_score=0.9),
        ]
        html = render_improvement_report_html(
            iterations, "score_plateau", 0.05, rubric, rubric
        )
        assert "<!DOCTYPE html>" in html
        assert "Rubric Improvement Report" in html
        assert "score_plateau" in html
        assert "bootstrap" in html.lower()

    def test_renders_issues_in_accordion(self):
        from autorubric.meta._display import render_improvement_report_html

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        issues = [_make_issue("clarity"), _make_issue("overlap", is_antipattern=True)]
        iterations = [_make_iteration_result(0, issues=issues)]
        html = render_improvement_report_html(
            iterations, "no_issues", 0.01, rubric, rubric
        )
        assert "clarity" in html
        assert "overlap" in html
        assert "ANTI-PATTERN" in html
        assert "QUALITY GAP" in html
        assert "Rubric #" in html

    def test_renders_final_rubric(self):
        from autorubric.meta._display import render_improvement_report_html

        original = Rubric([Criterion(name="x", weight=1.0, requirement="Old")])
        final = Rubric([
            Criterion(name="x", weight=1.0, requirement="Improved criterion"),
        ])
        iterations = [_make_iteration_result(0)]
        html = render_improvement_report_html(
            iterations, "no_issues", 0.0, original, final
        )
        assert "Final Rubric" in html
        assert "Improved criterion" in html


# ============================================================================
# _serialize_iteration
# ============================================================================


class TestSerializeIteration:
    def test_produces_json_safe_dict(self):
        iter_result = _make_iteration_result(
            0,
            quality_score=0.75,
            agreement=0.85,
            issues=[_make_issue("clarity"), _make_issue("overlap", is_antipattern=True)],
            issues_fixed=["old_issue"],
            issues_introduced=["overlap"],
        )
        data = _serialize_iteration(iter_result)

        # Must be JSON-serializable
        serialized = json.dumps(data, default=str)
        roundtripped = json.loads(serialized)

        assert roundtripped["iteration"] == 0
        assert roundtripped["quality_score"] == 0.75
        assert roundtripped["agreement"] == 0.85
        assert len(roundtripped["issues"]) == 2
        assert roundtripped["issues"][0]["criterion_name"] == "clarity"
        assert roundtripped["issues"][1]["is_antipattern"] is True
        assert roundtripped["issues_fixed"] == ["old_issue"]
        assert roundtripped["issues_introduced"] == ["overlap"]
        assert roundtripped["accepted"] is True
        assert "quality_report" in roundtripped
        assert roundtripped["quality_report"]["score"] is not None

    def test_includes_quality_report_criterion_reports(self):
        iter_result = _make_iteration_result(0)
        data = _serialize_iteration(iter_result)
        qr = data["quality_report"]
        assert "criterion_reports" in qr
        assert len(qr["criterion_reports"]) == 1
        assert qr["criterion_reports"][0]["criterion"]["name"] == "test_criterion"

    def test_includes_rubric_criteria(self):
        iter_result = _make_iteration_result(0)
        data = _serialize_iteration(iter_result)
        assert "rubric_criteria" in data
        assert len(data["rubric_criteria"]) == 1
        assert data["rubric_criteria"][0]["requirement"] == "Requirement for test_criterion"

    def test_includes_token_usage(self):
        iter_result = _make_iteration_result(0)
        data = _serialize_iteration(iter_result)
        assert data["token_usage"]["prompt_tokens"] == 100
        assert data["token_usage"]["completion_tokens"] == 50
        assert data["token_usage"]["total_tokens"] == 150

    def test_handles_none_agreement(self):
        iter_result = _make_iteration_result(0, agreement=None)
        data = _serialize_iteration(iter_result)
        assert data["agreement"] is None
        assert data["per_criterion_agreement"] is None


# ============================================================================
# revise_rubric _capture
# ============================================================================


class TestReviseRubricCapture:
    @pytest.mark.asyncio
    async def test_capture_populates_prompts_and_response(self):
        from autorubric.meta._improve import revise_rubric

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])

        revised_json = json.dumps([
            {"name": "x", "weight": 1.0, "requirement": "Improved test"}
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.05)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )

        capture: dict = {}
        with patch("autorubric.meta._improve.LLMClient") as mock_cls:
            mock_cls.return_value.generate = generate_mock

            await revise_rubric(
                rubric, "task", [], "no validation", "no history", config,
                _capture=capture,
            )

        assert "system_prompt" in capture
        assert "user_prompt" in capture
        assert "llm_response" in capture
        assert len(capture["system_prompt"]) > 0
        assert "task" in capture["user_prompt"]
        assert revised_json in capture["llm_response"]

    @pytest.mark.asyncio
    async def test_none_capture_is_noop(self):
        from autorubric.meta._improve import revise_rubric

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])

        revised_json = json.dumps([
            {"name": "x", "weight": 1.0, "requirement": "Improved test"}
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.05)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )

        with patch("autorubric.meta._improve.LLMClient") as mock_cls:
            mock_cls.return_value.generate = generate_mock

            revised, cost = await revise_rubric(
                rubric, "task", [], "no validation", "no history", config,
                _capture=None,
            )

        assert len(revised.rubric) == 1
        assert cost == 0.05


# ============================================================================
# validate_agreement _capture
# ============================================================================


class TestValidateAgreementCapture:
    @pytest.mark.asyncio
    async def test_capture_populates_per_sample(self):
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        samples = ["sample1", "sample2"]

        mock_report = _make_ensemble_report(
            [_make_ensemble_criterion_report("x", 1.0, CriterionVerdict.MET)],
            score=1.0,
            mean_agreement=0.9,
        )

        capture: list = []
        with patch.object(rubric, "grade", new_callable=AsyncMock, return_value=mock_report):
            await validate_agreement(
                rubric,
                samples,
                [JudgeSpec(llm_config=LLMConfig(model="test"), judge_id="j1")],
                _capture=capture,
            )

        assert len(capture) == 2
        assert capture[0]["submission"] == "sample1"
        assert capture[1]["submission"] == "sample2"
        assert capture[0]["score"] == 1.0
        assert capture[0]["mean_agreement"] == 0.9
        assert "criterion_reports" in capture[0]
        assert len(capture[0]["criterion_reports"]) == 1

    @pytest.mark.asyncio
    async def test_none_capture_is_noop(self):
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        samples = ["sample1"]

        mock_report = _make_ensemble_report(
            [_make_ensemble_criterion_report("x", 1.0, CriterionVerdict.MET)],
            score=1.0,
            mean_agreement=0.9,
        )

        with patch.object(rubric, "grade", new_callable=AsyncMock, return_value=mock_report):
            agreement, per_crit, cost = await validate_agreement(
                rubric,
                samples,
                [JudgeSpec(llm_config=LLMConfig(model="test"), judge_id="j1")],
                _capture=None,
            )

        assert agreement > 0


# ============================================================================
# Artifact persistence integration tests
# ============================================================================


class TestArtifactPersistence:
    @staticmethod
    def _make_eval_mock(quality_report):
        """Create an eval mock that writes an HTML stub when output_html_path is given."""
        async def _eval_side_effect(rubric, *args, **kwargs):
            html_path = kwargs.get("output_html_path")
            if html_path:
                Path(html_path).write_text("<html>stub</html>", encoding="utf-8")
            return quality_report
        return AsyncMock(side_effect=_eval_side_effect)

    @pytest.mark.asyncio
    async def test_artifacts_written_on_convergence(self, tmp_path):
        """When save_artifacts=True and loop converges on first iteration,
        all expected artifact files are created."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        quality_report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report("clarity", 1.0, CriterionVerdict.MET),
            ],
            score=1.0,
            mean_agreement=1.0,
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),

            save_artifacts=True,
            artifacts_dir=tmp_path / "artifacts",
            show_progress=False,
            max_iterations=5,
        )

        with patch(
            "autorubric.meta._improve.evaluate_rubric_in_context",
            self._make_eval_mock(quality_report),
        ):
            result = await improve_rubric(
                rubric,
                "Write a summary",
                config=config,
            )

        artifacts_dir = tmp_path / "artifacts"

        # Rubric JSON
        assert (artifacts_dir / "rubric-iter-00.json").exists()
        # Per-iteration JSON
        assert (artifacts_dir / "iter-00.json").exists()
        # Eval HTML (always generated when artifacts_dir is set)
        assert (artifacts_dir / "eval-iter-00.html").exists()
        # Improvement report HTML
        assert (artifacts_dir / "improvement_report.html").exists()
        # Summary JSON
        assert (artifacts_dir / "summary.json").exists()

    @pytest.mark.asyncio
    async def test_summary_json_structure(self, tmp_path):
        """summary.json contains expected keys and structure."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        quality_report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report("clarity", 1.0, CriterionVerdict.MET),
            ],
            score=1.0,
            mean_agreement=1.0,
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),

            save_artifacts=True,
            artifacts_dir=tmp_path / "artifacts",
            show_progress=False,
        )

        with patch(
            "autorubric.meta._improve.evaluate_rubric_in_context",
            new_callable=AsyncMock,
            return_value=quality_report,
        ):
            await improve_rubric(rubric, "Write a summary", config=config)

        summary_path = tmp_path / "artifacts" / "summary.json"
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        assert "original_rubric" in summary
        assert "final_rubric" in summary
        assert summary["task_prompt"] == "Write a summary"
        assert summary["convergence_reason"] == "no_issues"
        assert summary["best_iteration"] == 0
        assert summary["total_iterations"] == 1
        assert "config" in summary
        assert summary["config"]["mode"] == "in_context"
        assert summary["config"]["eval_llm_model"] == "test-model"
        assert "iterations_summary" in summary
        assert len(summary["iterations_summary"]) == 1
        assert summary["iterations_summary"][0]["quality_score"] == 1.0

    @pytest.mark.asyncio
    async def test_iter_json_structure(self, tmp_path):
        """iter-00.json contains expected keys from _serialize_iteration."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        quality_report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report("clarity", 1.0, CriterionVerdict.MET),
            ],
            score=1.0,
            mean_agreement=1.0,
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),

            save_artifacts=True,
            artifacts_dir=tmp_path / "artifacts",
            show_progress=False,
        )

        with patch(
            "autorubric.meta._improve.evaluate_rubric_in_context",
            new_callable=AsyncMock,
            return_value=quality_report,
        ):
            await improve_rubric(rubric, "Write a summary", config=config)

        iter_path = tmp_path / "artifacts" / "iter-00.json"
        with open(iter_path, encoding="utf-8") as f:
            iter_data = json.load(f)

        assert iter_data["iteration"] == 0
        assert iter_data["quality_score"] == 1.0
        assert "issues" in iter_data
        assert "quality_report" in iter_data
        assert "rubric_criteria" in iter_data

    @pytest.mark.asyncio
    async def test_iter_json_with_revision(self, tmp_path):
        """iter-00.json includes revision data when a revision occurs."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        report_with_issues = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "clarity", 1.0, CriterionVerdict.UNMET, reason="Not clear"
                ),
            ],
            score=0.5,
        )
        report_clean = _make_ensemble_report(
            [
                _make_ensemble_criterion_report(
                    "clarity", 1.0, CriterionVerdict.MET, reason="Clear now"
                ),
            ],
            score=1.0,
        )

        eval_mock = AsyncMock(side_effect=[report_with_issues, report_clean])

        revised_criteria_json = json.dumps([
            {"name": "clarity", "weight": 1.0, "requirement": "Is clear and concise"}
        ])
        generate_result = GenerateResult(
            content=revised_criteria_json, cost=0.005
        )
        generate_mock = AsyncMock(return_value=generate_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),

            save_artifacts=True,
            artifacts_dir=tmp_path / "artifacts",
            show_progress=False,
            max_iterations=5,
        )

        with (
            patch(
                "autorubric.meta._improve.evaluate_rubric_in_context",
                eval_mock,
            ),
            patch(
                "autorubric.meta._improve.LLMClient",
            ) as mock_llm_client_cls,
        ):
            mock_client_instance = MagicMock()
            mock_client_instance.generate = generate_mock
            mock_llm_client_cls.return_value = mock_client_instance

            await improve_rubric(rubric, "Write a summary", config=config)

        # Iteration 0 had issues and a revision
        iter0_path = tmp_path / "artifacts" / "iter-00.json"
        with open(iter0_path, encoding="utf-8") as f:
            iter0 = json.load(f)

        assert "revision" in iter0
        assert "system_prompt" in iter0["revision"]
        assert "user_prompt" in iter0["revision"]
        assert "llm_response" in iter0["revision"]

        # Iteration 1 converged (no revision)
        iter1_path = tmp_path / "artifacts" / "iter-01.json"
        with open(iter1_path, encoding="utf-8") as f:
            iter1 = json.load(f)

        assert "revision" not in iter1

    @pytest.mark.asyncio
    async def test_html_generated_without_html_display_mode(self, tmp_path):
        """eval HTML and improvement_report.html are generated even when display != 'html'."""
        rubric = Rubric([
            Criterion(name="clarity", weight=1.0, requirement="Is clear"),
        ])

        quality_report = _make_ensemble_report(
            [
                _make_ensemble_criterion_report("clarity", 1.0, CriterionVerdict.MET),
            ],
            score=1.0,
            mean_agreement=1.0,
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test-model"),
            revision_llm=LLMConfig(model="test-model"),

            save_artifacts=True,
            artifacts_dir=tmp_path / "artifacts",
            show_progress=False,
            display=None,  # Explicitly not "html"
        )

        with patch(
            "autorubric.meta._improve.evaluate_rubric_in_context",
            self._make_eval_mock(quality_report),
        ):
            await improve_rubric(rubric, "Write a summary", config=config)

        artifacts_dir = tmp_path / "artifacts"
        assert (artifacts_dir / "eval-iter-00.html").exists()
        assert (artifacts_dir / "improvement_report.html").exists()


# ============================================================================
# compute_expected_scores
# ============================================================================


class TestComputeExpectedScores:
    def test_computes_scores_from_ground_truth(self):
        rubric = Rubric([
            Criterion(name="a", weight=10.0, requirement="First"),
            Criterion(name="b", weight=5.0, requirement="Second"),
        ])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(
                    submission="s1",
                    description="Both met",
                    ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
                ),
                DataItem(
                    submission="s2",
                    description="First met only",
                    ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
                ),
                DataItem(
                    submission="s3",
                    description="None met",
                    ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.UNMET],
                ),
            ],
        )
        scores = compute_expected_scores(dataset)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(10.0 / 15.0)
        assert scores[2] == pytest.approx(0.0)

    def test_raises_without_rubric(self):
        rubric = Rubric([Criterion(name="a", weight=1.0, requirement="Test")])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(submission="s1", description="d1"),
            ],
        )
        # Remove rubric after construction to test the guard
        dataset.rubric = None
        with pytest.raises(ValueError, match="must have a rubric"):
            compute_expected_scores(dataset)


# ============================================================================
# validate_ground_truth
# ============================================================================


class TestValidateGroundTruth:
    @pytest.mark.asyncio
    async def test_computes_correlation(self):
        rubric = Rubric([
            Criterion(name="a", weight=10.0, requirement="First"),
        ])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(
                    submission="s1",
                    description="d1",
                    ground_truth=[CriterionVerdict.MET],
                ),
                DataItem(
                    submission="s2",
                    description="d2",
                    ground_truth=[CriterionVerdict.UNMET],
                ),
                DataItem(
                    submission="s3",
                    description="d3",
                    ground_truth=[CriterionVerdict.MET],
                ),
            ],
        )
        expected_scores = compute_expected_scores(dataset)

        # Mock rubric.grade to return scores matching the ground truth
        report_met = _make_ensemble_report(
            [_make_ensemble_criterion_report("a", 10.0, CriterionVerdict.MET)],
            score=1.0,
        )
        report_unmet = _make_ensemble_report(
            [_make_ensemble_criterion_report("a", 10.0, CriterionVerdict.UNMET)],
            score=0.0,
        )

        grader = CriterionGrader(llm_config=LLMConfig(model="test"))

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            side_effect=[report_met, report_unmet, report_met],
        ):
            correlation, per_item, cost = await validate_ground_truth(
                rubric, dataset, expected_scores, grader,
            )

        # Perfect correlation when rubric scores match expected scores
        assert correlation == pytest.approx(1.0)
        assert len(per_item) == 3
        assert per_item[0] == (1.0, 1.0)
        assert per_item[1] == (0.0, 0.0)
        assert per_item[2] == (1.0, 1.0)

    @pytest.mark.asyncio
    async def test_fallback_for_small_n(self):
        """When n < 3, uses 1-MAE instead of Spearman."""
        rubric = Rubric([
            Criterion(name="a", weight=10.0, requirement="First"),
        ])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(
                    submission="s1",
                    description="d1",
                    ground_truth=[CriterionVerdict.MET],
                ),
                DataItem(
                    submission="s2",
                    description="d2",
                    ground_truth=[CriterionVerdict.UNMET],
                ),
            ],
        )
        expected_scores = compute_expected_scores(dataset)

        report_met = _make_ensemble_report(
            [_make_ensemble_criterion_report("a", 10.0, CriterionVerdict.MET)],
            score=1.0,
        )
        report_unmet = _make_ensemble_report(
            [_make_ensemble_criterion_report("a", 10.0, CriterionVerdict.UNMET)],
            score=0.0,
        )

        grader = CriterionGrader(llm_config=LLMConfig(model="test"))

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            side_effect=[report_met, report_unmet],
        ):
            metric, per_item, cost = await validate_ground_truth(
                rubric, dataset, expected_scores, grader,
            )

        # Perfect match: MAE=0, so metric = 1 - 0 = 1.0
        assert metric == pytest.approx(1.0)


# ============================================================================
# format_ground_truth_for_prompt
# ============================================================================


class TestFormatGroundTruthForPrompt:
    def test_formats_with_header(self):
        result = format_ground_truth_for_prompt(
            0.85,
            [(0.9, 1.0), (0.5, 0.6), (0.2, 0.0)],
        )
        assert result.startswith("## Validation Against Ground Truth")
        assert "0.85" in result
        assert "Submission 1" in result
        assert "Submission 2" in result
        assert "Submission 3" in result

    def test_shows_gap(self):
        result = format_ground_truth_for_prompt(
            0.5,
            [(0.8, 0.5)],
        )
        assert "gap: +0.30" in result

    def test_empty_per_item(self):
        result = format_ground_truth_for_prompt(0.0, [])
        assert "## Validation Against Ground Truth" in result
        assert "Adjust criteria" in result

    def test_with_item_reports_adds_diagnostics(self):
        """When item_reports is provided, a diagnostics section appears."""
        per_item = [(0.80, 0.30), (0.20, 0.70), (0.50, 0.50)]
        reports = [
            _make_ensemble_report([
                _make_ensemble_criterion_report("accuracy", 10.0, CriterionVerdict.MET, reason="Looks correct"),
                _make_ensemble_criterion_report("error", -5.0, CriterionVerdict.UNMET, reason="No error found"),
            ], score=0.80),
            _make_ensemble_report([
                _make_ensemble_criterion_report("accuracy", 10.0, CriterionVerdict.UNMET, reason="Missing info"),
                _make_ensemble_criterion_report("error", -5.0, CriterionVerdict.MET, reason="Has errors"),
            ], score=0.20),
            _make_ensemble_report([
                _make_ensemble_criterion_report("accuracy", 10.0, CriterionVerdict.MET),
            ], score=0.50),
        ]
        result = format_ground_truth_for_prompt(0.5, per_item, item_reports=reports)
        assert "## Grading Diagnostics for Largest Gaps" in result
        assert "Over-scored" in result
        assert "Under-scored" in result
        assert "Looks correct" in result
        assert "Missing info" in result

    def test_without_item_reports_unchanged(self):
        """Without item_reports the output is identical to the original."""
        per_item = [(0.8, 0.5), (0.3, 0.6)]
        without = format_ground_truth_for_prompt(0.5, per_item)
        with_none = format_ground_truth_for_prompt(0.5, per_item, item_reports=None)
        assert without == with_none
        assert "Grading Diagnostics" not in without


# ============================================================================
# _select_diagnostic_items
# ============================================================================


class TestSelectDiagnosticItems:
    def test_basic_selection(self):
        """Selects top-N items by |gap| per direction."""
        per_item = [
            (0.80, 0.30),  # gap +0.50 (over)
            (0.20, 0.70),  # gap -0.50 (under)
            (0.60, 0.50),  # gap +0.10 (over)
            (0.50, 0.50),  # gap  0.00 (skip)
        ]
        reports = [
            _make_ensemble_report([], score=s) for s, _ in per_item
        ]
        over, under = _select_diagnostic_items(per_item, reports, n_per_direction=3)
        assert len(over) == 2  # items 0 and 2
        assert len(under) == 1  # item 1
        # Sorted by descending |gap|
        assert over[0][0] == 0  # index 0, gap=+0.50
        assert over[1][0] == 2  # index 2, gap=+0.10
        assert under[0][0] == 1  # index 1, gap=-0.50

    def test_excludes_zero_gap(self):
        per_item = [(0.5, 0.5), (0.5, 0.5)]
        reports = [_make_ensemble_report([], score=0.5) for _ in per_item]
        over, under = _select_diagnostic_items(per_item, reports)
        assert over == []
        assert under == []

    def test_caps_at_n(self):
        per_item = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
        reports = [_make_ensemble_report([], score=s) for s, _ in per_item]
        over, under = _select_diagnostic_items(per_item, reports, n_per_direction=2)
        assert len(over) == 2
        assert over[0][0] == 0  # largest gap first
        assert over[1][0] == 1

    def test_handles_fewer_than_n(self):
        per_item = [(0.8, 0.3)]
        reports = [_make_ensemble_report([], score=0.8)]
        over, under = _select_diagnostic_items(per_item, reports, n_per_direction=5)
        assert len(over) == 1
        assert len(under) == 0


# ============================================================================
# _format_error_criteria
# ============================================================================


class TestFormatErrorCriteria:
    def test_over_scored_filters_met_positive_and_unmet_negative(self):
        """Over-scored: include MET positive-weight + UNMET negative-weight."""
        report = _make_ensemble_report([
            _make_ensemble_criterion_report("good", 10.0, CriterionVerdict.MET, reason="Reason A"),
            _make_ensemble_criterion_report("bad", -5.0, CriterionVerdict.UNMET, reason="Reason B"),
            _make_ensemble_criterion_report("missed", 10.0, CriterionVerdict.UNMET, reason="Should exclude"),
            _make_ensemble_criterion_report("caught", -5.0, CriterionVerdict.MET, reason="Should exclude"),
        ])
        lines = _format_error_criteria(report, over_scored=True)
        assert len(lines) == 2
        assert "good" in lines[0] and "MET" in lines[0]
        assert "bad" in lines[1] and "UNMET" in lines[1]

    def test_under_scored_filters_unmet_positive_and_met_negative(self):
        """Under-scored: include UNMET positive-weight + MET negative-weight."""
        report = _make_ensemble_report([
            _make_ensemble_criterion_report("good", 10.0, CriterionVerdict.UNMET, reason="Reason C"),
            _make_ensemble_criterion_report("bad", -5.0, CriterionVerdict.MET, reason="Reason D"),
            _make_ensemble_criterion_report("met_pos", 10.0, CriterionVerdict.MET, reason="Should exclude"),
            _make_ensemble_criterion_report("unmet_neg", -5.0, CriterionVerdict.UNMET, reason="Should exclude"),
        ])
        lines = _format_error_criteria(report, over_scored=False)
        assert len(lines) == 2
        assert "good" in lines[0] and "UNMET" in lines[0]
        assert "bad" in lines[1] and "MET" in lines[1]

    def test_skips_cannot_assess(self):
        report = _make_ensemble_report([
            _make_ensemble_criterion_report("x", 10.0, CriterionVerdict.CANNOT_ASSESS, reason="N/A"),
        ])
        assert _format_error_criteria(report, over_scored=True) == []
        assert _format_error_criteria(report, over_scored=False) == []

    def test_skips_none_verdict(self):
        report = _make_ensemble_report([
            _make_ensemble_criterion_report("x", 10.0, CriterionVerdict.MET),
        ])
        # Manually set verdict to None to simulate multi-choice criterion
        report.report[0].final_verdict = None
        assert _format_error_criteria(report, over_scored=True) == []

    def test_empty_report(self):
        report = _make_ensemble_report([])
        assert _format_error_criteria(report, over_scored=True) == []

    def test_none_report(self):
        report = _make_ensemble_report([])
        report.report = None
        assert _format_error_criteria(report, over_scored=True) == []

    def test_weight_sign_in_output(self):
        """Positive weights get '+' prefix, negative don't."""
        report = _make_ensemble_report([
            _make_ensemble_criterion_report("pos", 10.0, CriterionVerdict.MET, reason="r1"),
            _make_ensemble_criterion_report("neg", -5.0, CriterionVerdict.UNMET, reason="r2"),
        ])
        lines = _format_error_criteria(report, over_scored=True)
        assert "w=+10" in lines[0]
        assert "w=-5" in lines[1]


# ============================================================================
# Startup validation in ImprovementRunner.run()
# ============================================================================


class TestStartupValidation:
    @pytest.mark.asyncio
    async def test_mixed_ground_truth_raises(self):
        """Mixed ground_truth (some items with, some without) raises ValueError."""
        from autorubric.meta._improve import ImprovementRunner

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(
                    submission="s1",
                    description="d1",
                    ground_truth=[CriterionVerdict.MET],
                ),
                DataItem(
                    submission="s2",
                    description="d2",
                    ground_truth=None,
                ),
            ],
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            validation_data=dataset,
            save_artifacts=False,
            show_progress=False,
        )
        runner = ImprovementRunner(rubric, "prompt", config=config)
        with pytest.raises(ValueError, match="mixed is not supported"):
            await runner.run()

    @pytest.mark.asyncio
    async def test_no_ground_truth_single_llm_raises(self):
        """No ground_truth + single LLMConfig eval_llm raises ValueError."""
        from autorubric.meta._improve import ImprovementRunner

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(submission="s1", description="d1"),
                DataItem(submission="s2", description="d2"),
            ],
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            validation_data=dataset,
            save_artifacts=False,
            show_progress=False,
        )
        runner = ImprovementRunner(rubric, "prompt", config=config)
        with pytest.raises(ValueError, match="list\\[JudgeSpec\\]"):
            await runner.run()

    @pytest.mark.asyncio
    async def test_no_ground_truth_single_judge_raises(self):
        """No ground_truth + list[JudgeSpec] with < 2 judges raises ValueError."""
        from autorubric.meta._improve import ImprovementRunner

        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Test")])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(submission="s1", description="d1"),
                DataItem(submission="s2", description="d2"),
            ],
        )

        config = ImprovementConfig(
            eval_llm=[JudgeSpec(llm_config=LLMConfig(model="test"), judge_id="j1")],
            revision_llm=LLMConfig(model="test"),
            validation_data=dataset,
            save_artifacts=False,
            show_progress=False,
        )
        runner = ImprovementRunner(rubric, "prompt", config=config)
        with pytest.raises(ValueError, match=">= 2 judges"):
            await runner.run()


# ============================================================================
# Held-out strategy: validate_held_out
# ============================================================================


class TestValidateHeldOut:
    """Tests for the validate_held_out building block."""

    def _rubric_and_dataset(self):
        rubric = Rubric([
            Criterion(name="accuracy", weight=1.0, requirement="Must be accurate"),
            Criterion(name="style", weight=1.0, requirement="Must have good style"),
        ])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(
                    submission="Good submission",
                    description="d1",
                    ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
                ),
                DataItem(
                    submission="Bad submission",
                    description="d2",
                    ground_truth=[CriterionVerdict.UNMET, CriterionVerdict.UNMET],
                ),
                DataItem(
                    submission="Mixed submission",
                    description="d3",
                    ground_truth=[CriterionVerdict.MET, CriterionVerdict.UNMET],
                ),
            ],
        )
        return rubric, dataset

    def _make_grade_report(self, verdicts):
        """Build a mock grade report from a list of (name, weight, verdict) tuples."""
        criterion_reports = [
            _make_ensemble_criterion_report(name, weight, verdict)
            for name, weight, verdict in verdicts
        ]
        return _make_ensemble_report(criterion_reports, score=0.5)

    @pytest.mark.asyncio
    async def test_per_criterion_accuracy(self):
        """Verifies accuracy is computed correctly per criterion."""
        from autorubric.meta._improve import validate_held_out

        rubric, dataset = self._rubric_and_dataset()
        grader = CriterionGrader(llm_config=LLMConfig(model="test"))

        # LLM agrees with ground truth on all items for criterion 0 (accuracy),
        # disagrees on all items for criterion 1 (style).
        reports = [
            self._make_grade_report([
                ("accuracy", 1.0, CriterionVerdict.MET),
                ("style", 1.0, CriterionVerdict.UNMET),  # GT=MET, wrong
            ]),
            self._make_grade_report([
                ("accuracy", 1.0, CriterionVerdict.UNMET),
                ("style", 1.0, CriterionVerdict.MET),  # GT=UNMET, wrong
            ]),
            self._make_grade_report([
                ("accuracy", 1.0, CriterionVerdict.MET),
                ("style", 1.0, CriterionVerdict.MET),  # GT=UNMET, wrong
            ]),
        ]

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            side_effect=reports,
        ):
            result = await validate_held_out(
                rubric, dataset, grader,
                max_exemplars_per_criterion=5,
            )

        assert len(result.per_criterion) == 2
        # Criterion 0: all 3 correct
        acc_report = result.per_criterion[0]
        assert acc_report.criterion_name == "accuracy"
        assert acc_report.accuracy == pytest.approx(1.0)
        # Criterion 1: all 3 wrong
        style_report = result.per_criterion[1]
        assert style_report.criterion_name == "style"
        assert style_report.accuracy == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_fp_fn_rates(self):
        """Verifies false positive and false negative rates."""
        from autorubric.meta._improve import validate_held_out

        rubric, dataset = self._rubric_and_dataset()
        grader = CriterionGrader(llm_config=LLMConfig(model="test"))

        # For criterion 0 (accuracy):
        #   item0: GT=MET, LLM=MET -> TP
        #   item1: GT=UNMET, LLM=MET -> FP
        #   item2: GT=MET, LLM=UNMET -> FN
        reports = [
            self._make_grade_report([
                ("accuracy", 1.0, CriterionVerdict.MET),
                ("style", 1.0, CriterionVerdict.MET),
            ]),
            self._make_grade_report([
                ("accuracy", 1.0, CriterionVerdict.MET),  # FP
                ("style", 1.0, CriterionVerdict.UNMET),
            ]),
            self._make_grade_report([
                ("accuracy", 1.0, CriterionVerdict.UNMET),  # FN
                ("style", 1.0, CriterionVerdict.UNMET),
            ]),
        ]

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            side_effect=reports,
        ):
            result = await validate_held_out(
                rubric, dataset, grader,
            )

        acc = result.per_criterion[0]
        # FP rate = FP / (FP + TN) = 1 / (1 + 0) = 1.0
        assert acc.false_positive_rate == pytest.approx(1.0)
        # FN rate = FN / (FN + TP) = 1 / (1 + 1) = 0.5
        assert acc.false_negative_rate == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_exemplar_selection_capped(self):
        """Disagreement exemplars are capped at max_exemplars_per_criterion."""
        from autorubric.meta._improve import validate_held_out

        rubric = Rubric([
            Criterion(name="c", weight=1.0, requirement="Test"),
        ])
        # 5 items, all disagreeing
        items = [
            DataItem(
                submission=f"s{i}",
                description=f"d{i}",
                ground_truth=[CriterionVerdict.MET],
            )
            for i in range(5)
        ]
        dataset = RubricDataset(prompt="task", rubric=rubric, items=items)
        grader = CriterionGrader(llm_config=LLMConfig(model="test"))

        report = self._make_grade_report([("c", 1.0, CriterionVerdict.UNMET)])

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            return_value=report,
        ):
            result = await validate_held_out(
                rubric, dataset, grader,
                max_exemplars_per_criterion=2,
            )

        assert len(result.per_criterion[0].disagreement_exemplars) == 2

    @pytest.mark.asyncio
    async def test_on_item_complete_callback(self):
        """on_item_complete is called once per item."""
        from autorubric.meta._improve import validate_held_out

        rubric, dataset = self._rubric_and_dataset()
        grader = CriterionGrader(llm_config=LLMConfig(model="test"))
        report = self._make_grade_report([
            ("accuracy", 1.0, CriterionVerdict.MET),
            ("style", 1.0, CriterionVerdict.MET),
        ])
        callback = MagicMock()

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            return_value=report,
        ):
            await validate_held_out(
                rubric, dataset, grader,
                on_item_complete=callback,
            )

        assert callback.call_count == 3

    @pytest.mark.asyncio
    async def test_mean_accuracy(self):
        """mean_accuracy is the mean of per-criterion accuracies."""
        from autorubric.meta._improve import validate_held_out

        rubric = Rubric([
            Criterion(name="a", weight=1.0, requirement="A"),
            Criterion(name="b", weight=1.0, requirement="B"),
        ])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(
                    submission="s1", description="d1",
                    ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
                ),
            ],
        )
        grader = CriterionGrader(llm_config=LLMConfig(model="test"))

        # a: correct (MET==MET), b: wrong (UNMET!=MET)
        report = self._make_grade_report([
            ("a", 1.0, CriterionVerdict.MET),
            ("b", 1.0, CriterionVerdict.UNMET),
        ])

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            return_value=report,
        ):
            result = await validate_held_out(rubric, dataset, grader)

        # a: 100%, b: 0% -> mean = 50%
        assert result.mean_accuracy == pytest.approx(0.5)


# ============================================================================
# Held-out strategy: format_held_out_for_prompt
# ============================================================================


class TestFormatHeldOutForPrompt:
    """Tests for the format_held_out_for_prompt building block."""

    def _make_held_out_result(
        self,
        per_criterion=None,
        mean_accuracy=0.75,
    ):
        from autorubric.meta._improve import (
            CriterionErrorReport,
            CriterionExemplar,
            HeldOutValidationResult,
        )

        if per_criterion is None:
            per_criterion = [
                CriterionErrorReport(
                    criterion_index=0,
                    criterion_name="accuracy",
                    n_samples=4,
                    accuracy=0.50,
                    false_positive_rate=0.25,
                    false_negative_rate=0.25,
                    disagreement_exemplars=[
                        CriterionExemplar(
                            item_index=0,
                            submission_snippet="bad answer",
                            llm_verdict=CriterionVerdict.MET,
                            ground_truth_verdict=CriterionVerdict.UNMET,
                            llm_reason="Looks ok",
                            is_disagreement=True,
                        ),
                    ],
                    agreement_exemplars=[
                        CriterionExemplar(
                            item_index=1,
                            submission_snippet="good answer",
                            llm_verdict=CriterionVerdict.MET,
                            ground_truth_verdict=CriterionVerdict.MET,
                            llm_reason="Correct",
                            is_disagreement=False,
                        ),
                    ],
                ),
                CriterionErrorReport(
                    criterion_index=1,
                    criterion_name="style",
                    n_samples=4,
                    accuracy=1.0,
                    false_positive_rate=0.0,
                    false_negative_rate=0.0,
                    disagreement_exemplars=[],
                    agreement_exemplars=[],
                ),
            ]
        return HeldOutValidationResult(
            mean_accuracy=mean_accuracy,
            per_criterion=per_criterion,
            total_cost=0.05,
            item_reports=[],
        )

    def test_header_includes_accuracy(self):
        from autorubric.meta._improve import format_held_out_for_prompt

        result = self._make_held_out_result(mean_accuracy=0.75)
        text = format_held_out_for_prompt(result)
        assert "## Held-Out Validation (Mean Accuracy: 75%)" in text

    def test_per_criterion_sections_present(self):
        from autorubric.meta._improve import format_held_out_for_prompt

        result = self._make_held_out_result()
        text = format_held_out_for_prompt(result)
        assert "### Criterion 1: accuracy" in text
        assert "### Criterion 2: style" in text

    def test_worst_first_ordering(self):
        from autorubric.meta._improve import format_held_out_for_prompt

        result = self._make_held_out_result()
        text = format_held_out_for_prompt(result)
        # accuracy (50%) should come before style (100%) in the per-criterion summary
        acc_pos = text.find("accuracy")
        style_pos = text.find("style")
        assert acc_pos < style_pos

    def test_exemplar_capping(self):
        from autorubric.meta._improve import (
            CriterionErrorReport,
            CriterionExemplar,
            format_held_out_for_prompt,
        )

        exemplars = [
            CriterionExemplar(
                item_index=i,
                submission_snippet=f"snippet {i}",
                llm_verdict=CriterionVerdict.MET,
                ground_truth_verdict=CriterionVerdict.UNMET,
                llm_reason=f"reason {i}",
                is_disagreement=True,
            )
            for i in range(10)
        ]
        report = CriterionErrorReport(
            criterion_index=0,
            criterion_name="c",
            n_samples=10,
            accuracy=0.0,
            false_positive_rate=1.0,
            false_negative_rate=0.0,
            disagreement_exemplars=exemplars,
            agreement_exemplars=[],
        )
        from autorubric.meta._improve import HeldOutValidationResult

        ho_result = HeldOutValidationResult(
            mean_accuracy=0.0,
            per_criterion=[report],
            total_cost=None,
            item_reports=[],
        )
        text = format_held_out_for_prompt(ho_result, max_exemplars_per_criterion=2)
        # Only 2 exemplars shown even though 10 exist
        assert text.count("Judge verdict:") == 2

    def test_perfect_accuracy(self):
        from autorubric.meta._improve import (
            CriterionErrorReport,
            format_held_out_for_prompt,
            HeldOutValidationResult,
        )

        report = CriterionErrorReport(
            criterion_index=0,
            criterion_name="perfect",
            n_samples=4,
            accuracy=1.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            disagreement_exemplars=[],
            agreement_exemplars=[],
        )
        ho_result = HeldOutValidationResult(
            mean_accuracy=1.0,
            per_criterion=[report],
            total_cost=None,
            item_reports=[],
        )
        text = format_held_out_for_prompt(ho_result)
        assert "Mean Accuracy: 100%" in text
        assert "Disagreements" not in text

    def test_disagreement_content_shown(self):
        from autorubric.meta._improve import format_held_out_for_prompt

        result = self._make_held_out_result()
        text = format_held_out_for_prompt(result)
        assert "**Disagreements** (judge got these WRONG)" in text
        assert "bad answer" in text
        assert "Looks ok" in text

    def test_agreement_content_shown(self):
        from autorubric.meta._improve import format_held_out_for_prompt

        result = self._make_held_out_result()
        text = format_held_out_for_prompt(result)
        assert "**Agreements** (judge got these RIGHT)" in text
        assert "Correct" in text


# ============================================================================
# Held-out strategy: validate_criteria_structure
# ============================================================================


class TestValidateCriteriaStructure:
    """Tests for the validate_criteria_structure building block."""

    def test_same_count_same_names(self):
        from autorubric.meta._improve import validate_criteria_structure

        original = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
            Criterion(name="b", weight=1.0, requirement="R2"),
        ])
        revised = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1 improved"),
            Criterion(name="b", weight=1.0, requirement="R2 improved"),
        ])
        valid, error = validate_criteria_structure(original, revised)
        assert valid is True
        assert error is None

    def test_different_count_more(self):
        from autorubric.meta._improve import validate_criteria_structure

        original = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
        ])
        revised = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
            Criterion(name="b", weight=1.0, requirement="R2"),
        ])
        valid, error = validate_criteria_structure(original, revised)
        assert valid is False
        assert "1 -> 2" in error

    def test_different_count_fewer(self):
        from autorubric.meta._improve import validate_criteria_structure

        original = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
            Criterion(name="b", weight=1.0, requirement="R2"),
        ])
        revised = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
        ])
        valid, error = validate_criteria_structure(original, revised)
        assert valid is False
        assert "2 -> 1" in error

    def test_name_mismatch(self):
        from autorubric.meta._improve import validate_criteria_structure

        original = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
            Criterion(name="b", weight=1.0, requirement="R2"),
        ])
        revised = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
            Criterion(name="c", weight=1.0, requirement="R2"),
        ])
        valid, error = validate_criteria_structure(original, revised)
        assert valid is False
        assert "'b'" in error
        assert "'c'" in error

    def test_unnamed_criteria_name_check_skipped(self):
        from autorubric.meta._improve import validate_criteria_structure

        original = Rubric([
            Criterion(name=None, weight=1.0, requirement="R1"),
            Criterion(name="b", weight=1.0, requirement="R2"),
        ])
        revised = Rubric([
            Criterion(name=None, weight=1.0, requirement="R1 improved"),
            Criterion(name="b", weight=1.0, requirement="R2 improved"),
        ])
        valid, error = validate_criteria_structure(original, revised)
        assert valid is True
        assert error is None

    def test_one_side_unnamed_skips_name_check(self):
        """When one side has a name and the other doesn't, skip name check."""
        from autorubric.meta._improve import validate_criteria_structure

        original = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
        ])
        revised = Rubric([
            Criterion(name=None, weight=1.0, requirement="R1 improved"),
        ])
        valid, error = validate_criteria_structure(original, revised)
        assert valid is True
        assert error is None


# ============================================================================
# Held-out strategy: _check_held_out_convergence
# ============================================================================


class TestCheckHeldOutConvergence:
    """Tests for the _check_held_out_convergence building block."""

    def _default_config(self, **overrides):
        cfg = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            max_iterations=10,
            held_out_min_accuracy=0.90,
            score_plateau_threshold=0.02,
            plateau_patience=2,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_accuracy_met(self):
        from autorubric.meta._improve import _check_held_out_convergence

        config = self._default_config(held_out_min_accuracy=0.90)
        state = _ConvergenceState()
        result = _check_held_out_convergence(0, 0.92, config, state, 0.0)
        assert result == "held_out_accuracy_met"

    def test_max_iterations(self):
        from autorubric.meta._improve import _check_held_out_convergence

        config = self._default_config(max_iterations=5)
        state = _ConvergenceState()
        result = _check_held_out_convergence(4, 0.50, config, state, 0.0)
        assert result == "max_iterations"

    def test_score_plateau(self):
        from autorubric.meta._improve import _check_held_out_convergence

        config = self._default_config(
            plateau_patience=2, score_plateau_threshold=0.02,
        )
        state = _ConvergenceState(prev_score=0.80)

        # First plateau tick
        result = _check_held_out_convergence(0, 0.81, config, state, 0.0)
        assert result is None
        assert state.plateau_count == 1

        # Second tick triggers
        result = _check_held_out_convergence(1, 0.82, config, state, 0.0)
        assert result == "score_plateau"

    def test_cost_limit(self):
        from autorubric.meta._improve import _check_held_out_convergence

        config = self._default_config(max_total_cost=1.0)
        state = _ConvergenceState()
        result = _check_held_out_convergence(0, 0.50, config, state, 1.5)
        assert result == "cost_limit"

    def test_continues_when_no_stopping_condition(self):
        from autorubric.meta._improve import _check_held_out_convergence

        config = self._default_config()
        state = _ConvergenceState()
        result = _check_held_out_convergence(0, 0.50, config, state, 0.0)
        assert result is None

    def test_plateau_resets_on_improvement(self):
        from autorubric.meta._improve import _check_held_out_convergence

        config = self._default_config(
            plateau_patience=2, score_plateau_threshold=0.02,
        )
        state = _ConvergenceState(prev_score=0.80, plateau_count=1)

        result = _check_held_out_convergence(0, 0.85, config, state, 0.0)
        assert result is None
        assert state.plateau_count == 0

    def test_updates_prev_score(self):
        from autorubric.meta._improve import _check_held_out_convergence

        config = self._default_config()
        state = _ConvergenceState(prev_score=0.0)
        _check_held_out_convergence(0, 0.60, config, state, 0.0)
        assert state.prev_score == pytest.approx(0.60)


# ============================================================================
# Held-out strategy: revise_rubric_held_out
# ============================================================================


class TestReviseRubricHeldOut:
    """Tests for the revise_rubric_held_out building block."""

    @pytest.mark.asyncio
    async def test_uses_held_out_prompt_templates(self):
        from autorubric.meta._improve import revise_rubric_held_out

        rubric = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
        ])

        revised_json = json.dumps([
            {"name": "a", "weight": 1.0, "requirement": "Improved R1"}
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.05)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )

        with patch("autorubric.meta._improve.LLMClient") as mock_cls:
            mock_cls.return_value.generate = generate_mock

            revised, cost = await revise_rubric_held_out(
                rubric, "task prompt", "diagnostics text", "history text",
                config,
            )

        assert len(revised.rubric) == 1
        assert revised.rubric[0].requirement == "Improved R1"
        assert cost == 0.05

        # Verify the system prompt used is the held-out one
        call_args = generate_mock.call_args
        system_prompt = call_args[0][0]
        assert "held-out" in system_prompt.lower() or "grading diagnostics" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_returns_original_on_criteria_count_change(self):
        """When revised rubric has different criteria count, returns original."""
        from autorubric.meta._improve import revise_rubric_held_out

        rubric = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
        ])

        # LLM returns 2 criteria instead of 1
        revised_json = json.dumps([
            {"name": "a", "weight": 1.0, "requirement": "R1"},
            {"name": "b", "weight": 1.0, "requirement": "R2"},
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.02)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )

        with patch("autorubric.meta._improve.LLMClient") as mock_cls:
            mock_cls.return_value.generate = generate_mock

            revised, cost = await revise_rubric_held_out(
                rubric, "task", "diag", "hist", config,
            )

        # Should return original rubric, not the invalid revised one
        assert len(revised.rubric) == 1
        assert revised.rubric[0].requirement == "R1"
        assert cost == 0.02

    @pytest.mark.asyncio
    async def test_capture_populated(self):
        from autorubric.meta._improve import revise_rubric_held_out

        rubric = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
        ])

        revised_json = json.dumps([
            {"name": "a", "weight": 1.0, "requirement": "Better R1"}
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.03)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )

        capture: dict = {}
        with patch("autorubric.meta._improve.LLMClient") as mock_cls:
            mock_cls.return_value.generate = generate_mock

            await revise_rubric_held_out(
                rubric, "task", "diag", "hist", config,
                _capture=capture,
            )

        assert "system_prompt" in capture
        assert "user_prompt" in capture
        assert "llm_response" in capture
        assert len(capture["system_prompt"]) > 0
        assert "diag" in capture["user_prompt"]

    @pytest.mark.asyncio
    async def test_cost_is_returned(self):
        from autorubric.meta._improve import revise_rubric_held_out

        rubric = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
        ])

        revised_json = json.dumps([
            {"name": "a", "weight": 1.0, "requirement": "R1v2"}
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.07)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )

        with patch("autorubric.meta._improve.LLMClient") as mock_cls:
            mock_cls.return_value.generate = generate_mock
            _, cost = await revise_rubric_held_out(
                rubric, "task", "diag", "hist", config,
            )

        assert cost == 0.07

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        """Custom system_prompt is used when provided."""
        from autorubric.meta._improve import revise_rubric_held_out

        rubric = Rubric([
            Criterion(name="a", weight=1.0, requirement="R1"),
        ])

        revised_json = json.dumps([
            {"name": "a", "weight": 1.0, "requirement": "R1v2"}
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.01)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
        )

        with patch("autorubric.meta._improve.LLMClient") as mock_cls:
            mock_cls.return_value.generate = generate_mock

            await revise_rubric_held_out(
                rubric, "task", "diag", "hist", config,
                system_prompt="Custom system prompt",
            )

        call_args = generate_mock.call_args
        assert call_args[0][0] == "Custom system prompt"


# ============================================================================
# Held-out strategy: integration tests
# ============================================================================


class TestHeldOutIntegration:
    """Integration tests for the held-out strategy."""

    def _make_dataset_and_rubric(self):
        rubric = Rubric([
            Criterion(name="accuracy", weight=1.0, requirement="Must be accurate"),
        ])
        dataset = RubricDataset(
            prompt="Evaluate this",
            rubric=rubric,
            items=[
                DataItem(
                    submission="good",
                    description="d1",
                    ground_truth=[CriterionVerdict.MET],
                ),
                DataItem(
                    submission="bad",
                    description="d2",
                    ground_truth=[CriterionVerdict.UNMET],
                ),
            ],
        )
        return rubric, dataset

    @pytest.mark.asyncio
    async def test_converges_on_high_accuracy(self):
        """held_out strategy converges when accuracy >= threshold."""
        from autorubric.meta._improve import ImprovementRunner

        rubric, dataset = self._make_dataset_and_rubric()

        # LLM grades matching ground truth perfectly
        report_met = _make_ensemble_report(
            [_make_ensemble_criterion_report("accuracy", 1.0, CriterionVerdict.MET)],
            score=1.0,
        )
        report_unmet = _make_ensemble_report(
            [_make_ensemble_criterion_report("accuracy", 1.0, CriterionVerdict.UNMET)],
            score=0.0,
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            strategy="held_out",
            validation_data=dataset,
            held_out_min_accuracy=0.90,
            save_artifacts=False,
            show_progress=False,
            max_iterations=5,
        )

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            side_effect=[report_met, report_unmet],
        ):
            runner = ImprovementRunner(rubric, "Evaluate this", config=config)
            result = await runner.run()

        assert result.convergence_reason == "held_out_accuracy_met"
        assert len(result.iterations) == 1
        assert result.iterations[0].quality_score == pytest.approx(1.0)
        assert result.iterations[0].quality_report is None
        assert result.iterations[0].held_out_diagnostics is not None

    @pytest.mark.asyncio
    async def test_validation_data_required(self):
        """held_out strategy raises if validation_data is None."""
        from autorubric.meta._improve import ImprovementRunner

        rubric = Rubric([
            Criterion(name="x", weight=1.0, requirement="Test"),
        ])

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            strategy="held_out",
            validation_data=None,
            save_artifacts=False,
            show_progress=False,
        )

        runner = ImprovementRunner(rubric, "prompt", config=config)
        with pytest.raises(ValueError, match="validation_data is required"):
            await runner.run()

    @pytest.mark.asyncio
    async def test_all_items_must_have_ground_truth(self):
        """held_out strategy raises if any item lacks ground_truth."""
        from autorubric.meta._improve import ImprovementRunner

        rubric = Rubric([
            Criterion(name="x", weight=1.0, requirement="Test"),
        ])
        dataset = RubricDataset(
            prompt="task",
            rubric=rubric,
            items=[
                DataItem(
                    submission="s1", description="d1",
                    ground_truth=[CriterionVerdict.MET],
                ),
                DataItem(
                    submission="s2", description="d2",
                    ground_truth=None,
                ),
            ],
        )

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            strategy="held_out",
            validation_data=dataset,
            save_artifacts=False,
            show_progress=False,
        )

        runner = ImprovementRunner(rubric, "prompt", config=config)
        with pytest.raises(ValueError, match="ground_truth"):
            await runner.run()

    @pytest.mark.asyncio
    async def test_improve_rubric_convenience_api(self):
        """improve_rubric() with strategy='held_out' dispatches correctly."""
        rubric, dataset = self._make_dataset_and_rubric()

        # Perfect grading
        report_met = _make_ensemble_report(
            [_make_ensemble_criterion_report("accuracy", 1.0, CriterionVerdict.MET)],
            score=1.0,
        )
        report_unmet = _make_ensemble_report(
            [_make_ensemble_criterion_report("accuracy", 1.0, CriterionVerdict.UNMET)],
            score=0.0,
        )

        with patch.object(
            rubric, "grade", new_callable=AsyncMock,
            side_effect=[report_met, report_unmet],
        ):
            result = await improve_rubric(
                rubric,
                "Evaluate this",
                eval_llm=LLMConfig(model="test"),
                revision_llm=LLMConfig(model="test"),
                strategy="held_out",
                validation_data=dataset,
                save_artifacts=False,
                show_progress=False,
            )

        assert result.convergence_reason == "held_out_accuracy_met"
        assert result.best_rubric is rubric

    @pytest.mark.asyncio
    async def test_revision_loop_with_held_out(self):
        """Held-out loop: bad accuracy -> revise -> good accuracy -> converge."""
        from autorubric.meta._improve import (
            HeldOutValidationResult,
            CriterionErrorReport,
            ImprovementRunner,
        )

        rubric, dataset = self._make_dataset_and_rubric()

        # Iteration 0: 50% accuracy (wrong on 1 of 1 criteria)
        held_out_bad = HeldOutValidationResult(
            mean_accuracy=0.50,
            per_criterion=[
                CriterionErrorReport(
                    criterion_index=0,
                    criterion_name="accuracy",
                    n_samples=2,
                    accuracy=0.50,
                    false_positive_rate=0.0,
                    false_negative_rate=0.50,
                    disagreement_exemplars=[],
                    agreement_exemplars=[],
                ),
            ],
            total_cost=0.01,
            item_reports=[],
        )

        # Iteration 1: 100% accuracy
        held_out_good = HeldOutValidationResult(
            mean_accuracy=1.0,
            per_criterion=[
                CriterionErrorReport(
                    criterion_index=0,
                    criterion_name="accuracy",
                    n_samples=2,
                    accuracy=1.0,
                    false_positive_rate=0.0,
                    false_negative_rate=0.0,
                    disagreement_exemplars=[],
                    agreement_exemplars=[],
                ),
            ],
            total_cost=0.01,
            item_reports=[],
        )

        validate_mock = AsyncMock(side_effect=[held_out_bad, held_out_good])

        revised_json = json.dumps([
            {"name": "accuracy", "weight": 1.0, "requirement": "Improved accuracy"}
        ])
        gen_result = GenerateResult(content=revised_json, cost=0.01)
        generate_mock = AsyncMock(return_value=gen_result)

        config = ImprovementConfig(
            eval_llm=LLMConfig(model="test"),
            revision_llm=LLMConfig(model="test"),
            strategy="held_out",
            validation_data=dataset,
            held_out_min_accuracy=0.90,
            save_artifacts=False,
            show_progress=False,
            max_iterations=5,
        )

        with (
            patch(
                "autorubric.meta._improve.validate_held_out",
                validate_mock,
            ),
            patch("autorubric.meta._improve.LLMClient") as mock_llm_cls,
        ):
            mock_llm_cls.return_value.generate = generate_mock

            runner = ImprovementRunner(rubric, "Evaluate this", config=config)
            result = await runner.run()

        assert result.convergence_reason == "held_out_accuracy_met"
        assert len(result.iterations) == 2
        assert result.iterations[0].quality_score == pytest.approx(0.50)
        assert result.iterations[1].quality_score == pytest.approx(1.0)


# ============================================================================
# Held-out strategy: _serialize_iteration with held_out_diagnostics
# ============================================================================


class TestSerializeIterationHeldOut:
    """Tests for _serialize_iteration with held-out-specific fields."""

    def test_held_out_diagnostics_serialized(self):
        from autorubric.meta._improve import (
            CriterionErrorReport,
            CriterionExemplar,
            HeldOutValidationResult,
        )

        held_out = HeldOutValidationResult(
            mean_accuracy=0.75,
            per_criterion=[
                CriterionErrorReport(
                    criterion_index=0,
                    criterion_name="accuracy",
                    n_samples=4,
                    accuracy=0.75,
                    false_positive_rate=0.25,
                    false_negative_rate=0.0,
                    disagreement_exemplars=[
                        CriterionExemplar(
                            item_index=0,
                            submission_snippet="test",
                            llm_verdict=CriterionVerdict.MET,
                            ground_truth_verdict=CriterionVerdict.UNMET,
                            llm_reason="reason",
                            is_disagreement=True,
                        ),
                    ],
                    agreement_exemplars=[],
                ),
            ],
            total_cost=0.05,
            item_reports=[],
        )

        rubric = Rubric([_make_criterion("accuracy")])
        iter_result = IterationResult(
            iteration=0,
            rubric=rubric,
            quality_score=0.75,
            agreement=None,
            per_criterion_agreement=None,
            issues=[],
            issues_fixed=[],
            issues_introduced=[],
            accepted=True,
            rejection_reason=None,
            quality_report=None,
            token_usage=None,
            completion_cost=0.05,
            held_out_diagnostics=held_out,
        )

        data = _serialize_iteration(iter_result)

        # JSON-safe
        serialized = json.dumps(data, default=str)
        roundtripped = json.loads(serialized)

        # held_out_diagnostics present
        assert "held_out_diagnostics" in roundtripped
        ho = roundtripped["held_out_diagnostics"]
        assert ho["mean_accuracy"] == 0.75
        assert ho["total_cost"] == 0.05
        assert len(ho["per_criterion"]) == 1
        assert ho["per_criterion"][0]["criterion_name"] == "accuracy"
        assert ho["per_criterion"][0]["accuracy"] == 0.75
        assert ho["per_criterion"][0]["num_disagreements"] == 1
        assert ho["per_criterion"][0]["num_agreements"] == 0

    def test_quality_report_absent_when_none(self):
        rubric = Rubric([_make_criterion("test")])
        iter_result = IterationResult(
            iteration=0,
            rubric=rubric,
            quality_score=0.80,
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
            held_out_diagnostics=None,
        )

        data = _serialize_iteration(iter_result)
        serialized = json.dumps(data, default=str)
        roundtripped = json.loads(serialized)

        assert "quality_report" not in roundtripped
        assert "held_out_diagnostics" not in roundtripped

    def test_both_quality_report_and_held_out_absent(self):
        """When both quality_report and held_out_diagnostics are None, neither key appears."""
        rubric = Rubric([_make_criterion("test")])
        iter_result = IterationResult(
            iteration=0,
            rubric=rubric,
            quality_score=0.50,
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
        )

        data = _serialize_iteration(iter_result)
        assert "quality_report" not in data
        assert "held_out_diagnostics" not in data
