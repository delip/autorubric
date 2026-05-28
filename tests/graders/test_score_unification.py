"""Cross-path agreement for the unified scoring core (audit T4-A + T1-B).

After step (b) of the scoring-unification refactor, three scorers route through the
single ``autorubric.scoring.score_reports`` core:

- the live grader (``CriterionGrader`` / ``Rubric.grade``),
- ``Rubric.compute_score`` (ground-truth / expected scores),
- ``RubricDataset.compute_weighted_score``.

These tests prove the grader and ``Rubric.compute_score`` produce *identical* scores
across every ``CannotAssessStrategy`` x {binary, multi-choice} x {+/- weight}, for the
same underlying verdicts -- the explicit cross-check the audit requires. They also pin
the ``Rubric.compute_score`` SKIP-denominator regression (the dual-denominator bug the
unified core fixes) and the T1-B weight-sign-aware multi-choice FAIL penalization.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorubric import (
    CannotAssessConfig,
    CannotAssessStrategy,
    Criterion,
    CriterionOption,
    CriterionVerdict,
    Rubric,
    TokenUsage,
)
from autorubric.graders import CriterionGrader
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment, MultiChoiceJudgment

# ---------------------------------------------------------------------------
# Mixed rubric: +/- binary and +/- ordinal multi-choice (each with an NA option),
# plus an assessable binary and a non-NA multi-choice. Requirements are unique so
# the mock can key its verdict off the prompt text deterministically.
# ---------------------------------------------------------------------------

# Ordinal options shared by the multi-choice criteria. Lowest *scored* value is
# 0.3 (not 0.0); the 0.0 option is NA. This distinguishes worst_scored_option()
# (-> 0.3 / index 0) from a naive binary-style "0 for positive FAIL".
MC_OPTIONS = [
    CriterionOption(label="low", value=0.3),
    CriterionOption(label="mid", value=0.6),
    CriterionOption(label="high", value=1.0),
    CriterionOption(label="na", value=0.0, na=True),
]
NA_LABEL = "na"
NA_OPTION_NUMBER = 4  # 1-indexed position of the NA option (shuffle disabled)

# Requirement -> (verdict-for-Rubric.compute_score, mock-response-builder).
# Binary verdicts are CriterionVerdict; multi-choice verdicts are option labels.
REQ_POS_BIN_CA = "POS binary cannot-assess"
REQ_NEG_BIN_CA = "NEG binary cannot-assess"
REQ_POS_BIN_MET = "POS binary assessable met"
REQ_POS_MC_NA = "POS multi-choice NA"
REQ_NEG_MC_NA = "NEG multi-choice NA"
REQ_POS_MC_HIGH = "POS multi-choice assessable high"


def _mixed_rubric() -> Rubric:
    """A rubric mixing +/- binary, +/- ordinal multi-choice (with NA), plus one
    assessable binary and one non-NA multi-choice."""
    return Rubric(
        [
            Criterion(name="pbc", weight=10.0, requirement=REQ_POS_BIN_CA),
            Criterion(name="nbc", weight=-4.0, requirement=REQ_NEG_BIN_CA),
            Criterion(name="pbm", weight=5.0, requirement=REQ_POS_BIN_MET),
            Criterion(
                name="pmn",
                weight=8.0,
                requirement=REQ_POS_MC_NA,
                scale_type="ordinal",
                options=MC_OPTIONS,
            ),
            Criterion(
                name="nmn",
                weight=-6.0,
                requirement=REQ_NEG_MC_NA,
                scale_type="ordinal",
                options=MC_OPTIONS,
            ),
            Criterion(
                name="pmh",
                weight=7.0,
                requirement=REQ_POS_MC_HIGH,
                scale_type="ordinal",
                options=MC_OPTIONS,
            ),
        ]
    )


# The verdicts the mock encodes, in rubric order, expressed as Rubric.compute_score
# inputs (CriterionVerdict for binary, option label str for multi-choice). Includes
# CANNOT_ASSESS on BOTH binary signs and NA on BOTH multi-choice signs.
MIXED_VERDICTS: list[CriterionVerdict | str] = [
    CriterionVerdict.CANNOT_ASSESS,  # pbc (+10)
    CriterionVerdict.CANNOT_ASSESS,  # nbc (-4)
    CriterionVerdict.MET,  # pbm (+5)
    NA_LABEL,  # pmn (+8) -> NA
    NA_LABEL,  # nmn (-6) -> NA
    "high",  # pmh (+7) -> value 1.0
]


def _mock_client_for_mixed() -> MagicMock:
    """A per-criterion mock LLMClient returning the MIXED_VERDICTS deterministically.

    Keys off the unique requirement text present in the prompt (``<criterion>`` for
    binary, ``<question>`` for multi-choice). Returns GenerateResult with the parsed
    CriterionJudgment / MultiChoiceJudgment, mirroring the real grader's call path.
    """

    def _result(parsed: Any) -> GenerateResult:
        return GenerateResult(
            content="{}",
            thinking=None,
            raw_response=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=0.001,
            parsed=parsed,
        )

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult:
        # Binary criteria
        if REQ_POS_BIN_CA in user_prompt:
            return _result(
                CriterionJudgment(criterion_status=CriterionVerdict.CANNOT_ASSESS, explanation="x")
            )
        if REQ_NEG_BIN_CA in user_prompt:
            return _result(
                CriterionJudgment(criterion_status=CriterionVerdict.CANNOT_ASSESS, explanation="x")
            )
        if REQ_POS_BIN_MET in user_prompt:
            return _result(
                CriterionJudgment(criterion_status=CriterionVerdict.MET, explanation="x")
            )
        # Multi-choice criteria (shuffle disabled -> selected_option is 1-indexed
        # declaration position).
        if REQ_POS_MC_NA in user_prompt:
            return _result(MultiChoiceJudgment(selected_option=NA_OPTION_NUMBER, explanation="x"))
        if REQ_NEG_MC_NA in user_prompt:
            return _result(MultiChoiceJudgment(selected_option=NA_OPTION_NUMBER, explanation="x"))
        if REQ_POS_MC_HIGH in user_prompt:
            return _result(MultiChoiceJudgment(selected_option=3, explanation="x"))  # "high"
        raise AssertionError(f"Unexpected criterion in prompt:\n{user_prompt}")

    client = MagicMock()
    client.generate = AsyncMock(side_effect=mock_generate)
    return client


ALL_STRATEGIES = [
    CannotAssessStrategy.SKIP,
    CannotAssessStrategy.ZERO,
    CannotAssessStrategy.PARTIAL,
    CannotAssessStrategy.FAIL,
]


# ===========================================================================
# Cross-check: grader score == Rubric.compute_score, all strategies, both signs
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
async def test_grader_matches_compute_score_all_strategies(strategy):
    """The live grader and Rubric.compute_score agree for the SAME verdicts under
    every CannotAssessStrategy, across +/- binary CA and +/- multi-choice NA.

    This is the audit's explicit cross-path requirement (T4-A): both paths route
    through score_reports, so they must produce identical normalized and raw scores.
    """
    rubric = _mixed_rubric()

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_mock_client_for_mixed(),
    ):
        grader = CriterionGrader(
            llm_config=LLMConfig(model="test-model"),
            cannot_assess_config=CannotAssessConfig(strategy=strategy),
            shuffle_options=False,
        )
        report = await rubric.grade("submission", grader=grader)

    expected_norm = rubric.compute_score(
        MIXED_VERDICTS, normalize=True, cannot_assess_strategy=strategy
    )
    expected_raw = rubric.compute_score(
        MIXED_VERDICTS, normalize=False, cannot_assess_strategy=strategy
    )

    assert report.score == pytest.approx(expected_norm)
    assert report.raw_score == pytest.approx(expected_raw)


# ===========================================================================
# T1-B: weight-sign-aware multi-choice FAIL penalization (FAIL < ZERO)
# ===========================================================================


def test_negative_multi_choice_na_fail_strictly_worse_than_zero():
    """A -weight multi-choice NA under FAIL scores strictly below the same under ZERO.

    Under ZERO the NA contributes 0 (no penalty). Under FAIL, worst_scored_option()
    selects the highest-value scored option for a negative weight, so it subtracts a
    real penalty -- the T1-B weight-sign-aware fix. FAIL must therefore be < ZERO.
    """
    rubric = Rubric(
        [
            Criterion(name="pos", weight=10.0, requirement="assessable positive"),
            Criterion(
                name="neg",
                weight=-6.0,
                requirement="negative multi-choice NA",
                scale_type="ordinal",
                options=MC_OPTIONS,
            ),
        ]
    )
    verdicts: list[CriterionVerdict | str] = [CriterionVerdict.MET, NA_LABEL]

    fail_score = rubric.compute_score(
        verdicts, normalize=True, cannot_assess_strategy=CannotAssessStrategy.FAIL
    )
    zero_score = rubric.compute_score(
        verdicts, normalize=True, cannot_assess_strategy=CannotAssessStrategy.ZERO
    )

    assert fail_score < zero_score
    # ZERO: 10 / 10 = 1.0 (NA contributes nothing, no penalty).
    assert zero_score == pytest.approx(1.0)
    # FAIL: NA -> highest scored value 1.0 * -6 = -6; (10 - 6) / 10 = 0.4.
    assert fail_score == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_grader_negative_multi_choice_na_fail_strictly_worse_than_zero():
    """Same FAIL < ZERO inequality, proved through the live grader path (T1-B)."""
    rubric = Rubric(
        [
            Criterion(name="pos", weight=10.0, requirement=REQ_POS_BIN_MET),
            Criterion(
                name="neg",
                weight=-6.0,
                requirement=REQ_NEG_MC_NA,
                scale_type="ordinal",
                options=MC_OPTIONS,
            ),
        ]
    )

    async def grade_with(strategy: CannotAssessStrategy) -> float:
        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=_mock_client_for_mixed(),
        ):
            grader = CriterionGrader(
                llm_config=LLMConfig(model="test-model"),
                cannot_assess_config=CannotAssessConfig(strategy=strategy),
                shuffle_options=False,
            )
            report = await rubric.grade("submission", grader=grader)
        return report.score

    fail_score = await grade_with(CannotAssessStrategy.FAIL)
    zero_score = await grade_with(CannotAssessStrategy.ZERO)

    assert fail_score < zero_score
    assert zero_score == pytest.approx(1.0)
    assert fail_score == pytest.approx(0.4)


# ===========================================================================
# Rubric.compute_score SKIP regression (dual-denominator bug fix)
# ===========================================================================


def test_compute_score_skip_excludes_cannot_assess_from_denominator_binary():
    """Binary: SKIP must exclude the CA criterion from BOTH numerator and denominator.

    MET(w=5) + CANNOT_ASSESS(w=10) + UNMET(w=3) under SKIP -> 5/8 = 0.625.
    The old double-subtraction bug returned ~0.0 here.
    """
    rubric = Rubric(
        [
            Criterion(name="a", weight=5.0, requirement="A"),
            Criterion(name="b", weight=10.0, requirement="B"),
            Criterion(name="c", weight=3.0, requirement="C"),
        ]
    )
    verdicts = [
        CriterionVerdict.MET,
        CriterionVerdict.CANNOT_ASSESS,
        CriterionVerdict.UNMET,
    ]
    score = rubric.compute_score(
        verdicts, normalize=True, cannot_assess_strategy=CannotAssessStrategy.SKIP
    )
    assert score == pytest.approx(5.0 / 8.0)
    assert score == pytest.approx(0.625)
    assert score != pytest.approx(0.0)


def test_compute_score_skip_excludes_na_from_denominator_multi_choice():
    """Multi-choice NA analog: SKIP excludes the NA criterion from the denominator.

    MET(binary, w=5) + NA(multi-choice, w=10) + high(multi-choice, w=3) under SKIP
    -> (5 + 1.0*3) / (5 + 3) = 8/8 = 1.0 (NA's weight 10 excluded from denom).
    """
    rubric = Rubric(
        [
            Criterion(name="a", weight=5.0, requirement="A"),
            Criterion(
                name="b",
                weight=10.0,
                requirement="B",
                scale_type="ordinal",
                options=MC_OPTIONS,
            ),
            Criterion(
                name="c",
                weight=3.0,
                requirement="C",
                scale_type="ordinal",
                options=MC_OPTIONS,
            ),
        ]
    )
    verdicts: list[CriterionVerdict | str] = [CriterionVerdict.MET, NA_LABEL, "high"]
    score = rubric.compute_score(
        verdicts, normalize=True, cannot_assess_strategy=CannotAssessStrategy.SKIP
    )
    assert score == pytest.approx(1.0)
