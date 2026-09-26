"""Mapping a decision model's answers to per-criterion reports, and its confidence.

Each answer becomes one ``CriterionReport``: a verdict (binary) or a selected option
(multi-choice), the answer's ``probabilities`` re-keyed to AutoRubric's scheme, and the
``confidence`` AutoRubric computes for the selected outcome. A decision model gives no
explanation, so ``reason`` and ``reasoning`` are ``None``. A missing or malformed answer
raises ``ValueError``, which ``classify_grading_error`` routes as a ``parse`` failure.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

from autorubric import (
    Criterion,
    CriterionOption,
    CriterionVerdict,
    DecisionModelConfig,
    classify_grading_error,
)
from autorubric.decision import answer_to_report, selection_confidence

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CANNOT_ASSESS = CriterionVerdict.CANNOT_ASSESS


def config(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(**{"model": "jev-latest", **overrides})


def answers(**by_id: dict[str, Any]) -> dict[str, Any]:
    """SDK answer objects keyed by question id, decoded from their wire form like a response."""
    payload = {"model": "jev-test", "usage": {"input_tokens": 1}, "answers": by_id}
    return SystemOneResponse.model_validate_json(json.dumps(payload)).answers


def noul(p: float) -> dict[str, Any]:
    return {"type": "noul", "noul": p}


def choice(selected: str, probabilities: dict[str, float], confidence: float = 0.5) -> dict:
    return {
        "type": "choice",
        "choice": selected,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def score(expected: float, probabilities: list[float], confidence: float = 0.5) -> dict:
    return {
        "type": "score",
        "score": expected,
        "confidence": confidence,
        "legend": {str(i): f"level {i}" for i in range(len(probabilities))},
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
    }


BINARY = Criterion(name="light", weight=5.0, requirement="Mentions light")
NEGATIVE = Criterion(name="myth", weight=-5.0, requirement="Claims plants eat soil")


def ordinal_with_na_in_the_middle(weight: float = 3.0) -> Criterion:
    return Criterion(
        name="clarity",
        weight=weight,
        requirement="How clear is it?",
        options=[
            CriterionOption(label="Unclear", value=0.0),
            CriterionOption(label="Not an explanation", value=0.0, na=True),
            CriterionOption(label="Mostly clear", value=0.6),
            CriterionOption(label="Very clear", value=1.0),
        ],
        scale_type="ordinal",
    )


NOMINAL = Criterion(
    name="tone",
    weight=2.0,
    requirement="Which tone?",
    options=[
        CriterionOption(label="Formal", value=1.0),
        CriterionOption(label="Casual", value=0.5),
        CriterionOption(label="Hostile", value=0.0),
    ],
    scale_type="nominal",
).with_guaranteed_na_option()


def report_for(criterion: Criterion, answer: dict[str, Any], cfg: DecisionModelConfig, idx=0):
    return answer_to_report(criterion, idx, answers(**{f"c{idx}": answer}), cfg)


# =============================================================================
# Noul (binary "noul" and "noul_framed")
# =============================================================================


class TestNoul:
    @pytest.mark.parametrize("framing", ["noul", "noul_framed"])
    def test_above_the_threshold_is_met(self, framing):
        report = report_for(BINARY, noul(0.83), config(binary_framing=framing))
        assert report.verdict is MET
        assert report.probabilities == {"MET": 0.83, "UNMET": pytest.approx(0.17)}
        assert list(report.probabilities) == ["MET", "UNMET"]
        assert report.confidence == pytest.approx(0.66)

    def test_below_the_threshold_is_unmet(self):
        report = report_for(BINARY, noul(0.2), config())
        assert report.verdict is UNMET
        assert report.probabilities == {"MET": 0.2, "UNMET": 0.8}
        assert report.confidence == pytest.approx(0.6)

    @pytest.mark.parametrize(("weight", "expected"), [(5.0, UNMET), (0.0, UNMET), (-5.0, MET)])
    def test_exact_tie_resolves_to_the_weight_sign_worst_case(self, weight, expected):
        """At exactly ``p == decision_threshold`` neither side wins: the verdict is the
        score-minimizing one, the same rule as an ensemble tie."""
        criterion = Criterion(weight=weight, requirement="r")
        report = report_for(criterion, noul(0.5), config())
        assert report.verdict is expected
        assert report.confidence == 0.0

    @pytest.mark.parametrize(("weight", "expected"), [(1.0, UNMET), (-1.0, MET)])
    def test_exact_tie_at_a_non_default_threshold(self, weight, expected):
        criterion = Criterion(weight=weight, requirement="r")
        report = report_for(criterion, noul(0.7), config(decision_threshold=0.7))
        assert report.verdict is expected
        # UNMET is issued with P(UNMET) = 0.3 < 1/2: no support, clamped to 0.
        # MET is issued with P(MET) = 0.7: (2 * 0.7 - 1) / 1 = 0.4.
        assert report.confidence == pytest.approx(0.0 if expected is UNMET else 0.4)

    @pytest.mark.parametrize(
        ("threshold", "p", "verdict", "confidence"),
        [
            (0.7, 0.6, UNMET, 0.0),  # UNMET with P(UNMET) = 0.4: clamped
            (0.7, 0.9, MET, 0.8),
            (0.3, 0.4, MET, 0.0),  # MET with P(MET) = 0.4: clamped
            (0.3, 0.1, UNMET, 0.8),
        ],
    )
    def test_non_default_threshold_confidence_is_for_the_issued_verdict(
        self, threshold, p, verdict, confidence
    ):
        report = report_for(BINARY, noul(p), config(decision_threshold=threshold))
        assert report.verdict is verdict
        assert report.confidence == pytest.approx(confidence)

    @pytest.mark.parametrize(("p", "verdict"), [(1.0, MET), (0.0, UNMET)])
    def test_certain_answers(self, p, verdict):
        report = report_for(BINARY, noul(p), config())
        assert report.verdict is verdict
        assert report.confidence == 1.0

    def test_negative_criterion_met_is_p_yes(self):
        """For a negative criterion the yes outcome is still MET (the error is made)."""
        report = report_for(NEGATIVE, noul(0.9), config())
        assert report.verdict is MET
        assert report.probabilities == {"MET": 0.9, "UNMET": pytest.approx(0.1)}

    def test_the_report_is_the_criterion_plus_the_verdict_and_nothing_invented(self):
        report = report_for(BINARY, noul(0.83), config())
        assert (report.name, report.weight, report.requirement) == ("light", 5.0, "Mentions light")
        assert report.options is None
        assert report.multi_choice_verdict is None
        assert report.reason is None
        assert report.reasoning is None
        assert report.error is None
        assert report.shuffle_order is None


# =============================================================================
# Binary Choice
# =============================================================================


class TestBinaryChoice:
    PROBS = {"UNMET": 0.2, "CANNOT_ASSESS": 0.1, "MET": 0.7}

    @pytest.mark.parametrize("selected", ["MET", "UNMET", "CANNOT_ASSESS"])
    def test_the_selected_option_is_the_verdict(self, selected):
        report = report_for(BINARY, choice(selected, self.PROBS), config(binary_framing="choice"))
        assert report.verdict is CriterionVerdict(selected)

    def test_cannot_assess_is_an_abstention(self):
        report = report_for(
            BINARY, choice("CANNOT_ASSESS", self.PROBS), config(binary_framing="choice")
        )
        assert report.is_na

    def test_probabilities_are_keyed_by_verdict_value_in_verdict_order(self):
        report = report_for(BINARY, choice("MET", self.PROBS), config(binary_framing="choice"))
        assert list(report.probabilities.items()) == [
            ("MET", 0.7),
            ("UNMET", 0.2),
            ("CANNOT_ASSESS", 0.1),
        ]

    def test_confidence_is_typesafes_three_option_formula(self):
        """(3 x largest probability - 1) / 2 for the selected, most probable option."""
        report = report_for(BINARY, choice("MET", self.PROBS), config(binary_framing="choice"))
        assert report.confidence == pytest.approx((3 * 0.7 - 1) / 2)

    def test_the_threshold_does_not_apply(self):
        report = report_for(
            BINARY,
            choice("MET", {"MET": 0.6, "UNMET": 0.3, "CANNOT_ASSESS": 0.1}),
            config(binary_framing="choice", decision_threshold=0.9),
        )
        assert report.verdict is MET


# =============================================================================
# Multi-choice Choice
# =============================================================================


NOMINAL_PROBS = {
    "Casual": 0.1,
    "Cannot assess / not applicable": 0.05,
    "Formal": 0.8,
    "Hostile": 0.05,
}


class TestMultiChoiceChoice:
    def test_the_selected_label_is_the_option(self):
        report = report_for(NOMINAL, choice("Formal", NOMINAL_PROBS), config())
        verdict = report.multi_choice_verdict
        assert (verdict.selected_index, verdict.selected_label, verdict.value, verdict.na) == (
            0,
            "Formal",
            1.0,
            False,
        )
        assert report.verdict is None

    def test_probabilities_are_keyed_by_original_option_index_for_every_option(self):
        report = report_for(NOMINAL, choice("Formal", NOMINAL_PROBS), config())
        assert list(report.probabilities.items()) == [
            ("0", 0.8),
            ("1", 0.1),
            ("2", 0.05),
            ("3", 0.05),
        ]

    def test_confidence_counts_every_option_offered_including_na(self):
        report = report_for(NOMINAL, choice("Formal", NOMINAL_PROBS), config())
        assert report.confidence == pytest.approx((4 * 0.8 - 1) / 3)

    def test_the_na_option_is_an_abstention(self):
        probs = {**NOMINAL_PROBS, "Cannot assess / not applicable": 0.8, "Formal": 0.05}
        report = report_for(NOMINAL, choice("Cannot assess / not applicable", probs), config())
        verdict = report.multi_choice_verdict
        assert (verdict.selected_index, verdict.na, verdict.value) == (3, True, 0.0)
        assert report.is_na

    def test_ordinal_choice_selects_by_label_and_keeps_the_na_option(self):
        probs = {"Unclear": 0.1, "Not an explanation": 0.0, "Mostly clear": 0.7, "Very clear": 0.2}
        report = report_for(
            ordinal_with_na_in_the_middle(), choice("Mostly clear", probs), config()
        )
        assert report.multi_choice_verdict.selected_index == 2
        assert report.multi_choice_verdict.value == 0.6
        assert report.probabilities == {"0": 0.1, "1": 0.0, "2": 0.7, "3": 0.2}

    def test_value_is_the_selected_options_value(self):
        for label, index in (("Formal", 0), ("Casual", 1), ("Hostile", 2)):
            report = report_for(NOMINAL, choice(label, NOMINAL_PROBS), config())
            assert report.multi_choice_verdict.value == NOMINAL.options[index].value


# =============================================================================
# Score
# =============================================================================


def ladder(weight: float = 3.0, values: tuple[float, ...] = (0.0, 0.5, 1.0)) -> Criterion:
    return Criterion(
        weight=weight,
        requirement="How good?",
        options=[CriterionOption(label=f"L{i}", value=v) for i, v in enumerate(values)],
        scale_type="ordinal",
    )


SCORE = config(ordinal_framing="score")


class TestScore:
    @pytest.mark.parametrize(("expected", "level"), [(0.2, 0), (0.8, 1), (1.2, 1), (1.7, 2)])
    def test_the_expected_score_snaps_to_the_nearest_level(self, expected, level):
        report = report_for(ladder(), score(expected, [0.3, 0.4, 0.3]), SCORE)
        assert report.multi_choice_verdict.selected_index == level

    def test_levels_map_to_original_option_indices_around_the_na_option(self):
        """Levels are the non-NA options in order: 0 -> option 0, 1 -> 2, 2 -> 3."""
        criterion = ordinal_with_na_in_the_middle()
        report = report_for(criterion, score(1.1, [0.1, 0.7, 0.2]), SCORE)
        verdict = report.multi_choice_verdict
        assert (verdict.selected_index, verdict.selected_label, verdict.value) == (
            2,
            "Mostly clear",
            0.6,
        )
        assert list(report.probabilities.items()) == [("0", 0.1), ("2", 0.7), ("3", 0.2)]

    def test_the_na_option_is_absent_from_probabilities_not_zero(self):
        report = report_for(ordinal_with_na_in_the_middle(), score(1.0, [0.1, 0.8, 0.1]), SCORE)
        assert "1" not in report.probabilities

    def test_score_cannot_abstain(self):
        criterion = ordinal_with_na_in_the_middle()
        for expected in (0.0, 0.5, 1.0, 1.5, 2.0):
            report = report_for(criterion, score(expected, [0.3, 0.4, 0.3]), SCORE)
            assert not report.multi_choice_verdict.na
            assert not report.is_na

    @pytest.mark.parametrize(
        ("weight", "expected", "level"),
        [
            (3.0, 0.5, 0),
            (3.0, 1.5, 2),
            (-3.0, 0.5, 0),
            (-3.0, 1.5, 2),
        ],
    )
    def test_an_exact_half_snaps_with_round_half_to_even_whatever_the_weight(
        self, weight, expected, level
    ):
        """The level is ``round(expected)``, Python's half-to-even rounding: 0.5 goes down
        and 1.5 goes up, for either weight sign. It is not the ensemble tie rule."""
        report = report_for(ladder(weight), score(expected, [0.25, 0.5, 0.25]), SCORE)
        assert report.multi_choice_verdict.selected_index == level

    @pytest.mark.parametrize(
        ("expected", "level"),
        [(1.5000000001, 2), (1.4999999999, 1), (2.5, 2), (3.5, 4), (0.5000000001, 1)],
    )
    def test_the_level_is_exactly_round_of_the_expected_score(self, expected, level):
        """No tolerance band around a half: the level is ``round(s)`` for any ``s``."""
        five_levels = ladder(values=(0.0, 0.25, 0.5, 0.75, 1.0))
        report = report_for(five_levels, score(expected, [0.2] * 5), SCORE)
        assert report.multi_choice_verdict.selected_index == level == round(expected)

    def test_the_snap_follows_level_order_not_option_values(self):
        descending = ladder(values=(1.0, 0.5, 0.0))
        report = report_for(descending, score(0.5, [0.5, 0.5, 0.0]), SCORE)
        assert report.multi_choice_verdict.selected_index == 0  # round(0.5) == 0

    @pytest.mark.parametrize(("expected", "level"), [(-0.4, 0), (2.4, 2), (7.0, 2)])
    def test_the_expected_score_is_clamped_to_the_scale(self, expected, level):
        report = report_for(ladder(), score(expected, [0.3, 0.4, 0.3]), SCORE)
        assert report.multi_choice_verdict.selected_index == level

    def test_value_is_the_snapped_options_value_never_the_fractional_score(self):
        report = report_for(ladder(), score(1.3, [0.1, 0.5, 0.4]), SCORE)
        assert report.multi_choice_verdict.value == 0.5

    def test_confidence_is_for_the_snapped_level_over_the_non_na_levels(self):
        report = report_for(ordinal_with_na_in_the_middle(), score(1.1, [0.1, 0.7, 0.2]), SCORE)
        assert report.confidence == pytest.approx((3 * 0.7 - 1) / 2)

    def test_a_snapped_level_with_little_support_has_zero_confidence(self):
        """A bimodal answer's expected score lands on a level the model barely supports."""
        report = report_for(ladder(), score(1.0, [0.45, 0.1, 0.45]), SCORE)
        assert report.multi_choice_verdict.selected_index == 1
        assert report.confidence == 0.0

    def test_nominal_criteria_are_never_scored(self):
        report = report_for(NOMINAL, choice("Casual", NOMINAL_PROBS), SCORE)
        assert report.multi_choice_verdict.selected_index == 1


# =============================================================================
# Malformed answers: per-criterion parse errors
# =============================================================================


def assert_parse_error(match: str, criterion, answers_by_id, cfg, idx=0) -> None:
    with pytest.raises(ValueError, match=match) as excinfo:
        answer_to_report(criterion, idx, answers_by_id, cfg)
    assert classify_grading_error(excinfo.value) == "parse"


class TestMalformedAnswers:
    def test_missing_answer_id(self):
        assert_parse_error("no answer for criterion c3", BINARY, answers(c0=noul(0.9)), config(), 3)

    @pytest.mark.parametrize(
        ("criterion", "cfg", "answer", "expected", "got"),
        [
            (
                BINARY,
                config(),
                choice("MET", {"MET": 1.0, "UNMET": 0.0, "CANNOT_ASSESS": 0.0}),
                "noul",
                "choice",
            ),
            (BINARY, config(binary_framing="choice"), noul(0.9), "choice", "noul"),
            (ladder(), config(), score(1.0, [0.2, 0.6, 0.2]), "choice", "score"),
            (ladder(), SCORE, choice("L0", {"L0": 1.0, "L1": 0.0, "L2": 0.0}), "score", "choice"),
        ],
    )
    def test_wrong_answer_type(self, criterion, cfg, answer, expected, got):
        assert_parse_error(
            f"expected a {expected} answer, got {got!r}", criterion, answers(c0=answer), cfg
        )

    def test_binary_choice_outside_the_verdicts(self):
        answer = choice("MAYBE", {"MET": 0.5, "UNMET": 0.5, "CANNOT_ASSESS": 0.0})
        assert_parse_error("'MAYBE'", BINARY, answers(c0=answer), config(binary_framing="choice"))

    def test_multi_choice_label_not_offered(self):
        probs = {**NOMINAL_PROBS}
        assert_parse_error("'Sarcastic'", NOMINAL, answers(c0=choice("Sarcastic", probs)), config())

    def test_labels_match_exactly(self):
        assert_parse_error(
            "'formal'", NOMINAL, answers(c0=choice("formal", NOMINAL_PROBS)), config()
        )

    @pytest.mark.parametrize(
        "probs",
        [
            {"MET": 0.7, "UNMET": 0.3},  # a verdict offered but missing
            {"MET": 0.7, "UNMET": 0.2, "CANNOT_ASSESS": 0.1, "OTHER": 0.0},  # not offered
        ],
    )
    def test_probabilities_must_cover_exactly_the_offered_outcomes(self, probs):
        assert_parse_error(
            "probabilities",
            BINARY,
            answers(c0=choice("MET", probs)),
            config(binary_framing="choice"),
        )

    def test_score_probabilities_must_cover_exactly_the_levels(self):
        assert_parse_error("probabilities", ladder(), answers(c0=score(1.0, [0.5, 0.5])), SCORE)

    @pytest.mark.parametrize("bad", [1.2, -0.1, math.nan, math.inf])
    def test_a_probability_outside_zero_to_one_is_malformed(self, bad):
        answer = ChoiceAnswer.model_construct(
            type="choice",
            choice="MET",
            confidence=0.5,
            probabilities={"MET": bad, "UNMET": 0.2, "CANNOT_ASSESS": 0.1},
        )
        assert_parse_error("probabilit", BINARY, {"c0": answer}, config(binary_framing="choice"))

    @pytest.mark.parametrize("bad", [1.5, -0.2, math.nan, True])
    def test_a_noul_outside_zero_to_one_is_malformed(self, bad):
        answer = NoulAnswer.model_construct(type="noul", noul=bad)
        assert_parse_error("probability", BINARY, {"c0": answer}, config())

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_a_non_finite_expected_score_is_malformed(self, bad):
        answer = ScoreAnswer.model_construct(
            type="score",
            score=bad,
            confidence=0.5,
            legend={0: "L0", 1: "L1", 2: "L2"},
            probabilities={0: 0.2, 1: 0.6, 2: 0.2},
        )
        assert_parse_error("score", ladder(), {"c0": answer}, SCORE)

    def test_an_inexpressible_criterion_has_no_mapping(self):
        duplicates = Criterion(
            requirement="Which?",
            options=[
                CriterionOption(label="Same", value=0.0),
                CriterionOption(label="Same", value=1.0),
            ],
            scale_type="nominal",
        )
        answer = choice("Same", {"Same": 1.0})
        assert_parse_error("not unique", duplicates, answers(c0=answer), config())


# =============================================================================
# Confidence
# =============================================================================


class TestSelectionConfidence:
    @pytest.mark.parametrize(
        ("p", "n", "expected"),
        [
            (0.5, 2, 0.0),
            (0.75, 2, 0.5),
            (1.0, 2, 1.0),
            (0.0, 2, 0.0),
            (1 / 3, 3, 0.0),
            (0.9, 3, 0.85),  # TypeSafe's worked example: probabilities 0.90 / 0.06 / 0.04
            (0.6, 5, 0.5),
            (1.0, 255, 1.0),
        ],
    )
    def test_formula(self, p, n, expected):
        assert selection_confidence(p, n) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize(("p", "n"), [(0.2, 3), (0.1, 2), (0.0, 10)])
    def test_below_chance_clamps_to_zero(self, p, n):
        assert selection_confidence(p, n) == 0.0

    def test_equals_twice_the_distance_from_one_half_for_noul_at_the_default_threshold(self):
        for p in (0.5, 0.62, 0.97):
            assert selection_confidence(p, 2) == pytest.approx(2 * abs(p - 0.5))

    @pytest.mark.parametrize(("p", "n"), [(0.5, 1), (0.5, 0), (1.2, 3), (-0.1, 3), (math.nan, 3)])
    def test_rejects_invalid_input(self, p, n):
        with pytest.raises(ValueError):
            selection_confidence(p, n)
