"""Regression against real decision-model outputs recorded before the library had this path.

``fixtures/recorded_jev.json`` was derived offline from recorded runs of TypeSafe's Jev
through a reference grader that kept every raw answer (see its ``about`` field). It pins:

- the questions that grader built for two rubrics under every framing, which the library
  must reproduce exactly (same framing, same wire form, same order);
- raw answers and the outcomes that grader recorded, which the library must reproduce
  except at an exact Noul tie (P(yes) equal to the threshold), where it deliberately
  applies the ensemble tie rule instead. Score answers are snapped with ``round``, as that
  grader snapped them, so every recorded Score outcome is reproduced, halves included.

No test here makes a request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typesafe_sdk import SystemOneResponse

from autorubric import Criterion, DecisionModelConfig
from autorubric.decision import answer_to_report, build_questions, build_state

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded_jev.json").read_text(encoding="utf-8")
)

# The endpoint reports probabilities rounded to two decimals and computes its own Choice
# confidence from the unrounded distribution, so the library's confidence, computed from
# the reported probabilities, agrees only up to that rounding (scaled by K / (K - 1)).
REPORTED_CONFIDENCE_TOLERANCE = 0.025


def config(case: dict[str, Any]) -> DecisionModelConfig:
    return DecisionModelConfig(
        model="jev-latest",
        binary_framing=case["binary_framing"],
        ordinal_framing=case["ordinal_framing"],
    )


def decoded_answers(raw: dict[str, Any]) -> dict[str, Any]:
    payload = {"model": "jev-1.13.0", "usage": {"input_tokens": 1}, "answers": {"c0": raw}}
    return SystemOneResponse.model_validate_json(json.dumps(payload)).answers


def test_the_state_matches_the_recorded_graders_state():
    state = FIXTURE["state"]
    built = build_state(state["submission"], query=state["input"])
    assert list(built.items()) == list(FIXTURE["jev_state"].items())


@pytest.mark.parametrize(
    "case",
    FIXTURE["questions"],
    ids=lambda c: f"{c['rubric']}-{c['binary_framing']}-{c['ordinal_framing']}",
)
def test_questions_match_the_recorded_graders_questions_exactly(case):
    rubric = [Criterion(**criterion) for criterion in FIXTURE["rubrics"][case["rubric"]]]
    questions, errors = build_questions(rubric, config(case), FIXTURE["state"])
    assert errors == {}
    built = [[qid, question.model_dump(mode="json")] for qid, question in questions.items()]
    # Compared as JSON text, so key order (the order options are offered in) counts too.
    assert json.dumps(built) == json.dumps(case["questions"])


def is_tie(raw: dict[str, Any], cfg: DecisionModelConfig) -> bool:
    """A Noul answer exactly at the threshold: the one outcome the library maps differently."""
    return raw["type"] == "noul" and raw["noul"] == cfg.decision_threshold


def is_half(raw: dict[str, Any]) -> bool:
    """A Score answer exactly halfway between two levels."""
    return raw["type"] == "score" and raw["score"] % 1 == 0.5


@pytest.mark.parametrize("case", FIXTURE["answers"], ids=lambda c: c["id"])
def test_recorded_answers_map_to_the_recorded_outcomes(case):
    criterion = Criterion(**case["criterion"])
    cfg = config(case)
    raw = case["answer"]
    report = answer_to_report(criterion, 0, decoded_answers(raw), cfg)
    if criterion.options is None:
        outcome: Any = report.verdict.value
    else:
        outcome = report.multi_choice_verdict.selected_index

    if not is_tie(raw, cfg):
        assert outcome == case["recorded"]
    else:
        # Recorded: P(yes) >= threshold was MET. Library: the weight-sign worst case.
        assert case["recorded"] == "MET"
        assert criterion.weight > 0 and outcome == "UNMET"

    assert report.reason is None and report.reasoning is None


@pytest.mark.parametrize("case", FIXTURE["answers"], ids=lambda c: c["id"])
def test_recorded_answers_probability_keys_and_confidence(case):
    criterion = Criterion(**case["criterion"])
    raw = case["answer"]
    report = answer_to_report(criterion, 0, decoded_answers(raw), config(case))
    options = criterion.options or []
    if raw["type"] == "noul":
        assert list(report.probabilities) == ["MET", "UNMET"]
        p_met = raw["noul"]
        assert report.confidence == pytest.approx(2 * abs(p_met - 0.5))
    elif raw["type"] == "score":
        assert list(report.probabilities) == [str(i) for i, o in enumerate(options) if not o.na]
    else:
        expected_keys = (
            ["MET", "UNMET", "CANNOT_ASSESS"]
            if criterion.options is None
            else [str(i) for i in range(len(options))]
        )
        assert list(report.probabilities) == expected_keys
        assert abs(report.confidence - raw["confidence"]) <= REPORTED_CONFIDENCE_TOLERANCE


def test_the_fixture_covers_every_answer_kind_a_noul_tie_and_score_halves():
    """Guards the fixture itself: a regeneration must keep these cases."""
    kinds = {(c["answer"]["type"], bool(c["criterion"].get("options"))) for c in FIXTURE["answers"]}
    assert kinds == {("noul", False), ("choice", False), ("choice", True), ("score", True)}
    ties = [c for c in FIXTURE["answers"] if is_tie(c["answer"], config(c))]
    assert {c["answer"]["type"] for c in ties} == {"noul"}
    halves = [c["answer"]["score"] for c in FIXTURE["answers"] if is_half(c["answer"])]
    # Half-to-even rounding goes down at 0.5 and up at 1.5: both directions are recorded,
    # so the recorded outcomes pin round rather than any fixed direction for halves.
    assert {round(s) > s for s in halves} == {True, False}
    na_selections = [
        c
        for c in FIXTURE["answers"]
        if c["criterion"].get("options") and c["criterion"]["options"][c["recorded"]].get("na")
    ]
    assert na_selections
