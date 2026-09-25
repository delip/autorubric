"""Framing goldens: the exact state and questions a decision model receives.

A decision model grades an item with one request: the material to judge is the shared
``state`` and each criterion of the effective rubric becomes one question with id
``c{criterion_idx}``. Criterion text is never rewritten: requirements and option labels are
sent verbatim, and only the structure around them differs by framing.

Expected values are written out literally (not rebuilt from the library's constants), so
any change to what a decision model is asked fails here.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from typesafe_sdk import Choice, Noul, Score

from autorubric import Criterion, CriterionOption, DecisionModelConfig, classify_grading_error
from autorubric.decision import build_question, build_questions, build_state, question_id
from autorubric.types import CANONICAL_NA_OPTION

MET_DEF = "The thing described in the criterion IS present in the submission"
UNMET_DEF = "The thing described in the criterion IS NOT present in the submission"
CA_DEF = "Insufficient evidence to determine either way (use rarely)"
NEG_MET_DEF = "The submission advocates, states, or recommends the problematic thing"
NEG_UNMET_DEF = "The submission does NOT make this error, OR mentions it only to warn against it"

TASK = "Determine whether this criterion is satisfied by the `submission`."
TASK_OUTPUT = (
    "Determine whether this criterion is satisfied by the `output`; the `thinking` is context only."
)
REFERENCE_SUBMISSION = (
    "Use the `reference_submission` only to calibrate expectations; judge the `submission` on "
    "its own merits, not by its resemblance to the reference."
)
REFERENCE_OUTPUT = (
    "Use the `reference_submission` only to calibrate expectations; judge the `output` on "
    "its own merits, not by its resemblance to the reference."
)

GUIDELINES_CLAUSE = "Apply the `guidelines`; the criterion text governs."

# The framed template of the experiments that motivated the framing, verbatim.
EXPERIMENTS_FRAMED_TEMPLATE = (
    "Determine whether this criterion is satisfied by the `submission`. Criterion: {requirement}"
)

REQ = "States that light is absorbed by chlorophyll"
NEG_REQ = "Claims that plants absorb mostly green light"

PLAIN = {"submission": "Chlorophyll absorbs red and blue light."}
WITH_REFERENCE = {
    "input": "How do plants use light?",
    "reference_submission": "Chlorophyll absorbs red and blue light and reflects green.",
    "submission": "Chlorophyll absorbs red and blue light.",
}
STRUCTURED = {"thinking": "Recall the pigments.", "output": "Chlorophyll absorbs light."}
STRUCTURED_WITH_REFERENCE = {
    "reference_submission": "Chlorophyll absorbs red and blue light.",
    "thinking": "Recall the pigments.",
    "output": "Chlorophyll absorbs light.",
}

GUIDELINES = "Writers are grade 8-12 learners; 'cited' means any attribution."
WITH_GUIDELINES = {
    "guidelines": GUIDELINES,
    "submission": "Chlorophyll absorbs red and blue light.",
}
WITH_GUIDELINES_AND_REFERENCE = {"guidelines": GUIDELINES, **WITH_REFERENCE}
STRUCTURED_WITH_GUIDELINES = {"guidelines": GUIDELINES, **STRUCTURED}
STRUCTURED_WITH_EVERY_FIELD = {
    "guidelines": GUIDELINES,
    "input": "How do plants use light?",
    **STRUCTURED_WITH_REFERENCE,
}
GUIDELINES_STATES = [
    WITH_GUIDELINES,
    WITH_GUIDELINES_AND_REFERENCE,
    STRUCTURED_WITH_GUIDELINES,
    STRUCTURED_WITH_EVERY_FIELD,
]


def config(**overrides: Any) -> DecisionModelConfig:
    return DecisionModelConfig(**{"model": "jev-latest", **overrides})


def binary(requirement: str = REQ, weight: float = 5.0, name: str | None = None) -> Criterion:
    return Criterion(weight=weight, requirement=requirement, name=name)


def ordinal(na: bool = False, requirement: str = "How clear is the explanation?") -> Criterion:
    options = [
        CriterionOption(label="Unclear", value=0.0),
        CriterionOption(label="Mostly clear", value=0.5),
        CriterionOption(label="Very clear", value=1.0),
    ]
    if na:
        options.insert(1, CriterionOption(label="Not an explanation", value=0.0, na=True))
    return Criterion(weight=3.0, requirement=requirement, options=options, scale_type="ordinal")


def nominal() -> Criterion:
    return Criterion(
        weight=2.0,
        requirement="Which tone does the answer take?",
        options=[
            CriterionOption(label="Formal", value=1.0),
            CriterionOption(label="Casual", value=0.5),
            CriterionOption(label="Hostile", value=0.0),
        ],
        scale_type="nominal",
    )


def wire(question: Any) -> dict[str, Any]:
    """A question's wire form, exactly as the SDK sends it."""
    return question.model_dump(mode="json")


def wire_items(questions: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Question ids and wire forms, in request order."""
    return [(qid, wire(q)) for qid, q in questions.items()]


# =============================================================================
# State
# =============================================================================


class TestState:
    def test_plain_submission_only(self):
        assert build_state("Chlorophyll absorbs light.") == {
            "submission": "Chlorophyll absorbs light."
        }

    def test_fields_appear_only_when_present_in_a_fixed_order(self):
        state = build_state("S", query="Q", reference_submission="R")
        assert list(state.items()) == [
            ("input", "Q"),
            ("reference_submission", "R"),
            ("submission", "S"),
        ]

    @pytest.mark.parametrize("absent", [None, ""])
    def test_absent_or_empty_optional_fields_are_left_out(self, absent):
        """Present means non-empty, the same rule the LLM user prompt applies."""
        assert build_state("S", query=absent, reference_submission=absent) == {"submission": "S"}

    @pytest.mark.parametrize(
        "text", ["  padded \n", "", "a <thinking> tag that never closes", "<think>x</think>"]
    )
    def test_plain_submission_is_sent_unchanged(self, text):
        """Without a section there is nothing to recover: the text goes out as-is, never
        stripped."""
        assert build_state(text) == {"submission": text}

    def test_structured_submission_is_split_into_thinking_and_output(self):
        state = build_state(
            "<thinking> Recall the pigments. </thinking>\n<output> Absorbs. </output>"
        )
        assert list(state.items()) == [("thinking", "Recall the pigments."), ("output", "Absorbs.")]

    def test_output_only_structure_still_splits(self):
        """``Grader.grade`` flattens ``{"output": ...}`` (no thinking) to an output section."""
        assert build_state("<output>Absorbs.</output>") == {"thinking": "", "output": "Absorbs."}

    def test_thinking_without_output_markers_takes_the_rest_as_output(self):
        assert build_state("<thinking>T</thinking> the rest ") == {
            "thinking": "T",
            "output": "the rest",
        }

    def test_structured_field_order_with_every_field(self):
        state = build_state(
            "<thinking>T</thinking><output>O</output>", query="Q", reference_submission="R"
        )
        assert list(state) == ["input", "reference_submission", "thinking", "output"]


# =============================================================================
# Binary framings
# =============================================================================


class TestBinaryFramings:
    def test_noul_is_the_bare_requirement(self):
        question = build_question(binary(), config(binary_framing="noul"), PLAIN)
        assert isinstance(question, Noul)
        assert wire(question) == {"type": "noul", "instructions": REQ}

    def test_noul_framed_is_the_default(self):
        assert config().binary_framing == "noul_framed"
        question = build_question(binary(), config(), PLAIN)
        assert isinstance(question, Noul)
        assert wire(question) == {
            "type": "noul",
            "instructions": f"{TASK} Criterion: {REQ}",
            "criteria": {"true": MET_DEF, "false": UNMET_DEF},
        }

    def test_choice_offers_the_three_verdicts_with_their_definitions(self):
        question = build_question(binary(), config(binary_framing="choice"), PLAIN)
        assert isinstance(question, Choice)
        assert wire(question) == {
            "type": "choice",
            "instructions": f"{TASK} Criterion: {REQ}",
            "criteria": {"MET": MET_DEF, "UNMET": UNMET_DEF, "CANNOT_ASSESS": CA_DEF},
        }
        assert list(wire(question)["criteria"]) == ["MET", "UNMET", "CANNOT_ASSESS"]

    def test_framed_instructions_reproduce_the_experiments_template(self):
        question = build_question(binary(), config(), PLAIN)
        assert question.instructions == EXPERIMENTS_FRAMED_TEMPLATE.format(requirement=REQ)

    @pytest.mark.parametrize("framing", ["noul_framed", "choice"])
    def test_negative_weight_selects_the_negative_definitions(self, framing):
        question = build_question(
            binary(NEG_REQ, weight=-4.0), config(binary_framing=framing), PLAIN
        )
        outcomes = wire(question)["criteria"]
        if framing == "noul_framed":
            assert outcomes == {"true": NEG_MET_DEF, "false": NEG_UNMET_DEF}
        else:
            assert outcomes == {"MET": NEG_MET_DEF, "UNMET": NEG_UNMET_DEF, "CANNOT_ASSESS": CA_DEF}
        assert question.instructions == f"{TASK} Criterion: {NEG_REQ}"

    def test_negative_weight_bare_noul_is_still_the_bare_requirement(self):
        question = build_question(
            binary(NEG_REQ, weight=-4.0), config(binary_framing="noul"), PLAIN
        )
        assert wire(question) == {"type": "noul", "instructions": NEG_REQ}

    def test_zero_weight_uses_the_positive_definitions(self):
        """Only a negative weight marks a negative criterion, as in the LLM user prompt."""
        question = build_question(binary(weight=0.0), config(), PLAIN)
        assert wire(question)["criteria"] == {"true": MET_DEF, "false": UNMET_DEF}

    @pytest.mark.parametrize("framing", ["noul_framed", "choice"])
    def test_reference_adds_its_usage_sentence_before_the_criterion(self, framing):
        question = build_question(binary(), config(binary_framing=framing), WITH_REFERENCE)
        assert question.instructions == f"{TASK} {REFERENCE_SUBMISSION} Criterion: {REQ}"

    def test_input_alone_adds_no_sentence(self):
        state = {"input": "How do plants use light?", "submission": "S"}
        question = build_question(binary(), config(), state)
        assert question.instructions == f"{TASK} Criterion: {REQ}"

    @pytest.mark.parametrize("framing", ["noul_framed", "choice"])
    def test_structured_submission_judges_the_output(self, framing):
        question = build_question(binary(), config(binary_framing=framing), STRUCTURED)
        assert question.instructions == f"{TASK_OUTPUT} Criterion: {REQ}"

    def test_structured_submission_with_reference_judges_the_output_against_it(self):
        question = build_question(binary(), config(), STRUCTURED_WITH_REFERENCE)
        assert question.instructions == f"{TASK_OUTPUT} {REFERENCE_OUTPUT} Criterion: {REQ}"

    @pytest.mark.parametrize("state", [WITH_REFERENCE, STRUCTURED, STRUCTURED_WITH_REFERENCE])
    def test_bare_noul_gets_no_clause_and_sees_extra_fields_only_through_the_state(self, state):
        question = build_question(binary(), config(binary_framing="noul"), state)
        assert wire(question) == {"type": "noul", "instructions": REQ}

    @pytest.mark.parametrize(
        "requirement",
        [
            "Uses {braces} and a trailing space ",
            "Multi-line\nrequirement\twith tabs",
            "Criterion: already labelled",
            "",
        ],
    )
    @pytest.mark.parametrize("state", [PLAIN, STRUCTURED_WITH_REFERENCE])
    def test_requirement_is_verbatim_and_last(self, requirement, state):
        question = build_question(binary(requirement), config(), state)
        assert question.instructions.endswith("Criterion: " + requirement)
        assert build_question(
            binary(requirement), config(binary_framing="noul"), state
        ).instructions == (requirement)


# =============================================================================
# Multi-choice framings
# =============================================================================


class TestMultiChoiceFramings:
    @pytest.mark.parametrize("ordinal_framing", ["choice", "score"])
    def test_nominal_is_always_a_choice_over_the_labels(self, ordinal_framing):
        question = build_question(nominal(), config(ordinal_framing=ordinal_framing), PLAIN)
        assert isinstance(question, Choice)
        assert wire(question) == {
            "type": "choice",
            "instructions": "Which tone does the answer take?",
            "criteria": {"Formal": None, "Casual": None, "Hostile": None},
        }

    def test_ordinal_choice_is_the_default_and_includes_the_na_option(self):
        assert config().ordinal_framing == "choice"
        question = build_question(ordinal(na=True), config(), PLAIN)
        assert isinstance(question, Choice)
        assert wire(question) == {
            "type": "choice",
            "instructions": "How clear is the explanation?",
            "criteria": {
                "Unclear": None,
                "Not an explanation": None,
                "Mostly clear": None,
                "Very clear": None,
            },
        }
        assert list(wire(question)["criteria"]) == [
            "Unclear",
            "Not an explanation",
            "Mostly clear",
            "Very clear",
        ]

    def test_ordinal_score_offers_the_non_na_levels_in_order(self):
        question = build_question(ordinal(na=True), config(ordinal_framing="score"), PLAIN)
        assert isinstance(question, Score)
        assert wire(question) == {
            "type": "score",
            "instructions": "How clear is the explanation?",
            "criteria": ["Unclear", "Mostly clear", "Very clear"],
        }

    def test_auto_injected_na_option_is_a_choice_key_but_never_a_score_level(self):
        effective = ordinal().with_guaranteed_na_option()
        choice = wire(build_question(effective, config(), PLAIN))
        assert list(choice["criteria"]) == [
            "Unclear",
            "Mostly clear",
            "Very clear",
            CANONICAL_NA_OPTION.label,
        ]
        score = wire(build_question(effective, config(ordinal_framing="score"), PLAIN))
        assert score["criteria"] == ["Unclear", "Mostly clear", "Very clear"]

    @pytest.mark.parametrize("state", [WITH_REFERENCE, STRUCTURED, STRUCTURED_WITH_REFERENCE])
    @pytest.mark.parametrize("ordinal_framing", ["choice", "score"])
    def test_multi_choice_gets_no_clause(self, state, ordinal_framing):
        question = build_question(ordinal(), config(ordinal_framing=ordinal_framing), state)
        assert question.instructions == "How clear is the explanation?"

    def test_labels_are_verbatim(self):
        criterion = Criterion(
            requirement="Pick one",
            options=[
                CriterionOption(label=" padded label ", value=0.0),
                CriterionOption(label="Label with {braces}", value=1.0),
            ],
            scale_type="nominal",
        )
        assert list(wire(build_question(criterion, config(), PLAIN))["criteria"]) == [
            " padded label ",
            "Label with {braces}",
        ]

    def test_binary_framing_does_not_affect_multi_choice(self):
        for framing in ("noul", "noul_framed", "choice"):
            question = build_question(nominal(), config(binary_framing=framing), PLAIN)
            assert wire(question)["criteria"] == {"Formal": None, "Casual": None, "Hostile": None}


# =============================================================================
# One request per rubric: ids, order, inexpressible criteria
# =============================================================================


def duplicate_labels() -> Criterion:
    return Criterion(
        name="tone",
        requirement="Which tone?",
        options=[
            CriterionOption(label="Formal", value=1.0),
            CriterionOption(label="Casual", value=0.5),
            CriterionOption(label="Formal", value=0.0),
        ],
        scale_type="nominal",
    )


def many_options(n: int, scale_type: str = "nominal") -> Criterion:
    return Criterion(
        requirement="Which category?",
        options=[CriterionOption(label=f"option {i}", value=i / n) for i in range(n)],
        scale_type=scale_type,
    )


class TestRubricQuestions:
    def test_question_ids_are_the_effective_rubric_indices(self):
        assert [question_id(i) for i in range(3)] == ["c0", "c1", "c2"]

    def test_one_question_per_criterion_in_rubric_order(self):
        rubric = [binary(), ordinal().with_guaranteed_na_option(), nominal(), binary(NEG_REQ, -1)]
        questions, errors = build_questions(rubric, config(), PLAIN)
        assert errors == {}
        assert list(questions) == ["c0", "c1", "c2", "c3"]
        assert wire_items(questions) == [
            (
                "c0",
                {
                    "type": "noul",
                    "instructions": f"{TASK} Criterion: {REQ}",
                    "criteria": {"true": MET_DEF, "false": UNMET_DEF},
                },
            ),
            (
                "c1",
                {
                    "type": "choice",
                    "instructions": "How clear is the explanation?",
                    "criteria": {
                        "Unclear": None,
                        "Mostly clear": None,
                        "Very clear": None,
                        "Cannot assess / not applicable": None,
                    },
                },
            ),
            (
                "c2",
                {
                    "type": "choice",
                    "instructions": "Which tone does the answer take?",
                    "criteria": {"Formal": None, "Casual": None, "Hostile": None},
                },
            ),
            (
                "c3",
                {
                    "type": "noul",
                    "instructions": f"{TASK} Criterion: {NEG_REQ}",
                    "criteria": {"true": NEG_MET_DEF, "false": NEG_UNMET_DEF},
                },
            ),
        ]

    def test_unnamed_and_duplicate_named_criteria_get_distinct_ids(self):
        rubric = [binary(name="same"), binary("Another requirement", name="same"), binary()]
        questions, _ = build_questions(rubric, config(), PLAIN)
        assert list(questions) == ["c0", "c1", "c2"]

    def test_duplicate_choice_labels_are_inexpressible_and_named(self):
        with pytest.raises(ValueError, match=r"not unique.*\['Formal'\]"):
            build_question(duplicate_labels(), config(), PLAIN)

    def test_auto_na_label_colliding_with_an_author_label_is_inexpressible(self):
        criterion = Criterion(
            requirement="Which?",
            options=[
                CriterionOption(label="Yes", value=1.0),
                CriterionOption(label="No", value=0.0),
                CriterionOption(label=CANONICAL_NA_OPTION.label, value=0.5),
            ],
            scale_type="nominal",
        ).with_guaranteed_na_option()
        with pytest.raises(ValueError, match="not unique"):
            build_question(criterion, config(), PLAIN)

    def test_choice_accepts_up_to_255_options(self):
        question = build_question(many_options(255), config(), PLAIN)
        assert len(wire(question)["criteria"]) == 255

    def test_choice_over_255_options_is_inexpressible(self):
        with pytest.raises(ValueError, match="256 options.*at most 255"):
            build_question(many_options(256), config(), PLAIN)

    def test_score_accepts_two_to_ten_levels(self):
        for n in (2, 10):
            question = build_question(
                many_options(n, "ordinal"), config(ordinal_framing="score"), PLAIN
            )
            assert len(wire(question)["criteria"]) == n

    def test_score_over_ten_levels_is_inexpressible(self):
        with pytest.raises(ValueError, match="11 non-NA levels.*2 to 10"):
            build_question(many_options(11, "ordinal"), config(ordinal_framing="score"), PLAIN)

    def test_eleven_ordinal_options_are_fine_as_a_choice(self):
        question = build_question(many_options(11, "ordinal"), config(), PLAIN)
        assert len(wire(question)["criteria"]) == 11

    def test_inexpressible_criteria_are_left_out_under_their_ids(self):
        rubric = [binary(), duplicate_labels(), ordinal(), many_options(11, "ordinal")]
        questions, errors = build_questions(rubric, config(ordinal_framing="score"), PLAIN)
        assert list(questions) == ["c0", "c2"]
        assert sorted(errors) == [1, 3]
        assert all(isinstance(e, ValueError) for e in errors.values())
        assert "c1" in str(errors[1]) and "'tone'" in str(errors[1])
        assert "c3" in str(errors[3])

    def test_build_errors_route_as_parse_failures(self):
        _, errors = build_questions([duplicate_labels(), many_options(256)], config(), PLAIN)
        assert [classify_grading_error(e) for e in errors.values()] == ["parse", "parse"]

    def test_all_inexpressible_gives_no_questions(self):
        questions, errors = build_questions([duplicate_labels()], config(), PLAIN)
        assert questions == {}
        assert list(errors) == [0]

    def test_empty_rubric_gives_no_questions(self):
        assert build_questions([], config(), PLAIN) == ({}, {})


# =============================================================================
# Rubric guidelines
# =============================================================================


class TestGuidelinesState:
    def test_guidelines_are_the_first_state_field(self):
        state = build_state("S", query="Q", reference_submission="R", guidelines="G")
        assert list(state.items()) == [
            ("guidelines", "G"),
            ("input", "Q"),
            ("reference_submission", "R"),
            ("submission", "S"),
        ]

    def test_guidelines_come_before_a_structured_submission(self):
        state = build_state(
            "<thinking>T</thinking><output>O</output>",
            query="Q",
            reference_submission="R",
            guidelines="G",
        )
        assert list(state.items()) == [
            ("guidelines", "G"),
            ("input", "Q"),
            ("reference_submission", "R"),
            ("thinking", "T"),
            ("output", "O"),
        ]

    def test_guidelines_alone_with_the_submission(self):
        assert build_state("S", guidelines="G") == {"guidelines": "G", "submission": "S"}

    @pytest.mark.parametrize("absent", [None, "", "  \n\t"], ids=["none", "empty", "blank"])
    def test_absent_or_blank_guidelines_are_left_out(self, absent):
        """Blank guidelines mean none, as for ``Rubric.guidelines``."""
        assert build_state("S", query="Q", guidelines=absent) == {"input": "Q", "submission": "S"}

    def test_guidelines_are_sent_verbatim(self):
        text = "  Leading and trailing space.\n\n{braces} and `ticks`\n"
        assert build_state("S", guidelines=text)["guidelines"] == text

    def test_non_string_guidelines_raise_type_error(self):
        with pytest.raises(TypeError, match="guidelines"):
            build_state("S", guidelines=3)  # a str or None only


class TestGuidelinesFraming:
    """Framed binary questions name the guidelines and their precedence; the bare Noul and
    multi-choice questions see them only through the state."""

    def test_noul_framed_golden(self):
        question = build_question(binary(), config(), WITH_GUIDELINES)
        assert wire(question) == {
            "type": "noul",
            "instructions": f"{TASK} {GUIDELINES_CLAUSE} Criterion: {REQ}",
            "criteria": {"true": MET_DEF, "false": UNMET_DEF},
        }

    def test_choice_golden(self):
        question = build_question(binary(), config(binary_framing="choice"), WITH_GUIDELINES)
        assert wire(question) == {
            "type": "choice",
            "instructions": f"{TASK} {GUIDELINES_CLAUSE} Criterion: {REQ}",
            "criteria": {"MET": MET_DEF, "UNMET": UNMET_DEF, "CANNOT_ASSESS": CA_DEF},
        }

    @pytest.mark.parametrize("framing", ["noul_framed", "choice"])
    def test_negative_weight_keeps_its_definitions_and_gains_the_clause(self, framing):
        question = build_question(
            binary(NEG_REQ, weight=-4.0), config(binary_framing=framing), WITH_GUIDELINES
        )
        assert question.instructions == f"{TASK} {GUIDELINES_CLAUSE} Criterion: {NEG_REQ}"
        outcomes = wire(question)["criteria"]
        assert NEG_MET_DEF in outcomes.values() and NEG_UNMET_DEF in outcomes.values()

    @pytest.mark.parametrize("framing", ["noul_framed", "choice"])
    @pytest.mark.parametrize(
        ("state", "instructions"),
        [
            (WITH_GUIDELINES, f"{TASK} {GUIDELINES_CLAUSE} Criterion: {REQ}"),
            (
                WITH_GUIDELINES_AND_REFERENCE,
                f"{TASK} {GUIDELINES_CLAUSE} {REFERENCE_SUBMISSION} Criterion: {REQ}",
            ),
            (STRUCTURED_WITH_GUIDELINES, f"{TASK_OUTPUT} {GUIDELINES_CLAUSE} Criterion: {REQ}"),
            (
                STRUCTURED_WITH_EVERY_FIELD,
                f"{TASK_OUTPUT} {GUIDELINES_CLAUSE} {REFERENCE_OUTPUT} Criterion: {REQ}",
            ),
        ],
        ids=["plain", "reference", "structured", "structured-reference"],
    )
    def test_clauses_follow_the_state_field_order(self, framing, state, instructions):
        """Task sentence, then one sentence per optional field in state order (guidelines
        before the reference), then the requirement, verbatim and last."""
        question = build_question(binary(), config(binary_framing=framing), state)
        assert question.instructions == instructions

    @pytest.mark.parametrize("state", GUIDELINES_STATES)
    @pytest.mark.parametrize("weight", [5.0, -4.0])
    def test_bare_noul_gets_no_clause(self, state, weight):
        question = build_question(binary(weight=weight), config(binary_framing="noul"), state)
        assert wire(question) == {"type": "noul", "instructions": REQ}

    @pytest.mark.parametrize("state", GUIDELINES_STATES)
    @pytest.mark.parametrize("ordinal_framing", ["choice", "score"])
    def test_multi_choice_gets_no_clause(self, state, ordinal_framing):
        cfg = config(ordinal_framing=ordinal_framing)
        assert build_question(ordinal(), cfg, state) == build_question(ordinal(), cfg, PLAIN)
        assert build_question(nominal(), cfg, state) == build_question(nominal(), cfg, PLAIN)

    @pytest.mark.parametrize(
        "requirement", ["Apply the `guidelines`; the criterion text governs.", "Criterion: x", ""]
    )
    def test_requirement_is_verbatim_and_last(self, requirement):
        question = build_question(binary(requirement), config(), WITH_GUIDELINES_AND_REFERENCE)
        assert question.instructions.endswith(" Criterion: " + requirement)
        assert question.instructions.count(GUIDELINES_CLAUSE) == 1 + (
            requirement == GUIDELINES_CLAUSE
        )

    def test_without_guidelines_the_questions_are_unchanged(self):
        """A state without guidelines poses exactly the questions posed before guidelines
        existed (the goldens above), for every framing."""
        rubric = [binary(), binary(NEG_REQ, weight=-4.0), ordinal(na=True), nominal()]
        for framing in ("noul", "noul_framed", "choice"):
            for ordinal_framing in ("choice", "score"):
                cfg = config(binary_framing=framing, ordinal_framing=ordinal_framing)
                for plain, blank in [
                    (build_state("S"), build_state("S", guidelines="")),
                    (
                        build_state("S", query="Q", reference_submission="R"),
                        build_state("S", query="Q", reference_submission="R", guidelines=None),
                    ),
                ]:
                    assert plain == blank
                    built, _ = build_questions(rubric, cfg, plain)
                    again, _ = build_questions(rubric, cfg, blank)
                    assert wire_items(built) == wire_items(again)


# =============================================================================
# Invariants
# =============================================================================

STATES = [
    build_state("S"),
    build_state("S", query="Q"),
    build_state("S", query="Q", reference_submission="R"),
    build_state("<thinking>T</thinking><output>O</output>"),
    build_state("<output>O</output>", query="Q", reference_submission="R"),
    build_state("S", guidelines="G"),
    build_state("S", query="Q", reference_submission="R", guidelines="G"),
    build_state("<thinking>T</thinking><output>O</output>", guidelines="G"),
    build_state("<output>O</output>", query="Q", reference_submission="R", guidelines="G"),
]


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("framing", ["noul", "noul_framed", "choice"])
def test_every_field_the_instructions_name_is_in_the_state(state, framing):
    """Backticked names in framed instructions are state keys; the text never refers the
    model to a field it was not sent."""
    question = build_question(binary(), config(binary_framing=framing), state)
    named = set(re.findall(r"`([^`]+)`", question.instructions))
    assert named <= set(state)


@pytest.mark.parametrize("state", STATES)
def test_questions_do_not_depend_on_state_values(state):
    """Only which fields are present shapes a question, never their text."""
    other = {key: value + " (changed)" for key, value in state.items()}
    for framing in ("noul", "noul_framed", "choice"):
        cfg = config(binary_framing=framing)
        assert wire(build_question(binary(), cfg, state)) == wire(
            build_question(binary(), cfg, other)
        )
