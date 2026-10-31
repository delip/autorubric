"""Tests for prompt construction (multi-choice NA / abstain rendering)."""

import pytest

from autorubric.prompts import (
    GUIDELINES_BLOCK,
    MULTI_CHOICE_FEW_SHOT_ADDITION,
    MULTI_CHOICE_SYSTEM_PROMPT,
    _binary_criterion_block,
    _few_shot_prefix,
    _format_multi_choice_examples,
    _multi_choice_few_shot_prefix,
    _pre_1_7_few_shot_user_prompt,
    _pre_1_7_system_prompt,
    _pre_1_7_user_prompt,
    _rubric_examples_prefix,
    _shared_user_content,
    build_few_shot_user_prompt,
    build_multi_choice_few_shot_user_prompt,
    build_multi_choice_user_prompt,
    build_rubric_user_prompt,
    build_user_prompt,
)
from autorubric.types import Criterion, CriterionOption, CriterionVerdict, FewShotExample

_NA_MARKER = "(cannot assess / not applicable)"


def _criterion_with_unmarked_na() -> Criterion:
    """Multi-choice criterion whose NA option's label does NOT itself signal NA."""
    return Criterion(
        name="q",
        requirement="How specific?",
        weight=6.0,
        scale_type="ordinal",
        options=[
            CriterionOption(label="Vague", value=0.0),
            CriterionOption(label="Specific", value=1.0),
            CriterionOption(label="Insufficient data", value=0.0, na=True),
        ],
    )


def test_na_option_is_marked_in_rendered_options():
    """An NA option is visibly marked so the judge can recognize the abstain choice."""
    prompt = build_multi_choice_user_prompt(_criterion_with_unmarked_na(), "submission")
    # The NA option line carries the marker.
    assert f"3. Insufficient data {_NA_MARKER}" in prompt
    # Scored options are NOT marked.
    assert "1. Vague" in prompt
    assert _NA_MARKER not in prompt.split("Vague")[0]


def test_na_marker_not_doubled_when_label_already_signals_na():
    """A label that already signals NA (e.g. 'N/A') is not given a redundant marker."""
    criterion = Criterion(
        name="q",
        requirement="How specific?",
        weight=6.0,
        scale_type="ordinal",
        options=[
            CriterionOption(label="Vague", value=0.0),
            CriterionOption(label="Specific", value=1.0),
            CriterionOption(label="N/A", value=0.0, na=True),
        ],
    )
    prompt = build_multi_choice_user_prompt(criterion, "submission")
    assert "3. N/A" in prompt
    # No appended "(cannot assess / not applicable)" suffix on the N/A line.
    assert f"N/A {_NA_MARKER}" not in prompt


def test_scored_only_criterion_has_no_na_marker():
    """A criterion with no NA option renders no abstain marker."""
    criterion = Criterion(
        name="q",
        requirement="How good?",
        weight=5.0,
        scale_type="ordinal",
        options=[
            CriterionOption(label="Bad", value=0.0),
            CriterionOption(label="Good", value=1.0),
        ],
    )
    prompt = build_multi_choice_user_prompt(criterion, "submission")
    assert _NA_MARKER not in prompt


def test_system_prompt_na_guidance_is_unconditional():
    """The multi-choice system prompt presents the abstain channel unconditionally."""
    # New, unconditional phrasing referencing the marked abstain option.
    assert "cannot assess / not applicable" in MULTI_CHOICE_SYSTEM_PROMPT
    assert "abstain" in MULTI_CHOICE_SYSTEM_PROMPT.lower()
    # The old conditional opener is gone.
    assert "Some options may be marked" not in MULTI_CHOICE_SYSTEM_PROMPT


def _empty_submission_section() -> str:
    """The 'Empty or refusal submissions' guidance block of the multi-choice system prompt."""
    start = MULTI_CHOICE_SYSTEM_PROMPT.index("Empty or refusal submissions:")
    end = MULTI_CHOICE_SYSTEM_PROMPT.index("Contradictory submissions:", start)
    return MULTI_CHOICE_SYSTEM_PROMPT[start:end]


def test_empty_submission_instruction_drops_ordinal_only_contradiction():
    """The empty/refusal instruction no longer hard-codes ordinal-only / anti-NA guidance.

    The old text said "Select the lowest-quality option on the scale — not NA", which is
    meaningless for nominal scales and contradicts the guaranteed NA channel.
    """
    # The anti-NA contradiction is gone (the NA channel is guaranteed).
    assert "not NA" not in MULTI_CHOICE_SYSTEM_PROMPT
    # The ordinal-only "lowest-quality option on the scale" phrasing is gone.
    assert "lowest-quality option on the scale" not in MULTI_CHOICE_SYSTEM_PROMPT


def test_empty_submission_instruction_is_scale_aware():
    """Empty/refusal guidance covers BOTH an ordinal (failure option) and a
    nominal (NA) branch, driven by whether any option describes an empty submission."""
    section = _empty_submission_section()
    # Still has a dedicated empty/refusal block.
    assert "Empty or refusal submissions:" in section
    # Ordinal branch: a failure / lowest-quality option, scored (not abstained).
    assert "absence" in section
    assert "failure" in section
    assert "lowest quality" in section.lower()
    # Nominal branch: fall back to the NA / "cannot assess" abstain option.
    assert "NA" in section
    assert "cannot assess" in section.lower()
    # No anti-NA contradiction inside the block.
    assert "not NA" not in section


def test_nominal_empty_submission_example_maps_to_na():
    """A worked nominal empty-submission example demonstrates empty -> NA,
    alongside the preserved ordinal empty -> lowest-option example."""
    # Two worked empty-submission examples now exist (ordinal lowest-option + nominal NA).
    assert MULTI_CHOICE_SYSTEM_PROMPT.count('Submission: ""') >= 2
    # The nominal example explains that no scored category applies (-> NA).
    assert "no applicable category" in MULTI_CHOICE_SYSTEM_PROMPT


def test_ordinal_empty_submission_example_preserved():
    """The existing ordinal empty -> lowest-quality-option example is retained
    (empty answers on quality scales are still scored, not excluded)."""
    assert "represents the lowest level of clarity" in MULTI_CHOICE_SYSTEM_PROMPT


def _contradictory_section() -> str:
    """The 'Contradictory submissions' guidance block of the multi-choice system prompt."""
    start = MULTI_CHOICE_SYSTEM_PROMPT.index("Contradictory submissions:")
    end = MULTI_CHOICE_SYSTEM_PROMPT.index("Borderline between two options:", start)
    return MULTI_CHOICE_SYSTEM_PROMPT[start:end]


def test_contradictory_submission_ordinal_branch_preserved():
    """The ordered-scale conservative default ('weaker' reading) is kept."""
    section = _contradictory_section()
    assert "weaker" in section
    assert "conservative default" in section


def test_contradictory_submission_instruction_is_scale_aware():
    """Genuine-ambiguity guidance covers an unordered/nominal branch that
    falls back to NA, since 'weaker interpretation' is undefined for unordered categories."""
    section = _contradictory_section()
    # Nominal branch: abstain via the NA / "cannot assess" option.
    assert "NA" in section
    assert "cannot assess" in section.lower()
    # A cue that the nominal branch is about unordered / categorical options.
    assert "unordered" in section.lower() or "categor" in section.lower()


def test_nominal_contradictory_submission_example_maps_to_na():
    """A worked nominal contradictory example demonstrates ambiguous -> NA."""
    assert "no single category applies" in MULTI_CHOICE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Shared verdict definitions (one source for LLM and decision-model framings)
# ---------------------------------------------------------------------------


def test_verdict_definitions_are_single_line_strings():
    """The shared definitions are plain one-line texts, usable verbatim as option
    descriptions outside the LLM system prompt."""
    from autorubric import prompts

    for name in (
        "MET_DEFINITION",
        "UNMET_DEFINITION",
        "CANNOT_ASSESS_DEFINITION",
        "NEGATIVE_MET_DEFINITION",
        "NEGATIVE_UNMET_DEFINITION",
    ):
        value = getattr(prompts, name)
        assert isinstance(value, str) and value, name
        assert "\n" not in value and value == value.strip(), name


def test_binary_system_prompt_is_built_from_the_verdict_definitions():
    """GRADER_SYSTEM_PROMPT_DEFAULT states each definition at its verdict line, so the
    LLM prompt and any other framing that reuses the constants cannot drift apart."""
    from autorubric.prompts import (
        CANNOT_ASSESS_DEFINITION,
        GRADER_SYSTEM_PROMPT_DEFAULT,
        MET_DEFINITION,
        NEGATIVE_MET_DEFINITION,
        NEGATIVE_UNMET_DEFINITION,
        UNMET_DEFINITION,
    )

    assert (
        "Your verdict must be one of:\n"
        f'- "MET": {MET_DEFINITION}\n'
        f'- "UNMET": {UNMET_DEFINITION}\n'
        f'- "CANNOT_ASSESS": {CANNOT_ASSESS_DEFINITION}\n'
    ) in GRADER_SYSTEM_PROMPT_DEFAULT
    assert (
        "NEGATIVE CRITERIA describe active errors or mistakes.\n"
        f"- MET: {NEGATIVE_MET_DEFINITION}\n"
        f"- UNMET: {NEGATIVE_UNMET_DEFINITION}\n"
    ) in GRADER_SYSTEM_PROMPT_DEFAULT


class TestPerCriterionLayout:
    """Shared content first and the criterion last; with examples, examples first."""

    CRITERION = Criterion(name="c", requirement="States the boiling point", weight=1.0)
    OTHER = Criterion(name="d", requirement="Names the unit", weight=1.0)
    MULTI = Criterion(
        name="m",
        requirement="How clear?",
        weight=1.0,
        options=[
            CriterionOption(label="Vague", value=0.0),
            CriterionOption(label="Clear", value=1.0),
        ],
    )
    FLIPPED = Criterion(
        name="m",
        requirement="How clear?",
        weight=1.0,
        options=[
            CriterionOption(label="Clear", value=1.0),
            CriterionOption(label="Vague", value=0.0),
        ],
    )
    SHARED_ARGS = {"query": "Q?", "reference_submission": "REF", "guidelines": "Be strict."}
    EXAMPLE = FewShotExample(submission="ex", verdict=CriterionVerdict.MET, reason=None)

    def test_criterion_ends_a_prompt_without_examples(self) -> None:
        block = _binary_criterion_block(self.CRITERION)
        assert build_user_prompt(self.CRITERION, "S", **self.SHARED_ARGS).endswith(block)

    def test_a_binary_prompt_with_examples_ends_with_the_item(self) -> None:
        prompt = build_few_shot_user_prompt(self.CRITERION, "S", [self.EXAMPLE], **self.SHARED_ARGS)
        assert (
            prompt.index("<guidelines>")
            < prompt.index("<examples>")
            < prompt.index("<criterion>")
            < prompt.index("<input>")
            < prompt.index("<submission>")
        )
        assert prompt.endswith("<submission>\nS\n</submission>")

    def test_a_multi_choice_prompt_keeps_its_options_after_the_prefix(self) -> None:
        prompt = build_multi_choice_few_shot_user_prompt(
            self.MULTI, "S", [("ex", 1, None)], **self.SHARED_ARGS
        )
        prefix = _multi_choice_few_shot_prefix(
            self.MULTI, [("ex", 1, None)], guidelines="Be strict."
        )
        assert prompt.startswith(prefix + "\n\n<options>")
        assert "<options>" not in prefix
        assert prefix.endswith("</question>")

    def test_the_prefix_is_the_same_for_every_option_order(self) -> None:
        one = _multi_choice_few_shot_prefix(self.MULTI, [("ex", 1, None)])
        other = _multi_choice_few_shot_prefix(self.FLIPPED, [("ex", 0, None)])
        assert one == other
        assert "<selected_option>" not in one
        assert "<selected_label>Clear</selected_label>" in one

    def test_the_binary_prefix_does_not_depend_on_the_item(self) -> None:
        prefix = _few_shot_prefix(self.CRITERION, [self.EXAMPLE], guidelines="Be strict.")
        for submission in ("S", "T"):
            prompt = build_few_shot_user_prompt(
                self.CRITERION, submission, [self.EXAMPLE], guidelines="Be strict."
            )
            assert prompt.startswith(prefix + "\n\n")

    def test_shared_block_is_equal_across_criteria_of_an_item(self) -> None:
        shared = _shared_user_content("S", "Q?", "REF", guidelines="Be strict.")
        for criterion in (self.CRITERION, self.OTHER):
            assert build_user_prompt(criterion, "S", **self.SHARED_ARGS).startswith(shared + "\n\n")

    def test_whole_rubric_prefix_is_guidelines_and_examples(self) -> None:
        prompt = build_rubric_user_prompt(
            [("c0", self.MULTI)], "S", examples_text="<examples>\nE\n</examples>", guidelines="G"
        )
        prefix = _rubric_examples_prefix("<examples>\nE\n</examples>", guidelines="G")
        assert prompt.startswith(prefix + "\n\n<criteria>")

    @pytest.mark.parametrize("with_guidelines", [False, True])
    def test_pre_1_7_text_without_examples(self, with_guidelines: bool) -> None:
        guidelines = "Be strict." if with_guidelines else None
        block = _binary_criterion_block(self.CRITERION)
        shared = _shared_user_content("S", "Q?", "REF", guidelines=guidelines)
        prompt = build_user_prompt(self.CRITERION, "S", "Q?", "REF", guidelines=guidelines)
        head = GUIDELINES_BLOCK.format(guidelines="Be strict.") if with_guidelines else ""
        assert _pre_1_7_user_prompt(prompt, shared, block, guidelines) == (
            f"{head}{block}\n\n<input>Q?</input>\n\n"
            "<reference_submission>\nREF\n</reference_submission>\n\n"
            "<submission>\nS\n</submission>"
        )

    @pytest.mark.parametrize("with_guidelines", [False, True])
    def test_pre_1_7_text_with_examples(self, with_guidelines: bool) -> None:
        guidelines = "Be strict." if with_guidelines else None
        head = GUIDELINES_BLOCK.format(guidelines="Be strict.") if with_guidelines else ""
        assert _pre_1_7_few_shot_user_prompt(
            "POSED", "EXAMPLES", "S", "Q?", "REF", guidelines=guidelines
        ) == (
            f"{head}POSED\n\nEXAMPLES\n\n<input>Q?</input>\n\n"
            "<reference_submission>\nREF\n</reference_submission>\n\n"
            "<submission>\nS\n</submission>"
        )

    def test_pre_1_7_examples_number_their_options(self) -> None:
        text = _format_multi_choice_examples(self.MULTI, [("ex", 1, None)], False, numbered=True)
        assert "<selected_option>2</selected_option>\n<selected_label>Clear" in text

    def test_pre_1_7_system_prompt_mentions_the_option_number(self) -> None:
        old = _pre_1_7_system_prompt(MULTI_CHOICE_SYSTEM_PROMPT + MULTI_CHOICE_FEW_SHOT_ADDITION)
        assert "- <selected_option>: The option number that was selected\n" in old
        assert "- <selected_option>" not in MULTI_CHOICE_FEW_SHOT_ADDITION
