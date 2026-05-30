"""Tests for prompt construction (multi-choice NA / abstain rendering)."""

from autorubric.prompts import (
    MULTI_CHOICE_SYSTEM_PROMPT,
    build_multi_choice_user_prompt,
)
from autorubric.types import Criterion, CriterionOption

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
