"""Tests for prompt construction (multi-choice NA / abstain rendering, T2-A)."""

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
