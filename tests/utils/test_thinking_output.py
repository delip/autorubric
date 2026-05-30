"""Tests for thinking/output token support."""

import pytest

from autorubric import (
    LengthPenalty,
    ThinkingOutputDict,
    compute_length_penalty,
    normalize_to_grade_input,
    parse_thinking_output,
)


class TestParseThinkingOutput:
    """Tests for parse_thinking_output function."""

    def test_parse_with_markers(self):
        """Test parsing text with thinking and output markers."""
        text = "<thinking>reasoning</thinking><output>answer</output>"
        result = parse_thinking_output(text)
        assert result["thinking"] == "reasoning"
        assert result["output"] == "answer"

    def test_parse_case_insensitive(self):
        """Test that markers are case-insensitive."""
        text = "<THINKING>think</THINKING><OUTPUT>ans</OUTPUT>"
        result = parse_thinking_output(text)
        assert result["thinking"] == "think"
        assert result["output"] == "ans"

    def test_parse_no_markers_treated_as_output(self):
        """Test plain text without markers is treated as output."""
        result = parse_thinking_output("plain text")
        assert result["thinking"] == ""
        assert result["output"] == "plain text"

    def test_parse_only_thinking_marker(self):
        """Test text with only thinking marker - rest becomes output."""
        text = "<thinking>reason</thinking>remaining text"
        result = parse_thinking_output(text)
        assert result["thinking"] == "reason"
        assert result["output"] == "remaining text"


class TestNormalizeToGradeInput:
    """Tests for normalize_to_grade_input function."""

    def test_normalize_string(self):
        """Test normalizing plain string."""
        result = normalize_to_grade_input("text")
        assert result["output"] == "text"

    def test_normalize_dict(self):
        """Test normalizing dict input."""
        result = normalize_to_grade_input({"thinking": "a", "output": "b"})
        assert result["thinking"] == "a"
        assert result["output"] == "b"

    def test_normalize_invalid_type_raises_error(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="to_grade must be a string or dict"):
            normalize_to_grade_input(123)  # type: ignore[arg-type]

    def test_normalize_dict_with_non_string_raises_error(self):
        """Test that non-string values raise ValueError."""
        with pytest.raises(ValueError, match="'output' must be a string"):
            normalize_to_grade_input({"output": 123})  # type: ignore[dict-item]


class TestComputeLengthPenaltyWithPenaltyType:
    """Tests for compute_length_penalty with penalty_type."""

    @pytest.mark.parametrize(
        ("penalty_type", "thinking", "output", "expected_predicate"),
        [
            # OUTPUT_ONLY ignores a long thinking section -> short output -> no penalty.
            ("OUTPUT_ONLY", " ".join(["x"] * 100), "short", lambda p: p == 0.0),
            # THINKING_ONLY ignores a long output section -> short thinking -> no penalty.
            ("THINKING_ONLY", "brief", " ".join(["x"] * 100), lambda p: p == 0.0),
            # ALL counts both sections -> combined length exceeds budget -> penalty.
            ("ALL", "one two", "three four five six", lambda p: p > 0),
        ],
    )
    def test_penalty_type_selects_counted_sections(
        self, penalty_type, thinking, output, expected_predicate
    ):
        """penalty_type dispatch counts only the configured section(s)."""
        config = LengthPenalty(
            free_budget=5, max_cap=10, penalty_at_cap=1.0, penalty_type=penalty_type
        )
        text = ThinkingOutputDict(thinking=thinking, output=output)
        penalty = compute_length_penalty(text, config)
        assert expected_predicate(penalty)

    def test_backwards_compatible_string_input(self):
        """Test plain string input still works."""
        config = LengthPenalty(free_budget=2, max_cap=5, penalty_at_cap=1.0)
        penalty = compute_length_penalty("one two three four", config)
        assert penalty > 0

    def test_penalty_type_default_is_all(self):
        """Test default penalty_type is 'ALL'."""
        config = LengthPenalty(free_budget=5, max_cap=10)
        assert config.penalty_type == "ALL"
