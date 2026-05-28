"""Type definitions for rubrics and evaluation components."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

CountFn = Callable[[str], int]


# ============================================================================
# Token Usage and Cost Tracking
# ============================================================================


@dataclass
class TokenUsage:
    """Token usage statistics from LLM API calls.

    Attributes:
        prompt_tokens: Number of tokens in the prompt/input.
        completion_tokens: Number of tokens in the completion/output.
        total_tokens: Total tokens (prompt + completion).
        cache_creation_input_tokens: Tokens used to create cache entries (Anthropic).
        cache_read_input_tokens: Tokens read from cache (Anthropic).

    Example:
        >>> usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        >>> print(f"Total tokens: {usage.total_tokens}")
        Total tokens: 150
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Anthropic prompt caching fields
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together."""
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    def __radd__(self, other: "TokenUsage | int") -> "TokenUsage":
        """Support sum() by handling 0 + TokenUsage."""
        if other == 0:
            return self
        if isinstance(other, TokenUsage):
            return self.__add__(other)
        return NotImplemented


class ThinkingOutputDict(TypedDict, total=False):
    """Dict format for submissions with separate thinking and output sections.

    Both fields are optional to allow partial submissions or gradual construction.
    When used with length penalty, missing fields are treated as empty strings.
    """

    thinking: str
    output: str


ToGradeInput = str | ThinkingOutputDict
"""Union type for to_grade parameter.

Accepts either a plain string or a dict with thinking/output keys.
"""

PenaltyType = Literal["ALL", "OUTPUT_ONLY", "THINKING_ONLY"]
"""Type for penalty_type field: specifies which sections to count for length penalty."""


class LengthPenalty(BaseModel):
    """Configuration for applying length-based penalties during grading.

    The penalty is computed as:
    - 0 if count <= free_budget
    - penalty_at_cap if count >= max_cap
    - penalty_at_cap * ((count - free_budget) / (max_cap - free_budget)) ** exponent otherwise

    By default, the penalty is subtracted from the final score (which is normalized to 0-1).
    For training use cases with raw scores, use absolute penalty values (e.g., 50.0).

    Attributes:
        free_budget: Number of tokens/words allowed before any penalty applies.
        max_cap: Number of tokens/words at which the maximum penalty is applied.
        penalty_at_cap: Maximum penalty value (always subtracted from score). For normalized
            scores, use fractional values like 0.5 (lose up to 50% of score). For training
            with raw scores, use absolute values like 50.0 (subtract up to 50 points).
        exponent: Controls the penalty curve steepness. Higher = more lenient near free_budget.
        count_fn: Function to count tokens/words in text. If None, uses whitespace word count.
            For accurate token counting, pass a tokenizer-based function like:
            `lambda text: len(tokenizer.encode(text))`
        penalty_type: Which text to count for penalty calculation:
            - "ALL": Count both thinking and output tokens (default)
            - "OUTPUT_ONLY": Count only output tokens (useful for RL training)
            - "THINKING_ONLY": Count only thinking tokens

    Example:
        >>> # Default: word-based counting with sensible defaults for normalized scores
        >>> penalty = LengthPenalty()
        >>>
        >>> # For training with raw (unnormalized) scores - absolute penalty values
        >>> penalty = LengthPenalty(
        ...     free_budget=8000,
        ...     max_cap=10000,
        ...     penalty_at_cap=50.0,  # Subtract up to 50 points from raw score
        ...     exponent=1.6,
        ... )
        >>>
        >>> # Custom tokenizer-based counting (e.g., with HuggingFace)
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> penalty = LengthPenalty(
        ...     free_budget=8000,
        ...     max_cap=10000,
        ...     count_fn=lambda text: len(tokenizer.encode(text))
        ... )
        >>>
        >>> # Only penalize output tokens (allow long thinking)
        >>> penalty = LengthPenalty(
        ...     free_budget=8000,
        ...     penalty_type="OUTPUT_ONLY",
        ... )
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    free_budget: int = 6000
    max_cap: int = 8000
    penalty_at_cap: float = 0.5
    exponent: float = 1.6
    count_fn: CountFn | None = None
    penalty_type: PenaltyType = "ALL"


ScaleType = Literal["ordinal", "nominal"]
"""Scale type for multi-choice criteria.

- ordinal: Options have inherent order (e.g., 1-4 satisfaction scale)
- nominal: Options are unordered categories (e.g., "too few", "too many", "just right")
"""

OrdinalAggregation = Literal["mean", "median", "weighted_mean", "mode"]
"""Aggregation strategy for ordinal multi-choice criteria.

- mean: Average of score values across judges
- median: Median of score values
- weighted_mean: Weighted average by judge weight
- mode: Most common selection
"""

NominalAggregation = Literal["mode", "weighted_mode", "unanimous"]
"""Aggregation strategy for nominal multi-choice criteria.

- mode: Most common selection (majority vote)
- weighted_mode: Weight votes by judge weight
- unanimous: All judges must agree
"""


class CriterionOption(BaseModel):
    """A single option in a multi-choice criterion.

    Attributes:
        label: Display text shown to the LLM judge.
        value: Score value (0.0-1.0) when this option is selected. REQUIRED.
        na: If True, this option indicates "not applicable" and is treated like
            CANNOT_ASSESS (excluded from scoring).

    Example:
        >>> # Ordinal scale with explicit values
        >>> options = [
        ...     CriterionOption(label="Very dissatisfied", value=0.0),
        ...     CriterionOption(label="Dissatisfied", value=0.33),
        ...     CriterionOption(label="Satisfied", value=0.67),
        ...     CriterionOption(label="Very satisfied", value=1.0),
        ... ]
        >>>
        >>> # Option with NA
        >>> na_option = CriterionOption(label="N/A - No claims made", value=0.0, na=True)
    """

    model_config = ConfigDict(frozen=True)

    label: str
    value: float
    na: bool = False

    @model_validator(mode="after")
    def validate_value_range(self) -> "CriterionOption":
        """Validate that non-NA options have values in [0, 1]."""
        if not self.na and not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Option value must be in [0, 1], got {self.value}")
        return self


class Criterion(BaseModel):
    """A single evaluation criterion with a weight and requirement description.

    Supports both binary (MET/UNMET) and multi-choice criteria. If `options` is None,
    the criterion is binary. If `options` is provided, the criterion is multi-choice.

    Attributes:
        weight: Scoring weight. Positive for desired traits, negative for errors/penalties.
            Defaults to 10.0 for uniform weighting when not specified.
        requirement: Description of what the criterion evaluates.
        name: Optional short identifier for the criterion (e.g., "clarity", "accuracy").
            Useful for referencing criteria in reports and debugging.
        options: List of options for multi-choice criteria. If None, criterion is binary.
        scale_type: For multi-choice, indicates if options are ordinal (ordered) or
            nominal (unordered categories). Affects aggregation strategy selection.
        aggregation: Per-criterion aggregation strategy override. If None, uses grader default.

    Example:
        >>> # Binary criterion (existing behavior)
        >>> binary = Criterion(
        ...     name="accuracy",
        ...     weight=10.0,
        ...     requirement="The response is factually accurate"
        ... )
        >>>
        >>> # Multi-choice ordinal criterion
        >>> ordinal = Criterion(
        ...     name="satisfaction",
        ...     weight=10.0,
        ...     requirement="How satisfied would you be?",
        ...     options=[
        ...         CriterionOption(label="1", value=0.0),
        ...         CriterionOption(label="2", value=0.33),
        ...         CriterionOption(label="3", value=0.67),
        ...         CriterionOption(label="4", value=1.0),
        ...     ],
        ...     scale_type="ordinal",
        ... )
    """

    model_config = ConfigDict(frozen=True)

    weight: float = 10.0
    requirement: str
    name: str | None = None
    # Multi-choice support
    options: list[CriterionOption] | None = None
    scale_type: ScaleType = "ordinal"
    aggregation: str | None = None  # Per-criterion override

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary (MET/UNMET) criterion."""
        return self.options is None

    @property
    def is_multi_choice(self) -> bool:
        """Check if this is a multi-choice criterion."""
        return self.options is not None

    def get_option_value(self, index: int) -> float:
        """Get the score value for an option by index.

        Args:
            index: Zero-based index of the option.

        Returns:
            The score value for the option.

        Raises:
            ValueError: If this is a binary criterion or index is out of range.
        """
        if self.options is None:
            raise ValueError("Binary criterion has no options")
        if index < 0 or index >= len(self.options):
            raise ValueError(f"Option index {index} out of range [0, {len(self.options)})")
        return self.options[index].value

    def find_option_by_label(self, label: str) -> int:
        """Find option index by label (case-insensitive, whitespace-normalized).

        Used for resolving ground truth labels to indices for metrics computation.

        Args:
            label: The label to search for.

        Returns:
            Zero-based index of the matching option.

        Raises:
            ValueError: If this is a binary criterion or label not found.
        """
        if self.options is None:
            raise ValueError("Binary criterion has no options")
        normalized_label = label.strip().lower()
        for i, opt in enumerate(self.options):
            if opt.label.strip().lower() == normalized_label:
                return i
        available = [opt.label for opt in self.options]
        raise ValueError(f"Label '{label}' not found. Available: {available}")

    def worst_scored_option(self) -> tuple[int, CriterionOption]:
        """Return (index, option) of the score-minimizing scored (non-NA) option.

        Weight-sign aware: for non-negative weight, returns the option with the
        lowest ``value``; for negative weight, returns the option with the
        highest ``value`` (the worst case flips because a high ``value`` on a
        negative-weight criterion subtracts more from the score). NA options
        are excluded — this returns the score-minimizing *scored* option, the
        analog of binary UNMET (for positive weight) or MET (for negative
        weight).

        Ties resolve to the first such option in declaration order
        (deterministic, matches ``min``/``max`` semantics).

        Shared by the grader's ``unknown``-error worst-case path
        (``criterion_grader.py``) and the metrics' ``na_mode="as_unmet"`` remap
        (``metrics/_helpers.py``) so the two layers cannot drift.

        Returns:
            Tuple of ``(index, CriterionOption)`` for the score-minimizing
            non-NA option.

        Raises:
            ValueError: If this is a binary criterion or has no non-NA option.
                The ``Criterion`` validator guarantees ≥2 non-NA options for
                multi-choice criteria, so the no-non-NA case is defensive.
        """
        if self.options is None:
            raise ValueError("Binary criterion has no scored options")
        scored = [(i, opt) for i, opt in enumerate(self.options) if not opt.na]
        if not scored:
            raise ValueError("Criterion has no non-NA option")
        if self.weight < 0:
            return max(scored, key=lambda io: io[1].value)
        return min(scored, key=lambda io: io[1].value)

    @model_validator(mode="after")
    def validate_options(self) -> "Criterion":
        """Validate multi-choice options if present."""
        if self.options is not None:
            if len(self.options) < 2:
                raise ValueError("Multi-choice criterion must have at least 2 options")
            # Ensure at least 2 non-NA options
            non_na = [o for o in self.options if not o.na]
            if len(non_na) < 2:
                raise ValueError("Must have at least 2 non-NA options")
        return self


class CriterionVerdict(str, Enum):
    """Status of a criterion evaluation.

    - MET: The criterion is satisfied by the submission
    - UNMET: The criterion is not satisfied by the submission
    - CANNOT_ASSESS: Insufficient evidence to make a determination
    """

    MET = "MET"
    UNMET = "UNMET"
    CANNOT_ASSESS = "CANNOT_ASSESS"


class CannotAssessStrategy(str, Enum):
    """Strategy for handling CANNOT_ASSESS verdicts in score calculation.

    - SKIP: Exclude the criterion from scoring entirely (adjust denominator)
    - ZERO: Treat as 0 contribution (same as UNMET for positive criteria)
    - PARTIAL: Treat as partial credit (configurable fraction)
    - FAIL: Treat as worst case (UNMET for positive, MET for negative)
    """

    SKIP = "skip"
    ZERO = "zero"
    PARTIAL = "partial"
    FAIL = "fail"


class CannotAssessConfig(BaseModel):
    """Configuration for handling CANNOT_ASSESS verdicts.

    Attributes:
        strategy: How to handle CANNOT_ASSESS verdicts in score calculation.
            Default is SKIP, which excludes unassessable criteria from scoring.
        partial_credit: Fraction of weight to award when strategy is PARTIAL.
            Must be between 0.0 and 1.0. Default is 0.5.

    Example:
        >>> # Default: skip unassessable criteria
        >>> config = CannotAssessConfig()
        >>>
        >>> # Be conservative: treat cannot-assess as failure
        >>> config = CannotAssessConfig(strategy=CannotAssessStrategy.FAIL)
        >>>
        >>> # Give partial credit
        >>> config = CannotAssessConfig(
        ...     strategy=CannotAssessStrategy.PARTIAL,
        ...     partial_credit=0.3
        ... )
    """

    model_config = ConfigDict(frozen=True)

    strategy: CannotAssessStrategy = CannotAssessStrategy.SKIP
    partial_credit: float = 0.5


# ============================================================================
# Multi-Choice Verdict Types
# ============================================================================


class MultiChoiceVerdict(BaseModel):
    """Verdict for a multi-choice criterion evaluation.

    Stores both index (stable, for metrics computation) and label (readable, for reports).
    This design enables future metrics like kappa, accuracy, and confusion matrices.

    Attributes:
        selected_index: Zero-based index of the selected option. STABLE for metrics.
        selected_label: Label text of the selected option. READABLE for reports.
        value: Score contribution of the selected option (0.0-1.0).
        na: True if the selected option is marked as NA (not applicable).

    Example:
        >>> verdict = MultiChoiceVerdict(
        ...     selected_index=2,
        ...     selected_label="Satisfied",
        ...     value=0.67,
        ...     na=False
        ... )
    """

    model_config = ConfigDict(frozen=True)

    selected_index: int
    selected_label: str
    value: float
    na: bool = False


class AggregatedMultiChoiceVerdict(MultiChoiceVerdict):
    """Extended verdict for ensemble aggregation results.

    Stores both discrete (snapped to nearest option) and continuous (actual mean/median)
    results to support different metrics:
    - Discrete (selected_index, value): for exact accuracy, kappa
    - Continuous (aggregated_value): for RMSE, MAE on scores

    Attributes:
        aggregated_value: Continuous aggregated value before snapping to nearest option.
            For ordinal scales with mean/median aggregation, this may differ from `value`.
            For nominal scales with mode aggregation, this equals `value`.

    Example:
        >>> # Mean of [0.0, 0.33, 0.67] = 0.33, snapped to option 1
        >>> verdict = AggregatedMultiChoiceVerdict(
        ...     selected_index=1,
        ...     selected_label="Dissatisfied",
        ...     value=0.33,        # Value of snapped option
        ...     aggregated_value=0.33,  # Actual mean
        ...     na=False
        ... )
    """

    aggregated_value: float


@dataclass
class MultiChoiceJudgeVote:
    """Individual judge's vote for a multi-choice criterion (ensemble mode).

    Preserves full vote details for per-judge metrics and inter-judge agreement analysis
    (Krippendorff's alpha, Fleiss' kappa, per-judge accuracy).

    Attributes:
        judge_id: Identifier for the judge (e.g., "gpt-4", "claude-sonnet").
        selected_index: Zero-based index of selected option. STABLE for metrics.
        selected_label: Label of selected option. READABLE for reports.
        value: Score value of selected option.
        reason: Judge's explanation for the selection.
        weight: Judge's voting weight (default 1.0).
        na: True if selected option is NA.
        shuffle_order: Permutation used when presenting options to the judge.
        error: Set (with a category prefix) when this vote's verdict was synthesized
            because the judge call failed. None for genuine votes. Mirrors
            ``JudgeVote.error`` for multi-choice criteria.
    """

    judge_id: str
    selected_index: int
    selected_label: str
    value: float
    reason: str
    weight: float = 1.0
    na: bool = False
    shuffle_order: list[int] | None = None
    error: str | None = None


class CriterionReport(Criterion):
    """A criterion with its evaluation result.

    Supports both binary (MET/UNMET/CANNOT_ASSESS) and multi-choice verdicts.
    For binary criteria, use `verdict`. For multi-choice, use `multi_choice_verdict`.

    Attributes:
        verdict: Binary verdict (MET/UNMET/CANNOT_ASSESS). None for multi-choice criteria.
        multi_choice_verdict: Multi-choice verdict with selected option. None for binary.
        reason: Explanation for the verdict from the LLM judge.
        shuffle_order: Permutation used when presenting multi-choice options to the LLM.
            Maps shuffled position → original index. None for binary criteria or when
            shuffle_options is disabled.
        error: Set when this verdict was synthesized because the judge call failed,
            rather than produced by a genuine judgment. The string is prefixed with the
            failure category (``"infrastructure: ..."``, ``"parse: ..."``, or
            ``"unknown: ..."``). None for genuine verdicts. See ``is_error``.
    """

    verdict: CriterionVerdict | None = None
    multi_choice_verdict: MultiChoiceVerdict | AggregatedMultiChoiceVerdict | None = None
    reason: str
    shuffle_order: list[int] | None = None
    error: str | None = None

    @property
    def score_value(self) -> float:
        """Get the score contribution (0-1) for this criterion.

        For binary criteria: 1.0 if MET, 0.0 otherwise.
        For multi-choice: the value of the selected option.
        """
        if self.verdict is not None:
            return 1.0 if self.verdict == CriterionVerdict.MET else 0.0
        if self.multi_choice_verdict is not None:
            return self.multi_choice_verdict.value
        return 0.0

    @property
    def is_na(self) -> bool:
        """Check if this criterion was marked NA or CANNOT_ASSESS.

        Returns True for:
        - Binary criteria with CANNOT_ASSESS verdict
        - Multi-choice criteria with NA option selected
        """
        if self.verdict == CriterionVerdict.CANNOT_ASSESS:
            return True
        if self.multi_choice_verdict is not None:
            return self.multi_choice_verdict.na
        return False

    @property
    def is_error(self) -> bool:
        """Whether this verdict was synthesized due to a judge-call failure.

        Use this instead of inspecting ``reason`` to distinguish error-induced
        verdicts from genuine judgments.
        """
        return self.error is not None


class EvaluationReport(BaseModel):
    """Final evaluation result with score and per-criterion reports.

    For training use cases, set normalize=False in the grader to get raw weighted sums
    instead of normalized 0-1 scores.

    Attributes:
        score: The final score (0-1 if normalized, raw weighted sum otherwise).
        raw_score: The unnormalized weighted sum.
        llm_raw_score: The original score returned by the LLM (same as raw_score).
        report: Per-criterion breakdown with verdicts and explanations.
        cannot_assess_count: Number of criteria with CANNOT_ASSESS verdict.
        error: Optional error message if grading failed (e.g., JSON parse error).
            When set, score defaults to 0.0. Training pipelines should filter these out.
        token_usage: Aggregated token usage across all LLM calls made during grading.
            For CriterionGrader, this is the sum across all criterion evaluations.
        completion_cost: Total cost in USD for all LLM calls made during grading.
            Calculated using LiteLLM's completion_cost() function.

    Example:
        >>> result = await rubric.grade(to_grade=response, grader=grader)
        >>> print(f"Score: {result.score:.2f}")
        >>> if result.cannot_assess_count:
        ...     print(f"Could not assess {result.cannot_assess_count} criteria")
        >>> if result.token_usage:
        ...     print(f"Tokens: {result.token_usage.total_tokens}")
        >>> if result.completion_cost:
        ...     print(f"Cost: ${result.completion_cost:.6f}")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: float
    raw_score: float | None = None
    llm_raw_score: float | None = None
    report: list[CriterionReport] | None = None
    cannot_assess_count: int = 0
    error: str | None = None
    token_usage: TokenUsage | None = None
    completion_cost: float | None = None


# ============================================================================
# Structured Output Types for LLM Responses
# ============================================================================


class CriterionJudgment(BaseModel):
    """Structured LLM output for single criterion evaluation.

    Used with LiteLLM's response_format parameter to ensure
    type-safe, validated responses from the judge LLM.

    Note: This is separate from CriterionReport because:
    - CriterionReport includes 'weight' and 'requirement' fields that come
      from the rubric, not from the LLM
    - The LLM only outputs the judgment (status + explanation)
    """

    model_config = ConfigDict(frozen=True)

    criterion_status: CriterionVerdict = Field(
        description=(
            "Whether the criterion is satisfied (MET), not satisfied (UNMET), "
            "or cannot be determined (CANNOT_ASSESS)"
        )
    )
    explanation: str = Field(
        description=(
            "Brief explanation of why the criterion is or isn't met, or why it cannot be assessed"
        )
    )
    reasoning: str | None = Field(
        default=None,
        description=(
            "Extended thinking/reasoning trace from the LLM (populated when thinking is enabled)"
        ),
    )


class MultiChoiceJudgment(BaseModel):
    """Structured LLM output for multi-choice criterion evaluation.

    Used with LiteLLM's response_format parameter to ensure type-safe,
    validated responses from the judge LLM for multi-choice criteria.

    Note: The LLM uses 1-indexed option numbers for human readability.
    The grader converts to 0-indexed internally.

    Attributes:
        selected_option: 1-indexed number of the selected option (1, 2, 3, etc.)
        explanation: Brief explanation of why this option was selected.
        reasoning: Extended thinking/reasoning trace (populated when thinking enabled).
    """

    model_config = ConfigDict(frozen=True)

    selected_option: int = Field(description="The number of your chosen option (1, 2, 3, etc.)")
    explanation: str = Field(
        description="Brief explanation of why this option best describes the submission"
    )
    reasoning: str | None = Field(
        default=None,
        description=(
            "Extended thinking/reasoning trace from the LLM (populated when thinking is enabled)"
        ),
    )


# ============================================================================
# Ensemble Grading Types
# ============================================================================

AggregationStrategy = Literal["majority", "weighted", "unanimous", "any"]
"""Strategy for aggregating votes from multiple judges.

- majority: Simple majority vote (> 50% must agree)
- weighted: Weighted vote based on judge weights
- unanimous: All judges must agree for MET
- any: Any judge voting MET results in MET
"""


@dataclass
class JudgeVote:
    """A single judge's vote on a criterion.

    Attributes:
        judge_id: Identifier for the judge (e.g., "gpt-4", "claude-sonnet").
        verdict: The judge's verdict (MET/UNMET).
        reason: The judge's explanation for the verdict.
        weight: Judge's voting weight (default 1.0).
        error: Set (with a category prefix) when this vote's verdict was synthesized
            because the judge call failed. None for genuine votes.
    """

    judge_id: str
    verdict: CriterionVerdict
    reason: str
    weight: float = 1.0
    error: str | None = None


@dataclass
class EnsembleCriterionReport:
    """A criterion report with ensemble voting details.

    Supports both binary and multi-choice criteria:
    - Binary: Use `final_verdict` and `votes` (list of JudgeVote)
    - Multi-choice: Use `final_multi_choice_verdict` and `multi_choice_votes`

    Attributes:
        criterion: The criterion being evaluated.
        final_verdict: Aggregated binary verdict from all judges. None for multi-choice.
        final_reason: Combined reasoning from judges.
        votes: Individual binary votes from each judge. Empty for multi-choice.
        agreement: Proportion of judges agreeing with final verdict (0-1).
        final_multi_choice_verdict: Aggregated multi-choice verdict. None for binary.
        multi_choice_votes: Individual multi-choice votes. Empty for binary.
        error: Set (with a category prefix) when the final verdict was driven entirely by
            judge-call failures (every contributing vote errored). None when at least one
            genuine judgment was available. See ``is_error``.
    """

    criterion: Criterion
    final_verdict: CriterionVerdict | None
    final_reason: str
    votes: list[JudgeVote] = field(default_factory=list)
    agreement: float = field(default=0.0)
    # Multi-choice support
    final_multi_choice_verdict: AggregatedMultiChoiceVerdict | None = field(default=None)
    multi_choice_votes: list[MultiChoiceJudgeVote] = field(default_factory=list)
    error: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Compute agreement if not set."""
        if self.agreement == 0.0:
            if self.votes:
                agreeing = sum(1 for v in self.votes if v.verdict == self.final_verdict)
                self.agreement = agreeing / len(self.votes)
            elif self.multi_choice_votes and self.final_multi_choice_verdict:
                # For multi-choice, count votes matching the final selected index
                final_idx = self.final_multi_choice_verdict.selected_index
                agreeing = sum(1 for v in self.multi_choice_votes if v.selected_index == final_idx)
                self.agreement = agreeing / len(self.multi_choice_votes)

    @property
    def score_value(self) -> float:
        """Get the score contribution (0-1) for this criterion."""
        if self.final_verdict is not None:
            return 1.0 if self.final_verdict == CriterionVerdict.MET else 0.0
        if self.final_multi_choice_verdict is not None:
            return self.final_multi_choice_verdict.value
        return 0.0

    @property
    def is_na(self) -> bool:
        """Check if this criterion was marked NA or CANNOT_ASSESS."""
        if self.final_verdict == CriterionVerdict.CANNOT_ASSESS:
            return True
        if self.final_multi_choice_verdict is not None:
            return self.final_multi_choice_verdict.na
        return False

    @property
    def is_error(self) -> bool:
        """Whether the final verdict was driven entirely by judge-call failures."""
        return self.error is not None


class EnsembleEvaluationReport(BaseModel):
    """Evaluation report with ensemble voting details.

    Extends EvaluationReport with per-judge breakdown and agreement metrics.

    Attributes:
        score: The final aggregated score (0-1 if normalized).
        raw_score: The unnormalized weighted sum.
        llm_raw_score: Same as raw_score (for compatibility with EvaluationReport).
        report: Per-criterion breakdown with ensemble voting details.
        judge_scores: Individual scores from each judge.
        mean_agreement: Average agreement across all criteria.
        cannot_assess_count: Number of criteria with CANNOT_ASSESS final verdict.
        token_usage: Total token usage across all judges.
        completion_cost: Total cost across all judges.
        error: Error message if grading failed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: float
    raw_score: float | None = None
    llm_raw_score: float | None = None
    report: list[EnsembleCriterionReport] | None = None
    judge_scores: dict[str, float] = Field(default_factory=dict)
    mean_agreement: float = 0.0
    cannot_assess_count: int = 0
    token_usage: TokenUsage | None = None
    completion_cost: float | None = None
    error: str | None = None


# ============================================================================
# Few-Shot Configuration Types
# ============================================================================


@dataclass
class FewShotExample:
    """A single few-shot example for criterion evaluation.

    Attributes:
        submission: The content that was evaluated. Can be plain text or JSON-serialized.
        verdict: The ground truth verdict for this criterion.
        reason: Optional explanation for why the verdict was assigned.

    Example:
        >>> example = FewShotExample(
        ...     submission="The Industrial Revolution began in Britain...",
        ...     verdict=CriterionVerdict.MET,
        ...     reason="Correctly identifies Britain as origin"
        ... )
    """

    submission: str
    verdict: CriterionVerdict
    reason: str | None = None


@dataclass
class FewShotConfig:
    """Configuration for few-shot example selection.

    Attributes:
        n_examples: Total number of examples to include per criterion.
        balance_verdicts: If True, attempt to balance MET/UNMET/CANNOT_ASSESS.
            If False, randomly sample without balancing.
        include_reason: If True, include the reason/explanation in examples.
            Note: Ground truth datasets typically don't have reasons.
        seed: Random seed for reproducible sampling.

    Example:
        >>> config = FewShotConfig(
        ...     n_examples=3,
        ...     balance_verdicts=True,
        ...     seed=42
        ... )
    """

    n_examples: int = 3
    balance_verdicts: bool = True
    include_reason: bool = False
    seed: int | None = None
