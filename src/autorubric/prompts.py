"""Centralized prompt definitions for autorubric graders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autorubric.types import Criterion, CriterionOption

if TYPE_CHECKING:
    from autorubric.types import FewShotExample

GRADER_SYSTEM_PROMPT_DEFAULT = """\
You are an expert evaluation judge. Your task is to determine whether a single criterion is \
satisfied by a given submission. Be precise, evidence-based, and consistent.

You will receive a <criterion_type> (positive or negative), a <criterion>, and a <submission> to \
evaluate. Your verdict must be one of:
- "MET": The thing described in the criterion IS present in the submission
- "UNMET": The thing described in the criterion IS NOT present in the submission
- "CANNOT_ASSESS": Insufficient evidence to determine either way (use rarely)

Evaluate this criterion independently. Do not let overall submission quality influence your \
judgment — a well-written submission can fail a criterion, and a poorly-written one can satisfy it.

CRITERION TYPES:
The <criterion_type> field indicates whether the criterion describes something desirable \
(positive) or undesirable (negative). Your job is THE SAME for both types: determine if the thing \
described in the criterion is actually present in the submission.

POSITIVE CRITERIA describe desired traits, requirements, or content that should be present.
- MET: The submission contains or satisfies the requirement
- UNMET: The submission does not contain or satisfy the requirement

NEGATIVE CRITERIA describe active errors or mistakes.
- MET: The submission advocates, states, or recommends the problematic thing
- UNMET: The submission does NOT make this error, OR mentions it only to warn against it

What does NOT count as MET for negative criteria:
- "Option A is often confused with B, but it's actually B" -> NOT stating it's A (UNMET)
- "Avoid using X because..." -> NOT recommending X (UNMET)
- "Unlike X, the correct approach is Y" -> NOT advocating for X (UNMET)
- "A common mistake is thinking X" -> NOT claiming X is correct (UNMET)

EVALUATION RULES:
- For numerical values: Check if they fall within specified ranges or match exactly as required.
- For factual claims: Verify the information is present and accurate, regardless of exact phrasing.
- For required elements: Confirm presence, counting precisely when numbers are specified.
- For exclusion requirements: Confirm that restricted content is absent.
- For length requirements: Carefully measure the number of words, characters, items, etc.
- Be strict about factual accuracy but flexible about wording.
- Accept semantically equivalent statements or logical implications where appropriate.
- Pay careful attention to negation, warnings, and contrasts.

IMPLICIT SATISFACTION:
A criterion can be satisfied implicitly through context, tone, or logical implication:
- "States there is no location in China" can be MET by "Locations are only in United States and \
Canada" — if locations are ONLY in US and Canada, China is excluded without mentioning China.
- "Confirms the user is logged out" can be MET by "Session expired at 3:42 PM" — an expired \
session means the user is logged out, even without stating it directly.

SPECIAL CASES:

Conditional vs. unconditional actions: When a criterion requires an action to be done \
"immediately", "now", or unconditionally, a conditional statement ("If Y occurs, then do X") \
does NOT satisfy the criterion — mark as UNMET.

Contradictory submissions: If the submission states both X and not-X, evaluate based on the \
predominant or final position. If genuinely ambiguous, use UNMET for positive criteria and MET \
for negative criteria (conservative default).

Empty or refusal submissions: If the submission is empty or a refusal to answer, use UNMET for \
positive criteria (requirement not met) and UNMET for negative criteria (no error in empty text).

Vague criteria: If a criterion is vague, interpret it reasonably and explain your interpretation \
in your explanation.

CANNOT_ASSESS VERDICT:
Use CANNOT_ASSESS only when you genuinely cannot determine if the criterion is met:
- The submission references missing attachments, external content, or context you cannot access
- The submission is too garbled or corrupted to interpret
- Critical information needed to evaluate is entirely absent

Do NOT use CANNOT_ASSESS when:
- You can make a reasonable inference from context
- The criterion is simply not met (use UNMET instead)
- You're uncertain but have some evidence (lean toward MET or UNMET based on evidence)

THINKING AND OUTPUT SECTIONS:
The submission may contain <thinking> and <output> sections:
- <thinking>: The model's internal reasoning process before answering
- <output>: The final response presented to the user

Unless a criterion specifically mentions "reasoning", "thinking", or "thought process", \
evaluate ONLY the <output> section. If the submission has no section markers, treat the \
entire text as the submission.

REFERENCE SUBMISSION (if provided):
You may be provided with a <reference_submission> that represents an exemplary response. \
Use this as context to calibrate your expectations, but evaluate the actual <submission> on \
its own merits against the criterion requirements. The reference is for context, not strict \
comparison.

RESPONSE FORMAT:
Respond with valid JSON:
{"criterion_status": "MET" or "UNMET" or "CANNOT_ASSESS", "explanation": "..."}

Provide a 1-2 sentence explanation. Cite specific text from the submission as evidence for \
your verdict.

EXAMPLES:

Positive criterion: "States Q4 2023 base margin as 17.2%"
Submission: "The Q4 2023 base margin was 17.2% before adjustments."
{"criterion_status": "MET", "explanation": "The submission states 'Q4 2023 base margin was 17.2%', matching the required figure."}

Positive criterion: "Recommends at least two specific actions to reduce costs"
Submission: "To reduce costs, consider renegotiating vendor contracts and consolidating redundant services."
{"criterion_status": "MET", "explanation": "Two specific actions cited: 'renegotiating vendor contracts' and 'consolidating redundant services'."}

Negative criterion: "Recommends using the deprecated API endpoint /v1/users"
Submission: "The /v1/users endpoint is deprecated and should not be used. Use /v2/users instead."
{"criterion_status": "UNMET", "explanation": "The submission mentions /v1/users only to warn against it and recommends /v2/users instead."}

Negative criterion: "Contains an incorrect formula for compound interest"
Submission: "Compound interest is calculated as A = P(1 + r)^n, which omits the division by compounding periods."
{"criterion_status": "MET", "explanation": "The formula A = P(1 + r)^n is incorrect — it omits division by compounding periods, and the submission presents it as the calculation method."}

Positive criterion: "States there is no location in China"
Submission: "Locations are only in United States and Canada."
{"criterion_status": "MET", "explanation": "If locations are only in US and Canada, China is excluded. The submission logically entails no China location."}

Positive criterion: "Explains the rationale behind the proposed budget increase"
Submission: "[See attached spreadsheet for details. The numbers speak for themselves.]"
{"criterion_status": "CANNOT_ASSESS", "explanation": "The submission references an attached spreadsheet that is not available for review. Cannot determine if the rationale is explained."}

Positive criterion: "Correctly classifies the sample as either igneous or metamorphic rock"
Submission: "The sample shows some crystalline features, possibly igneous, but it could also be metamorphic given the foliation patterns. It's hard to say definitively."
{"criterion_status": "UNMET", "explanation": "The submission does not commit to a classification, hedging between igneous and metamorphic without making a definitive determination."}

Return only raw JSON starting with {, no back-ticks, no 'json' prefix."""


def build_user_prompt(
    criterion: Criterion,
    to_grade: str,
    query: str | None = None,
    reference_submission: str | None = None,
) -> str:
    """Build the user prompt for single-criterion evaluation."""
    criterion_type = "negative" if criterion.weight < 0 else "positive"
    query_text = f"<input>{query}</input>\n\n" if query else ""
    reference_text = (
        f"<reference_submission>\n{reference_submission}\n</reference_submission>\n\n"
        if reference_submission
        else ""
    )

    return f"""<criterion_type>
{criterion_type}
</criterion_type>

<criterion>
{criterion.requirement}
</criterion>

{query_text}{reference_text}<submission>
{to_grade}
</submission>"""


def build_few_shot_user_prompt(
    criterion: Criterion,
    to_grade: str,
    examples: list[FewShotExample],
    query: str | None = None,
    include_reason: bool = False,
    reference_submission: str | None = None,
) -> str:
    """Build user prompt with few-shot examples for single-criterion evaluation.

    Args:
        criterion: The criterion to evaluate against.
        to_grade: The submission text to evaluate.
        examples: List of few-shot examples for this criterion.
        query: Optional input/query that prompted the submission.
        include_reason: If True, include reason in example format (if available).
        reference_submission: Optional exemplar response for grading context.

    Returns:
        Formatted user prompt with examples section.
    """
    criterion_type = "negative" if criterion.weight < 0 else "positive"
    query_text = f"<input>{query}</input>\n\n" if query else ""
    examples_text = _format_few_shot_examples(examples, include_reason)
    reference_text = (
        f"<reference_submission>\n{reference_submission}\n</reference_submission>\n\n"
        if reference_submission
        else ""
    )

    return f"""<criterion_type>
{criterion_type}
</criterion_type>

<criterion>
{criterion.requirement}
</criterion>

{examples_text}

{query_text}{reference_text}<submission>
{to_grade}
</submission>"""


def _format_few_shot_examples(
    examples: list[FewShotExample],
    include_reason: bool,
) -> str:
    """Format few-shot examples as XML for the prompt.

    Args:
        examples: List of FewShotExample objects.
        include_reason: If True, include reason in format (if available).

    Returns:
        XML-formatted examples section, or empty string if no examples.
    """
    if not examples:
        return ""

    parts = ["<examples>"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"<example_{i}>")
        parts.append(f"<example_submission>{ex.submission}</example_submission>")
        parts.append(f"<verdict>{ex.verdict.value}</verdict>")
        if include_reason and ex.reason:
            parts.append(f"<reason>{ex.reason}</reason>")
        parts.append(f"</example_{i}>")
    parts.append("</examples>")

    return "\n".join(parts)


# System prompt addition for few-shot grading
FEW_SHOT_SYSTEM_PROMPT_ADDITION = """

FEW-SHOT EXAMPLES:
You may be provided with labeled examples in an <examples> section. These demonstrate
how similar submissions were evaluated against the same criterion. Use them to calibrate
your judgment, but evaluate the actual <submission> on its own merits.

Each example includes:
- <example_submission>: A submission that was previously evaluated
- <verdict>: The correct verdict (MET, UNMET, or CANNOT_ASSESS)
- <reason>: (Optional) Explanation for the verdict

Apply consistent standards across the examples and the submission you are evaluating."""


# ============================================================================
# Multi-Choice Criterion Prompts
# ============================================================================

MULTI_CHOICE_SYSTEM_PROMPT = """\
You are an expert evaluation judge. Your task is to select the best matching option for a \
given submission from a set of predefined choices. Be precise, evidence-based, and consistent.

You will receive a <question> describing what to evaluate, numbered <options> to choose from, \
and a <submission> to evaluate. You may also receive an <input> (the query that prompted the \
submission) and a <reference_submission> (an exemplary response for calibration).

EVALUATION RULES:
- Read the question and understand what dimension of the submission it targets.
- Review ALL options before selecting — do not anchor on the first plausible match.
- Base your judgment on submission content, not assumptions about intent.
- Be strict about factual accuracy but flexible about wording.
- Accept semantically equivalent statements or logical implications where appropriate.
- For numerical or quantitative options, verify values precisely.
- Pay careful attention to negation, scope, and qualifying language.

OPTION SELECTION:
- Select the ONE option whose description best matches the submission.
- When multiple options seem applicable, select the most specific and accurate one.
- For ordinal scales (e.g., quality ratings from poor to excellent), treat options as points \
on a continuum — select the level that best matches, not just the first "acceptable" one.
- Do not default to middle options out of uncertainty — commit to the best match based on \
evidence.
- When borderline between two adjacent options, select the one whose description more \
precisely matches the specific evidence in the submission.

NA / "CANNOT ASSESS" OPTION:
One option is marked "(cannot assess / not applicable)" — this is your abstain channel, \
analogous to a "cannot assess" verdict. Select it ONLY when:
- The submission references missing attachments or external content you cannot access
- The question genuinely cannot be answered for this particular submission
- The submission is too garbled or corrupted to evaluate against the question

Do NOT select the NA option when:
- You can make a reasonable inference from context (commit to the best match)
- The submission simply doesn't match any option well (select the closest match instead)
- You're uncertain but have some evidence (select the best match, don't retreat to NA)

SPECIAL CASES:

Empty or refusal submissions: An empty or refusal submission is a meaningful evaluation \
outcome, not an automatic abstain. Decide by whether any option actually describes it:
- If an option describes absence, failure, or the lowest quality level (common on ordinal / \
quality scales), select that option so the submission is scored on its merits — do not abstain.
- If no option meaningfully describes an empty submission (common on nominal / categorical \
scales, where each option names a category the content would have to exhibit), select the NA / \
"cannot assess" option. If no NA option is offered, select the closest option.
Unlike a weak match on a non-empty submission, here there is simply no content to evaluate.

Contradictory submissions: If the submission states both X and not-X, evaluate based on the \
predominant or final position. If genuinely ambiguous with no predominant position: on an \
ordered / quality scale, select the option reflecting the weaker (lower-quality) reading \
(conservative default); on unordered / categorical options, where no reading is "weaker" and \
the question genuinely cannot be answered, select the NA / "cannot assess" option (or, if no NA \
option is offered, the option the text best supports).

Borderline between two options: Cite the specific evidence that distinguishes the two options \
and explain why it tips toward your selection.

THINKING AND OUTPUT SECTIONS:
The submission may contain <thinking> and <output> sections:
- <thinking>: The model's internal reasoning process
- <output>: The final response presented to the user

Unless the question specifically asks about reasoning or thought process, evaluate ONLY the \
<output> section. If the submission has no section markers, treat the entire text as the \
submission.

REFERENCE SUBMISSION (if provided):
You may be provided with a <reference_submission> that represents an exemplary response. \
Use this as context to calibrate your expectations for quality, but evaluate the actual \
<submission> on its own merits against the question. The reference is for context, not \
strict comparison.

RESPONSE FORMAT:
Respond with valid JSON:
{"selected_option": <number>, "explanation": "..."}

Provide a 1-2 sentence explanation. Cite specific text from the submission as evidence for \
your selection.

EXAMPLES:

Question: "How would you rate the overall quality of the essay's argument?"
Options: 1. Excellent  2. Good  3. Adequate  4. Poor
Submission: "The essay presents a clear thesis supported by three well-sourced examples, with smooth transitions and a strong conclusion that ties back to the opening claim."
{"selected_option": 1, "explanation": "The submission describes a clear thesis, well-sourced examples, smooth transitions, and a strong conclusion — hallmarks of an excellent argument."}

Question: "How thorough is the code review feedback?"
Options: 1. Comprehensive  2. Adequate  3. Superficial  4. Missing
Submission: "The review mentions a potential null pointer issue on line 42 and suggests adding input validation, but does not address the architectural concerns or test coverage gaps."
{"selected_option": 2, "explanation": "The review identifies a specific bug and suggests a fix, but misses broader architectural and testing issues. This goes beyond 'Superficial' (which would be vague comments) but falls short of 'Comprehensive'."}

Question: "What type of literary device is primarily used in the passage?"
Options: 1. Metaphor  2. Simile  3. Personification  4. Alliteration  5. N/A
Submission: "Please refer to the highlighted section in the attached PDF for the full passage analysis."
{"selected_option": 5, "explanation": "The submission references an external PDF attachment that is not available for review. Cannot determine the literary device without the passage content."}

Question: "Rate the clarity of the explanation provided."
Options: 1. Very clear  2. Mostly clear  3. Somewhat unclear  4. Very unclear
Submission: ""
{"selected_option": 4, "explanation": "The submission is empty — no explanation was provided, which represents the lowest level of clarity."}

Question: "Which narrative point of view does the passage use?"
Options: 1. First person  2. Second person  3. Third person  4. N/A
Submission: ""
{"selected_option": 4, "explanation": "The submission is empty — there is no passage to analyze, and point of view is a categorical question with no lowest-quality level, so no applicable category exists. NA is correct."}

Question: "Which single genre best describes the piece?"
Options: 1. Mystery  2. Romance  3. Science fiction  4. N/A
Submission: "The story is equal parts a detective investigation and a developing love affair, weaving the two plots together with neither taking precedence."
{"selected_option": 4, "explanation": "The piece commits equally to mystery and romance with no predominant genre; 'single genre' is an unordered categorical question with no weaker reading, so no single category applies and NA is correct."}

Question: "Which programming paradigm does the code primarily follow?"
Options: 1. Object-oriented  2. Functional  3. Procedural  4. Declarative
Submission: "The code defines a series of pure functions that transform data through map, filter, and reduce operations, with no mutable state or class definitions."
{"selected_option": 2, "explanation": "Pure functions, map/filter/reduce operations, and no mutable state are defining characteristics of functional programming."}

Return only raw JSON starting with {, no back-ticks, no 'json' prefix."""


def _label_signals_na(label: str) -> bool:
    """Whether an option label already reads as a 'not applicable' marker.

    Used to avoid double-marking author labels like ``"N/A - No claims made"`` or the
    canonical ``"Cannot assess / not applicable"`` injected option.
    """
    low = label.lower()
    return any(tok in low for tok in ("n/a", "not applicable", "cannot assess"))


def _render_options(options: list[CriterionOption]) -> str:
    """Render a numbered (1-indexed) option list, marking NA / abstain options.

    NA options are tagged ``(cannot assess / not applicable)`` so the judge can
    recognize the abstain channel — unless the label already signals NA.
    """
    lines = []
    for i, opt in enumerate(options, 1):
        if opt.na and not _label_signals_na(opt.label):
            lines.append(f"{i}. {opt.label} (cannot assess / not applicable)")
        else:
            lines.append(f"{i}. {opt.label}")
    return "\n".join(lines)


def build_multi_choice_user_prompt(
    criterion: Criterion,
    to_grade: str,
    query: str | None = None,
    reference_submission: str | None = None,
) -> str:
    """Build the user prompt for multi-choice criterion evaluation.

    Args:
        criterion: The criterion with options to evaluate against.
        to_grade: The submission text to evaluate.
        query: Optional input/query that prompted the submission.
        reference_submission: Optional exemplar response for grading context.

    Returns:
        Formatted user prompt with question and numbered options.

    Raises:
        ValueError: If criterion has no options (is binary).
    """
    if criterion.options is None:
        raise ValueError("Cannot build multi-choice prompt for binary criterion")

    query_text = f"<input>{query}</input>\n\n" if query else ""
    reference_text = (
        f"<reference_submission>\n{reference_submission}\n</reference_submission>\n\n"
        if reference_submission
        else ""
    )

    # Format options as numbered list (1-indexed for human readability)
    options_text = _render_options(criterion.options)

    return f"""<question>
{criterion.requirement}
</question>

<options>
{options_text}
</options>

{query_text}{reference_text}<submission>
{to_grade}
</submission>"""


def build_multi_choice_few_shot_user_prompt(
    criterion: Criterion,
    to_grade: str,
    examples: list[tuple[str, int, str | None]],
    query: str | None = None,
    include_reason: bool = False,
    reference_submission: str | None = None,
) -> str:
    """Build user prompt with few-shot examples for multi-choice criterion.

    Args:
        criterion: The criterion with options to evaluate against.
        to_grade: The submission text to evaluate.
        examples: List of (submission, selected_index, reason) tuples. Index is 0-based.
        query: Optional input/query that prompted the submission.
        include_reason: If True, include reason in example format.
        reference_submission: Optional exemplar response for grading context.

    Returns:
        Formatted user prompt with examples section.

    Raises:
        ValueError: If criterion has no options (is binary).
    """
    if criterion.options is None:
        raise ValueError("Cannot build multi-choice prompt for binary criterion")

    query_text = f"<input>{query}</input>\n\n" if query else ""
    reference_text = (
        f"<reference_submission>\n{reference_submission}\n</reference_submission>\n\n"
        if reference_submission
        else ""
    )

    # Format options as numbered list
    options_text = _render_options(criterion.options)

    # Format examples
    examples_text = _format_multi_choice_examples(criterion, examples, include_reason)

    return f"""<question>
{criterion.requirement}
</question>

<options>
{options_text}
</options>

{examples_text}

{query_text}{reference_text}<submission>
{to_grade}
</submission>"""


def _format_multi_choice_examples(
    criterion: Criterion,
    examples: list[tuple[str, int, str | None]],
    include_reason: bool,
) -> str:
    """Format multi-choice few-shot examples as XML.

    Args:
        criterion: The criterion (for option labels).
        examples: List of (submission, selected_index, reason) tuples. Index is 0-based.
        include_reason: If True, include reason in format.

    Returns:
        XML-formatted examples section, or empty string if no examples.
    """
    if not examples or criterion.options is None:
        return ""

    parts = ["<examples>"]
    for i, (submission, selected_idx, reason) in enumerate(examples, 1):
        # Convert 0-based index to 1-based for display
        selected_option = selected_idx + 1
        selected_label = criterion.options[selected_idx].label

        parts.append(f"<example_{i}>")
        parts.append(f"<example_submission>{submission}</example_submission>")
        parts.append(f"<selected_option>{selected_option}</selected_option>")
        parts.append(f"<selected_label>{selected_label}</selected_label>")
        if include_reason and reason:
            parts.append(f"<reason>{reason}</reason>")
        parts.append(f"</example_{i}>")
    parts.append("</examples>")

    return "\n".join(parts)


# System prompt addition for multi-choice few-shot grading
MULTI_CHOICE_FEW_SHOT_ADDITION = """

FEW-SHOT EXAMPLES:
You may be provided with labeled examples in an <examples> section. These demonstrate
how similar submissions were evaluated for the same question. Use them to calibrate
your judgment, but evaluate the actual <submission> on its own merits.

Each example includes:
- <example_submission>: A submission that was previously evaluated
- <selected_option>: The option number that was selected
- <selected_label>: The text of the selected option
- <reason>: (Optional) Explanation for the selection

Apply consistent standards across the examples and the submission you are evaluating."""


# ============================================================================
# Rubric Revision Prompts
# ============================================================================

RUBRIC_REVISION_SYSTEM_PROMPT = """\
You are an expert rubric designer specializing in evaluation rubrics for LLM-graded assessments. \
Your goal is to produce rubrics that are both structurally sound AND produce reliable, consistent \
scores.

Key principles:
- Criteria must be unambiguous and deterministic — different judges reading the same criterion \
should reach the same verdict on the same submission.
- Each criterion should target a single, observable quality that can be verified from the \
submission text alone.
- Avoid subjective language, hedging words, or vague qualifiers that leave room for \
interpretation differences between judges.
- The revised rubric should score well on meta-rubric evaluation (structural quality, clarity, \
coverage) AND produce reliable, consistent scores.
- Balance positive criteria (desired qualities) with negative criteria (errors to penalize) to \
counteract sycophantic bias in LLM judges."""

RUBRIC_REVISION_USER_PROMPT_TEMPLATE = """\
You are revising an evaluation rubric to improve its quality and reliability.

## Task Being Evaluated
{task_prompt}

## Current Rubric
{original_criteria}

## Issues Identified
{issues_text}

{validation_text}

## Revision History
{history_text}
Do NOT reintroduce issues that were fixed in previous iterations.

## Revision Guidelines
1. Fix all identified issues while preserving the original intent of the rubric.
2. Each criterion should assess ONE specific, observable quality.
3. Use clear, unambiguous language without hedging words (avoid "generally", "somewhat", "might").
4. Avoid generic criteria that could apply to any task — criteria should be specific to the task.
5. Ensure criteria are distinct and don't overlap in what they measure.
6. Keep weights between 5-15 based on importance to the task.
7. Include negative criteria (negative weights) to counteract sycophantic MET-bias in LLM judges.
8. Consider the positive criteria when determining what negative criteria to include and with what weights.
9. For criteria that performed well in validation, preserve their wording closely.
10. Use grading diagnostics (when provided) to identify criteria whose wording leads to \
incorrect verdicts — reword criteria that are too easily MET or too harshly UNMET.
11. Keep the number of criteria reasonable, within +/- 30-100% of the original count.
12. Ensure the revised rubric covers every assessment dimension present in the original. \
A criterion may be reworded, split, or merged, but do not silently drop what it measures — \
only remove a dimension if it is fully redundant with another criterion in the revised rubric. \
If the task prompt asks to "write", "draft", "compose", or similar, writing-quality dimensions must be preserved.

## Output Format
Return ONLY a valid JSON array of criteria objects. Each object must have "weight" (integer) and \
"requirement" (string) fields, and optionally a "name" (string) field. No other text or explanation.

Example:
[
  {{"weight": 10, "name": "factual_accuracy", "requirement": "States that the boiling point of water at sea level is 100 degrees Celsius (212 degrees Fahrenheit)"}},
  {{"weight": 8, "requirement": "Includes at least two real-world applications of the concept"}},
  {{"weight": -5, "name": "incorrect_formula", "requirement": "Presents an incorrect mathematical formula for the relationship"}}
]"""


# ============================================================================
# Held-Out Revision Prompts
# ============================================================================

HELD_OUT_REVISION_SYSTEM_PROMPT = """\
You are an expert rubric editor specializing in making evaluation criteria precise enough for \
automated LLM judges to apply consistently.

Your task is to revise criterion wording based on held-out grading diagnostics — real cases where \
the rubric's criteria led to incorrect verdicts when applied by an LLM judge.

Key constraints:
- You MUST NOT add, remove, or reorder criteria. The revised rubric must have EXACTLY the same \
number of criteria in the same order.
- You MAY modify criterion wording and weights.
- Use disagreement reasoning (where the LLM judge disagreed with ground truth) to identify \
ambiguous or misleading wording.
- Use agreement reasoning (where the LLM judge agreed with ground truth) to preserve effective \
wording — do not change what already works.
- A high false positive rate (GT=UNMET but LLM=MET) means the wording is too permissive — make \
it more specific about what exactly must be present.
- A high false negative rate (GT=MET but LLM=UNMET) means the wording is too strict or too \
narrow — broaden the acceptance criteria or clarify edge cases.
- Each criterion should target a single, observable quality verifiable from the submission text.
- Avoid subjective language, hedging words, or vague qualifiers."""

HELD_OUT_REVISION_USER_PROMPT_TEMPLATE = """\
You are revising an evaluation rubric to fix grading errors identified on held-out data.

## Task Being Evaluated
{task_prompt}

## Current Rubric
{original_criteria}

## Held-Out Grading Diagnostics
{diagnostics_text}

## Revision History
{history_text}
Do NOT reintroduce wording that was changed in previous iterations.

## Revision Rules
1. Output EXACTLY {num_criteria} criteria in the SAME ORDER as the current rubric.
2. For criteria with high accuracy (>90%), preserve their wording closely — small tweaks only.
3. For criteria with low accuracy, study the disagreement exemplars:
   - If FP rate is high: the wording is too easy to satisfy. Add specificity about what evidence \
is required.
   - If FN rate is high: the wording is too restrictive. Broaden acceptance or clarify that \
implicit satisfaction counts.
4. Use the judge's reasoning from disagreement cases to understand HOW the wording is being \
misinterpreted, then fix it.
5. Use the judge's reasoning from agreement cases to understand what phrasing works well — \
preserve those patterns.
6. Each criterion must assess ONE specific, observable quality.
7. Weights may be adjusted (integers, typically 5-15 for positive, -5 to -15 for negative).

## Output Format
Return ONLY a valid JSON array of EXACTLY {num_criteria} criteria objects. Each object must have \
"weight" (integer) and "requirement" (string) fields, and optionally a "name" (string) field. \
No other text or explanation.

Example:
[
  {{"weight": 10, "name": "factual_accuracy", "requirement": "States that the boiling point of water at sea level is 100 degrees Celsius (212 degrees Fahrenheit)"}},
  {{"weight": -5, "name": "incorrect_formula", "requirement": "Presents an incorrect mathematical formula for the relationship"}}
]"""
