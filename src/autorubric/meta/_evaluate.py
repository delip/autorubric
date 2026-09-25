"""Meta-rubric evaluation functions for assessing rubric quality."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from autorubric.decision import DecisionModelConfig
from autorubric.graders import CriterionGrader
from autorubric.llm import LLMConfig
from autorubric.rubric import Rubric
from autorubric.types import CriterionJudgment, EnsembleEvaluationReport, MultiChoiceJudgment

from ._display import display_meta_rubric_result

_DATA_DIR = Path(__file__).parent / "data"
_STANDALONE_META_RUBRIC_PATH = _DATA_DIR / "meta_rubric_standalone.json"
_IN_CONTEXT_META_RUBRIC_PATH = _DATA_DIR / "meta_rubric_in_context.json"

DisplayMode = Literal["stdout", "html"]

_CRITERIA_REF_INSTRUCTION = (
    "The rubric criteria below are numbered with an index field. "
    "When evaluating, identify which specific criteria (by index) "
    "are relevant to your assessment.\n\n"
)

# Follows _CRITERIA_REF_INSTRUCTION only for a rubric with guidelines, so the meta-judge
# prompts for a rubric without them are unchanged. It states the rule the judges grading
# with the rubric are given: the criterion text governs, the guidelines clarify it.
_GUIDELINES_REF_INSTRUCTION = (
    'The rubric also has guidelines (the "guidelines" field): free text that applies to '
    "every criterion when grading. The criterion text governs; the guidelines clarify how "
    "to apply it. Assess each criterion as a rater would apply it, together with the "
    "guidelines.\n\n"
)


def _meta_judge_instruction(rubric: Rubric) -> str:
    """The instruction that precedes the rubric in the meta-judge's submission."""
    if rubric.guidelines is None:
        return _CRITERIA_REF_INSTRUCTION
    return _CRITERIA_REF_INSTRUCTION + _GUIDELINES_REF_INSTRUCTION


def _rubric_for_meta_judge(rubric: Rubric) -> dict[str, Any]:
    """The rubric under review as the meta-judge sees it.

    Its criteria carry a 1-based ``index`` for ``affected_criteria``. The guidelines, when
    the rubric has them, come first, as in the rubric's on-disk dict form; without them
    the data is exactly what it was before guidelines existed.
    """
    criteria = [
        {
            "index": i + 1,
            "name": c.name,
            "weight": c.weight,
            "requirement": c.requirement,
            **({"options": [o.model_dump() for o in c.options]} if c.options else {}),
        }
        for i, c in enumerate(rubric.rubric)
    ]
    if rubric.guidelines is None:
        return {"criteria": criteria}
    return {"guidelines": rubric.guidelines, "criteria": criteria}


def _rubric_summary(rubric: Rubric) -> str:
    """The display line naming what the meta-judge evaluated."""
    summary = f"Evaluating rubric with {len(rubric.rubric)} criteria"
    if rubric.guidelines is not None:
        summary += " and guidelines"
    return summary


def _not_an_llm(what: str) -> ValueError:
    """The error for a decision model given where meta-rubric work needs an LLM.

    Meta-rubric evaluation and rubric improvement work from generated text: the
    meta-judge's explanations, the evaluation judges' explanations of their verdicts and
    the revised rubric. A decision model (``DecisionModelConfig``) returns probabilities,
    not text, so it can fill none of these roles. Every such entry point raises this
    error, before it builds a client or makes a call.

    Args:
        what: The argument at fault, as the message names it (e.g. ``"llm_config"``).

    Returns:
        The ``ValueError`` to raise.
    """
    return ValueError(
        f"{what} must be an LLM, not a decision model (DecisionModelConfig): meta-rubric "
        "evaluation and rubric improvement work from generated text (judges' explanations, "
        "revised rubrics), which a decision model does not produce"
    )


def _reject_decision_model_judges(judges: Iterable[object], what: str) -> None:
    """Raise ``_not_an_llm`` for the first judge that is a decision model.

    Args:
        judges: ``JudgeSpec``s; one whose ``llm_config`` is a ``DecisionModelConfig`` is
            rejected.
        what: How the message names a judge; its ``judge_id`` follows.

    Raises:
        ValueError: If a judge is a decision model.
    """
    for judge in judges:
        if isinstance(getattr(judge, "llm_config", None), DecisionModelConfig):
            raise _not_an_llm(f"{what} {getattr(judge, 'judge_id', None)!r}")


class MetaCriterionJudgment(CriterionJudgment):
    """Extended judgment for meta-rubric evaluation with structured criterion references."""

    model_config = CriterionJudgment.model_config

    affected_criteria: list[int] = Field(
        default_factory=list,
        description=(
            "1-based indices of the rubric criteria this evaluation pertains to. "
            "Use the index field from the criteria list. "
            "Return an empty list for rubric-wide issues not specific to any criterion."
        ),
    )


class MultiChoiceMetaJudgment(MultiChoiceJudgment):
    """Multi-choice analog of MetaCriterionJudgment with structured criterion references."""

    model_config = MultiChoiceJudgment.model_config

    affected_criteria: list[int] = Field(
        default_factory=list,
        description=(
            "1-based indices of the rubric criteria this evaluation pertains to. "
            "Use the index field from the criteria list. "
            "Return an empty list for rubric-wide issues not specific to any criterion."
        ),
    )


def get_standalone_meta_rubric() -> Rubric:
    """Load the standalone meta-rubric for evaluating rubrics in isolation."""
    return Rubric.from_file(str(_STANDALONE_META_RUBRIC_PATH))


def get_in_context_meta_rubric() -> Rubric:
    """Load the in-context meta-rubric for evaluating rubrics with task context."""
    return Rubric.from_file(str(_IN_CONTEXT_META_RUBRIC_PATH))


async def evaluate_rubric_standalone(
    rubric: Rubric,
    llm_config: LLMConfig,
    *,
    display: DisplayMode | None = None,
    output_html_path: Path | str | None = None,
) -> EnsembleEvaluationReport:
    """Evaluate a rubric's quality in isolation using the standalone meta-rubric.

    This evaluates the rubric's intrinsic quality without considering any specific
    task context. It checks for clarity, structure, LLM-friendliness, and common
    anti-patterns.

    Args:
        rubric: The rubric to evaluate. Its guidelines, when it has them, are shown to
            the meta-judge as part of the rubric under review.
        llm_config: LLM configuration for the evaluation. The meta-judge must be an LLM:
            meta-rubric evaluation works from its explanations, which a decision model
            (``DecisionModelConfig``) does not produce.
        display: Output format - None for no display, "stdout" for terminal,
            "html" for HTML file.
        output_html_path: Path for HTML output (required when display="html").

    Returns:
        EnsembleEvaluationReport with score, raw_score, and per-criterion verdicts.

    Raises:
        ValueError: If llm_config is a decision model, or if display="html" but
            output_html_path is not provided.
    """
    if isinstance(llm_config, DecisionModelConfig):
        raise _not_an_llm("llm_config")
    if display == "html" and output_html_path is None:
        raise ValueError("output_html_path is required when display='html'")

    meta_rubric = get_standalone_meta_rubric()
    grader = CriterionGrader(
        judge_model_config=llm_config,
        binary_response_format=MetaCriterionJudgment,
        multi_choice_response_format=MultiChoiceMetaJudgment,
    )

    submission = _meta_judge_instruction(rubric) + json.dumps(
        _rubric_for_meta_judge(rubric), indent=2
    )

    result = await meta_rubric.grade(to_grade=submission, grader=grader)

    if display is not None:
        rubric_summary = _rubric_summary(rubric)
        display_meta_rubric_result(
            result,
            _STANDALONE_META_RUBRIC_PATH,
            title="Standalone Evaluation (rubric quality in isolation)",
            rubric_summary=rubric_summary,
            mode=display,
            output_html_path=output_html_path,
        )

    return result


async def evaluate_rubric_in_context(
    rubric: Rubric,
    task_prompt: str,
    llm_config: LLMConfig,
    *,
    display: DisplayMode | None = None,
    output_html_path: Path | str | None = None,
) -> EnsembleEvaluationReport:
    """Evaluate a rubric's quality in the context of a specific task.

    This evaluates both intrinsic rubric quality and how well the rubric
    aligns with the given task prompt. It checks for task alignment,
    coverage of key aspects, and task-specific anti-patterns.

    Args:
        rubric: The rubric to evaluate. Its guidelines, when it has them, are shown to
            the meta-judge as part of the rubric under review.
        task_prompt: The task prompt the rubric is designed to evaluate.
        llm_config: LLM configuration for the evaluation. The meta-judge must be an LLM:
            meta-rubric evaluation works from its explanations, which a decision model
            (``DecisionModelConfig``) does not produce.
        display: Output format - None for no display, "stdout" for terminal,
            "html" for HTML file.
        output_html_path: Path for HTML output (required when display="html").

    Returns:
        EnsembleEvaluationReport with score, raw_score, and per-criterion verdicts.

    Raises:
        ValueError: If llm_config is a decision model, or if display="html" but
            output_html_path is not provided.
    """
    if isinstance(llm_config, DecisionModelConfig):
        raise _not_an_llm("llm_config")
    if display == "html" and output_html_path is None:
        raise ValueError("output_html_path is required when display='html'")

    meta_rubric = get_in_context_meta_rubric()
    grader = CriterionGrader(
        judge_model_config=llm_config,
        binary_response_format=MetaCriterionJudgment,
        multi_choice_response_format=MultiChoiceMetaJudgment,
    )

    submission = _meta_judge_instruction(rubric) + json.dumps(
        {
            "task_prompt": task_prompt,
            "rubric": _rubric_for_meta_judge(rubric),
        },
        indent=2,
    )

    result = await meta_rubric.grade(to_grade=submission, grader=grader)

    if display is not None:
        prompt_preview = task_prompt[:100] + "..." if len(task_prompt) > 100 else task_prompt
        rubric_summary = f"{_rubric_summary(rubric)}\nTask: {prompt_preview}"
        display_meta_rubric_result(
            result,
            _IN_CONTEXT_META_RUBRIC_PATH,
            title="In-Context Evaluation (rubric + task prompt)",
            rubric_summary=rubric_summary,
            mode=display,
            output_html_path=output_html_path,
        )

    return result
