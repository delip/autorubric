"""Meta-rubric evaluation functions for assessing rubric quality."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from autorubric.graders import CriterionGrader
from autorubric.llm import LLMConfig
from autorubric.rubric import Rubric
from autorubric.types import CriterionJudgment, EnsembleEvaluationReport

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
        rubric: The rubric to evaluate.
        llm_config: LLM configuration for the evaluation.
        display: Output format - None for no display, "stdout" for terminal,
            "html" for HTML file.
        output_html_path: Path for HTML output (required when display="html").

    Returns:
        EnsembleEvaluationReport with score, raw_score, and per-criterion verdicts.

    Raises:
        ValueError: If display="html" but output_html_path is not provided.
    """
    if display == "html" and output_html_path is None:
        raise ValueError("output_html_path is required when display='html'")

    meta_rubric = get_standalone_meta_rubric()
    grader = CriterionGrader(
        llm_config=llm_config,
        binary_response_format=MetaCriterionJudgment,
    )

    rubric_data = {
        "criteria": [
            {
                "index": i + 1,
                "name": c.name,
                "weight": c.weight,
                "requirement": c.requirement,
                **({"options": [o.model_dump() for o in c.options]} if c.options else {}),
            }
            for i, c in enumerate(rubric.rubric)
        ]
    }
    submission = _CRITERIA_REF_INSTRUCTION + json.dumps(rubric_data, indent=2)

    result = await meta_rubric.grade(to_grade=submission, grader=grader)

    if display is not None:
        rubric_summary = f"Evaluating rubric with {len(rubric.rubric)} criteria"
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
        rubric: The rubric to evaluate.
        task_prompt: The task prompt the rubric is designed to evaluate.
        llm_config: LLM configuration for the evaluation.
        display: Output format - None for no display, "stdout" for terminal,
            "html" for HTML file.
        output_html_path: Path for HTML output (required when display="html").

    Returns:
        EnsembleEvaluationReport with score, raw_score, and per-criterion verdicts.

    Raises:
        ValueError: If display="html" but output_html_path is not provided.
    """
    if display == "html" and output_html_path is None:
        raise ValueError("output_html_path is required when display='html'")

    meta_rubric = get_in_context_meta_rubric()
    grader = CriterionGrader(
        llm_config=llm_config,
        binary_response_format=MetaCriterionJudgment,
    )

    rubric_data = {
        "criteria": [
            {
                "index": i + 1,
                "name": c.name,
                "weight": c.weight,
                "requirement": c.requirement,
                **({"options": [o.model_dump() for o in c.options]} if c.options else {}),
            }
            for i, c in enumerate(rubric.rubric)
        ]
    }
    submission = _CRITERIA_REF_INSTRUCTION + json.dumps(
        {
            "task_prompt": task_prompt,
            "rubric": rubric_data,
        },
        indent=2,
    )

    result = await meta_rubric.grade(to_grade=submission, grader=grader)

    if display is not None:
        prompt_preview = task_prompt[:100] + "..." if len(task_prompt) > 100 else task_prompt
        rubric_summary = (
            f"Evaluating rubric with {len(rubric.rubric)} criteria\n"
            f"Task: {prompt_preview}"
        )
        display_meta_rubric_result(
            result,
            _IN_CONTEXT_META_RUBRIC_PATH,
            title="In-Context Evaluation (rubric + task prompt)",
            rubric_summary=rubric_summary,
            mode=display,
            output_html_path=output_html_path,
        )

    return result
