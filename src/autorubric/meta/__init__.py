"""Meta-rubric module for evaluating and improving rubric quality.

This module provides utilities to assess the quality of grading rubrics using
meta-rubrics (rubrics for evaluating rubrics). Two evaluation modes are supported:

1. Standalone evaluation: Assesses rubric quality in isolation
2. In-context evaluation: Assesses rubric quality relative to a task prompt

Additionally, the ``improve_rubric`` function provides an iterative improvement
loop that optimizes for meta-rubric quality (validity) and validation reliability.
Supports ground-truth validation (Spearman ρ) and multi-judge agreement modes.

Example usage:
    from autorubric import LLMConfig, Rubric
    from autorubric.meta import evaluate_rubric_standalone, evaluate_rubric_in_context

    llm_config = LLMConfig(model="gpt-4o", temperature=0.0)
    rubric = Rubric.from_file("my_rubric.json")

    # Standalone evaluation (terminal output)
    result = await evaluate_rubric_standalone(rubric, llm_config, display="stdout")

    # In-context evaluation (HTML output)
    result = await evaluate_rubric_in_context(
        rubric, "Write a summary of the article.", llm_config,
        display="html", output_html_path="report.html"
    )

    # Iterative improvement with ground-truth validation
    from autorubric.dataset import RubricDataset
    from autorubric.meta import improve_rubric
    dataset = RubricDataset.from_file("validation_data.json")
    result = await improve_rubric(
        rubric, "Write a summary of the article.",
        eval_llm=llm_config, revision_llm=llm_config,
        validation_data=dataset,
    )
"""

from ._evaluate import (
    evaluate_rubric_in_context,
    evaluate_rubric_standalone,
    get_in_context_meta_rubric,
    get_standalone_meta_rubric,
)
from ._improve import (
    ConvergenceFn,
    CriterionErrorReport,
    CriterionExemplar,
    EvidenceFn,
    HeldOutValidationResult,
    ImprovementConfig,
    ImprovementProgressDisplay,
    ImprovementResult,
    ImprovementRunner,
    IssueDetail,
    IterationResult,
    behavioral_plateau_converged,
    build_revision_history,
    compute_expected_scores,
    diff_issues,
    extract_issues,
    format_agreement_for_prompt,
    format_ground_truth_for_prompt,
    format_held_out_for_prompt,
    format_issues_for_prompt,
    improve_rubric,
    pareto_accept,
    revise_rubric,
    revise_rubric_held_out,
    validate_agreement,
    validate_criteria_structure,
    validate_ground_truth,
    validate_held_out,
)
from ._signals import compute_reward_variance

__all__ = [
    "compute_reward_variance",
    "evaluate_rubric_in_context",
    "evaluate_rubric_standalone",
    "get_in_context_meta_rubric",
    "get_standalone_meta_rubric",
    # Improvement - convenience API
    "improve_rubric",
    # Improvement - runner
    "ImprovementRunner",
    # Improvement - types
    "ConvergenceFn",
    "EvidenceFn",
    "CriterionErrorReport",
    "CriterionExemplar",
    "HeldOutValidationResult",
    "ImprovementConfig",
    "ImprovementProgressDisplay",
    "ImprovementResult",
    "IssueDetail",
    "IterationResult",
    # Improvement - building blocks
    "behavioral_plateau_converged",
    "build_revision_history",
    "compute_expected_scores",
    "diff_issues",
    "extract_issues",
    "format_agreement_for_prompt",
    "format_ground_truth_for_prompt",
    "format_held_out_for_prompt",
    "format_issues_for_prompt",
    "pareto_accept",
    "revise_rubric",
    "revise_rubric_held_out",
    "validate_agreement",
    "validate_criteria_structure",
    "validate_ground_truth",
    "validate_held_out",
]
