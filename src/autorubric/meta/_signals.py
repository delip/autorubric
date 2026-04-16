"""Behavioral signal computation for rubric criteria."""

from __future__ import annotations

import statistics
from dataclasses import replace

from autorubric.graders import CriterionGrader
from autorubric.llm import LLMConfig
from autorubric.rubric import Rubric
from autorubric.types import CriterionVerdict


def _criterion_key(name: str | None, requirement: str) -> str:
    return name or requirement[:30]


async def compute_reward_variance(
    rubric: Rubric,
    items: list[str],
    *,
    llm_config: LLMConfig,
    n_samples: int = 3,
    seed: int | None = None,
    task_prompt: str | None = None,
) -> dict[str, float]:
    """Compute per-criterion verdict variance across repeated evaluations.

    For each item, grades it ``n_samples`` times with different seeds and
    temperature > 0 to measure how much each criterion's verdict varies.
    High variance indicates unreliable criteria.

    Args:
        rubric: The rubric to analyze.
        items: Submission texts to grade.
        llm_config: LLM configuration. Temperature is forced > 0 for variance
            measurement; seed is overridden per sample.
        n_samples: Number of repeated gradings per item.
        seed: Base seed for reproducibility. Each sample uses ``seed + i``.
        task_prompt: Optional task prompt passed as ``query`` to grading.

    Returns:
        Dict mapping criterion name to variance of the binary score
        (1.0 for MET, 0.0 for UNMET) across all (item, sample) pairs.
        CANNOT_ASSESS verdicts are excluded. Range: 0.0 (perfectly
        consistent) to 0.25 (maximum variance for a binary variable).
    """
    if not items:
        return {
            _criterion_key(c.name, c.requirement): 0.0
            for c in rubric.rubric
        }

    base_seed = seed or 0
    # Force temperature > 0 for meaningful variance
    base_config = replace(llm_config, temperature=max(llm_config.temperature, 0.5))

    verdicts_by_criterion: dict[str, list[float]] = {
        _criterion_key(c.name, c.requirement): []
        for c in rubric.rubric
    }

    for sample_idx in range(n_samples):
        sample_config = replace(base_config, seed=base_seed + sample_idx)
        grader = CriterionGrader(llm_config=sample_config)

        for item in items:
            report = await rubric.grade(
                to_grade=item,
                grader=grader,
                query=task_prompt,
            )
            if report.report is None:
                continue
            for cr in report.report:
                key = _criterion_key(cr.name, cr.requirement)
                if key not in verdicts_by_criterion:
                    continue
                if cr.verdict == CriterionVerdict.CANNOT_ASSESS:
                    continue
                verdicts_by_criterion[key].append(
                    1.0 if cr.verdict == CriterionVerdict.MET else 0.0
                )

    result: dict[str, float] = {}
    for key, values in verdicts_by_criterion.items():
        if len(values) < 2:
            result[key] = 0.0
        else:
            result[key] = statistics.variance(values)

    return result
