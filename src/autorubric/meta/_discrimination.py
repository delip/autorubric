"""Whole-rubric discrimination evaluator.

Given a rubric and a task prompt, generate synthetic submissions spanning a
quality range and grade them with the rubric. Reports score spread, standard
deviation, and rank-correlation between intended quality level and observed
score.

This complements ``evaluate_rubric_standalone`` / ``evaluate_rubric_in_context``:
those judge each criterion independently; this judges whether the rubric *as a
whole* meaningfully separates submission quality. The two questions are
orthogonal — a rubric can have well-formed criteria yet still produce flat
scores across the quality range, or vice versa.

Calibration caveat
------------------
When ``gen_llm`` and ``grader_llm`` are the same model, the generator may
produce outputs the grader is biased toward, inflating the apparent
discrimination. Pass distinct models for high-stakes use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autorubric.graders import CriterionGrader
from autorubric.llm import LLMClient, LLMConfig
from autorubric.rubric import Rubric

_GEN_SYSTEM_PROMPT = (
    "You generate sample submissions for a task at a specified quality level. "
    "Given a task prompt and a target quality level, produce one submission "
    "representative of that level. Output only the submission text — no "
    "preamble, commentary, or quality markers."
)


def _level_descriptor(level: int, n_levels: int) -> str:
    """Map an ordinal level (1-based) to a human-readable quality label."""
    if n_levels <= 1:
        return "average quality"
    bands = ["poor", "fair", "adequate", "good", "excellent"]
    # Linearly map level [1, n_levels] to band index [0, len(bands)-1].
    idx = round((level - 1) * (len(bands) - 1) / (n_levels - 1))
    idx = max(0, min(len(bands) - 1, idx))
    return bands[idx]


@dataclass
class DiscriminationReport:
    """Whole-rubric discrimination diagnostics.

    Attributes:
        score_range: ``max(per_level_scores) - min(per_level_scores)``. Higher
            means the rubric distinguishes the quality range more strongly.
        score_std: Standard deviation of ``per_level_scores``.
        monotonicity: Spearman rank correlation between the intended quality
            level (1..n_levels) and the rubric's per-level score. ``+1`` is
            perfectly monotonic; ``0`` indicates no rank order is preserved.
        n_levels: Number of quality bands the generator was asked to produce.
        per_level_scores: Rubric score for each generated submission, indexed
            level 1..n_levels.
        submissions: The text of each generated submission, indexed identically.
        total_cost: Sum of generation + grading completion costs, when
            available; otherwise None.
    """

    score_range: float
    score_std: float
    monotonicity: float
    n_levels: int
    per_level_scores: list[float] = field(default_factory=list)
    submissions: list[str] = field(default_factory=list)
    total_cost: float | None = None


async def _generate_submission(
    task_prompt: str,
    level: int,
    n_levels: int,
    gen_client: LLMClient,
) -> tuple[str, float | None]:
    descriptor = _level_descriptor(level, n_levels)
    user_prompt = (
        f"Task:\n{task_prompt}\n\n"
        f"Quality level: {level} of {n_levels} ({descriptor}).\n\n"
        "Produce the submission text only."
    )
    result = await gen_client.generate(
        system_prompt=_GEN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        return_result=True,
    )
    return result.content.strip(), result.cost


def _spearman(intended_ranks: list[int], scores: list[float]) -> float:
    """Spearman ρ between two equal-length sequences. Returns 0.0 if undefined."""
    n = len(scores)
    if n < 2:
        return 0.0

    def _ranks(values: list[float]) -> list[float]:
        ordered = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[ordered[j + 1]] == values[ordered[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based average rank
            for k in range(i, j + 1):
                ranks[ordered[k]] = avg
            i = j + 1
        return ranks

    rx = _ranks([float(r) for r in intended_ranks])
    ry = _ranks(scores)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = sum((rx[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


async def evaluate_rubric_discrimination(
    rubric: Rubric,
    task_prompt: str,
    *,
    gen_llm: LLMConfig,
    grader_llm: LLMConfig,
    n_levels: int = 5,
) -> DiscriminationReport:
    """Assess whole-rubric discrimination by grading synthetic submissions.

    Generates ``n_levels`` submissions spanning a quality range using
    ``gen_llm``, grades each with the rubric using ``grader_llm``, and reports
    score spread plus rank correlation between intended and observed ordering.

    Args:
        rubric: The rubric whose discriminative power is being assessed.
        task_prompt: The task the rubric is designed to evaluate. Provided to
            the generator so its synthetic submissions are task-shaped.
        gen_llm: LLM that generates synthetic submissions. Should be capable
            enough to produce plausibly-tiered outputs for the task.
        grader_llm: LLM used to grade each generated submission with the
            rubric. **Should differ from ``gen_llm``** for unbiased
            discrimination — using the same model risks calibration-circular
            results (the grader rewards the generator's stylistic habits).
        n_levels: Number of quality bands to generate (default 5).

    Returns:
        DiscriminationReport with score_range, score_std, monotonicity,
        per-level scores, generated submissions, and total cost.

    Raises:
        ValueError: If ``n_levels < 2``.
    """
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2 to measure discrimination")

    gen_client = LLMClient(gen_llm)
    grader = CriterionGrader(llm_config=grader_llm)

    submissions: list[str] = []
    per_level_scores: list[float] = []
    total_cost: float = 0.0

    for level in range(1, n_levels + 1):
        text, gen_cost = await _generate_submission(
            task_prompt, level, n_levels, gen_client
        )
        submissions.append(text)
        if gen_cost is not None:
            total_cost += gen_cost

        report = await rubric.grade(
            to_grade=text,
            grader=grader,
            query=task_prompt,
        )
        per_level_scores.append(report.score)
        if report.completion_cost is not None:
            total_cost += report.completion_cost

    score_range = max(per_level_scores) - min(per_level_scores)
    mean_score = sum(per_level_scores) / n_levels
    score_std = (
        sum((s - mean_score) ** 2 for s in per_level_scores) / n_levels
    ) ** 0.5
    monotonicity = _spearman(list(range(1, n_levels + 1)), per_level_scores)

    return DiscriminationReport(
        score_range=score_range,
        score_std=score_std,
        monotonicity=monotonicity,
        n_levels=n_levels,
        per_level_scores=per_level_scores,
        submissions=submissions,
        total_cost=total_cost if total_cost > 0 else None,
    )
