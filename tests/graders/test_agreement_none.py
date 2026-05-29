"""Issue #6b: empty-rubric ``mean_agreement`` is ``None``, not a fabricated ``1.0``.

Inter-judge agreement is ``None`` when there is nothing to agree on (no criteria).
``EnsembleEvaluationReport.mean_agreement`` is ``float | None``; "perfect agreement"
(``1.0``) must never be fabricated for an empty rubric.
"""

import pytest

from autorubric.graders import CriterionGrader
from autorubric.graders.criterion_grader import JudgeCriterionResults
from autorubric.llm import LLMConfig


@pytest.mark.asyncio
async def test_empty_rubric_mean_agreement_is_none() -> None:
    """Aggregating judges with zero criteria yields ``mean_agreement is None``."""
    grader = CriterionGrader(llm_config=LLMConfig(model="test-model"))
    judge_results = [JudgeCriterionResults(judge_id="j1", weight=1.0, criterion_results=[])]

    report = await grader.aggregate(judge_results)

    assert report.mean_agreement is None
