"""A judge's weight must be a positive, finite number.

Aggregation sums judge weights (``weighted``, ``weighted_mean``, ``weighted_mode``) and
compares their sums with zero (``unanimous``, ``any``), so a zero, negative or non-finite
weight would silently change verdicts: a zero-weight UNMET vote could not block
``unanimous``, and weights of +1 and -1 could cancel out.
"""

import math

import pytest

from autorubric import LLMConfig
from autorubric.graders import CriterionGrader, JudgeSpec

CONFIG = LLMConfig(model="openai/gpt-4.1-mini")


@pytest.mark.parametrize("weight", [0, 0.0, -1, -0.5, math.nan, math.inf, -math.inf, True])
def test_a_weight_that_is_not_a_positive_finite_number_is_refused(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        JudgeSpec(CONFIG, "judge", weight=weight)


@pytest.mark.parametrize("weight", [1, 0.25, 3.0])
def test_positive_finite_weights_are_accepted(weight: float) -> None:
    assert JudgeSpec(CONFIG, "judge", weight=weight).weight == weight


def test_the_grader_refuses_a_weight_changed_after_the_spec_was_built() -> None:
    spec = JudgeSpec(CONFIG, "a")
    spec.weight = 0.0

    with pytest.raises(ValueError, match="weight"):
        CriterionGrader(judges=[spec, JudgeSpec(CONFIG, "b")])
