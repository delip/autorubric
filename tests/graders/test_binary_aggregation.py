"""Tests for binary ensemble vote aggregation strategies.

Pins the semantics of ``CriterionGrader._aggregate_votes`` for each
``AggregationStrategy``:

- ``majority`` is an **unweighted head-count** (> 50% of judges), distinct from
  ``weighted`` (sum of judge weights). This is the fix: previously the two
  branches were byte-identical.
- ``weighted`` decides by summed judge weights.
- ``unanimous`` / ``any`` behave as documented.
- CANNOT_ASSESS votes are excluded from the count.
- Ties resolve to the weight-sign worst case: UNMET for weight >= 0, MET for
  weight < 0 (the binary analog of ``Criterion.worst_scored_option``).
"""

import pytest

from autorubric.graders import CriterionGrader
from autorubric.llm import LLMConfig
from autorubric.types import AggregationStrategy, CriterionVerdict, JudgeVote

MET = CriterionVerdict.MET
UNMET = CriterionVerdict.UNMET
CANNOT_ASSESS = CriterionVerdict.CANNOT_ASSESS


def _grader(aggregation: AggregationStrategy) -> CriterionGrader:
    return CriterionGrader(llm_config=LLMConfig(model="test-model"), aggregation=aggregation)


def _votes(*specs: tuple[CriterionVerdict, float]) -> list[JudgeVote]:
    return [
        JudgeVote(judge_id=f"judge-{i}", verdict=verdict, reason="r", weight=weight)
        for i, (verdict, weight) in enumerate(specs)
    ]


def test_majority_is_head_count_not_weighted() -> None:
    """The pin: with unequal weights, majority (heads) and weighted disagree.

    3 judges, weights [3, 1, 1], verdicts [UNMET, MET, MET]:
    - majority: 2 MET heads vs 1 UNMET head -> MET
    - weighted: 2.0 MET weight vs 3.0 UNMET weight -> UNMET
    """
    votes = _votes((UNMET, 3.0), (MET, 1.0), (MET, 1.0))

    majority_verdict, _ = _grader("majority")._aggregate_votes(votes, weight=10.0)
    weighted_verdict, _ = _grader("weighted")._aggregate_votes(votes, weight=10.0)

    assert majority_verdict == MET
    assert weighted_verdict == UNMET
    assert majority_verdict != weighted_verdict


def test_majority_ignores_lopsided_minority_weight() -> None:
    """A single heavily-weighted dissenter cannot overturn the head-count."""
    votes = _votes((UNMET, 100.0), (MET, 1.0), (MET, 1.0))
    verdict, _ = _grader("majority")._aggregate_votes(votes, weight=10.0)
    assert verdict == MET


def test_majority_excludes_cannot_assess_from_count() -> None:
    """CANNOT_ASSESS votes drop out; [MET, CA, UNMET] -> tie among assessable -> UNMET."""
    votes = _votes((MET, 1.0), (CANNOT_ASSESS, 1.0), (UNMET, 1.0))
    verdict, _ = _grader("majority")._aggregate_votes(votes, weight=10.0)
    assert verdict == UNMET


def test_weighted_uses_summed_weights() -> None:
    """Weighted lets a heavy judge win: [MET@3, UNMET, UNMET] -> MET (3 > 2)."""
    votes = _votes((MET, 3.0), (UNMET, 1.0), (UNMET, 1.0))
    verdict, _ = _grader("weighted")._aggregate_votes(votes, weight=10.0)
    assert verdict == MET


def test_unanimous_requires_all_met() -> None:
    grader = _grader("unanimous")
    assert grader._aggregate_votes(_votes((MET, 1.0), (MET, 1.0)), weight=10.0)[0] == MET
    assert grader._aggregate_votes(_votes((MET, 1.0), (UNMET, 1.0)), weight=10.0)[0] == UNMET


def test_any_met_wins() -> None:
    grader = _grader("any")
    assert grader._aggregate_votes(_votes((UNMET, 5.0), (MET, 1.0)), weight=10.0)[0] == MET
    assert grader._aggregate_votes(_votes((UNMET, 1.0), (UNMET, 1.0)), weight=10.0)[0] == UNMET


def test_all_cannot_assess_returns_cannot_assess() -> None:
    votes = _votes((CANNOT_ASSESS, 1.0), (CANNOT_ASSESS, 1.0))
    verdict, reason = _grader("majority")._aggregate_votes(votes, weight=10.0)
    assert verdict == CANNOT_ASSESS


# Ties resolve to the score-minimizing verdict by weight sign (worst case),
# matching Criterion.worst_scored_option / the unknown-error path. Positive weight →
# UNMET (earns 0); negative (penalty) weight → MET (subtracts the full penalty).
#
# Both dispatch paths preserved as params: majority is an equal-weight 4-vote even split
# (head-count branch); weighted is 2 equal-summed votes (summed-weight branch). The
# positive-weight majority row also absorbs the former standalone conservative-tie test.


@pytest.mark.parametrize(
    ("aggregation", "votes_spec", "weight", "expected"),
    [
        # majority: 2 MET vs 2 UNMET, equal weights (head-count even split).
        ("majority", ((MET, 1.0), (MET, 1.0), (UNMET, 1.0), (UNMET, 1.0)), 10.0, UNMET),
        ("majority", ((MET, 1.0), (MET, 1.0), (UNMET, 1.0), (UNMET, 1.0)), -10.0, MET),
        # weighted: equal summed weights (summed-weight tie).
        ("weighted", ((MET, 2.0), (UNMET, 2.0)), 5.0, UNMET),
        ("weighted", ((MET, 2.0), (UNMET, 2.0)), -5.0, MET),
    ],
)
def test_binary_tie_falls_to_weight_sign_worst_case(
    aggregation: AggregationStrategy,
    votes_spec: tuple[tuple[CriterionVerdict, float], ...],
    weight: float,
    expected: CriterionVerdict,
) -> None:
    """Binary tie → score-minimizing verdict by weight sign (UNMET for >=0, MET for <0)."""
    votes = _votes(*votes_spec)
    verdict, _ = _grader(aggregation)._aggregate_votes(votes, weight=weight)
    assert verdict == expected
