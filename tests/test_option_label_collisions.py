"""Option labels that would name two options are refused.

A multi-choice label (a ground-truth label, the label ``fill_ground_truth`` records from a
judge's choice) is resolved to an option by ``Criterion.find_option_by_label``, which compares
labels ignoring case and surrounding whitespace and returns the first match. Two options whose
labels compare equal therefore cannot both be named: a label meant for the second resolves to
the first, silently. Two kinds of pair are refused when the criterion is built:

- two of the author's own options with the same label, since the judge is shown both labels
  too;
- a scored option labelled as ``CANONICAL_NA_OPTION`` in a criterion with no NA option. The
  grader adds that option to such a criterion (``Criterion.with_guaranteed_na_option``), and
  every label reader resolves labels among the options shown with it added, whichever way the
  labels were produced. Beside the scored option, a judge's choice of the added NA option was
  recorded under its label and read back as the scored option. ``with_guaranteed_na_option``
  refuses the pair too, for a criterion built without validation.

Nothing here reaches the network: ``litellm.acompletion`` is patched.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autorubric import (
    Criterion,
    CriterionOption,
    LLMConfig,
    Rubric,
)
from autorubric.graders import CriterionGrader
from autorubric.meta._improve import _revised_rubric
from autorubric.metrics import resolve_ground_truth
from autorubric.types import CANONICAL_NA_OPTION

NA_LABEL = CANONICAL_NA_OPTION.label


def _criterion(*labels: str, na: tuple[bool, ...] | None = None) -> Criterion:
    flags = na or (False,) * len(labels)
    values = [i / max(1, len(labels) - 1) for i in range(len(labels))]
    return Criterion(
        name="clarity",
        weight=1.0,
        requirement="How clear is the answer?",
        options=[
            CriterionOption(label=label, value=0.0 if flag else value, na=flag)
            for label, value, flag in zip(labels, values, flags, strict=True)
        ],
    )


def _unvalidated(*labels: str) -> Criterion:
    """A criterion built without validation, as ``model_construct`` builds one."""
    valid = _criterion(*(f"option {i}" for i in range(len(labels))))
    options = [
        option.model_copy(update={"label": label})
        for option, label in zip(valid.options or [], labels, strict=True)
    ]
    return valid.model_copy(update={"options": options})


# A scored option labelled as the NA option the grader adds, and no NA option of its own,
# built past validation (the only way such a criterion can exist).
COLLIDING = _unvalidated("Unclear", NA_LABEL)


# =============================================================================
# The author's own options
# =============================================================================


class TestDuplicateLabels:
    @pytest.mark.parametrize(
        "labels",
        [
            ("Good", "Good"),
            ("Good", "good"),
            ("Good", "  GOOD "),
            ("Bad", "Good", "Fine", "good"),
        ],
    )
    def test_two_options_with_the_same_label_are_refused(self, labels):
        with pytest.raises(ValueError, match="same label"):
            _criterion(*labels)

    def test_the_message_names_both_options(self):
        with pytest.raises(ValueError, match=r"Options 2 and 4 .*'good'"):
            _criterion("Bad", "Good", "Fine", "good")

    def test_an_na_option_counts_too(self):
        with pytest.raises(ValueError, match="same label"):
            _criterion("Bad", "Good", "n/a", "N/A", na=(False, False, False, True))

    def test_two_na_options_with_one_label_are_refused(self):
        with pytest.raises(ValueError, match="same label"):
            _criterion("Bad", "Good", "N/A", "N/A", na=(False, False, True, True))

    @pytest.mark.parametrize(
        "labels",
        [
            ("Good", "Goods"),
            ("not ok", "not  ok"),  # find_option_by_label leaves inner whitespace alone
            ("1", "2", "3"),
        ],
    )
    def test_distinct_labels_are_accepted(self, labels):
        criterion = _criterion(*labels)
        for i, label in enumerate(labels):
            assert criterion.find_option_by_label(label) == i

    def test_a_rubric_file_with_them_is_refused(self):
        data = [
            {
                "requirement": "How clear is the answer?",
                "options": [{"label": "Clear", "value": 1.0}, {"label": "clear", "value": 0.0}],
            }
        ]
        with pytest.raises(ValueError, match="same label"):
            Rubric.from_dict(data)


# =============================================================================
# The NA option the grader adds
# =============================================================================


class TestTheAddedNaOption:
    @pytest.mark.parametrize("label", [NA_LABEL, NA_LABEL.upper(), f"  {NA_LABEL.lower()} "])
    def test_a_scored_option_with_its_label_is_refused(self, label):
        with pytest.raises(ValueError, match="NA option"):
            _criterion("Unclear", label)

    def test_the_message_names_the_criterion_and_the_option(self):
        with pytest.raises(ValueError, match=r"'clarity'.*option 2"):
            _criterion("Unclear", NA_LABEL)

    def test_a_rubric_file_with_one_is_refused(self):
        data = [
            {
                "name": "clarity",
                "requirement": "How clear is the answer?",
                "options": [{"label": "Unclear", "value": 0.0}, {"label": NA_LABEL, "value": 1.0}],
            }
        ]
        with pytest.raises(ValueError, match="NA option"):
            Rubric.from_dict(data)

    def test_a_revised_rubric_with_one_is_refused(self):
        criteria = [
            {
                "requirement": "How clear is the answer?",
                "options": [{"label": "Unclear", "value": 0.0}, {"label": NA_LABEL, "value": 1.0}],
            }
        ]
        with pytest.raises(ValueError, match="NA option"):
            _revised_rubric(criteria, Rubric([_criterion("Unclear", "Clear")]))

    def test_it_is_not_added_to_a_criterion_built_without_validation(self):
        with pytest.raises(ValueError, match="NA option"):
            COLLIDING.with_guaranteed_na_option()

    def test_beside_an_na_option_of_its_own_a_scored_option_may_have_the_label(self):
        criterion = _criterion("Unclear", NA_LABEL, "N/A", na=(False, False, True))
        assert criterion.with_guaranteed_na_option() is criterion
        assert criterion.find_option_by_label(NA_LABEL) == 1

    def test_an_na_option_of_its_own_may_have_the_label(self):
        criterion = _criterion("Unclear", "Clear", NA_LABEL, na=(False, False, True))
        assert criterion.with_guaranteed_na_option() is criterion

    def test_other_criteria_get_it_added(self):
        criterion = _criterion("Unclear", "Clear")
        shown = criterion.with_guaranteed_na_option()
        assert shown.options == [*criterion.options, CANONICAL_NA_OPTION]  # type: ignore[misc]


class TestCriteriaBuiltWithoutValidation:
    """Whatever adds the NA option refuses a criterion that got past validation."""

    @pytest.mark.asyncio
    async def test_grading_with_auto_na_option_is_refused(self):
        async def acompletion(**params):
            raise AssertionError("no judge call is made")

        grader = CriterionGrader(
            judge_model_config=LLMConfig(model="openai/gpt-4.1-mini"), shuffle_options=False
        )
        with patch("litellm.acompletion", side_effect=acompletion):
            with pytest.raises(ValueError, match="NA option"):
                await Rubric([COLLIDING]).grade("An answer.", grader)

    def test_compute_score_refuses_it(self):
        with pytest.raises(ValueError, match="NA option"):
            Rubric([COLLIDING]).compute_score([NA_LABEL])

    def test_resolve_ground_truth_refuses_it(self):
        with pytest.raises(ValueError, match="NA option"):
            resolve_ground_truth([NA_LABEL], [COLLIDING])
