"""``FewShotConfig`` refuses the values example selection would misread (#39).

Selection slices the shuffled training items with ``n_examples``, so a negative value used
to show every eligible item but the last few, and ``0`` showed none while the system prompt
still announced examples. ``seed`` enters the few-shot RNG key as text, so ``True`` or
``1.0`` silently drew other examples than ``1``; ``balance_verdicts`` and
``include_reason`` are read for their truth value, so the string ``"False"`` meant True.
"""

import dataclasses
import math
from typing import Any, cast

import numpy as np
import pytest

from autorubric import CriterionVerdict, FewShotConfig, LLMConfig, Rubric, RubricDataset
from autorubric.graders import CriterionGrader
from autorubric.types import Criterion

CONFIG = LLMConfig(model="openai/gpt-4.1-mini")


def _training_data() -> RubricDataset:
    dataset = RubricDataset(
        prompt="Q", rubric=Rubric([Criterion(name="a", weight=1.0, requirement="A")])
    )
    verdicts = [CriterionVerdict.MET, CriterionVerdict.UNMET] * 3
    for i, verdict in enumerate(verdicts):
        dataset.add_item(submission=f"s{i}", description="d", ground_truth=[verdict])
    return dataset


class TestNExamples:
    @pytest.mark.parametrize("n", [0, -1, -2, True, False, 1.0, 2.5, "3", None, math.nan])
    def test_a_value_that_is_not_an_integer_of_at_least_one_is_refused(self, n: int) -> None:
        with pytest.raises(ValueError, match=r"FewShotConfig\.n_examples must be an integer >= 1"):
            FewShotConfig(n_examples=n)

    @pytest.mark.parametrize("n", [1, 3, 50])
    def test_positive_integers_are_accepted(self, n: int) -> None:
        assert FewShotConfig(n_examples=n).n_examples == n

    def test_a_numpy_integer_is_accepted_as_a_plain_int(self) -> None:
        n = cast(int, np.int64(3))  # accepted, though the annotation names only int
        config = FewShotConfig(n_examples=n)

        assert config.n_examples == 3
        assert type(config.n_examples) is int

    @pytest.mark.parametrize("n", [np.int64(0), np.int64(-2), np.bool_(True), np.float64(3.0)])
    def test_a_numpy_value_that_is_not_a_positive_integer_is_refused(self, n: int) -> None:
        with pytest.raises(ValueError, match=r"FewShotConfig\.n_examples must be an integer >= 1"):
            FewShotConfig(n_examples=n)

    def test_the_message_shows_the_value(self) -> None:
        with pytest.raises(ValueError, match="got -2"):
            FewShotConfig(n_examples=-2)


class TestSeed:
    @pytest.mark.parametrize("seed", [True, False, 1.0, "42", math.nan])
    def test_a_seed_that_is_not_an_integer_or_none_is_refused(self, seed: int | None) -> None:
        with pytest.raises(ValueError, match=r"FewShotConfig\.seed must be None or an integer"):
            FewShotConfig(seed=seed)

    @pytest.mark.parametrize("seed", [None, 0, 42, -7, 2**40])
    def test_none_and_any_integer_are_accepted(self, seed: int | None) -> None:
        assert FewShotConfig(seed=seed).seed == seed

    def test_a_numpy_integer_seed_is_accepted_as_a_plain_int(self) -> None:
        seed = cast(int, np.int64(42))
        config = FewShotConfig(seed=seed)

        assert config.seed == 42
        assert type(config.seed) is int

    @pytest.mark.parametrize("seed", [np.bool_(True), np.float64(1.0)])
    def test_a_numpy_seed_that_is_not_an_integer_is_refused(self, seed: int) -> None:
        with pytest.raises(ValueError, match=r"FewShotConfig\.seed must be None or an integer"):
            FewShotConfig(seed=seed)


class TestFlags:
    @pytest.mark.parametrize("field", ["balance_verdicts", "include_reason"])
    @pytest.mark.parametrize("value", [0, 1, "False", "True", None])
    def test_a_flag_that_is_not_a_bool_is_refused(self, field: str, value: bool) -> None:
        with pytest.raises(ValueError, match=rf"FewShotConfig\.{field} must be True or False"):
            FewShotConfig(**{field: value})

    @pytest.mark.parametrize("field", ["balance_verdicts", "include_reason"])
    @pytest.mark.parametrize("value", [True, False])
    def test_true_and_false_are_accepted(self, field: str, value: bool) -> None:
        assert getattr(FewShotConfig(**{field: value}), field) is value


def test_the_defaults_are_valid() -> None:
    config = FewShotConfig()
    assert (config.n_examples, config.balance_verdicts, config.include_reason, config.seed) == (
        3,
        True,
        False,
        None,
    )


def test_replace_validates_again() -> None:
    with pytest.raises(ValueError, match="n_examples"):
        dataclasses.replace(FewShotConfig(), n_examples=0)


class TestTheGrader:
    @pytest.mark.parametrize("n", [-2, 0])
    @pytest.mark.parametrize("balance", [True, False])
    def test_the_grader_refuses_a_value_set_after_the_config_was_built(
        self, n: int, balance: bool
    ) -> None:
        config = FewShotConfig(balance_verdicts=balance, seed=1)
        config.n_examples = n

        with pytest.raises(ValueError, match="n_examples"):
            CriterionGrader(
                judge_model_config=CONFIG, training_data=_training_data(), few_shot_config=config
            )

    def test_the_grader_refuses_a_bad_seed_set_after_the_config_was_built(self) -> None:
        config = FewShotConfig()
        config.seed = True

        with pytest.raises(ValueError, match="seed"):
            CriterionGrader(
                judge_model_config=CONFIG, training_data=_training_data(), few_shot_config=config
            )

    def test_the_grader_keeps_its_own_copy(self) -> None:
        config = FewShotConfig(n_examples=2, seed=1)
        grader = CriterionGrader(
            judge_model_config=CONFIG, training_data=_training_data(), few_shot_config=config
        )
        config.n_examples = -2

        assert grader._few_shot_config == FewShotConfig(n_examples=2, seed=1)
        assert grader._few_shot_config is not config

    def test_a_valid_count_still_selects_that_many_examples(self) -> None:
        grader = CriterionGrader(
            judge_model_config=CONFIG,
            training_data=_training_data(),
            few_shot_config=FewShotConfig(n_examples=4, seed=1),
        )

        assert len(grader._criterion_examples[(0, "default")]) == 4

    def test_a_numpy_count_selects_that_many_examples(self) -> None:
        n = cast(int, np.int64(4))
        grader = CriterionGrader(
            judge_model_config=CONFIG,
            training_data=_training_data(),
            few_shot_config=FewShotConfig(n_examples=n, seed=1),
        )

        assert len(grader._criterion_examples[(0, "default")]) == 4


class TestTheGraderSeed:
    """The grader's own ``seed`` fills an unset ``FewShotConfig.seed``.

    The grader checks its ``seed`` argument itself (``tests/graders/test_grader_seed.py``),
    so a bad one is refused as the grader's ``seed``, never as ``FewShotConfig.seed``, and
    the copy's seed is the grader's plain ``int``.
    """

    @staticmethod
    def _grader(seed: Any) -> CriterionGrader:
        return CriterionGrader(
            judge_model_config=CONFIG,
            seed=seed,
            training_data=_training_data(),
            few_shot_config=FewShotConfig(n_examples=4, balance_verdicts=False),
        )

    def test_a_numpy_grader_seed_draws_what_the_equal_int_draws(self) -> None:
        def examples(seed: Any) -> list:
            return self._grader(seed)._criterion_examples[(0, "default")]

        assert examples(np.int64(42)) == examples(42)

    def test_the_coordinated_seed_is_a_plain_int(self) -> None:
        seed = self._grader(np.int64(7))._few_shot_config.seed
        assert seed == 7
        assert type(seed) is int

    @pytest.mark.parametrize("seed", [True, 1.0])
    def test_a_bad_grader_seed_is_refused_as_the_graders_own(self, seed: Any) -> None:
        with pytest.raises(ValueError, match=r"^seed must be None or an integer"):
            self._grader(seed)
