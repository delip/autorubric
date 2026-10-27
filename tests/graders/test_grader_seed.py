"""``CriterionGrader(seed=...)`` refuses a seed it would misread.

The master seed enters every shuffle and few-shot RNG key as text, so ``True`` or ``1.0``
used to shuffle options and draw examples differently from ``1`` without any error. A
numpy-integer seed shuffled as the equal ``int`` does (its text is the same), but the
manifest records ``master_seed`` only for an ``int``, so the run's seed went unrecorded.
The grader now takes any integer but a ``bool``, as ``FewShotConfig.seed`` does, and keeps
it as a plain ``int``.
"""

import math
from typing import Any

import numpy as np
import pytest

from autorubric import CriterionVerdict, FewShotConfig, LLMConfig, Rubric, RubricDataset
from autorubric.eval import _serialize_grader_config
from autorubric.graders import CriterionGrader
from autorubric.types import Criterion, CriterionOption

CONFIG = LLMConfig(model="openai/gpt-4.1-mini")

MULTI_CHOICE = Criterion(
    name="quality",
    weight=1.0,
    requirement="How good is it?",
    options=[CriterionOption(label=f"level {i}", value=i / 4) for i in range(5)],
    scale_type="ordinal",
)


def _grader(seed: Any, **kwargs: Any) -> CriterionGrader:
    return CriterionGrader(judge_model_config=CONFIG, seed=seed, **kwargs)


def _training_data() -> RubricDataset:
    dataset = RubricDataset(
        prompt="Q", rubric=Rubric([Criterion(name="a", weight=1.0, requirement="A")])
    )
    for i, verdict in enumerate([CriterionVerdict.MET, CriterionVerdict.UNMET] * 3):
        dataset.add_item(submission=f"s{i}", description="d", ground_truth=[verdict])
    return dataset


class TestRefused:
    @pytest.mark.parametrize(
        "seed", [True, False, 1.0, 42.0, "42", math.nan, np.bool_(True), np.float64(1.0)]
    )
    def test_a_seed_that_is_not_an_integer_or_none_is_refused(self, seed: Any) -> None:
        with pytest.raises(ValueError, match=r"seed must be None or an integer"):
            _grader(seed)

    def test_the_message_shows_the_value(self) -> None:
        with pytest.raises(ValueError, match=r"got True"):
            _grader(True)

    def test_the_seed_is_checked_before_any_client_is_built(self, monkeypatch) -> None:
        """A bad seed is a configuration error, reported before the judges' clients exist."""
        built: list[object] = []
        monkeypatch.setattr(
            "autorubric.graders.criterion_grader.LLMClient",
            lambda *args, **kwargs: built.append(args),
        )
        with pytest.raises(ValueError, match="seed"):
            _grader(1.0)
        assert built == []


class TestAccepted:
    @pytest.mark.parametrize("seed", [0, 1, 42, -7, 2**40])
    def test_any_integer_is_kept(self, seed: int) -> None:
        assert _grader(seed).seed == seed

    @pytest.mark.parametrize("seed", [np.int64(42), np.int32(42), np.uint8(42)])
    def test_a_numpy_integer_is_kept_as_a_plain_int(self, seed: Any) -> None:
        grader = _grader(seed)
        assert grader.seed == 42
        assert type(grader.seed) is int

    def test_none_draws_a_plain_int(self) -> None:
        assert type(_grader(None).seed) is int


class TestNumpySeedBehavesAsTheEqualInt:
    def test_it_shuffles_options_as_the_int_does(self) -> None:
        def order(seed: Any) -> list[int]:
            return _grader(seed)._presented_criterion(MULTI_CHOICE, 0, "default", "an answer")[1]

        assert order(np.int64(42)) == order(42)

    def test_it_draws_the_examples_the_int_draws(self) -> None:
        def examples(seed: Any) -> list:
            grader = _grader(
                seed,
                training_data=_training_data(),
                few_shot_config=FewShotConfig(n_examples=4, balance_verdicts=False),
            )
            return grader._criterion_examples[(0, "default")]

        assert examples(np.int64(42)) == examples(42)

    def test_it_coordinates_an_unset_few_shot_seed_as_a_plain_int(self) -> None:
        grader = _grader(np.int64(42), training_data=_training_data())
        assert grader._few_shot_config.seed == 42
        assert type(grader._few_shot_config.seed) is int

    def test_the_manifest_records_it(self) -> None:
        assert _serialize_grader_config(_grader(np.int64(42)))["master_seed"] == 42
        assert _serialize_grader_config(_grader(42))["master_seed"] == 42
