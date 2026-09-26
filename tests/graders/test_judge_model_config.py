"""``judge_model_config`` (canonical) and the deprecated ``llm_config`` alias.

Covers ``CriterionGrader(judge_model_config=...)`` with ``llm_config=`` as a deprecated
alias, the ``JudgeSpec`` keyword alias and read/write property, the unchanged ``JudgeSpec``
dataclass surface (checked against fixtures captured from the library before the alias
existed), and that the library never triggers its own deprecation warning.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import inspect
import json
import pickle
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The function pytest itself uses to apply ini ``filterwarnings`` lines, reused so the
# guard tests apply the suite's configured filters exactly as pytest does.
from _pytest.config import apply_warning_filters

import autorubric
from autorubric import Criterion, CriterionOption, CriterionVerdict, DataItem, Rubric, RubricDataset
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.meta import (
    ImprovementConfig,
    ImprovementRunner,
    evaluate_rubric_in_context,
    evaluate_rubric_standalone,
)
from autorubric.types import (
    CriterionJudgment,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    JudgeVote,
    MultiChoiceJudgment,
    TokenUsage,
)

LEGACY_DIR = Path(__file__).resolve().parents[1] / "golden" / "legacy"
DEPRECATION_MESSAGE = "llm_config is deprecated; use judge_model_config"
PACKAGE_DIR = Path(autorubric.__file__).resolve().parent
SUITE_DEPRECATION_GUARD = "error::DeprecationWarning:autorubric"


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(model="test-model")


def _deprecations(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def _raised_inside_library(w: warnings.WarningMessage) -> bool:
    return Path(w.filename).resolve().is_relative_to(PACKAGE_DIR)


# ---------------------------------------------------------------------------
# CriterionGrader
# ---------------------------------------------------------------------------


class TestCriterionGraderKeyword:
    def test_judge_model_config_builds_the_default_single_judge(self, config):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grader = CriterionGrader(judge_model_config=config)
        assert _deprecations(caught) == []
        assert grader._judges == [JudgeSpec(config, "default", 1.0)]
        assert grader._judges[0].llm_config is config
        assert not grader.is_ensemble

    def test_llm_config_warns_at_the_callers_line(self, config):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = inspect.currentframe()
            assert frame is not None
            call_line = frame.f_lineno + 1
            grader = CriterionGrader(llm_config=config)
        deprecations = _deprecations(caught)
        assert len(deprecations) == 1
        warning = deprecations[0]
        assert warning.category is DeprecationWarning
        assert str(warning.message) == DEPRECATION_MESSAGE
        # The stack level skips the library's frames: attributed to this call, not the library.
        assert Path(warning.filename).resolve() == Path(__file__).resolve()
        assert warning.lineno == call_line
        assert grader._judges == [JudgeSpec(config, "default", 1.0)]

    def test_llm_config_is_reported_by_pytest_warns(self, config):
        with pytest.warns(DeprecationWarning, match="^llm_config is deprecated; use judge"):
            CriterionGrader(llm_config=config)

    def test_llm_config_builds_the_same_grader_state(self, config):
        new = CriterionGrader(judge_model_config=config, seed=11)
        with pytest.warns(DeprecationWarning):
            old = CriterionGrader(llm_config=config, seed=11)
        assert old._judges == new._judges
        assert old.seed == new.seed
        assert old._system_prompt == new._system_prompt
        assert old._multi_choice_system_prompt == new._multi_choice_system_prompt
        assert list(old._clients) == list(new._clients) == ["default"]

    @pytest.mark.asyncio
    async def test_llm_config_grades_identically(self, config):
        """Both spellings produce the same report from the same judge outputs."""
        rubric = Rubric(
            [
                Criterion(name="a", weight=2.0, requirement="Mentions Paris"),
                Criterion(name="b", weight=-1.0, requirement="Contains profanity"),
                Criterion(
                    name="c",
                    weight=1.0,
                    requirement="How clear?",
                    scale_type="ordinal",
                    options=[
                        CriterionOption(label="Unclear", value=0.0),
                        CriterionOption(label="Clear", value=1.0),
                    ],
                ),
            ]
        )

        async def generate(system_prompt, user_prompt, response_format=None, **kwargs):
            if issubclass(response_format, MultiChoiceJudgment):
                parsed: Any = MultiChoiceJudgment(selected_option=1, explanation="mc")
            else:
                status = (
                    CriterionVerdict.MET
                    if "Mentions Paris" in user_prompt
                    else CriterionVerdict.UNMET
                )
                parsed = CriterionJudgment(criterion_status=status, explanation="bin")
            return GenerateResult(
                content="{}",
                usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                cost=0.01,
                parsed=parsed,
            )

        client = MagicMock()
        client.generate = AsyncMock(side_effect=generate)
        with patch("autorubric.graders.criterion_grader.LLMClient", return_value=client):
            new = CriterionGrader(judge_model_config=config, seed=3)
            with pytest.warns(DeprecationWarning):
                old = CriterionGrader(llm_config=config, seed=3)
            new_report = await rubric.grade("Paris is lovely.", grader=new, query="q")
            old_report = await rubric.grade("Paris is lovely.", grader=old, query="q")
        assert old_report.model_dump() == new_report.model_dump()
        assert new_report.score is not None

    def test_both_spellings_raise_without_warning(self, config):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="judge_model_config") as excinfo:
                CriterionGrader(judge_model_config=config, llm_config=LLMConfig(model="other"))
        assert "llm_config" in str(excinfo.value)
        assert _deprecations(caught) == []

    def test_both_spellings_raise_even_for_the_same_object(self, config):
        with pytest.raises(ValueError, match="Pass only one of judge_model_config and llm_config"):
            CriterionGrader(judge_model_config=config, llm_config=config)

    def test_neither_single_config_nor_judges_names_new_keyword(self):
        with pytest.raises(ValueError, match="^Must provide either judge_model_config or judges$"):
            CriterionGrader()

    def test_single_config_and_judges_names_new_keyword(self, config):
        judges = [JudgeSpec(config, "a")]
        with pytest.raises(ValueError, match="^Cannot provide both judge_model_config and judges$"):
            CriterionGrader(judge_model_config=config, judges=judges)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(
                ValueError, match="^Cannot provide both judge_model_config and judges$"
            ):
                CriterionGrader(llm_config=config, judges=judges)

    def test_signature_is_keyword_only_with_alias_last(self):
        params = list(inspect.signature(CriterionGrader.__init__).parameters.values())[1:]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)
        assert params[0].name == "judge_model_config"
        assert params[1].name == "judges"
        assert params[-1].name == "llm_config"
        assert params[0].default is None and params[-1].default is None


# ---------------------------------------------------------------------------
# JudgeSpec
# ---------------------------------------------------------------------------


def _legacy_judge_spec_fixture() -> dict[str, Any]:
    return json.loads((LEGACY_DIR / "judge_spec.json").read_text(encoding="utf-8"))


class TestJudgeSpecAlias:
    def test_three_spellings_build_equal_objects_without_warning(self, config):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            positional = JudgeSpec(config, "gemini", 2.0)
            stored_name = JudgeSpec(llm_config=config, judge_id="gemini", weight=2.0)
            canonical = JudgeSpec(judge_model_config=config, judge_id="gemini", weight=2.0)
            mixed = JudgeSpec(config, judge_id="gemini", weight=2.0)
        assert caught == []
        assert positional == stored_name == canonical == mixed
        for spec in (positional, stored_name, canonical, mixed):
            assert spec.llm_config is config
            assert spec.judge_id == "gemini"
            assert spec.weight == 2.0

    def test_weight_defaults_to_one(self, config):
        assert JudgeSpec(config, "a").weight == 1.0
        assert JudgeSpec(judge_model_config=config, judge_id="a").weight == 1.0

    def test_both_spellings_raise(self, config):
        with pytest.raises(ValueError, match="judge_model_config"):
            JudgeSpec(config, "a", judge_model_config=config)
        with pytest.raises(ValueError, match="judge_model_config"):
            JudgeSpec(llm_config=config, judge_model_config=config, judge_id="a")

    def test_missing_config_is_a_type_error(self):
        with pytest.raises(TypeError, match="missing 1 required argument: 'judge_model_config'"):
            JudgeSpec(judge_id="a")

    def test_missing_judge_id_is_a_type_error(self, config):
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'judge_id'"):
            JudgeSpec(config)
        with pytest.raises(TypeError, match="'judge_id'"):
            JudgeSpec(judge_model_config=config)

    def test_missing_everything_names_both_arguments(self):
        with pytest.raises(TypeError, match="missing 2 required arguments") as excinfo:
            JudgeSpec()
        assert "'judge_model_config'" in str(excinfo.value)
        assert "'judge_id'" in str(excinfo.value)

    def test_explicit_none_is_passed_through_as_before(self):
        """Only a *missing* argument is an error; values are stored unvalidated, as the
        generated dataclass ``__init__`` always did."""
        spec = JudgeSpec(None, None)  # the generated __init__ stored these too
        assert spec.llm_config is None and spec.judge_id is None

    def test_unknown_keyword_and_extra_positional_are_type_errors(self, config):
        with pytest.raises(TypeError):
            JudgeSpec(config, "a", 1.0, "extra")
        with pytest.raises(TypeError):
            JudgeSpec(config, "a", model_config=config)

    def test_property_reads_and_writes_the_stored_field(self, config):
        spec = JudgeSpec(config, "a")
        assert spec.judge_model_config is config
        other = LLMConfig(model="other-model")
        spec.judge_model_config = other
        assert spec.llm_config is other
        third = LLMConfig(model="third-model")
        spec.llm_config = third
        assert spec.judge_model_config is third
        assert "judge_model_config" not in vars(spec)
        assert set(vars(spec)) == {"llm_config", "judge_id", "weight"}


class TestJudgeSpecDataclassSurfaceUnchanged:
    """The dataclass surface is identical to the pre-alias library (captured fixture)."""

    def test_fields_defaults_and_match_args(self):
        fixture = _legacy_judge_spec_fixture()
        fields = dataclasses.fields(JudgeSpec)
        assert [f.name for f in fields] == fixture["field_names"]
        defaults = {
            f.name: (None if f.default is dataclasses.MISSING else f.default) for f in fields
        }
        assert defaults == fixture["field_defaults"]
        assert list(JudgeSpec.__match_args__) == fixture["match_args"]
        assert (JudgeSpec.__hash__ is None) is fixture["hash_is_none"]

    def test_repr_asdict_eq_and_replace(self, config):
        fixture = _legacy_judge_spec_fixture()
        stub = JudgeSpec("stub-config", "judge-a", 0.5)
        assert repr(stub) == fixture["stub_repr"]
        assert dataclasses.asdict(stub) == fixture["stub_asdict"]
        assert (stub == JudgeSpec("stub-config", "judge-a", 0.5)) is fixture[
            "stub_equals_positional_copy"
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            replaced = dataclasses.replace(stub, llm_config="other-config", weight=1.5)
        assert caught == []
        assert repr(replaced) == fixture["replaced_stub_repr"]
        assert list(dataclasses.asdict(JudgeSpec(config, "g"))) == fixture["asdict_keys"]

    def test_replace_round_trip(self, config):
        spec = JudgeSpec(config, "gemini", 2.0)
        assert dataclasses.replace(spec) == spec
        other = LLMConfig(model="other-model")
        assert dataclasses.replace(spec, llm_config=other).judge_model_config is other

    def test_pickle_and_copy_round_trip(self, config):
        spec = JudgeSpec(config, "gemini", 2.0)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.loads(pickle.dumps(spec, protocol=protocol)) == spec
        assert copy.copy(spec) == spec
        assert copy.deepcopy(spec) == spec

    def test_unpickles_a_judge_spec_pickled_before_the_alias(self):
        fixture = _legacy_judge_spec_fixture()
        spec = pickle.loads(base64.b64decode(fixture["pickle_b64"]))
        assert type(spec) is JudgeSpec
        assert set(vars(spec)) == {"llm_config", "judge_id", "weight"}
        assert isinstance(spec.llm_config, LLMConfig)
        assert spec.llm_config.model == "golden-model"
        assert spec.llm_config.max_parallel_requests == 3
        assert spec.judge_id == "gemini"
        assert spec.weight == 2.0
        assert spec.judge_model_config is spec.llm_config


# ---------------------------------------------------------------------------
# The library never triggers its own deprecation
# ---------------------------------------------------------------------------


def _meta_judgment_client() -> MagicMock:
    """A mock LLMClient that answers any (meta-)judgment schema."""

    async def generate(system_prompt, user_prompt, response_format=None, **kwargs):
        if issubclass(response_format, MultiChoiceJudgment):
            parsed: Any = response_format(selected_option=1, explanation="ok")
        else:
            parsed = response_format(criterion_status=CriterionVerdict.MET, explanation="ok")
        return GenerateResult(content="{}", parsed=parsed)

    client = MagicMock()
    client.generate = AsyncMock(side_effect=generate)
    return client


def _clean_report(names: list[str]) -> EnsembleEvaluationReport:
    reports = [
        EnsembleCriterionReport(
            criterion=Criterion(name=name, weight=1.0, requirement=f"req {name}"),
            final_verdict=CriterionVerdict.MET,
            final_reason="fine",
            votes=[JudgeVote(judge_id="j", verdict=CriterionVerdict.MET, reason="fine")],
        )
        for name in names
    ]
    return EnsembleEvaluationReport(score=1.0, raw_score=2.0, report=reports, mean_agreement=1.0)


def _labelled_dataset() -> RubricDataset:
    rubric = Rubric(
        [
            Criterion(name="intro", weight=1.0, requirement="Has an intro"),
            Criterion(name="conclusion", weight=1.0, requirement="Has a conclusion"),
        ]
    )
    items = [
        DataItem(
            submission=f"essay {i}",
            description=f"item {i}",
            ground_truth=[CriterionVerdict.MET, CriterionVerdict.MET],
        )
        for i in range(2)
    ]
    return RubricDataset(prompt="Write an essay", rubric=rubric, items=items, name="ds")


class TestLibraryNeverTriggersItsOwnDeprecation:
    def _assert_no_internal_deprecation(self, caught: list[warnings.WarningMessage]) -> None:
        internal = [w for w in _deprecations(caught) if _raised_inside_library(w)]
        assert internal == [], [f"{w.filename}:{w.lineno}: {w.message}" for w in internal]

    def test_single_config_path(self, config):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CriterionGrader(judge_model_config=config)
        self._assert_no_internal_deprecation(caught)
        assert _deprecations(caught) == []

    def test_detector_attributes_a_callers_deprecation_to_the_caller(self, config):
        """Control: a deprecated call from *outside* the package is not flagged as internal,
        so an internal call (attributed by the stack level to its library call site) would be."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CriterionGrader(llm_config=config)
        assert len(_deprecations(caught)) == 1
        assert not _raised_inside_library(_deprecations(caught)[0])

    @pytest.mark.asyncio
    async def test_meta_rubric_evaluation(self, config):
        rubric = Rubric([Criterion(name="x", weight=1.0, requirement="Is clear")])
        with (
            patch(
                "autorubric.graders.criterion_grader.LLMClient",
                return_value=_meta_judgment_client(),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            standalone = await evaluate_rubric_standalone(rubric, config)
            in_context = await evaluate_rubric_in_context(rubric, "Write an essay", config)
        assert standalone.report and in_context.report
        self._assert_no_internal_deprecation(caught)

    @pytest.mark.asyncio
    async def test_improvement_loop_with_ground_truth_validation(self, config):
        dataset = _labelled_dataset()
        rubric = dataset.rubric
        assert rubric is not None
        improvement = ImprovementConfig(
            eval_llm=config,
            revision_llm=config,
            validation_data=dataset,
            max_iterations=1,
            save_artifacts=False,
            show_progress=False,
            display=None,
        )
        with (
            patch(
                "autorubric.meta._improve.evaluate_rubric_in_context",
                new_callable=AsyncMock,
                return_value=_clean_report(["q1"]),
            ),
            patch.object(
                rubric,
                "grade",
                new_callable=AsyncMock,
                return_value=_clean_report(["intro", "conclusion"]),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            result = await ImprovementRunner(rubric, "Write an essay", config=improvement).run()
        assert result.iterations
        self._assert_no_internal_deprecation(caught)

    @pytest.mark.asyncio
    async def test_held_out_improvement_loop(self, config):
        dataset = _labelled_dataset()
        rubric = dataset.rubric
        assert rubric is not None
        improvement = ImprovementConfig(
            eval_llm=config,
            revision_llm=config,
            strategy="held_out",
            validation_data=dataset,
            max_iterations=1,
            held_out_min_accuracy=0.0,
            save_artifacts=False,
            show_progress=False,
            display=None,
        )
        with (
            patch.object(
                rubric,
                "grade",
                new_callable=AsyncMock,
                return_value=_clean_report(["intro", "conclusion"]),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            result = await ImprovementRunner(rubric, "Write an essay", config=improvement).run()
        assert result.iterations
        self._assert_no_internal_deprecation(caught)


@contextlib.contextmanager
def _suite_configured_filters(
    pytestconfig: pytest.Config,
) -> Iterator[list[warnings.WarningMessage]]:
    """Record warnings under the suite's configured ini ``filterwarnings`` alone.

    pytest applies a run's own ``-W`` options on top of the ini filters, and they may
    deliberately change the outcome: ``-W error::DeprecationWarning`` makes a caller's
    deprecation an error as well, ``-W always::DeprecationWarning`` lifts the guard, and
    ``-p no:warnings`` applies no filters at all. Starting from ``always`` and applying only
    the configured ini filters, exactly as pytest applies them, checks the suite's own
    configuration whatever filters a particular run adds (``PYTHONWARNINGS`` included).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_warning_filters(pytestconfig.getini("filterwarnings"), [])
        yield caught


class TestSuiteWideDeprecationGuard:
    """``tests/conftest.py`` adds the ini filter ``error::DeprecationWarning:autorubric``, so
    a ``DeprecationWarning`` attributed to any ``autorubric`` module is an error on every
    path the suite exercises, not only the paths tested above. These tests check that
    configuration through ``_suite_configured_filters``, so their outcome does not depend
    on the warning filters of the run."""

    def test_guard_is_configured(self, pytestconfig):
        assert SUITE_DEPRECATION_GUARD in pytestconfig.getini("filterwarnings")

    @pytest.mark.parametrize("module", ["autorubric", "autorubric.graders.probe"])
    def test_deprecation_attributed_to_the_library_is_an_error(self, pytestconfig, module):
        with _suite_configured_filters(pytestconfig):
            with pytest.raises(DeprecationWarning, match="library probe"):
                warnings.warn_explicit(
                    "library probe", DeprecationWarning, "probe.py", 1, module=module
                )

    def test_deprecation_attributed_to_a_caller_only_warns(self, pytestconfig):
        with _suite_configured_filters(pytestconfig) as caught:
            warnings.warn_explicit(
                "caller probe", DeprecationWarning, "probe.py", 1, module=__name__
            )
        assert [str(w.message) for w in caught] == ["caller probe"]

    def test_callers_deprecated_keyword_only_warns(self, pytestconfig, config):
        """The stack level attributes the warning to this test module, not the library,
        so the guard leaves it a warning."""
        with _suite_configured_filters(pytestconfig) as caught:
            CriterionGrader(llm_config=config)
        deprecations = _deprecations(caught)
        assert [str(w.message) for w in deprecations] == [DEPRECATION_MESSAGE]
        assert Path(deprecations[0].filename).resolve() == Path(__file__).resolve()
