"""Common fixtures for autorubric tests."""

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autorubric import Criterion, CriterionVerdict, Rubric, TokenUsage
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import CriterionJudgment

CriterionList = list[Criterion]


def pytest_configure(config: pytest.Config) -> None:
    """Make every ``DeprecationWarning`` attributed to an ``autorubric`` module an error.

    The library must never trigger its own deprecations (it calls ``CriterionGrader`` with
    ``judge_model_config=``, never the deprecated ``llm_config=``), so this guards every
    path the suite runs. In an ini filter the module field is a regular expression matched
    at the start of the module name, so it covers all ``autorubric.*`` modules. A caller's
    own deprecated call is unaffected: ``CriterionGrader.__init__`` warns with
    ``stacklevel=2``, which attributes the warning to the caller (a test module), and
    ``pytest.warns`` still captures it. A stack level that stopped inside the library would
    attribute it to ``autorubric``, and this guard would turn every caller's deprecated call
    into an error. pytest applies a run's own ``-W``
    options after ini filters, so they take precedence (``-W always::DeprecationWarning``
    lifts the guard for that run); ``TestSuiteWideDeprecationGuard`` in
    ``tests/graders/test_judge_model_config.py`` checks this configuration independently of
    the run's filters.
    """
    config.addinivalue_line("filterwarnings", "error::DeprecationWarning:autorubric")


class _ExperimentsWatch:
    """The entries of the working directory's ``experiments/`` the suite has seen.

    ``EvalRunner`` experiments (``experiments_dir="experiments"``) and the improvement loop's
    artifacts (``ImprovementConfig.save_artifacts`` with no ``artifacts_dir``) default to
    ``./experiments``, which in a checkout is the repository's own (gitignored) directory,
    where real experiment runs live. A test must write to ``tmp_path`` instead. The
    directory is listed only when its modification time changed, so the check costs one
    ``stat`` per test.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime = self._stat_mtime()
        self._entries = self._list()

    def _stat_mtime(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _list(self) -> frozenset[str]:
        try:
            return frozenset(os.listdir(self.path))
        except FileNotFoundError:
            return frozenset()

    def new_entries(self) -> list[str]:
        """The entries created since the last call (or since the watch began)."""
        mtime = self._stat_mtime()
        if mtime == self._mtime:
            return []
        entries = self._list()
        created = sorted(entries - self._entries)
        self._mtime, self._entries = mtime, entries
        return created


@pytest.fixture(scope="session")
def _experiments_watch() -> _ExperimentsWatch:
    return _ExperimentsWatch(Path.cwd() / "experiments")


@pytest.fixture(autouse=True)
def _no_new_experiments_in_the_working_directory(
    request: pytest.FixtureRequest, _experiments_watch: _ExperimentsWatch
) -> Iterator[None]:
    """Fail a test that creates anything under the working directory's ``experiments/``.

    Tests run one at a time, so whatever appears there during a test is the test's. The
    created entries are reported, not removed: the directory also holds real runs.
    """
    yield
    created = _experiments_watch.new_entries()
    if created:
        pytest.fail(
            f"{request.node.nodeid} created {', '.join(created)} in {_experiments_watch.path}; "
            "write to tmp_path instead (e.g. ImprovementConfig(artifacts_dir=tmp_path) or "
            "evaluate(..., experiments_dir=tmp_path))",
            pytrace=False,
        )


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create a mock LLMConfig for testing."""
    return LLMConfig(model="test-model")


@pytest.fixture
def sample_output() -> str:
    return "Paris is the capital of France. It is a beautiful city with rich history."


@pytest.fixture
def sample_criteria() -> CriterionList:
    return [
        Criterion(
            weight=2.0,
            requirement="Output mentions Paris",
        ),
        Criterion(
            weight=1.0,
            requirement="Output mentions France",
        ),
        Criterion(
            weight=1.0,
            requirement="Output is written in complete sentences",
        ),
        Criterion(
            weight=-0.5,
            requirement="Output contains profanity or offensive language",
        ),
    ]


@pytest.fixture
def sample_rubric(sample_criteria: CriterionList) -> Rubric:
    return Rubric(sample_criteria)


def _extract_field(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        raise ValueError("Expected field not found in prompt")
    return match.group(1).strip()


def create_per_criterion_mock_client(sample_criteria: CriterionList | None = None) -> MagicMock:
    """Create a mock LLMClient for CriterionGrader tests.

    This mock returns GenerateResult objects with CriterionJudgment in the parsed field.
    """
    criterion_pattern = re.compile(r"<criterion>(.*?)</criterion>", re.DOTALL)
    type_pattern = re.compile(r"<criterion_type>(.*?)</criterion_type>", re.DOTALL)
    positive_requirements_met = {
        "Output mentions Paris",
        "Output mentions France",
        "Output is written in complete sentences",
    }
    negative_errors_present = {
        "Output contains profanity or offensive language": False,  # Error NOT present
    }

    async def mock_generate(
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        return_result: bool = False,
        **kwargs: Any,
    ) -> GenerateResult | CriterionJudgment:
        criterion_text = _extract_field(criterion_pattern, user_prompt)
        criterion_type = _extract_field(type_pattern, user_prompt).lower()

        if criterion_type == "negative":
            # For negative criteria: criterion_status=MET means error IS present (bad)
            # criterion_status=UNMET means error is NOT present (good)
            error_present = negative_errors_present.get(criterion_text, False)
            explanation = (
                "Error detected in the output."
                if error_present
                else "Error not present in the output."
            )
            judgment = CriterionJudgment(
                criterion_status=(
                    CriterionVerdict.MET if error_present else CriterionVerdict.UNMET
                ),
                explanation=explanation,
            )
        else:
            criteria_met = criterion_text in positive_requirements_met
            explanation = (
                "Requirement satisfied by the submission."
                if criteria_met
                else "Requirement not satisfied by the submission."
            )
            judgment = CriterionJudgment(
                criterion_status=(CriterionVerdict.MET if criteria_met else CriterionVerdict.UNMET),
                explanation=explanation,
            )

        # Return GenerateResult when return_result=True (as the grader now uses)
        if return_result:
            return GenerateResult(
                content="{}",  # JSON content (not used when parsed is set)
                thinking=None,
                raw_response=None,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                cost=0.001,
                parsed=judgment,
            )
        return judgment

    mock_client = MagicMock()
    mock_client.generate = AsyncMock(side_effect=mock_generate)
    return mock_client


@pytest.fixture
def per_criterion_mock_client(sample_criteria: CriterionList) -> MagicMock:
    """Mock LLMClient for CriterionGrader tests."""
    return create_per_criterion_mock_client(sample_criteria)
