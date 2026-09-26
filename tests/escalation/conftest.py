"""Fixtures for the cascade replay and diagnostics tests.

The decision-model tests' guards apply here too: no ``httpx2`` transport may send, the
``TYPESAFE_*`` variables are cleared, and the rate-limit pool starts fresh for each test.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typesafe_sdk

# Autouse guards, registered for this directory by importing them.
from decision.conftest import (  # noqa: F401
    _fresh_rate_limit_pool,
    _isolate_typesafe_env,
    _no_network,
)

from autorubric import EscalationConfig, EvalResult, LLMConfig, RubricDataset, evaluate
from autorubric.graders import CriterionGrader, JudgeSpec
from escalation.cascade_runs import (
    GEMINI,
    PER_CRITERION,
    REQUEST_ERROR,
    SEED,
    THRESHOLD,
    Runs,
    ScriptedLLM,
    ScriptedSDK,
    ScriptedSDKClient,
    dataset,
    decision_model_scripts,
    dm,
    llm_scripts,
)


@pytest.fixture
def decision_model(monkeypatch: pytest.MonkeyPatch) -> ScriptedSDK:
    """The TypeSafe SDK client, answering from ``ScriptedSDK.scripts`` by submission."""
    sdk = ScriptedSDK()
    monkeypatch.setattr(
        typesafe_sdk, "AsyncTypeSafeClient", lambda **kwargs: ScriptedSDKClient(sdk, **kwargs)
    )
    return sdk


@pytest.fixture
def make_grader(decision_model: ScriptedSDK) -> Iterator[Any]:
    """Build ``CriterionGrader``s whose LLM judges are ``ScriptedLLM``s.

    ``llm_script`` (``(submission, requirement) -> answer``) is shared by every LLM judge.
    Decision-model response caches are closed at teardown.
    """
    built: list[CriterionGrader] = []

    def build(*, llm_script: dict[tuple[str, str], Any] | None = None, **kwargs: Any) -> Any:
        def client(config: LLMConfig) -> ScriptedLLM:
            return ScriptedLLM(config.model, llm_script or {})

        with patch("autorubric.graders.criterion_grader.LLMClient", side_effect=client):
            grader = CriterionGrader(**kwargs)
        built.append(grader)
        return grader

    yield build
    for grader in built:
        for client in grader._decision_clients.values():
            client.close()


@pytest.fixture
def cascade_runs(
    make_grader: Any, decision_model: ScriptedSDK, tmp_path: Path
) -> Callable[..., Any]:
    """Grade one dataset three times: a live cascade, its decision model alone, and its
    escalation judges alone.

    The decision model answers from ``decision_model_scripts`` (``decision_model.scripts``
    can be changed before a run), the LLM judges from ``llm_script`` (by default
    ``llm_scripts()``) and their prompts. ``judges`` are the escalation judges and the LLM
    run's judges; ``settings`` go to the cascade and to the LLM run (aggregation, scoring).
    The cascade uses ``SEED``, the LLM run ``llm_seed``.
    """
    decision_model.scripts = decision_model_scripts(REQUEST_ERROR)
    names = (f"run-{n}" for n in itertools.count())

    async def run(grader: CriterionGrader, data: RubricDataset) -> EvalResult:
        return await evaluate(
            data, grader, show_progress=False, experiments_dir=tmp_path, experiment_name=next(names)
        )

    async def build(
        judges: list[JudgeSpec] = GEMINI,
        *,
        data: RubricDataset | None = None,
        dm_config: Any = None,
        threshold: float = THRESHOLD,
        per_criterion: dict[str, float] | None = PER_CRITERION,
        llm_script: dict[tuple[str, str], Any] | None = None,
        llm_seed: int = SEED,
        **settings: Any,
    ) -> Runs:
        data = data if data is not None else dataset()
        config = dm_config if dm_config is not None else dm()
        script = llm_script if llm_script is not None else llm_scripts()
        escalation = EscalationConfig(judges, threshold, per_criterion=per_criterion)
        cascade = make_grader(
            llm_script=script,
            judge_model_config=config,
            escalation=escalation,
            seed=SEED,
            **settings,
        )
        live = await run(cascade, data)
        dm_run = await run(make_grader(judge_model_config=config), data)
        llm_grader = make_grader(llm_script=script, judges=list(judges), seed=llm_seed, **settings)
        llm_run = await run(llm_grader, data)
        return Runs(data, live, dm_run, llm_run, cascade, llm_grader)

    return build
