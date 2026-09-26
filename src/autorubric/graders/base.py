"""Shared abstractions for grader implementations."""

import asyncio
import inspect
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, cast

from autorubric.types import (
    Criterion,
    EnsembleEvaluationReport,
    EvaluationReport,
    LengthPenalty,
    ToGradeInput,
)
from autorubric.utils import (
    _normalize_guidelines,
    compute_length_penalty,
    normalize_to_grade_input,
)

# Directories of the code a warning about grader use is never attributed to, each with a
# trailing separator: the autorubric package, and asyncio, whose event loop runs the tasks
# ``evaluate`` grades items in. Code objects record the path their module was loaded from,
# the same path as ``__file__``, so a frame runs such code exactly when its file name starts
# with one of these prefixes; both sides are compared through ``os.path.normcase``, so case
# and separator spellings (Windows) cannot hide a match.
_INTERNAL_DIRS = tuple(
    os.path.normcase(os.path.dirname(os.path.abspath(path))) + os.sep
    for path in (os.path.dirname(__file__), inspect.getfile(asyncio))
)


def _caller_stacklevel() -> int:
    """The ``stacklevel`` naming the code that asked for grading, for a warning issued by
    this function's caller.

    A warning about how a grader is used belongs to the code that uses it, which may call
    ``Grader.grade`` directly or through ``Rubric.grade``, ``evaluate`` or the improvement
    loop, so no fixed ``stacklevel`` names it. The level returned names the first frame
    outside autorubric and asyncio: the line that called autorubric, or, for items graded in
    tasks (``evaluate``), the line that started the event loop (e.g. ``asyncio.run(...)``).
    This is what ``warnings.warn``'s ``skip_file_prefixes`` does from Python 3.12;
    autorubric supports 3.11.
    """
    frame = sys._getframe(1)  # the caller, which ``stacklevel=1`` names
    level = 1
    while frame is not None and os.path.normcase(frame.f_code.co_filename).startswith(
        _INTERNAL_DIRS
    ):
        frame = frame.f_back
        level += 1
    return level


def _accepts_guidelines(target: Callable[..., Any], *, through_kwargs: bool = True) -> bool:
    """Whether ``target`` can be passed ``guidelines=``: it has a ``guidelines`` parameter
    that can be passed by keyword or, when ``through_kwargs`` is true (the default), takes
    ``**kwargs``. ``target`` is a grader method or a ``Rubric`` class (whose signature is
    its ``__init__``'s). A callable whose signature cannot be read is taken not to accept
    it."""
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (through_kwargs and parameter.kind is inspect.Parameter.VAR_KEYWORD)
        or (
            parameter.name == "guidelines"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for parameter in parameters
    )


def _instance_dict(grader: object) -> dict[str, Any] | None:
    """The grader's own writable attribute dict, or ``None`` when it has none."""
    instance_dict = getattr(grader, "__dict__", None)
    return instance_dict if isinstance(instance_dict, dict) else None


def _takes_guidelines(grader: object, method_name: str) -> bool:
    """Whether ``grader``'s method ``method_name`` accepts ``guidelines=``.

    The signature is inspected once per grader instance and cached on the instance, not on
    the class: the method is looked up through the instance, where an instance attribute
    (a wrapper, a test double) can replace the class's, and the "ignores rubric guidelines"
    warning is per instance too. The cache entry remembers the implementation it describes,
    so a method replaced later is inspected again. It lives in the instance ``__dict__``,
    which needs no ``Grader.__init__`` call and no attribute assignment a subclass could
    intercept; an object without one is inspected on every call.
    """
    method = getattr(grader, method_name)
    implementation = getattr(method, "__func__", method)
    instance_dict = _instance_dict(grader)
    cache: dict[str, tuple[object, bool]] = (
        instance_dict.setdefault("_guidelines_support", {}) if instance_dict is not None else {}
    )
    cached = cache.get(method_name)
    if cached is not None and cached[0] is implementation:
        return cached[1]
    accepts = _accepts_guidelines(method)
    cache[method_name] = (implementation, accepts)
    return accepts


def _warn_ignores_guidelines(grader: object) -> None:
    """Warn, once per grader instance, that ``grader`` grades without the rubric guidelines.

    The warning is a ``UserWarning`` attributed to the code that asked for grading
    (``_caller_stacklevel``). Whether an instance has warned is kept in its ``__dict__``,
    like ``_takes_guidelines``' cache; an object without one warns on every call. The flag
    is set only once ``warnings.warn`` returns: under a warnings-as-errors filter the
    warning raises on every call, as Python's own warning registry makes it, instead of
    raising once and then letting the instance grade without the guidelines unannounced.
    """
    instance_dict = _instance_dict(grader)
    if instance_dict is not None and instance_dict.get("_guidelines_warned"):
        return
    warnings.warn(
        f"{type(grader).__name__} ignores rubric guidelines",
        UserWarning,
        stacklevel=_caller_stacklevel(),
    )
    if instance_dict is not None:
        instance_dict["_guidelines_warned"] = True


class Grader(ABC):
    """Base class for LLM-backed grading implementations.

    All graders require an LLMConfig for the LLM client. Subclasses must
    implement judge() and aggregate() methods.

    Rubric guidelines reach ``judge`` only if it accepts them: a ``judge`` with a
    ``guidelines`` parameter that can be passed by keyword (e.g. keyword-only
    ``guidelines: str | None = None``) or with ``**kwargs`` receives them as
    ``guidelines=``. A ``judge`` with the abstract signature keeps working unchanged; the
    grader then grades without the guidelines and warns once (see ``grade``).

    Args:
        length_penalty: Optional configuration for penalizing overly long outputs.
            When provided, a penalty based on the token/word count is subtracted
            from the final score.
        normalize: If True (default), scores are normalized to 0-1. If False, raw
            weighted sums are returned, which is useful for RL training scenarios.
    """

    def __init__(
        self,
        *,
        length_penalty: LengthPenalty | None = None,
        normalize: bool = True,
    ):
        self.length_penalty: LengthPenalty | None = length_penalty
        self.normalize: bool = normalize

    @abstractmethod
    async def judge(
        self,
        to_grade: str,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
    ) -> Any:
        """Collect raw judge results for the provided submission.

        An implementation may also take the rubric's guidelines, as a ``guidelines``
        parameter that can be passed by keyword or through ``**kwargs``; ``grade`` passes
        them, when there are any, only to an implementation that does.

        Args:
            to_grade: The text to evaluate.
            rubric: List of criteria to evaluate against.
            query: Optional input/query that prompted the response.
            reference_submission: Optional exemplar response for grading context.

        Returns:
            Raw judge results (format depends on implementation).
        """
        pass

    @abstractmethod
    async def aggregate(self, judge_results: Any, *, normalize: bool = True) -> EvaluationReport:
        """Transform judge results into an EvaluationReport.

        Args:
            judge_results: Raw results from judge().
            normalize: If True, normalize score to 0-1. If False, return raw weighted sum.

        Returns:
            EvaluationReport with score and optional per-criterion breakdown.
        """
        pass

    async def grade(
        self,
        to_grade: ToGradeInput,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
        *,
        guidelines: str | None = None,
    ) -> EvaluationReport:
        """Grade the submission against the rubric.

        This is the main entry point for the grader.

        Args:
            to_grade: The text to evaluate. Can be either:
                - A string (optionally with <thinking>/<output> markers)
                - A dict with 'thinking' and 'output' keys
            rubric: List of criteria to evaluate against.
            query: Optional input/query that prompted the response.
            reference_submission: Optional exemplar response for grading context.
            guidelines: Optional rubric guidelines: free text that applies to every
                criterion (``Rubric.grade`` passes ``Rubric.guidelines``). Blank text means
                none. They are passed to ``judge`` as ``guidelines=`` only if ``judge``
                accepts that keyword (a ``guidelines`` parameter or ``**kwargs``; checked
                once per grader instance). Otherwise, and whenever there are no guidelines,
                ``judge`` is called exactly as without them:
                ``judge(to_grade, rubric, query, reference_submission)``.

        Returns:
            EvaluationReport with score and optional per-criterion breakdown.
            If normalize=True (default), score is 0-1. If normalize=False, score is raw
            weighted sum. If length_penalty was configured, the penalty is subtracted from
            the score. The raw_score field contains the unnormalized weighted sum before
            length penalty.

        Raises:
            TypeError: If ``guidelines`` is neither a ``str`` nor ``None``.

        Warns:
            UserWarning: "<GraderClass> ignores rubric guidelines", once per grader
                instance, when guidelines are given and ``judge`` does not accept them.
        """
        guidelines = _normalize_guidelines(guidelines)
        if guidelines is not None and not _takes_guidelines(self, "judge"):
            _warn_ignores_guidelines(self)
            guidelines = None

        # Convert to_grade to string for judge() call (maintains compatibility)
        if isinstance(to_grade, str):
            to_grade_str = to_grade
        else:
            # Dict format - reconstruct string with markers for judge()
            thinking = to_grade.get("thinking", "")
            output = to_grade.get("output", "")
            parts = []
            if thinking:
                parts.append(f"<thinking>{thinking}</thinking>")
            if output:
                parts.append(f"<output>{output}</output>")
            to_grade_str = "\n".join(parts) if parts else ""

        # Call judge with string format (maintains compatibility). Guidelines go as a
        # keyword, and only to a judge that accepts it (checked above); the abstract
        # signature does not declare it, hence the cast.
        judge = cast(Callable[..., Awaitable[Any]], self.judge)
        if guidelines is None:
            judge_results = await judge(to_grade_str, rubric, query, reference_submission)
        else:
            judge_results = await judge(
                to_grade_str, rubric, query, reference_submission, guidelines=guidelines
            )
        report = await self.aggregate(judge_results, normalize=self.normalize)

        if self.length_penalty is not None:
            # A grade-FAILURE has no score (errored/empty report): there is nothing to
            # penalize, so return it unchanged rather than subtracting from None.
            if report.score is None:
                return report

            # Normalize to_grade to dict format for penalty calculation
            to_grade_normalized = normalize_to_grade_input(to_grade)

            # Compute penalty
            penalty = compute_length_penalty(to_grade_normalized, self.length_penalty)

            # Apply penalty (penalty is always non-negative, so we subtract)
            adjusted_score = report.score - penalty
            if self.normalize:
                adjusted_score = max(0.0, adjusted_score)

            # Return the same report type with adjusted score
            if isinstance(report, EnsembleEvaluationReport):
                return EnsembleEvaluationReport(
                    score=adjusted_score,
                    raw_score=report.raw_score,
                    llm_raw_score=report.llm_raw_score,
                    report=report.report,
                    judge_scores=report.judge_scores,
                    mean_agreement=report.mean_agreement,
                    cannot_assess_count=report.cannot_assess_count,
                    token_usage=report.token_usage,
                    completion_cost=report.completion_cost,
                    error=report.error,
                )
            else:
                return EvaluationReport(
                    score=adjusted_score,
                    raw_score=report.raw_score,
                    llm_raw_score=report.llm_raw_score,
                    report=report.report,
                    cannot_assess_count=report.cannot_assess_count,
                    error=report.error,
                    token_usage=report.token_usage,
                    completion_cost=report.completion_cost,
                )

        return report
