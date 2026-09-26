"""Unified criterion-based grader with compositional few-shot and ensemble support."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import itertools
import logging
import math
import random
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel

from autorubric.decision import (
    DecisionModelClient,
    DecisionModelConfig,
    _is_real_number,
    answer_to_report,
    build_questions,
    build_state,
)
from autorubric.graders.base import Grader, _caller_stacklevel
from autorubric.llm import GenerateResult, LLMClient, LLMConfig, classify_grading_error
from autorubric.prompts import (
    FEW_SHOT_SYSTEM_PROMPT_ADDITION,
    GRADER_SYSTEM_PROMPT_DEFAULT,
    MULTI_CHOICE_FEW_SHOT_ADDITION,
    MULTI_CHOICE_SYSTEM_PROMPT,
    build_few_shot_user_prompt,
    build_multi_choice_few_shot_user_prompt,
    build_multi_choice_user_prompt,
    build_user_prompt,
)
from autorubric.scoring import score_reports
from autorubric.types import (
    AggregatedMultiChoiceVerdict,
    AggregationStrategy,
    CannotAssessConfig,
    Criterion,
    CriterionJudgment,
    CriterionReport,
    CriterionVerdict,
    EnsembleCriterionReport,
    EnsembleEvaluationReport,
    FewShotConfig,
    FewShotExample,
    JudgeVote,
    LengthPenalty,
    MultiChoiceJudgeVote,
    MultiChoiceJudgment,
    MultiChoiceVerdict,
    NominalAggregation,
    OrdinalAggregation,
    TokenUsage,
    _binary_worst_verdict,
)
from autorubric.utils import _normalize_guidelines

if TYPE_CHECKING:
    from autorubric.dataset import DataItem, RubricDataset

logger = logging.getLogger(__name__)


def _derive_shuffle_rng(
    master_seed: int,
    item_key: str,
    criterion_idx: int,
    judge_id: str,
) -> random.Random:
    """Derive a deterministic, concurrency-safe RNG for option shuffling.

    Produces a unique random.Random instance for each combination of
    master seed, item content, criterion, and judge.
    """
    key = f"{master_seed}:{item_key}:{criterion_idx}:{judge_id}"
    derived = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**31)
    return random.Random(derived)


# Sentinel used in the item-key slot of ``_derive_shuffle_rng`` for few-shot example
# selection. Few-shot examples are a fixed property of (criterion, judge), not of the
# item being graded, so the per-call item content is intentionally not part of the key.
FEW_SHOT_DOMAIN = "few_shot"


def _ground_truth_reason(item: DataItem, criterion_idx: int) -> str | None:
    """A training item's written reason for its ground truth on a criterion, if it has one.

    The reason rides along with the few-shot example drawn from the item; it never
    affects which items are drawn.
    """
    reasons = item.ground_truth_reasons
    return reasons[criterion_idx] if reasons is not None else None


_ESCALATION_NAMES_CHECKED: ContextVar[bool] = ContextVar("_ESCALATION_NAMES_CHECKED", default=False)
"""True while a batch (``EvalRunner.run``, ``fill_ground_truth``) grades a dataset's items.
The batch checked the cascade's ``EscalationConfig.per_criterion`` names against the union
of its items' rubrics before grading (``_escalation_names_checked``), so
``CriterionGrader.judge`` skips its per-rubric check: with per-item rubrics a name absent
from one item's rubric may be in another's."""


def _escalation_names_checked(
    grader: Grader, dataset: RubricDataset
) -> contextlib.AbstractContextManager[None]:
    """Check a cascade's ``per_criterion`` names once for a batch grading ``dataset``'s items.

    Thresholds are looked up by criterion name, and with per-item rubrics a name absent from
    one item's rubric can be in another's, so a batch checks the names once, against every
    item's rubric (``get_item_rubric``), and warns about the names no item's rubric has.
    Wrap the batch's grading in ``with _escalation_names_checked(grader, dataset):``. The
    check runs when this is called, so that its warning names the code that asked for
    grading (a ``contextlib`` frame is not autorubric's); the context returned sets
    ``_ESCALATION_NAMES_CHECKED`` for the grading inside it, tasks started there included,
    so that ``CriterionGrader.judge`` skips its per-rubric check. A grader that is not a
    ``CriterionGrader`` is neither checked nor flagged.

    Args:
        grader: The batch's grader.
        dataset: The dataset whose items the batch grades.

    Returns:
        The context to grade the items in.
    """
    if not isinstance(grader, CriterionGrader):
        return contextlib.nullcontext()
    grader._check_escalation_names(
        (dataset.get_item_rubric(idx).rubric for idx in range(len(dataset))),
        "any item's rubric",
    )
    return _escalation_names_marked_checked()


@contextlib.contextmanager
def _escalation_names_marked_checked() -> Iterator[None]:
    """Set ``_ESCALATION_NAMES_CHECKED`` for the body of the ``with`` statement."""
    token = _ESCALATION_NAMES_CHECKED.set(True)
    try:
        yield
    finally:
        _ESCALATION_NAMES_CHECKED.reset(token)


def _combine_errors(errors: list[str | None]) -> str | None:
    """Combine per-judge error strings into an ensemble-level error.

    Returns a combined message only when *every* contributing judge errored (so the
    final verdict was driven entirely by failures). Returns None if any judge produced
    a genuine judgment, or if there were no contributing judges.
    """
    if not errors or any(e is None for e in errors):
        return None
    return " | ".join(e for e in errors if e is not None)


def _aggregate_error(votes: Sequence[JudgeVote | MultiChoiceJudgeVote]) -> str | None:
    """Ensemble error string for a list of votes (see ``_combine_errors``).

    Single source of truth for the ensemble-level ``error`` across both the binary
    (``JudgeVote``) and multi-choice (``MultiChoiceJudgeVote``) aggregation paths.
    """
    return _combine_errors([v.error for v in votes])


def _join_vote_reasons(votes: Sequence[JudgeVote | MultiChoiceJudgeVote]) -> str | None:
    """Ensemble ``final_reason`` for a list of votes: ``"judge_id: reason"`` per vote,
    joined with ``" | "``.

    Votes whose ``reason`` is ``None`` (the judge gives no explanation) are skipped by an
    identity check, not a truthiness check, so an LLM vote with an empty explanation still
    renders as ``"judge_id: "``. ``None`` when no vote carried a reason. Single source for
    both the binary and multi-choice aggregation paths.
    """
    reasons = [f"{v.judge_id}: {v.reason}" for v in votes if v.reason is not None]
    return " | ".join(reasons) if reasons else None


def _inject_affected_criteria(reason: str | None, judgment: object) -> str | None:
    """Append an [Affects: #i, #j] tag when the judgment carries non-empty affected_criteria.

    Shared by the binary and multi-choice success paths so the structured-output
    affected_criteria convention (used by meta-rubric evaluation) behaves identically
    for both criterion types.
    """
    affected = getattr(judgment, "affected_criteria", None)
    if affected:
        tag = ", ".join(f"#{i}" for i in affected)
        reason = f"{reason} [Affects: {tag}]"
    return reason


def _require_llm_reason(reason: str | None) -> str:
    """Return an LLM judgment's tagged reason, rejecting a null one.

    ``reason`` is the judgment's ``explanation`` after ``_inject_affected_criteria``. An
    LLM judge always explains its verdict, so its reason is a string, possibly empty;
    ``reason is None`` is reserved for judges that return probabilities instead of text.
    A null explanation (possible only with a custom response format that makes the field
    optional) is therefore a malformed judgment. It raises ``ValueError``, which
    ``classify_grading_error`` routes as ``parse``, exactly as the report's own validation
    did while ``reason`` was a plain ``str``. Like that validation, the check applies to
    the reason as stored, after tagging: a tagged null explanation is the string
    ``"None [Affects: ...]"`` and is accepted, as it always was.

    Shared by the binary and multi-choice success paths. Each calls it only after reading
    the judgment's required fields (its verdict field and ``explanation``), so a judgment
    that lacks one of them still fails on the missing field (an ``unknown`` error), as it
    did when the report's own validation made this check.

    Raises:
        ValueError: If ``reason`` is ``None``.
    """
    if reason is None:
        raise ValueError("LLM judgment has no explanation (explanation is null)")
    return reason


def _top_tied_keys(scores: Mapping[int, float]) -> list[int]:
    """Option indices tied for the maximum score (vote count or summed judge weight).

    Used to surface mode/weighted_mode tie candidates for resolution via
    ``Criterion.worst_option_among``. A clear winner yields a single-element list.
    """
    top = max(scores.values())
    return [i for i, s in scores.items() if s == top]


@dataclass
class JudgeSpec:
    """Specification for a single judge in an ensemble.

    ``judge_model_config`` is the preferred keyword for the judge's configuration, and the
    positional form is unchanged. ``llm_config=`` keeps working without a warning: it is
    the name of the stored dataclass field, which is deliberately not renamed so that
    ``dataclasses.fields``/``asdict``/``replace``, the ``repr``, equality and pickles stay
    exactly as they were. ``dataclasses.replace`` therefore takes ``llm_config=``, and
    passing both ``judge_model_config=`` and ``llm_config=`` raises ``ValueError``.

    The configuration decides the judge's kind: an ``LLMConfig`` makes an LLM judge, which
    is asked about each criterion in its own call, and a ``DecisionModelConfig`` makes a
    decision-model judge, which is asked about every criterion of an item in one request.
    Both kinds mix freely in one ensemble.

    Attributes:
        llm_config: The judge's model configuration, an ``LLMConfig`` or a
            ``DecisionModelConfig`` (the stored field; also readable and writable as the
            ``judge_model_config`` property).
        judge_id: Unique identifier for this judge (e.g., "gpt-4", "claude-sonnet").
        weight: Voting weight for weighted aggregation (default 1.0).

    Example:
        >>> gemini = LLMConfig(model="gemini/gemini-3-flash-preview")
        >>> JudgeSpec(judge_model_config=gemini, judge_id="gemini")
        >>> JudgeSpec(gemini, "gemini", weight=2.0)  # positional form
        >>> JudgeSpec(DecisionModelConfig(model="jev-latest"), "jev")  # a decision model
    """

    llm_config: LLMConfig | DecisionModelConfig
    judge_id: str
    weight: float = 1.0

    # ``@dataclass`` keeps an ``__init__`` defined in the class body instead of generating
    # one, while ``fields()``, ``repr``, ``__eq__``, ``__match_args__`` and ``replace()``
    # are still derived from the three fields above. The overloads are the two accepted
    # call shapes; the sentinel defaults tell a missing argument from an explicit ``None``.
    @overload
    def __init__(
        self,
        llm_config: LLMConfig | DecisionModelConfig,
        judge_id: str,
        weight: float = 1.0,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        judge_model_config: LLMConfig | DecisionModelConfig,
        judge_id: str,
        weight: float = 1.0,
    ) -> None: ...

    def __init__(
        self,
        llm_config: Any = dataclasses.MISSING,
        judge_id: Any = dataclasses.MISSING,
        weight: float = 1.0,
        *,
        judge_model_config: Any = dataclasses.MISSING,
    ) -> None:
        """Initialize from ``judge_model_config`` (or its stored-field name ``llm_config``).

        Args:
            llm_config: The judge's model configuration, positionally or by the stored
                field name. Mutually exclusive with ``judge_model_config``.
            judge_id: Unique identifier for this judge.
            weight: Voting weight for weighted aggregation (default 1.0).
            judge_model_config: The judge's model configuration (preferred keyword).

        Raises:
            ValueError: If both ``judge_model_config`` and ``llm_config`` are passed.
            TypeError: If the configuration or ``judge_id`` is missing.
        """
        if judge_model_config is not dataclasses.MISSING:
            if llm_config is not dataclasses.MISSING:
                raise ValueError(
                    "Pass only one of judge_model_config and llm_config "
                    "(llm_config is the stored field name of judge_model_config)"
                )
            llm_config = judge_model_config
        missing = []
        if llm_config is dataclasses.MISSING:
            missing.append("'judge_model_config' (or its alias 'llm_config')")
        if judge_id is dataclasses.MISSING:
            missing.append("'judge_id'")
        if missing:
            # Mirror Python's own wording; a lone judge_id is exactly the positional
            # argument the generated dataclass __init__ used to report.
            kind = "positional argument" if missing == ["'judge_id'"] else "argument"
            plural = "s" if len(missing) > 1 else ""
            raise TypeError(
                f"JudgeSpec.__init__() missing {len(missing)} required {kind}{plural}: "
                + " and ".join(missing)
            )
        self.llm_config = llm_config
        self.judge_id = judge_id
        self.weight = weight

    @property
    def judge_model_config(self) -> LLMConfig | DecisionModelConfig:
        """The judge's model configuration (read/write alias of the ``llm_config`` field)."""
        return self.llm_config

    @judge_model_config.setter
    def judge_model_config(self, value: LLMConfig | DecisionModelConfig) -> None:
        self.llm_config = value


def _repeated_judge_ids(judges: Iterable[JudgeSpec]) -> list[str]:
    """The ``judge_id``s that more than one of ``judges`` has, each once, sorted."""
    counts = Counter(judge.judge_id for judge in judges)
    return sorted(judge_id for judge_id, count in counts.items() if count > 1)


@dataclass(frozen=True)
class EscalationConfig:
    """Configuration of a confidence cascade: which LLM judges take over, and when.

    Passed as ``CriterionGrader(escalation=...)`` to a grader whose one judge is a decision
    model. The decision model answers every criterion of an item, in its one request per
    item; a criterion is **escalated** when its vote errored, abstained (``CANNOT_ASSESS``
    or an NA option), or has a ``confidence`` below the criterion's threshold
    (``threshold_for``). Only escalated criteria go to the escalation judges, and only
    their votes decide such a criterion's verdict (aggregated with the grader's
    ``aggregation`` / ``ordinal_aggregation`` / ``nominal_aggregation``); the decision
    model's vote is kept with ``superseded=True`` and never used as a fallback.

    Validated at construction (``dataclasses.replace`` validates again) and frozen: its
    fields cannot be reassigned. The ``judges`` list and the ``per_criterion`` dict can
    still be changed in place, so a grader validates and keeps its own copy
    (``dataclasses.replace``) when it is built; later changes to the config do not reach
    it.

    Attributes:
        judges: The escalation judges, LLM judges only, in order. A bare ``LLMConfig``
            given to the constructor becomes ``[JudgeSpec(config, "escalation")]``. To
            compare a cascade with a separate LLM run judge for judge (the same prompts,
            option shuffles and few-shot examples included), grade that run with these
            judges (``judges=escalation.judges``, their ``judge_id``s included) and the
            cascade's ``seed`` and other LLM settings (``replay_escalation`` lists them).
        threshold: A criterion is escalated when the decision model's confidence is below
            it, in [0, 1]. ``0.0`` escalates only errors and abstentions; ``1.0`` everything
            short of certainty.
        per_criterion: Thresholds for particular criteria, keyed by criterion name (each
            in [0, 1]); other criteria, and unnamed ones, use ``threshold``. A grader
            checks the names against the rubrics it grades and warns about names that
            match no criterion.

    Example:
        >>> jev = DecisionModelConfig(model="jev-latest")
        >>> grader = CriterionGrader(
        ...     judge_model_config=jev,
        ...     escalation=EscalationConfig(
        ...         judges=LLMConfig(model="gemini/gemini-3-flash-preview"),
        ...         threshold=0.72,
        ...         per_criterion={"factuality": 0.95},
        ...     ),
        ... )
    """

    judges: list[JudgeSpec]
    threshold: float
    per_criterion: Mapping[str, float] | None = None

    # ``@dataclass`` keeps an ``__init__`` defined in the class body; ``fields()``, ``repr``,
    # ``__eq__`` and ``replace()`` still come from the three fields above. The constructor
    # takes ``judges`` in a wider form than the stored field, which always holds the list.
    def __init__(
        self,
        judges: LLMConfig | Sequence[JudgeSpec],
        threshold: float,
        per_criterion: Mapping[str, float] | None = None,
    ) -> None:
        """Validate and store the configuration.

        Args:
            judges: One LLM judge as a bare ``LLMConfig`` (its ``judge_id`` is
                ``"escalation"``), or a non-empty list of ``JudgeSpec`` of LLM judges.
            threshold: The confidence threshold, in [0, 1].
            per_criterion: Thresholds by criterion name, each in [0, 1]. Copied.

        Raises:
            ValueError: If ``judges`` is empty, holds anything but ``JudgeSpec``, or names
                a decision model; if ``threshold`` or a ``per_criterion`` threshold is not a
                number in [0, 1]; or if ``per_criterion`` is not a mapping from criterion
                names (strings).
        """
        name = type(self).__name__
        if isinstance(judges, LLMConfig):
            specs = [JudgeSpec(judge_model_config=judges, judge_id="escalation")]
        elif isinstance(judges, DecisionModelConfig):
            raise ValueError(
                f"{name}.judges must be LLM judges, not a decision model "
                "(DecisionModelConfig): they judge the criteria the decision model escalates"
            )
        elif isinstance(judges, (list, tuple)):
            specs = list(judges)
        else:
            raise ValueError(
                f"{name}.judges must be an LLMConfig or a list of JudgeSpec; "
                f"got {type(judges).__name__}"
            )
        if not specs:
            raise ValueError(f"{name}.judges must name at least one judge")
        for i, spec in enumerate(specs):
            if not isinstance(spec, JudgeSpec):
                raise ValueError(
                    f"{name}.judges[{i}] must be a JudgeSpec; got {type(spec).__name__} "
                    "(a single judge can be passed as a bare LLMConfig, not in a list)"
                )
        ids = [s.judge_id for s in specs if isinstance(s.llm_config, DecisionModelConfig)]
        if ids:
            raise ValueError(
                f"{name}.judges must be LLM judges, not decision models: "
                f"{', '.join(map(repr, ids))} {'is a' if len(ids) == 1 else 'are'} "
                "DecisionModelConfig; escalation judges judge the criteria the decision model "
                "escalates"
            )
        per_criterion = _check_thresholds(
            threshold,
            per_criterion,
            threshold_name=f"{name}.threshold",
            per_criterion_name=f"{name}.per_criterion",
        )
        # Frozen: the generated __setattr__ refuses assignment, so store past it.
        object.__setattr__(self, "judges", specs)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "per_criterion", per_criterion)

    def threshold_for(self, criterion: Criterion) -> float:
        """The confidence threshold below which ``criterion`` is escalated.

        Its ``per_criterion`` threshold, looked up by the criterion's name, else
        ``threshold``; an unnamed criterion always gets ``threshold``.
        """
        return _threshold_for(criterion, self.threshold, self.per_criterion)


def _check_thresholds(
    threshold: float,
    per_criterion: Mapping[str, float] | None,
    *,
    threshold_name: str,
    per_criterion_name: str,
) -> dict[str, float] | None:
    """Validate a cascade's escalation thresholds; return the per-criterion ones as a copy.

    The one check of ``EscalationConfig`` and of an offline cascade replay
    (``autorubric.escalation.replay_escalation``), so both accept exactly the same
    thresholds.

    Args:
        threshold: The global threshold, which must be a real number in [0, 1] (no ``bool``;
            NaN fails the range).
        per_criterion: ``None``, or a mapping from criterion names (``str``) to thresholds,
            each a real number in [0, 1].
        threshold_name: How error messages name ``threshold``.
        per_criterion_name: How error messages name ``per_criterion``.

    Returns:
        ``per_criterion`` copied into a ``dict``, or ``None``.

    Raises:
        ValueError: If a threshold is not a real number in [0, 1], or if ``per_criterion``
            is not a mapping from criterion names.
    """
    if not (_is_real_number(threshold) and 0 <= threshold <= 1):
        raise ValueError(f"{threshold_name} must be a number in [0, 1]; got {threshold!r}")
    if per_criterion is None:
        return None
    if not isinstance(per_criterion, Mapping):
        raise ValueError(
            f"{per_criterion_name} must map criterion names to thresholds; "
            f"got {type(per_criterion).__name__}"
        )
    for key, value in per_criterion.items():
        if not isinstance(key, str):
            raise ValueError(f"{per_criterion_name} keys are criterion names (str); got {key!r}")
        if not (_is_real_number(value) and 0 <= value <= 1):
            raise ValueError(
                f"{per_criterion_name}[{key!r}] must be a number in [0, 1]; got {value!r}"
            )
    return dict(per_criterion)


def _threshold_for(
    criterion: Criterion, threshold: float, per_criterion: Mapping[str, float] | None
) -> float:
    """The confidence threshold below which a cascade escalates ``criterion``.

    Its ``per_criterion`` threshold, looked up by the criterion's name, else ``threshold``;
    an unnamed criterion always gets ``threshold``. The one lookup of
    ``EscalationConfig.threshold_for`` and of an offline cascade replay
    (``autorubric.escalation.replay_escalation``), so both escalate the same criteria.
    """
    if per_criterion is not None and criterion.name is not None:
        return per_criterion.get(criterion.name, threshold)
    return threshold


def _warn_unknown_threshold_names(
    per_criterion: Mapping[str, float] | None,
    rubrics: Iterable[Iterable[Criterion]],
    *,
    source: str,
    where: str,
) -> bool:
    """Warn about per-criterion thresholds whose names match no criterion of ``rubrics``.

    Thresholds are looked up by criterion name (``_threshold_for``), so such a name sets no
    threshold; it is most likely a typo or a stale name. The warning is a ``UserWarning``
    attributed to the code that called autorubric. The one check of a live cascade
    (``CriterionGrader._check_escalation_names``) and of an offline cascade replay
    (``autorubric.escalation.replay_escalation``).

    Args:
        per_criterion: The thresholds by criterion name, if any.
        rubrics: The criteria the names are checked against, as one or more rubrics.
        source: What ``per_criterion`` is, for the message (e.g.
            ``"EscalationConfig.per_criterion"``).
        where: What ``rubrics`` are, for the message (e.g. "the rubric being graded").

    Returns:
        Whether it warned. It never does without per-criterion thresholds.
    """
    if not per_criterion:
        return False
    names = {criterion.name for criterion in itertools.chain.from_iterable(rubrics)}
    unknown = [name for name in per_criterion if name not in names]
    if not unknown:
        return False
    warnings.warn(
        f"{source} names {', '.join(map(repr, unknown))} match no criterion of {where}; "
        "thresholds are looked up by criterion name, so these have no effect",
        UserWarning,
        stacklevel=_caller_stacklevel(),
    )
    return True


@dataclass
class CriterionResult:
    """Result from evaluating a single criterion by a single judge.

    ``usage`` and ``cost`` belong to the call that produced the result. An LLM judge makes
    one call per criterion, so each result carries its own. A decision-model judge makes
    one request per item, whose usage and cost ride on the item's first result (the other
    results carry ``None``), so ``JudgeCriterionResults.total_usage``/``total_cost`` are the
    per-item totals for either kind. ``cost`` is ``None`` when it is unknown.
    """

    report: CriterionReport
    usage: TokenUsage | None = None
    cost: float | None = None


@dataclass
class JudgeCriterionResults:
    """All criterion results from a single judge, one entry per criterion, in rubric order.

    ``role`` is the judge's part in grading the item. A ``"primary"`` judge (every judge of
    a grader without a cascade, and a cascade's decision model) judges every criterion, so
    each entry is a result. An ``"escalation"`` judge of a cascade judges only the criteria
    the decision model escalated and holds ``None`` at the others; its list is full-length
    all the same, so every judge's entries line up by criterion index. ``reports``,
    ``total_usage`` and ``total_cost`` cover the criteria the judge judged.
    """

    judge_id: str
    weight: float
    criterion_results: list[CriterionResult | None] = field(default_factory=list)
    role: Literal["primary", "escalation"] = "primary"

    @property
    def reports(self) -> list[CriterionReport]:
        return [r.report for r in self.criterion_results if r is not None]

    @property
    def total_usage(self) -> TokenUsage | None:
        usages = [r.usage for r in self.criterion_results if r is not None and r.usage is not None]
        return sum(usages, TokenUsage()) if usages else None

    @property
    def total_cost(self) -> float | None:
        costs = [r.cost for r in self.criterion_results if r is not None and r.cost is not None]
        return sum(costs) if costs else None


def _escalates(report: CriterionReport, threshold: float) -> bool:
    """Whether a cascade escalates the criterion its decision model judged in ``report``.

    It does when the judgment failed (any ``error`` category), when the decision model
    abstained (``CANNOT_ASSESS``, or an NA option), and when its ``confidence`` is below
    ``threshold``. A failed judgment has no confidence; nor, therefore, can it clear the
    threshold. The one rule of the live cascade (``CriterionGrader.judge``) and of an
    offline replay (``autorubric.escalation.replay_escalation``).
    """
    return (
        report.is_error
        or report.is_na
        or report.confidence is None
        or report.confidence < threshold
    )


def _failed_judgment_result(
    criterion: Criterion,
    judge_id: str,
    error: Exception,
    *,
    shuffle_order: list[int] | None = None,
) -> CriterionResult:
    """The result a judge gets for ``criterion`` when judging it failed with ``error``.

    The single source of failure reports for every judge kind: an LLM judge's failed call,
    a decision model's failed request (once for each criterion posed in it), an answer
    that cannot be used, and a criterion a decision model cannot be asked about. The
    failure is classified by ``classify_grading_error`` and logged with the judge's id.
    Infrastructure (API/network) and parse/validation failures are not the submission's
    fault, so they abstain: ``CANNOT_ASSESS`` (binary) or ``na=True`` (multi-choice), which
    the default ``CannotAssessStrategy.SKIP`` excludes from scoring, instead of a
    score-affecting worst-case verdict. Only truly unknown errors keep the conservative
    worst case. The report surfaces the failure structurally (``error``, prefixed with the
    category; see ``is_error``) and in ``reason``, and carries no usage or cost.

    Args:
        criterion: The criterion as evaluated (after its NA option is guaranteed).
        judge_id: The judge whose judgment failed.
        error: The exception the failure raised.
        shuffle_order: The option permutation the judge was shown for a multi-choice
            criterion, if any, recorded on the report.

    Returns:
        The failure's ``CriterionResult``.
    """
    category = classify_grading_error(error)
    # The outcome fields each criterion kind's report has always been built with, so the
    # report is the same pydantic object field for field (fields set included).
    outcome: dict[str, Any]
    if criterion.options is None:
        if category == "unknown":
            verdict = _binary_worst_verdict(criterion.weight)
        else:
            verdict = CriterionVerdict.CANNOT_ASSESS
        logger.warning(
            f"Error evaluating criterion '{criterion.requirement[:50]}...' "
            f"with judge '{judge_id}' [{category}]: {error}"
        )
        outcome = {"verdict": verdict}
    else:
        logger.warning(
            f"Error evaluating multi-choice criterion '{criterion.requirement[:50]}...' "
            f"with judge '{judge_id}' [{category}]: {error}"
        )

        options = criterion.options
        if category == "unknown":
            # Conservative worst case, weight-sign-aware, among scored (non-NA)
            # options only — mirrors the binary worst case (MET if weight < 0 else
            # UNMET). Never auto-select an NA option for an unknown error; NA/skip
            # is reserved for infrastructure/parse failures. Ties resolve to the
            # first such option in declaration order (deterministic). The shared
            # helper is also used by the metrics layer's na_mode="as_unmet" remap
            # so the two layers cannot drift.
            worst_idx, worst_option = criterion.worst_scored_option()
            multi_choice_verdict = MultiChoiceVerdict(
                selected_index=worst_idx,
                selected_label=worst_option.label,
                value=worst_option.value,
                na=False,
            )
        else:
            # Infrastructure/parse: abstain (excluded under SKIP). Prefer a genuine
            # NA option — guaranteed when auto_na_option is on — so the abstain verdict
            # points at a real na=True option (resolves the contradiction for the
            # default case). With no NA option (forced-choice + no author NA) emit a
            # GENUINE abstain that selects no option (selected_index/label=None), rather
            # than forcing na=True onto a scored option (the old contradiction). It
            # stays na=True → excluded under SKIP, so infra/parse never penalizes.
            na_idx = criterion.na_option_index
            if na_idx is not None:
                na_option = options[na_idx]
                multi_choice_verdict = MultiChoiceVerdict(
                    selected_index=na_idx,
                    selected_label=na_option.label,
                    value=na_option.value,
                    na=True,
                )
            else:
                multi_choice_verdict = MultiChoiceVerdict(
                    selected_index=None,
                    selected_label=None,
                    value=0.0,
                    na=True,
                )
        outcome = {
            "verdict": None,
            "multi_choice_verdict": multi_choice_verdict,
            "shuffle_order": shuffle_order,
        }

    report = CriterionReport(
        requirement=criterion.requirement,
        **outcome,
        reason=f"Judge call failed ({category}): {str(error)}",
        error=f"{category}: {str(error)}",
        weight=criterion.weight,
        name=criterion.name,
        options=criterion.options,
        scale_type=criterion.scale_type,
        aggregation=criterion.aggregation,
    )
    return CriterionResult(report=report, usage=None, cost=None)


def _ensemble_evaluation_report(
    ensemble_reports: list[EnsembleCriterionReport],
    judge_scores: dict[str, float | None],
    score: Callable[[list[CriterionReport], bool], float],
    *,
    normalize: bool,
    token_usage: TokenUsage | None,
    completion_cost: float | None,
) -> EnsembleEvaluationReport:
    """An item's report, from its criteria's ensemble reports.

    ``score`` and ``raw_score`` are the weighted score over the final verdicts (normalized as
    ``normalize`` says, and raw), ``mean_agreement`` is the criteria's mean ``agreement``
    (``None`` for an empty rubric, never a fabricated 1.0), and ``cannot_assess_count``
    counts the criteria whose final verdict abstains (``CANNOT_ASSESS`` or an NA option).
    Shared by ``CriterionGrader.aggregate`` and the offline cascade replay
    (``autorubric.escalation.replay_escalation``), so a replayed report is assembled
    exactly as a live one.

    Args:
        ensemble_reports: The item's criterion reports, in rubric order.
        judge_scores: Each judge's own score over the rubric (``None`` where undefined).
        score: The weighted scoring of criterion reports, called as
            ``score(reports, normalize)`` (``score_reports`` under the grader's
            ``CannotAssessConfig``).
        normalize: Whether ``score`` is normalized to [0, 1].
        token_usage: The item's token usage, if known.
        completion_cost: The item's cost, if known.

    Returns:
        The item's ``EnsembleEvaluationReport``.
    """
    # The final verdicts as criterion reports, for scoring.
    final_reports = []
    for er in ensemble_reports:
        if er.final_multi_choice_verdict is not None:
            # Multi-choice criterion
            final_reports.append(
                CriterionReport(
                    weight=er.criterion.weight,
                    requirement=er.criterion.requirement,
                    name=er.criterion.name,
                    options=er.criterion.options,
                    scale_type=er.criterion.scale_type,
                    aggregation=er.criterion.aggregation,
                    verdict=None,  # Binary verdict is None
                    multi_choice_verdict=er.final_multi_choice_verdict,
                    reason=er.final_reason,
                )
            )
        else:
            # Binary criterion
            final_reports.append(
                CriterionReport(
                    weight=er.criterion.weight,
                    requirement=er.criterion.requirement,
                    name=er.criterion.name,
                    verdict=er.final_verdict,
                    reason=er.final_reason,
                )
            )
    final_score = score(final_reports, normalize)
    raw_score = score(final_reports, False)

    # Calculate agreement
    mean_agreement = (
        sum(er.agreement for er in ensemble_reports) / len(ensemble_reports)
        if ensemble_reports
        else None  # No criteria to agree on -> not measured (never fabricate 1.0)
    )

    # Count CANNOT_ASSESS (binary) and NA (multi-choice)
    cannot_assess_count = sum(
        1
        for er in ensemble_reports
        if (er.final_verdict == CriterionVerdict.CANNOT_ASSESS)
        or (er.final_multi_choice_verdict is not None and er.final_multi_choice_verdict.na)
    )

    return EnsembleEvaluationReport(
        score=final_score,
        raw_score=raw_score,
        llm_raw_score=raw_score,
        report=ensemble_reports,
        judge_scores=judge_scores,
        mean_agreement=mean_agreement,
        cannot_assess_count=cannot_assess_count,
        token_usage=token_usage,
        completion_cost=completion_cost,
    )


class CriterionGrader(Grader):
    """Unified criterion-based grader with compositional few-shot and ensemble support.

    This grader evaluates each criterion independently and supports:
    - Single-judge mode (via judge_model_config)
    - Ensemble mode with multiple judges (via judges)
    - Few-shot prompting (via training_data + few_shot_config)

    All combinations work: single judge, single + few-shot, ensemble, ensemble + few-shot.

    Parameters are orthogonal:
    - judge_model_config OR judges: Choose single-judge or ensemble mode
    - training_data + few_shot_config: Enable few-shot prompting (applies to every LLM judge)

    A judge is an LLM (``LLMConfig``) or a decision model (``DecisionModelConfig``), alone
    or mixed in one ensemble. An LLM judge is asked about each criterion in its own call. A
    decision-model judge is asked about the whole rubric in **one request per item**: the
    submission goes once as shared state and every criterion it can express becomes one
    question; a criterion it cannot express fails alone, and a failed request fails every
    criterion posed in it, through the same failure routing as an LLM judge's failed call.
    Its votes carry ``probabilities`` and ``confidence`` and no explanation (``reason`` is
    ``None``). In a mixed ensemble every judge votes on every criterion, and the votes
    aggregate exactly as an all-LLM ensemble's do.

    A confidence cascade (``escalation=EscalationConfig(...)``) puts one decision model
    first and LLM judges behind it. The decision model judges every criterion in its one
    request per item; a criterion it errored or abstained on, or answered with a
    ``confidence`` below the criterion's threshold, is escalated: each escalation judge
    judges it (and only the escalated criteria), with the same prompt it would get in an
    otherwise identical grader without the cascade (the same ``seed``, ``judge_id`` and
    LLM settings). An escalated
    criterion's report is marked ``escalated=True``, keeps the decision model's vote with
    ``superseded=True``, and takes its verdict from the escalation judges' votes alone,
    aggregated as any ensemble's; it never falls back to the decision model's vote. The
    decision model's ``judge_scores`` entry is its own score over every criterion; an
    escalation judge's is ``None``, as it never judges a whole rubric.

    Some settings exist for LLM judges only. Few-shot examples, the system prompts and
    ``shuffle_options`` apply to the LLM judges of a mixed ensemble or a cascade and never
    reach a decision model. A grader whose judges are all decision models rejects few-shot
    examples and warns when a system prompt is given or ``shuffle_options`` is ``False``,
    since it has no judge to apply them to. A custom response format describes a generated judgment,
    which only an LLM judge produces, so it cannot be combined with any decision-model
    judge.

    Example:
        >>> from autorubric import DecisionModelConfig, LLMConfig, FewShotConfig, RubricDataset
        >>> from autorubric.graders import CriterionGrader, JudgeSpec
        >>>
        >>> # Single LLM
        >>> grader = CriterionGrader(
        ...     judge_model_config=LLMConfig(model="gemini/gemini-3-flash-preview")
        ... )
        >>>
        >>> # Single LLM + few-shot
        >>> train, test = dataset.split_train_test(n_train=100)
        >>> grader = CriterionGrader(
        ...     judge_model_config=LLMConfig(model="gemini/gemini-3-flash-preview"),
        ...     training_data=train,
        ...     few_shot_config=FewShotConfig(n_examples=3),
        ... )
        >>>
        >>> # Ensemble
        >>> grader = CriterionGrader(
        ...     judges=[
        ...         JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini"),
        ...         JudgeSpec(LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"), "claude"),
        ...     ],
        ...     aggregation="majority",
        ... )
        >>>
        >>> # Ensemble + few-shot
        >>> grader = CriterionGrader(
        ...     judges=[JudgeSpec(...), JudgeSpec(...)],
        ...     aggregation="majority",
        ...     training_data=train,
        ...     few_shot_config=FewShotConfig(n_examples=3),
        ... )
        >>>
        >>> # Decision model: one request per item for the whole rubric
        >>> grader = CriterionGrader(judge_model_config=DecisionModelConfig(model="jev-latest"))
        >>>
        >>> # Mixed ensemble of a decision model and an LLM
        >>> grader = CriterionGrader(
        ...     judges=[
        ...         JudgeSpec(DecisionModelConfig(model="jev-latest"), "jev"),
        ...         JudgeSpec(LLMConfig(model="gemini/gemini-3-flash-preview"), "gemini"),
        ...     ],
        ... )
        >>>
        >>> # Cascade: the decision model first, an LLM for what it is unsure of
        >>> grader = CriterionGrader(
        ...     judge_model_config=DecisionModelConfig(model="jev-latest"),
        ...     escalation=EscalationConfig(
        ...         judges=LLMConfig(model="gemini/gemini-3-flash-preview"), threshold=0.72
        ...     ),
        ... )
    """

    # Class-level defaults of the cascade state: a grader whose ``__init__`` never ran (a
    # subclass that skips it, or one unpickled from a version without the cascade) has no
    # cascade, and a grader has not warned about unknown ``per_criterion`` names until
    # ``judge`` does.
    _escalation: EscalationConfig | None = None
    _escalation_names_warned: bool = False

    def __init__(
        self,
        *,
        # Single-judge mode
        judge_model_config: LLMConfig | DecisionModelConfig | None = None,
        # Ensemble mode (mutually exclusive with judge_model_config)
        judges: list[JudgeSpec] | None = None,
        aggregation: AggregationStrategy = "majority",
        # Multi-choice aggregation strategies
        ordinal_aggregation: OrdinalAggregation = "mean",
        nominal_aggregation: NominalAggregation = "mode",
        # Few-shot mode (orthogonal - applies to every LLM judge)
        training_data: RubricDataset | None = None,
        few_shot_config: FewShotConfig | None = None,
        # Common parameters
        system_prompt: str | None = None,
        multi_choice_system_prompt: str | None = None,
        length_penalty: LengthPenalty | None = None,
        normalize: bool = True,
        cannot_assess_config: CannotAssessConfig | None = None,
        # Position bias mitigation
        shuffle_options: bool = True,
        # Multi-choice abstain channel
        auto_na_option: bool = True,
        # Reproducibility
        seed: int | None = None,
        # Structured output override for binary criteria
        binary_response_format: type[BaseModel] | None = None,
        # Structured output override for multi-choice criteria
        multi_choice_response_format: type[BaseModel] | None = None,
        # Confidence cascade: LLM judges behind a decision model
        escalation: EscalationConfig | None = None,
        # Deprecated alias of judge_model_config (every parameter is keyword-only, so
        # its position at the end changes no call)
        llm_config: LLMConfig | DecisionModelConfig | None = None,
    ):
        """Initialize the criterion grader.

        Args:
            judge_model_config: Configuration of the single judge (single-judge mode), which
                is given ``judge_id="default"``: an ``LLMConfig`` for an LLM judge or a
                ``DecisionModelConfig`` for a decision-model judge. Mutually exclusive with
                judges.
            judges: List of JudgeSpec for ensemble mode, each an LLM or a decision model,
                each with its own ``judge_id``. Mutually exclusive with judge_model_config.
            aggregation: Strategy for aggregating votes in ensemble mode (binary criteria).
            ordinal_aggregation: Strategy for aggregating ordinal multi-choice votes.
                Central tendency: "mean", "median", "weighted_mean", "mode". Conservative/
                permissive (analogs of binary unanimous/any): "min" (lowest selected
                option) / "max" (highest selected option).
            nominal_aggregation: Strategy for aggregating nominal multi-choice votes.
                Options: "mode", "weighted_mode", "unanimous". "unanimous" abstains via the
                NA option on disagreement, or falls back to mode + warns if there is no NA
                option.
            training_data: Dataset for few-shot examples. If provided, enables few-shot
                prompting for the LLM judges; examples are selected for, and sent to, LLM
                judges only, never to a decision-model judge, whose request is exactly what
                it would be without them. An example shows its item's ground truth for the
                criterion and, when ``few_shot_config.include_reason`` is True, the item's
                written reason for it (``DataItem.ground_truth_reasons``), if any.
            few_shot_config: Configuration for few-shot example selection (LLM judges only).
            system_prompt: Custom system prompt for binary criteria. Applies to LLM judges
                only (a decision model's request has no system prompt); in a mixed ensemble
                the LLM judges use it.
            multi_choice_system_prompt: Custom system prompt for multi-choice criteria. Applies
                to LLM judges only, like ``system_prompt``.
            length_penalty: Optional length penalty configuration.
            normalize: If True, normalize score to [0, 1]. If False, return raw sum.
            cannot_assess_config: Configuration for handling CANNOT_ASSESS verdicts.
            shuffle_options: If True (default), randomize the order of multi-choice options
                presented to the LLM to mitigate position bias. Each judge/call sees a
                different random order, and responses are mapped back to original indices.
                Disable for deterministic behavior in tests. Applies to LLM judges only: a
                decision model's question offers the options in the rubric's order, and in a
                mixed ensemble the LLM judges shuffle as set here.
            auto_na_option: If True (default), auto-inject a canonical NA / "cannot assess"
                option into any multi-choice criterion that lacks one, giving the judge a
                first-class abstain channel analogous to binary CANNOT_ASSESS. The injected
                option is appended at the end (highest index) so existing option indices are
                preserved. Set False for forced-choice classification (the judge must pick a
                scored option). Never strips an author-supplied NA option — author intent wins.
            seed: Master seed for all non-LLM randomness (option shuffling, few-shot
                example selection). Auto-generated when None so that randomness is always
                pinned and reproducible. Inspect via the ``seed`` property after construction.
            binary_response_format: Pydantic model to use as the structured output schema
                for binary criterion judgments. Must be a subclass of (or compatible with)
                CriterionJudgment. If the model includes an ``affected_criteria`` field
                (list[int]), matching indices are injected as an ``[Affects: ...]`` tag
                into the reason string. Defaults to CriterionJudgment. LLM judges only: it
                describes a generated judgment, which a decision model does not produce, so
                it cannot be combined with any decision-model judge.
            multi_choice_response_format: Pydantic model to use as the structured output
                schema for multi-choice criterion judgments. Must be a subclass of (or
                compatible with) MultiChoiceJudgment. If the model includes an
                ``affected_criteria`` field (list[int]), matching indices are injected as
                an ``[Affects: ...]`` tag into the reason string (same convention as
                binary_response_format). Defaults to MultiChoiceJudgment. LLM judges only,
                like binary_response_format.
            escalation: Makes the grader a confidence cascade (see the class docstring):
                the escalation judges, all LLMs, judge the criteria the decision model
                errored or abstained on or answered with a confidence below the
                criterion's threshold. The grader's one judge (``judge_model_config``, or a
                single ``JudgeSpec`` in ``judges``) must then be a decision model. The
                escalation judges count as judges for the LLM-only settings: few-shot
                examples, the system prompts and ``shuffle_options`` apply to them. The
                grader validates and keeps its own copy of the config (``dataclasses.replace``),
                so changing the config's lists in place later does not affect it.
            llm_config: Deprecated alias of ``judge_model_config``; builds the same single
                judge and emits a ``DeprecationWarning``.

        Raises:
            ValueError: If neither judge_model_config nor judges is provided, if both are
                provided, or if both judge_model_config and its deprecated alias
                llm_config are provided; if escalation is given while the grader's judges
                are not exactly one decision model, with a ``judge_id`` that repeats among
                the decision model and the escalation judges, or with a config that no
                longer validates (its ``judges`` list or ``per_criterion`` dict changed in
                place since it was built, e.g. emptied); if few_shot_config or
                training_data is given while every judge is a decision model (few-shot
                examples apply to LLM judges only); if binary_response_format or
                multi_choice_response_format is given while any judge is a decision model;
                or if a decision-model judge has no API key, has a key the SDK would reject
                (anything but printable ASCII without whitespace), or has a resolved base
                URL (``api_base`` or the ``TYPESAFE_BASE_URL`` environment variable) that is
                not an absolute http(s) URL free of whitespace and control characters,
                query, fragment and credentials, or whose host the SDK's HTTP stack cannot
                encode (e.g. a non-ASCII name that is not a valid internationalized domain
                name). Such a judge fails here, never as an abstention or a worst case on
                every item.
            ImportError: If a judge is a decision model and the TypeSafe SDK is not
                installed (``pip install 'autorubric[typesafe]'``).

        Warns:
            UserWarning: If system_prompt or multi_choice_system_prompt is not None, or
                shuffle_options is False, while every judge is a decision model: these
                settings apply to LLM judges only, so they have no effect. One warning per
                setting that differs from its default (None, None and True); a cascade's
                escalation judges are LLM judges, so a cascade never warns.
            FutureWarning: If a ``judge_id`` repeats among ``judges``. Judges that share
                one are conflated: those of one kind (LLM or decision model) all call the
                model of the last of them, and all share one ``judge_scores`` entry, one set
                of per-judge metrics, and the same option shuffles and few-shot examples.
                Grading is otherwise unchanged; a repeat will be a ``ValueError`` in the
                next major version, as it already is in a cascade.
        """
        if llm_config is not None:
            if judge_model_config is not None:
                raise ValueError(
                    "Pass only one of judge_model_config and llm_config "
                    "(llm_config is a deprecated alias of judge_model_config)"
                )
            warnings.warn(
                "llm_config is deprecated; use judge_model_config",
                DeprecationWarning,
                stacklevel=2,
            )
            judge_model_config = llm_config

        super().__init__(length_penalty=length_penalty, normalize=normalize)

        # Validate: must have either judge_model_config or judges, not both, not neither
        if judge_model_config is None and judges is None:
            raise ValueError("Must provide either judge_model_config or judges")
        if judge_model_config is not None and judges is not None:
            raise ValueError("Cannot provide both judge_model_config and judges")

        # Normalize to ensemble representation (single judge = ensemble of 1).
        # The validation above guarantees exactly one of judge_model_config/judges is set,
        # so by this point `judges` in the else branch is a non-None list[JudgeSpec]; the
        # explicit attribute annotation lets the type checker see that without a
        # suppression comment.
        self._judges: list[JudgeSpec]
        if judge_model_config is not None:
            self._judges = [
                JudgeSpec(judge_model_config=judge_model_config, judge_id="default", weight=1.0)
            ]
        else:
            assert judges is not None
            self._judges = judges

        # The grader's own copy of the cascade config, validated again: the config's list and
        # dict can have been changed in place since it was built, and can be again later.
        # Everything the cascade does reads this copy. Its escalation judges
        # (``_escalation_judges``) are kept apart from ``_judges``, which judge every
        # criterion: they judge only the criteria the decision model escalates. Every other
        # per-judge concern (clients, few-shot examples, the LLM-only settings, the
        # manifest, ``judge_ids``) covers both lists.
        self._escalation = dataclasses.replace(escalation) if escalation is not None else None
        if self._escalation is not None:
            primary = self._judges
            if len(primary) != 1 or not isinstance(primary[0].llm_config, DecisionModelConfig):
                described = [
                    f"{j.judge_id!r} (a decision model)"
                    if isinstance(j.llm_config, DecisionModelConfig)
                    else f"{j.judge_id!r} (an LLM)"
                    for j in primary
                ]
                raise ValueError(
                    "escalation needs exactly one primary judge, a decision model "
                    "(judge_model_config=DecisionModelConfig(...), or judges=[JudgeSpec("
                    "DecisionModelConfig(...), ...)]), which judges every criterion and "
                    f"escalates the ones it is unsure of; got {', '.join(described)}"
                )
            repeated = _repeated_judge_ids((*primary, *self._escalation_judges))
            if repeated:
                raise ValueError(
                    "judge_ids must be unique across the decision model and the escalation "
                    f"judges; repeated: {', '.join(repr(judge_id) for judge_id in repeated)}"
                )
        # A panel's judge_ids should be unique too, but a repeat has always been accepted,
        # so it warns until the next major version makes it the cascade's ValueError. A
        # cascade's primary is one judge, so this never warns for a cascade.
        repeated = _repeated_judge_ids(self._judges)
        if repeated:
            warnings.warn(
                f"judge_ids should be unique; repeated: {', '.join(map(repr, repeated))}. "
                "Judges that share a judge_id are conflated: those of one kind (LLM or "
                "decision model) all call the model of the last of them, and all share one "
                "judge_scores entry, one set of per-judge metrics, and the same option "
                "shuffles and few-shot examples. Give each judge its own judge_id; a "
                "repeated judge_id will be a ValueError in the next major version.",
                FutureWarning,
                stacklevel=2,
            )
        all_judges = [*self._judges, *self._escalation_judges]

        # Settings a decision-model judge cannot use, checked before any client is built so
        # that a configuration error is reported as one.
        decision_model_ids = [
            j.judge_id for j in all_judges if isinstance(j.llm_config, DecisionModelConfig)
        ]
        every_judge_is_decision_model = 0 < len(decision_model_ids) == len(all_judges)
        if every_judge_is_decision_model and (
            few_shot_config is not None or training_data is not None
        ):
            raise ValueError(
                "few-shot examples apply to LLM judges only, and every judge of this "
                "grader is a decision model: remove few_shot_config and training_data, "
                "or add an LLM judge"
            )
        response_formats = [
            name
            for name, value in (
                ("binary_response_format", binary_response_format),
                ("multi_choice_response_format", multi_choice_response_format),
            )
            if value is not None
        ]
        if response_formats and decision_model_ids:
            raise ValueError(
                f"{' and '.join(response_formats)} cannot be combined with decision-model "
                f"judges ({', '.join(repr(judge_id) for judge_id in decision_model_ids)}): "
                "a response format describes a generated judgment, which only an LLM judge "
                "produces"
            )
        if every_judge_is_decision_model:
            # A setting that differs from its default asks for something no judge here
            # can do: a prompt (None is "no prompt"), or shuffle_options=False (a decision
            # model never shuffles, so True, the default, is what it does anyway).
            changed = (
                ("system_prompt", system_prompt is not None),
                ("multi_choice_system_prompt", multi_choice_system_prompt is not None),
                ("shuffle_options", not shuffle_options),
            )
            for name, is_changed in changed:
                if is_changed:
                    warnings.warn(
                        f"{name} applies to LLM judges only; this grader has none (every "
                        "judge is a decision model), so it has no effect",
                        UserWarning,
                        stacklevel=2,
                    )

        self._aggregation = aggregation
        self._ordinal_aggregation = ordinal_aggregation
        self._nominal_aggregation = nominal_aggregation
        self._training_data = training_data
        self._cannot_assess_config = cannot_assess_config or CannotAssessConfig()
        self._shuffle_options = shuffle_options
        self._auto_na_option = auto_na_option
        self._seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        # Coordinate few-shot seed with master seed when unset
        fsc = few_shot_config or FewShotConfig()
        if fsc.seed is None and training_data is not None:
            fsc = dataclasses.replace(fsc, seed=self._seed)
        self._few_shot_config = fsc
        self._binary_response_format = binary_response_format or CriterionJudgment
        self._multi_choice_response_format = multi_choice_response_format or MultiChoiceJudgment

        # Build system prompts (separate for binary and multi-choice)
        if system_prompt is None:
            self._system_prompt = GRADER_SYSTEM_PROMPT_DEFAULT
            if training_data is not None:
                self._system_prompt += FEW_SHOT_SYSTEM_PROMPT_ADDITION
        else:
            self._system_prompt = system_prompt

        if multi_choice_system_prompt is None:
            self._multi_choice_system_prompt = MULTI_CHOICE_SYSTEM_PROMPT
            if training_data is not None:
                self._multi_choice_system_prompt += MULTI_CHOICE_FEW_SHOT_ADDITION
        else:
            self._multi_choice_system_prompt = multi_choice_system_prompt

        # Create each judge's client: an LLMClient for an LLM judge, a DecisionModelClient
        # for a decision-model judge. Building one resolves its credentials, so a missing
        # SDK or API key fails here rather than on every item.
        self._clients: dict[str, LLMClient] = {}
        self._decision_clients: dict[str, DecisionModelClient] = {}
        for judge in all_judges:
            config = judge.llm_config
            if isinstance(config, DecisionModelConfig):
                self._decision_clients[judge.judge_id] = DecisionModelClient(config)
            else:
                self._clients[judge.judge_id] = LLMClient(config)

        # Pre-compute few-shot examples if training data provided
        # Note: For multi-choice, examples are stored as (submission, selected_index, reason)
        self._criterion_examples: dict[tuple[int, str], list[FewShotExample]] = {}
        self._multi_choice_examples: dict[tuple[int, str], list[tuple[str, int, str | None]]] = {}
        if training_data is not None:
            self._prepare_examples()

    @property
    def is_ensemble(self) -> bool:
        """Whether this grader has several judges that each judge every criterion.

        A cascade is not an ensemble: its escalation judges judge only the criteria its
        decision model escalates.
        """
        return len(self._judges) > 1

    @property
    def _escalation_judges(self) -> list[JudgeSpec]:
        """A cascade's escalation judges (its ``EscalationConfig.judges``); none without one."""
        return self._escalation.judges if self._escalation is not None else []

    @property
    def judge_ids(self) -> list[str]:
        """The ``judge_id`` of every judge, in order: the judges that judge every criterion
        (``judges``, or the single ``"default"`` judge), then a cascade's escalation judges.
        """
        return [j.judge_id for j in (*self._judges, *self._escalation_judges)]

    @property
    def has_few_shot(self) -> bool:
        """Whether this grader uses few-shot prompting."""
        return self._training_data is not None

    @property
    def seed(self) -> int:
        """The master seed governing all non-LLM randomness in this grader."""
        return self._seed

    # =========================================================================
    # Few-Shot Example Preparation
    # =========================================================================

    def _prepare_examples(self) -> None:
        """Pre-compute few-shot examples for each criterion and each LLM judge.

        The LLM judges include a cascade's escalation judges. Decision-model judges get
        none: few-shot examples apply to LLM judges only. Each judge's selection is keyed on
        its own ``judge_id``, so leaving decision models out changes no LLM judge's
        examples, and an escalation judge gets the examples it would get in a grader without
        the cascade with the same ``judge_id``.
        """
        training_data = self._training_data
        if training_data is None:
            return

        n_criteria = training_data.num_criteria
        rubric_criteria = training_data.rubric.rubric if training_data.rubric else []
        llm_judges = [
            j
            for j in (*self._judges, *self._escalation_judges)
            if not isinstance(j.llm_config, DecisionModelConfig)
        ]

        for criterion_idx in range(n_criteria):
            criterion = (
                rubric_criteria[criterion_idx] if criterion_idx < len(rubric_criteria) else None
            )
            # Select examples per judge so that ensemble judges see decorrelated few-shot
            # subsets/orderings, mirroring per-judge option shuffling.
            for judge in llm_judges:
                if criterion is not None and criterion.is_multi_choice:
                    examples = self._select_multi_choice_examples(criterion_idx, judge.judge_id)
                    self._multi_choice_examples[(criterion_idx, judge.judge_id)] = examples
                else:
                    binary_examples = self._select_examples_for_criterion(
                        criterion_idx, judge.judge_id
                    )
                    self._criterion_examples[(criterion_idx, judge.judge_id)] = binary_examples

    def _select_examples_for_criterion(
        self, criterion_idx: int, judge_id: str
    ) -> list[FewShotExample]:
        """Select stratified examples for a specific criterion and judge.

        Each example carries its item's reason for this criterion
        (``DataItem.ground_truth_reasons``), if it has one; reasons never affect the draw.
        """
        if self._training_data is None:
            return []

        config = self._few_shot_config
        seed = config.seed
        # Guaranteed non-None when training_data is present (see __init__ few-shot seed
        # coordination); _prepare_examples is the only caller and runs only in that case.
        assert seed is not None
        rng = _derive_shuffle_rng(seed, FEW_SHOT_DOMAIN, criterion_idx, judge_id)

        # Group items by verdict for this criterion
        verdict_groups: dict[CriterionVerdict, list] = {
            CriterionVerdict.MET: [],
            CriterionVerdict.UNMET: [],
            CriterionVerdict.CANNOT_ASSESS: [],
        }

        for item in self._training_data:
            if item.ground_truth is None:
                continue
            verdict = item.ground_truth[criterion_idx]
            verdict_groups[verdict].append(item)

        available_verdicts = [v for v, items in verdict_groups.items() if items]
        if not available_verdicts:
            return []

        n_examples = config.n_examples

        if config.balance_verdicts and len(available_verdicts) > 1:
            return self._balanced_selection(verdict_groups, n_examples, criterion_idx, rng)
        else:
            all_items = [item for items in verdict_groups.values() for item in items]
            rng.shuffle(all_items)
            return [
                FewShotExample(
                    submission=item.submission,
                    verdict=item.ground_truth[criterion_idx],  # type: ignore
                    reason=_ground_truth_reason(item, criterion_idx),
                )
                for item in all_items[:n_examples]
            ]

    def _balanced_pick(
        self,
        groups: dict,
        n_examples: int,
        rng: random.Random,
        transform,
        identity,
    ) -> list:
        """Select items with balanced distribution across non-empty groups.

        Divides slots evenly, assigns remainder to the first groups, then
        fills any remaining slots from unused items across all groups.

        Args:
            groups: Mapping from group key to list of items.
            n_examples: Total number of items to select.
            rng: Random instance for shuffling.
            transform: Convert an item to the output type.
            identity: Extract a hashable dedup key from an item.
        """
        results = []
        available = [(k, v) for k, v in groups.items() if v]
        if not available:
            return results

        base_per_group = n_examples // len(available)
        remainder = n_examples % len(available)

        used_ids: set = set()
        for i, (_key, items) in enumerate(available):
            pool = items.copy()
            rng.shuffle(pool)
            count = min(base_per_group + (1 if i < remainder else 0), len(pool))
            for item in pool[:count]:
                results.append(transform(item))
                used_ids.add(identity(item))

        if len(results) < n_examples:
            remaining = [
                item
                for _k, items in groups.items()
                for item in items
                if identity(item) not in used_ids
            ]
            rng.shuffle(remaining)
            for item in remaining[: n_examples - len(results)]:
                results.append(transform(item))

        return results

    def _balanced_selection(
        self,
        verdict_groups: dict[CriterionVerdict, list],
        n_examples: int,
        criterion_idx: int,
        rng: random.Random,
    ) -> list[FewShotExample]:
        """Select examples with balanced verdict distribution."""
        return self._balanced_pick(
            groups=verdict_groups,
            n_examples=n_examples,
            rng=rng,
            transform=lambda item: FewShotExample(
                submission=item.submission,
                verdict=item.ground_truth[criterion_idx],  # type: ignore
                reason=_ground_truth_reason(item, criterion_idx),
            ),
            identity=lambda item: item.submission,
        )

    def _label_to_option_index(self, criterion_idx: int, label: str) -> int | None:
        """Map a ground truth label string to a 0-based option index."""
        if self._training_data is None or self._training_data.rubric is None:
            return None
        criteria = self._training_data.rubric.rubric
        if criterion_idx >= len(criteria):
            return None
        try:
            return criteria[criterion_idx].find_option_by_label(label)
        except ValueError:
            return None

    def _select_multi_choice_examples(
        self, criterion_idx: int, judge_id: str
    ) -> list[tuple[str, int, str | None]]:
        """Select few-shot examples for a multi-choice criterion and judge.

        Groups training items by their selected option index and balances
        across options when configured. Ground truth labels are converted
        to 0-based option indices. Each example carries its item's reason for
        this criterion (``DataItem.ground_truth_reasons``), if it has one;
        reasons never affect the draw.
        """
        if self._training_data is None:
            return []

        config = self._few_shot_config
        seed = config.seed
        # Guaranteed non-None when training_data is present (see __init__ few-shot seed
        # coordination); _prepare_examples is the only caller and runs only in that case.
        assert seed is not None
        rng = _derive_shuffle_rng(seed, FEW_SHOT_DOMAIN, criterion_idx, judge_id)

        # Group items by option index (converting label to index)
        option_groups: dict[int, list] = {}
        for item in self._training_data:
            if item.ground_truth is None:
                continue
            label = item.ground_truth[criterion_idx]
            idx = (
                self._label_to_option_index(criterion_idx, label)
                if isinstance(label, str)
                else label
            )
            if idx is None:
                continue
            option_groups.setdefault(idx, []).append((item, idx))

        available_options = [k for k, v in option_groups.items() if v]
        if not available_options:
            return []

        n_examples = config.n_examples

        if config.balance_verdicts and len(available_options) > 1:
            sorted_groups = {k: option_groups[k] for k in sorted(available_options)}
            return self._balanced_pick(
                groups=sorted_groups,
                n_examples=n_examples,
                rng=rng,
                transform=lambda pair: (
                    pair[0].submission,
                    pair[1],
                    _ground_truth_reason(pair[0], criterion_idx),
                ),
                identity=lambda pair: pair[0].submission,
            )
        else:
            all_pairs = [
                (item, resolved_idx)
                for pairs in option_groups.values()
                for item, resolved_idx in pairs
            ]
            rng.shuffle(all_pairs)
            return [
                (item.submission, resolved_idx, _ground_truth_reason(item, criterion_idx))
                for item, resolved_idx in all_pairs[:n_examples]
            ]

    # =========================================================================
    # Single Criterion Evaluation
    # =========================================================================

    async def _judge_single_criterion(
        self,
        judge: JudgeSpec,
        criterion: Criterion,
        criterion_idx: int,
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
        *,
        guidelines: str | None = None,
    ) -> CriterionResult:
        """Judge a single criterion with a single judge.

        Handles both binary (MET/UNMET) and multi-choice criteria.
        """
        client = self._clients[judge.judge_id]

        # Dispatch to appropriate handler based on criterion type
        if criterion.is_multi_choice:
            return await self._judge_multi_choice_criterion(
                client,
                judge,
                criterion,
                criterion_idx,
                to_grade,
                query,
                reference_submission,
                guidelines=guidelines,
            )
        else:
            return await self._judge_binary_criterion(
                client,
                judge,
                criterion,
                criterion_idx,
                to_grade,
                query,
                reference_submission,
                guidelines=guidelines,
            )

    async def _judge_binary_criterion(
        self,
        client: LLMClient,
        judge: JudgeSpec,
        criterion: Criterion,
        criterion_idx: int,
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
        *,
        guidelines: str | None = None,
    ) -> CriterionResult:
        """Judge a binary (MET/UNMET) criterion."""
        examples = self._criterion_examples.get((criterion_idx, judge.judge_id), [])

        # Build prompt (with or without few-shot examples)
        if examples:
            user_prompt = build_few_shot_user_prompt(
                criterion=criterion,
                to_grade=to_grade,
                examples=examples,
                query=query,
                include_reason=self._few_shot_config.include_reason,
                reference_submission=reference_submission,
                guidelines=guidelines,
            )
        else:
            user_prompt = build_user_prompt(
                criterion, to_grade, query, reference_submission, guidelines=guidelines
            )

        try:
            result: GenerateResult = await client.generate(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                response_format=self._binary_response_format,
                return_result=True,
            )

            judgment = result.parsed
            # Read explanation, then criterion_status, then check for a null explanation:
            # a judgment missing a field fails on the first one read, and a null
            # explanation is a parse failure only when the verdict is readable.
            reason = _inject_affected_criteria(judgment.explanation, judgment)
            verdict = judgment.criterion_status
            reason = _require_llm_reason(reason)

            report = CriterionReport(
                requirement=criterion.requirement,
                verdict=verdict,
                reason=reason,
                # Preserve the extended-thinking deliberation trace. getattr
                # guards custom binary_response_format models that lack the field.
                reasoning=getattr(judgment, "reasoning", None),
                weight=criterion.weight,
                name=criterion.name,
                options=criterion.options,
                scale_type=criterion.scale_type,
                aggregation=criterion.aggregation,
            )
            return CriterionResult(report=report, usage=result.usage, cost=result.cost)

        except Exception as e:
            # Classified by classify_grading_error: infrastructure/parse abstain
            # (CANNOT_ASSESS), unknown keeps the conservative worst case.
            return _failed_judgment_result(criterion, judge.judge_id, e)

    async def _judge_multi_choice_criterion(
        self,
        client: LLMClient,
        judge: JudgeSpec,
        criterion: Criterion,
        criterion_idx: int,
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
        *,
        guidelines: str | None = None,
    ) -> CriterionResult:
        """Judge a multi-choice criterion.

        If shuffle_options is enabled, options are presented to the LLM in a
        randomized order to mitigate position bias. The response is mapped back
        to the original option indices.
        """
        if criterion.options is None:
            raise ValueError("Multi-choice criterion must have options")

        # Shuffle options to mitigate position bias
        # shuffled_indices[shuffled_pos] = original_pos
        if self._shuffle_options:
            original_indices = list(range(len(criterion.options)))
            shuffled_indices = original_indices.copy()
            item_key = hashlib.sha256(to_grade.encode()).hexdigest()[:16]
            rng = _derive_shuffle_rng(self._seed, item_key, criterion_idx, judge.judge_id)
            rng.shuffle(shuffled_indices)

            # Create shuffled options list
            shuffled_options = [criterion.options[i] for i in shuffled_indices]

            # Create criterion with shuffled options for prompt building

            prompt_criterion = Criterion(
                weight=criterion.weight,
                requirement=criterion.requirement,
                name=criterion.name,
                options=shuffled_options,
                scale_type=criterion.scale_type,
                aggregation=criterion.aggregation,
            )
        else:
            shuffled_indices = list(range(len(criterion.options)))
            prompt_criterion = criterion

        examples = self._multi_choice_examples.get((criterion_idx, judge.judge_id), [])

        # Build prompt (with or without few-shot examples)
        if examples:
            # Note: If shuffling is enabled and few-shot examples are used,
            # we need to transform example indices to match shuffled order.
            # Create inverse mapping: original_to_shuffled[original_pos] = shuffled_pos
            original_to_shuffled = {orig: shuf for shuf, orig in enumerate(shuffled_indices)}
            transformed_examples = [
                (submission, original_to_shuffled[orig_idx], reason)
                for submission, orig_idx, reason in examples
            ]
            user_prompt = build_multi_choice_few_shot_user_prompt(
                criterion=prompt_criterion,
                to_grade=to_grade,
                examples=transformed_examples,
                query=query,
                include_reason=self._few_shot_config.include_reason,
                reference_submission=reference_submission,
                guidelines=guidelines,
            )
        else:
            user_prompt = build_multi_choice_user_prompt(
                prompt_criterion, to_grade, query, reference_submission, guidelines=guidelines
            )

        try:
            result: GenerateResult = await client.generate(
                system_prompt=self._multi_choice_system_prompt,
                user_prompt=user_prompt,
                response_format=self._multi_choice_response_format,
                return_result=True,
            )

            judgment: MultiChoiceJudgment = result.parsed

            # Convert 1-indexed response to 0-indexed (in shuffled space)
            shuffled_idx = judgment.selected_option - 1

            # Validate index is in range
            if shuffled_idx < 0 or shuffled_idx >= len(criterion.options):
                raise ValueError(
                    f"Selected option {judgment.selected_option} out of range "
                    f"[1, {len(criterion.options)}]"
                )

            # Map back from shuffled position to original index
            original_idx = shuffled_indices[shuffled_idx]

            selected_option = criterion.options[original_idx]
            multi_choice_verdict = MultiChoiceVerdict(
                selected_index=original_idx,
                selected_label=selected_option.label,
                value=selected_option.value,
                na=selected_option.na,
            )

            reason = _require_llm_reason(_inject_affected_criteria(judgment.explanation, judgment))

            report = CriterionReport(
                requirement=criterion.requirement,
                verdict=None,  # Binary verdict is None for multi-choice
                multi_choice_verdict=multi_choice_verdict,
                reason=reason,
                # Preserve the extended-thinking deliberation trace.
                reasoning=getattr(judgment, "reasoning", None),
                weight=criterion.weight,
                name=criterion.name,
                options=criterion.options,
                scale_type=criterion.scale_type,
                aggregation=criterion.aggregation,
                shuffle_order=shuffled_indices if self._shuffle_options else None,
            )
            return CriterionResult(report=report, usage=result.usage, cost=result.cost)

        except Exception as e:
            # Classified by classify_grading_error: infrastructure/parse abstain (na=True),
            # unknown keeps the conservative worst case among the scored options.
            return _failed_judgment_result(
                criterion,
                judge.judge_id,
                e,
                shuffle_order=shuffled_indices if self._shuffle_options else None,
            )

    async def _judge_all_criteria_for_judge(
        self,
        judge: JudgeSpec,
        rubric: list[Criterion],
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
        *,
        guidelines: str | None = None,
    ) -> JudgeCriterionResults:
        """Evaluate all criteria for a single judge, one result per criterion, in order.

        An LLM judge is asked about each criterion in its own call, all in parallel, each
        user prompt starting with the rubric's guidelines when there are any. A
        decision-model judge is asked about the whole rubric in one request
        (``_judge_all_criteria_with_decision_model``), the guidelines in its state.
        """
        if isinstance(judge.llm_config, DecisionModelConfig):
            results = await self._judge_all_criteria_with_decision_model(
                judge, rubric, to_grade, query, reference_submission, guidelines=guidelines
            )
        else:
            tasks = [
                self._judge_single_criterion(
                    judge,
                    criterion,
                    idx,
                    to_grade,
                    query,
                    reference_submission,
                    guidelines=guidelines,
                )
                for idx, criterion in enumerate(rubric)
            ]
            results = await asyncio.gather(*tasks)
        return JudgeCriterionResults(
            judge_id=judge.judge_id,
            weight=judge.weight,
            criterion_results=list(results),
        )

    async def _judge_all_criteria_with_decision_model(
        self,
        judge: JudgeSpec,
        rubric: list[Criterion],
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
        *,
        guidelines: str | None = None,
    ) -> list[CriterionResult]:
        """Judge every criterion with a decision-model judge in one request.

        The state is built once from the submission (a ``<thinking>``/``<output>``
        submission is sent as its two parts) and its context, the rubric's guidelines first
        when there are any, and every criterion of the effective rubric
        that a question can express is posed under the id ``c{criterion_idx}``. Exactly one
        request is made, and none when no criterion can be expressed. Its answers map back
        to one result per criterion, in rubric order.

        Failures are scoped as tightly as they occur, and each is reported by the helper
        that reports an LLM judge's failed call (``_failed_judgment_result``, routed by
        ``classify_grading_error``):

        - A criterion that cannot be expressed is left out of the request and fails alone
          (``parse``).
        - A failed request fails every criterion posed in it, each with the request's
          error; criteria are never re-sent one at a time.
        - A missing or unusable answer fails only its criterion (``parse``).

        The request's usage and cost ride on the first result, so the judge's
        ``total_usage``/``total_cost`` are the item's totals.

        Args:
            judge: The decision-model judge.
            rubric: The effective rubric (NA options already guaranteed).
            to_grade: The submission, as ``judge`` receives it.
            query: The input that prompted the submission.
            reference_submission: An exemplar response for grading context.
            guidelines: The rubric's guidelines.

        Returns:
            One ``CriterionResult`` per criterion of ``rubric``, in order.
        """
        client = self._decision_clients[judge.judge_id]
        config = client.config
        state = build_state(
            to_grade,
            query=query,
            reference_submission=reference_submission,
            guidelines=guidelines,
        )
        questions, unposed = build_questions(rubric, config, state)

        response = None
        request_error: Exception | None = None
        if questions:
            try:
                response = await client.system_one(state, questions)
            except Exception as e:
                request_error = e

        # A criterion fails with its own error if it could not be posed, else with the
        # request's if the request failed, else its answer is mapped (and may fail alone).
        answers = response.answers if response is not None else {}
        results: list[CriterionResult] = []
        for criterion_idx, criterion in enumerate(rubric):
            error: Exception | None = unposed.get(criterion_idx, request_error)
            if error is None:
                try:
                    report = answer_to_report(criterion, criterion_idx, answers, config)
                except Exception as e:
                    error = e
                else:
                    results.append(CriterionResult(report=report))
                    continue
            results.append(_failed_judgment_result(criterion, judge.judge_id, error))

        if response is not None and results:
            results[0] = dataclasses.replace(
                results[0],
                usage=client.token_usage(response),
                cost=client.completion_cost(response),
            )
        return results

    # =========================================================================
    # Judge and Aggregate (Grader Interface)
    # =========================================================================

    def _effective_criterion(self, criterion: Criterion) -> Criterion:
        """Return the criterion as actually evaluated.

        When ``auto_na_option`` is enabled, multi-choice criteria are guaranteed an
        NA/"cannot assess" option (the abstain channel) via
        :meth:`Criterion.with_guaranteed_na_option`. Binary criteria and the
        ``auto_na_option=False`` case are returned unchanged. Pure function of the
        criterion, so prompt building, verdict mapping, scoring, ensemble aggregation,
        and the metrics layer can all reconstruct the identical option set.
        """
        if self._auto_na_option and criterion.is_multi_choice:
            return criterion.with_guaranteed_na_option()
        return criterion

    async def judge(
        self,
        to_grade: str,
        rubric: list[Criterion],
        query: str | None = None,
        reference_submission: str | None = None,
        *,
        guidelines: str | None = None,
    ) -> list[JudgeCriterionResults]:
        """Judge all criteria with all judges (parallel across judges).

        In a cascade the decision model judges every criterion first; then every escalation
        judge judges the escalated criteria (``_escalates``: an error, an abstention, or a
        confidence below ``EscalationConfig.threshold_for`` the criterion), all at once, and
        none when nothing is escalated.

        Args:
            to_grade: The submission, as ``Grader.grade`` passes it.
            rubric: The criteria.
            query: Optional input/query that prompted the submission.
            reference_submission: Optional exemplar response for grading context.
            guidelines: Optional rubric guidelines (``Grader.grade`` passes them when the
                rubric has any; blank text means none). Every judge sees them: each LLM user
                prompt starts with ``GUIDELINES_BLOCK``, and a decision model's state holds
                them as its first field, with the framed binary questions naming them. With
                none, every prompt and request is exactly what it is without guidelines.

        Returns:
            One ``JudgeCriterionResults`` per judge, in judge order: in a cascade, the
            decision model's (``role="primary"``), then each escalation judge's
            (``role="escalation"``, ``None`` at the criteria not escalated).

        Raises:
            TypeError: If ``guidelines`` is neither a ``str`` nor ``None``.

        Warns:
            UserWarning: Once per grader, when an ``EscalationConfig.per_criterion`` name
                matches no criterion of ``rubric`` (not while ``EvalRunner`` or
                ``fill_ground_truth`` grades a dataset, which checks the names against
                every item's rubric instead).
        """
        # Checked before any judge runs; blank guidelines become None.
        guidelines = _normalize_guidelines(guidelines)
        if not self._escalation_names_warned and not _ESCALATION_NAMES_CHECKED.get():
            # The flag is set only once the warning is issued, so under a
            # warnings-as-errors filter it raises on every call instead of only once.
            if self._check_escalation_names([rubric], "the rubric being graded"):
                self._escalation_names_warned = True
        # Normalize once to the effective rubric (abstain channel guaranteed for
        # multi-choice under auto_na_option). Same length/order, so criterion_idx — and
        # thus the shuffle RNG key — stays aligned; the user's rubric is never mutated.
        effective_rubric = [self._effective_criterion(c) for c in rubric]
        tasks = [
            self._judge_all_criteria_for_judge(
                judge,
                effective_rubric,
                to_grade,
                query,
                reference_submission,
                guidelines=guidelines,
            )
            for judge in self._judges
        ]
        results = list(await asyncio.gather(*tasks))
        escalation = self._escalation
        if escalation is None:
            return results

        # A cascade: ``_judges`` is its one decision model, which judged every criterion.
        (primary,) = results
        escalated = [
            criterion_idx
            for criterion_idx, (criterion, report) in enumerate(
                zip(effective_rubric, primary.reports, strict=True)
            )
            if _escalates(report, escalation.threshold_for(criterion))
        ]
        escalation_results = await asyncio.gather(
            *(
                self._judge_escalated_criteria(
                    judge,
                    effective_rubric,
                    escalated,
                    to_grade,
                    query,
                    reference_submission,
                    guidelines=guidelines,
                )
                for judge in self._escalation_judges
            )
        )
        return [primary, *escalation_results]

    async def _judge_escalated_criteria(
        self,
        judge: JudgeSpec,
        rubric: list[Criterion],
        escalated: list[int],
        to_grade: str,
        query: str | None = None,
        reference_submission: str | None = None,
        *,
        guidelines: str | None = None,
    ) -> JudgeCriterionResults:
        """Judge a cascade's escalated criteria with one escalation judge, all in parallel.

        Each goes through ``_judge_single_criterion`` under its index in the effective
        rubric, the index that keys option shuffling and few-shot selection, so the judge's
        prompt for it is the one it would get in a grader without the cascade with the same
        ``seed`` and ``judge_id``.

        Args:
            judge: The escalation judge (an LLM).
            rubric: The effective rubric.
            escalated: The indices of the escalated criteria, in rubric order.
            to_grade: The submission, as ``judge`` receives it.
            query: The input that prompted the submission.
            reference_submission: An exemplar response for grading context.
            guidelines: The rubric's guidelines.

        Returns:
            The judge's full-length results (``role="escalation"``): a result at each
            escalated criterion, ``None`` at the others.
        """
        results = await asyncio.gather(
            *(
                self._judge_single_criterion(
                    judge,
                    rubric[criterion_idx],
                    criterion_idx,
                    to_grade,
                    query,
                    reference_submission,
                    guidelines=guidelines,
                )
                for criterion_idx in escalated
            )
        )
        criterion_results: list[CriterionResult | None] = [None] * len(rubric)
        for criterion_idx, result in zip(escalated, results, strict=True):
            criterion_results[criterion_idx] = result
        return JudgeCriterionResults(
            judge_id=judge.judge_id,
            weight=judge.weight,
            criterion_results=criterion_results,
            role="escalation",
        )

    def _check_escalation_names(self, rubrics: Iterable[Iterable[Criterion]], where: str) -> bool:
        """Warn about ``EscalationConfig.per_criterion`` names that match no criterion.

        A name that no criterion of any of ``rubrics`` has sets no threshold (thresholds
        are looked up by criterion name), which is most likely a typo or a stale name. The
        warning is a ``UserWarning`` attributed to the code that asked for grading.

        Args:
            rubrics: The criteria the names are checked against, as one or more rubrics.
            where: What ``rubrics`` are, for the message (e.g. "the rubric being graded").

        Returns:
            Whether it warned. It never does for a grader without per-criterion thresholds.
        """
        per_criterion = self._escalation.per_criterion if self._escalation is not None else None
        return _warn_unknown_threshold_names(
            per_criterion, rubrics, source="EscalationConfig.per_criterion", where=where
        )

    async def aggregate(
        self, judge_results: list[JudgeCriterionResults], *, normalize: bool = True
    ) -> EnsembleEvaluationReport:
        """Aggregate results from all judges into final report.

        Handles both binary and multi-choice criteria:
        - Binary: Uses JudgeVote and _aggregate_votes()
        - Multi-choice: Uses MultiChoiceJudgeVote and _aggregate_multi_choice_votes()

        In a cascade the escalated criteria are the ones the escalation judges judged
        (their results are ``None`` everywhere else). On those, the primary judge's vote is
        marked ``superseded=True`` and the final verdict, reason and error come from the
        votes that are not superseded, which are the escalation judges'; the report is
        marked ``escalated=True``. ``judge_scores`` holds each primary judge's score over its
        own verdicts (superseded ones included) and ``None`` for each escalation judge,
        which never judges a whole rubric.
        """
        if not judge_results:
            # Empty/failed aggregation has no score: emit None, not a fabricated 0.0
            # (which is a valid catastrophic score, indistinguishable from a real zero).
            return EnsembleEvaluationReport(
                score=None,
                raw_score=None,
                llm_raw_score=None,
                error="No judge results to aggregate",
            )

        n_criteria = len(judge_results[0].criterion_results)
        # A cascade's escalated criteria: an escalation judge has a result at each of them,
        # and only there (none at all without a cascade).
        escalated = {
            criterion_idx
            for judge_result in judge_results
            if judge_result.role == "escalation"
            for criterion_idx, cr in enumerate(judge_result.criterion_results)
            if cr is not None
        }

        # Build ensemble criterion reports
        ensemble_reports: list[EnsembleCriterionReport] = []
        for criterion_idx in range(n_criteria):
            is_escalated = criterion_idx in escalated
            # Each judge's result on this criterion, in judge order; an escalation judge
            # has one only where the criterion was escalated.
            judged: list[tuple[JudgeCriterionResults, CriterionResult]] = [
                (judge_result, cr)
                for judge_result in judge_results
                if (cr := judge_result.criterion_results[criterion_idx]) is not None
            ]
            # Get criterion from the first judge's result
            criterion_report = judged[0][1].report

            if criterion_report.is_multi_choice:
                # Multi-choice: build MultiChoiceJudgeVote list
                mc_votes: list[MultiChoiceJudgeVote] = []
                # Each MultiChoiceJudgeVote carries its own .error (parity with JudgeVote),
                # so the ensemble error is derived from the votes via _aggregate_error,
                # mirroring the binary path. Every judge call synthesizes a verdict, so the
                # `mcv is not None` guard below never drops an errored vote.
                for judge_result, cr in judged:
                    mcv = cr.report.multi_choice_verdict
                    if mcv is not None:
                        mc_votes.append(
                            MultiChoiceJudgeVote(
                                judge_id=judge_result.judge_id,
                                selected_index=mcv.selected_index,
                                selected_label=mcv.selected_label,
                                value=mcv.value,
                                reason=cr.report.reason,
                                weight=judge_result.weight,
                                na=mcv.na,
                                shuffle_order=cr.report.shuffle_order,
                                error=cr.report.error,
                                reasoning=cr.report.reasoning,
                                probabilities=cr.report.probabilities,
                                confidence=cr.report.confidence,
                                superseded=is_escalated and judge_result.role == "primary",
                            )
                        )
                aggregated_mc_votes = [v for v in mc_votes if not v.superseded]

                # Aggregate multi-choice votes
                final_mc_verdict, final_reason = self._aggregate_multi_choice_votes(
                    aggregated_mc_votes, criterion_report
                )

                ensemble_reports.append(
                    EnsembleCriterionReport(
                        criterion=Criterion(
                            weight=criterion_report.weight,
                            requirement=criterion_report.requirement,
                            name=criterion_report.name,
                            options=criterion_report.options,
                            scale_type=criterion_report.scale_type,
                            aggregation=criterion_report.aggregation,
                        ),
                        final_verdict=None,  # Binary verdict is None for multi-choice
                        final_reason=final_reason,
                        votes=[],  # Binary votes empty for multi-choice
                        final_multi_choice_verdict=final_mc_verdict,
                        multi_choice_votes=mc_votes,
                        error=_aggregate_error(aggregated_mc_votes),
                        escalated=is_escalated,
                    )
                )
            else:
                # Binary: build JudgeVote list
                votes: list[JudgeVote] = []
                for judge_result, cr in judged:
                    votes.append(
                        JudgeVote(
                            judge_id=judge_result.judge_id,
                            verdict=cr.report.verdict,
                            reason=cr.report.reason,
                            weight=judge_result.weight,
                            error=cr.report.error,
                            reasoning=cr.report.reasoning,
                            probabilities=cr.report.probabilities,
                            confidence=cr.report.confidence,
                            superseded=is_escalated and judge_result.role == "primary",
                        )
                    )
                aggregated_votes = [v for v in votes if not v.superseded]

                final_verdict, final_reason = self._aggregate_votes(
                    aggregated_votes, criterion_report.weight
                )

                ensemble_reports.append(
                    EnsembleCriterionReport(
                        criterion=Criterion(
                            weight=criterion_report.weight,
                            requirement=criterion_report.requirement,
                            name=criterion_report.name,
                        ),
                        final_verdict=final_verdict,
                        final_reason=final_reason,
                        votes=votes,
                        error=_aggregate_error(aggregated_votes),
                        escalated=is_escalated,
                    )
                )

        # Calculate per-judge scores: a primary judge's over its own verdicts on every
        # criterion; an escalation judge's is undefined, None by role on every item.
        judge_scores: dict[str, float | None] = {}
        for judge_result in judge_results:
            judge_scores[judge_result.judge_id] = (
                self._calculate_score_from_reports(judge_result.reports, normalize)
                if judge_result.role == "primary"
                else None
            )

        # Aggregate token usage and cost
        total_usage = TokenUsage()
        total_cost = 0.0
        for jr in judge_results:
            if jr.total_usage:
                total_usage = total_usage + jr.total_usage
            if jr.total_cost:
                total_cost += jr.total_cost

        return _ensemble_evaluation_report(
            ensemble_reports,
            judge_scores,
            self._calculate_score_from_reports,
            normalize=normalize,
            token_usage=total_usage if total_usage.total_tokens > 0 else None,
            completion_cost=total_cost if total_cost > 0 else None,
        )

    def _aggregate_votes(
        self, votes: list[JudgeVote], weight: float
    ) -> tuple[CriterionVerdict, str | None]:
        """Aggregate votes from multiple judges into a single verdict.

        ``weight`` is the criterion weight (not a judge weight); it is used only to break
        ties in ``majority``/``weighted`` via :func:`_binary_worst_verdict` — UNMET for
        weight ≥ 0, MET for weight < 0 (consistent with ``worst_scored_option``).
        """
        if not votes:
            return CriterionVerdict.CANNOT_ASSESS, "No votes"

        # Filter out CANNOT_ASSESS for aggregation (unless all are CANNOT_ASSESS)
        assessable_votes = [v for v in votes if v.verdict != CriterionVerdict.CANNOT_ASSESS]
        if not assessable_votes:
            return CriterionVerdict.CANNOT_ASSESS, "All judges could not assess"

        met_weight = sum(v.weight for v in assessable_votes if v.verdict == CriterionVerdict.MET)
        unmet_weight = sum(
            v.weight for v in assessable_votes if v.verdict == CriterionVerdict.UNMET
        )

        if self._aggregation == "majority":
            # True head-count: count judges, ignore weights (> 50%); tie -> worst case.
            met_count = sum(1 for v in assessable_votes if v.verdict == CriterionVerdict.MET)
            unmet_count = sum(1 for v in assessable_votes if v.verdict == CriterionVerdict.UNMET)
            verdict = self._decide_binary(met_count, unmet_count, weight)
        elif self._aggregation == "weighted":
            verdict = self._decide_binary(met_weight, unmet_weight, weight)
        elif self._aggregation == "unanimous":
            verdict = CriterionVerdict.MET if unmet_weight == 0 else CriterionVerdict.UNMET
        elif self._aggregation == "any":
            verdict = CriterionVerdict.MET if met_weight > 0 else CriterionVerdict.UNMET
        else:
            verdict = self._decide_binary(met_weight, unmet_weight, weight)

        return verdict, _join_vote_reasons(votes)

    @staticmethod
    def _decide_binary(met: float, unmet: float, weight: float) -> CriterionVerdict:
        """MET if ``met`` strictly wins, UNMET if ``unmet`` strictly wins, else the
        weight-sign worst case. ``met``/``unmet`` are head-counts or summed
        weights depending on the strategy."""
        if met > unmet:
            return CriterionVerdict.MET
        if unmet > met:
            return CriterionVerdict.UNMET
        return _binary_worst_verdict(weight)

    def _aggregate_multi_choice_votes(
        self,
        votes: list[MultiChoiceJudgeVote],
        criterion: CriterionReport,
    ) -> tuple[AggregatedMultiChoiceVerdict, str | None]:
        """Aggregate multi-choice votes from multiple judges.

        Uses ordinal_aggregation for ordinal scale criteria and
        nominal_aggregation for nominal scale criteria.

        Args:
            votes: List of multi-choice votes from all judges.
            criterion: The criterion report (to access options and scale_type).

        Returns:
            Tuple of (aggregated verdict, combined reason). The reason is ``None`` when no
            vote carried one (see ``_join_vote_reasons``).
        """
        if not votes:
            # Return NA verdict if no votes
            if criterion.options:
                na_idx = criterion.na_option_index
                if na_idx is not None:
                    na_opt = criterion.options[na_idx]
                    return (
                        AggregatedMultiChoiceVerdict(
                            selected_index=na_idx,
                            selected_label=na_opt.label,
                            value=na_opt.value,
                            na=True,
                            aggregated_value=0.0,
                        ),
                        "No votes",
                    )
            # No NA option to abstain into: fall back to the weight-sign worst case
            # (consistent with worst_scored_option / the unknown-error path).
            if criterion.options:
                worst_idx, worst_opt = criterion.worst_scored_option()
                return (
                    AggregatedMultiChoiceVerdict(
                        selected_index=worst_idx,
                        selected_label=worst_opt.label,
                        value=worst_opt.value,
                        na=worst_opt.na,
                        aggregated_value=worst_opt.value,
                    ),
                    "No votes",
                )
            return (
                AggregatedMultiChoiceVerdict(
                    selected_index=0,
                    selected_label="",
                    value=0.0,
                    na=False,
                    aggregated_value=0.0,
                ),
                "No votes",
            )

        # Filter out NA votes for aggregation (unless all are NA). The extra
        # ``selected_index is not None`` is redundant at runtime (a None index always
        # carries na=True, so it is already excluded by ``not v.na``) but narrows the
        # type so the downstream index reads need no None-guards.
        assessable_votes = [v for v in votes if not v.na and v.selected_index is not None]
        if not assessable_votes:
            # All votes are NA. Prefer a vote that abstained into a GENUINE NA option
            # (selected_index is not None) so the aggregate keeps a real NA index where one
            # exists (the default auto_na_option case); fall back to a clean None-abstain
            # only when every NA vote is itself a no-NA-option error-abstain.
            na_vote = next((v for v in votes if v.selected_index is not None), votes[0])
            return (
                AggregatedMultiChoiceVerdict(
                    selected_index=na_vote.selected_index,
                    selected_label=na_vote.selected_label,
                    value=na_vote.value,
                    na=True,
                    aggregated_value=na_vote.value,
                ),
                _join_vote_reasons(votes),
            )

        # Check for per-criterion aggregation override
        agg_strategy = criterion.aggregation
        scale_type = criterion.scale_type

        # Determine which aggregation to use
        if scale_type == "ordinal":
            agg = agg_strategy or self._ordinal_aggregation
            result = self._aggregate_ordinal_votes(assessable_votes, criterion, agg)
        else:  # nominal
            agg = agg_strategy or self._nominal_aggregation
            result = self._aggregate_nominal_votes(assessable_votes, criterion, agg)

        return result, _join_vote_reasons(votes)

    def _aggregate_ordinal_votes(
        self,
        votes: list[MultiChoiceJudgeVote],
        criterion: Criterion,
        strategy: str,
    ) -> AggregatedMultiChoiceVerdict:
        """Aggregate ordinal multi-choice votes.

        Strategies:
        - mean: Average of score values, snap to nearest option
        - median: Median of score values, snap to nearest option
        - weighted_mean: Weighted average by judge weight
        - mode: Most common selection
        - min: Lowest-value option any judge selected (conservative; analog of binary
          ``unanimous``)
        - max: Highest-value option any judge selected (permissive; analog of binary ``any``)

        Tie-breaking (mode count tie, mean/median snap equidistance): the
        score-minimizing tied option by weight sign via ``criterion.worst_option_among``
        (lowest value for weight ≥ 0, highest for weight < 0; lowest index on a value
        tie). ``min``/``max`` already resolve value ties to the lowest index.
        """
        options = criterion.options or []
        values = [v.value for v in votes]
        weights = [v.weight for v in votes]
        # Assessable votes always carry a concrete index (None is reserved for the
        # error-abstain, filtered out before aggregation). The guard makes that contract
        # explicit and narrows the type to ``int`` for the index-based strategies.
        indices = [v.selected_index for v in votes if v.selected_index is not None]

        if strategy == "mean":
            aggregated_value = sum(values) / len(values)
        elif strategy == "median":
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                aggregated_value = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
            else:
                aggregated_value = sorted_values[n // 2]
        elif strategy == "weighted_mean":
            total_weight = sum(weights)
            if total_weight > 0:
                aggregated_value = sum(v * w for v, w in zip(values, weights)) / total_weight
            else:
                aggregated_value = sum(values) / len(values)
        elif strategy == "mode":
            # Most common selection; count ties -> worst tied option by weight sign.
            most_common_idx = criterion.worst_option_among(_top_tied_keys(Counter(indices)))
            selected_option = options[most_common_idx]
            return AggregatedMultiChoiceVerdict(
                selected_index=most_common_idx,
                selected_label=selected_option.label,
                value=selected_option.value,
                na=selected_option.na,
                aggregated_value=selected_option.value,  # For mode, no continuous value
            )
        elif strategy in ("min", "max"):
            # Conservative / permissive analogs of binary unanimous / any: the lowest-
            # (min) or highest- (max) value option any judge selected. Value ties resolve
            # to the lowest option index (deterministic; tie rules out of scope).
            scored = [(v.value, idx) for v in votes if (idx := v.selected_index) is not None]
            if strategy == "min":
                _, chosen_idx = min(scored, key=lambda t: (t[0], t[1]))
            else:
                _, chosen_idx = max(scored, key=lambda t: (t[0], -t[1]))
            selected_option = options[chosen_idx]
            return AggregatedMultiChoiceVerdict(
                selected_index=chosen_idx,
                selected_label=selected_option.label,
                value=selected_option.value,
                na=selected_option.na,
                aggregated_value=selected_option.value,
            )
        else:
            # Default to mean
            aggregated_value = sum(values) / len(values)

        # Snap to nearest non-NA option by value; equidistant ties -> worst tied option
        # by weight sign (deterministic, independent of option declaration order).
        distances = [
            (abs(opt.value - aggregated_value), i) for i, opt in enumerate(options) if not opt.na
        ]
        min_diff = min(d for d, _ in distances)
        tied = [i for d, i in distances if math.isclose(d, min_diff, abs_tol=1e-9)]
        closest_idx = criterion.worst_option_among(tied)

        selected_option = options[closest_idx]
        return AggregatedMultiChoiceVerdict(
            selected_index=closest_idx,
            selected_label=selected_option.label,
            value=selected_option.value,
            na=selected_option.na,
            aggregated_value=aggregated_value,  # Store continuous value before snap
        )

    def _aggregate_nominal_votes(
        self,
        votes: list[MultiChoiceJudgeVote],
        criterion: Criterion,
        strategy: str,
    ) -> AggregatedMultiChoiceVerdict:
        """Aggregate nominal multi-choice votes.

        Strategies:
        - mode: Most common selection (majority)
        - weighted_mode: Weight votes by judge weight
        - unanimous: All judges must select the same option. On disagreement, abstain
          via the criterion's NA option (verdict na=True); if there is no NA option,
          fall back to mode and warn.

        Tie-breaking (mode count tie, weighted_mode equal-weight tie): the
        score-minimizing tied option by weight sign via ``criterion.worst_option_among``
        (lowest value for weight ≥ 0, highest for weight < 0; lowest index on a value
        tie) — deterministic, independent of judge order.
        """
        options = criterion.options or []
        # Assessable votes always carry a concrete index (None is reserved for the
        # error-abstain, filtered out before aggregation). The guard makes that contract
        # explicit and narrows the type to ``int``.
        indices = [v.selected_index for v in votes if v.selected_index is not None]

        if strategy == "mode":
            most_common_idx = criterion.worst_option_among(_top_tied_keys(Counter(indices)))
        elif strategy == "weighted_mode":
            # Accumulate weights per index; equal-weight ties -> worst tied option.
            weight_per_idx: dict[int, float] = {}
            for v in votes:
                if v.selected_index is None:
                    continue
                weight_per_idx[v.selected_index] = (
                    weight_per_idx.get(v.selected_index, 0.0) + v.weight
                )
            most_common_idx = criterion.worst_option_among(_top_tied_keys(weight_per_idx))
        elif strategy == "unanimous":
            unique_indices = set(indices)
            if len(unique_indices) == 1:
                most_common_idx = indices[0]
            else:
                # Judges disagree -> abstain via the NA option if one exists (na=True flows
                # through the SKIP scoring path). Never set na=True against a real option.
                # With no NA option, fall back to mode and warn.
                na_idx = criterion.na_option_index
                if na_idx is not None:
                    most_common_idx = na_idx
                else:
                    logger.warning(
                        "Nominal 'unanimous' aggregation: judges disagreed (indices %s) "
                        "and the criterion has no NA option; falling back to mode.",
                        sorted(unique_indices),
                    )
                    most_common_idx = criterion.worst_option_among(_top_tied_keys(Counter(indices)))

        selected_option = options[most_common_idx]
        return AggregatedMultiChoiceVerdict(
            selected_index=most_common_idx,
            selected_label=selected_option.label,
            value=selected_option.value,
            na=selected_option.na,
            aggregated_value=selected_option.value,  # For nominal, discrete = continuous
        )

    def _calculate_score_from_reports(
        self, reports: list[CriterionReport], normalize: bool
    ) -> float:
        """Calculate score from criterion reports via the shared scoring core."""
        return score_reports(reports, self._cannot_assess_config, normalize)
